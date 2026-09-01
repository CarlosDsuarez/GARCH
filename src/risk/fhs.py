"""Filtered Historical Simulation (FHS) for VaR and Expected Shortfall.

Why FHS — and not the usual alternatives
----------------------------------------
PARAMETRIC NORMAL VaR assumes Gaussian returns. Credit returns have severe
excess kurtosis; a 99% normal VaR systematically understates the real loss.

PURE HISTORICAL SIMULATION assigns equal probability to every day in the
window. A 2020-crisis day and a quiet 2021 day weigh the same in tomorrow's
number. It ignores whether VIX is 12 or 40 *today*. The VaR then reacts with
a long lag, rises after the crisis is already in the window, and falls when
the crash rolls out — the worst possible timing.

FHS fixes both at once: it keeps the *empirical* shape of the innovation
distribution (real skew and kurtosis, no parametric assumption) and *updates
the scale* with today's conditional volatility. That is why CCPs use it for
dynamic initial margin.

This module never implements DCC-GARCH. Published DCC examples are 2–3
assets; at 20+ names the likelihood is unstable, correlations thrash or
collapse to the unconditional mean, and the "dynamic" part vanishes. Route B
is univariate GJR margins plus a t-copula.

Sign convention
---------------
VaR and ES are **positive loss magnitudes**. A 99% one-day VaR of 0.02 means
a two-percent loss. Sign inconsistency across risk modules is a classic
production failure mode.

Horizon aggregation
-------------------
Returns in this engine are **simple**. A path of daily simple returns
:math:`r_1,\\ldots,r_h` is aggregated as

.. math::

    R^* = \\prod_{j=1}^{h}(1+r_j^*) - 1.

Do not mix this with a log-return sum. The gap is order
:math:`h\\sigma^2/2` and grows with the horizon.

:math:`\\mathrm{VaR}_h = \\mathrm{VaR}_1\\sqrt{h}` is forbidden. GARCH
variance mean-reverts, and the leverage indicator compounds non-linearly.
Multi-day risk is obtained only by simulating GJR paths.

Algorithm (Barone-Adesi, Giannopoulos, Vosper)
----------------------------------------------
Fit GJR-GARCH(1,1)-skewt; extract :math:`z_t=\\varepsilon_t/\\sigma_t`;
**rescale** :math:`z` to unit *sample* standard deviation; forecast
:math:`\\sigma_{T+1}` from the observed residual at :math:`T`; bootstrap
:math:`z^*` with equal probability :math:`1/T` (recency lives in
:math:`\\sigma_{T+1}`, not in the draw weights); rebuild
:math:`r^*=\\mu+\\sigma_{T+1}z^*`; report
:math:`\\mathrm{VaR}_\\alpha=-Q_{1-\\alpha}(r^*)` and
:math:`\\mathrm{ES}_\\alpha=-E[r^*\\mid r^*\\le Q_{1-\\alpha}]`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from arch.univariate import arch_model
from scipy import optimize, stats as scipy_stats
from scipy.special import gammaln
from statsmodels.stats.diagnostic import acorr_ljungbox

from data.quality import assert_no_stale_zero_returns
from risk.schema import FHSConfig

logger = logging.getLogger(__name__)

__all__ = [
    "FHSEngine",
    "FHSError",
    "ReportPaths",
    "SimulatedPaths",
    "TCopula",
    "TCopulaFHSEngine",
    "VaRES",
    "portfolio_returns",
]


class FHSError(ValueError):
    """FHS sample, filter, or copula is not usable ([C7])."""


@dataclass(frozen=True)
class VaRES:
    alpha: float
    horizon: int
    var: float
    es: float
    n_simulations: int
    sigma_forecast: float


@dataclass(frozen=True)
class SimulatedPaths:
    daily: np.ndarray
    aggregated: np.ndarray
    aggregation: str


@dataclass(frozen=True)
class ReportPaths:
    markdown: Path
    html: Path
    plot: Path


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Route A: apply *current* weights to the full asset-return history."""
    if asset_returns.empty:
        raise FHSError("asset return frame is empty")
    w = weights.astype("float64")
    missing = [name for name in w.index if name not in asset_returns.columns]
    if missing:
        raise FHSError(f"weights refer to missing assets: {missing}")
    aligned = asset_returns.loc[:, w.index].astype("float64")
    if aligned.isna().any().any():
        raise FHSError("asset returns contain NaN; do not silently fill")
    port = aligned.mul(w, axis=1).sum(axis=1)
    port.name = "portfolio"
    return port


def _require_positive_definite(corr: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (corr + corr.T)
    eigval, eigvec = np.linalg.eigh(matrix)
    eigval = np.maximum(eigval, 1e-8)
    rebuilt = eigvec @ np.diag(eigval) @ eigvec.T
    scale = np.sqrt(np.clip(np.diag(rebuilt), 1e-12, None))
    rebuilt = rebuilt / np.outer(scale, scale)
    np.fill_diagonal(rebuilt, 1.0)
    return rebuilt


def _t_copula_loglik(x: np.ndarray, corr: np.ndarray, nu: float) -> float:
    n_obs, n_dim = x.shape
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:
        return float("-inf")
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        return float("-inf")
    quad = np.einsum("ti,ij,tj->t", x, inv, x)
    ll_mv = (
        gammaln((nu + n_dim) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * n_dim * np.log(nu * np.pi)
        - 0.5 * logdet
        - ((nu + n_dim) / 2.0) * np.log1p(quad / nu)
    )
    ll_uni = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - ((nu + 1.0) / 2.0) * np.log1p((x * x) / nu)
    )
    return float(ll_mv.sum() - ll_uni.sum())


class TCopula:
    """Student-t copula on uniform margins (symmetric tail dependence)."""

    def __init__(self) -> None:
        self.corr: np.ndarray | None = None
        self.nu: float | None = None

    def fit(self, uniforms: np.ndarray) -> TCopula:
        u = np.clip(np.asarray(uniforms, dtype=float), 1e-6, 1.0 - 1e-6)
        if u.ndim != 2 or u.shape[1] < 2:
            raise FHSError("t-copula requires a (T, N) uniform matrix with N>=2")
        best_ll = float("-inf")
        best_nu = 6.0
        best_corr = np.eye(u.shape[1])
        for nu in np.linspace(3.0, 25.0, 12):
            scores = scipy_stats.t.ppf(u, df=nu)
            if not np.isfinite(scores).all():
                continue
            corr = _require_positive_definite(np.corrcoef(scores, rowvar=False))
            ll = _t_copula_loglik(scores, corr, float(nu))
            if ll > best_ll:
                best_ll = ll
                best_nu = float(nu)
                best_corr = corr
        if not np.isfinite(best_ll):
            raise FHSError("t-copula MLE failed")
        self.nu = best_nu
        self.corr = best_corr
        logger.info("t-copula fit nu=%.2f rho_01=%.3f ll=%.2f", best_nu, best_corr[0, 1], best_ll)
        return self

    def simulate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.corr is None or self.nu is None:
            raise FHSError("t-copula simulate() requires fit()")
        dim = self.corr.shape[0]
        chol = np.linalg.cholesky(self.corr)
        gaussian = rng.standard_normal((int(n), dim)) @ chol.T
        chi = rng.chisquare(self.nu, size=int(n)) / self.nu
        student = gaussian / np.sqrt(chi)[:, None]
        uniforms = scipy_stats.t.cdf(student, df=self.nu)
        return np.clip(uniforms, 1e-10, 1.0 - 1e-10)


def _gjr_result_usable(result: object, y: pd.Series) -> bool:
    params = getattr(result, "params", None)
    if params is None or not np.isfinite(np.asarray(params, dtype=float)).all():
        return False
    llf = float(getattr(result, "loglikelihood", float("nan")))
    if not np.isfinite(llf):
        return False
    try:
        mu = _param(result, ("mu", "Const"))
        omega = _param(result, ("omega",))
    except FHSError:
        return False
    data_std = float(np.std(y.to_numpy(dtype=float), ddof=1)) or 1e-8
    if abs(mu) > max(0.5, 25.0 * data_std):
        return False
    try:
        phi = _param(result, ("y[1]", "r[1]", f"{y.name}[1]"))
        if abs(phi) >= 0.99:
            return False
    except FHSError:
        pass
    if omega <= 0.0:
        return False
    vol = np.asarray(getattr(result, "conditional_volatility", []), dtype=float)
    if vol.size == 0 or not np.all(np.isfinite(vol)) or not np.all(vol > 0.0):
        return False
    median_vol = float(np.median(vol))
    return 1e-6 * data_std <= median_vol <= 200.0 * data_std


class _ManualGJR:
    def __init__(
        self,
        params: pd.Series,
        resid: np.ndarray,
        conditional_volatility: np.ndarray,
        loglikelihood: float,
    ) -> None:
        self.params = params
        self.resid = resid
        self.conditional_volatility = conditional_volatility
        self.loglikelihood = loglikelihood
        self.scale = 1.0
        self.convergence_flag = 0


def _qmle_gjr(y: pd.Series, *, mean_spec: str) -> _ManualGJR:
    """Bounded Gaussian QMLE for GJR(1,1). Last-resort filter when arch fails."""
    r = y.to_numpy(dtype=float)
    n = int(r.size)
    var = float(np.var(r, ddof=1)) or 1e-6
    mu0 = float(np.mean(r))
    ar = mean_spec == "AR(1)"

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float, float, float]:
        mu = float(theta[0])
        phi = float(theta[1]) if ar else 0.0
        omega = float(np.exp(theta[1 + int(ar)]))
        alpha = float(0.25 / (1.0 + np.exp(-theta[2 + int(ar)])))
        gamma = float(0.40 * np.tanh(theta[3 + int(ar)]))
        if alpha + gamma < 0.0:
            gamma = -alpha
        room = max(0.995 - alpha - 0.5 * max(gamma, 0.0), 1e-4)
        beta = float(room / (1.0 + np.exp(-theta[4 + int(ar)])))
        return mu, phi, omega, alpha, gamma, beta

    def residuals_and_var(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, phi, omega, alpha, gamma, beta = unpack(theta)
        eps = np.empty(n)
        sig2 = np.empty(n)
        eps[0] = r[0] - mu
        sig2[0] = var
        for t in range(1, n):
            mu_t = mu + phi * r[t - 1] if ar else mu
            eps[t] = r[t] - mu_t
            sig2[t] = omega + (alpha + gamma * float(eps[t - 1] < 0.0)) * eps[t - 1] ** 2 + beta * sig2[t - 1]
            sig2[t] = max(sig2[t], 1e-16)
        return eps, sig2

    def nll(theta: np.ndarray) -> float:
        eps, sig2 = residuals_and_var(theta)
        return float(0.5 * np.sum(np.log(sig2) + (eps * eps) / sig2))

    x0 = np.array(
        ([mu0, 0.25] if ar else [mu0])
        + [np.log(max(var * 0.04, 1e-12)), 0.0, 0.0, 2.2],
        dtype=float,
    )
    opt = optimize.minimize(nll, x0, method="L-BFGS-B")
    if not opt.success and not np.isfinite(opt.fun):
        raise FHSError(f"manual GJR QMLE failed: {opt.message}")
    mu, phi, omega, alpha, gamma, beta = unpack(opt.x)
    eps, sig2 = residuals_and_var(opt.x)
    params = {"mu": mu, "omega": omega, "alpha[1]": alpha, "gamma[1]": gamma, "beta[1]": beta}
    if ar:
        params["r[1]"] = phi
        params["Const"] = mu
    return _ManualGJR(
        params=pd.Series(params, dtype=float),
        resid=eps,
        conditional_volatility=np.sqrt(sig2),
        loglikelihood=float(-opt.fun),
    )


def _default_gjr_start(y: pd.Series, *, mean_spec: str, dist: str) -> np.ndarray:
    mu = float(y.mean())
    var = float(y.var(ddof=1)) or 1e-6
    alpha, gamma, beta = 0.04, 0.04, 0.90
    omega = max(var * (1.0 - alpha - 0.5 * gamma - beta), 1e-12)
    mean = [mu, 0.10] if mean_spec == "AR(1)" else [mu]
    vol = [omega, alpha, gamma, beta]
    dist_params = [8.0, 0.0] if dist == "skewt" else [8.0]
    return np.array(mean + vol + dist_params, dtype=float)


def _param(result: object, names: tuple[str, ...]) -> float:
    params = result.params  # type: ignore[attr-defined]
    for name in names:
        if name in params.index:
            return float(params[name])
    for name in params.index:
        if any(key in str(name) for key in names):
            return float(params[name])
    raise FHSError(f"GJR parameter not found among {names}; have {list(params.index)}")


def _var_es_from_draws(
    draws: np.ndarray,
    *,
    alpha: float,
    horizon: int,
    sigma_forecast: float,
) -> VaRES:
    if draws.size < 10:
        raise FHSError("not enough simulated returns for VaR/ES")
    q = float(np.quantile(draws, 1.0 - alpha))
    var = -q
    tail = draws[draws <= q]
    es = -float(tail.mean()) if tail.size else var
    if var <= 0.0:
        logger.warning("VaR at alpha=%.3f is non-positive (%.6f); check the P&L sign", alpha, var)
    return VaRES(
        alpha=float(alpha),
        horizon=int(horizon),
        var=float(var),
        es=float(es),
        n_simulations=int(draws.shape[0]),
        sigma_forecast=float(sigma_forecast),
    )


class FHSEngine:
    """Univariate FHS (Route A). ``fit`` / ``var_es`` / ``simulate_paths``."""

    def __init__(self, config: FHSConfig) -> None:
        self.config = config
        self.returns: pd.Series | None = None
        self.eps: pd.Series | None = None
        self.sigma: pd.Series | None = None
        self.z: pd.Series | None = None
        self.mean_spec: str = "Constant"
        self.omega: float = float("nan")
        self.alpha: float = float("nan")
        self.gamma: float = float("nan")
        self.beta: float = float("nan")
        self.mu_const: float = 0.0
        self.phi: float = 0.0
        self.mu_one_step: float = float("nan")
        self.sigma_one_step: float = float("nan")
        self.comparison_table: pd.DataFrame | None = None

    def fit(self, returns: pd.Series) -> FHSEngine:
        series = _as_return_series(returns)
        try:
            assert_no_stale_zero_returns(series)
        except ValueError as exc:
            raise FHSError(str(exc)) from exc
        min_obs = self.config.filter.min_observations
        if series.shape[0] < min_obs:
            raise FHSError(
                f"insufficient observations for FHS: {series.shape[0]} < "
                f"min_observations={min_obs}"
            )
        mean_spec = self._choose_mean(series)
        try:
            fitted = self._fit_gjr(series, mean_spec=mean_spec)
        except FHSError:
            if mean_spec != "AR(1)":
                raise
            logger.warning("AR(1) GJR unusable; refitting with a constant mean")
            mean_spec = "Constant"
            fitted = self._fit_gjr(series, mean_spec=mean_spec)
        if float(getattr(fitted, "scale", 1.0)) != 1.0:
            raise FHSError(
                f"arch rescaled the series (scale={fitted.scale}); rescale=False was required"
            )
        resid = pd.Series(np.asarray(fitted.resid, dtype=float), index=series.index, name="eps")
        sigma = pd.Series(
            np.asarray(fitted.conditional_volatility, dtype=float),
            index=series.index,
            name="sigma",
        )
        aligned = pd.concat({"r": series, "eps": resid, "sigma": sigma}, axis=1).dropna()
        if aligned.shape[0] < min_obs:
            raise FHSError(
                f"insufficient aligned residuals after the mean equation: "
                f"{aligned.shape[0]} < min_observations={min_obs}"
            )
        if (aligned["sigma"] <= 0).any():
            raise FHSError("conditional volatility must be strictly positive")
        z_raw = aligned["eps"] / aligned["sigma"]
        sample_std = float(z_raw.std(ddof=1))
        if sample_std <= 0.0 or not np.isfinite(sample_std):
            raise FHSError("standardized residual sample std is not usable")
        z = (z_raw / sample_std).rename("z")
        self.returns = series.rename("r")
        self.eps = aligned["eps"]
        self.sigma = aligned["sigma"]
        self.z = z
        self.mean_spec = mean_spec
        self.omega = _param(fitted, ("omega",))
        self.alpha = _param(fitted, ("alpha[1]", "alpha"))
        try:
            self.gamma = _param(fitted, ("gamma[1]", "gamma"))
        except FHSError:
            self.gamma = 0.0
        self.beta = _param(fitted, ("beta[1]", "beta"))
        self.mu_const = _param(fitted, ("mu", "Const"))
        self.phi = 0.0
        if mean_spec == "AR(1)":
            self.phi = _param(fitted, ("y[1]", "r[1]", f"{series.name}[1]"))
        last_eps = float(self.eps.iloc[-1])
        last_sig2 = float(self.sigma.iloc[-1] ** 2)
        sig2_next = self.omega + (
            self.alpha + self.gamma * float(last_eps < 0.0)
        ) * last_eps**2 + self.beta * last_sig2
        if sig2_next <= 0.0:
            raise FHSError(f"GJR one-step variance is non-positive: {sig2_next}")
        self.sigma_one_step = float(np.sqrt(sig2_next))
        if mean_spec == "AR(1)":
            self.mu_one_step = self.mu_const + self.phi * float(self.returns.iloc[-1])
        else:
            self.mu_one_step = self.mu_const
        logger.info(
            "FHS fit n=%s mean=%s omega=%.6g alpha=%.4f gamma=%.4f beta=%.4f "
            "sigma_T+1=%.6f z_std_before_rescale=%.4f",
            int(aligned.shape[0]),
            mean_spec,
            self.omega,
            self.alpha,
            self.gamma,
            self.beta,
            self.sigma_one_step,
            sample_std,
        )
        return self

    def fit_portfolio(self, asset_returns: pd.DataFrame, weights: pd.Series) -> FHSEngine:
        return self.fit(portfolio_returns(asset_returns, weights))

    def simulate_paths(
        self,
        horizon: int,
        n_simulations: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> SimulatedPaths:
        self._require_fit()
        if horizon < 1:
            raise FHSError("horizon must be >= 1")
        if self.config.fhs.scale_sqrt_h:
            raise FHSError("sqrt(h) scaling is forbidden")
        n_sim = int(n_simulations or self.config.fhs.n_simulations)
        generator = rng or np.random.default_rng(self.config.fhs.seed)
        z_pool = self.z.to_numpy(dtype=float)
        daily = np.empty((n_sim, horizon), dtype=float)
        sigma2 = np.full(n_sim, self.sigma_one_step**2, dtype=float)
        last_r = float(self.returns.iloc[-1])
        for step in range(horizon):
            z_star = generator.choice(z_pool, size=n_sim, replace=True)
            if self.mean_spec == "AR(1)":
                lagged = last_r if step == 0 else daily[:, step - 1]
                mu = self.mu_const + self.phi * lagged
            else:
                mu = self.mu_one_step
            eps = np.sqrt(sigma2) * z_star
            daily[:, step] = mu + eps
            sigma2 = (
                self.omega
                + (self.alpha + self.gamma * (eps < 0.0).astype(float)) * eps * eps
                + self.beta * sigma2
            )
            sigma2 = np.maximum(sigma2, 1e-16)
        aggregated = np.prod(1.0 + daily, axis=1) - 1.0
        return SimulatedPaths(daily=daily, aggregated=aggregated, aggregation="simple")

    def var_es(
        self,
        alpha: float | None = None,
        horizon: int | None = None,
    ) -> VaRES | dict[float, VaRES]:
        self._require_fit()
        h = int(horizon or self.config.fhs.default_horizon)
        paths = self.simulate_paths(horizon=h)
        alphas = self.config.fhs.alphas if alpha is None else [float(alpha)]
        reports = {
            a: _var_es_from_draws(
                paths.aggregated,
                alpha=a,
                horizon=h,
                sigma_forecast=self.sigma_one_step,
            )
            for a in alphas
        }
        if alpha is None:
            return reports
        return reports[float(alpha)]

    def write_report(self, returns: pd.Series | None = None) -> ReportPaths:
        self._require_fit()
        self.comparison_table = self._build_comparison()
        plot_cfg = self.config.plot
        out_dir = Path(plot_cfg.output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / plot_cfg.filename
        _plot_comparison(self.comparison_table, plot_path, plot_cfg)
        md = _render_report(self, self.comparison_table, plot_path)
        md_path = Path(self.config.output.report_markdown)
        html_path = Path(self.config.output.report_html)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        html_path.write_text(_as_html(md), encoding="utf-8")
        logger.info("wrote FHS report %s plot=%s", md_path, plot_path)
        return ReportPaths(markdown=md_path, html=html_path, plot=plot_path)

    def _choose_mean(self, series: pd.Series) -> str:
        spec = self.config.filter
        lb = acorr_ljungbox(series.to_numpy(dtype=float), lags=[spec.ljung_box_lags], return_df=True)
        pvalue = float(lb["lb_pvalue"].iloc[-1])
        if pvalue <= spec.ljung_box_pvalue_min:
            logger.info(
                "Ljung-Box Q(%s) on r p=%.4f <= %.2f; escalating mean to AR(1)",
                spec.ljung_box_lags,
                pvalue,
                spec.ljung_box_pvalue_min,
            )
            return "AR(1)"
        return "Constant"

    def _fit_gjr(self, series: pd.Series, *, mean_spec: str):
        spec = self.config.filter
        y = series.astype("float64")
        options = {"ftol": spec.ftol, "maxiter": spec.maxiter}
        attempts = (
            (spec.dist, spec.o),
            ("t", spec.o),
            ("t", 0),
        )
        last_error = "no GJR attempt produced usable parameters"
        candidates: list[object] = []
        for dist, o in attempts:
            model = self._build_gjr(y, mean_spec=mean_spec, dist=dist, o=o)
            start = _default_gjr_start(y, mean_spec=mean_spec, dist=dist)
            if o == 0:
                gamma_index = 4 if mean_spec == "AR(1)" else 3
                start = np.delete(start, gamma_index)
            try:
                result = model.fit(
                    disp="off",
                    options=options,
                    starting_values=start,
                    show_warning=False,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.exception("GJR fit failed dist=%s o=%s", dist, o)
                continue
            if _gjr_result_usable(result, y):
                candidates.append(result)
            else:
                last_error = f"unusable params: {result.params.to_dict()}"
        try:
            manual = _qmle_gjr(y, mean_spec=mean_spec)
        except FHSError as exc:
            last_error = str(exc)
        else:
            if _gjr_result_usable(manual, y):
                candidates.append(manual)
            else:
                last_error = f"unusable QMLE params: {manual.params.to_dict()}"
        if not candidates:
            raise FHSError(f"GJR-GARCH produced unusable parameters: {last_error}")
        data_std = float(np.std(y.to_numpy(dtype=float), ddof=1)) or 1e-8

        def _vol_gap(result: object) -> float:
            vol = np.asarray(result.conditional_volatility, dtype=float)
            return abs(float(np.median(vol)) - data_std)

        chosen = min(candidates, key=_vol_gap)
        if isinstance(chosen, _ManualGJR):
            logger.warning("selected bounded GJR QMLE (conditional vol closest to sample std)")
        return chosen

    def _build_gjr(self, y: pd.Series, *, mean_spec: str, dist: str, o: int):
        spec = self.config.filter
        kwargs = {
            "vol": "GARCH",
            "p": spec.p,
            "o": o,
            "q": spec.q,
            "dist": dist,
            "rescale": spec.rescale,
        }
        if mean_spec == "AR(1)":
            return arch_model(y, mean="ARX", lags=spec.ar_lags, **kwargs)
        return arch_model(y, mean="Constant", **kwargs)

    def _require_fit(self) -> pd.Series:
        if self.returns is None or self.z is None or self.sigma is None or self.eps is None:
            raise FHSError("var_es/simulate_paths require fit()")
        return self.returns

    def _build_comparison(self) -> pd.DataFrame:
        r = self._require_fit()
        window = self.config.comparison.historical_window
        alpha = 0.99 if 0.99 in self.config.fhs.alphas else self.config.fhs.alphas[-1]
        z_score = float(scipy_stats.norm.ppf(1.0 - alpha))
        rows: list[dict[str, object]] = []
        dates: list[pd.Timestamp] = []
        for i in range(window, len(r)):
            hist = r.iloc[i - window : i]
            var_hs = -float(np.quantile(hist.to_numpy(dtype=float), 1.0 - alpha))
            var_n = -(float(hist.mean()) + float(hist.std(ddof=1)) * z_score)
            z_hist = self.z.reindex(r.index).iloc[:i].dropna()
            if z_hist.shape[0] < 20:
                continue
            sigma_t = float(self.sigma.reindex(r.index).iloc[i])
            if self.mean_spec == "AR(1)":
                mu_t = self.mu_const + self.phi * float(r.iloc[i - 1])
            else:
                mu_t = self.mu_const
            var_fhs = -(mu_t + sigma_t * float(np.quantile(z_hist.to_numpy(), 1.0 - alpha)))
            realized = float(r.iloc[i])
            dates.append(pd.Timestamp(r.index[i]))
            rows.append(
                {
                    "return": realized,
                    "var_fhs": var_fhs,
                    "var_historical": var_hs,
                    "var_normal": var_n,
                    "violation_fhs": realized < -var_fhs,
                    "violation_historical": realized < -var_hs,
                    "violation_normal": realized < -var_n,
                }
            )
        if not rows:
            raise FHSError("not enough observations to build the VaR comparison table")
        table = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
        table.index.name = "date"
        return table


class TCopulaFHSEngine:
    """Route B: univariate GJR-FHS margins + t-copula. Not DCC."""

    def __init__(self, config: FHSConfig) -> None:
        self.config = config
        self.margins: dict[str, FHSEngine] = {}
        self.copula: TCopula | None = None
        self.asset_names: list[str] = []

    def fit(self, asset_returns: pd.DataFrame) -> TCopulaFHSEngine:
        frame = asset_returns.astype("float64").dropna(how="any")
        if frame.shape[1] < 2:
            raise FHSError("Route B requires at least two assets")
        if frame.shape[0] < self.config.filter.min_observations:
            raise FHSError(
                f"insufficient observations for FHS: {frame.shape[0]} < "
                f"min_observations={self.config.filter.min_observations}"
            )
        z_cols: list[pd.Series] = []
        for name in frame.columns:
            engine = FHSEngine(self.config).fit(frame[name])
            self.margins[str(name)] = engine
            z_cols.append(engine.z.rename(str(name)))
        z_frame = pd.concat(z_cols, axis=1).dropna()
        uniforms = z_frame.rank(method="average") / (z_frame.shape[0] + 1.0)
        self.copula = TCopula().fit(uniforms.to_numpy(dtype=float))
        self.asset_names = [str(c) for c in z_frame.columns]
        logger.info(
            "Route B fitted N=%s T=%s via t-copula (DCC was not used)",
            len(self.asset_names),
            z_frame.shape[0],
        )
        return self

    def simulate_paths(
        self,
        weights: pd.Series,
        horizon: int,
        n_simulations: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> SimulatedPaths:
        if self.copula is None or not self.margins:
            raise FHSError("Route B simulate_paths() requires fit()")
        if horizon < 1:
            raise FHSError("horizon must be >= 1")
        w = weights.astype("float64").reindex(self.asset_names)
        if w.isna().any():
            raise FHSError(f"weights missing assets {list(w.index[w.isna()])}")
        n_sim = int(n_simulations or self.config.fhs.n_simulations)
        generator = rng or np.random.default_rng(self.config.fhs.seed)
        n_assets = len(self.asset_names)
        asset_daily = np.empty((n_sim, horizon, n_assets), dtype=float)
        sigma2 = np.column_stack(
            [np.full(n_sim, self.margins[name].sigma_one_step**2) for name in self.asset_names]
        )
        last_r = np.array([float(self.margins[name].returns.iloc[-1]) for name in self.asset_names])
        z_pools = [self.margins[name].z.to_numpy(dtype=float) for name in self.asset_names]
        for step in range(horizon):
            uniforms = self.copula.simulate(n_sim, generator)
            for i, name in enumerate(self.asset_names):
                engine = self.margins[name]
                z_star = np.quantile(z_pools[i], uniforms[:, i], method="linear")
                if engine.mean_spec == "AR(1)":
                    lagged = last_r[i] if step == 0 else asset_daily[:, step - 1, i]
                    mu = engine.mu_const + engine.phi * lagged
                else:
                    mu = engine.mu_one_step
                eps = np.sqrt(sigma2[:, i]) * z_star
                asset_daily[:, step, i] = mu + eps
                sigma2[:, i] = np.maximum(
                    engine.omega
                    + (engine.alpha + engine.gamma * (eps < 0.0).astype(float)) * eps * eps
                    + engine.beta * sigma2[:, i],
                    1e-16,
                )
        port = asset_daily @ w.to_numpy(dtype=float)
        aggregated = np.prod(1.0 + port, axis=1) - 1.0
        return SimulatedPaths(daily=port, aggregated=aggregated, aggregation="simple")

    def var_es(
        self,
        weights: pd.Series,
        alpha: float,
        horizon: int = 1,
    ) -> VaRES:
        paths = self.simulate_paths(weights, horizon=horizon)
        sigma = float(
            np.sqrt(
                sum(
                    (float(weights.get(name, 0.0)) * self.margins[name].sigma_one_step) ** 2
                    for name in self.asset_names
                )
            )
        )
        return _var_es_from_draws(
            paths.aggregated,
            alpha=float(alpha),
            horizon=int(horizon),
            sigma_forecast=sigma,
        )


def _as_return_series(returns: pd.Series) -> pd.Series:
    out = returns.astype("float64").dropna().sort_index()
    out.name = out.name or "r"
    if out.index.has_duplicates:
        raise FHSError("duplicate dates in the return series")
    return out


def _plot_comparison(table: pd.DataFrame, dest: Path, plot_cfg: object) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(plot_cfg.output_directory)))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(plot_cfg.figsize_width, plot_cfg.figsize_height))
    ax.plot(table.index, table["return"].to_numpy(), color="0.45", lw=0.7, label="retorno realizado")
    ax.plot(table.index, -table["var_fhs"].to_numpy(), color="tab:blue", label="VaR FHS 99%")
    ax.plot(
        table.index,
        -table["var_historical"].to_numpy(),
        color="tab:orange",
        label="VaR histórico",
    )
    ax.plot(
        table.index,
        -table["var_normal"].to_numpy(),
        color="tab:green",
        label="VaR normal paramétrico",
    )
    viol = table["violation_fhs"].astype(bool)
    if viol.any():
        ax.scatter(
            table.index[viol],
            table.loc[viol, "return"],
            color="tab:red",
            s=16,
            zorder=5,
            label="violación FHS",
        )
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_title("VaR FHS vs histórico vs normal (violaciones FHS en rojo)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=plot_cfg.dpi)
    plt.close(fig)


def _render_report(engine: FHSEngine, table: pd.DataFrame, plot: Path) -> str:
    both = engine.var_es(horizon=1)
    assert isinstance(both, dict)
    n_viol = int(table["violation_fhs"].sum())
    n = int(table.shape[0])
    hit = n_viol / n if n else float("nan")
    n_fit = int(engine.returns.shape[0]) if engine.returns is not None else 0
    z_std = float(engine.z.std(ddof=1)) if engine.z is not None else float("nan")
    lines = [
        "# Filtered Historical Simulation — VaR y Expected Shortfall",
        "",
        "VaR y ES se reportan como **magnitudes de pérdida positivas**.",
        "Agregación multi-día: retornos **simples** "
        r"\(R^*=\prod_j(1+r_j^*)-1\). Prohibido \(\mathrm{VaR}_1\sqrt{h}\).",
        "",
        "## Por qué FHS y no las alternativas",
        "",
        "- **VaR paramétrico normal:** los retornos de crédito tienen curtosis en "
        "exceso; el VaR gaussiano al 99% subestima la pérdida.",
        "- **Simulación histórica pura (ventana 250d en producción):** pondera "
        "igual un día de crisis y un día plácido e ignora la σ de hoy. El VaR "
        "sube cuando el crash ya está en la ventana y baja cuando sale.",
        "- **FHS:** conserva la forma empírica de z y actualiza la escala con "
        r"\(\sigma_{T+1}\) del GJR.",
        "",
        f"- Media: **{engine.mean_spec}**. Filtro: GJR-GARCH(1,1)-skewt.",
        f"- n={n_fit}  sigma_T+1={engine.sigma_one_step:.6f}",
        f"- Residuos reescalados: std muestral(z) = {z_std:.6f}",
        "",
        "## VaR / ES (h = 1)",
        "",
        "| α | VaR | ES |",
        "|---|-----|----|",
    ]
    for alpha in sorted(both):
        item = both[alpha]
        lines.append(f"| {alpha:.3f} | {item.var:.6f} | {item.es:.6f} |")
    lines += [
        "",
        "## Comparación y violaciones",
        "",
        f"Ventana histórica de esta corrida: {engine.config.comparison.historical_window} días "
        f"(250 en producción). Violaciones FHS: **{n_viol}/{n}** (hit rate={hit:.3%}).",
        f"Violaciones histórico: {int(table['violation_historical'].sum())}/{n}. "
        f"Violaciones normal: {int(table['violation_normal'].sum())}/{n}.",
        "",
        f"![comparación]({plot})",
        "",
        "## Ruta B",
        "",
        "Interfaz `TCopulaFHSEngine`: GJR univariado por activo → CDF empírica → "
        "t-cópulas. DCC-GARCH no está implementado a propósito.",
        "",
    ]
    return "\n".join(lines)


def _as_html(markdown: str) -> str:
    escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>"
        "<title>FHS VaR / ES</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:2rem auto;}"
        "pre{white-space:pre-wrap}</style></head><body><pre>"
        f"{escaped}</pre></body></html>\n"
    )
