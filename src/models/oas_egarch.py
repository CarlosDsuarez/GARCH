"""Asymmetric EGARCH for credit OAS with matrix-pricing mean correction.

Transformation and sign convention
----------------------------------
OAS is bounded below at zero and its volatility scales with the level: a 20 bp
move from 300 bp is not the same event as 20 bp from 800 bp. Work in logs

.. math::

    L_t = \\ln(\\mathrm{OAS}_t),
    \\qquad
    r_t = -100 \\, (L_t - L_{t-1}).

The minus sign is mandatory. GARCH leverage terms (GJR :math:`I[\\varepsilon<0]`,
EGARCH :math:`\\gamma e_{t-1}`) were written for **equity prices**, where bad
news is a negative return. In credit, bad news is a **wider** spread. Feeding
:math:`+100\\Delta L_t` into those indicators activates them on tightenings
and estimates the leverage effect backwards.

Under :math:`r_t` as defined here, :math:`r_t<0` is a widening. Then
EGARCH :math:`\\gamma<0` (GJR :math:`\\gamma>0`) is the central hypothesis.

Matrix pricing
--------------
ICE BofA OAS is matrix-priced. Stale quotes induce spurious AR(1) in
:math:`r_t`. That predictable mean must be absorbed **before** the variance
equation, or GARCH attributes it to :math:`\\alpha` (upward-biased persistence).

References
----------
Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New
Approach. *Econometrica*, 59(2), 347–370.
Hansen, B. E. (1994). Autoregressive Conditional Density Estimation.
*International Economic Review*, 35(3), 705–730.
Lo, A. W. and MacKinlay, A. C. (1988). Stock Market Prices Do Not Follow
Random Walks. *Review of Financial Studies*, 1(1), 41–66.
Ljung, G. M. and Box, G. E. P. (1978). On a Measure of Lack of Fit in Time
Series Models. *Biometrika*, 65(2), 297–303.
Bollerslev, T. (1986). Generalized Autoregressive Conditional
Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch.univariate import ARX, EGARCH, SkewStudent, StudentsT, arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from data.quality import assert_no_stale_zero_returns
from models.schema import (
    MeanConfig,
    ModelConfig,
    SignTestConfig,
    VarianceConfig,
    load_model_config,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CreditStressSeries",
    "EstimationReport",
    "ForecastResult",
    "MeanDiagnostics",
    "ModelInvalidError",
    "OASVolatilityModel",
    "SignConventionError",
    "assert_leverage_sign",
    "build_credit_stress_return",
    "fit_oas_universe",
    "half_life_days",
    "load_model_config",
    "lo_mackinlay_variance_ratio",
    "pre_mean_diagnostics",
]


class ModelInvalidError(ValueError):
    """Mandatory diagnostic failed or the specification is not usable ([C7])."""


class SignConventionError(ModelInvalidError):
    """Credit-stress sign convention is broken."""


@dataclass(frozen=True)
class CreditStressSeries:
    r: pd.Series
    log_level: pd.Series
    oas: pd.Series
    n_dropped_nan: int
    n_nonpositive_dropped: int


@dataclass(frozen=True)
class MeanDiagnostics:
    acf: np.ndarray
    rho_1: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    variance_ratios: pd.DataFrame
    matrix_pricing_contaminated: bool


@dataclass(frozen=True)
class ForecastResult:
    variance: dict[int, float]
    volatility: dict[int, float]
    simulations: int
    seed: int
    method: str
    horizon_max: int


@dataclass
class EstimationReport:
    series_id: str
    n_obs: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    mean_spec: str
    dist: str
    params: pd.DataFrame
    llf: float
    aic: float
    bic: float
    half_life_days: float
    pre_mean: MeanDiagnostics
    residual_ljung_box_pvalue: float
    residual_ljung_box_stat: float
    nu: float | None
    nu_warnings: tuple[str, ...]
    gamma: float | None
    gamma_pvalue: float | None
    gamma_significant: bool
    leverage_confirmed: bool
    symmetric_params: pd.DataFrame | None
    seed: int
    converged: bool
    scale: float
    plot_path: Path | None = None

    def as_text(self) -> str:
        lines = [
            f"OAS EGARCH  series={self.series_id}",
            f"n_obs={self.n_obs}  window={self.start} / {self.end}",
            f"mean={self.mean_spec}  dist={self.dist}  converged={self.converged}",
            f"llf={self.llf:.4f}  AIC={self.aic:.4f}  BIC={self.bic:.4f}",
            f"half_life_days={self.half_life_days:.2f}  seed={self.seed}",
            f"rho_1={self.pre_mean.rho_1:.4f}  LB_p(r)={self.pre_mean.ljung_box_pvalue:.4f}  "
            f"matrix_pricing={self.pre_mean.matrix_pricing_contaminated}",
            f"LB_p(std_resid)={self.residual_ljung_box_pvalue:.4f}",
            f"gamma={self.gamma}  p={self.gamma_pvalue}  "
            f"leverage_confirmed={self.leverage_confirmed}",
            "parameters:",
            self.params.to_string(index=False),
        ]
        if self.nu_warnings:
            lines.append("nu warnings: " + " | ".join(self.nu_warnings))
        if self.symmetric_params is not None:
            lines.append("symmetric EGARCH (o=0) for comparison:")
            lines.append(self.symmetric_params.to_string(index=False))
        return "\n".join(lines)


def build_credit_stress_return(
    oas_series: pd.Series,
    config: ModelConfig,
) -> CreditStressSeries:
    """Map an OAS level series into a credit-stress return.

    .. math::

        L_t = \\ln(\\mathrm{OAS}_t)

        r_t = -S \\, (L_t - L_{t-1})

    with :math:`S` = ``transform.percent_scale`` (percent units). Missing
    *levels* are dropped before the difference ([D1]): forward-fill would
    create exact-zero returns and bias GARCH :math:`\\alpha`.

    A mechanical sign check is mandatory: every day with
    :math:`\\Delta\\mathrm{OAS}_t>0` must have :math:`r_t<0`.
    """
    if not config.transform.invert_sign:
        raise SignConventionError("invert_sign must be true")

    n_raw = int(oas_series.shape[0])
    n_nan = int(oas_series.isna().sum())
    cleaned = oas_series.dropna()
    n_nonpos = int((cleaned <= 0).sum())
    if n_nonpos:
        raise ModelInvalidError(
            f"non-positive OAS values cannot be logged: {n_nonpos} points"
        )
    if cleaned.index.has_duplicates:
        raise ModelInvalidError("duplicate dates in OAS level series")

    log_level = np.log(cleaned.astype("float64"))
    d_log = log_level.diff()
    scale = config.transform.percent_scale
    r = (-scale * d_log).dropna()
    r.name = oas_series.name
    aligned_oas = cleaned.loc[r.index]
    aligned_log = log_level.loc[r.index]

    _assert_mechanical_sign(cleaned, r)

    logger.info(
        "credit-stress return n_raw=%s n_dropped_nan=%s n_obs=%s window=%s/%s",
        n_raw,
        n_nan,
        int(r.shape[0]),
        r.index.min(),
        r.index.max(),
    )
    return CreditStressSeries(
        r=r,
        log_level=aligned_log,
        oas=aligned_oas,
        n_dropped_nan=n_nan,
        n_nonpositive_dropped=n_nonpos,
    )


def _assert_mechanical_sign(oas_level: pd.Series, r: pd.Series) -> None:
    delta = oas_level.diff()
    frame = pd.concat({"doas": delta, "r": r}, axis=1).dropna()
    widening = frame["doas"] > 0
    tightening = frame["doas"] < 0
    if widening.any() and not bool((frame.loc[widening, "r"] < 0).all()):
        raise SignConventionError(
            "sign inversion broken: OAS widening is not mapped to r_t < 0"
        )
    if tightening.any() and not bool((frame.loc[tightening, "r"] > 0).all()):
        raise SignConventionError(
            "sign inversion broken: OAS tightening is not mapped to r_t > 0"
        )


def assert_leverage_sign(
    r: pd.Series,
    cond_vol: pd.Series,
    spec: SignTestConfig,
) -> dict[str, float]:
    """Require higher next-day conditional vol after the widest spread days.

    EGARCH updates :math:`\\sigma_{t+1}` from :math:`e_t`. After the largest
    widenings (:math:`r_t` in the lower tail) mean :math:`\\sigma_{t+1}` must
    exceed the mean after the largest tightenings. Failure means the series
    was fed with the equity sign convention.
    """
    frame = pd.concat({"r": r, "vol": cond_vol}, axis=1).dropna()
    next_vol = frame["vol"].shift(-1)
    q = spec.tail_percentile / 100.0
    wide = frame["r"] <= frame["r"].quantile(q)
    tight = frame["r"] >= frame["r"].quantile(1.0 - q)
    wide_vol = next_vol[wide].dropna()
    tight_vol = next_vol[tight].dropna()
    if (
        wide_vol.shape[0] < spec.min_tail_observations
        or tight_vol.shape[0] < spec.min_tail_observations
    ):
        raise ModelInvalidError(
            "not enough tail observations for the leverage sign test"
        )
    mean_wide = float(wide_vol.mean())
    mean_tight = float(tight_vol.mean())
    logger.info(
        "leverage sign test mean_sigma_after_widening=%.6f after_tightening=%.6f",
        mean_wide,
        mean_tight,
    )
    if mean_wide <= mean_tight:
        raise SignConventionError(
            "conditional volatility is not higher after widenings; "
            "the credit-stress sign convention is inverted"
        )
    return {"mean_vol_after_widening": mean_wide, "mean_vol_after_tightening": mean_tight}


def half_life_days(beta: float, spec: VarianceConfig) -> float:
    """Shock half-life :math:`\\ln(m)/\\ln(\\beta)` for :math:`0<\\beta<1`.

    :math:`m` is ``half_life_mass`` (one half). Requires
    :math:`|\\beta|<` ``stationarity_abs_beta_max``.
    """
    if not (0.0 < abs(beta) < spec.stationarity_abs_beta_max):
        raise ModelInvalidError(
            f"EGARCH beta={beta} violates stationarity |beta| < "
            f"{spec.stationarity_abs_beta_max}"
        )
    if beta <= 0.0:
        raise ModelInvalidError(
            f"half-life is undefined for non-positive beta={beta}"
        )
    return float(math.log(spec.half_life_mass) / math.log(beta))


def lo_mackinlay_variance_ratio(
    r: pd.Series,
    horizons: list[int],
) -> pd.DataFrame:
    """Overlapping Lo–MacKinlay (1988) variance-ratio statistics.

    .. math::

        \\hat\\mu = n^{-1}\\sum r_t,
        \\qquad
        \\hat\\sigma_a^2 = (n-1)^{-1}\\sum (r_t-\\hat\\mu)^2

        m = q(n-q+1)(1-q/n)

        \\hat\\sigma_c^2(q) = m^{-1}\\sum_{t=q}^{n}
            \\Bigl(\\sum_{i=0}^{q-1} r_{t-i} - q\\hat\\mu\\Bigr)^2

        VR(q) = \\hat\\sigma_c^2(q)/\\hat\\sigma_a^2

    Homoskedastic :math:`z` uses
    :math:`\\theta(q)=2(2q-1)(q-1)/(3qn)`.
    Heteroskedasticity-robust :math:`z^*` uses :math:`\\phi(q)` from
    Lo and MacKinlay (1988, eq. 20–22).

    Positive serial correlation (matrix pricing) produces :math:`VR(q)>1`.
    :math:`VR<1` is mean reversion. Both z-statistics are reported.
    """
    x = r.dropna().astype("float64").to_numpy()
    n = int(x.size)
    mu = float(x.mean())
    centered = x - mu
    sigma_a = float(np.sum(centered**2) / (n - 1))
    if sigma_a <= 0.0:
        raise ModelInvalidError("variance ratio undefined for zero-variance series")

    rows: list[dict[str, float]] = []
    for q in horizons:
        if q < 2 or q >= n:
            raise ModelInvalidError(f"variance-ratio horizon q={q} is not usable")
        m = q * (n - q + 1) * (1.0 - q / n)
        rolling = pd.Series(x).rolling(q).sum().dropna().to_numpy()
        sigma_c = float(np.sum((rolling - q * mu) ** 2) / m)
        vr = sigma_c / sigma_a
        theta = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n)
        z_homo = (vr - 1.0) / math.sqrt(theta)
        phi = _lm_phi(centered, q)
        z_het = (vr - 1.0) / math.sqrt(phi) if phi > 0.0 else float("nan")
        p_het = float(2.0 * (1.0 - stats.norm.cdf(abs(z_het)))) if phi > 0.0 else float("nan")
        rows.append(
            {
                "q": q,
                "vr": vr,
                "z_homo": z_homo,
                "z_het": z_het,
                "p_het": p_het,
            }
        )
    return pd.DataFrame(rows)


def _lm_phi(centered: np.ndarray, q: int) -> float:
    n = centered.size
    denom = float(np.sum(centered**2) ** 2)
    if denom <= 0.0:
        return 0.0
    phi = 0.0
    for k in range(1, q):
        num = n * float(np.sum(centered[k:] ** 2 * centered[:-k] ** 2))
        delta = num / denom
        weight = (2.0 * (q - k) / q) ** 2
        phi += weight * delta
    return phi


def pre_mean_diagnostics(r: pd.Series, spec: MeanConfig) -> MeanDiagnostics:
    """ACF, Ljung–Box Q, and Lo–MacKinlay VR on the credit-stress return.

    Matrix-pricing contamination is flagged when
    :math:`\\rho_1` exceeds ``rho1_matrix_pricing_threshold`` **and**
    Ljung–Box rejects at ``ljung_box_pvalue_min``.
    """
    values = r.dropna().astype("float64")
    acf_vals = acf(values, nlags=spec.acf_lags, fft=False, adjusted=False)
    rho_1 = float(acf_vals[1])
    lb = acorr_ljungbox(values, lags=[spec.ljung_box_lags], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[-1])
    lb_p = float(lb["lb_pvalue"].iloc[-1])
    vrs = lo_mackinlay_variance_ratio(values, list(spec.variance_ratio_horizons))
    contaminated = (rho_1 > spec.rho1_matrix_pricing_threshold) and (
        lb_p < spec.ljung_box_pvalue_min
    )
    logger.info(
        "pre-mean diagnostics rho_1=%.4f LB_p=%.4f matrix_pricing=%s",
        rho_1,
        lb_p,
        contaminated,
    )
    return MeanDiagnostics(
        acf=np.asarray(acf_vals, dtype=float),
        rho_1=rho_1,
        ljung_box_stat=lb_stat,
        ljung_box_pvalue=lb_p,
        variance_ratios=vrs,
        matrix_pricing_contaminated=contaminated,
    )


class _ARMA11(ARX):
    """Joint ARMA(1,1) mean: :math:`r_t=\\mu+\\phi r_{t-1}+\\theta e_{t-1}+e_t`."""

    def __init__(self, y: pd.Series, **kwargs: Any) -> None:
        super().__init__(y, lags=1, constant=True, **kwargs)
        self._name = "ARMA(1,1)"

    @cached_property
    def num_params(self) -> int:
        return int(self.regressors.shape[1]) + 1

    def parameter_names(self) -> list[str]:
        return list(super().parameter_names()) + ["theta"]

    def starting_values(self) -> np.ndarray:
        return np.append(super().starting_values(), 0.0)

    def bounds(self) -> list[tuple[float, float]]:
        # ARCHModel.bounds() already allocates self.num_params slots.
        # Appending theta a second time makes SLSQP see more bounds than x0.
        bounds = list(super().bounds())
        bounds[-1] = (-0.999, 0.999)
        return bounds

    def constraints(self) -> tuple[np.ndarray, np.ndarray]:
        a, b = super().constraints()
        if a.size == 0:
            return np.empty((0, self.num_params)), np.empty(0)
        pad = np.zeros((a.shape[0], 1))
        return np.hstack([a, pad]), b

    def resids(
        self,
        params: np.ndarray,
        y: np.ndarray | None = None,
        regressors: np.ndarray | None = None,
    ) -> np.ndarray:
        theta = float(params[-1])
        ar_resid = np.asarray(super().resids(params[:-1], y=y, regressors=regressors))
        errors = np.empty_like(ar_resid, dtype=float)
        errors[0] = ar_resid[0]
        for t in range(1, ar_resid.shape[0]):
            errors[t] = ar_resid[t] - theta * errors[t - 1]
        return errors


class OASVolatilityModel:
    """AR/ARMA + EGARCH(1,1) on credit-stress returns.

    Variance equation (arch / Nelson 1991 convention)

    .. math::

        \\ln\\sigma_t^2
        = \\omega
        + \\alpha\\bigl(|e_{t-1}|-\\mathbb{E}|e_{t-1}|\\bigr)
        + \\gamma e_{t-1}
        + \\beta\\ln\\sigma_{t-1}^2

    with :math:`e_t=\\varepsilon_t/\\sigma_t`. Under Gaussian innovations
    :math:`\\mathbb{E}|e|=\\sqrt{2/\\pi}`; ``arch`` uses the moment that
    matches the chosen :math:`t` / skew-t density.

    ``rescale`` is forced off. Forecasts use simulation (no closed form for
    multi-step :math:`\\mathbb{E}\\exp(\\cdot)`).
    """

    def __init__(self, config: ModelConfig, *, series_id: str) -> None:
        self.config = config
        self.series_id = series_id
        self.stress: CreditStressSeries | None = None
        self.pre_mean: MeanDiagnostics | None = None
        self.result: Any | None = None
        self.mean_spec: str = "AR(1)"
        self.dist: str | None = None
        self.report: EstimationReport | None = None
        self.symmetric_result: Any | None = None

    @classmethod
    def from_yaml(cls, path: str | Path, *, series_id: str) -> OASVolatilityModel:
        return cls(load_model_config(path), series_id=series_id)

    def fit(self, oas_series: pd.Series) -> OASVolatilityModel:
        """Estimate on information contained in ``oas_series`` only ([C1])."""
        self._forbid_gaussian()
        stress = build_credit_stress_return(oas_series, self.config)
        if stress.r.shape[0] < self.config.min_observations:
            raise ModelInvalidError(
                f"insufficient observations for GARCH: {stress.r.shape[0]} < "
                f"{self.config.min_observations}"
            )
        try:
            assert_no_stale_zero_returns(stress.r)
        except ValueError as exc:
            raise ModelInvalidError(str(exc)) from exc
        self.stress = stress
        self.pre_mean = pre_mean_diagnostics(stress.r, self.config.mean)
        fitted = self._fit_mean_and_dist(stress.r, mean_spec="AR(1)")
        std = _std_resid(fitted)
        lb_stat, lb_p = _ljung_box_p(std, self.config.mean.ljung_box_lags)
        if lb_p <= self.config.mean.ljung_box_pvalue_min:
            logger.info("AR(1) residuals still autocorrelated; escalating to ARMA(1,1)")
            fitted = self._fit_mean_and_dist(stress.r, mean_spec="ARMA(1,1)")
            std = _std_resid(fitted)
            lb_stat, lb_p = _ljung_box_p(std, self.config.mean.ljung_box_lags)
        if lb_p <= self.config.mean.ljung_box_pvalue_min:
            raise ModelInvalidError(
                f"Ljung-Box Q({self.config.mean.ljung_box_lags}) on standardized "
                f"residuals p={lb_p:.4f} <= {self.config.mean.ljung_box_pvalue_min}"
            )

        self.result = fitted
        vol_index = stress.r.index[-len(fitted.conditional_volatility) :]
        cond_vol = pd.Series(np.asarray(fitted.conditional_volatility), index=vol_index)
        r_aligned = stress.r.reindex(cond_vol.index)
        assert_leverage_sign(r_aligned, cond_vol, self.config.sign_test)
        self.report = self._build_report(fitted, lb_stat, lb_p)
        logger.info(
            "fit series=%s n_obs=%s window=%s/%s converged=%s llf=%.4f dist=%s mean=%s",
            self.series_id,
            self.report.n_obs,
            self.report.start,
            self.report.end,
            self.report.converged,
            self.report.llf,
            self.report.dist,
            self.report.mean_spec,
        )
        return self

    def forecast(self, horizon: int | None = None) -> ForecastResult:
        if self.result is None:
            raise ModelInvalidError("forecast() requires fit()")
        spec = self.config.forecast
        h_max = int(horizon or max(spec.horizons))
        rng = np.random.RandomState(self.config.seed)
        fcast = self.result.forecast(
            horizon=h_max,
            method=spec.method,
            simulations=spec.simulations,
            reindex=spec.reindex,
            random_state=rng,
        )
        row = fcast.variance.iloc[-1]
        variance: dict[int, float] = {}
        volatility: dict[int, float] = {}
        for h in spec.horizons:
            if h > h_max:
                continue
            col = f"h.{h}"
            if col not in row.index:
                matches = [c for c in row.index if str(c).endswith(str(h))]
                col = matches[0] if matches else row.index[h - 1]
            var_h = float(row[col])
            variance[h] = var_h
            volatility[h] = math.sqrt(var_h) if var_h > 0 else float("nan")
        return ForecastResult(
            variance=variance,
            volatility=volatility,
            simulations=spec.simulations,
            seed=self.config.seed,
            method=spec.method,
            horizon_max=h_max,
        )

    def diagnostics(self) -> dict[str, Any]:
        if self.report is None or self.pre_mean is None:
            raise ModelInvalidError("diagnostics() requires fit()")
        return {
            "pre_mean": self.pre_mean,
            "variance_ratios": self.pre_mean.variance_ratios,
            "residual_ljung_box_pvalue": self.report.residual_ljung_box_pvalue,
            "nu_warnings": self.report.nu_warnings,
            "leverage_confirmed": self.report.leverage_confirmed,
        }

    def summary(self) -> EstimationReport:
        if self.report is None:
            raise ModelInvalidError("summary() requires fit()")
        return self.report

    def plot_conditional_vol(self, path: str | Path | None = None) -> Path:
        if self.result is None or self.stress is None or self.report is None:
            raise ModelInvalidError("plot requires fit()")
        import os

        os.environ.setdefault("MPLCONFIGDIR", str(Path(self.config.plot.output_directory)))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_cfg = self.config.plot
        out_dir = Path(plot_cfg.output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = Path(path) if path is not None else out_dir / plot_cfg.filename_template.format(
            series_id=self.series_id
        )
        n_vol = len(self.result.conditional_volatility)
        vol = pd.Series(
            np.asarray(self.result.conditional_volatility),
            index=self.stress.r.index[-n_vol:],
        )
        fig, ax_oas = plt.subplots(figsize=(plot_cfg.figsize_width, plot_cfg.figsize_height))
        ax_oas.plot(self.stress.oas.index, self.stress.oas.to_numpy(), color="tab:blue", label="OAS")
        ax_oas.set_ylabel("OAS (percent)")
        ax_vol = ax_oas.twinx()
        ax_vol.plot(vol.index, vol.to_numpy(), color="tab:red", alpha=0.8, label="EGARCH sigma")
        ax_vol.set_ylabel("conditional volatility")
        ax_oas.set_title(f"{self.series_id} OAS vs EGARCH conditional volatility")
        fig.tight_layout()
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=plot_cfg.dpi)
        plt.close(fig)
        self.report.plot_path = dest
        return dest

    def _forbid_gaussian(self) -> None:
        for name in self.config.distribution.candidates:
            if name.lower() in {n.lower() for n in self.config.distribution.forbidden}:
                raise ModelInvalidError(f"forbidden innovation density: {name}")
            if name.lower() in {"normal", "gaussian"}:
                raise ModelInvalidError("dist='normal' is forbidden for credit returns")

    def _fit_mean_and_dist(self, r: pd.Series, *, mean_spec: str) -> Any:
        fits: list[tuple[str, Any]] = []
        for dist in self.config.distribution.candidates:
            try:
                res = self._fit_one(r, mean_spec=mean_spec, dist=dist)
            except Exception:
                logger.exception("EGARCH %s + %s failed to converge", mean_spec, dist)
                continue
            if not _converged(res):
                logger.warning("optimizer did not converge for %s %s", mean_spec, dist)
                continue
            if float(getattr(res, "scale", 1.0)) != 1.0:
                raise ModelInvalidError(
                    f"arch rescaled the series (scale={res.scale}); rescale=False was required"
                )
            fits.append((dist, res))
        if not fits:
            raise ModelInvalidError(f"no converging EGARCH specification for {mean_spec}")
        dist, best = min(fits, key=lambda item: float(item[1].bic))
        self.mean_spec = mean_spec
        self.dist = dist
        logger.info("selected dist=%s by BIC=%.4f among %s", dist, float(best.bic), [d for d, _ in fits])
        return best

    def _fit_one(self, r: pd.Series, *, mean_spec: str, dist: str) -> Any:
        vol = self.config.variance
        y = r.astype("float64")
        if mean_spec == "AR(1)":
            model = arch_model(
                y,
                mean="ARX",
                lags=self.config.mean.ar_lags,
                vol=vol.vol,
                p=vol.p,
                o=vol.o,
                q=vol.q,
                dist=dist,
                rescale=vol.rescale,
            )
        elif mean_spec == "ARMA(1,1)":
            volatility = EGARCH(p=vol.p, o=vol.o, q=vol.q)
            distribution = StudentsT() if dist == "t" else SkewStudent()
            model = _ARMA11(
                y,
                volatility=volatility,
                distribution=distribution,
                rescale=vol.rescale,
            )
        else:
            raise ModelInvalidError(f"unknown mean specification {mean_spec}")
        return model.fit(
            disp="off",
            options={"ftol": vol.ftol, "maxiter": vol.maxiter},
        )

    def _build_report(self, fitted: Any, lb_stat: float, lb_p: float) -> EstimationReport:
        assert self.stress is not None and self.pre_mean is not None
        params = _param_table(fitted)
        beta = _first_param(params, ("beta[1]", "beta"))
        gamma = _first_param(params, ("gamma[1]", "gamma"))
        gamma_p = _first_pvalue(params, ("gamma[1]", "gamma"))
        nu = _first_param(params, ("nu", "eta"))
        sig = self.config.variance.significance_level
        gamma_sig = gamma_p is not None and gamma_p < sig
        expected = self.config.variance.expected_egarch_gamma_sign
        leverage = bool(
            gamma is not None and gamma_sig and math.copysign(1, gamma) == expected
        )
        symmetric = None
        if not gamma_sig:
            logger.info("gamma not significant; estimating symmetric EGARCH o=0")
            symmetric = self._fit_symmetric(self.stress.r)
        nu_warnings = _nu_warnings(nu, self.config)
        hl = half_life_days(float(beta), self.config.variance) if beta is not None else float("nan")
        return EstimationReport(
            series_id=self.series_id,
            n_obs=int(self.stress.r.shape[0]),
            start=pd.Timestamp(self.stress.r.index.min()),
            end=pd.Timestamp(self.stress.r.index.max()),
            mean_spec=self.mean_spec,
            dist=str(self.dist),
            params=params,
            llf=float(fitted.loglikelihood),
            aic=float(fitted.aic),
            bic=float(fitted.bic),
            half_life_days=hl,
            pre_mean=self.pre_mean,
            residual_ljung_box_pvalue=lb_p,
            residual_ljung_box_stat=lb_stat,
            nu=nu,
            nu_warnings=tuple(nu_warnings),
            gamma=gamma,
            gamma_pvalue=gamma_p,
            gamma_significant=gamma_sig,
            leverage_confirmed=leverage,
            symmetric_params=symmetric,
            seed=self.config.seed,
            converged=_converged(fitted),
            scale=float(getattr(fitted, "scale", 1.0)),
        )

    def _fit_symmetric(self, r: pd.Series) -> pd.DataFrame | None:
        vol = self.config.variance
        try:
            model = arch_model(
                r.astype("float64"),
                mean="ARX",
                lags=self.config.mean.ar_lags,
                vol=vol.vol,
                p=vol.p,
                o=0,
                q=vol.q,
                dist=self.dist or "t",
                rescale=vol.rescale,
            )
            res = model.fit(disp="off", options={"ftol": vol.ftol, "maxiter": vol.maxiter})
            self.symmetric_result = res
            return _param_table(res)
        except Exception:
            logger.exception("symmetric EGARCH comparison failed")
            return None


def fit_oas_universe(
    series_by_id: dict[str, pd.Series],
    config: ModelConfig,
) -> pd.DataFrame:
    """Estimate the OAS panel and return a comparative parameter table."""
    rows: list[dict[str, Any]] = []
    for series_id, oas in series_by_id.items():
        model = OASVolatilityModel(config, series_id=series_id)
        model.fit(oas)
        report = model.summary()
        try:
            model.plot_conditional_vol()
        except Exception:
            logger.exception("plot failed for %s", series_id)
        meta = config.oas_universe.get(series_id)
        rows.append(
            {
                "series_id": series_id,
                "label": meta.label if meta else series_id,
                "quality_rank": meta.quality_rank if meta else None,
                "n_obs": report.n_obs,
                "mean_spec": report.mean_spec,
                "dist": report.dist,
                "llf": report.llf,
                "aic": report.aic,
                "bic": report.bic,
                "rho_1": report.pre_mean.rho_1,
                "matrix_pricing": report.pre_mean.matrix_pricing_contaminated,
                "gamma": report.gamma,
                "gamma_pvalue": report.gamma_pvalue,
                "leverage_confirmed": report.leverage_confirmed,
                "beta": _first_param(report.params, ("beta[1]", "beta")),
                "alpha": _first_param(report.params, ("alpha[1]", "alpha")),
                "omega": _first_param(report.params, ("omega",)),
                "nu": report.nu,
                "half_life_days": report.half_life_days,
                "lb_p_std_resid": report.residual_ljung_box_pvalue,
                "seed": report.seed,
            }
        )
    table = pd.DataFrame(rows)
    if "quality_rank" in table.columns:
        table = table.sort_values("quality_rank")
    return table


def _param_table(result: Any) -> pd.DataFrame:
    stderr = getattr(result, "std_err", None)
    if stderr is None:
        stderr = result.std_errors
    return pd.DataFrame(
        {
            "name": list(result.params.index),
            "value": result.params.to_numpy(dtype=float),
            "std_err": stderr.to_numpy(dtype=float),
            "tstat": result.tvalues.to_numpy(dtype=float),
            "pvalue": result.pvalues.to_numpy(dtype=float),
        }
    )


def _first_param(params: pd.DataFrame, names: tuple[str, ...]) -> float | None:
    indexed = params.set_index("name")
    for name in names:
        if name in indexed.index:
            return float(indexed.loc[name, "value"])
    for name in indexed.index:
        if any(key in str(name) for key in names):
            return float(indexed.loc[name, "value"])
    return None


def _first_pvalue(params: pd.DataFrame, names: tuple[str, ...]) -> float | None:
    indexed = params.set_index("name")
    for name in names:
        if name in indexed.index:
            return float(indexed.loc[name, "pvalue"])
    for name in indexed.index:
        if any(key in str(name) for key in names):
            return float(indexed.loc[name, "pvalue"])
    return None


def _std_resid(result: Any) -> pd.Series:
    return pd.Series(result.std_resid).dropna()


def _ljung_box_p(std_resid: pd.Series, lags: int) -> tuple[float, float]:
    lb = acorr_ljungbox(std_resid.dropna(), lags=[lags], return_df=True)
    return float(lb["lb_stat"].iloc[-1]), float(lb["lb_pvalue"].iloc[-1])


def _converged(result: Any) -> bool:
    flag = getattr(result, "convergence_flag", 0)
    return int(flag) == 0


def _nu_warnings(nu: float | None, config: ModelConfig) -> list[str]:
    if nu is None:
        return []
    notes: list[str] = []
    if nu < config.distribution.nu_fourth_moment_min:
        notes.append(
            f"nu={nu:.3f} < {config.distribution.nu_fourth_moment_min}: "
            "fourth moment does not exist; variance forecasts remain valid "
            "but sample-kurtosis diagnostics do not"
        )
    if nu > config.distribution.nu_normal_collapse:
        notes.append(
            f"nu={nu:.3f} > {config.distribution.nu_normal_collapse}: "
            "Student-t is collapsing toward the normal; check sample length "
            "and whether a single extreme event dominates"
        )
    return notes
