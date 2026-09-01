"""GJR-GARCH(1,1)-t on the first difference of the Excess Bond Premium.

Target
------
EBP in levels is not stationary in short samples. Work with

.. math::

    \\Delta\\mathrm{EBP}_t = \\mathrm{EBP}_t - \\mathrm{EBP}_{t-1},
    \\qquad
    r_t = -\\Delta\\mathrm{EBP}_t.

No log: EBP can be negative (spreads compressed versus expected default).
Rising EBP is bad news, so the sign is inverted to match
:math:`I[\\varepsilon_{t-1}<0]`.

Variance (Glosten–Jagannathan–Runkle 1993)

.. math::

    \\sigma_t^2
    = \\omega
    + \\alpha\\varepsilon_{t-1}^2
    + \\gamma\\varepsilon_{t-1}^2 I[\\varepsilon_{t-1}<0]
    + \\beta\\sigma_{t-1}^2

with :math:`\\omega>0`, :math:`\\alpha\\ge 0`, :math:`\\alpha+\\gamma\\ge 0`,
:math:`\\beta\\ge 0`, and :math:`\\alpha+\\gamma/2+\\beta<1`. After the sign
convention, the hypothesis is :math:`\\gamma>0`. Half-life uses the GJR
persistence, not the EGARCH :math:`\\beta` formula.

Signal layers (partial circularity)
----------------------------------
Daily EBP is a Chow-Lin output anchored on VIX / HY OAS. Comparing daily
EBP vol to OAS vol is partly circular. Three layers are mandatory:

1. ``primary_monthly`` — official monthly EBP; the only layer that may
   *originate* a decision.
2. ``daily_full`` — intra-month timing after the monthly signal fired.
3. ``daily_vix_only`` — robustness; if the daily signal dies without the
   OAS anchor, it was a disaggregation artifact.

References
----------
Glosten, L. R., Jagannathan, R. and Runkle, D. E. (1993). On the Relation
between the Expected Value and the Volatility of the Nominal Excess Return
on Stocks. *Journal of Finance*, 48(5), 1779–1801.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from arch.univariate import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from data.quality import assert_no_stale_zero_returns
from models.oas_egarch import (
    ForecastResult,
    ModelInvalidError,
    SignConventionError,
    assert_leverage_sign,
)
from models.schema import (
    EbpModelConfig,
    EbpStationarityConfig,
    EbpVarianceConfig,
    ModelConfig,
    load_model_config,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ComparativeEBPReport",
    "EBPVolatilityModel",
    "EbpEstimationReport",
    "EbpStressSeries",
    "SignalLayer",
    "assert_difference_stationary",
    "assert_gjr_constraints",
    "build_ebp_stress_return",
    "gjr_half_life",
    "originate_signal",
    "signal_may_originate",
    "write_comparative_report",
]

SignalLayer = Literal["primary_monthly", "daily_full", "daily_vix_only"]


@dataclass(frozen=True)
class EbpStressSeries:
    r: pd.Series
    level: pd.Series
    delta: pd.Series
    n_dropped_nan: int


@dataclass
class EbpEstimationReport:
    series_id: str
    layer: SignalLayer
    n_obs: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    mean_spec: str
    dist: str
    params: pd.DataFrame
    llf: float
    aic: float
    bic: float
    half_life: float
    residual_ljung_box_pvalue: float
    residual_ljung_box_stat: float
    nu: float | None
    gamma: float | None
    gamma_pvalue: float | None
    leverage_confirmed: bool
    adf_pvalue: float
    kpss_pvalue: float
    omega: float | None
    alpha: float | None
    beta: float | None
    persistence: float | None
    seed: int
    converged: bool
    scale: float
    may_originate: bool

    def as_text(self) -> str:
        return "\n".join(
            [
                f"EBP GJR-GARCH  series={self.series_id} layer={self.layer}",
                f"n_obs={self.n_obs}  window={self.start} / {self.end}",
                f"mean={self.mean_spec}  dist={self.dist}  converged={self.converged}",
                f"llf={self.llf:.4f}  AIC={self.aic:.4f}  BIC={self.bic:.4f}",
                f"half_life={self.half_life:.2f}  persistence={self.persistence}",
                f"gamma={self.gamma}  p={self.gamma_pvalue}  "
                f"leverage_confirmed={self.leverage_confirmed}",
                f"may_originate={self.may_originate}",
                "parameters:",
                self.params.to_string(index=False),
            ]
        )


@dataclass(frozen=True)
class ComparativeEBPReport:
    table: pd.DataFrame
    plot_path: Path
    robustness_correlation: float
    robustness_artifact: bool


def _require_ebp(config: ModelConfig) -> EbpModelConfig:
    if config.ebp is None:
        raise ModelInvalidError("ModelConfig.ebp is required for EBP volatility")
    return config.ebp


def build_ebp_stress_return(ebp_levels: pd.Series, config: ModelConfig) -> EbpStressSeries:
    """Map official or disaggregated EBP levels into a credit-stress return.

    .. math::

        r_t = -(\\mathrm{EBP}_t - \\mathrm{EBP}_{t-1})
    """
    ebp_cfg = _require_ebp(config)
    if not ebp_cfg.invert_sign:
        raise SignConventionError("invert_sign must be true for EBP")
    n_nan = int(ebp_levels.isna().sum())
    cleaned = ebp_levels.dropna().astype("float64")
    if cleaned.index.has_duplicates:
        raise ModelInvalidError("duplicate dates in EBP level series")
    delta = cleaned.diff().dropna()
    r = (-delta).copy()
    r.name = ebp_levels.name
    _assert_mechanical_sign(cleaned, r)
    logger.info(
        "EBP stress return n_raw=%s n_dropped_nan=%s n_obs=%s window=%s/%s",
        int(ebp_levels.shape[0]),
        n_nan,
        int(r.shape[0]),
        r.index.min(),
        r.index.max(),
    )
    return EbpStressSeries(
        r=r,
        level=cleaned.loc[r.index],
        delta=delta,
        n_dropped_nan=n_nan,
    )


def _assert_mechanical_sign(level: pd.Series, r: pd.Series) -> None:
    delta = level.diff()
    frame = pd.concat({"debp": delta, "r": r}, axis=1).dropna()
    rising = frame["debp"] > 0
    falling = frame["debp"] < 0
    if rising.any() and not bool((frame.loc[rising, "r"] < 0).all()):
        raise SignConventionError("sign inversion broken: rising EBP is not r_t < 0")
    if falling.any() and not bool((frame.loc[falling, "r"] > 0).all()):
        raise SignConventionError("sign inversion broken: falling EBP is not r_t > 0")


def assert_difference_stationary(
    levels: pd.Series,
    spec: EbpStationarityConfig,
) -> dict[str, float]:
    """Require :math:`\\Delta\\mathrm{EBP}` to be I(0) by ADF and KPSS ([C7])."""
    diffs = levels.dropna().astype("float64").diff().dropna()
    if diffs.shape[0] < spec.adf_maxlag + 10:
        raise ModelInvalidError("not enough observations for ADF/KPSS on ΔEBP")
    adf = adfuller(
        diffs.to_numpy(),
        maxlag=spec.adf_maxlag,
        regression=spec.adf_regression,
        autolag="AIC",
        result_object=False,
    )
    adf_p = float(adf[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_out = kpss(diffs.to_numpy(), regression=spec.kpss_regression, nlags="auto")
    kpss_p = float(kpss_out[1])
    logger.info("ΔEBP stationarity ADF_p=%.4f KPSS_p=%.4f", adf_p, kpss_p)
    if adf_p > spec.adf_pvalue_max:
        raise ModelInvalidError(
            f"ΔEBP is not ADF-stationary: p={adf_p:.4f} > {spec.adf_pvalue_max}"
        )
    if kpss_p < spec.kpss_pvalue_min:
        raise ModelInvalidError(
            f"ΔEBP fails KPSS stationarity: p={kpss_p:.4f} < {spec.kpss_pvalue_min}"
        )
    return {"adf_pvalue": adf_p, "kpss_pvalue": kpss_p}


def assert_gjr_constraints(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    spec: EbpVarianceConfig | Any,
) -> float:
    """Enforce GJR positivity and :math:`\\alpha+\\gamma/2+\\beta<1`."""
    cap = float(spec.stationarity_alpha_half_gamma_beta_max)
    if omega <= 0.0:
        raise ModelInvalidError(f"GJR omega={omega} must be > 0")
    if alpha < 0.0:
        raise ModelInvalidError(f"GJR alpha={alpha} must be >= 0")
    if alpha + gamma < 0.0:
        raise ModelInvalidError(f"GJR alpha+gamma={alpha + gamma} must be >= 0")
    if beta < 0.0:
        raise ModelInvalidError(f"GJR beta={beta} must be >= 0")
    persistence = alpha + 0.5 * gamma + beta
    if persistence >= cap:
        raise ModelInvalidError(
            f"GJR variance not stationary: α+γ/2+β={persistence} >= {cap}"
        )
    return persistence


def gjr_half_life(
    alpha: float,
    gamma: float,
    beta: float,
    spec: EbpVarianceConfig | Any,
) -> float:
    """Shock half-life :math:`\\ln(m)/\\ln(\\alpha+\\gamma/2+\\beta)`."""
    persistence = assert_gjr_constraints(1.0, alpha, gamma, beta, spec)
    if persistence <= 0.0:
        raise ModelInvalidError(f"GJR half-life undefined for persistence={persistence}")
    return float(math.log(spec.half_life_mass) / math.log(persistence))


def signal_may_originate(layer: SignalLayer, config: ModelConfig) -> bool:
    _require_ebp(config)
    return layer == "primary_monthly"


def originate_signal(
    layer: SignalLayer,
    *,
    fired: bool,
    config: ModelConfig,
) -> bool:
    if not signal_may_originate(layer, config):
        raise ModelInvalidError(
            "daily disaggregated EBP may refine timing only; "
            "it cannot originate a signal"
        )
    return bool(fired)


def write_comparative_report(
    *,
    monthly_official: pd.Series,
    daily_full_anchor: pd.Series,
    daily_vix_only: pd.Series,
    config: ModelConfig,
    path: str | Path | None = None,
) -> ComparativeEBPReport:
    """Three columns, one axis, same scale: monthly vs daily-full vs VIX-only."""
    ebp_cfg = _require_ebp(config)
    table = pd.DataFrame(
        {
            "monthly_official": monthly_official.astype("float64"),
            "daily_full_anchor": daily_full_anchor.reindex(monthly_official.index).astype(
                "float64"
            ),
            "daily_vix_only": daily_vix_only.reindex(monthly_official.index).astype(
                "float64"
            ),
        }
    )
    robustness = float(table["daily_full_anchor"].corr(table["daily_vix_only"]))
    artifact = bool(
        not np.isfinite(robustness) or robustness < ebp_cfg.signal.robustness_min_correlation
    )
    if artifact:
        logger.warning(
            "daily EBP signal is an artifact of the HY OAS anchor: "
            "corr(full, vix-only)=%.3f < %.3f",
            robustness,
            ebp_cfg.signal.robustness_min_correlation,
        )
    plot_cfg = ebp_cfg.plot
    out_dir = Path(plot_cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = Path(path) if path is not None else out_dir / plot_cfg.filename_template.format(
        layer="all"
    )
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(plot_cfg.figsize_width, plot_cfg.figsize_height))
    ax.plot(table.index, table["monthly_official"], label="monthly official (primary)", linewidth=2.0)
    ax.plot(table.index, table["daily_full_anchor"], label="daily full anchors (timing)", alpha=0.85)
    ax.plot(table.index, table["daily_vix_only"], label="daily VIX-only (robustness)", alpha=0.85)
    ax.set_ylabel("conditional volatility (common scale)")
    ax.set_title("EBP volatility: monthly vs daily-full vs VIX-only")
    ax.legend(loc="upper left")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=plot_cfg.dpi)
    plt.close(fig)
    table_path = Path(ebp_cfg.output.comparative_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path)
    logger.info("wrote comparative EBP report plot=%s table=%s", dest, table_path)
    return ComparativeEBPReport(
        table=table,
        plot_path=dest,
        robustness_correlation=robustness,
        robustness_artifact=artifact,
    )


class EBPVolatilityModel:
    """Constant / AR(1) mean + GJR-GARCH(1,1)-t on :math:`r_t=-\\Delta\\mathrm{EBP}_t`."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        series_id: str = "EBP",
        layer: SignalLayer = "primary_monthly",
    ) -> None:
        self.config = config
        self.ebp = _require_ebp(config)
        self.series_id = series_id
        self.layer = layer
        self.stress: EbpStressSeries | None = None
        self.result: Any | None = None
        self.mean_spec: str = "Constant"
        self.report: EbpEstimationReport | None = None
        self.stationarity: dict[str, float] | None = None

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        series_id: str = "EBP",
        layer: SignalLayer = "primary_monthly",
    ) -> EBPVolatilityModel:
        return cls(load_model_config(path), series_id=series_id, layer=layer)

    def fit(self, ebp_levels: pd.Series) -> EBPVolatilityModel:
        if self.layer != "primary_monthly":
            logger.info(
                "fitting %s for intra-month timing only; will not originate a signal",
                self.layer,
            )
        self.stationarity = assert_difference_stationary(ebp_levels, self.ebp.stationarity)
        stress = build_ebp_stress_return(ebp_levels, self.config)
        if stress.r.shape[0] < self.ebp.min_observations:
            raise ModelInvalidError(
                f"insufficient observations for GJR-GARCH: {stress.r.shape[0]} < "
                f"{self.ebp.min_observations}"
            )
        try:
            assert_no_stale_zero_returns(stress.r)
        except ValueError as exc:
            raise ModelInvalidError(str(exc)) from exc
        self.stress = stress
        fitted = self._fit_mean(stress.r, "Constant")
        std = pd.Series(fitted.std_resid).dropna()
        lb_stat, lb_p = _ljung_box(std, self.ebp.mean.ljung_box_lags)
        if lb_p <= self.ebp.mean.ljung_box_pvalue_min:
            logger.info("constant-mean residuals autocorrelated; escalating to AR(1)")
            fitted = self._fit_mean(stress.r, "AR(1)")
            std = pd.Series(fitted.std_resid).dropna()
            lb_stat, lb_p = _ljung_box(std, self.ebp.mean.ljung_box_lags)
        if lb_p <= self.ebp.mean.ljung_box_pvalue_min:
            raise ModelInvalidError(
                f"Ljung-Box Q({self.ebp.mean.ljung_box_lags}) on standardized "
                f"residuals p={lb_p:.4f} <= {self.ebp.mean.ljung_box_pvalue_min}"
            )
        if float(getattr(fitted, "scale", 1.0)) != 1.0:
            raise ModelInvalidError(
                f"arch rescaled the series (scale={fitted.scale}); rescale=False was required"
            )
        params = _param_table(fitted)
        omega = _first_param(params, ("omega",))
        alpha = _first_param(params, ("alpha[1]", "alpha"))
        gamma = _first_param(params, ("gamma[1]", "gamma"))
        beta = _first_param(params, ("beta[1]", "beta"))
        if None in (omega, alpha, gamma, beta):
            raise ModelInvalidError("GJR-GARCH did not return omega/alpha/gamma/beta")
        persistence = assert_gjr_constraints(omega, alpha, gamma, beta, self.ebp.variance)
        vol_index = stress.r.index[-len(fitted.conditional_volatility) :]
        cond_vol = pd.Series(np.asarray(fitted.conditional_volatility), index=vol_index)
        assert_leverage_sign(stress.r.reindex(cond_vol.index), cond_vol, self.ebp.sign_test)
        self.result = fitted
        self.report = EbpEstimationReport(
            series_id=self.series_id,
            layer=self.layer,
            n_obs=int(stress.r.shape[0]),
            start=pd.Timestamp(stress.r.index.min()),
            end=pd.Timestamp(stress.r.index.max()),
            mean_spec=self.mean_spec,
            dist="t",
            params=params,
            llf=float(fitted.loglikelihood),
            aic=float(fitted.aic),
            bic=float(fitted.bic),
            half_life=gjr_half_life(alpha, gamma, beta, self.ebp.variance),
            residual_ljung_box_pvalue=lb_p,
            residual_ljung_box_stat=lb_stat,
            nu=_first_param(params, ("nu", "eta")),
            gamma=gamma,
            gamma_pvalue=_first_pvalue(params, ("gamma[1]", "gamma")),
            leverage_confirmed=True,
            adf_pvalue=self.stationarity["adf_pvalue"],
            kpss_pvalue=self.stationarity["kpss_pvalue"],
            omega=omega,
            alpha=alpha,
            beta=beta,
            persistence=persistence,
            seed=self.config.seed,
            converged=int(getattr(fitted, "convergence_flag", 0)) == 0,
            scale=float(getattr(fitted, "scale", 1.0)),
            may_originate=signal_may_originate(self.layer, self.config),
        )
        logger.info(
            "fit EBP series=%s layer=%s n_obs=%s gamma=%s persistence=%.4f",
            self.series_id,
            self.layer,
            self.report.n_obs,
            gamma,
            persistence,
        )
        return self

    def forecast(self, horizon: int | None = None) -> ForecastResult:
        if self.result is None:
            raise ModelInvalidError("forecast() requires fit()")
        spec = self.ebp.forecast
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
        if self.report is None or self.stationarity is None:
            raise ModelInvalidError("diagnostics() requires fit()")
        return {
            "stationarity": self.stationarity,
            "residual_ljung_box_pvalue": self.report.residual_ljung_box_pvalue,
            "leverage_confirmed": self.report.leverage_confirmed,
            "persistence": self.report.persistence,
            "may_originate": self.report.may_originate,
            "layer": self.layer,
        }

    def summary(self) -> EbpEstimationReport:
        if self.report is None:
            raise ModelInvalidError("summary() requires fit()")
        return self.report

    def conditional_volatility(self) -> pd.Series:
        if self.result is None or self.stress is None:
            raise ModelInvalidError("conditional_volatility() requires fit()")
        n_vol = len(self.result.conditional_volatility)
        return pd.Series(
            np.asarray(self.result.conditional_volatility),
            index=self.stress.r.index[-n_vol:],
            name=f"{self.series_id}_{self.layer}",
        )

    def _fit_mean(self, r: pd.Series, mean_spec: str) -> Any:
        vol = self.ebp.variance
        y = r.astype("float64")
        dist = self.ebp.distribution.candidates[0]
        if dist.lower() in {name.lower() for name in self.ebp.distribution.forbidden}:
            raise ModelInvalidError(f"forbidden innovation density: {dist}")
        kwargs: dict[str, Any] = {
            "vol": vol.vol,
            "p": vol.p,
            "o": vol.o,
            "q": vol.q,
            "dist": dist,
            "rescale": vol.rescale,
        }
        if mean_spec == "Constant":
            model = arch_model(y, mean="Constant", **kwargs)
        elif mean_spec == "AR(1)":
            model = arch_model(y, mean="ARX", lags=self.ebp.mean.ar_lags, **kwargs)
        else:
            raise ModelInvalidError(f"unknown mean specification {mean_spec}")
        result = model.fit(
            disp="off",
            options={"ftol": vol.ftol, "maxiter": vol.maxiter},
        )
        self.mean_spec = mean_spec
        return result


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
    return None


def _ljung_box(std_resid: pd.Series, lags: int) -> tuple[float, float]:
    lb = acorr_ljungbox(std_resid.dropna(), lags=[lags], return_df=True)
    return float(lb["lb_stat"].iloc[-1]), float(lb["lb_pvalue"].iloc[-1])
