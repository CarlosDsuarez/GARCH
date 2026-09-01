"""Pre- and post-estimation diagnostic battery for GARCH fits ([C7]).

Block A runs on the input series *before* estimation. Block B runs on the
standardized residuals :math:`z_t=\\varepsilon_t/\\sigma_t` *after* estimation.
A blocking FAIL raises :class:`DiagnosticGateError` — the pipeline does not
log-and-continue.

Stationarity (A1)
-----------------
ADF :math:`H_0`: unit root. KPSS :math:`H_0`: stationarity. The nulls are
opposites; both tests are required.

    ADF reject + KPSS not reject → stationary, proceed.
    ADF not reject + KPSS reject → non-stationary, difference.
    Both reject or neither rejects → ambiguous (fractional integration or
    a structural break). Investigate; do not ignore.

ARCH LM (A2 / B2)
-----------------
Engle (1982). Regress :math:`\\varepsilon_t^2` on :math:`q` own lags:

.. math::

    \\varepsilon_t^2 = a_0 + \\sum_{i=1}^{q} a_i \\varepsilon_{t-i}^2 + u_t,
    \\qquad n R^2 \\sim \\chi^2(q).

If A2 does not reject, GARCH is unnecessary — that is a **WARN**, never
hidden. If B2 rejects, the variance equation failed; that is a **FAIL**.

Engle–Ng sign bias (B3)
-----------------------
Regress :math:`z_t^2` on :math:`S^-_{t-1}=I[\\varepsilon_{t-1}<0]`,
:math:`S^-_{t-1}\\varepsilon_{t-1}`, and
:math:`(1-S^-_{t-1})\\varepsilon_{t-1}`. Joint significance of the three
coefficients. Rejection after EGARCH/GJR means the asymmetric spec did not
capture the leverage effect.

Parameter stationarity (B5)
---------------------------
EGARCH: :math:`|\\beta|<1`. GJR: :math:`\\alpha+\\gamma/2+\\beta<1`.
Persistence above ``igarch_persistence`` is the IGARCH frontier: shocks
never decay and long-horizon forecasts diverge. Half-life
:math:`\\ln(m)/\\ln(\\text{persistence})`.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

from diagnostics.schema import DiagnosticConfig, load_diagnostics_config

logger = logging.getLogger(__name__)

Verdict = Literal["PASS", "WARN", "FAIL"]
Family = Literal["EGARCH", "GJR", "GARCH"]
DistName = Literal["normal", "t", "skewt"]

_ADF_REGRESSION_LABEL = {
    "c": "constant, no trend",
    "ct": "constant and linear trend",
    "ctt": "constant, linear and quadratic trend",
    "n": "no constant, no trend",
}

__all__ = [
    "DiagnosticGateError",
    "DiagnosticReport",
    "DiagnosticResult",
    "DiagnosticSuite",
    "FittedGarchSnapshot",
    "diagnostic_gate",
    "joint_stationarity_verdict",
    "render_markdown",
    "require_post_estimation",
]


class DiagnosticGateError(Exception):
    """Blocking diagnostic FAIL — do not use the series or the fitted model."""

    def __init__(self, report: DiagnosticReport) -> None:
        self.report = report
        names = ", ".join(r.name for r in report.blocking_failures) or "unknown"
        super().__init__(f"Diagnostic gate FAIL ({report.stage}): {names}")


@dataclass(frozen=True)
class DiagnosticResult:
    name: str
    statistic: float | None
    pvalue: float | None
    criterion: str
    verdict: Verdict
    message: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticReport:
    stage: Literal["pre", "post"]
    series_name: str
    n_obs: int
    results: tuple[DiagnosticResult, ...]
    blocking_codes: tuple[str, ...] = ()

    def by_name(self, name: str) -> DiagnosticResult:
        for row in self.results:
            if row.name == name:
                return row
        available = [row.name for row in self.results]
        raise KeyError(f"{name!r} not in report {available}")

    @property
    def blocking_failures(self) -> tuple[DiagnosticResult, ...]:
        return tuple(
            row
            for row in self.results
            if row.verdict == "FAIL" and self._blocks(row.name)
        )

    @property
    def has_blocking_fail(self) -> bool:
        return bool(self.blocking_failures)

    def _blocks(self, name: str) -> bool:
        if name.startswith("A1."):
            if name != "A1.joint":
                return False
            return "A1" in self.blocking_codes or "A1.joint" in self.blocking_codes
        prefix = name.split(".", 1)[0]
        return prefix in self.blocking_codes or name in self.blocking_codes

    def to_markdown(self) -> str:
        return render_markdown(self)


@dataclass(frozen=True)
class FittedGarchSnapshot:
    """Adapter so the suite does not import OAS / EBP / FHS fit objects."""

    z: pd.Series
    family: Family
    dist: DistName
    alpha: float
    beta: float
    gamma: float = 0.0
    omega: float = 1e-6
    eps: pd.Series | None = None
    nu: float | None = None
    lambda_skew: float | None = None
    converged: bool = True
    loglikelihood: float = 0.0
    restart_loglikelihoods: tuple[float, ...] | None = None
    name: str = "garch"

    def residuals(self) -> pd.Series:
        if self.eps is None:
            raise ValueError(
                "eps is required; Engle–Ng size-bias terms cannot substitute z_t"
            )
        return self.eps


def joint_stationarity_verdict(adf_p: float, kpss_p: float, alpha: float) -> Verdict:
    """Map the opposite-null pair (ADF, KPSS) onto PASS / WARN / FAIL."""
    adf_reject = adf_p < alpha
    kpss_reject = kpss_p < alpha
    if adf_reject and not kpss_reject:
        return "PASS"
    if (not adf_reject) and kpss_reject:
        return "FAIL"
    return "WARN"


def render_markdown(report: DiagnosticReport) -> str:
    """Render a DiagnosticReport as a Markdown table for backtest reports."""
    lines = [
        f"# Econometric diagnostics ({report.stage})",
        "",
        f"Series: `{report.series_name}` · n = {report.n_obs}",
        "",
        "| Test | Statistic | p-value | Criterion | Verdict | Interpretation |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in report.results:
        lines.append(
            "| {name} | {stat} | {pval} | {crit} | **{verdict}** | {msg} |".format(
                name=_md_cell(row.name),
                stat=_fmt_num(row.statistic),
                pval=_fmt_num(row.pvalue),
                crit=_md_cell(row.criterion),
                verdict=row.verdict,
                msg=_md_cell(row.message),
            )
        )
    fails = report.blocking_failures
    lines.append("")
    if fails:
        names = ", ".join(row.name for row in fails)
        lines.append(f"**Blocking FAIL:** {names}. Do not use this model.")
    else:
        lines.append("No blocking FAIL.")
    return "\n".join(lines) + "\n"


def require_post_estimation(
    suite: DiagnosticSuite,
    *,
    adapter: Callable[[Any], FittedGarchSnapshot] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Run block B after a fit and raise if a blocking test FAILs."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            fitted = adapter(result) if adapter is not None else result
            if not isinstance(fitted, FittedGarchSnapshot):
                raise TypeError(
                    "require_post_estimation expected FittedGarchSnapshot "
                    f"(got {type(fitted).__name__}); pass adapter="
                )
            report = suite.run_post_estimation(fitted)
            suite.last_post_report = report
            if report.has_blocking_fail:
                raise DiagnosticGateError(report)
            return result

        return wrapped

    return decorator


@contextmanager
def diagnostic_gate(
    series: pd.Series,
    suite: DiagnosticSuite,
    *,
    fitted_key: str = "fitted",
) -> Iterator[dict[str, Any]]:
    """Run block A, yield a bag for the fit, then run block B."""
    pre = suite.run_pre_estimation(series)
    suite.last_pre_report = pre
    if pre.has_blocking_fail:
        raise DiagnosticGateError(pre)
    bag: dict[str, Any] = {}
    yield bag
    fitted = bag.get(fitted_key)
    if not isinstance(fitted, FittedGarchSnapshot):
        raise TypeError(
            "diagnostic_gate bag must set 'fitted' to a FittedGarchSnapshot"
        )
    post = suite.run_post_estimation(fitted)
    suite.last_post_report = post
    if post.has_blocking_fail:
        raise DiagnosticGateError(post)


class DiagnosticSuite:
    """[C7] quality gate: pre-estimation on the series, post-estimation on z_t."""

    def __init__(self, config: DiagnosticConfig) -> None:
        self.config = config
        self.last_pre_report: DiagnosticReport | None = None
        self.last_post_report: DiagnosticReport | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> DiagnosticSuite:
        return cls(load_diagnostics_config(path))

    def run_pre_estimation(self, series: pd.Series) -> DiagnosticReport:
        x = _as_series(series)
        logger.info("pre-estimation diagnostics series=%s n=%d", x.name, int(x.size))
        rows: list[DiagnosticResult] = []
        rows.extend(self._a1_stationarity(x))
        rows.extend(self._a2_arch_lm(x))
        rows.extend(self._a3_mean_acf(x))
        rows.append(self._a4_normality(x))
        rows.extend(self._a5_breaks(x))
        report = DiagnosticReport(
            stage="pre",
            series_name=str(x.name),
            n_obs=int(x.size),
            results=tuple(rows),
            blocking_codes=tuple(self.config.blocking_pre),
        )
        self.last_pre_report = report
        _log_report(report)
        return report

    def run_post_estimation(
        self,
        fitted_model: FittedGarchSnapshot,
        *,
        qq_path: Path | None = None,
    ) -> DiagnosticReport:
        z = _as_series(fitted_model.z, name="z")
        logger.info(
            "post-estimation diagnostics family=%s dist=%s n=%d",
            fitted_model.family,
            fitted_model.dist,
            int(z.size),
        )
        dest = (
            qq_path
            if qq_path is not None
            else Path(self.config.plot.output_directory) / self.config.plot.qq_filename
        )
        rows: list[DiagnosticResult] = [
            self._b1_mean_lb(z),
            *self._b2_residual_arch(z),
            self._b3_engle_ng(z, fitted_model),
            self._b4_distribution(z, fitted_model, dest),
            self._b5_stationarity(fitted_model),
            *self._b6_optimizer(fitted_model),
        ]
        report = DiagnosticReport(
            stage="post",
            series_name=str(fitted_model.name),
            n_obs=int(z.size),
            results=tuple(rows),
            blocking_codes=tuple(self.config.blocking_post),
        )
        self.last_post_report = report
        _log_report(report)
        return report

    def _a1_stationarity(self, x: pd.Series) -> list[DiagnosticResult]:
        alpha = self.config.significance
        adf_cfg = self.config.adf
        trend_label = _ADF_REGRESSION_LABEL[adf_cfg.regression]
        adf_out = adfuller(
            x.to_numpy(),
            autolag=adf_cfg.autolag,
            regression=adf_cfg.regression,
            result_object=False,
        )
        adf_stat, adf_p, adf_lag = float(adf_out[0]), float(adf_out[1]), int(adf_out[2])
        adf_reject = adf_p < alpha
        adf = DiagnosticResult(
            name="A1.ADF",
            statistic=adf_stat,
            pvalue=adf_p,
            criterion=(
                f"ADF autolag={adf_cfg.autolag}, regression={adf_cfg.regression} "
                f"({trend_label}). Reject H0 (unit root) at {alpha:g}."
            ),
            verdict="PASS" if adf_reject else "FAIL",
            message=(
                f"ADF {'rejects' if adf_reject else 'does not reject'} a unit root "
                f"(lags selected by {adf_cfg.autolag}: {adf_lag})."
            ),
            extras={"lags": adf_lag, "autolag": adf_cfg.autolag, "regression": adf_cfg.regression},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_out = kpss(
                x.to_numpy(),
                regression=self.config.kpss.regression,
                nlags=self.config.kpss.nlags,
                result_object=False,
            )
        kpss_stat, kpss_p = float(kpss_out[0]), float(kpss_out[1])
        kpss_reject = kpss_p < alpha
        kpss_row = DiagnosticResult(
            name="A1.KPSS",
            statistic=kpss_stat,
            pvalue=kpss_p,
            criterion=(
                f"KPSS regression={self.config.kpss.regression} "
                f"({trend_label}), nlags={self.config.kpss.nlags}. "
                f"Do not reject H0 (stationarity) at {alpha:g}."
            ),
            verdict="FAIL" if kpss_reject else "PASS",
            message=(
                f"KPSS {'rejects' if kpss_reject else 'does not reject'} stationarity."
            ),
            extras={"regression": self.config.kpss.regression, "nlags": self.config.kpss.nlags},
        )
        joint_v = joint_stationarity_verdict(adf_p, kpss_p, alpha)
        joint_msg = {
            "PASS": (
                "ADF rejects a unit root and KPSS does not reject stationarity; "
                "the series is stationary. Proceed."
            ),
            "FAIL": (
                "ADF does not reject a unit root and KPSS rejects stationarity; "
                "the series is not stationary. Difference before estimating GARCH."
            ),
            "WARN": (
                "ADF and KPSS disagree or are both inconclusive "
                "(both reject, or neither rejects). Possible fractional integration "
                "or a structural break. Investigate before proceeding; do not ignore."
            ),
        }[joint_v]
        joint = DiagnosticResult(
            name="A1.joint",
            statistic=adf_stat,
            pvalue=adf_p,
            criterion=(
                "ADF reject + KPSS not reject → stationary; "
                "ADF not reject + KPSS reject → difference; "
                "both/neither → ambiguous."
            ),
            verdict=joint_v,
            message=joint_msg,
            extras={"adf_pvalue": adf_p, "kpss_pvalue": kpss_p},
        )
        return [adf, kpss_row, joint]

    def _a2_arch_lm(self, x: pd.Series) -> list[DiagnosticResult]:
        alpha = self.config.significance
        rows: list[DiagnosticResult] = []
        for q in self.config.arch_lm_lags:
            stat, pval = _engle_arch_lm(x, q)
            present = pval < alpha
            if present:
                verdict: Verdict = "PASS"
                message = (
                    f"Engle ARCH LM(q={q}) rejects: conditional heteroskedasticity "
                    "is present. A GARCH variance equation is justified."
                )
            else:
                verdict = "WARN"
                message = (
                    f"Engle ARCH LM(q={q}) does not reject: no conditional "
                    "heteroskedasticity detected. A GARCH model is unnecessary "
                    "for this series. This is a legitimate result and must be reported."
                )
            rows.append(
                DiagnosticResult(
                    name=f"A2.ARCH_LM_{q}",
                    statistic=stat,
                    pvalue=pval,
                    criterion=f"n R^2 ~ chi^2({q}); reject H0 (no ARCH) at {alpha:g} to justify GARCH.",
                    verdict=verdict,
                    message=message,
                    extras={"lags": q},
                )
            )
        return rows

    def _a3_mean_acf(self, x: pd.Series) -> list[DiagnosticResult]:
        alpha = self.config.significance
        rho1 = float(x.autocorr(lag=1))
        rho_warn = abs(rho1) > self.config.rho1_warn
        rho_row = DiagnosticResult(
            name="A3.rho1",
            statistic=rho1,
            pvalue=None,
            criterion=(
                f"|rho_1| ≤ {self.config.rho1_warn:g} (matrix-pricing diagnostic)."
            ),
            verdict="WARN" if rho_warn else "PASS",
            message=(
                f"rho_1={rho1:.3f}. "
                + (
                    "Large lag-1 autocorrelation is the matrix-pricing fingerprint; "
                    "the mean equation needs AR lags or ARMA(1,1)."
                    if rho_warn
                    else "Lag-1 autocorrelation is within the matrix-pricing threshold."
                )
            ),
            extras={"rho_1": rho1},
        )
        rows = [rho_row]
        for lag in self.config.ljung_box_lags:
            stat, pval = _ljung_box(x, lag)
            reject = pval < alpha
            rows.append(
                DiagnosticResult(
                    name=f"A3.LjungBox_{lag}",
                    statistic=stat,
                    pvalue=pval,
                    criterion=f"Ljung-Box Q({lag}) p-value > {alpha:g} (no mean autocorrelation).",
                    verdict="WARN" if reject else "PASS",
                    message=(
                        f"Ljung-Box Q({lag}) {'rejects' if reject else 'does not reject'} "
                        "white-noise in the mean. "
                        + (
                            "Add AR lags or ARMA(1,1) before GARCH."
                            if reject
                            else "No evidence of mean autocorrelation at this lag."
                        )
                    ),
                    extras={"lags": lag, "rho_1": rho1},
                )
            )
        return rows

    def _a4_normality(self, x: pd.Series) -> DiagnosticResult:
        """Quantify non-normality. Never FAIL: Gaussian innovations are already forbidden."""
        values = x.to_numpy(dtype=float)
        jb_stat, jb_p = scipy_stats.jarque_bera(values)
        sk = float(scipy_stats.skew(values, bias=False))
        kt = float(scipy_stats.kurtosis(values, fisher=False, bias=False))
        reject = float(jb_p) < self.config.significance
        if reject:
            verdict: Verdict = "WARN"
            message = (
                f"Jarque-Bera rejects normality (skew={sk:.3f}, kurtosis={kt:.3f}). "
                "This quantifies the case for t / skew-t; it is not a licence to use Gaussian innovations."
            )
        else:
            verdict = "PASS"
            message = (
                f"Jarque-Bera does not reject normality (skew={sk:.3f}, kurtosis={kt:.3f}). "
                "Gaussian innovations remain forbidden for credit returns."
            )
        return DiagnosticResult(
            name="A4.JarqueBera",
            statistic=float(jb_stat),
            pvalue=float(jb_p),
            criterion=(
                "Quantify skew and kurtosis; do not use this test to choose a Normal "
                "(already forbidden)."
            ),
            verdict=verdict,
            message=message,
            extras={"skew": sk, "kurtosis": kt},
        )

    def _a5_breaks(self, x: pd.Series) -> list[DiagnosticResult]:
        alpha = self.config.significance
        r2 = (x ** 2).to_numpy(dtype=float)
        try:
            cusum_stat, cusum_p = _cusum_of_squares(r2)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("CUSUM of squares failed: %s", exc)
            cusum_stat, cusum_p = float("nan"), float("nan")
        cusum_reject = math.isfinite(cusum_p) and cusum_p < alpha
        break_idx = _icss_break_indices(r2, self.config.icss_critical)
        dates = [_index_stamp(x.index, i) for i in break_idx]
        if break_idx:
            icss_verdict: Verdict = "WARN"
            icss_stat = float(len(break_idx))
            icss_msg = (
                "Variance breaks at "
                + ", ".join(dates)
                + ". Full-sample GARCH mixes regimes and produces parameters "
                "that describe none of them. This is direct evidence for MS-GARCH."
            )
        else:
            icss_verdict = "PASS"
            icss_stat = 0.0
            icss_msg = "Inclán–Tiao ICSS finds no variance break."
        cusum = DiagnosticResult(
            name="A5.CUSUM",
            statistic=cusum_stat,
            pvalue=cusum_p,
            criterion=f"CUSUM of squares on r_t^2; reject stability at {alpha:g}.",
            verdict="WARN" if cusum_reject or not math.isfinite(cusum_p) else "PASS",
            message=(
                "CUSUM of squares could not be computed; do not treat this as a non-rejection."
                if not math.isfinite(cusum_p)
                else (
                    "CUSUM of squares rejects variance constancy; inspect ICSS dates."
                    if cusum_reject
                    else "CUSUM of squares does not reject variance constancy."
                )
            ),
            extras={"break_dates": dates},
        )
        icss = DiagnosticResult(
            name="A5.ICSS",
            statistic=icss_stat,
            pvalue=None,
            criterion=(
                f"Inclán–Tiao ICSS: max |sqrt(T/2) D_k| > {self.config.icss_critical:g}."
            ),
            verdict=icss_verdict,
            message=icss_msg,
            extras={"break_dates": dates, "break_indices": break_idx},
        )
        return [cusum, icss]

    def _b1_mean_lb(self, z: pd.Series) -> DiagnosticResult:
        lag = int(self.config.ljung_box_lags[0])
        alpha = self.config.significance
        stat, pval = _ljung_box(z, lag)
        ok = pval > alpha
        return DiagnosticResult(
            name="B1.LjungBox_z",
            statistic=stat,
            pvalue=pval,
            criterion=f"Ljung-Box Q({lag}) on z_t: p-value > {alpha:g}.",
            verdict="PASS" if ok else "FAIL",
            message=(
                "Standardized residuals have no remaining mean autocorrelation."
                if ok
                else (
                    "Ljung-Box on z_t rejects: the mean equation is misspecified. "
                    "Add AR lags or pass to ARMA(1,1). Do not use this model."
                )
            ),
            extras={"lags": lag},
        )

    def _b2_residual_arch(self, z: pd.Series) -> list[DiagnosticResult]:
        lag = int(self.config.ljung_box_lags[0])
        alpha = self.config.significance
        z2 = z ** 2
        lb_stat, lb_p = _ljung_box(z2, lag)
        lb_ok = lb_p > alpha
        fail_msg = (
            "Leftover ARCH: the variance equation did not capture all conditional "
            "heteroskedasticity. Try p=2 or q=2, or a different specification. "
            "A GARCH that leaves ARCH in the residuals has not done its job."
        )
        rows = [
            DiagnosticResult(
                name="B2.LjungBox_z2",
                statistic=lb_stat,
                pvalue=lb_p,
                criterion=f"Ljung-Box Q({lag}) on z_t^2: p-value > {alpha:g}.",
                verdict="PASS" if lb_ok else "FAIL",
                message=(
                    "No remaining autocorrelation in squared standardized residuals."
                    if lb_ok
                    else fail_msg
                ),
                extras={"lags": lag},
            )
        ]
        max_q = max(self.config.arch_lm_lags)
        for q in self.config.arch_lm_lags:
            lm_stat, lm_p = _engle_arch_lm(z, q)
            lm_ok = lm_p > alpha
            names = [f"B2.ARCH_LM_{q}"]
            if q == max_q:
                names.append("B2.ARCH_LM")
            for name in names:
                rows.append(
                    DiagnosticResult(
                        name=name,
                        statistic=lm_stat,
                        pvalue=lm_p,
                        criterion=f"Engle ARCH LM(q={q}) on z_t: p-value > {alpha:g}.",
                        verdict="PASS" if lm_ok else "FAIL",
                        message=(
                            "Engle LM finds no leftover ARCH in standardized residuals."
                            if lm_ok
                            else fail_msg
                        ),
                        extras={"lags": q},
                    )
                )
        return rows

    def _b3_engle_ng(self, z: pd.Series, fitted: FittedGarchSnapshot) -> DiagnosticResult:
        alpha = self.config.significance
        family = fitted.family
        if fitted.eps is None:
            return DiagnosticResult(
                name="B3.EngleNg",
                statistic=None,
                pvalue=None,
                criterion=(
                    "Joint significance of sign, negative-size and positive-size bias. "
                    "Size terms require ε_{t-1}, not z_{t-1}."
                ),
                verdict="FAIL",
                message=(
                    "eps was not supplied. Engle–Ng size-bias terms use ε_{t-1}; "
                    "substituting z_t would make the test uninformative. Do not use this model."
                ),
                extras={"family": family},
            )
        eps = _as_series(fitted.eps, name="eps")
        if not z.index.equals(eps.index):
            return DiagnosticResult(
                name="B3.EngleNg",
                statistic=None,
                pvalue=None,
                criterion="z_t and ε_t must share an index.",
                verdict="FAIL",
                message="z and eps indexes do not match; Engle–Ng cannot be computed.",
                extras={"family": family},
            )
        stat, pval = _engle_ng_joint(z, eps)
        reject = pval < alpha
        asymmetric = family in {"EGARCH", "GJR"}
        if not reject:
            verdict: Verdict = "PASS"
            message = (
                "Engle–Ng joint test does not reject: no remaining sign / size bias."
            )
        elif asymmetric:
            verdict = "FAIL"
            message = (
                f"Engle–Ng rejects after {family}: the asymmetric specification did not "
                "capture the leverage effect. The EGARCH/GJR choice is empirically unsupported."
            )
        else:
            verdict = "WARN"
            message = (
                "Engle–Ng rejects on a symmetric GARCH: this is evidence to use "
                "EGARCH or GJR rather than a symmetric variance equation."
            )
        return DiagnosticResult(
            name="B3.EngleNg",
            statistic=stat,
            pvalue=pval,
            criterion=(
                "Joint significance of sign, negative-size and positive-size bias; "
                f"p-value > {alpha:g}. Blocking FAIL only after EGARCH/GJR."
            ),
            verdict=verdict,
            message=message,
            extras={"family": family},
        )

    def _b4_distribution(
        self,
        z: pd.Series,
        fitted: FittedGarchSnapshot,
        qq_path: Path,
    ) -> DiagnosticResult:
        values = z.to_numpy(dtype=float)
        try:
            cdf = _assumed_cdf(fitted)
            ks_stat, ks_p = scipy_stats.kstest(values, cdf)
            _write_qq_plot(values, fitted, qq_path, self.config)
        except ValueError as exc:
            return DiagnosticResult(
                name="B4.KS",
                statistic=None,
                pvalue=None,
                criterion=(
                    f"Kolmogorov–Smirnov of z_t vs {fitted.dist} (estimated parameters). "
                    "QQ-plot is mandatory."
                ),
                verdict="WARN",
                message=str(exc),
                extras={"qq_path": str(qq_path), "dist": fitted.dist, "nu": fitted.nu},
            )
        alpha = self.config.significance
        ok = float(ks_p) > alpha
        return DiagnosticResult(
            name="B4.KS",
            statistic=float(ks_stat),
            pvalue=float(ks_p),
            criterion=(
                f"Kolmogorov–Smirnov of z_t vs {fitted.dist} (estimated parameters); "
                f"p-value > {alpha:g}. QQ-plot is mandatory (tails show there first)."
            ),
            verdict="PASS" if ok else "WARN",
            message=(
                f"KS {'does not reject' if ok else 'rejects'} the assumed {fitted.dist} "
                f"distribution. QQ-plot written to {qq_path}."
            ),
            extras={"qq_path": str(qq_path), "dist": fitted.dist, "nu": fitted.nu},
        )

    def _b5_stationarity(self, fitted: FittedGarchSnapshot) -> DiagnosticResult:
        cap = 1.0
        igarch = self.config.igarch_persistence
        extras: dict[str, Any] = {
            "persistence": None,
            "family": fitted.family,
            "alpha": fitted.alpha,
            "beta": fitted.beta,
            "gamma": fitted.gamma,
        }
        formula = (
            "|beta| < 1"
            if fitted.family == "EGARCH"
            else (
                "alpha + gamma/2 + beta < 1"
                if fitted.family == "GJR"
                else "alpha + beta < 1"
            )
        )
        if fitted.family == "GJR":
            violations: list[str] = []
            if fitted.omega <= 0.0:
                violations.append(f"omega={fitted.omega} must be > 0")
            if fitted.alpha < 0.0:
                violations.append(f"alpha={fitted.alpha} must be >= 0")
            if fitted.alpha + fitted.gamma < 0.0:
                violations.append(f"alpha+gamma={fitted.alpha + fitted.gamma} must be >= 0")
            if fitted.beta < 0.0:
                violations.append(f"beta={fitted.beta} must be >= 0")
            if violations:
                extras["persistence"] = _persistence(fitted)
                return DiagnosticResult(
                    name="B5.stationarity",
                    statistic=_persistence(fitted),
                    pvalue=None,
                    criterion=f"{fitted.family}: positivity and {formula}.",
                    verdict="FAIL",
                    message="GJR constraints violated: " + "; ".join(violations) + ".",
                    extras=extras,
                )
        persistence = _persistence(fitted)
        extras["persistence"] = persistence
        if not math.isfinite(persistence) or persistence <= 0.0:
            extras["half_life"] = math.inf
            return DiagnosticResult(
                name="B5.stationarity",
                statistic=persistence,
                pvalue=None,
                criterion=f"{fitted.family}: {formula}.",
                verdict="FAIL",
                message=(
                    f"{fitted.family} persistence={persistence} is not in (0, 1); "
                    "half-life is undefined and the variance process is not stationary."
                ),
                extras=extras,
            )
        if persistence >= cap:
            extras["half_life"] = math.inf
            return DiagnosticResult(
                name="B5.stationarity",
                statistic=persistence,
                pvalue=None,
                criterion=f"{fitted.family}: {formula}.",
                verdict="FAIL",
                message=(
                    f"{fitted.family} persistence={persistence:.6f} is not strictly "
                    "less than 1; the variance process is not covariance-stationary."
                ),
                extras=extras,
            )
        half_life = float(math.log(self.config.half_life_mass) / math.log(persistence))
        extras["half_life"] = half_life
        if persistence > igarch:
            return DiagnosticResult(
                name="B5.stationarity",
                statistic=persistence,
                pvalue=None,
                criterion=f"{fitted.family}: {formula}; IGARCH warning if persistence > {igarch:g}.",
                verdict="WARN",
                message=(
                    f"IGARCH frontier: persistence={persistence:.6f} > {igarch:g}. "
                    f"Volatility shocks never decay; implied half-life is {half_life:.1f} days. "
                    "Long-horizon forecasts diverge."
                ),
                extras=extras,
            )
        return DiagnosticResult(
            name="B5.stationarity",
            statistic=persistence,
            pvalue=None,
            criterion=f"{fitted.family}: {formula}.",
            verdict="PASS",
            message=(
                f"{fitted.family} persistence={persistence:.4f} < 1; "
                f"shock half-life is {half_life:.2f} days."
            ),
            extras=extras,
        )

    def _b6_optimizer(self, fitted: FittedGarchSnapshot) -> list[DiagnosticResult]:
        conv = DiagnosticResult(
            name="B6.convergence",
            statistic=1.0 if fitted.converged else 0.0,
            pvalue=None,
            criterion="Optimizer reports convergence.",
            verdict="PASS" if fitted.converged else "FAIL",
            message=(
                "Optimizer reported convergence."
                if fitted.converged
                else "Optimizer did not converge; reported parameters are not a maximum."
            ),
            extras={"converged": fitted.converged, "loglikelihood": fitted.loglikelihood},
        )
        n_req = self.config.optimizer_restarts
        llfs = fitted.restart_loglikelihoods
        if llfs is None:
            restart = DiagnosticResult(
                name="B6.restarts",
                statistic=None,
                pvalue=None,
                criterion=f"Re-estimate from {n_req} distinct starting points; log-likelihoods within {self.config.llf_atol:g}.",
                verdict="FAIL",
                message=(
                    "Restart log-likelihoods were not supplied; optimizer stability "
                    "is unverified. Multiple local optima are a common EGARCH failure "
                    "mode on short samples. Do not use this model."
                ),
                extras={"n_restarts": 0},
            )
            return [conv, restart]
        arr = np.asarray(llfs, dtype=float)
        finite = np.isfinite(arr)
        if arr.size < n_req or int(finite.sum()) < n_req:
            restart = DiagnosticResult(
                name="B6.restarts",
                statistic=None,
                pvalue=None,
                criterion=f"Re-estimate from {n_req} distinct starting points; log-likelihoods within {self.config.llf_atol:g}.",
                verdict="FAIL",
                message=(
                    f"Need {n_req} finite restart log-likelihoods "
                    f"(got {int(finite.sum())} finite of {arr.size}). "
                    "Non-finite or missing restarts are treated as disagreement."
                ),
                extras={"n_restarts": int(arr.size), "loglikelihoods": [float(v) for v in arr]},
            )
            return [conv, restart]
        spread = float(np.max(arr[finite]) - np.min(arr[finite]))
        agree = spread <= self.config.llf_atol
        if agree:
            verdict: Verdict = "PASS"
            message = (
                f"{arr.size} restart log-likelihoods agree within {self.config.llf_atol:g} "
                f"(spread={spread:.4g})."
            )
        else:
            verdict = "FAIL"
            message = (
                f"Restart log-likelihoods disagree (spread={spread:.4g} over {arr.size} starts; "
                f"need {n_req} within {self.config.llf_atol:g}). The likelihood surface has "
                "multiple local optima; reported parameters are not reliable."
            )
        restart = DiagnosticResult(
            name="B6.restarts",
            statistic=spread,
            pvalue=None,
            criterion=(
                f"Re-estimate from {n_req} distinct starting points; "
                f"log-likelihoods within {self.config.llf_atol:g}."
            ),
            verdict=verdict,
            message=message,
            extras={"n_restarts": int(arr.size), "loglikelihoods": [float(v) for v in arr]},
        )
        return [conv, restart]


def _as_series(values: pd.Series | np.ndarray | Sequence[float], *, name: str | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        out = values.astype(float).dropna()
        if name is not None and out.name is None:
            out = out.rename(name)
        if out.empty:
            raise ValueError("diagnostic series is empty after dropping NA")
        return out
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("diagnostic series is empty")
    return pd.Series(arr, name=name or "r")


def _engle_arch_lm(x: pd.Series, lags: int) -> tuple[float, float]:
    """Engle (1982) ARCH LM: :math:`n R^2 \\sim \\chi^2(q)`."""
    if lags < 1:
        raise ValueError("ARCH LM lags must be >= 1")
    e2 = x.astype(float) ** 2
    lagged = {f"l{i}": e2.shift(i) for i in range(1, lags + 1)}
    frame = pd.concat({"y": e2, **lagged}, axis=1).dropna()
    if frame.shape[0] <= lags + 2:
        raise ValueError(f"ARCH LM(q={lags}) needs more observations than {frame.shape[0]}")
    y = frame["y"]
    design = sm.add_constant(frame.drop(columns=["y"]), has_constant="add")
    ols = sm.OLS(y, design).fit()
    stat = float(ols.nobs) * float(ols.rsquared)
    pval = float(scipy_stats.chi2.sf(stat, lags))
    return stat, pval


def _ljung_box(x: pd.Series, lags: int) -> tuple[float, float]:
    table = acorr_ljungbox(x.dropna(), lags=[lags], return_df=True)
    return float(table["lb_stat"].iloc[-1]), float(table["lb_pvalue"].iloc[-1])


def _engle_ng_joint(z: pd.Series, eps: pd.Series) -> tuple[float, float]:
    """Engle–Ng (1993) joint sign / size-bias test, :math:`n R^2 \\sim \\chi^2(3)`."""
    aligned = pd.concat({"z": z.astype(float), "eps": eps.astype(float)}, axis=1).dropna()
    z2 = aligned["z"] ** 2
    e_lag = aligned["eps"].shift(1)
    frame = pd.concat({"z2": z2, "e_lag": e_lag}, axis=1).dropna()
    s_minus = (frame["e_lag"] < 0).astype(float)
    design = sm.add_constant(
        pd.DataFrame(
            {
                "sign": s_minus.to_numpy(),
                "neg_size": (s_minus * frame["e_lag"]).to_numpy(),
                "pos_size": ((1.0 - s_minus) * frame["e_lag"]).to_numpy(),
            },
            index=frame.index,
        ),
        has_constant="add",
    )
    ols = sm.OLS(frame["z2"].to_numpy(), design).fit()
    stat = float(ols.nobs) * float(ols.rsquared)
    pval = float(scipy_stats.chi2.sf(stat, 3))
    return stat, pval


def _icss_break_indices(squared: np.ndarray, crit: float, *, min_len: int = 30) -> list[int]:
    """Inclán–Tiao (1994) Iterative Cumulative Sum of Squares break dates."""
    squared = np.asarray(squared, dtype=float)
    if squared.size < min_len:
        return []
    found: list[int] = []
    stack: list[tuple[int, int]] = [(0, int(squared.size))]
    while stack:
        lo, hi = stack.pop()
        loc = _icss_scan(squared, lo, hi, crit, min_len)
        if loc is None:
            continue
        found.append(loc)
        stack.append((lo, loc))
        stack.append((loc, hi))
    return sorted(set(found))


def _icss_scan(
    squared: np.ndarray,
    lo: int,
    hi: int,
    crit: float,
    min_len: int,
) -> int | None:
    n = hi - lo
    if n < min_len:
        return None
    c = np.cumsum(squared[lo:hi])
    total = float(c[-1])
    if total <= 0.0:
        return None
    k = np.arange(1, n + 1, dtype=float)
    d = c / total - k / n
    it = np.sqrt(n / 2.0) * np.abs(d[:-1])
    j = int(np.argmax(it))
    if float(it[j]) <= crit:
        return None
    loc = lo + j + 1
    if loc - lo < min_len // 2 or hi - loc < min_len // 2:
        return None
    return loc


def _persistence(fitted: FittedGarchSnapshot) -> float:
    if fitted.family == "EGARCH":
        return float(abs(fitted.beta))
    if fitted.family == "GJR":
        return float(fitted.alpha + 0.5 * fitted.gamma + fitted.beta)
    return float(fitted.alpha + fitted.beta)


def _assumed_cdf(fitted: FittedGarchSnapshot) -> Callable[[np.ndarray], np.ndarray]:
    dist = fitted.dist
    if dist == "normal":
        return lambda s: scipy_stats.norm.cdf(np.asarray(s, dtype=float))
    nu = _require_nu(fitted)
    if dist == "t":
        scale = _standardized_t_scale(nu)
        return lambda s, _nu=nu, _scale=scale: scipy_stats.t.cdf(
            np.asarray(s, dtype=float), df=_nu, loc=0.0, scale=_scale
        )
    lam = fitted.lambda_skew if fitted.lambda_skew is not None else 0.0
    from arch.univariate.distribution import SkewStudent

    skewt = SkewStudent()
    params = np.array([nu, lam], dtype=float)
    return lambda s, _p=params, _d=skewt: np.asarray(_d.cdf(np.asarray(s, dtype=float), _p))


def _assumed_ppf(fitted: FittedGarchSnapshot, p: np.ndarray) -> np.ndarray:
    dist = fitted.dist
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    if dist == "normal":
        return scipy_stats.norm.ppf(p)
    nu = _require_nu(fitted)
    if dist == "t":
        scale = _standardized_t_scale(nu)
        return scipy_stats.t.ppf(p, df=nu, loc=0.0, scale=scale)
    lam = fitted.lambda_skew if fitted.lambda_skew is not None else 0.0
    from arch.univariate.distribution import SkewStudent

    return np.asarray(SkewStudent().ppf(p, np.array([nu, lam], dtype=float)), dtype=float)


def _require_nu(fitted: FittedGarchSnapshot) -> float:
    if fitted.nu is None:
        raise ValueError(f"{fitted.dist} distribution requires estimated nu")
    return float(fitted.nu)


def _standardized_t_scale(nu: float) -> float:
    """Scale so scipy.stats.t has unit variance: :math:`\\sqrt{(\\nu-2)/\\nu}`."""
    if nu <= 2.0:
        raise ValueError(f"Student-t nu={nu} has no variance; cannot standardize")
    return math.sqrt((nu - 2.0) / nu)


def _cusum_of_squares(squared: np.ndarray) -> tuple[float, float]:
    """Inclán–Tiao CUSUM-of-squares statistic and Brownian-bridge p-value."""
    squared = np.asarray(squared, dtype=float)
    n = int(squared.size)
    if n < 10:
        raise ValueError("CUSUM of squares needs more observations")
    c = np.cumsum(squared)
    total = float(c[-1])
    if total <= 0.0:
        raise ValueError("CUSUM of squares is undefined when sum of squares is 0")
    k = np.arange(1, n + 1, dtype=float)
    d = c / total - k / n
    stat = float(np.sqrt(n / 2.0) * np.max(np.abs(d[:-1])))
    pval = float(scipy_stats.kstwobign.sf(stat))
    return stat, pval


def _write_qq_plot(
    z: np.ndarray,
    fitted: FittedGarchSnapshot,
    path: Path,
    config: DiagnosticConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = np.sort(np.asarray(z, dtype=float))
    probs = (np.arange(1, ordered.size + 1, dtype=float) - 0.5) / ordered.size
    theoretical = _assumed_ppf(fitted, probs)
    fig, ax = plt.subplots(
        figsize=(config.plot.figsize_width, config.plot.figsize_height)
    )
    ax.scatter(theoretical, ordered, s=10, alpha=0.65, color="#1f4e79")
    lo = float(min(np.nanmin(theoretical), np.nanmin(ordered)))
    hi = float(max(np.nanmax(theoretical), np.nanmax(ordered)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlabel(f"Theoretical quantiles ({fitted.dist})")
    ax.set_ylabel(r"Sample quantiles of $z_t$")
    ax.set_title("QQ plot of standardized residuals")
    fig.tight_layout()
    fig.savefig(path, dpi=config.plot.dpi)
    plt.close(fig)
    logger.info("QQ-plot written to %s", path)


def _index_stamp(index: pd.Index, loc: int) -> str:
    loc = min(max(loc, 0), len(index) - 1)
    value = index[loc]
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%Y-%m-%d")


def _fmt_num(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.4g}"


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _log_report(report: DiagnosticReport) -> None:
    for row in report.results:
        logger.info(
            "%s %s stat=%s p=%s — %s",
            row.verdict,
            row.name,
            _fmt_num(row.statistic),
            _fmt_num(row.pvalue),
            row.message,
        )
    if report.has_blocking_fail:
        logger.error(
            "blocking diagnostic FAIL: %s",
            ", ".join(row.name for row in report.blocking_failures),
        )
