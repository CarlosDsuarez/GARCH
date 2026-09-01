"""Statistical backtests for tail-risk models (VaR and Expected Shortfall).

A VaR number without a formal backtest is an opinion with decimals. Each
test below catches a *different* failure mode. Skipping any one of them
leaves a hole that a decorative model can walk through.

Hit sequence
------------
:math:`I_t=1` if :math:`r_t < -\\mathrm{VaR}_t` (VaR is a **positive** loss
magnitude). :math:`p=1-\\alpha` is the expected hit rate.

Kupiec POF (unconditional coverage)
-----------------------------------
Counts exceptions. A model that prints the right *number* of hits, all
packed into a single crisis week, **passes** Kupiec and is still useless.
That is why Christoffersen is mandatory.

Christoffersen independence / conditional coverage
--------------------------------------------------
Clustering of hits is the fingerprint of a filter that missed the vol
regime: the model fails several days in a row when the crisis arrives.
:math:`LR_{cc}=LR_{uc}+LR_{ind}\\sim\\chi^2(2)`.

Engle–Manganelli DQ
-------------------
:math:`Hit_t=I_t-p` must be a martingale difference. The Wald test includes
:math:`d\\cdot\\mathrm{VaR}_t`, which Kupiec and Christoffersen never see:
systematic failure when the VaR level itself is high or low.

Acerbi–Székely (2014) ES backtest
---------------------------------
ES is not elicitable; it is still backtestable via statistics *conditional
on exceptions*. :math:`E[Z_1]=E[Z_2]=0` under a correct model. A
significantly **negative** Z means tail losses exceed the predicted ES.
p-values come from M Monte Carlo draws under the predictive null
(exponential exceedances with mean ES), never from a χ² table. Default
:math:`M=10\\,000`, seed from config.

Basel traffic light (250 days, 99%)
-----------------------------------
Green 0–4, yellow 5–9, red ≥10. Included so a risk committee can read the
page in the language it already knows.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

__all__ = [
    "AcerbiResult",
    "BacktestSuiteResult",
    "BaselTrafficLight",
    "ChristoffersenResult",
    "ConditionalCoverageResult",
    "DynamicQuantileResult",
    "LikelihoodRatioResult",
    "ModelWindowReport",
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "basel_traffic_light",
    "christoffersen_cc",
    "christoffersen_independence",
    "dynamic_quantile_test",
    "hit_series",
    "kupiec_pof",
    "run_full_backtest_suite",
]

CHI2_1_5PCT = 3.841458820694124
CHI2_2_5PCT = 5.991464547107979


class BacktestError(ValueError):
    """Hit series or likelihood inputs are not usable ([C7])."""


@dataclass(frozen=True)
class LikelihoodRatioResult:
    statistic: float
    pvalue: float
    df: int
    reject: bool
    n: int
    x: int
    p: float
    name: str = "kupiec_pof"


@dataclass(frozen=True)
class ChristoffersenResult:
    statistic: float
    pvalue: float
    df: int
    reject: bool
    n_00: int
    n_01: int
    n_10: int
    n_11: int
    pi_01: float
    pi_11: float
    pi: float
    name: str = "christoffersen_ind"


@dataclass(frozen=True)
class ConditionalCoverageResult:
    statistic: float
    pvalue: float
    df: int
    reject: bool
    lr_uc: float
    lr_ind: float
    name: str = "christoffersen_cc"


@dataclass(frozen=True)
class DynamicQuantileResult:
    statistic: float
    pvalue: float
    df: int
    reject: bool
    lags: int
    coefficients: pd.Series
    name: str = "engle_manganelli_dq"


@dataclass(frozen=True)
class AcerbiResult:
    statistic: float
    pvalue: float
    reject: bool
    n_simulations: int
    seed: int
    n_hits: int
    name: str


@dataclass(frozen=True)
class BaselTrafficLight:
    zone: Literal["green", "yellow", "red"]
    exceptions: int
    window: int
    alpha: float = 0.99


@dataclass
class ModelWindowReport:
    model: str
    window: str
    n: int
    x: int
    p: float
    kupiec: LikelihoodRatioResult
    independence: ChristoffersenResult
    conditional_coverage: ConditionalCoverageResult
    dq: DynamicQuantileResult
    z1: AcerbiResult
    z2: AcerbiResult
    basel: BaselTrafficLight
    verdict: str


@dataclass
class BacktestSuiteResult:
    alpha: float
    models: list[str]
    windows: list[str]
    reports: dict[tuple[str, str], ModelWindowReport]
    comparison: pd.DataFrame
    plot_path: Path | None
    verdict: str
    rolling_hits: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def report(self, model: str, window: str) -> ModelWindowReport:
        try:
            return self.reports[(model, window)]
        except KeyError as exc:
            raise BacktestError(f"no report for model={model!r} window={window!r}") from exc


def hit_series(returns: pd.Series, var: pd.Series) -> pd.Series:
    """:math:`I_t=1` iff the realized return is worse than ``-VaR_t``."""
    aligned = pd.concat({"r": returns, "var": var}, axis=1).dropna()
    if aligned.empty:
        raise BacktestError("returns and VaR have no overlapping dates")
    if (aligned["var"] < 0).any():
        raise BacktestError("VaR must be a positive loss magnitude")
    hits = (aligned["r"] < -aligned["var"]).astype(int)
    hits.name = "hit"
    return hits


def kupiec_pof(
    n: int,
    x: int,
    p: float,
    *,
    significance: float = 0.05,
    critical: float = CHI2_1_5PCT,
) -> LikelihoodRatioResult:
    """Unconditional coverage (proportion of failures). χ²(1)."""
    if n <= 0 or x < 0 or x > n:
        raise BacktestError(f"invalid Kupiec counts n={n} x={x}")
    if not (0.0 < p < 1.0):
        raise BacktestError("p must lie in (0, 1)")
    if x == 0:
        lr = -2.0 * n * math.log(1.0 - p)
    elif x == n:
        lr = -2.0 * n * math.log(p)
    else:
        phat = x / n
        lr = -2.0 * (
            (n - x) * math.log(1.0 - p)
            + x * math.log(p)
            - (n - x) * math.log(1.0 - phat)
            - x * math.log(phat)
        )
        if abs(lr) < 1e-12:
            lr = 0.0
    pvalue = float(scipy_stats.chi2.sf(lr, 1))
    return LikelihoodRatioResult(
        statistic=float(lr),
        pvalue=pvalue,
        df=1,
        reject=bool(lr > critical or pvalue < significance),
        n=int(n),
        x=int(x),
        p=float(p),
    )


def _transition_counts(hits: pd.Series) -> tuple[int, int, int, int]:
    flag = hits.astype(int).to_numpy()
    if flag.size < 2:
        raise BacktestError("Christoffersen needs at least two observations")
    lagged = flag[:-1]
    current = flag[1:]
    n_00 = int(((lagged == 0) & (current == 0)).sum())
    n_01 = int(((lagged == 0) & (current == 1)).sum())
    n_10 = int(((lagged == 1) & (current == 0)).sum())
    n_11 = int(((lagged == 1) & (current == 1)).sum())
    return n_00, n_01, n_10, n_11


def _count_log(count: int, probability: float) -> float:
    if count == 0:
        return 0.0
    if probability <= 0.0:
        return float("-inf")
    return count * math.log(probability)


def christoffersen_independence(
    hits: pd.Series,
    *,
    significance: float = 0.05,
    critical: float = CHI2_1_5PCT,
) -> ChristoffersenResult:
    """Independence of hits. χ²(1). Empty transition rows ⇒ LR_ind = 0."""
    n_00, n_01, n_10, n_11 = _transition_counts(hits)
    n_0 = n_00 + n_01
    n_1 = n_10 + n_11
    n_trans = n_0 + n_1
    pi = (n_01 + n_11) / n_trans if n_trans else float("nan")
    pi_01 = n_01 / n_0 if n_0 else float("nan")
    pi_11 = n_11 / n_1 if n_1 else float("nan")
    if n_0 == 0 or n_1 == 0:
        lr = 0.0
    else:
        num = _count_log(n_00 + n_10, 1.0 - pi) + _count_log(n_01 + n_11, pi)
        den = (
            _count_log(n_00, 1.0 - pi_01)
            + _count_log(n_01, pi_01)
            + _count_log(n_10, 1.0 - pi_11)
            + _count_log(n_11, pi_11)
        )
        lr = -2.0 * (num - den)
        if not math.isfinite(lr):
            lr = float("inf")
    pvalue = float(scipy_stats.chi2.sf(lr, 1)) if math.isfinite(lr) else 0.0
    return ChristoffersenResult(
        statistic=float(lr),
        pvalue=pvalue,
        df=1,
        reject=bool(lr > critical or pvalue < significance),
        n_00=n_00,
        n_01=n_01,
        n_10=n_10,
        n_11=n_11,
        pi_01=float(pi_01) if pi_01 == pi_01 else float("nan"),
        pi_11=float(pi_11) if pi_11 == pi_11 else float("nan"),
        pi=float(pi) if pi == pi else float("nan"),
    )


def christoffersen_cc(
    hits: pd.Series,
    p: float,
    *,
    significance: float = 0.05,
    critical: float = CHI2_2_5PCT,
) -> ConditionalCoverageResult:
    """Joint conditional coverage :math:`LR_{cc}=LR_{uc}+LR_{ind}\\sim\\chi^2(2)`."""
    flag = hits.astype(int)
    uc = kupiec_pof(n=int(flag.shape[0]), x=int(flag.sum()), p=p, significance=significance)
    ind = christoffersen_independence(flag, significance=significance)
    lr = uc.statistic + ind.statistic
    pvalue = float(scipy_stats.chi2.sf(lr, 2)) if math.isfinite(lr) else 0.0
    return ConditionalCoverageResult(
        statistic=float(lr),
        pvalue=pvalue,
        df=2,
        reject=bool(lr > critical or pvalue < significance),
        lr_uc=uc.statistic,
        lr_ind=ind.statistic,
    )


def dynamic_quantile_test(
    hits: pd.Series,
    var: pd.Series,
    p: float,
    *,
    lags: int = 4,
    significance: float = 0.05,
) -> DynamicQuantileResult:
    """Engle–Manganelli DQ: Wald test that Hit is unpredictable. χ²(K+2)."""
    if lags < 1:
        raise BacktestError("DQ lags must be >= 1")
    frame = pd.concat({"hit": hits.astype(float), "var": var.astype(float)}, axis=1).dropna()
    hit = frame["hit"] - float(p)
    data = pd.DataFrame({"y": hit, "var": frame["var"]}, index=frame.index)
    for lag in range(1, lags + 1):
        data[f"hit_l{lag}"] = hit.shift(lag)
    data = data.dropna()
    y = data["y"].to_numpy(dtype=float)
    x_cols = ["var", *[f"hit_l{k}" for k in range(1, lags + 1)]]
    x = sm.add_constant(data[x_cols].to_numpy(dtype=float), has_constant="add")
    if x.shape[0] <= x.shape[1]:
        raise BacktestError("insufficient rows for the DQ regression")
    ols = sm.OLS(y, x).fit()
    cov = np.asarray(ols.cov_params(), dtype=float)
    beta = np.asarray(ols.params, dtype=float)
    try:
        wald = float(beta.T @ np.linalg.inv(cov) @ beta)
    except np.linalg.LinAlgError:
        wald = float(beta.T @ np.linalg.pinv(cov) @ beta)
    df = int(beta.size)
    pvalue = float(scipy_stats.chi2.sf(wald, df))
    coef = pd.Series(beta, index=["c", "d_var", *[f"b{k}" for k in range(1, lags + 1)]])
    return DynamicQuantileResult(
        statistic=wald,
        pvalue=pvalue,
        df=df,
        reject=bool(pvalue < significance),
        lags=int(lags),
        coefficients=coef,
    )


def _align_tail(returns: pd.Series, es: pd.Series, hits: pd.Series) -> pd.DataFrame:
    frame = pd.concat(
        {"r": returns, "es": es, "hit": hits.astype(int)},
        axis=1,
    ).dropna()
    if frame.empty:
        raise BacktestError("returns, ES and hits have no overlap")
    if (frame["es"] <= 0).any():
        raise BacktestError("ES must be a positive loss magnitude")
    return frame


def _z1_stat(frame: pd.DataFrame) -> float:
    tail = frame.loc[frame["hit"] == 1]
    if tail.empty:
        return float("nan")
    return float((tail["r"] / tail["es"]).mean() + 1.0)


def _z2_stat(frame: pd.DataFrame, p: float) -> float:
    n = int(frame.shape[0])
    if n == 0 or p <= 0.0:
        raise BacktestError("Z2 requires n>0 and p>0")
    return float((frame["r"] * frame["hit"] / frame["es"]).sum() / (n * p) + 1.0)


def _simulate_acerbi(
    frame: pd.DataFrame,
    p: float,
    *,
    n_simulations: int,
    seed: int,
    which: Literal["z1", "z2"],
) -> np.ndarray:
    """Predictive null: I~Bern(p), exceedance magnitude ~ Exp(1)×ES.

    Then :math:`E[r/\\mathrm{ES}\\mid I=1]=-1`, so both Z statistics have
    mean zero. M and ``seed`` are part of the audit trail.
    """
    rng = np.random.default_rng(seed)
    n = int(frame.shape[0])
    es = frame["es"].to_numpy(dtype=float)
    indicator = rng.random((n_simulations, n)) < p
    expo = rng.exponential(1.0, size=(n_simulations, n))
    r_over_es = np.where(indicator, -expo, 0.0)
    if which == "z1":
        hit_count = indicator.sum(axis=1)
        totals = r_over_es.sum(axis=1)
        mean_rel = np.divide(totals, hit_count, out=np.zeros_like(totals), where=hit_count > 0)
        return (mean_rel + 1.0).astype(float)
    return (r_over_es.sum(axis=1) / (n * p) + 1.0).astype(float)


def _acerbi_pvalue(observed: float, simulated: np.ndarray) -> float:
    if not math.isfinite(observed):
        return 1.0
    return float(np.mean(simulated <= observed))


def acerbi_szekely_z1(
    returns: pd.Series,
    es: pd.Series,
    hits: pd.Series,
    *,
    p: float = 0.01,
    n_simulations: int = 10_000,
    seed: int = 7,
    significance: float = 0.05,
) -> AcerbiResult:
    """Mean relative exceedance. Negative Z1 ⇒ tails worse than ES."""
    frame = _align_tail(returns, es, hits)
    observed = _z1_stat(frame)
    simulated = _simulate_acerbi(
        frame, p, n_simulations=n_simulations, seed=seed, which="z1"
    )
    pvalue = _acerbi_pvalue(observed, simulated)
    return AcerbiResult(
        statistic=float(observed) if observed == observed else float("nan"),
        pvalue=pvalue,
        reject=bool(math.isfinite(observed) and observed < 0.0 and pvalue < significance),
        n_simulations=int(n_simulations),
        seed=int(seed),
        n_hits=int(frame["hit"].sum()),
        name="acerbi_z1",
    )


def acerbi_szekely_z2(
    returns: pd.Series,
    es: pd.Series,
    hits: pd.Series,
    *,
    p: float,
    n_simulations: int = 10_000,
    seed: int = 7,
    significance: float = 0.05,
) -> AcerbiResult:
    """Joint frequency-and-magnitude ES test."""
    frame = _align_tail(returns, es, hits)
    observed = _z2_stat(frame, p)
    simulated = _simulate_acerbi(
        frame, p, n_simulations=n_simulations, seed=seed, which="z2"
    )
    pvalue = _acerbi_pvalue(observed, simulated)
    return AcerbiResult(
        statistic=float(observed),
        pvalue=pvalue,
        reject=bool(observed < 0.0 and pvalue < significance),
        n_simulations=int(n_simulations),
        seed=int(seed),
        n_hits=int(frame["hit"].sum()),
        name="acerbi_z2",
    )


def basel_traffic_light(*, exceptions: int, window: int = 250, alpha: float = 0.99) -> BaselTrafficLight:
    """Official 250-day / 99% zones: green 0–4, yellow 5–9, red ≥10."""
    if exceptions < 0:
        raise BacktestError("exception count cannot be negative")
    if exceptions <= 4:
        zone: Literal["green", "yellow", "red"] = "green"
    elif exceptions <= 9:
        zone = "yellow"
    else:
        zone = "red"
    return BaselTrafficLight(zone=zone, exceptions=int(exceptions), window=int(window), alpha=float(alpha))


def _verdict(
    kupiec: LikelihoodRatioResult,
    cc: ConditionalCoverageResult,
    dq: DynamicQuantileResult,
    z1: AcerbiResult,
    basel: BaselTrafficLight,
) -> str:
    if kupiec.reject or cc.reject or dq.reject or z1.reject or basel.zone == "red":
        return "reject"
    if basel.zone == "yellow":
        return "scrutiny"
    return "pass"


def _slice_window(
    returns: pd.Series,
    var: pd.Series,
    es: pd.Series,
    window: str,
    rolling: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    aligned = pd.concat({"r": returns, "var": var, "es": es}, axis=1).dropna()
    if window == "rolling_250":
        if aligned.shape[0] < 2:
            raise BacktestError("rolling window is empty")
        aligned = aligned.iloc[-min(rolling, aligned.shape[0]) :]
    return aligned["r"], aligned["var"], aligned["es"]


def _evaluate_model(
    *,
    model: str,
    window: str,
    returns: pd.Series,
    var: pd.Series,
    es: pd.Series,
    p: float,
    alpha: float,
    lags: int,
    rolling: int,
    n_sim: int,
    seed: int,
    significance: float,
) -> ModelWindowReport:
    r, v, e = _slice_window(returns, var, es, window, rolling)
    hits = hit_series(r, v)
    n = int(hits.shape[0])
    x = int(hits.sum())
    uc = kupiec_pof(n=n, x=x, p=p, significance=significance)
    ind = christoffersen_independence(hits, significance=significance)
    cc = christoffersen_cc(hits, p=p, significance=significance)
    dq = dynamic_quantile_test(hits, v.reindex(hits.index), p=p, lags=lags, significance=significance)
    z1 = acerbi_szekely_z1(
        r, e.reindex(hits.index), hits, p=p, n_simulations=n_sim, seed=seed, significance=significance
    )
    z2 = acerbi_szekely_z2(
        r, e.reindex(hits.index), hits, p=p, n_simulations=n_sim, seed=seed, significance=significance
    )
    basel_n = min(n, rolling)
    basel_x = int(hits.iloc[-basel_n:].sum())
    basel = basel_traffic_light(
        exceptions=basel_x,
        window=250 if window == "rolling_250" else basel_n,
        alpha=alpha,
    )
    verdict = _verdict(uc, cc, dq, z1, basel)
    logger.info(
        "VaR backtest model=%s window=%s n=%s x=%s Kupiec_p=%.3f CC_p=%.3f DQ_p=%.3f "
        "Z1=%.3f Basel=%s verdict=%s",
        model,
        window,
        n,
        x,
        uc.pvalue,
        cc.pvalue,
        dq.pvalue,
        z1.statistic,
        basel.zone,
        verdict,
    )
    return ModelWindowReport(
        model=model,
        window=window,
        n=n,
        x=x,
        p=p,
        kupiec=uc,
        independence=ind,
        conditional_coverage=cc,
        dq=dq,
        z1=z1,
        z2=z2,
        basel=basel,
        verdict=verdict,
    )


def _plot_violations(returns: pd.Series, var: pd.Series, dest: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(dest.parent))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hits = hit_series(returns, var)
    aligned = pd.concat({"r": returns, "var": var, "hit": hits}, axis=1).dropna()
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.plot(aligned.index, aligned["r"].to_numpy(), color="0.4", lw=0.7, label="retorno")
    ax.plot(aligned.index, -aligned["var"].to_numpy(), color="tab:blue", lw=1.0, label="-VaR")
    viol = aligned["hit"].astype(bool)
    if viol.any():
        ax.scatter(
            aligned.index[viol],
            aligned.loc[viol, "r"],
            color="tab:red",
            s=18,
            zorder=5,
            label="violación",
        )
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.set_title("Violaciones de VaR (clustering visible a ojo)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)


def run_full_backtest_suite(
    returns: pd.Series,
    var_series: pd.Series,
    es_series: pd.Series,
    alpha: float,
    *,
    models: Mapping[str, tuple[pd.Series, pd.Series]] | None = None,
    plot_directory: str | Path | None = None,
    seed: int = 7,
    acerbi_simulations: int = 10_000,
    dq_lags: int = 4,
    rolling_window: int = 250,
    significance: float = 0.05,
    primary_name: str = "FHS-GJR-GARCH",
) -> BacktestSuiteResult:
    """Run the full battery on the full sample and the last ``rolling_window`` days."""
    if not (0.0 < alpha < 1.0):
        raise BacktestError("alpha must lie in (0, 1)")
    p = 1.0 - float(alpha)
    book: dict[str, tuple[pd.Series, pd.Series]] = dict(models or {})
    if primary_name not in book:
        book = {primary_name: (var_series, es_series), **book}
    windows = ["full", "rolling_250"]
    reports: dict[tuple[str, str], ModelWindowReport] = {}
    for name, (var, es) in book.items():
        for window in windows:
            reports[(name, window)] = _evaluate_model(
                model=name,
                window=window,
                returns=returns,
                var=var,
                es=es,
                p=p,
                alpha=alpha,
                lags=dq_lags,
                rolling=rolling_window,
                n_sim=acerbi_simulations,
                seed=seed,
                significance=significance,
            )
    rows = []
    for (name, window), item in reports.items():
        rows.append(
            {
                "model": name,
                "window": window,
                "n": item.n,
                "hits": item.x,
                "hit_rate": item.x / item.n if item.n else float("nan"),
                "kupiec_p": item.kupiec.pvalue,
                "christoffersen_p": item.independence.pvalue,
                "cc_p": item.conditional_coverage.pvalue,
                "dq_p": item.dq.pvalue,
                "z1": item.z1.statistic,
                "z1_p": item.z1.pvalue,
                "z2": item.z2.statistic,
                "z2_p": item.z2.pvalue,
                "basel": item.basel.zone,
                "verdict": item.verdict,
            }
        )
    comparison = pd.DataFrame(rows)
    hits_primary = hit_series(returns, var_series)
    rolling_hits = hits_primary.rolling(
        rolling_window, min_periods=max(20, rolling_window // 5)
    ).sum()
    plot_path = None
    if plot_directory is not None:
        dest = Path(plot_directory) / "var_violations.png"
        _plot_violations(returns, var_series, dest)
        plot_path = dest
    primary = reports[(primary_name, "full")]
    worst = "pass"
    if any(item.verdict == "scrutiny" for item in reports.values()):
        worst = "scrutiny"
    if any(item.verdict == "reject" for item in reports.values()):
        worst = "reject"
    return BacktestSuiteResult(
        alpha=float(alpha),
        models=list(book),
        windows=windows,
        reports=reports,
        comparison=comparison,
        plot_path=plot_path,
        verdict=primary.verdict if worst == "pass" else worst,
        rolling_hits=rolling_hits,
    )
