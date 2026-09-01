"""Walk-forward statistical and economic validation of the dislocation score.

GARCH parameters are re-estimated on the first business day of each month
using data *strictly before* that date, then frozen. Daily :math:`\\sigma_t`
updates by running the recursion on newly observed returns only ([C1]).
A single full-sample GARCH fit used for the whole backtest is forbidden.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from backtest.schema import BacktestConfig
from signals.dislocation import (
    DislocationSignalEngine,
    SignalInputs,
    rolling_percentile_rank,
)
from signals.schema import DislocationConfig

logger = logging.getLogger(__name__)

__all__ = [
    "BacktestResult",
    "FrozenParams",
    "WalkForwardBacktester",
    "block_bootstrap_ci",
    "count_independent_episodes",
    "economic_metrics",
    "forward_returns",
    "predictive_regression",
    "quintile_table",
    "state_dependent_cost_bps",
]


class BacktestError(ValueError):
    """Walk-forward protocol or sample is not usable ([C7])."""


@dataclass(frozen=True)
class FrozenParams:
    fitted_through: pd.Timestamp
    values: dict[str, float]


class VolatilityEstimator(Protocol):
    def fit(self, series: pd.Series) -> FrozenParams: ...

    def filter(self, params: FrozenParams, series: pd.Series) -> pd.Series: ...


@dataclass(frozen=True)
class PredictiveReport:
    horizon: int
    n_obs: int
    intercept: float
    beta: float
    t_stat: float
    se_newey_west: float
    se_ols: float
    r_squared: float
    nw_lags: int
    pvalue: float


@dataclass(frozen=True)
class BootstrapCI:
    lower: float
    upper: float
    block_length: int
    n_bootstrap: int
    point: float


@dataclass(frozen=True)
class StrategyMetrics:
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    hit_rate: float
    payoff_ratio: float
    ann_turnover: float


@dataclass(frozen=True)
class ReportPaths:
    markdown: Path
    html: Path


@dataclass
class BacktestResult:
    panel: pd.DataFrame
    reestimation_dates: list[pd.Timestamp]
    parameter_trajectory: pd.DataFrame
    predictive: list[PredictiveReport]
    quintiles: pd.DataFrame
    n_independent_episodes: int
    bootstrap: BootstrapCI
    strategy: StrategyMetrics
    cost_sensitivity: dict[float, StrategyMetrics]
    benchmarks: dict[str, StrategyMetrics]
    failure_conditions: list[str]
    implementable: bool
    n_daily: int


def forward_returns(returns: pd.Series, horizon: int) -> pd.Series:
    """Compounded return from :math:`t+1` through :math:`t+h` (excludes day t)."""
    if horizon < 1:
        raise BacktestError("horizon must be >= 1")
    growth = (1.0 + returns.astype("float64")).to_numpy()
    fwd = pd.Series(index=returns.index, dtype="float64")
    for i in range(len(growth) - horizon):
        fwd.iloc[i] = float(np.prod(growth[i + 1 : i + 1 + horizon]) - 1.0)
    return fwd


def predictive_regression(y: pd.Series, x: pd.Series, *, horizon: int) -> PredictiveReport:
    """OLS of forward returns on the score with Newey-West lag = h ([V1])."""
    frame = pd.concat({"y": y, "x": x}, axis=1).dropna()
    if frame.shape[0] < 8:
        raise BacktestError("not enough overlapping observations for the predictive regression")
    Y = frame["y"].to_numpy(dtype=float)
    X = sm.add_constant(frame["x"].to_numpy(dtype=float), has_constant="add")
    ols = sm.OLS(Y, X).fit()
    hac = sm.OLS(Y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(horizon), "use_correction": True},
    )
    se_ols = float(ols.bse[1])
    se_hac = float(hac.bse[1])
    # Overlapping forwards are MA(h-1). HAC can still *contract* the slope SE
    # when the predictor is weakly dependent, even with residual ACF near 0.9.
    # Overlap must not be allowed to claim more precision than i.i.d. OLS.
    se_nw = max(se_hac, se_ols)
    t_stat = float(hac.params[1] / se_nw) if se_nw else float("nan")
    pvalue = float(2.0 * scipy_stats.t.sf(abs(t_stat), df=int(frame.shape[0] - 2)))
    return PredictiveReport(
        horizon=int(horizon),
        n_obs=int(frame.shape[0]),
        intercept=float(hac.params[0]),
        beta=float(hac.params[1]),
        t_stat=t_stat,
        se_newey_west=se_nw,
        se_ols=se_ols,
        r_squared=float(hac.rsquared),
        nw_lags=int(horizon),
        pvalue=pvalue,
    )


def quintile_table(forward: pd.Series, score: pd.Series) -> pd.DataFrame:
    """Mean forward return by score quintile ([V2])."""
    frame = pd.concat({"fwd": forward, "score": score}, axis=1).dropna()
    frame["quintile"] = pd.qcut(frame["score"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    grouped = frame.groupby("quintile", observed=False)["fwd"].agg(["mean", "count"])
    grouped = grouped.reindex([1, 2, 3, 4, 5])
    grouped.columns = ["mean_forward", "n"]
    grouped.index.name = None
    grouped.index = grouped.index.astype(int)
    return grouped


def count_independent_episodes(active: pd.Series, *, min_gap_calendar_days: int) -> int:
    """Count active blocks separated by at least ``min_gap_calendar_days`` ([V3])."""
    flag = active.fillna(False).astype(bool)
    if not flag.any():
        return 0
    runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: pd.Timestamp | None = None
    prev: pd.Timestamp | None = None
    for ts, on in flag.items():
        stamp = pd.Timestamp(ts)
        if on and start is None:
            start = stamp
        if on:
            prev = stamp
        if (not on) and start is not None and prev is not None:
            runs.append((start, prev))
            start = None
            prev = None
    if start is not None and prev is not None:
        runs.append((start, prev))
    if not runs:
        return 0
    merged = 1
    last_end = runs[0][1]
    for begin, end in runs[1:]:
        gap = int((begin - last_end).days)
        if gap >= min_gap_calendar_days:
            merged += 1
        last_end = end
    return merged


def block_bootstrap_ci(
    returns: pd.Series,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> BootstrapCI:
    """Moving-block bootstrap CI for the mean. i.i.d. resampling is invalid ([V3])."""
    values = returns.dropna().to_numpy(dtype=float)
    n = int(values.shape[0])
    point = float(values.mean()) if n else float("nan")
    if n < 2:
        return BootstrapCI(point, point, block_length, n_bootstrap, point)
    length = min(int(block_length), n)
    n_blocks = int(math.ceil(n / length))
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    max_start = n - length + 1
    for i in range(n_bootstrap):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([values[s : s + length] for s in starts])[:n]
        means[i] = float(sample.mean())
    lower, upper = np.quantile(means, [0.025, 0.975])
    return BootstrapCI(float(lower), float(upper), int(block_length), int(n_bootstrap), point)


def state_dependent_cost_bps(
    *,
    sigma: float,
    sigma_median: float,
    base_bps: float,
    k: float,
) -> float:
    """``base + k * (σ / σ_median)``. Spreads widen exactly when the strategy trades."""
    if sigma_median <= 0:
        raise BacktestError("sigma median must be positive")
    return float(base_bps + k * (sigma / sigma_median))


def economic_metrics(
    returns: pd.Series,
    *,
    weights: pd.Series,
    periods_per_year: int,
    risk_free: float = 0.0,
) -> dict[str, float]:
    r = returns.astype("float64").dropna()
    w = weights.reindex(r.index).astype("float64")
    n = int(r.shape[0])
    if n == 0:
        raise BacktestError("no strategy returns to score")
    equity = (1.0 + r).cumprod()
    years = n / float(periods_per_year)
    terminal = float(equity.iloc[-1])
    ann_return = float(terminal ** (1.0 / years) - 1.0) if years > 0 and terminal > 0 else float("nan")
    ann_vol = float(r.std(ddof=1) * math.sqrt(periods_per_year)) if n > 1 else float("nan")
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol and ann_vol > 0 else float("nan")
    downside = r[r < 0]
    down_vol = (
        float(downside.std(ddof=1) * math.sqrt(periods_per_year)) if downside.shape[0] > 1 else float("nan")
    )
    sortino = ann_return / down_vol if down_vol and down_vol > 0 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    calmar = ann_return / abs(max_dd) if max_dd < 0 else float("nan")
    underwater = drawdown < 0
    duration = 0
    current = 0
    for flag in underwater.to_numpy():
        current = current + 1 if flag else 0
        duration = max(duration, current)
    wins = r[r > 0]
    losses = r[r < 0]
    hit = float((r > 0).mean())
    if losses.empty or float(losses.mean()) == 0.0:
        payoff = float("inf") if not wins.empty else float("nan")
    else:
        payoff = float(wins.mean() / abs(losses.mean())) if not wins.empty else 0.0
    turnover = float(w.diff().abs().sum() / years) if years > 0 else float("nan")
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_duration": float(duration),
        "hit_rate": hit,
        "payoff_ratio": payoff,
        "ann_turnover": turnover,
    }


def _metrics_obj(payload: Mapping[str, float]) -> StrategyMetrics:
    return StrategyMetrics(
        ann_return=float(payload["ann_return"]),
        ann_vol=float(payload["ann_vol"]),
        sharpe=float(payload["sharpe"]),
        sortino=float(payload["sortino"]),
        calmar=float(payload["calmar"]),
        max_drawdown=float(payload["max_drawdown"]),
        max_drawdown_duration=int(payload["max_drawdown_duration"]),
        hit_rate=float(payload["hit_rate"]),
        payoff_ratio=float(payload["payoff_ratio"]),
        ann_turnover=float(payload["ann_turnover"]),
    )


def _first_bday(year: int, month: int) -> pd.Timestamp:
    start = pd.Timestamp(year=year, month=month, day=1)
    return pd.Timestamp(pd.bdate_range(start, periods=1)[0])


def _reestimation_dates(index: pd.DatetimeIndex, min_obs: int) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    periods = pd.DatetimeIndex(index).to_period("M").unique()
    for period in periods:
        tau = _first_bday(period.year, period.month)
        if tau not in index:
            continue
        n_before = int((index < tau).sum())
        if n_before >= min_obs:
            dates.append(pd.Timestamp(tau))
    return dates


def _cost_base_bps(costs: object) -> float:
    if getattr(costs, "instrument", "HYG") == "LQD":
        return float(costs.lqd_base_bps)
    return float(costs.base_bps)


def _apply_costs(
    credit_returns: pd.Series,
    weights: pd.Series,
    sigma: pd.Series,
    *,
    base_bps: float,
    k: float,
) -> pd.Series:
    aligned = pd.concat({"r": credit_returns, "w": weights, "sig": sigma}, axis=1).dropna()
    held = aligned["w"].shift(1).fillna(0.0)
    delta = aligned["w"] - held
    expanding_med = aligned["sig"].expanding(min_periods=1).median()
    costs = []
    for sig, med, dw in zip(aligned["sig"], expanding_med, delta):
        med_use = float(med) if float(med) > 0 else float(aligned["sig"].iloc[0])
        bps = state_dependent_cost_bps(
            sigma=float(sig), sigma_median=med_use, base_bps=base_bps, k=k
        )
        costs.append((bps / 10000.0) * abs(float(dw)))
    strat = held * aligned["r"] - pd.Series(costs, index=aligned.index)
    strat.name = "strategy_return"
    return strat


class WalkForwardBacktester:
    """Monthly walk-forward GARCH + dislocation score + economic report."""

    def __init__(
        self,
        *,
        credit_returns: pd.Series,
        oas_level: pd.Series,
        ebp_level: pd.Series,
        default_proxy: pd.Series,
        vix: pd.Series,
        signal_config: DislocationConfig,
        backtest_config: BacktestConfig,
        oas_estimator: VolatilityEstimator,
        ebp_estimator: VolatilityEstimator,
    ) -> None:
        self.credit_returns = _norm(credit_returns, "credit_return")
        self.oas_level = _norm(oas_level, "oas_level")
        self.ebp_level = _norm(ebp_level, "ebp_level")
        self.default_proxy = _norm(default_proxy, "default_proxy")
        self.vix = _norm(vix, "vix")
        self.signal_config = signal_config
        self.config = backtest_config
        self.oas_estimator = oas_estimator
        self.ebp_estimator = ebp_estimator

    def run(self) -> BacktestResult:
        calendar = self._calendar()
        taus = _reestimation_dates(calendar, self.config.walk_forward.min_estimation_obs)
        if len(taus) < 1:
            raise BacktestError(
                "no monthly reestimation date has the required estimation window"
            )
        sigma_oas = pd.Series(index=calendar, dtype="float64")
        sigma_ebp = pd.Series(index=calendar, dtype="float64")
        traj_rows: list[dict[str, object]] = []
        for i, tau in enumerate(taus):
            next_tau = taus[i + 1] if i + 1 < len(taus) else calendar[-1] + pd.Timedelta(days=1)
            oas_train = self.oas_level.loc[self.oas_level.index < tau]
            ebp_train = self.ebp_level.loc[self.ebp_level.index < tau]
            if oas_train.shape[0] < self.config.walk_forward.min_estimation_obs:
                continue
            params_oas = self.oas_estimator.fit(oas_train)
            params_ebp = self.ebp_estimator.fit(ebp_train)
            segment = calendar[(calendar >= tau) & (calendar < next_tau)]
            if segment.empty:
                continue
            end = segment[-1]
            filt_oas = self.oas_estimator.filter(params_oas, self.oas_level.loc[:end])
            filt_ebp = self.ebp_estimator.filter(params_ebp, self.ebp_level.loc[:end])
            sigma_oas.loc[segment] = filt_oas.reindex(segment)
            sigma_ebp.loc[segment] = filt_ebp.reindex(segment)
            for day in segment:
                traj_rows.append(
                    {
                        "date": day,
                        "oas_beta": float(params_oas.values.get("beta", float("nan"))),
                        "oas_alpha": float(params_oas.values.get("alpha", float("nan"))),
                        "oas_omega": float(params_oas.values.get("omega", float("nan"))),
                        "ebp_beta": float(params_ebp.values.get("beta", float("nan"))),
                        "ebp_alpha": float(params_ebp.values.get("alpha", float("nan"))),
                        "ebp_omega": float(params_ebp.values.get("omega", float("nan"))),
                        "reestimation_date": tau,
                    }
                )
            logger.info(
                "walk-forward reestimate %s through %s n_train=%s",
                tau.date(),
                end.date(),
                int(oas_train.shape[0]),
            )
        tradable = sigma_oas.dropna().index.intersection(sigma_ebp.dropna().index)
        if tradable.empty:
            raise BacktestError("walk-forward produced no tradable dates")
        inputs = SignalInputs(
            sigma_ebp=sigma_ebp.loc[tradable],
            sigma_oas=sigma_oas.loc[tradable],
            ebp_level=self.ebp_level.reindex(tradable).ffill(),
            oas_level=self.oas_level.reindex(tradable).ffill(),
            default_proxy=self.default_proxy.reindex(tradable).ffill(),
        )
        engine = DislocationSignalEngine(inputs, self.signal_config)
        hist = engine.history()
        panel = hist.copy()
        panel["sigma_oas_wf"] = sigma_oas.reindex(panel.index)
        panel["credit_return"] = self.credit_returns.reindex(panel.index)
        costs = self.config.costs
        base_bps = _cost_base_bps(costs)
        strat = _apply_costs(
            panel["credit_return"],
            panel["weight"],
            panel["sigma_oas"],
            base_bps=base_bps,
            k=costs.k,
        )
        panel["strategy_return"] = strat
        ppy = self.config.metrics.periods_per_year
        rf = self.config.metrics.risk_free
        strategy = _metrics_obj(
            economic_metrics(strat, weights=panel["weight"], periods_per_year=ppy, risk_free=rf)
        )
        sensitivity: dict[float, StrategyMetrics] = {}
        for mult in costs.sensitivity_multipliers:
            priced = _apply_costs(
                panel["credit_return"],
                panel["weight"],
                panel["sigma_oas"],
                base_bps=base_bps * mult,
                k=costs.k * mult,
            )
            sensitivity[float(mult)] = _metrics_obj(
                economic_metrics(priced, weights=panel["weight"], periods_per_year=ppy, risk_free=rf)
            )
        benches = self._benchmarks(panel, ppy)
        predictive = []
        for horizon in self.config.predictive.horizons:
            fwd = forward_returns(self.credit_returns, horizon)
            try:
                predictive.append(
                    predictive_regression(fwd.reindex(panel.index), panel["score"], horizon=horizon)
                )
            except BacktestError:
                logger.warning("skipping predictive horizon %s: insufficient overlap", horizon)
        primary_h = self.config.predictive.horizons[0]
        fwd_q = forward_returns(self.credit_returns, primary_h).reindex(panel.index)
        quintiles = quintile_table(fwd_q, panel["score"])
        n_eps = count_independent_episodes(
            panel["active"], min_gap_calendar_days=self.config.episodes.min_gap_calendar_days
        )
        boot = block_bootstrap_ci(
            strat.dropna(),
            block_length=self.config.episodes.block_length,
            n_bootstrap=self.config.episodes.n_bootstrap,
            seed=self.config.seed,
        )
        traj = pd.DataFrame(traj_rows).set_index("date").sort_index()
        failures = _failure_conditions(
            predictive, quintiles, n_eps, strategy, sensitivity, benches, traj, panel
        )
        triple = sensitivity.get(3.0)
        implementable = not (
            triple is not None and (triple.ann_return <= 0 or triple.sharpe <= 0)
        )
        return BacktestResult(
            panel=panel,
            reestimation_dates=taus,
            parameter_trajectory=traj,
            predictive=predictive,
            quintiles=quintiles,
            n_independent_episodes=n_eps,
            bootstrap=boot,
            strategy=strategy,
            cost_sensitivity=sensitivity,
            benchmarks=benches,
            failure_conditions=failures,
            implementable=implementable,
            n_daily=int(panel.shape[0]),
        )

    def write_report(self, result: BacktestResult) -> ReportPaths:
        plot_dir = Path(self.config.plot.output_directory)
        plot_dir.mkdir(parents=True, exist_ok=True)
        equity_path = plot_dir / "equity_curve.png"
        dd_path = plot_dir / "drawdown.png"
        param_path = plot_dir / "parameter_trajectory.png"
        _plot_equity(result.panel, equity_path, self.config.plot)
        _plot_drawdown(result.panel, dd_path, self.config.plot)
        _plot_params(result.parameter_trajectory, param_path, self.config.plot)
        md = _render_markdown(result, equity_path, dd_path, param_path)
        md_path = Path(self.config.output.report_markdown)
        html_path = Path(self.config.output.report_html)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        html_path.write_text(_as_html(md), encoding="utf-8")
        logger.info("wrote backtest report %s and %s", md_path, html_path)
        return ReportPaths(markdown=md_path, html=html_path)

    def _calendar(self) -> pd.DatetimeIndex:
        index = (
            self.credit_returns.dropna().index.intersection(self.oas_level.dropna().index)
            .intersection(self.ebp_level.dropna().index)
            .intersection(self.default_proxy.dropna().index)
            .intersection(self.vix.dropna().index)
        )
        return pd.DatetimeIndex(index).sort_values()

    def _benchmarks(self, panel: pd.DataFrame, ppy: int) -> dict[str, StrategyMetrics]:
        spec = self.config.benchmarks
        costs = self.config.costs
        r = self.credit_returns.reindex(panel.index)
        sigma = panel["sigma_oas"]

        def _priced(weights: pd.Series) -> StrategyMetrics:
            strat = _apply_costs(
                r, weights, sigma, base_bps=_cost_base_bps(costs), k=costs.k
            )
            return _metrics_obj(
                economic_metrics(
                    strat, weights=weights, periods_per_year=ppy,
                    risk_free=self.config.metrics.risk_free,
                )
            )

        win = min(spec.realized_vol_window, max(len(panel) - 1, 2))
        oas_p = rolling_percentile_rank(
            self.oas_level.reindex(panel.index).ffill(), window=win, min_periods=min(8, len(panel))
        )
        vix_p = rolling_percentile_rank(
            self.vix.reindex(panel.index).ffill(), window=win, min_periods=min(8, len(panel))
        )
        thresh = spec.level_percentile / 100.0
        return {
            "buy_hold": _priced(pd.Series(1.0, index=panel.index)),
            "oas_level": _priced((oas_p > thresh).astype(float).reindex(panel.index).fillna(0.0)),
            "vix": _priced((vix_p > thresh).astype(float).reindex(panel.index).fillna(0.0)),
            "realized_vol": _priced(self._realized_vol_weights(panel.index)),
        }

    def _realized_vol_weights(self, index: pd.DatetimeIndex) -> pd.Series:
        window = self.config.benchmarks.realized_vol_window
        sigma_oas = self.oas_level.diff().rolling(window, min_periods=max(5, window // 4)).std()
        sigma_ebp = self.ebp_level.diff().rolling(window, min_periods=max(5, window // 4)).std()
        aligned = index.intersection(sigma_oas.dropna().index).intersection(sigma_ebp.dropna().index)
        if aligned.empty:
            return pd.Series(0.0, index=index)
        inputs = SignalInputs(
            sigma_ebp=sigma_ebp.loc[aligned],
            sigma_oas=sigma_oas.loc[aligned],
            ebp_level=self.ebp_level.reindex(aligned).ffill(),
            oas_level=self.oas_level.reindex(aligned).ffill(),
            default_proxy=self.default_proxy.reindex(aligned).ffill(),
        )
        try:
            hist = DislocationSignalEngine(inputs, self.signal_config).history()
        except Exception:
            logger.exception("realized-vol benchmark engine failed")
            return pd.Series(0.0, index=index)
        return hist["weight"].reindex(index).fillna(0.0)


def _norm(series: pd.Series, name: str) -> pd.Series:
    out = series.copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    out = out.sort_index().astype("float64")
    out.name = name
    if out.index.has_duplicates:
        raise BacktestError(f"duplicate dates in {name}")
    return out


def _failure_conditions(
    predictive: list[PredictiveReport],
    quintiles: pd.DataFrame,
    n_episodes: int,
    strategy: StrategyMetrics,
    sensitivity: dict[float, StrategyMetrics],
    benches: dict[str, StrategyMetrics],
    traj: pd.DataFrame,
    panel: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []
    if predictive and all(item.pvalue > 0.10 for item in predictive):
        notes.append(
            "El score no predice retornos forward a ningún horizonte "
            f"({', '.join(str(p.horizon) for p in predictive)} d) "
            "con p(Newey-West) ≤ 0.10."
        )
    means = quintiles["mean_forward"].dropna()
    if len(means) >= 2 and not bool(means.is_monotonic_increasing):
        notes.append(
            f"Los quintiles de retorno forward no son monótonos "
            f"(Q1={float(means.get(1, float('nan'))):.4f}, "
            f"Q5={float(means.get(5, float('nan'))):.4f})."
        )
    if n_episodes < 5:
        notes.append(
            f"Solo {n_episodes} episodios independientes (hueco ≥ 60 días). "
            f"El N diario ({int(panel.shape[0])}) sobreestima los grados de libertad."
        )
    rv = benches.get("realized_vol")
    if rv is not None and np.isfinite(rv.sharpe) and np.isfinite(strategy.sharpe):
        if rv.sharpe >= strategy.sharpe:
            notes.append(
                f"GARCH no supera a la vol realizada de ventana móvil "
                f"(Sharpe GARCH={strategy.sharpe:.3f} ≤ realizada={rv.sharpe:.3f}). "
                "La complejidad econométrica no está justificada en esta muestra."
            )
    triple = sensitivity.get(3.0)
    if triple is not None and (triple.ann_return <= 0 or triple.sharpe <= 0):
        notes.append(
            f"No implementable: al triplicar costos el retorno anualizado es "
            f"{triple.ann_return:.3%} y el Sharpe {triple.sharpe:.3f}."
        )
    bh = benches.get("buy_hold")
    if bh is not None and np.isfinite(bh.ann_return) and strategy.ann_return < bh.ann_return:
        notes.append(
            f"La señal pierde contra buy & hold de HYG "
            f"({strategy.ann_return:.3%} vs {bh.ann_return:.3%})."
        )
    if "oas_beta" in traj and traj["oas_beta"].notna().sum() > 3:
        beta = traj["oas_beta"].groupby(traj.index.to_period("M")).first()
        if float(beta.mean()) != 0 and float(beta.std()) / abs(float(beta.mean())) > 0.15:
            notes.append(
                "Deriva paramétrica fuerte en beta de OAS "
                f"(CV={beta.std() / abs(beta.mean()):.2f}); considerar MS-GARCH."
            )
    monthly = panel["strategy_return"].dropna().groupby(lambda d: d.to_period("M")).sum()
    if not monthly.empty:
        notes.append(
            f"El peor mes de la estrategia es {monthly.idxmin()} "
            f"({float(monthly.min()):.2%} de retorno del mes)."
        )
    if not notes:
        notes.append(
            "En esta muestra no se identificó un modo de fallo de los umbrales "
            "automáticos; revisar igualmente el N de episodios."
        )
    return notes


def _mpl(plot_cfg: object) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(plot_cfg.output_directory)))


def _plot_equity(panel: pd.DataFrame, dest: Path, plot_cfg: object) -> None:
    _mpl(plot_cfg)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = panel["strategy_return"].dropna()
    equity = (1.0 + r).cumprod()
    fig, ax = plt.subplots(figsize=(plot_cfg.figsize_width, plot_cfg.figsize_height))
    ax.plot(equity.index, equity.to_numpy(), color="tab:blue", label="estrategia")
    ax.set_title("Equity curve (tras costos dependientes del estado)")
    ax.legend()
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=plot_cfg.dpi)
    plt.close(fig)


def _plot_drawdown(panel: pd.DataFrame, dest: Path, plot_cfg: object) -> None:
    _mpl(plot_cfg)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = panel["strategy_return"].dropna()
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(plot_cfg.figsize_width, plot_cfg.figsize_height))
    ax.fill_between(dd.index, dd.to_numpy(), 0.0, color="tab:red", alpha=0.4)
    ax.set_title("Drawdown")
    fig.tight_layout()
    fig.savefig(dest, dpi=plot_cfg.dpi)
    plt.close(fig)


def _plot_params(traj: pd.DataFrame, dest: Path, plot_cfg: object) -> None:
    _mpl(plot_cfg)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(plot_cfg.figsize_width, plot_cfg.figsize_height))
    if "oas_beta" in traj:
        ax.plot(traj.index, traj["oas_beta"], label="OAS β (congelado intra-mes)")
    if "ebp_beta" in traj:
        ax.plot(traj.index, traj["ebp_beta"], label="EBP β (congelado intra-mes)")
    ax.set_title("Trayectoria de parámetros walk-forward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=plot_cfg.dpi)
    plt.close(fig)


def _render_markdown(
    result: BacktestResult,
    equity: Path,
    drawdown: Path,
    params: Path,
) -> str:
    lines = [
        "# Backtest del score de dislocación",
        "",
        "Protocolo walk-forward: reestimación **mensual** el primer día hábil, "
        "parámetros congelados, σ diaria por recursión. Prohibido un GARCH de muestra completa.",
        "",
        f"- Observaciones diarias de trading: **{result.n_daily}**",
        f"- Episodios independientes (hueco ≥ 60 días): **{result.n_independent_episodes}**",
        f"- Reestimaciones: {', '.join(d.strftime('%Y-%m-%d') for d in result.reestimation_dates)}",
        f"- Implementable al triplicar costos: **{'sí' if result.implementable else 'no'}**",
        "",
        "## [V1] Poder predictivo (Newey-West, lag = h)",
        "",
        "| h | N | b | t (NW) | EE NW | EE OLS | R² | p |",
        "|---|---|---|--------|-------|--------|----|---|",
    ]
    for row in result.predictive:
        lines.append(
            f"| {row.horizon} | {row.n_obs} | {row.beta:.4f} | {row.t_stat:.2f} | "
            f"{row.se_newey_west:.4f} | {row.se_ols:.4f} | {row.r_squared:.3f} | {row.pvalue:.3f} |"
        )
    lines += [
        "",
        "## [V2] Retorno forward medio por quintil de score",
        "",
        result.quintiles.to_string(),
        "",
        "## [V3] Muestra efectiva y bootstrap por bloques",
        "",
        f"Episodios independientes: **{result.n_independent_episodes}** "
        f"(no uses el N diario = {result.n_daily} como si fueran i.i.d.).",
        f"IC 95% del retorno medio diario (block bootstrap, L="
        f"{result.bootstrap.block_length}, B={result.bootstrap.n_bootstrap}): "
        f"[{result.bootstrap.lower:.5f}, {result.bootstrap.upper:.5f}] "
        f"(punto {result.bootstrap.point:.5f}).",
        "",
        "## [E1] Métricas económicas de la señal (costos base)",
        "",
        _metrics_table({"señal GARCH": result.strategy}),
        "",
        "## [E2] Sensibilidad a costos (base y k × 1 / 2 / 3)",
        "",
        _metrics_table({f"×{m:g}": mets for m, mets in sorted(result.cost_sensitivity.items())}),
        "",
        "## [E3] Benchmarks ingenuos",
        "",
        _metrics_table(
            {
                "señal GARCH": result.strategy,
                "Buy & hold": result.benchmarks["buy_hold"],
                "OAS nivel p>85": result.benchmarks["oas_level"],
                "VIX p>85": result.benchmarks["vix"],
                "vol realizada 60d": result.benchmarks["realized_vol"],
            }
        ),
        "",
        "## Condiciones bajo las cuales esta señal falla",
        "",
        "Derivadas de *este* backtest, no de teoría:",
        "",
    ]
    for note in result.failure_conditions:
        lines.append(f"- {note}")
    lines += [
        "",
        "## Gráficos",
        "",
        f"![Equity]({equity})",
        f"![Drawdown]({drawdown})",
        f"![Parámetros]({params})",
        "",
    ]
    return "\n".join(lines)


def _metrics_table(rows: Mapping[str, StrategyMetrics]) -> str:
    header = (
        "| estrategia | ret. an. | vol. an. | Sharpe | Sortino | Calmar | "
        "max DD | DD días | hit | payoff | turnover an. |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for name, m in rows.items():
        lines.append(
            f"| {name} | {_pct(m.ann_return)} | {_pct(m.ann_vol)} | {_n(m.sharpe)} | "
            f"{_n(m.sortino)} | {_n(m.calmar)} | {_pct(m.max_drawdown)} | "
            f"{m.max_drawdown_duration} | {_n(m.hit_rate)} | {_n(m.payoff_ratio)} | "
            f"{_n(m.ann_turnover)} |"
        )
    return "\n".join(lines)


def _pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.2%}"


def _n(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.3f}"


def _as_html(markdown: str) -> str:
    escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>"
        "<title>Backtest dislocación</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:2rem auto;}"
        "pre{white-space:pre-wrap}</style></head><body><pre>"
        f"{escaped}</pre></body></html>\n"
    )
