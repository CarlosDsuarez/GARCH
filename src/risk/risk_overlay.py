"""Risk overlay: aggregate exposure on top of an existing optimiser.

Architectural split
-------------------
The HRP (or other) optimiser chooses *relative* weights :math:`w^{\\mathrm{raw}}`
with :math:`\\sum_i w_i=1`. This module chooses the *level* of deployment
:math:`m_t\\in[m_{\\min},1]`. Final weights:

.. math::

    w^{\\mathrm{final}}_t = m_t\\, w^{\\mathrm{raw}}_t,
    \\qquad
    w^{\\mathrm{cash}}_t = 1-m_t.

Mixing selection and sizing in one objective makes it impossible to tell
which decision produced a bad year. They stay separate on purpose.

Multiplier (minimum, not product)
---------------------------------
Three overlapping views of the same risk. The product would drive exposure
toward zero exactly when valuations are cheapest. The binding constraint is
the right risk-budget logic:

.. math::

    m^{\\mathrm{vol}}_t = \\min\\bigl(m_{\\mathrm{cap}},\\,
        \\sigma^\\star / \\sigma^{\\mathrm{ann}}_{t}\\bigr)

    m^{\\mathrm{ES}}_t = \\min\\bigl(1,\\,
        \\mathrm{ES}^\\star / \\mathrm{ES}_t\\bigr)

    m^{\\mathrm{regime}}_t =
    \\begin{cases}
      1 & \\text{calm}\\\\
      \\kappa & \\text{stress, exogenous confirm}\\\\
      1-(1-\\kappa)/2 & \\text{stress, unconfirmed}
    \\end{cases}

    m_t = \\min\\bigl(m^{\\mathrm{vol}}_t, m^{\\mathrm{ES}}_t, m^{\\mathrm{regime}}_t\\bigr).

:math:`\\sigma^{\\mathrm{ann}}` is the GJR one-step variance of the *portfolio*
return, annualised. ES is a **positive** loss at ``es_alpha`` over
``es_horizon`` days (production default: 97.5%, **1 day**). Do not scale a
one-day ES by :math:`\\sqrt{h}`.

:math:`m_{\\mathrm{cap}}=1` unless the mandate sets ``allow_leverage``. That is
a policy choice, not an implementation default.

Turnover
--------
:math:`m_t` is computed daily. It is executed only when
:math:`|m^{\\mathrm{smooth}}_t - m^{\\mathrm{exec}}_{t-1}|` exceeds the band
(default 5 percentage points of exposure). EMA smoothing (default
:math:`\\alpha=0.3`) is applied **before** the band. Incremental annualised
turnover above 200% versus the base book is an alarm, not a knob to hide.

Autoportfolio regime detector
-----------------------------
The HMM / dual-regime stack in ``strategy-lab`` (autoportfolio) is **not**
deleted. Pass it in as a boolean stress series (map 4-state HMM via
:func:`hmm_states_to_stress`). :func:`compare_regime_signals` is the research
deliverable that decides whether MS-GARCH, the legacy detector, or their
conjunction should govern production.

[C1]
----
``sigma_ann[t]``, ``es[t]`` and the regime flag at :math:`t` must already be
causal (forecasts formed at :math:`t-1`). This overlay never looks at
:math:`t+1`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from backtest.signal_backtest import StrategyMetrics, economic_metrics
from risk.schema import OverlayConfig

logger = logging.getLogger(__name__)

__all__ = [
    "OverlayBacktestResult",
    "OverlayError",
    "OverlayExplanation",
    "OverlayReportPaths",
    "RegimeComparison",
    "RiskOverlay",
    "annualize_daily_sigma",
    "compare_regime_signals",
    "hmm_states_to_stress",
    "overlay_sensitivity",
    "run_overlay_backtest",
]

Binding = Literal["vol", "es", "regime", "tie"]


class OverlayError(ValueError):
    """Overlay input or date is not usable ([C7])."""


@dataclass(frozen=True)
class OverlayExplanation:
    date: pd.Timestamp
    m_vol: float
    m_es: float
    m_regime: float
    m_raw: float
    m_smoothed: float
    m_executed: float
    binding: Binding
    reason: str
    es_horizon: int
    es_alpha: float


@dataclass(frozen=True)
class OverlayReportPaths:
    report_markdown: Path
    plot_path: Path | None


@dataclass
class OverlayBacktestResult:
    metrics: dict[str, StrategyMetrics]
    multipliers: pd.DataFrame
    incremental_turnover: dict[str, float]
    turnover_alarm: dict[str, bool]


@dataclass
class RegimeComparison:
    agreement_rate: float
    n_days: int
    n_agree: int
    discrepancies: pd.DataFrame
    episode_leaders: pd.DataFrame
    overlay_ms: StrategyMetrics
    overlay_legacy: StrategyMetrics
    overlay_conjunction: StrategyMetrics


def annualize_daily_sigma(sigma_daily: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """Convert GJR one-day :math:`\\sigma` to annualised units."""
    if periods_per_year < 1:
        raise OverlayError("periods_per_year must be >= 1")
    series = _as_series(sigma_daily, "sigma_daily")
    if (series <= 0.0).any():
        raise OverlayError("daily sigma must be strictly positive")
    return (series * float(np.sqrt(periods_per_year))).rename("sigma_ann")


def hmm_states_to_stress(
    states: pd.Series,
    stress_ids: Sequence[int] = (2, 3),
) -> pd.Series:
    """Map autoportfolio 4-state HMM labels to a boolean stress flag.

    Does **not** replace the HMM. The overlay still consumes the boolean;
    production continues to own the four-state model.
    """
    series = states.copy()
    series.index = pd.DatetimeIndex(series.index).normalize()
    allowed = {int(i) for i in stress_ids}
    return series.map(lambda s: int(s) in allowed).astype(bool).rename("legacy_stress")


class RiskOverlay:
    """Compute and execute the exposure multiplier on HRP weights."""

    def __init__(self, config: OverlayConfig) -> None:
        if config.aggregator != "min":
            raise OverlayError("aggregator must be min; the product over-de-risks")
        self.config = config
        self.sigma_ann: pd.Series | None = None
        self.es: pd.Series | None = None
        self.states: pd.DataFrame | None = None
        self.legacy_stress: pd.Series | None = None
        self.ms_stress: pd.Series | None = None
        self.conjunction_stress: pd.Series | None = None
        self.panel: pd.DataFrame | None = None

    def build(
        self,
        sigma_ann: pd.Series,
        es: pd.Series,
        states: pd.DataFrame,
        *,
        legacy_stress: pd.Series | None = None,
    ) -> RiskOverlay:
        sig = _as_series(sigma_ann, "sigma_ann")
        es_s = _as_series(es, "es")
        st = _as_state_frame(states)
        index = sig.index.intersection(es_s.index).intersection(st.index)
        if index.empty:
            raise OverlayError("sigma, ES and regime states have no overlapping dates")
        sig = sig.reindex(index)
        es_s = es_s.reindex(index)
        st = st.reindex(index)
        if sig.isna().any() or es_s.isna().any() or st.isna().any().any():
            raise OverlayError("sigma, ES or regime states contain NaN on the overlap")
        if (sig <= 0.0).any() or (es_s <= 0.0).any():
            raise OverlayError("sigma and ES must be strictly positive")
        self.sigma_ann = sig
        self.es = es_s
        self.states = st
        self.ms_stress = st["label"].astype(str) == "stress"
        if legacy_stress is None:
            logger.warning(
                "no autoportfolio regime series supplied; full_legacy and "
                "conjunction treat legacy as never-in-stress"
            )
            self.legacy_stress = pd.Series(False, index=index, dtype=bool)
        else:
            legacy = legacy_stress.reindex(index)
            if legacy.isna().any():
                raise OverlayError("legacy stress series has NaN on the overlay index")
            self.legacy_stress = legacy.astype(bool)
        self.conjunction_stress = self.ms_stress & self.legacy_stress
        self.panel = self._build_panel()
        logger.info(
            "risk overlay n=%s kappa=%.2f sigma*=%.3f band=%.3f es_horizon=%s",
            int(len(index)),
            self.config.kappa,
            self.config.sigma_target,
            self.config.band,
            self.config.es_horizon,
        )
        return self

    def compute_multiplier(self, date: object) -> float:
        ts = self._require_date(date)
        return float(self.panel.loc[ts, "m_exec_full_ms"])

    def apply(self, weights: pd.Series, date: object) -> pd.Series:
        ts = self._require_date(date)
        w = weights.astype("float64").copy()
        if "cash" in w.index:
            raise OverlayError("w_raw must not include cash; the overlay adds it")
        total = float(w.sum())
        if abs(total - 1.0) > 1e-8:
            raise OverlayError(f"w_raw must sum to 1, got {total}")
        m = self.compute_multiplier(ts)
        deployed = w * m
        deployed["cash"] = 1.0 - m
        return deployed

    def explain(self, date: object) -> OverlayExplanation:
        ts = self._require_date(date)
        row = self.panel.loc[ts]
        m_vol = float(row["m_vol"])
        m_es = float(row["m_es"])
        m_reg = float(row["m_regime_ms"])
        m_raw = float(row["m_raw_full_ms"])
        binding, reason = _binding(m_vol, m_es, m_reg, m_raw)
        return OverlayExplanation(
            date=ts,
            m_vol=m_vol,
            m_es=m_es,
            m_regime=m_reg,
            m_raw=m_raw,
            m_smoothed=float(row["m_smooth_full_ms"]),
            m_executed=float(row["m_exec_full_ms"]),
            binding=binding,
            reason=reason,
            es_horizon=int(self.config.es_horizon),
            es_alpha=float(self.config.es_alpha),
        )

    def write_report(self, returns: pd.Series) -> OverlayReportPaths:
        self._require_panel()
        backtest = run_overlay_backtest(self, returns)
        table, heat = overlay_sensitivity(self, returns)
        episodes = [
            (ep.label, pd.Timestamp(ep.start), pd.Timestamp(ep.end))
            for ep in self.config.episodes
        ]
        comparison = compare_regime_signals(self, returns=returns, episodes=episodes)
        md_path = Path(self.config.output.report_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(
            _render_markdown(self, backtest, table, comparison),
            encoding="utf-8",
        )
        return OverlayReportPaths(report_markdown=md_path, plot_path=heat)

    def _build_panel(self) -> pd.DataFrame:
        cfg = self.config
        cap = float(cfg.leverage_cap)
        m_vol = np.minimum(cap, cfg.sigma_target / self.sigma_ann.to_numpy(dtype=float))
        m_es = np.minimum(1.0, cfg.es_budget / self.es.to_numpy(dtype=float))
        labels = self.states["label"].astype(str).to_numpy()
        confirmed = self.states["exogenous_confirms"].astype(bool).to_numpy()
        m_reg_ms = np.array(
            [_regime_m(lab, conf, cfg.kappa) for lab, conf in zip(labels, confirmed, strict=True)],
            dtype=float,
        )
        m_reg_leg = np.where(self.legacy_stress.to_numpy(dtype=bool), cfg.kappa, 1.0)
        m_reg_and = np.where(self.conjunction_stress.to_numpy(dtype=bool), cfg.kappa, 1.0)
        index = self.sigma_ann.index
        frame = pd.DataFrame(
            {
                "m_vol": m_vol,
                "m_es": m_es,
                "m_regime_ms": m_reg_ms,
                "m_regime_legacy": m_reg_leg,
                "m_regime_and": m_reg_and,
            },
            index=index,
        )
        specs: dict[str, np.ndarray] = {
            "base": np.ones(len(index), dtype=float),
            "vol_only": m_vol,
            "full_ms": np.minimum(np.minimum(m_vol, m_es), m_reg_ms),
            "full_legacy": np.minimum(np.minimum(m_vol, m_es), m_reg_leg),
            "conjunction": np.minimum(np.minimum(m_vol, m_es), m_reg_and),
        }
        for name, raw in specs.items():
            clipped = np.clip(raw, cfg.m_min, cap)
            smooth, executed = _smooth_and_band(clipped, alpha=cfg.smoothing, band=cfg.band)
            frame[f"m_raw_{name}"] = clipped
            frame[f"m_smooth_{name}"] = smooth
            frame[f"m_exec_{name}"] = executed
        return frame

    def _require_panel(self) -> None:
        if self.panel is None or self.sigma_ann is None or self.states is None:
            raise OverlayError("build() is required before compute_multiplier/apply/explain")

    def _require_date(self, date: object) -> pd.Timestamp:
        self._require_panel()
        ts = pd.Timestamp(date).normalize()
        if ts not in self.panel.index:
            raise OverlayError(f"date {ts.date()} is not in the overlay sample")
        return ts


def run_overlay_backtest(overlay: RiskOverlay, returns: pd.Series) -> OverlayBacktestResult:
    """Four books: base, vol-only, full MS-GARCH overlay, full legacy overlay."""
    overlay._require_panel()
    r = _as_series(returns, "r").reindex(overlay.panel.index).dropna()
    panel = overlay.panel.loc[r.index]
    names = ("base", "vol_only", "full_ms", "full_legacy")
    metrics: dict[str, StrategyMetrics] = {}
    incremental: dict[str, float] = {}
    alarms: dict[str, bool] = {}
    ppy = overlay.config.periods_per_year
    for name in names:
        m = panel[f"m_exec_{name}"]
        pnl = (m * r).rename(name)
        payload = economic_metrics(pnl, weights=m, periods_per_year=ppy)
        metrics[name] = _to_metrics(payload)
    base_to = metrics["base"].ann_turnover
    for name in names:
        extra = metrics[name].ann_turnover - base_to
        incremental[name] = float(extra)
        alarms[name] = bool(extra > overlay.config.turnover_alarm)
        if alarms[name]:
            logger.warning(
                "overlay %s adds %.0f%% annualised turnover over the base book "
                "(alarm threshold %.0f%%); costs will dominate — revisit band/smoothing",
                name,
                100.0 * extra,
                100.0 * overlay.config.turnover_alarm,
            )
    return OverlayBacktestResult(
        metrics=metrics,
        multipliers=panel[[f"m_exec_{n}" for n in names]].copy(),
        incremental_turnover=incremental,
        turnover_alarm=alarms,
    )


def compare_regime_signals(
    overlay: RiskOverlay,
    *,
    returns: pd.Series,
    episodes: Sequence[tuple[str, pd.Timestamp, pd.Timestamp]] | None = None,
) -> RegimeComparison:
    """Concordance of MS-GARCH vs the autoportfolio detector. Research deliverable."""
    overlay._require_panel()
    if overlay.ms_stress is None or overlay.legacy_stress is None:
        raise OverlayError("build() with both detectors is required")
    ms = overlay.ms_stress.astype(bool)
    legacy = overlay.legacy_stress.astype(bool)
    aligned = pd.concat({"ms": ms, "legacy": legacy}, axis=1).dropna()
    agree = aligned["ms"] == aligned["legacy"]
    n = int(aligned.shape[0])
    n_agree = int(agree.sum())
    disc_rows = []
    for ts, row in aligned.loc[~agree].iterrows():
        direction = (
            "ms_stress_legacy_calm" if bool(row["ms"]) else "ms_calm_legacy_stress"
        )
        disc_rows.append(
            {
                "date": pd.Timestamp(ts),
                "ms": bool(row["ms"]),
                "legacy": bool(row["legacy"]),
                "direction": direction,
            }
        )
    discrepancies = pd.DataFrame(disc_rows)
    if episodes is None:
        episodes = [
            (ep.label, pd.Timestamp(ep.start), pd.Timestamp(ep.end))
            for ep in overlay.config.episodes
        ]
    leader_rows = []
    for label, start, end in episodes:
        window = aligned.loc[
            (aligned.index >= pd.Timestamp(start).normalize())
            & (aligned.index <= pd.Timestamp(end).normalize())
        ]
        ms_first = _first_true(window["ms"]) if not window.empty else None
        leg_first = _first_true(window["legacy"]) if not window.empty else None
        if ms_first is None and leg_first is None:
            leader, lead_days = "neither", 0
        elif ms_first is None:
            leader, lead_days = "legacy", 0
        elif leg_first is None:
            leader, lead_days = "ms", 0
        elif ms_first < leg_first:
            leader = "ms"
            lead_days = int(window.loc[ms_first:leg_first].shape[0] - 1)
        elif leg_first < ms_first:
            leader = "legacy"
            lead_days = int(window.loc[leg_first:ms_first].shape[0] - 1)
        else:
            leader, lead_days = "tie", 0
        leader_rows.append(
            {
                "episode": label,
                "ms_first": ms_first,
                "legacy_first": leg_first,
                "leader": leader,
                "lead_days": lead_days,
            }
        )
    backtest = run_overlay_backtest(overlay, returns)
    r = _as_series(returns, "r").reindex(overlay.panel.index).dropna()
    m_and = overlay.panel.loc[r.index, "m_exec_conjunction"]
    pnl_and = (m_and * r).rename("conjunction")
    conj_metrics = _to_metrics(
        economic_metrics(pnl_and, weights=m_and, periods_per_year=overlay.config.periods_per_year)
    )
    return RegimeComparison(
        agreement_rate=float(n_agree / n) if n else float("nan"),
        n_days=n,
        n_agree=n_agree,
        discrepancies=discrepancies,
        episode_leaders=pd.DataFrame(leader_rows),
        overlay_ms=backtest.metrics["full_ms"],
        overlay_legacy=backtest.metrics["full_legacy"],
        overlay_conjunction=conj_metrics,
    )


def overlay_sensitivity(
    overlay: RiskOverlay,
    returns: pd.Series,
) -> tuple[pd.DataFrame, Path]:
    """Grid of kappa × sigma_target × band → Sharpe and max drawdown."""
    overlay._require_panel()
    grid = overlay.config.sensitivity
    rows: list[dict[str, float]] = []
    for kappa in grid.kappa:
        for sigma_star in grid.sigma_target:
            for band in grid.band:
                cfg = overlay.config.model_copy(
                    update={
                        "kappa": float(kappa),
                        "sigma_target": float(sigma_star),
                        "band": float(band),
                    }
                )
                alt = RiskOverlay(cfg).build(
                    overlay.sigma_ann,
                    overlay.es,
                    overlay.states,
                    legacy_stress=overlay.legacy_stress,
                )
                bt = run_overlay_backtest(alt, returns)
                m = bt.metrics["full_ms"]
                rows.append(
                    {
                        "kappa": float(kappa),
                        "sigma_target": float(sigma_star),
                        "band": float(band),
                        "sharpe": m.sharpe,
                        "max_drawdown": m.max_drawdown,
                        "ann_turnover": m.ann_turnover,
                    }
                )
    table = pd.DataFrame(rows)
    dest = Path(overlay.config.plot.output_directory) / overlay.config.plot.filename
    _plot_sensitivity(table, dest, overlay.config.plot)
    return table, dest


def _to_metrics(payload: dict[str, float]) -> StrategyMetrics:
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


def _regime_m(label: str, confirmed: bool, kappa: float) -> float:
    if label != "stress":
        return 1.0
    if confirmed:
        return float(kappa)
    return 1.0 - (1.0 - float(kappa)) / 2.0


def _smooth_and_band(
    raw: np.ndarray,
    *,
    alpha: float,
    band: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(raw.size)
    smooth = np.empty(n, dtype=float)
    executed = np.empty(n, dtype=float)
    smooth[0] = raw[0]
    executed[0] = raw[0]
    for t in range(1, n):
        smooth[t] = alpha * raw[t] + (1.0 - alpha) * smooth[t - 1]
        if abs(smooth[t] - executed[t - 1]) > band:
            executed[t] = smooth[t]
        else:
            executed[t] = executed[t - 1]
    return smooth, executed


def _binding(m_vol: float, m_es: float, m_reg: float, m_raw: float) -> tuple[Binding, str]:
    parts = {"vol": m_vol, "es": m_es, "regime": m_reg}
    tied = [name for name, value in parts.items() if abs(value - m_raw) <= 1e-12]
    if len(tied) == 1:
        name = tied[0]
        reason = (
            f"{name} is binding: m_vol={m_vol:.3f}, m_es={m_es:.3f}, "
            f"m_regime={m_reg:.3f} → min={m_raw:.3f} (not the product "
            f"{m_vol * m_es * m_reg:.3f})"
        )
        return name, reason  # type: ignore[return-value]
    reason = (
        f"tie among {', '.join(tied)}: m_vol={m_vol:.3f}, m_es={m_es:.3f}, "
        f"m_regime={m_reg:.3f} → min={m_raw:.3f}"
    )
    return "tie", reason


def _first_true(flag: pd.Series) -> pd.Timestamp | None:
    hits = flag[flag.astype(bool)]
    if hits.empty:
        return None
    return pd.Timestamp(hits.index[0])


def _as_series(series: pd.Series, name: str) -> pd.Series:
    out = series.copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    out = out.sort_index().astype("float64")
    out.name = name
    if out.index.has_duplicates:
        raise OverlayError(f"duplicate dates in {name}")
    return out


def _as_state_frame(states: pd.DataFrame) -> pd.DataFrame:
    if "label" not in states.columns or "exogenous_confirms" not in states.columns:
        raise OverlayError("regime states need columns label and exogenous_confirms")
    frame = states.copy()
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise OverlayError("duplicate dates in regime states")
    return frame[["label", "exogenous_confirms"]]


def _plot_sensitivity(table: pd.DataFrame, dest: Path, plot_cfg: object) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(plot_cfg.output_directory)))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(plot_cfg.figsize_width, plot_cfg.figsize_height),
    )
    band_default = float(table["band"].median())
    slice_ = table[np.isclose(table["band"], band_default)]
    if slice_["sigma_target"].nunique() == 1:
        axes[0].bar(slice_["kappa"].astype(str), slice_["sharpe"])
        axes[1].bar(slice_["kappa"].astype(str), slice_["max_drawdown"])
        axes[0].set_xlabel("kappa")
        axes[1].set_xlabel("kappa")
    else:
        pivot_s = slice_.pivot_table(index="kappa", columns="sigma_target", values="sharpe")
        pivot_d = slice_.pivot_table(index="kappa", columns="sigma_target", values="max_drawdown")
        axes[0].imshow(pivot_s.to_numpy(), aspect="auto")
        axes[1].imshow(pivot_d.to_numpy(), aspect="auto")
        axes[0].set_xticks(range(pivot_s.shape[1]), [f"{c:.2f}" for c in pivot_s.columns])
        axes[0].set_yticks(range(pivot_s.shape[0]), [f"{r:.2f}" for r in pivot_s.index])
        axes[1].set_xticks(range(pivot_d.shape[1]), [f"{c:.2f}" for c in pivot_d.columns])
        axes[1].set_yticks(range(pivot_d.shape[0]), [f"{r:.2f}" for r in pivot_d.index])
        axes[0].set_xlabel("sigma*")
        axes[1].set_xlabel("sigma*")
    axes[0].set_title("Sharpe")
    axes[1].set_title("Max drawdown")
    fig.suptitle("Overlay sensitivity (band slice)")
    fig.tight_layout()
    fig.savefig(dest, dpi=plot_cfg.dpi)
    plt.close(fig)


def _render_markdown(
    overlay: RiskOverlay,
    backtest: OverlayBacktestResult,
    sensitivity: pd.DataFrame,
    comparison: RegimeComparison | None,
) -> str:
    cfg = overlay.config
    lines = [
        "# Risk overlay report",
        "",
        "This layer does **not** replace HRP / the autoportfolio optimiser. It",
        "scales aggregate exposure. Relative weights stay with the optimiser.",
        "",
        f"- aggregator: **{cfg.aggregator}** (product is forbidden)",
        f"- sigma target: {cfg.sigma_target:.2%} annualised",
        f"- ES budget: {cfg.es_budget:.2%} at {cfg.es_alpha:.3%} over **{cfg.es_horizon}-day** horizon",
        f"- kappa: {cfg.kappa:.2f}",
        f"- smoothing α: {cfg.smoothing:.2f}; rebalance band: {cfg.band:.2f}",
        f"- leverage cap: {cfg.leverage_cap:.2f} (allow_leverage={cfg.allow_leverage})",
        "",
        "## [E1] Four-configuration backtest",
        "",
        "| book | ann. return | ann. vol | Sharpe | Sortino | Calmar | max DD | turnover | Δ turnover | alarm |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in backtest.metrics.items():
        extra = backtest.incremental_turnover[name]
        alarm = backtest.turnover_alarm[name]
        lines.append(
            f"| {name} | {m.ann_return:.3%} | {m.ann_vol:.3%} | {m.sharpe:.3f} | "
            f"{m.sortino:.3f} | {m.calmar:.3f} | {m.max_drawdown:.3%} | "
            f"{m.ann_turnover:.3f} | {extra:.3f} | {alarm} |"
        )
    lines.extend(["", "## Autoportfolio vs MS-GARCH", ""])
    if comparison is None:
        lines.append("Legacy detector not supplied.")
    else:
        lines.extend(
            [
                f"- agreement: {comparison.agreement_rate:.1%} of {comparison.n_days} days",
                f"- discrepancy days: {int(comparison.discrepancies.shape[0])}",
                "",
                "### Episode lead-lag",
                "",
                comparison.episode_leaders.to_string(index=False),
                "",
                "### Overlay performance by detector",
                "",
                f"- MS-GARCH Sharpe {comparison.overlay_ms.sharpe:.3f}, max DD {comparison.overlay_ms.max_drawdown:.3%}",
                f"- legacy Sharpe {comparison.overlay_legacy.sharpe:.3f}, max DD {comparison.overlay_legacy.max_drawdown:.3%}",
                f"- conjunction Sharpe {comparison.overlay_conjunction.sharpe:.3f}, max DD {comparison.overlay_conjunction.max_drawdown:.3%}",
                "",
                "Conjunction de-risks the *regime* leg only when **both** detectors",
                "are in stress. It does not silently replace autoportfolio.",
            ]
        )
    lines.extend(
        [
            "",
            "## Sensitivity (Sharpe and max drawdown)",
            "",
            sensitivity.to_string(index=False),
            "",
            "If overlay incremental turnover exceeds 200% annualised, costs dominate.",
            "Do not shrink the band to hide that — revisit the input vol series.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
