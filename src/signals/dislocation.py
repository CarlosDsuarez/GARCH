"""Translate conditional credit volatilities into a continuous dislocation score.

Economic target
---------------
Identify the state where credit is in acute stress *and* that stress is the
price of risk (EBP), not expected default. That is a liquidity-provision
setup: dealers are forced sellers on VaR, not solvency. The opposite state
— stress with confirmed fundamental deterioration — is a reduce/avoid signal.

Score ([S3])
------------
Percentiles are causal rolling ranks ([S1], [C1]). Then

.. math::

    \\mathrm{stress}_t = w_v\\, p^{\\sigma}_{\\mathrm{EBP},t}
                       + w_\\ell\\, p^{\\mathrm{level}}_{\\mathrm{EBP},t}

    \\mathrm{score}_t
    = \\mathrm{stress}_t (1 - p^{\\mathrm{fund}}_t)
    - (1 - \\mathrm{stress}_t)\\, p^{\\mathrm{fund}}_t

so :math:`\\mathrm{score}\\in[-1,+1]`. Near :math:`+1` is pure dislocation
(buy credit); near :math:`-1` is fundamental deterioration (cut exposure).

``ScoringFunction`` is a protocol so alternative formulas can be injected
without editing the engine.

Inputs must already be point-in-time: each series is indexed by the first
date the value was knowable. The engine never reads past ``date``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from signals.schema import DislocationConfig, OptionBConfig, load_signal_config

logger = logging.getLogger(__name__)

__all__ = [
    "DefaultBilinearScore",
    "DislocationSignalEngine",
    "ScoreExplanation",
    "ScoringFunction",
    "SignalError",
    "SignalInputs",
    "apply_hysteresis",
    "build_default_proxy",
    "load_option_b_default_rate",
    "rolling_percentile_rank",
]


class SignalError(ValueError):
    """Mandatory signal diagnostic failed or the date is not usable ([C7])."""


class ScoringFunction(Protocol):
    def __call__(self, p_vol_ebp: float, p_level_ebp: float, p_fund: float) -> float:
        """Map causal percentiles in ``[0, 1]`` to a score in ``[-1, +1]``."""


@dataclass(frozen=True)
class DefaultBilinearScore:
    vol_weight: float = 0.5
    level_weight: float = 0.5

    def __call__(self, p_vol_ebp: float, p_level_ebp: float, p_fund: float) -> float:
        stress = self.vol_weight * p_vol_ebp + self.level_weight * p_level_ebp
        fund = p_fund
        score = stress * (1.0 - fund) - (1.0 - stress) * fund
        return float(np.clip(score, -1.0, 1.0))


@dataclass(frozen=True)
class SignalInputs:
    sigma_ebp: pd.Series
    sigma_oas: pd.Series
    ebp_level: pd.Series
    oas_level: pd.Series
    default_proxy: pd.Series


@dataclass(frozen=True)
class ScoreExplanation:
    date: pd.Timestamp
    score: float
    stress: float
    fundamental: float
    p_vol_ebp: float
    p_level_ebp: float
    p_fund: float
    sigma_ebp: float
    sigma_oas: float
    ebp_level: float
    oas_level: float
    default_proxy: float
    default_proxy_option: str
    active: bool
    weight: float
    window: int
    n_obs_used: int


def rolling_percentile_rank(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Causal rolling percentile of the *current* value inside the window.

    At date :math:`t` the window is :math:`\\{y_{t-W+1},\\ldots,y_t\\}`.
    The rank is :math:`n^{-1}\\#\\{y\\le y_t\\}`. Observations after :math:`t`
    are never consulted ([C1], [S1]).
    """
    if window < 2:
        raise ValueError("rolling percentile window must be >= 2")
    min_p = window if min_periods is None else int(min_periods)
    values = series.astype("float64")

    def _last_percentile(arr: np.ndarray) -> float:
        current = arr[-1]
        if np.isnan(current):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if valid.size < min_p:
            return np.nan
        return float(np.sum(valid <= current) / valid.size)

    ranked = values.rolling(window=window, min_periods=min_p).apply(
        _last_percentile, raw=True
    )
    ranked.name = series.name
    return ranked


def build_default_proxy(
    option: str,
    *,
    gz_spread: pd.Series | None = None,
    ebp: pd.Series | None = None,
    ccc_oas: pd.Series | None = None,
    bbb_oas: pd.Series | None = None,
) -> pd.Series:
    """Construct the fundamental-risk proxy selected in configuration."""
    choice = option.upper()
    if choice == "A":
        if gz_spread is None or ebp is None:
            raise SignalError("option A requires gz_spread and ebp")
        aligned = pd.concat({"gz": gz_spread, "ebp": ebp}, axis=1).dropna()
        proxy = (aligned["gz"] - aligned["ebp"]).rename("default_proxy")
        logger.info("default proxy A = GZ spread − EBP; n=%s", int(proxy.shape[0]))
        return proxy
    if choice == "C":
        if ccc_oas is None or bbb_oas is None:
            raise SignalError("option C requires ccc_oas and bbb_oas")
        aligned = pd.concat({"ccc": ccc_oas, "bbb": bbb_oas}, axis=1).dropna()
        if (aligned["bbb"] <= 0).any():
            raise SignalError("BBB OAS must be strictly positive for the quality ratio")
        proxy = (aligned["ccc"] / aligned["bbb"]).rename("default_proxy")
        logger.info("default proxy C = CCC/BBB OAS; n=%s", int(proxy.shape[0]))
        return proxy
    if choice == "B":
        raise SignalError(
            "option B is a dated manual ingest; use load_option_b_default_rate"
        )
    raise SignalError(f"unknown default-proxy option {option}")


def load_option_b_default_rate(
    path: str | Path,
    *,
    asof: pd.Timestamp | str,
    spec: OptionBConfig,
) -> pd.Series:
    """Load the trailing-12m HY default rate, keeping only published rows.

    A row is available at ``asof`` only when ``observation_date <= asof``.
    ``source`` and ``observation_date`` of the latest available report are
    stored on ``Series.attrs`` for the audit trail.
    """
    frame = pd.read_csv(path)
    required = [
        spec.date_column,
        spec.value_column,
        spec.source_column,
        spec.observation_date_column,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SignalError(
            f"option B CSV missing {missing}; need reference date, rate, "
            "source, and observation_date (report publication date)"
        )
    frame[spec.date_column] = pd.to_datetime(frame[spec.date_column])
    frame[spec.observation_date_column] = pd.to_datetime(frame[spec.observation_date_column])
    asof_ts = pd.Timestamp(asof).normalize()
    keep = frame.loc[frame[spec.observation_date_column] <= asof_ts].copy()
    keep = keep.sort_values(spec.observation_date_column)
    series = (
        keep.set_index(spec.date_column)[spec.value_column]
        .astype("float64")
        .sort_index()
        .rename("default_proxy")
    )
    if not keep.empty:
        last = keep.iloc[-1]
        series.attrs["source"] = str(last[spec.source_column])
        series.attrs["observation_date"] = pd.Timestamp(
            last[spec.observation_date_column]
        ).strftime("%Y-%m-%d")
        logger.info(
            "option B as-of %s n=%s last_source=%s observation_date=%s",
            asof_ts.date(),
            int(series.shape[0]),
            series.attrs["source"],
            series.attrs["observation_date"],
        )
    else:
        logger.info("option B as-of %s: no published observations", asof_ts.date())
    return series


def apply_hysteresis(
    scores: pd.Series,
    *,
    activate: float,
    deactivate: float,
) -> pd.Series:
    """Forward-only on/off path. Future scores never change the state at t."""
    if deactivate >= activate:
        raise SignalError("hysteresis deactivate must be strictly below activate")
    state = False
    flags: list[bool] = []
    for value in scores.to_numpy(dtype=float):
        if np.isnan(value):
            flags.append(state)
            continue
        if not state and value > activate:
            state = True
        elif state and value < deactivate:
            state = False
        flags.append(state)
    return pd.Series(flags, index=scores.index, dtype=bool, name="active")


def _as_series(series: pd.Series, name: str) -> pd.Series:
    out = series.copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    out = out.sort_index().astype("float64")
    out.name = name
    if out.index.has_duplicates:
        raise SignalError(f"duplicate dates in {name}")
    return out


class DislocationSignalEngine:
    """Causal dislocation score, hysteresis flag, and vol-targeted weight."""

    def __init__(
        self,
        inputs: SignalInputs,
        config: DislocationConfig,
        *,
        scoring: ScoringFunction | None = None,
    ) -> None:
        self.config = config
        self.inputs = SignalInputs(
            sigma_ebp=_as_series(inputs.sigma_ebp, "sigma_ebp"),
            sigma_oas=_as_series(inputs.sigma_oas, "sigma_oas"),
            ebp_level=_as_series(inputs.ebp_level, "ebp_level"),
            oas_level=_as_series(inputs.oas_level, "oas_level"),
            default_proxy=_as_series(inputs.default_proxy, "default_proxy"),
        )
        self.scoring: ScoringFunction = scoring or DefaultBilinearScore(
            vol_weight=config.score.vol_weight,
            level_weight=config.score.level_weight,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        inputs: SignalInputs,
        **kwargs: object,
    ) -> DislocationSignalEngine:
        return cls(inputs, load_signal_config(path), **kwargs)  # type: ignore[arg-type]

    def compute_score(self, date: pd.Timestamp | str) -> float:
        return self.explain(date).score

    def get_position_weight(self, date: pd.Timestamp | str) -> float:
        return self.explain(date).weight

    def explain(self, date: pd.Timestamp | str) -> ScoreExplanation:
        ts = pd.Timestamp(date).normalize()
        last = self._last_available()
        if ts > last:
            raise SignalError(f"date {ts.date()} is beyond the available sample")
        frame = self._aligned_through(ts)
        ranked = self._rank_frame(frame)
        if ranked.empty:
            raise SignalError(
                f"insufficient history for rolling percentiles at {ts.date()} "
                f"(need {self.config.percentile.min_periods} observations)"
            )
        row = ranked.iloc[-1]
        scores = ranked["score"]
        active_path = apply_hysteresis(
            scores,
            activate=self.config.hysteresis.activate,
            deactivate=self.config.hysteresis.deactivate,
        )
        sigma = float(frame["sigma_oas"].iloc[-1])
        score = float(row["score"])
        weight = self._weight(score, sigma, bool(active_path.iloc[-1]))
        n_obs = int(min(len(frame), self.config.percentile.window))
        logger.info(
            "dislocation t=%s score=%.4f stress=%.4f fund=%.4f active=%s w=%.4f",
            ts.date(),
            score,
            float(row["stress"]),
            float(row["fundamental"]),
            bool(active_path.iloc[-1]),
            weight,
        )
        return ScoreExplanation(
            date=ts,
            score=score,
            stress=float(row["stress"]),
            fundamental=float(row["fundamental"]),
            p_vol_ebp=float(row["p_vol_ebp"]),
            p_level_ebp=float(row["p_level_ebp"]),
            p_fund=float(row["p_fund"]),
            sigma_ebp=float(frame["sigma_ebp"].iloc[-1]),
            sigma_oas=sigma,
            ebp_level=float(frame["ebp_level"].iloc[-1]),
            oas_level=float(frame["oas_level"].iloc[-1]),
            default_proxy=float(frame["default_proxy"].iloc[-1]),
            default_proxy_option=self.config.default_proxy.option,
            active=bool(active_path.iloc[-1]),
            weight=weight,
            window=self.config.percentile.window,
            n_obs_used=n_obs,
        )

    def history(self) -> pd.DataFrame:
        last = self._last_available()
        frame = self._aligned_through(last)
        ranked = self._rank_frame(frame)
        if ranked.empty:
            raise SignalError("insufficient history to build a score series")
        active = apply_hysteresis(
            ranked["score"],
            activate=self.config.hysteresis.activate,
            deactivate=self.config.hysteresis.deactivate,
        )
        weights = [
            self._weight(
                float(ranked.loc[ts, "score"]),
                float(frame.loc[ts, "sigma_oas"]),
                bool(active.loc[ts]),
            )
            for ts in ranked.index
        ]
        out = ranked.copy()
        out["active"] = active.astype(bool)
        out["weight"] = weights
        out["sigma_ebp"] = frame["sigma_ebp"].reindex(out.index)
        out["sigma_oas"] = frame["sigma_oas"].reindex(out.index)
        out["ebp_level"] = frame["ebp_level"].reindex(out.index)
        out["oas_level"] = frame["oas_level"].reindex(out.index)
        out["default_proxy"] = frame["default_proxy"].reindex(out.index)
        return out

    def plot_history(self, path: str | Path | None = None) -> Path:
        hist = self.history()
        plot_cfg = self.config.plot
        out_dir = Path(plot_cfg.output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = Path(path) if path is not None else out_dir / plot_cfg.filename
        os.environ.setdefault("MPLCONFIGDIR", str(out_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(plot_cfg.figsize_width, plot_cfg.figsize_height))
        ax.plot(hist.index, hist["score"].to_numpy(), color="tab:blue", label="dislocation score")
        ax.axhline(
            self.config.hysteresis.activate,
            color="tab:green",
            ls="--",
            lw=1.0,
            label=f"activate {self.config.hysteresis.activate:.2f}",
        )
        ax.axhline(
            self.config.hysteresis.deactivate,
            color="tab:orange",
            ls="--",
            lw=1.0,
            label=f"deactivate {self.config.hysteresis.deactivate:.2f}",
        )
        ax.axhline(0.0, color="0.5", lw=0.8)
        x_min, x_max = hist.index.min(), hist.index.max()
        for episode in self.config.episodes:
            start = pd.Timestamp(episode.start)
            end = pd.Timestamp(episode.end)
            if end < x_min or start > x_max:
                continue
            ax.axvspan(
                max(start, x_min),
                min(end, x_max),
                color="tab:red",
                alpha=0.12,
                label=episode.label,
            )
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("score")
        ax.set_title("Credit dislocation score (EBP stress vs fundamental default)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        handles, labels = ax.get_legend_handles_labels()
        seen: set[str] = set()
        uniq = [(h, lab) for h, lab in zip(handles, labels) if lab not in seen and not seen.add(lab)]
        if uniq:
            ax.legend(*zip(*uniq), loc="upper left", fontsize=8)
        fig.tight_layout()
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=plot_cfg.dpi)
        plt.close(fig)
        table_path = Path(self.config.output.score_table)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(table_path)
        logger.info("wrote dislocation plot %s and table %s", dest, table_path)
        return dest

    def _last_available(self) -> pd.Timestamp:
        ends = [
            series.dropna().index.max()
            for series in (
                self.inputs.sigma_ebp,
                self.inputs.sigma_oas,
                self.inputs.ebp_level,
                self.inputs.oas_level,
                self.inputs.default_proxy,
            )
            if series.notna().any()
        ]
        if not ends:
            raise SignalError("all signal inputs are empty")
        return max(ends)

    def _aligned_through(self, date: pd.Timestamp) -> pd.DataFrame:
        columns = {
            "sigma_ebp": self.inputs.sigma_ebp,
            "sigma_oas": self.inputs.sigma_oas,
            "ebp_level": self.inputs.ebp_level,
            "oas_level": self.inputs.oas_level,
            "default_proxy": self.inputs.default_proxy,
        }
        cut: dict[str, pd.Series] = {}
        calendar: pd.DatetimeIndex | None = None
        for name, series in columns.items():
            available = series.loc[:date].dropna()
            if available.empty:
                raise SignalError(f"no {name} observations on or before {date.date()}")
            cut[name] = available
            calendar = available.index if calendar is None else calendar.union(available.index)
        assert calendar is not None
        calendar = calendar.sort_values()
        frame = pd.DataFrame(index=calendar)
        for name, series in cut.items():
            frame[name] = series.reindex(calendar).ffill()
        cleaned = frame.dropna()
        if cleaned.empty:
            raise SignalError(f"no overlapping observations on or before {date.date()}")
        return cleaned

    def _rank_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        window = self.config.percentile.window
        min_p = self.config.percentile.min_periods
        ranked = pd.DataFrame(
            {
                "p_vol_ebp": rolling_percentile_rank(frame["sigma_ebp"], window, min_p),
                "p_level_ebp": rolling_percentile_rank(frame["ebp_level"], window, min_p),
                "p_fund": rolling_percentile_rank(frame["default_proxy"], window, min_p),
            },
            index=frame.index,
        ).dropna()
        if ranked.empty:
            return ranked
        stresses: list[float] = []
        funds: list[float] = []
        scores: list[float] = []
        for p_vol, p_level, p_fund in ranked.to_numpy():
            stress = (
                self.config.score.vol_weight * float(p_vol)
                + self.config.score.level_weight * float(p_level)
            )
            score = float(self.scoring(float(p_vol), float(p_level), float(p_fund)))
            stresses.append(stress)
            funds.append(float(p_fund))
            scores.append(score)
        ranked = ranked.copy()
        ranked["stress"] = stresses
        ranked["fundamental"] = funds
        ranked["score"] = scores
        return ranked

    def _weight(self, score: float, sigma: float, active: bool) -> float:
        spec = self.config.position
        if sigma < spec.min_sigma:
            raise SignalError(
                f"sigma_oas_forecast={sigma} is below min_sigma={spec.min_sigma}"
            )
        if spec.size_only_when_active and not active:
            return 0.0
        raw = spec.k * score / sigma
        return float(np.clip(raw, -spec.abs_cap, spec.abs_cap))
