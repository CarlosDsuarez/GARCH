"""Dislocation score: rolling percentiles, default proxies, hysteresis, no look-ahead.

Synthetic schema: ISO dates (YYYY-MM-DD), invented levels (OAS = 5.0, EBP = 0.45,
GZ spread = 2.50). No production Fed or Moody's vintages are stored here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from signals.dislocation import (
    DefaultBilinearScore,
    DislocationSignalEngine,
    SignalError,
    SignalInputs,
    apply_hysteresis,
    build_default_proxy,
    load_option_b_default_rate,
    rolling_percentile_rank,
)
from signals.schema import DislocationConfig, load_signal_config


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SIGNAL = REPO_ROOT / "config" / "signal.yaml"


def _signal_payload(tmp_path: Path, **overrides) -> dict:
    payload = {
        "percentile": {"window": 8, "min_periods": 8},
        "default_proxy": {
            "option": "A",
            "gz_spread_column": "gz_spread",
            "ebp_column": "ebp",
            "ccc_series_id": "BAMLH0A3HYC",
            "bbb_series_id": "BAMLC0A4CBBB",
            "option_b": {
                "path": str(tmp_path / "hy_default_rate.csv"),
                "date_column": "date",
                "value_column": "default_rate",
                "source_column": "source",
                "observation_date_column": "observation_date",
                "notes": "Synthetic Moody's-style trailing 12m HY default rate.",
            },
        },
        "score": {
            "function": "default_bilinear",
            "vol_weight": 0.5,
            "level_weight": 0.5,
        },
        "hysteresis": {"activate": 0.60, "deactivate": 0.30},
        "position": {
            "k": 0.10,
            "abs_cap": 1.0,
            "size_only_when_active": False,
            "min_sigma": 1.0e-8,
        },
        "episodes": [
            {"label": "COVID-19 credit crash", "start": "2020-03-01", "end": "2020-03-31"},
            {"label": "2022 QT / gilt / credit", "start": "2022-09-01", "end": "2022-11-30"},
            {"label": "SVB / regional banks", "start": "2023-03-01", "end": "2023-03-31"},
        ],
        "plot": {
            "output_directory": str(tmp_path / "signal_plots"),
            "filename": "dislocation_score.png",
            "dpi": 80,
            "figsize_width": 8.0,
            "figsize_height": 4.0,
        },
        "output": {"score_table": str(tmp_path / "dislocation_score.csv")},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


def make_signal_config(tmp_path: Path, **overrides) -> DislocationConfig:
    return DislocationConfig.model_validate(_signal_payload(tmp_path, **overrides))


def _series(values: list[float], start: str = "2020-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype="float64")


def _inputs_from_frame(frame: pd.DataFrame) -> SignalInputs:
    return SignalInputs(
        sigma_ebp=frame["sigma_ebp"],
        sigma_oas=frame["sigma_oas"],
        ebp_level=frame["ebp_level"],
        oas_level=frame["oas_level"],
        default_proxy=frame["default_proxy"],
    )


def _monotonic_panel(n: int = 40) -> pd.DataFrame:
    """Rising EBP stress, stable fundamentals — dislocation-like panel."""
    idx = pd.bdate_range("2020-01-02", periods=n)
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "sigma_ebp": 0.05 + 0.01 * t,
            "sigma_oas": 0.20 + 0.002 * t,
            "ebp_level": 0.20 + 0.02 * t,
            "oas_level": 4.0 + 0.05 * t,
            "default_proxy": np.full(n, 1.50),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Production config
# ---------------------------------------------------------------------------


def test_production_signal_config_has_750_day_window_and_hysteresis() -> None:
    config = load_signal_config(PRODUCTION_SIGNAL)
    assert config.percentile.window == 750
    assert config.hysteresis.activate == pytest.approx(0.60)
    assert config.hysteresis.deactivate == pytest.approx(0.30)
    assert config.hysteresis.deactivate < config.hysteresis.activate
    assert config.default_proxy.option == "A"
    assert {ep.label for ep in config.episodes} >= {
        "COVID-19 credit crash",
        "2022 QT / gilt / credit",
        "SVB / regional banks",
    }


def test_config_rejects_deactivate_above_activate(tmp_path: Path) -> None:
    payload = _signal_payload(tmp_path)
    payload["hysteresis"] = {"activate": 0.30, "deactivate": 0.60}
    with pytest.raises(ValidationError, match="deactivate"):
        DislocationConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# [S1] Rolling percentile — no look-ahead
# ---------------------------------------------------------------------------


def test_rolling_percentile_at_t_ignores_later_observations() -> None:
    idx = pd.bdate_range("2020-01-02", periods=7)
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 1000.0, 2000.0], index=idx)
    t = idx[4]
    rolled = rolling_percentile_rank(series, window=5, min_periods=5)
    # Window through t is {10,20,30,40,50}; 50 is the maximum → 1.0
    assert rolled.loc[t] == pytest.approx(1.0)
    # Full-sample rank of 50 is only 5/7 — the future spikes must not leak in
    full_sample = float(series.rank(pct=True).loc[t])
    assert full_sample < 0.80


def test_rolling_percentile_window_is_causal() -> None:
    idx = pd.bdate_range("2020-01-02", periods=6)
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
    rolled = rolling_percentile_rank(series, window=3, min_periods=3)
    t = idx[2]
    window = series.loc[:t].iloc[-3:]
    expected = float((window <= window.iloc[-1]).sum() / len(window))
    assert rolled.loc[t] == pytest.approx(expected)
    assert rolled.loc[idx[0]] != rolled.loc[idx[0]]  # NaN before min_periods


# ---------------------------------------------------------------------------
# Default-risk proxies A / B / C
# ---------------------------------------------------------------------------


def test_option_a_default_component_is_gz_minus_ebp() -> None:
    idx = pd.to_datetime(["2020-01-31", "2020-02-29"])
    gz = pd.Series([2.50, 2.80], index=idx)
    ebp = pd.Series([0.45, 0.40], index=idx)
    proxy = build_default_proxy("A", gz_spread=gz, ebp=ebp)
    assert proxy.iloc[0] == pytest.approx(2.05)
    assert proxy.iloc[1] == pytest.approx(2.40)


def test_option_c_quality_ratio_is_ccc_over_bbb() -> None:
    idx = pd.bdate_range("2020-01-02", periods=3)
    ccc = pd.Series([10.0, 12.0, 9.0], index=idx)
    bbb = pd.Series([2.0, 2.0, 3.0], index=idx)
    proxy = build_default_proxy("C", ccc_oas=ccc, bbb_oas=bbb)
    assert proxy.iloc[0] == pytest.approx(5.0)
    assert proxy.iloc[2] == pytest.approx(3.0)


def test_option_b_requires_source_and_respects_observation_date(tmp_path: Path) -> None:
    csv = tmp_path / "hy_default_rate.csv"
    csv.write_text(
        "date,default_rate,source,observation_date\n"
        "2020-01-31,3.50,Moody's Default Report Jan 2020,2020-02-15\n"
        "2020-02-29,3.80,Moody's Default Report Feb 2020,2020-03-16\n",
        encoding="utf-8",
    )
    config = make_signal_config(tmp_path)
    early = load_option_b_default_rate(csv, asof="2020-02-01", spec=config.default_proxy.option_b)
    late = load_option_b_default_rate(csv, asof="2020-02-15", spec=config.default_proxy.option_b)
    assert early.empty
    assert pd.Timestamp("2020-01-31") in late.index
    assert late.attrs["source"] == "Moody's Default Report Jan 2020"
    assert late.attrs["observation_date"] == "2020-02-15"


# ---------------------------------------------------------------------------
# [S3] Continuous score + interchangeable ScoringFunction
# ---------------------------------------------------------------------------


def test_default_score_extremes_are_plus_minus_one() -> None:
    fn = DefaultBilinearScore(vol_weight=0.5, level_weight=0.5)
    assert fn(1.0, 1.0, 0.0) == pytest.approx(1.0)
    assert fn(0.0, 0.0, 1.0) == pytest.approx(-1.0)
    assert fn(0.5, 0.5, 0.5) == pytest.approx(0.0)


def test_engine_uses_injected_scoring_function(tmp_path: Path) -> None:
    panel = _monotonic_panel(20)
    config = make_signal_config(tmp_path)

    class _Fixed:
        def __call__(self, p_vol_ebp: float, p_level_ebp: float, p_fund: float) -> float:
            return 0.42

    engine = DislocationSignalEngine(_inputs_from_frame(panel), config, scoring=_Fixed())
    t = panel.index[-1]
    assert engine.compute_score(t) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# [S4] Hysteresis
# ---------------------------------------------------------------------------


def test_hysteresis_activates_above_060_and_holds_until_030() -> None:
    idx = pd.bdate_range("2020-01-02", periods=5)
    scores = pd.Series([0.20, 0.70, 0.45, 0.25, 0.65], index=idx)
    active = apply_hysteresis(scores, activate=0.60, deactivate=0.30)
    assert list(active.astype(int)) == [0, 1, 1, 0, 1]


# ---------------------------------------------------------------------------
# [S5] Position sizing
# ---------------------------------------------------------------------------


def test_position_weight_is_k_score_over_sigma_and_capped(tmp_path: Path) -> None:
    panel = _monotonic_panel(20)
    config = make_signal_config(tmp_path, position={"k": 0.10, "abs_cap": 0.25})

    class _Half:
        def __call__(self, p_vol_ebp: float, p_level_ebp: float, p_fund: float) -> float:
            return 0.50

    engine = DislocationSignalEngine(_inputs_from_frame(panel), config, scoring=_Half())
    t = panel.index[-1]
    sigma = float(panel.loc[t, "sigma_oas"])
    raw = 0.10 * 0.50 / sigma
    weight = engine.get_position_weight(t)
    assert weight == pytest.approx(min(raw, 0.25))
    assert abs(weight) <= 0.25 + 1e-12


def test_near_zero_forecast_vol_fails_loud(tmp_path: Path) -> None:
    panel = _monotonic_panel(20)
    panel["sigma_oas"] = 0.0
    config = make_signal_config(tmp_path)
    engine = DislocationSignalEngine(_inputs_from_frame(panel), config)
    with pytest.raises(SignalError, match="sigma"):
        engine.get_position_weight(panel.index[-1])


# ---------------------------------------------------------------------------
# Engine API + explain
# ---------------------------------------------------------------------------


def test_explain_returns_full_decomposition(tmp_path: Path) -> None:
    panel = _monotonic_panel(20)
    config = make_signal_config(tmp_path)
    engine = DislocationSignalEngine(_inputs_from_frame(panel), config)
    t = panel.index[-1]
    expl = engine.explain(t)
    assert expl.date == pd.Timestamp(t)
    assert -1.0 <= expl.score <= 1.0
    assert expl.p_vol_ebp == pytest.approx(1.0)
    assert expl.p_level_ebp == pytest.approx(1.0)
    assert 0.0 <= expl.p_fund <= 1.0
    assert expl.sigma_oas == pytest.approx(float(panel.loc[t, "sigma_oas"]))
    assert expl.ebp_level == pytest.approx(float(panel.loc[t, "ebp_level"]))
    assert expl.oas_level == pytest.approx(float(panel.loc[t, "oas_level"]))
    assert expl.default_proxy == pytest.approx(float(panel.loc[t, "default_proxy"]))
    assert expl.weight == pytest.approx(engine.get_position_weight(t))
    assert expl.n_obs_used >= config.percentile.min_periods


def test_history_matches_compute_score_at_each_date(tmp_path: Path) -> None:
    panel = _monotonic_panel(20)
    engine = DislocationSignalEngine(_inputs_from_frame(panel), make_signal_config(tmp_path))
    hist = engine.history()
    for ts in hist.index:
        assert hist.loc[ts, "score"] == pytest.approx(engine.compute_score(ts))


def test_plot_history_writes_png_with_episode_overlay(tmp_path: Path) -> None:
    panel = _monotonic_panel(20)
    config = make_signal_config(tmp_path)
    engine = DislocationSignalEngine(_inputs_from_frame(panel), config)
    path = engine.plot_history()
    assert path.exists()
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Mandatory look-ahead test
# ---------------------------------------------------------------------------


@pytest.mark.blocking
def test_score_at_t_identical_on_dataset_truncated_at_t(tmp_path: Path) -> None:
    panel = _monotonic_panel(30)
    # A future shock that would change full-sample ranks if leaked
    panel.loc[panel.index[-3]:, "sigma_ebp"] = 50.0
    panel.loc[panel.index[-3]:, "ebp_level"] = 20.0
    t = panel.index[15]
    config = make_signal_config(tmp_path)
    full = DislocationSignalEngine(_inputs_from_frame(panel), config)
    truncated = DislocationSignalEngine(_inputs_from_frame(panel.loc[:t]), config)
    assert full.compute_score(t) == pytest.approx(truncated.compute_score(t), abs=1e-12)
    assert full.explain(t).active == truncated.explain(t).active
    assert full.get_position_weight(t) == pytest.approx(
        truncated.get_position_weight(t), abs=1e-12
    )


def test_compute_score_rejects_future_peek(tmp_path: Path) -> None:
    panel = _monotonic_panel(20)
    engine = DislocationSignalEngine(_inputs_from_frame(panel), make_signal_config(tmp_path))
    with pytest.raises(SignalError, match="beyond"):
        engine.compute_score(panel.index[-1] + pd.Timedelta(days=30))
