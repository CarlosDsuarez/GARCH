"""[T2] Look-ahead bias. The most important tests in the repository.

A pipeline that uses y_{t+1:T} at date t still produces numbers that look like
trading signals. Truncation and future-permutation are the detectors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helpers import (
    TRUE_GJR,
    causal_fhs_var,
    dislocation_panel,
    gjr_filter,
    permute_future,
    permute_future_frame,
    simulate_gjr_garch,
)
from models.regime import RegimeDetector, realized_log_variance, rolling_first_pc
from risk.risk_overlay import RiskOverlay
from risk.schema import OverlayConfig
from signals.dislocation import DislocationSignalEngine, SignalInputs, rolling_percentile_rank
from test_dislocation import make_signal_config
from test_regime import make_regime_config
from test_risk_overlay import _overlay_payload, _states


N_DATES = 20
ATOL = 1e-12


def _inputs(frame: pd.DataFrame) -> SignalInputs:
    return SignalInputs(
        sigma_ebp=frame["sigma_ebp"],
        sigma_oas=frame["sigma_oas"],
        ebp_level=frame["ebp_level"],
        oas_level=frame["oas_level"],
        default_proxy=frame["default_proxy"],
    )


def _eval_dates(index: pd.DatetimeIndex, n: int = N_DATES) -> list[pd.Timestamp]:
    lo = max(40, int(0.25 * len(index)))
    hi = len(index) - 3
    locs = np.linspace(lo, hi, n, dtype=int)
    return [pd.Timestamp(index[int(i)]) for i in locs]


@pytest.mark.blocking
def test_t2_1_truncation_sigma_score_var_regime_match_on_20_dates(tmp_path: Path) -> None:
    """Full dataset vs dataset truncated at t: outputs at t must be identical."""
    returns = simulate_gjr_garch(120, seed=8)
    panel = dislocation_panel(120)
    panel.index = returns.index
    params = {k: TRUE_GJR[k] for k in ("omega", "alpha", "gamma", "beta")}
    score_cfg = make_signal_config(tmp_path)

    dates = _eval_dates(returns.index)
    assert len(dates) >= N_DATES
    for t in dates:
        truncated_r = returns.loc[:t]
        truncated_panel = panel.loc[:t]

        sigma_full = gjr_filter(returns, **params).loc[t]
        sigma_cut = gjr_filter(truncated_r, **params).loc[t]
        assert sigma_full == pytest.approx(sigma_cut, abs=ATOL)

        full_engine = DislocationSignalEngine(_inputs(panel), score_cfg)
        cut_engine = DislocationSignalEngine(_inputs(truncated_panel), score_cfg)
        assert full_engine.compute_score(t) == pytest.approx(cut_engine.compute_score(t), abs=ATOL)

        var_full = causal_fhs_var(returns.loc[:t], **params)
        var_cut = causal_fhs_var(truncated_r, **params)
        assert var_full == pytest.approx(var_cut, abs=ATOL)

        rv_full = realized_log_variance(returns, 5)
        rv_cut = realized_log_variance(truncated_r, 5)
        assert rv_full.loc[t] == pytest.approx(rv_cut.loc[t], abs=ATOL)

        p_full = rolling_percentile_rank(rv_full, window=20, min_periods=20)
        p_cut = rolling_percentile_rank(rv_cut, window=20, min_periods=20)
        assert p_full.loc[t] == pytest.approx(p_cut.loc[t], abs=ATOL)


@pytest.mark.blocking
def test_t2_1_truncation_rolling_pc1_on_20_dates() -> None:
    idx = pd.bdate_range("2018-01-02", periods=80)
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(
        {
            "VIXCLS": 15.0 + np.cumsum(rng.normal(0, 0.4, size=80)),
            "HY": 4.0 + np.cumsum(rng.normal(0, 0.05, size=80)),
        },
        index=idx,
    )
    for t in _eval_dates(idx):
        full = rolling_first_pc(panel, window=20, min_periods=20)
        cut = rolling_first_pc(panel.loc[:t], window=20, min_periods=20)
        assert full.loc[t] == pytest.approx(cut.loc[t], abs=1e-10)


@pytest.mark.blocking
def test_t2_2_permuting_the_future_does_not_change_outputs_at_t(tmp_path: Path) -> None:
    returns = simulate_gjr_garch(100, seed=9)
    panel = dislocation_panel(100)
    panel.index = returns.index
    params = {k: TRUE_GJR[k] for k in ("omega", "alpha", "gamma", "beta")}
    score_cfg = make_signal_config(tmp_path)
    t = pd.Timestamp(returns.index[60])

    shuffled_r = permute_future(returns, t, seed=99)
    shuffled_panel = permute_future_frame(panel, t, seed=99)

    assert gjr_filter(returns, **params).loc[t] == pytest.approx(
        gjr_filter(shuffled_r, **params).loc[t], abs=ATOL
    )
    assert DislocationSignalEngine(_inputs(panel), score_cfg).compute_score(t) == pytest.approx(
        DislocationSignalEngine(_inputs(shuffled_panel), score_cfg).compute_score(t),
        abs=ATOL,
    )
    assert causal_fhs_var(returns.loc[:t], **params) == pytest.approx(
        causal_fhs_var(shuffled_r.loc[:t], **params), abs=ATOL
    )
    assert realized_log_variance(returns, 5).loc[t] == pytest.approx(
        realized_log_variance(shuffled_r, 5).loc[t], abs=ATOL
    )


@pytest.mark.blocking
def test_t2_2_overlay_multiplier_ignores_future_sigma(tmp_path: Path) -> None:
    idx = pd.bdate_range("2019-01-02", periods=40)
    sigma = pd.Series(0.12 + 0.001 * np.arange(40), index=idx, name="sigma_ann")
    es = pd.Series(0.02, index=idx, name="es")
    states = _states(idx, labels=["calm"] * 25 + ["stress"] * 15)
    cfg = OverlayConfig.model_validate(_overlay_payload(tmp_path))
    t = idx[20]
    overlay = RiskOverlay(cfg).build(sigma, es, states)
    shuffled = permute_future(sigma, t, seed=4)
    overlay_shuffled = RiskOverlay(cfg).build(shuffled, es, states)
    assert overlay.compute_multiplier(t) == pytest.approx(
        overlay_shuffled.compute_multiplier(t), abs=ATOL
    )


@pytest.mark.blocking
def test_t2_3_operational_code_never_requests_smoothed_probabilities() -> None:
    root = Path(__file__).resolve().parents[1] / "src"
    operational = [
        root / "risk" / "risk_overlay.py",
        root / "signals" / "dislocation.py",
        root / "backtest" / "signal_backtest.py",
        root / "risk" / "fhs.py",
    ]
    for path in operational:
        text = path.read_text(encoding="utf-8")
        assert "mode='smoothed'" not in text
        assert 'mode="smoothed"' not in text


@pytest.mark.blocking
def test_t2_3_get_regime_probability_defaults_to_filtered_and_logs_smoothed(
    tmp_path: Path, caplog
) -> None:
    idx = pd.bdate_range("2018-01-02", periods=60)
    p = pd.Series(np.linspace(0.1, 0.9, 60), index=idx)
    detector = RegimeDetector.from_probabilities(p, make_regime_config(tmp_path))
    t = idx[30]
    assert detector.get_regime_probability(t) == pytest.approx(float(p.loc[t]))
    assert detector.get_regime_probability(t, mode="filtered") == pytest.approx(float(p.loc[t]))
    with caplog.at_level("WARNING"):
        detector.get_regime_probability(t, mode="smoothed")
    text = caplog.text.lower()
    assert "look-ahead" in text or "lookahead" in text or "smoothed" in text
