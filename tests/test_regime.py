"""Markov-switching regime detection: Option A, filtered vs smoothed, 5-layer protocol.

Synthetic schema: ISO dates (YYYY-MM-DD), invented returns (~0.005 calm / ~0.03
stress) and invented exogenous levels. No live OAS/NFCI vintages are stored here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from models.regime import (
    MSGARCHBackend,
    RegimeDetector,
    RegimeError,
    apply_anti_chattering,
    assert_regimes_economically_valid,
    causal_exogenous_confirm,
    expected_durations,
    realized_log_variance,
    rolling_first_pc,
    unconditional_probabilities,
)
from models.schema import RegimeConfig, load_regime_config


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_REGIME = REPO_ROOT / "config" / "regime.yaml"


def _idx(n: int, start: str = "2016-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _regime_payload(tmp_path: Path, **overrides) -> dict:
    payload: dict = {
        "input": {
            "measure": "log_rv",
            "rv_window": 5,
            "pca_window": 20,
            "pca_min_periods": 20,
            "min_observations": 40,
        },
        "fit": {
            "k_regimes": 2,
            "switching_variance": True,
            "trend": "c",
            "search_reps": 8,
            "maxiter": 300,
            "seed": 7,
            "backend": "markov_log_variance",
        },
        "hysteresis": {"enter": 0.70, "exit": 0.30},
        "dwell": {"min_days": 5},
        "confirmation": {"consecutive_days": 2},
        "exogenous": {
            "percentile": 80.0,
            "window": 10,
            "min_periods": 10,
            "partial_derisk_fraction": 0.5,
        },
        "k3": {
            "min_unconditional": 0.10,
            "min_expected_duration_days": 10.0,
        },
        "transitions": {
            "alarm_per_year": 8.0,
            "round_trip_cost_bps": 10.0,
            "periods_per_year": 252,
        },
        "plot": {
            "output_directory": str(tmp_path / "regime_plots"),
            "filename": "regime_three_panel.png",
            "dpi": 80,
            "figsize_width": 10.0,
            "figsize_height": 8.0,
        },
        "output": {
            "report_markdown": str(tmp_path / "regime_report.md"),
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


def make_regime_config(tmp_path: Path, **overrides) -> RegimeConfig:
    return RegimeConfig.model_validate(_regime_payload(tmp_path, **overrides))


def _always_confirmed(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(True, index=index, dtype=bool)


# ---------------------------------------------------------------------------
# Production YAML / schema
# ---------------------------------------------------------------------------


def test_production_regime_config_is_option_a_two_state() -> None:
    config = load_regime_config(PRODUCTION_REGIME)
    assert config.fit.k_regimes == 2
    assert config.fit.backend == "markov_log_variance"
    assert config.fit.switching_variance is True
    assert config.input.measure == "log_rv"
    assert config.input.rv_window == 5
    assert config.hysteresis.enter == pytest.approx(0.70)
    assert config.hysteresis.exit == pytest.approx(0.30)
    assert config.dwell.min_days == 5
    assert config.confirmation.consecutive_days == 2
    assert config.exogenous.partial_derisk_fraction == pytest.approx(0.5)
    assert config.transitions.alarm_per_year == pytest.approx(8.0)
    assert config.fit.seed is not None
    assert config.k3.min_unconditional == pytest.approx(0.10)
    assert config.k3.min_expected_duration_days == pytest.approx(10.0)


def test_config_rejects_a_single_hysteresis_threshold(tmp_path: Path) -> None:
    payload = _regime_payload(tmp_path)
    payload["hysteresis"]["enter"] = 0.50
    payload["hysteresis"]["exit"] = 0.50
    with pytest.raises(ValidationError, match="enter"):
        RegimeConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Input measure / [C1]
# ---------------------------------------------------------------------------


def test_realized_log_variance_is_causal() -> None:
    idx = _idx(8)
    r = pd.Series([0.01, 0.02, -0.01, 0.0, 0.03, 0.04, -0.02, 0.01], index=idx)
    log_rv = realized_log_variance(r, window=5)
    assert np.isnan(log_rv.iloc[3])
    expected = float(np.log((r.iloc[:5] ** 2).sum()))
    assert log_rv.iloc[4] == pytest.approx(expected)
    r_future = r.copy()
    r_future.iloc[-1] = 0.99
    log_rv2 = realized_log_variance(r_future, window=5)
    assert log_rv.iloc[4] == pytest.approx(float(log_rv2.iloc[4]))


def test_rolling_pc1_does_not_use_future_observations() -> None:
    idx = _idx(60)
    a = np.linspace(0.0, 1.0, 60)
    b = np.zeros(60)
    b[40:] = np.linspace(0.0, 80.0, 20)
    panel = pd.DataFrame({"sigma_oas": a, "VIX": b}, index=idx)
    pc = rolling_first_pc(panel, window=15, min_periods=15)
    t = idx[25]
    assert np.isfinite(pc.loc[t])
    panel2 = panel.copy()
    panel2.iloc[45:, 1] = 1_000.0
    pc2 = rolling_first_pc(panel2, window=15, min_periods=15)
    assert pc.loc[t] == pytest.approx(float(pc2.loc[t]))


# ---------------------------------------------------------------------------
# Duration / K=3 validity
# ---------------------------------------------------------------------------


def test_expected_duration_is_one_over_one_minus_stay_prob() -> None:
    # Rows = from, columns = to: p_ij = P(s_t=j | s_{t-1}=i).
    transition = np.array([[0.95, 0.05], [0.10, 0.90]])
    durations = expected_durations(transition)
    assert durations[0] == pytest.approx(1.0 / 0.05)
    assert durations[1] == pytest.approx(1.0 / 0.10)
    pi = unconditional_probabilities(transition)
    assert pi.sum() == pytest.approx(1.0)
    assert pi[0] == pytest.approx((1.0 - 0.90) / (2.0 - 0.95 - 0.90))


def test_k3_rejects_a_tiny_or_ephemeral_third_regime() -> None:
    with pytest.raises(RegimeError, match="10%"):
        assert_regimes_economically_valid(
            unconditional=np.array([0.50, 0.48, 0.02]),
            durations=np.array([20.0, 18.0, 15.0]),
            min_unconditional=0.10,
            min_duration_days=10.0,
        )
    with pytest.raises(RegimeError, match="10 days"):
        assert_regimes_economically_valid(
            unconditional=np.array([0.40, 0.40, 0.20]),
            durations=np.array([20.0, 18.0, 3.0]),
            min_unconditional=0.10,
            min_duration_days=10.0,
        )


# ---------------------------------------------------------------------------
# Anti-chattering layers (injected probabilities — no estimator luck)
# ---------------------------------------------------------------------------


def test_hysteresis_holds_in_the_grey_band() -> None:
    idx = _idx(20)
    p = pd.Series([0.20] * 4 + [0.50, 0.60, 0.40, 0.55] * 4, index=idx)
    path = apply_anti_chattering(
        p,
        enter=0.70,
        exit=0.30,
        dwell=5,
        confirm_days=2,
        exogenous_confirmed=_always_confirmed(idx),
        partial_fraction=0.5,
    )
    assert (path["label"] == "calm").all()
    assert (path["derisk_fraction"] == 0.0).all()


def test_confirmation_requires_n_consecutive_days_above_enter() -> None:
    idx = _idx(12)
    values = [0.20] * 4 + [0.80] + [0.20] * 3 + [0.80, 0.80] + [0.50, 0.50]
    p = pd.Series(values, index=idx)
    path = apply_anti_chattering(
        p,
        enter=0.70,
        exit=0.30,
        dwell=1,
        confirm_days=2,
        exogenous_confirmed=_always_confirmed(idx),
        partial_fraction=0.5,
    )
    assert path["label"].iloc[4] == "calm"
    assert path["label"].iloc[8] == "calm"
    assert path["label"].iloc[9] == "stress"
    assert path["label"].iloc[10] == "stress"


def test_dwell_blocks_immediate_reversal_after_a_declared_stress() -> None:
    idx = _idx(16)
    values = [0.20] * 4 + [0.85, 0.85] + [0.05] * 10
    p = pd.Series(values, index=idx)
    path = apply_anti_chattering(
        p,
        enter=0.70,
        exit=0.30,
        dwell=5,
        confirm_days=2,
        exogenous_confirmed=_always_confirmed(idx),
        partial_fraction=0.5,
    )
    assert path["label"].iloc[5] == "stress"
    assert (path["label"].iloc[5:10] == "stress").all()
    assert path["label"].iloc[10] == "calm"


def test_unconfirmed_stress_is_partial_derisk() -> None:
    idx = _idx(10)
    p = pd.Series([0.20, 0.20, 0.85, 0.85, 0.85, 0.85, 0.85, 0.20, 0.20, 0.20], index=idx)
    confirmed = pd.Series(
        [False, False, False, False, True, True, True, False, False, False],
        index=idx,
    )
    path = apply_anti_chattering(
        p,
        enter=0.70,
        exit=0.30,
        dwell=1,
        confirm_days=2,
        exogenous_confirmed=confirmed,
        partial_fraction=0.5,
    )
    stress = path["label"] == "stress"
    assert stress.any()
    partial_days = path.loc[stress & ~path["exogenous_confirms"], "derisk_fraction"]
    full_days = path.loc[stress & path["exogenous_confirms"], "derisk_fraction"]
    assert np.allclose(partial_days.to_numpy(dtype=float), 0.5)
    assert np.allclose(full_days.to_numpy(dtype=float), 1.0)
    assert np.allclose(path.loc[~stress, "derisk_fraction"].to_numpy(dtype=float), 0.0)


def test_causal_exogenous_confirm_uses_only_trailing_window() -> None:
    idx = _idx(20)
    level = pd.Series([0.0] * 15 + [10.0] * 5, index=idx)
    flag = causal_exogenous_confirm(level, percentile=80.0, window=10, min_periods=10)
    assert not bool(flag.iloc[14])
    assert bool(flag.iloc[-1])
    level2 = level.copy()
    level2.iloc[-1] = 10_000.0
    flag2 = causal_exogenous_confirm(level2, percentile=80.0, window=10, min_periods=10)
    assert bool(flag.iloc[14]) == bool(flag2.iloc[14])


# ---------------------------------------------------------------------------
# Detector API / [C1-bis]
# ---------------------------------------------------------------------------


def test_get_regime_probability_defaults_to_filtered_and_smoothed_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    idx = _idx(8)
    filtered = pd.Series([0.10, 0.12, 0.11, 0.80, 0.82, 0.15, 0.16, 0.14], index=idx)
    smoothed = pd.Series([0.40, 0.50, 0.70, 0.90, 0.88, 0.60, 0.30, 0.20], index=idx)
    config = make_regime_config(tmp_path)
    detector = RegimeDetector.from_probabilities(
        filtered,
        config,
        p_stress_smoothed=smoothed,
        exogenous_confirmed=_always_confirmed(idx),
    )
    date = idx[3]
    assert detector.get_regime_probability(date) == pytest.approx(0.80)
    assert detector.get_regime_probability(date, mode="filtered") == pytest.approx(0.80)
    with caplog.at_level("WARNING"):
        p_s = detector.get_regime_probability(date, mode="smoothed")
    assert p_s == pytest.approx(0.90)
    text = caplog.text.lower()
    assert "smoothed" in text
    assert "look-ahead" in text or "lookahead" in text or "c1" in text


def test_unknown_date_fails_loud(tmp_path: Path) -> None:
    idx = _idx(5)
    p = pd.Series(0.2, index=idx)
    detector = RegimeDetector.from_probabilities(
        p, make_regime_config(tmp_path), exogenous_confirmed=_always_confirmed(idx)
    )
    with pytest.raises(RegimeError, match="date"):
        detector.get_regime_probability("1999-01-01")
    with pytest.raises(RegimeError, match="date"):
        detector.get_regime_state("1999-01-01")


def test_get_regime_state_uses_five_layers_not_raw_probability(tmp_path: Path) -> None:
    idx = _idx(12)
    p = pd.Series([0.20] * 4 + [0.80] + [0.50] * 7, index=idx)
    detector = RegimeDetector.from_probabilities(
        p, make_regime_config(tmp_path, dwell={"min_days": 1}),
        exogenous_confirmed=_always_confirmed(idx),
    )
    assert detector.get_regime_state(idx[4]).label == "calm"
    assert detector.get_regime_probability(idx[4]) == pytest.approx(0.80)


def test_smoothed_probability_leads_filtered_into_a_break() -> None:
    """Smoothed P(stress) uses y_{t+1:T}; that is look-ahead if used as a signal."""
    rng = np.random.default_rng(3)
    n1, n2, n3 = 120, 40, 120
    y = np.concatenate(
        [
            rng.normal(0.0, 0.4, n1),
            rng.normal(1.2, 0.6, n2),
            rng.normal(0.0, 0.4, n3),
        ]
    )
    idx = _idx(len(y), start="2010-01-04")
    observed = pd.Series(y, index=idx, name="log_rv")
    config_dir = Path("/tmp")
    config = RegimeConfig.model_validate(
        _regime_payload(config_dir, input={"min_observations": 80, "rv_window": 5})
    )
    detector = RegimeDetector(config)
    detector.fit_observed(observed)
    break_date = idx[n1]
    loc = observed.index.get_loc(break_date)
    filtered = detector.filtered_stress.iloc[loc - 3 : loc]
    smoothed = detector.smoothed_stress.iloc[loc - 3 : loc]
    assert float(smoothed.max()) > float(filtered.max())
    assert float(smoothed.iloc[-1]) > float(filtered.iloc[-1])


def test_msgarch_backend_is_reserved_for_v2() -> None:
    with pytest.raises(NotImplementedError, match="v2"):
        MSGARCHBackend().fit(pd.Series([0.0, 1.0]), k_regimes=2, seed=7)


def test_detector_accepts_a_substitute_backend(tmp_path: Path) -> None:
    idx = _idx(12)
    observed = pd.Series(np.linspace(-2.0, 2.0, 12), index=idx)
    filtered = pd.DataFrame({0: 1.0 - np.linspace(0.1, 0.9, 12), 1: np.linspace(0.1, 0.9, 12)}, index=idx)
    smoothed = filtered.copy()
    from models.regime import FittedRegime, RegimeBackend

    class _Stub:
        name = "stub"

        def fit(self, series: pd.Series, *, k_regimes: int, seed: int) -> FittedRegime:
            assert k_regimes == 2
            transition = np.array([[0.9, 0.1], [0.2, 0.8]])
            params = pd.DataFrame(
                {"const": [-1.0, 1.0], "sigma2": [0.1, 0.2], "implied_rv": [np.exp(-1.0), np.exp(1.0)]},
                index=["calm", "stress"],
            )
            return FittedRegime(
                filtered=filtered,
                smoothed=smoothed,
                transition=transition,
                params=params,
                log_likelihood=1.0,
                aic=2.0,
                bic=3.0,
                stress_regime_id=1,
            )

    backend: RegimeBackend = _Stub()
    config = make_regime_config(tmp_path)
    detector = RegimeDetector(config, backend=backend)
    report = detector.fit_observed(observed, exogenous_confirmed=_always_confirmed(idx))
    assert report.backend == "stub"
    assert report.expected_durations["calm"] == pytest.approx(10.0)
    assert report.expected_durations["stress"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Option A on a two-vol return path
# ---------------------------------------------------------------------------


def test_option_a_labels_the_high_vol_block_as_stress(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n_calm, n_hot, n_calm2 = 150, 50, 150
    r = np.concatenate(
        [
            rng.normal(0.0, 0.004, n_calm),
            rng.normal(0.0, 0.035, n_hot),
            rng.normal(0.0, 0.004, n_calm2),
        ]
    )
    idx = _idx(len(r))
    returns = pd.Series(r, index=idx, name="r")
    config = make_regime_config(
        tmp_path,
        input={"min_observations": 80, "rv_window": 5},
        fit={"search_reps": 12, "maxiter": 400, "seed": 11},
    )
    detector = RegimeDetector(config)
    report = detector.fit(returns, exogenous_confirmed=_always_confirmed(idx))
    assert report.k_regimes == 2
    assert report.backend == "markov_log_variance"
    assert report.expected_durations["calm"] > 1.0
    assert report.expected_durations["stress"] > 1.0
    assert report.unconditional_probabilities["calm"] + report.unconditional_probabilities["stress"] == pytest.approx(1.0)
    hot = returns.index[n_calm + 5 : n_calm + n_hot - 5]
    calm = returns.index[20:80]
    p_hot = np.array([detector.get_regime_probability(d) for d in hot])
    p_calm = np.array([detector.get_regime_probability(d) for d in calm])
    assert float(np.mean(p_hot)) > 0.6
    assert float(np.mean(p_calm)) < 0.4
    assert report.regime_parameters.loc["stress", "const"] > report.regime_parameters.loc["calm", "const"]


def test_transition_stats_always_report_annual_rate_and_alarm(tmp_path: Path) -> None:
    idx = _idx(252 * 2)
    # Flip 0.85/0.10 every 15 days with D=1, N=1 → many transitions.
    values = np.array([0.85 if (i // 15) % 2 == 0 else 0.10 for i in range(len(idx))])
    p = pd.Series(values, index=idx)
    config = make_regime_config(
        tmp_path,
        dwell={"min_days": 1},
        confirmation={"consecutive_days": 1},
    )
    detector = RegimeDetector.from_probabilities(
        p, config, exogenous_confirmed=_always_confirmed(idx)
    )
    stats = detector.transition_stats()
    assert stats.transitions_per_year > 8.0
    assert stats.alarm is True
    assert stats.n_transitions > 0
    assert stats.annual_friction_bps == pytest.approx(
        stats.transitions_per_year * config.transitions.round_trip_cost_bps
    )
    assert stats.confirmation_delay_days == 1
    assert stats.mean_detection_lag_days >= 0.0


def test_three_panel_plot_and_structured_report(tmp_path: Path) -> None:
    idx = _idx(40)
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.0, 0.01, 40), index=idx, name="r")
    p = pd.Series([0.15] * 10 + [0.85] * 12 + [0.15] * 18, index=idx)
    config = make_regime_config(tmp_path, dwell={"min_days": 3}, confirmation={"consecutive_days": 2})
    detector = RegimeDetector.from_probabilities(
        p, config, exogenous_confirmed=_always_confirmed(idx)
    )
    fig, path = detector.plot(returns)
    assert path.is_file()
    assert path.stat().st_size > 0
    assert len(fig.axes) == 3
    report = detector.write_report(returns)
    assert report.plot_path.is_file()
    md = Path(config.output.report_markdown).read_text(encoding="utf-8")
    assert "filtered" in md.lower() or "filtrada" in md.lower()
    assert "smoothed" in md.lower() or "suavizada" in md.lower()
    assert "look-ahead" in md.lower() or "lookahead" in md.lower()
    assert "transitions" in md.lower() or "transiciones" in md.lower()
    est = detector.estimation_report
    assert est is not None
    assert "calm" in est.expected_durations
    assert "stress" in est.unconditional_probabilities
