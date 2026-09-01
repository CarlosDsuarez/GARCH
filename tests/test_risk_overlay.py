"""Risk overlay: exposure multiplier on HRP weights, not a second optimiser.

Synthetic schema: ISO dates (YYYY-MM-DD), invented daily returns (~0.001) and
positive ES magnitudes. No live broker fills or autoportfolio state files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from risk.risk_overlay import (
    OverlayError,
    RiskOverlay,
    compare_regime_signals,
    hmm_states_to_stress,
    overlay_sensitivity,
    run_overlay_backtest,
)
from risk.schema import OverlayConfig, load_fhs_config


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_RISK = REPO_ROOT / "config" / "risk.yaml"


def _idx(n: int, start: str = "2019-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _overlay_payload(tmp_path: Path, **overrides) -> dict:
    payload: dict = {
        "sigma_target": 0.10,
        "periods_per_year": 252,
        "allow_leverage": False,
        "leverage_cap": 1.0,
        "m_min": 0.0,
        "es_alpha": 0.975,
        "es_horizon": 1,
        "es_budget": 0.025,
        "kappa": 0.50,
        "smoothing": 0.3,
        "band": 0.05,
        "turnover_alarm": 2.0,
        "legacy_stress_states": [2, 3],
        "aggregator": "min",
        "sensitivity": {
            "kappa": [0.30, 0.50, 0.70],
            "sigma_target": [0.08, 0.10, 0.12],
            "band": [0.03, 0.05, 0.10],
        },
        "episodes": [
            {"label": "COVID-19 credit crash", "start": "2020-03-01", "end": "2020-03-31"},
            {"label": "2022 QT / gilt / credit", "start": "2022-09-01", "end": "2022-11-30"},
            {"label": "SVB / regional banks", "start": "2023-03-01", "end": "2023-03-31"},
        ],
        "plot": {
            "output_directory": str(tmp_path / "overlay_plots"),
            "filename": "overlay_sensitivity.png",
            "dpi": 80,
            "figsize_width": 9.0,
            "figsize_height": 5.0,
        },
        "output": {
            "report_markdown": str(tmp_path / "overlay_report.md"),
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


def make_overlay_config(tmp_path: Path, **overrides) -> OverlayConfig:
    return OverlayConfig.model_validate(_overlay_payload(tmp_path, **overrides))


def _states(index: pd.DatetimeIndex, *, labels: list[str], confirmed: list[bool] | None = None) -> pd.DataFrame:
    flag = [True] * len(index) if confirmed is None else confirmed
    return pd.DataFrame({"label": labels, "exogenous_confirms": flag}, index=index)


def _inputs(
    index: pd.DatetimeIndex,
    *,
    sigma: float | pd.Series = 0.08,
    es: float | pd.Series = 0.01,
    labels: list[str] | None = None,
    confirmed: list[bool] | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    n = len(index)
    sig = pd.Series(sigma, index=index, dtype=float) if np.isscalar(sigma) else sigma
    es_s = pd.Series(es, index=index, dtype=float) if np.isscalar(es) else es
    lab = labels if labels is not None else ["calm"] * n
    return sig, es_s, _states(index, labels=lab, confirmed=confirmed)


# ---------------------------------------------------------------------------
# Production knobs / policy
# ---------------------------------------------------------------------------


def test_production_overlay_uses_min_not_product_and_forbids_leverage() -> None:
    config = load_fhs_config(PRODUCTION_RISK)
    assert config.overlay is not None
    ov = config.overlay
    assert ov.sigma_target == pytest.approx(0.10)
    assert ov.kappa == pytest.approx(0.50)
    assert ov.band == pytest.approx(0.05)
    assert ov.smoothing == pytest.approx(0.3)
    assert ov.es_alpha == pytest.approx(0.975)
    assert ov.es_horizon == 1
    assert ov.allow_leverage is False
    assert ov.leverage_cap == pytest.approx(1.0)
    assert ov.turnover_alarm == pytest.approx(2.0)
    assert ov.aggregator == "min"


def test_config_rejects_silent_leverage(tmp_path: Path) -> None:
    payload = _overlay_payload(tmp_path)
    payload["allow_leverage"] = False
    payload["leverage_cap"] = 1.5
    with pytest.raises(ValidationError, match="leverage"):
        OverlayConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Three components and the min aggregator
# ---------------------------------------------------------------------------


def test_vol_targeting_caps_at_one_when_forecast_is_below_target(tmp_path: Path) -> None:
    idx = _idx(8)
    sig, es, states = _inputs(idx, sigma=0.05)
    overlay = RiskOverlay(make_overlay_config(tmp_path, smoothing=1.0, band=0.0)).build(sig, es, states)
    expl = overlay.explain(idx[3])
    assert expl.m_vol == pytest.approx(1.0)
    assert overlay.compute_multiplier(idx[3]) <= 1.0


def test_vol_targeting_scales_when_forecast_exceeds_target(tmp_path: Path) -> None:
    idx = _idx(8)
    sig, es, states = _inputs(idx, sigma=0.20)
    overlay = RiskOverlay(make_overlay_config(tmp_path, smoothing=1.0, band=0.0)).build(sig, es, states)
    assert overlay.explain(idx[2]).m_vol == pytest.approx(0.10 / 0.20)


def test_es_constraint_binds_only_when_forecast_exceeds_budget(tmp_path: Path) -> None:
    idx = _idx(6)
    sig, es, states = _inputs(idx, es=0.01)
    overlay = RiskOverlay(
        make_overlay_config(tmp_path, es_budget=0.025, smoothing=1.0, band=0.0)
    ).build(sig, es, states)
    assert overlay.explain(idx[1]).m_es == pytest.approx(1.0)

    es_hot = pd.Series(0.05, index=idx)
    overlay_hot = RiskOverlay(
        make_overlay_config(tmp_path, es_budget=0.025, smoothing=1.0, band=0.0)
    ).build(sig, es_hot, states)
    assert overlay_hot.explain(idx[1]).m_es == pytest.approx(0.025 / 0.05)


def test_regime_kappa_and_unconfirmed_half_cut(tmp_path: Path) -> None:
    idx = _idx(6)
    kappa = 0.50
    labels = ["calm", "calm", "stress", "stress", "stress", "calm"]
    confirmed = [True, True, True, False, False, True]
    sig, es, states = _inputs(idx, labels=labels, confirmed=confirmed)
    overlay = RiskOverlay(
        make_overlay_config(tmp_path, kappa=kappa, smoothing=1.0, band=0.0)
    ).build(sig, es, states)
    assert overlay.explain(idx[0]).m_regime == pytest.approx(1.0)
    assert overlay.explain(idx[2]).m_regime == pytest.approx(kappa)
    assert overlay.explain(idx[3]).m_regime == pytest.approx(1.0 - (1.0 - kappa) / 2.0)


def test_final_multiplier_is_the_min_not_the_product(tmp_path: Path) -> None:
    idx = _idx(5)
    sig = pd.Series(0.20, index=idx)
    es = pd.Series(0.05, index=idx)
    states = _states(idx, labels=["stress"] * 5, confirmed=[True] * 5)
    overlay = RiskOverlay(
        make_overlay_config(tmp_path, es_budget=0.025, kappa=0.50, smoothing=1.0, band=0.0)
    ).build(sig, es, states)
    raw = overlay.explain(idx[2]).m_raw
    assert raw == pytest.approx(0.5)
    assert raw != pytest.approx(0.5 * 0.5 * 0.5)
    assert overlay.explain(idx[2]).binding in {"vol", "es", "regime", "tie"}


def test_apply_preserves_relative_hrp_weights_and_parks_the_rest_in_cash(tmp_path: Path) -> None:
    idx = _idx(4)
    sig = pd.Series(0.20, index=idx)
    es = pd.Series(0.01, index=idx)
    states = _states(idx, labels=["calm"] * 4)
    overlay = RiskOverlay(make_overlay_config(tmp_path, smoothing=1.0, band=0.0)).build(sig, es, states)
    w_raw = pd.Series({"HYG": 0.6, "LQD": 0.4})
    w_final = overlay.apply(w_raw, idx[1])
    m = overlay.compute_multiplier(idx[1])
    assert float(w_final[["HYG", "LQD"]].sum()) == pytest.approx(m)
    assert w_final["cash"] == pytest.approx(1.0 - m)
    assert w_final["HYG"] / m == pytest.approx(0.6)
    assert w_final["LQD"] / m == pytest.approx(0.4)
    assert float(w_final.sum()) == pytest.approx(1.0)


def test_explain_names_the_binding_constraint(tmp_path: Path) -> None:
    idx = _idx(3)
    sig = pd.Series(0.40, index=idx)
    es = pd.Series(0.01, index=idx)
    states = _states(idx, labels=["calm"] * 3)
    overlay = RiskOverlay(make_overlay_config(tmp_path, smoothing=1.0, band=0.0)).build(sig, es, states)
    expl = overlay.explain(idx[0])
    assert expl.binding == "vol"
    assert "vol" in expl.reason.lower() or "sigma" in expl.reason.lower()
    assert expl.es_horizon == 1
    assert expl.es_alpha == pytest.approx(0.975)


# ---------------------------------------------------------------------------
# Turnover control
# ---------------------------------------------------------------------------


def test_band_skips_rebalance_inside_five_points_of_exposure(tmp_path: Path) -> None:
    idx = _idx(6)
    sig = pd.Series([0.10, 0.101, 0.102, 0.103, 0.20, 0.20], index=idx)
    es = pd.Series(0.01, index=idx)
    states = _states(idx, labels=["calm"] * 6)
    overlay = RiskOverlay(
        make_overlay_config(tmp_path, smoothing=1.0, band=0.05)
    ).build(sig, es, states)
    m0 = overlay.compute_multiplier(idx[0])
    assert overlay.compute_multiplier(idx[1]) == pytest.approx(m0)
    assert overlay.compute_multiplier(idx[2]) == pytest.approx(m0)
    assert overlay.compute_multiplier(idx[4]) < m0 - 0.04


def test_smoothing_is_applied_before_the_band(tmp_path: Path) -> None:
    idx = _idx(5)
    sig = pd.Series([0.10, 0.50, 0.50, 0.50, 0.50], index=idx)
    es = pd.Series(0.01, index=idx)
    states = _states(idx, labels=["calm"] * 5)
    jumpy = RiskOverlay(make_overlay_config(tmp_path, smoothing=1.0, band=0.0)).build(sig, es, states)
    smooth = RiskOverlay(make_overlay_config(tmp_path, smoothing=0.3, band=0.0)).build(sig, es, states)
    assert jumpy.explain(idx[1]).m_raw == pytest.approx(0.10 / 0.50)
    assert smooth.explain(idx[1]).m_smoothed > jumpy.explain(idx[1]).m_raw
    assert smooth.explain(idx[1]).m_smoothed < jumpy.explain(idx[0]).m_raw


def test_future_sigma_does_not_change_todays_multiplier(tmp_path: Path) -> None:
    idx = _idx(10)
    sig, es, states = _inputs(idx, sigma=0.12)
    overlay = RiskOverlay(make_overlay_config(tmp_path)).build(sig, es, states)
    m_today = overlay.compute_multiplier(idx[2])
    sig2 = sig.copy()
    sig2.iloc[-1] = 0.80
    overlay2 = RiskOverlay(make_overlay_config(tmp_path)).build(sig2, es, states)
    assert overlay2.compute_multiplier(idx[2]) == pytest.approx(m_today)


# ---------------------------------------------------------------------------
# Four-config backtest, concordance, sensitivity
# ---------------------------------------------------------------------------


def test_four_config_backtest_reports_e1_metrics(tmp_path: Path) -> None:
    idx = _idx(80)
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.0004, 0.01, 80), index=idx, name="r")
    sig = pd.Series(np.where(np.arange(80) < 40, 0.08, 0.22), index=idx)
    es = pd.Series(np.where(np.arange(80) < 40, 0.012, 0.04), index=idx)
    labels = ["calm"] * 40 + ["stress"] * 40
    states = _states(idx, labels=labels, confirmed=[True] * 80)
    legacy = pd.Series([False] * 30 + [True] * 50, index=idx)
    overlay = RiskOverlay(make_overlay_config(tmp_path, smoothing=0.3, band=0.05))
    overlay.build(sig, es, states, legacy_stress=legacy)
    result = run_overlay_backtest(overlay, r)
    for name in ("base", "vol_only", "full_ms", "full_legacy"):
        metrics = result.metrics[name]
        assert metrics.sharpe == metrics.sharpe
        assert metrics.max_drawdown <= 0.0
        assert metrics.ann_turnover >= 0.0
        assert hasattr(metrics, "sortino")
        assert hasattr(metrics, "calmar")
        assert hasattr(metrics, "hit_rate")
        assert hasattr(metrics, "payoff_ratio")
    assert result.incremental_turnover["full_ms"] == pytest.approx(
        result.metrics["full_ms"].ann_turnover - result.metrics["base"].ann_turnover
    )


def test_compare_regime_signals_concordance_lead_and_conjunction(tmp_path: Path) -> None:
    idx = _idx(40, start="2020-02-03")
    ms = pd.Series([False] * 10 + [True] * 20 + [False] * 10, index=idx)
    legacy = pd.Series([False] * 14 + [True] * 16 + [False] * 10, index=idx)
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.01, 40), index=idx)
    sig, es, states = _inputs(
        idx,
        sigma=0.08,
        es=0.01,
        labels=["stress" if v else "calm" for v in ms],
        confirmed=[True] * 40,
    )
    overlay = RiskOverlay(make_overlay_config(tmp_path, smoothing=1.0, band=0.0))
    overlay.build(sig, es, states, legacy_stress=legacy)
    report = compare_regime_signals(
        overlay,
        returns=r,
        episodes=[("COVID-19 credit crash", idx[8], idx[30])],
    )
    assert 0.0 < report.agreement_rate < 1.0
    assert not report.discrepancies.empty
    assert set(report.discrepancies["direction"]) <= {
        "ms_stress_legacy_calm",
        "ms_calm_legacy_stress",
    }
    row = report.episode_leaders.iloc[0]
    assert row["leader"] == "ms"
    assert int(row["lead_days"]) > 0
    assert report.overlay_ms.ann_turnover >= 0.0
    assert report.overlay_legacy.ann_turnover >= 0.0
    assert report.overlay_conjunction.ann_turnover >= 0.0
    assert int(overlay.conjunction_stress.sum()) == int((ms & legacy).sum())


def test_hmm_four_state_mapping_does_not_replace_the_legacy_detector() -> None:
    idx = _idx(8)
    states = pd.Series([0, 0, 1, 2, 3, 3, 1, 0], index=idx)
    stress = hmm_states_to_stress(states, stress_ids=(2, 3))
    assert list(stress.astype(int)) == [0, 0, 0, 1, 1, 1, 0, 0]


def test_sensitivity_table_has_sharpe_and_max_drawdown(tmp_path: Path) -> None:
    idx = _idx(50)
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0.0003, 0.012, 50), index=idx)
    sig = pd.Series(0.14, index=idx)
    es = pd.Series(0.02, index=idx)
    labels = ["calm"] * 25 + ["stress"] * 25
    states = _states(idx, labels=labels)
    overlay = RiskOverlay(
        make_overlay_config(
            tmp_path,
            sensitivity={"kappa": [0.4, 0.6], "sigma_target": [0.10], "band": [0.05]},
        )
    ).build(sig, es, states)
    table, heat = overlay_sensitivity(overlay, r)
    assert {"kappa", "sigma_target", "band", "sharpe", "max_drawdown"}.issubset(table.columns)
    assert table.shape[0] == 2
    assert heat.is_file()
    paths = overlay.write_report(r)
    md = Path(paths.report_markdown).read_text(encoding="utf-8")
    assert "min" in md.lower()
    assert "conjunction" in md.lower() or "conjunción" in md.lower()
    assert "turnover" in md.lower()
    assert "1-day" in md.lower() or "1 día" in md.lower() or "horizon" in md.lower()


def test_unknown_date_and_nonsumming_weights_fail_loud(tmp_path: Path) -> None:
    idx = _idx(4)
    sig, es, states = _inputs(idx)
    overlay = RiskOverlay(make_overlay_config(tmp_path)).build(sig, es, states)
    with pytest.raises(OverlayError, match="date"):
        overlay.compute_multiplier("1999-01-01")
    with pytest.raises(OverlayError, match="sum"):
        overlay.apply(pd.Series({"HYG": 0.3, "LQD": 0.3}), idx[0])
