"""Walk-forward backtest: frozen GARCH params, Newey-West, episodes, costs.

Synthetic schema: ISO dates (YYYY-MM-DD), invented HYG returns (~0.001) and
OAS/EBP levels (5.0 / 0.45). No live ETF or Fed vintages are stored here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from backtest.signal_backtest import (
    FrozenParams,
    WalkForwardBacktester,
    block_bootstrap_ci,
    count_independent_episodes,
    economic_metrics,
    forward_returns,
    predictive_regression,
    quintile_table,
    state_dependent_cost_bps,
)
from backtest.schema import BacktestConfig, load_backtest_config
from test_dislocation import make_signal_config


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_BACKTEST = REPO_ROOT / "config" / "backtest.yaml"


def _bt_payload(tmp_path: Path, **overrides) -> dict:
    payload = {
        "walk_forward": {
            "min_estimation_obs": 25,
            "reestimate": "monthly",
            "reestimate_on": "first_business_day",
            "window": "expanding",
            "fit_uses_data_strictly_before_reestimate": True,
        },
        "predictive": {"horizons": [5, 10], "newey_west_lag_equals_horizon": True},
        "episodes": {"min_gap_calendar_days": 60, "block_length": 60, "n_bootstrap": 80},
        "costs": {
            "instrument": "HYG",
            "base_bps": 3.0,
            "k": 1.5,
            "lqd_base_bps": 2.0,
            "sensitivity_multipliers": [1.0, 2.0, 3.0],
        },
        "benchmarks": {"level_percentile": 85.0, "realized_vol_window": 20},
        "metrics": {"periods_per_year": 252, "risk_free": 0.0},
        "seed": 7,
        "plot": {
            "output_directory": str(tmp_path / "bt_plots"),
            "dpi": 80,
            "figsize_width": 8.0,
            "figsize_height": 4.0,
        },
        "output": {
            "report_markdown": str(tmp_path / "backtest_report.md"),
            "report_html": str(tmp_path / "backtest_report.html"),
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


def make_bt_config(tmp_path: Path, **overrides) -> BacktestConfig:
    return BacktestConfig.model_validate(_bt_payload(tmp_path, **overrides))


@dataclass
class RecordingEstimator:
    """Injected stand-in: frozen scale from the training window only."""

    fits: list[pd.Timestamp] = field(default_factory=list)
    filter_ends: list[pd.Timestamp] = field(default_factory=list)
    n_fits: int = 0

    def fit(self, series: pd.Series) -> FrozenParams:
        if series.empty:
            raise ValueError("empty estimation window")
        self.n_fits += 1
        through = pd.Timestamp(series.index.max())
        self.fits.append(through)
        return FrozenParams(
            fitted_through=through,
            values={
                "omega": 0.01,
                "alpha": 0.05,
                "beta": 0.80 + 0.01 * self.n_fits,
                "scale": float(series.diff().dropna().std() or 0.01),
                "n_obs": float(series.shape[0]),
                "fit_id": float(self.n_fits),
            },
        )

    def filter(self, params: FrozenParams, series: pd.Series) -> pd.Series:
        self.filter_ends.append(pd.Timestamp(series.index.max()))
        r = series.diff().dropna()
        lam = float(params.values["beta"])
        scale = float(params.values["scale"])
        sigma = np.empty(len(r), dtype=float)
        prev = scale
        for i, val in enumerate(r.to_numpy()):
            prev = float(np.sqrt((1.0 - lam) * val * val + lam * prev * prev))
            sigma[i] = prev
        out = pd.Series(sigma, index=r.index, name="sigma")
        return out.reindex(series.index).bfill()


def _panel(n: int = 90, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    r = rng.normal(0.0004, 0.008, size=n)
    oas = 5.0 + np.cumsum(rng.normal(0, 0.05, size=n))
    ebp = 0.40 + np.cumsum(rng.normal(0, 0.01, size=n))
    return pd.DataFrame(
        {
            "credit_return": r,
            "oas_level": np.clip(oas, 1.0, None),
            "ebp_level": ebp,
            "default_proxy": 1.5 + 0.05 * np.sin(np.arange(n) / 8.0),
            "vix": 15.0 + rng.normal(0, 2.0, size=n),
        },
        index=idx,
    )


def _run(tmp_path: Path, panel: pd.DataFrame | None = None, **bt_over):
    frame = panel if panel is not None else _panel()
    oas_est = RecordingEstimator()
    ebp_est = RecordingEstimator()
    bt = WalkForwardBacktester(
        credit_returns=frame["credit_return"],
        oas_level=frame["oas_level"],
        ebp_level=frame["ebp_level"],
        default_proxy=frame["default_proxy"],
        vix=frame["vix"],
        signal_config=make_signal_config(tmp_path),
        backtest_config=make_bt_config(tmp_path, **bt_over),
        oas_estimator=oas_est,
        ebp_estimator=ebp_est,
    )
    result = bt.run()
    return result, oas_est, ebp_est, bt


def test_production_backtest_config_is_monthly_walk_forward() -> None:
    config = load_backtest_config(PRODUCTION_BACKTEST)
    assert config.walk_forward.min_estimation_obs >= 1000
    assert config.walk_forward.reestimate == "monthly"
    assert config.walk_forward.reestimate_on == "first_business_day"
    assert config.costs.base_bps == pytest.approx(3.0)
    assert config.costs.lqd_base_bps == pytest.approx(2.0)
    assert config.episodes.block_length == 60
    assert config.episodes.min_gap_calendar_days == 60
    assert 3.0 in config.costs.sensitivity_multipliers


def test_config_rejects_daily_reestimation(tmp_path: Path) -> None:
    payload = _bt_payload(tmp_path)
    payload["walk_forward"]["reestimate"] = "daily"
    with pytest.raises(ValidationError, match="monthly"):
        BacktestConfig.model_validate(payload)


def test_estimator_never_sees_data_on_or_after_reestimation_date(tmp_path: Path) -> None:
    result, oas_est, _, bt = _run(tmp_path)
    assert oas_est.n_fits >= 2
    for through, tau in zip(oas_est.fits, result.reestimation_dates):
        assert through < pd.Timestamp(tau)


def test_parameters_are_frozen_between_monthly_reestimates(tmp_path: Path) -> None:
    result, _, _, _ = _run(tmp_path)
    traj = result.parameter_trajectory
    assert not traj.empty
    months = traj.index.to_period("M")
    for month in months.unique():
        block = traj.loc[months == month, "oas_beta"]
        assert block.nunique() == 1


def test_reestimation_dates_are_first_business_days(tmp_path: Path) -> None:
    result, _, _, _ = _run(tmp_path)
    for tau in result.reestimation_dates:
        month_days = pd.bdate_range(tau.replace(day=1), periods=5)
        assert pd.Timestamp(tau) == month_days[0]


@pytest.mark.blocking
def test_walk_forward_score_at_t_matches_truncated_run(tmp_path: Path) -> None:
    frame = _panel(100)
    t = frame.index[70]
    full, _, _, _ = _run(tmp_path, frame)
    trunc, _, _, _ = _run(tmp_path, frame.loc[:t])
    assert full.panel.loc[t, "score"] == pytest.approx(trunc.panel.loc[t, "score"], abs=1e-10)


def test_forward_return_does_not_include_same_day(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-02", periods=8)
    r = pd.Series([0.01] * 8, index=idx)
    fwd = forward_returns(r, horizon=3)
    t = idx[0]
    expected = (1.01**3) - 1.0
    assert fwd.loc[t] == pytest.approx(expected)


def test_predictive_regression_uses_newey_west_lag_equal_to_horizon() -> None:
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2018-01-02", periods=200)
    score = pd.Series(rng.normal(0, 1, size=200), index=idx)
    r = 0.02 * score.shift(1) + rng.normal(0, 0.05, size=200)
    r = pd.Series(r.to_numpy(), index=idx)
    fwd = forward_returns(r, horizon=10)
    aligned = pd.concat({"y": fwd, "x": score}, axis=1).dropna()
    report = predictive_regression(aligned["y"], aligned["x"], horizon=10)
    assert report.nw_lags == 10
    assert report.se_newey_west >= report.se_ols - 1e-15
    assert np.isfinite(report.r_squared)


def test_quintiles_are_monotone_when_score_ranks_future_returns() -> None:
    idx = pd.bdate_range("2020-01-02", periods=50)
    score = pd.Series(np.tile(np.arange(5, dtype=float), 10), index=idx)
    fwd = pd.Series(score.to_numpy() * 0.01, index=idx)
    table = quintile_table(fwd, score)
    assert list(table.index) == [1, 2, 3, 4, 5]
    assert table["mean_forward"].is_monotonic_increasing


def test_independent_episodes_merge_when_gap_under_60_days() -> None:
    idx = pd.bdate_range("2020-01-02", periods=80)
    active = pd.Series(False, index=idx)
    active.iloc[2:6] = True
    active.iloc[10:14] = True
    active.iloc[60:64] = True
    n_close = count_independent_episodes(active, min_gap_calendar_days=60)
    assert n_close == 2


def test_block_bootstrap_uses_requested_block_length() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, size=180))
    ci = block_bootstrap_ci(r, block_length=60, n_bootstrap=40, seed=7)
    assert ci.block_length == 60
    assert ci.n_bootstrap == 40
    assert ci.lower <= ci.upper


def test_state_dependent_cost_rises_with_forecast_vol() -> None:
    quiet = state_dependent_cost_bps(sigma=0.10, sigma_median=0.20, base_bps=3.0, k=1.5)
    stressed = state_dependent_cost_bps(sigma=0.40, sigma_median=0.20, base_bps=3.0, k=1.5)
    assert stressed > quiet
    assert quiet == pytest.approx(3.0 + 1.5 * 0.5)


def test_economic_metrics_include_required_fields() -> None:
    idx = pd.bdate_range("2020-01-02", periods=40)
    r = pd.Series(0.001, index=idx)
    w = pd.Series(1.0, index=idx)
    mets = economic_metrics(r, weights=w, periods_per_year=252)
    required = {
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "max_drawdown_duration",
        "hit_rate",
        "payoff_ratio",
        "ann_turnover",
    }
    assert required <= set(mets)


def test_backtest_report_has_benchmarks_cost_sensitivity_and_failure_section(
    tmp_path: Path,
) -> None:
    result, _, _, bt = _run(tmp_path)
    paths = bt.write_report(result)
    text = Path(paths.markdown).read_text(encoding="utf-8")
    assert "Newey-West" in text
    assert "quintil" in text.lower()
    assert "episodios independientes" in text.lower()
    assert "Buy & hold" in text
    assert "OAS" in text
    assert "VIX" in text
    assert "60" in text
    assert "Condiciones bajo las cuales" in text
    assert "3" in text
    assert paths.html.exists()
    assert result.cost_sensitivity[3.0].sharpe is not None
    assert {"buy_hold", "oas_level", "vix", "realized_vol"} <= set(result.benchmarks)
