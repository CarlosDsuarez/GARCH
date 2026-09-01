"""Pydantic schema for config/backtest.yaml — walk-forward knobs ([C2])."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_estimation_obs: int = Field(ge=2)
    reestimate: Literal["monthly"]
    reestimate_on: Literal["first_business_day"]
    window: Literal["expanding"]
    fit_uses_data_strictly_before_reestimate: bool

    @model_validator(mode="after")
    def _no_lookahead_fit(self) -> WalkForwardConfig:
        if not self.fit_uses_data_strictly_before_reestimate:
            raise ValueError(
                "GARCH must be estimated on data strictly before the reestimation "
                "date; otherwise the month's first return leaks into the parameters"
            )
        return self


class PredictiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizons: list[int] = Field(min_length=1)
    newey_west_lag_equals_horizon: bool

    @model_validator(mode="after")
    def _nw_required(self) -> PredictiveConfig:
        if not self.newey_west_lag_equals_horizon:
            raise ValueError("Newey-West lag must equal the forecast horizon")
        if any(h < 1 for h in self.horizons):
            raise ValueError("horizons must be positive")
        return self


class EpisodeStatsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_gap_calendar_days: int = Field(ge=1)
    block_length: int = Field(ge=2)
    n_bootstrap: int = Field(ge=10)


class CostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instrument: Literal["HYG", "LQD"]
    base_bps: float = Field(gt=0)
    k: float = Field(ge=0)
    lqd_base_bps: float = Field(gt=0)
    sensitivity_multipliers: list[float] = Field(min_length=1)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_percentile: float = Field(gt=50, lt=100)
    realized_vol_window: int = Field(ge=5)


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    periods_per_year: int = Field(gt=0)
    risk_free: float = Field(ge=0)


class BacktestPlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_directory: str
    dpi: int = Field(gt=0)
    figsize_width: float = Field(gt=0)
    figsize_height: float = Field(gt=0)


class BacktestOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_markdown: str
    report_html: str


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    walk_forward: WalkForwardConfig
    predictive: PredictiveConfig
    episodes: EpisodeStatsConfig
    costs: CostConfig
    benchmarks: BenchmarkConfig
    metrics: MetricsConfig
    seed: int
    plot: BacktestPlotConfig
    output: BacktestOutputConfig


def load_backtest_config(path: str | Path) -> BacktestConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return BacktestConfig.model_validate(payload)
