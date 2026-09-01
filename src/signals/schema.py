"""Pydantic schema for config/signal.yaml — dislocation-score knobs ([C2])."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PercentileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window: int = Field(ge=2)
    min_periods: int = Field(ge=2)

    @model_validator(mode="after")
    def _window_consistent(self) -> PercentileConfig:
        if self.min_periods > self.window:
            raise ValueError("min_periods cannot exceed the rolling window")
        return self


class OptionBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    date_column: str
    value_column: str
    source_column: str
    observation_date_column: str
    notes: str


class DefaultProxyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option: Literal["A", "B", "C"]
    gz_spread_column: str
    ebp_column: str
    ccc_series_id: str
    bbb_series_id: str
    option_b: OptionBConfig


class ScoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: Literal["default_bilinear"]
    vol_weight: float = Field(ge=0, le=1)
    level_weight: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ScoreConfig:
        if abs(self.vol_weight + self.level_weight - 1.0) > 1e-12:
            raise ValueError("vol_weight + level_weight must equal 1")
        return self


class HysteresisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activate: float = Field(gt=-1, le=1)
    deactivate: float = Field(ge=-1, lt=1)

    @model_validator(mode="after")
    def _ordered(self) -> HysteresisConfig:
        if self.deactivate >= self.activate:
            raise ValueError("hysteresis deactivate must be strictly below activate")
        return self


class PositionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: float = Field(gt=0)
    abs_cap: float = Field(gt=0)
    size_only_when_active: bool
    min_sigma: float = Field(gt=0)


class EpisodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    start: date
    end: date

    @model_validator(mode="after")
    def _span(self) -> EpisodeConfig:
        if self.end < self.start:
            raise ValueError(f"episode {self.label}: end precedes start")
        return self


class SignalPlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_directory: str
    filename: str
    dpi: int = Field(gt=0)
    figsize_width: float = Field(gt=0)
    figsize_height: float = Field(gt=0)


class SignalOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_table: str


class DislocationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    percentile: PercentileConfig
    default_proxy: DefaultProxyConfig
    score: ScoreConfig
    hysteresis: HysteresisConfig
    position: PositionConfig
    episodes: list[EpisodeConfig] = Field(min_length=1)
    plot: SignalPlotConfig
    output: SignalOutputConfig


def load_signal_config(path: str | Path) -> DislocationConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return DislocationConfig.model_validate(payload)
