"""Pydantic schema for config/risk.yaml — FHS / GJR filter knobs ([C2])."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_observations: int = Field(ge=2)
    dist: Literal["skewt"]
    mean: Literal["constant"]
    ljung_box_lags: int = Field(ge=1)
    ljung_box_pvalue_min: float = Field(gt=0, lt=1)
    p: int = Field(ge=1)
    o: int = Field(ge=1)
    q: int = Field(ge=1)
    rescale: bool
    ftol: float = Field(gt=0)
    maxiter: int = Field(ge=1)
    ar_lags: int = Field(ge=1)

    @field_validator("rescale")
    @classmethod
    def _no_silent_rescale(cls, value: bool) -> bool:
        if value:
            raise ValueError("arch rescale must be False so omega/alpha stay in return units")
        return value


class FHSBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alphas: list[float] = Field(min_length=1)
    n_simulations: int = Field(ge=10)
    seed: int
    aggregation: Literal["simple"]
    scale_sqrt_h: bool
    default_horizon: int = Field(ge=1)

    @model_validator(mode="after")
    def _no_sqrt_h(self) -> FHSBlock:
        if self.scale_sqrt_h:
            raise ValueError(
                "scale_sqrt_h is forbidden: VaR_h = VaR_1 * sqrt(h) is invalid under GARCH"
            )
        if any(not (0.0 < a < 1.0) for a in self.alphas):
            raise ValueError("alphas must lie in (0, 1)")
        return self


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    historical_window: int = Field(ge=20)
    parametric: Literal["normal"]


class RiskPlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_directory: str
    filename: str
    dpi: int = Field(gt=0)
    figsize_width: float = Field(gt=0)
    figsize_height: float = Field(gt=0)


class RiskOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_markdown: str
    report_html: str


class VaRBacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    significance: float = Field(gt=0, lt=1)
    dq_lags: int = Field(ge=1)
    rolling_window: int = Field(ge=20)
    acerbi_simulations: int = Field(ge=10)
    seed: int
    kupiec_critical: float = Field(gt=0)
    christoffersen_ind_critical: float = Field(gt=0)
    christoffersen_cc_critical: float = Field(gt=0)
    dq_critical_name: str = "chi2"


class OverlayEpisodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    start: date
    end: date

    @model_validator(mode="after")
    def _span(self) -> OverlayEpisodeConfig:
        if self.end < self.start:
            raise ValueError(f"episode {self.label}: end precedes start")
        return self


class OverlaySensitivityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kappa: list[float] = Field(min_length=1)
    sigma_target: list[float] = Field(min_length=1)
    band: list[float] = Field(min_length=1)


class OverlayPlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_directory: str
    filename: str
    dpi: int = Field(gt=0)
    figsize_width: float = Field(gt=0)
    figsize_height: float = Field(gt=0)


class OverlayOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_markdown: str


class OverlayConfig(BaseModel):
    """Exposure overlay on HRP weights. Does not replace the optimiser ([C2])."""

    model_config = ConfigDict(extra="forbid")

    sigma_target: float = Field(gt=0)
    periods_per_year: int = Field(gt=0)
    allow_leverage: bool
    leverage_cap: float = Field(gt=0)
    m_min: float = Field(ge=0)
    es_alpha: float = Field(gt=0, lt=1)
    es_horizon: int = Field(ge=1)
    es_budget: float = Field(gt=0)
    kappa: float = Field(gt=0, le=1)
    smoothing: float = Field(gt=0, le=1)
    band: float = Field(ge=0)
    turnover_alarm: float = Field(gt=0)
    legacy_stress_states: list[int] = Field(min_length=1)
    aggregator: Literal["min"]
    sensitivity: OverlaySensitivityConfig
    episodes: list[OverlayEpisodeConfig] = Field(min_length=1)
    plot: OverlayPlotConfig
    output: OverlayOutputConfig

    @model_validator(mode="after")
    def _leverage_is_a_policy_choice(self) -> OverlayConfig:
        if not self.allow_leverage and self.leverage_cap > 1.0 + 1e-12:
            raise ValueError(
                "leverage_cap > 1 requires allow_leverage=true; that is a mandate "
                "decision, not an implementation default"
            )
        if self.m_min > self.leverage_cap:
            raise ValueError("m_min cannot exceed leverage_cap")
        return self


class FHSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filter: FilterConfig
    fhs: FHSBlock
    comparison: ComparisonConfig
    plot: RiskPlotConfig
    output: RiskOutputConfig
    backtest: VaRBacktestConfig | None = None
    overlay: OverlayConfig | None = None


def load_fhs_config(path: str | Path) -> FHSConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return FHSConfig.model_validate(payload)
