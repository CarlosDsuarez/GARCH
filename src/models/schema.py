"""Pydantic schema for config/params.yaml — EGARCH and mean-equation knobs ([C2])."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TransformConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    percent_scale: float = Field(gt=0)
    invert_sign: bool

    @field_validator("invert_sign")
    @classmethod
    def _must_invert(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "Credit-stress returns must invert the OAS-log change so that "
                "widening maps to r_t < 0 (GARCH leverage convention)."
            )
        return value


class MeanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acf_lags: int = Field(ge=1)
    ljung_box_lags: int = Field(ge=1)
    ljung_box_pvalue_min: float = Field(gt=0, lt=1)
    rho1_matrix_pricing_threshold: float = Field(gt=0, lt=1)
    variance_ratio_horizons: list[int] = Field(min_length=1)
    ar_lags: int = Field(ge=1)
    ma_lags: int = Field(ge=1)


class VarianceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vol: Literal["EGARCH"]
    p: int = Field(ge=1)
    o: int = Field(ge=0)
    q: int = Field(ge=1)
    rescale: bool
    ftol: float = Field(gt=0)
    maxiter: int = Field(ge=1)
    half_life_mass: float = Field(gt=0, lt=1)
    stationarity_abs_beta_max: float = Field(gt=0)
    typical_beta_min: float
    typical_beta_max: float
    significance_level: float = Field(gt=0, lt=1)
    expected_egarch_gamma_sign: Literal[-1, 1]

    @field_validator("rescale")
    @classmethod
    def _no_silent_rescale(cls, value: bool) -> bool:
        if value:
            raise ValueError("rescale must be False so omega/alpha stay in percent units")
        return value


class DistributionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[str] = Field(min_length=1)
    selection_ic: Literal["bic"]
    forbidden: list[str]
    nu_fourth_moment_min: float = Field(gt=0)
    nu_normal_collapse: float = Field(gt=0)

    @model_validator(mode="after")
    def _ban_gaussian(self) -> DistributionConfig:
        lowered = {name.lower() for name in self.candidates}
        forbidden = {name.lower() for name in self.forbidden}
        if lowered & forbidden:
            raise ValueError("Gaussian innovations are forbidden for credit returns")
        if any(name in {"normal", "gaussian"} for name in lowered):
            raise ValueError("dist='normal' is forbidden ([Paso 5])")
        return self


class ForecastConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizons: list[int] = Field(min_length=1)
    method: Literal["simulation"]
    simulations: int = Field(ge=1)
    reindex: bool


class SignTestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tail_percentile: float = Field(gt=0, lt=50)
    min_tail_observations: int = Field(ge=1)


class OasUniverseEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    quality_rank: int


class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_directory: str
    dpi: int = Field(gt=0)
    figsize_width: float = Field(gt=0)
    figsize_height: float = Field(gt=0)
    filename_template: str


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparative_table: str


class EbpStationarityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adf_pvalue_max: float = Field(gt=0, lt=1)
    kpss_pvalue_min: float = Field(gt=0, lt=1)
    adf_regression: Literal["c"]
    kpss_regression: Literal["c"]
    adf_maxlag: int = Field(ge=1)


class EbpVarianceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vol: Literal["GARCH"]
    p: int = Field(ge=1)
    o: int = Field(ge=1)
    q: int = Field(ge=1)
    rescale: bool
    ftol: float = Field(gt=0)
    maxiter: int = Field(ge=1)
    half_life_mass: float = Field(gt=0, lt=1)
    significance_level: float = Field(gt=0, lt=1)
    expected_gjr_gamma_sign: Literal[-1, 1]
    stationarity_alpha_half_gamma_beta_max: float = Field(gt=0)

    @field_validator("rescale")
    @classmethod
    def _no_silent_rescale(cls, value: bool) -> bool:
        if value:
            raise ValueError("rescale must be False")
        return value


class EbpDisaggregationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["chow_lin"]
    fallback: Literal["denton_cholette"]
    include_constant: bool
    rho_grid_min: float
    rho_grid_max: float
    rho_grid_size: int = Field(ge=3)
    condition_number_max: float = Field(gt=0)
    consistency_atol: float = Field(gt=0)
    high_freq_rho_from_monthly: bool


class EbpSignalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_frequency: Literal["monthly"]
    daily_may_originate: bool
    robustness_min_correlation: float = Field(gt=0, le=1)

    @field_validator("daily_may_originate")
    @classmethod
    def _daily_is_timing_only(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "Daily disaggregated EBP must not originate a signal "
                "(partial circularity with HY OAS / VIX anchors)"
            )
        return value


class EbpModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invert_sign: bool
    min_observations: int = Field(gt=0)
    stationarity: EbpStationarityConfig
    variance: EbpVarianceConfig
    distribution: DistributionConfig
    mean: MeanConfig
    forecast: ForecastConfig
    sign_test: SignTestConfig
    disaggregation: EbpDisaggregationConfig
    signal: EbpSignalConfig
    plot: PlotConfig
    output: OutputConfig

    @field_validator("invert_sign")
    @classmethod
    def _must_invert(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "Rising EBP is bad news; invert ΔEBP so r_t < 0 maps to I[ε<0]"
            )
        return value


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    transform: TransformConfig
    mean: MeanConfig
    variance: VarianceConfig
    distribution: DistributionConfig
    forecast: ForecastConfig
    sign_test: SignTestConfig
    min_observations: int = Field(gt=0)
    oas_universe: dict[str, OasUniverseEntry]
    plot: PlotConfig
    output: OutputConfig
    ebp: EbpModelConfig | None = None


def load_model_config(path: str | Path) -> ModelConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ModelConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Markov-switching regime detector (config/regime.yaml)
# ---------------------------------------------------------------------------


class RegimeInputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measure: Literal["log_rv", "log_rv_oas", "rolling_pc1"]
    rv_window: int = Field(ge=2)
    pca_window: int = Field(ge=3)
    pca_min_periods: int = Field(ge=3)
    min_observations: int = Field(ge=10)

    @model_validator(mode="after")
    def _pca_window(self) -> RegimeInputConfig:
        if self.pca_min_periods > self.pca_window:
            raise ValueError("pca_min_periods cannot exceed pca_window")
        return self


class RegimeFitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k_regimes: int = Field(ge=2, le=3)
    switching_variance: bool
    trend: Literal["c"]
    search_reps: int = Field(ge=0)
    maxiter: int = Field(ge=1)
    seed: int
    backend: Literal["markov_log_variance", "msgarch"]


class RegimeHysteresisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enter: float = Field(gt=0, lt=1)
    exit: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _two_thresholds(self) -> RegimeHysteresisConfig:
        if self.enter <= self.exit:
            raise ValueError(
                "hysteresis enter must exceed exit (a single threshold chatters)"
            )
        return self


class RegimeDwellConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_days: int = Field(ge=1)


class RegimeConfirmationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consecutive_days: int = Field(ge=1)


class RegimeExogenousConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    percentile: float = Field(gt=50, lt=100)
    window: int = Field(ge=2)
    min_periods: int = Field(ge=2)
    partial_derisk_fraction: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _window(self) -> RegimeExogenousConfig:
        if self.min_periods > self.window:
            raise ValueError("exogenous min_periods cannot exceed window")
        return self


class RegimeK3Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_unconditional: float = Field(gt=0, lt=1)
    min_expected_duration_days: float = Field(gt=0)


class RegimeTransitionBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alarm_per_year: float = Field(gt=0)
    round_trip_cost_bps: float = Field(ge=0)
    periods_per_year: int = Field(gt=0)


class RegimePlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_directory: str
    filename: str
    dpi: int = Field(gt=0)
    figsize_width: float = Field(gt=0)
    figsize_height: float = Field(gt=0)


class RegimeOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_markdown: str


class RegimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: RegimeInputConfig
    fit: RegimeFitConfig
    hysteresis: RegimeHysteresisConfig
    dwell: RegimeDwellConfig
    confirmation: RegimeConfirmationConfig
    exogenous: RegimeExogenousConfig
    k3: RegimeK3Config
    transitions: RegimeTransitionBudgetConfig
    plot: RegimePlotConfig
    output: RegimeOutputConfig


def load_regime_config(path: str | Path) -> RegimeConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return RegimeConfig.model_validate(payload)
