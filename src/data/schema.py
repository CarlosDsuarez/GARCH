"""Pydantic schema for config/data.yaml — the only source of data parameters ([C2])."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str
    filename_template: str


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_observations: int = Field(gt=0)
    max_jump_sigma: float = Field(gt=0)
    reversal_tolerance: float = Field(ge=0, le=1)
    robust_sigma_constant: float = Field(gt=0)
    min_level_points_for_jump_check: int = Field(ge=3)


class CouponDropConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_abs_return: float = Field(gt=0)
    max_abs_return: float = Field(gt=0)
    min_spacing_calendar_days: int = Field(gt=0)
    max_spacing_calendar_days: int = Field(gt=0)
    min_events_per_year: float = Field(gt=0)
    days_per_year: float = Field(gt=0)
    min_candidates: int = Field(ge=2)
    max_unspaced_gaps: int = Field(ge=0)

    @model_validator(mode="after")
    def _band_ordered(self) -> CouponDropConfig:
        if self.max_abs_return <= self.min_abs_return:
            raise ValueError("max_abs_return must exceed min_abs_return")
        if self.max_spacing_calendar_days < self.min_spacing_calendar_days:
            raise ValueError("max_spacing_calendar_days must be >= min_spacing_calendar_days")
        return self


class DiscontinuedSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discontinued_on: date
    reason: str
    substitutes: list[str]


class FredSeriesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    frequency: Literal["daily", "weekly"]
    series_type: Literal["oas", "yield", "spread", "rate", "index"]
    unit: str
    non_negative: bool
    publication_lag_days: int = Field(ge=0)
    publication_weekday: (
        Literal["monday", "tuesday", "wednesday", "thursday", "friday"] | None
    ) = None
    primary: bool = False

    @model_validator(mode="after")
    def _weekly_release_rules(self) -> FredSeriesSpec:
        if self.frequency != "weekly":
            return self
        if self.publication_weekday is None:
            raise ValueError("weekly FRED series require publication_weekday")
        if self.publication_lag_days < 1:
            raise ValueError("weekly FRED series require publication_lag_days > 0")
        return self


class EtfSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    auto_adjust: bool

    @field_validator("auto_adjust")
    @classmethod
    def _require_total_return(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "Bond ETFs distribute coupons monthly. auto_adjust must be True "
                "so GARCH is not fed spurious ex-dividend price drops."
            )
        return value


class EtfDownloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # yfinance.download treats ``end`` as exclusive; FRED does not.
    end_date_exclusive: bool
    end_exclusive_shift_days: int = Field(ge=0)


class EbpDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    date_column: str
    ebp_column: str
    gz_spread_column: str
    recession_prob_column: str | None
    cache_id: str
    publication_lag_days: int = Field(ge=1)
    publication_lag_anchor: Literal["month_end"]
    sensitivity_lags_days: list[int] = Field(min_length=1)
    sensitivity_min_correlation: float = Field(gt=0, le=1)
    aggregation: Literal["average"]
    vix_series_id: str
    hy_oas_series_id: str
    t10y2y_series_id: str


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: date
    end_date: date | None
    timezone: str
    fred_api_key_env: str
    cache: CacheConfig
    validation: ValidationConfig
    etf_coupon_drop_detection: CouponDropConfig
    discontinued_series: dict[str, DiscontinuedSeries]
    fred_series: dict[str, FredSeriesSpec]
    etf_tickers: dict[str, EtfSpec]
    etf_download: EtfDownloadConfig
    ebp: EbpDataConfig | None = None


def load_data_config(path: str | Path) -> DataConfig:
    """Load and validate a YAML data-universe file."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return DataConfig.model_validate(payload)
