"""Shared fixtures for credit-data loader tests.

Synthetic series use ISO dates (YYYY-MM-DD) and invented levels
(e.g. OAS = 5.0). No production market data is stored here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.schema import DataConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = REPO_ROOT / "config" / "data.yaml"


def _fred_spec(
    *,
    series_type: str = "oas",
    frequency: str = "daily",
    non_negative: bool = True,
    publication_lag_days: int = 0,
    publication_weekday: str | None = None,
    primary: bool = False,
    description: str = "synthetic",
    unit: str = "percent",
) -> dict:
    spec: dict = {
        "description": description,
        "frequency": frequency,
        "series_type": series_type,
        "unit": unit,
        "non_negative": non_negative,
        "publication_lag_days": publication_lag_days,
        "primary": primary,
    }
    if publication_weekday is not None:
        spec["publication_weekday"] = publication_weekday
    return spec


def make_test_config(tmp_path: Path, **overrides) -> DataConfig:
    """Minimal validated config pointed at a temporary cache directory."""
    payload = {
        "start_date": date(2020, 1, 1),
        "end_date": date(2024, 12, 31),
        "timezone": "America/New_York",
        "fred_api_key_env": "FRED_API_KEY",
        "cache": {
            "directory": str(tmp_path / "cache"),
            "filename_template": "{series_id}.parquet",
        },
        "validation": {
            "min_observations": 50,
            "max_jump_sigma": 10.0,
            "reversal_tolerance": 0.10,
            "robust_sigma_constant": 1.4826,
            "min_level_points_for_jump_check": 3,
        },
        "etf_coupon_drop_detection": {
            "min_abs_return": 0.003,
            "max_abs_return": 0.006,
            "min_spacing_calendar_days": 15,
            "max_spacing_calendar_days": 35,
            "min_events_per_year": 8.0,
            "days_per_year": 365.25,
            "min_candidates": 2,
            "max_unspaced_gaps": 2,
        },
        "discontinued_series": {
            "TEDRATE": {
                "discontinued_on": date(2022, 1, 31),
                "reason": "LIBOR-SOFR transition",
                "substitutes": ["NFCI", "NFCICREDIT"],
            }
        },
        "fred_series": {
            "BAMLH0A0HYM2": _fred_spec(primary=True, description="HY OAS"),
            "T10Y2Y": _fred_spec(
                series_type="spread",
                non_negative=False,
                description="curve slope",
            ),
            "NFCI": _fred_spec(
                series_type="index",
                frequency="weekly",
                non_negative=False,
                publication_lag_days=7,
                publication_weekday="wednesday",
                unit="index",
                description="NFCI",
            ),
        },
        "etf_tickers": {
            "HYG": {
                "description": "HY ETF",
                "auto_adjust": True,
            }
        },
        "etf_download": {
            "end_date_exclusive": True,
            "end_exclusive_shift_days": 1,
        },
    }
    payload.update(overrides)
    return DataConfig.model_validate(payload)


class FakeFredClient:
    """In-memory FRED stand-in. Keys are series ids; values are daily/weekly Series."""

    def __init__(self, store: dict[str, pd.Series]) -> None:
        self.store = store
        self.calls: list[str] = []
        self.requests: list[dict] = []

    def get_series(
        self,
        series_id: str,
        observation_start=None,
        observation_end=None,
    ) -> pd.Series:
        self.calls.append(series_id)
        self.requests.append(
            {
                "series_id": series_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
            }
        )
        series = self.store[series_id].copy()
        if observation_start is not None:
            series = series.loc[pd.Timestamp(observation_start) :]
        if observation_end is not None:
            series = series.loc[: pd.Timestamp(observation_end)]
        return series


class FakeEtfDownloader:
    """Captures yfinance kwargs. ``end`` is exclusive, matching yfinance.download."""

    def __init__(self, store: dict[str, pd.DataFrame]) -> None:
        self.store = store
        self.calls: list[dict] = []

    def __call__(self, tickers, start=None, end=None, **kwargs) -> pd.DataFrame:
        self.calls.append({"tickers": tickers, "start": start, "end": end, **kwargs})
        key = tickers if isinstance(tickers, str) else tickers[0]
        frame = self.store[key].copy()
        if start is not None:
            frame = frame.loc[pd.Timestamp(start) :]
        if end is not None:
            frame = frame.loc[frame.index < pd.Timestamp(end)]
        return frame


@pytest.fixture
def repo_config_path() -> Path:
    return PRODUCTION_CONFIG


@pytest.fixture(scope="session", autouse=True)
def persist_synthetic_fixtures() -> None:
    from helpers import ensure_synthetic_fixtures

    ensure_synthetic_fixtures()
