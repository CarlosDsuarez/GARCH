"""Behaviour tests for credit data ingestion: [D1], [D3], ETF total return, cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import (
    FakeEtfDownloader,
    FakeFredClient,
    PRODUCTION_CONFIG,
    make_test_config,
)
from data.credit_loader import (
    CreditDataLoader,
    SeriesValidationError,
    detect_spurious_coupon_drops,
    load_data_config,
)


REQUIRED_FRED = [
    "BAMLH0A0HYM2",
    "BAMLC0A0CM",
    "BAMLC0A4CBBB",
    "BAMLH0A3HYC",
    "BAMLEMCBPIOAS",
    "BAMLH0A0HYM2EY",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "T10YIE",
    "DFII10",
    "VIXCLS",
    "NFCI",
    "NFCICREDIT",
    "STLFSI4",
]
REQUIRED_ETF = ["HYG", "JNK", "LQD", "EMB", "AGG", "IEF", "TLT", "SHY", "BKLN"]


def _daily_levels(
    start: str = "2020-01-02",
    periods: int = 80,
    level: float = 5.0,
    sigma: float = 0.02,
    seed: int = 7,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    changes = rng.normal(0.0, sigma, size=periods)
    values = level + np.cumsum(changes)
    return pd.Series(values, index=idx, name="BAMLH0A0HYM2")


def _loader(tmp_path: Path, **kwargs) -> CreditDataLoader:
    config = make_test_config(tmp_path)
    return CreditDataLoader(
        config,
        project_root=tmp_path,
        fred_client=kwargs.get("fred_client"),
        etf_downloader=kwargs.get("etf_downloader"),
    )


def _window(series: pd.Series) -> tuple[str, str]:
    return (
        series.index.min().strftime("%Y-%m-%d"),
        series.index.max().strftime("%Y-%m-%d"),
    )


# ---------------------------------------------------------------------------
# Production config universe
# ---------------------------------------------------------------------------


def test_production_config_contains_full_universe() -> None:
    config = load_data_config(PRODUCTION_CONFIG)
    assert set(REQUIRED_FRED) <= set(config.fred_series)
    assert set(REQUIRED_ETF) <= set(config.etf_tickers)
    assert "TEDRATE" not in config.fred_series
    assert "TEDRATE" in config.discontinued_series
    assert config.validation.min_observations >= 1000
    assert config.validation.max_jump_sigma == 10.0
    assert all(spec.auto_adjust for spec in config.etf_tickers.values())
    assert config.etf_download.end_date_exclusive is True
    weekly = {sid for sid, spec in config.fred_series.items() if spec.frequency == "weekly"}
    assert weekly == {"NFCI", "NFCICREDIT", "STLFSI4"}
    assert config.fred_series["NFCI"].publication_lag_days > 0
    assert config.fred_series["BAMLH0A0HYM2"].primary is True
    substitutes = config.discontinued_series["TEDRATE"].substitutes
    assert "NFCI" in substitutes


def test_fetch_fred_rejects_discontinued_tedrate(tmp_path: Path) -> None:
    loader = _loader(tmp_path, fred_client=FakeFredClient({}))
    with pytest.raises(ValueError, match="TEDRATE"):
        loader.fetch_fred(["TEDRATE"], start="2020-01-01", end="2020-12-31")


def test_fetch_fred_rejects_unknown_series_id(tmp_path: Path) -> None:
    loader = _loader(tmp_path, fred_client=FakeFredClient({}))
    with pytest.raises(ValueError, match="not in the configured"):
        loader.fetch_fred(["NOT_A_SERIES"], start="2020-01-01", end="2020-12-31")


def test_weekly_series_require_positive_publication_lag() -> None:
    from pydantic import ValidationError

    from data.schema import FredSeriesSpec

    with pytest.raises(ValidationError, match="publication_lag_days"):
        FredSeriesSpec(
            description="bad weekly",
            frequency="weekly",
            series_type="index",
            unit="index",
            non_negative=False,
            publication_lag_days=0,
            publication_weekday="wednesday",
        )


# ---------------------------------------------------------------------------
# [D1] no forward-fill on daily levels before differencing
# ---------------------------------------------------------------------------


def test_level_to_changes_drops_missing_dates_and_does_not_inject_zeros(
    tmp_path: Path,
) -> None:
    """Holiday NaNs must be dropped, not ffilled, before Δy_t is formed."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    level = pd.Series([5.0, np.nan, 5.2], index=idx, name="BAMLH0A0HYM2")
    loader = _loader(tmp_path)

    changes, n_dropped = loader.level_to_changes(level)

    assert n_dropped == 1
    assert len(changes) == 1
    assert changes.iloc[0] == pytest.approx(0.2)
    assert not np.any(np.isclose(changes.to_numpy(), 0.0))


def test_forward_fill_control_would_inject_a_zero_return() -> None:
    """Sanity check that the forbidden ffill path is what produces exact zeros."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    level = pd.Series([5.0, np.nan, 5.2], index=idx)
    ffilled_changes = level.ffill().diff().dropna()
    assert (ffilled_changes == 0.0).any()


def test_weekly_align_uses_asof_backward_and_honours_publication_lag(
    tmp_path: Path,
) -> None:
    """NFCI dated Wednesday T is available on T+7, never earlier ([C1], [D2])."""
    weekly = pd.Series(
        [1.0, 2.0],
        index=pd.to_datetime(["2024-01-03", "2024-01-10"]),
        name="NFCI",
    )
    loader = _loader(tmp_path)
    lagged = loader.apply_publication_lag(weekly, lag_days=7)
    daily_index = pd.bdate_range("2024-01-03", "2024-01-18")
    aligned = loader.align_weekly_to_daily(lagged, daily_index)

    jan9 = pd.Timestamp("2024-01-09")
    jan10 = pd.Timestamp("2024-01-10")
    jan16 = pd.Timestamp("2024-01-16")
    jan17 = pd.Timestamp("2024-01-17")
    assert jan9 in aligned.index
    assert pd.isna(aligned.loc[jan9])
    assert aligned.loc[jan10] == pytest.approx(1.0)
    assert aligned.loc[jan16] == pytest.approx(1.0)
    assert aligned.loc[jan17] == pytest.approx(2.0)


def test_align_weekly_never_propagates_future_values(tmp_path: Path) -> None:
    weekly = pd.Series(
        [9.0],
        index=pd.to_datetime(["2024-01-10"]),
        name="NFCI",
    )
    loader = _loader(tmp_path)
    daily_index = pd.bdate_range("2024-01-03", "2024-01-12")
    aligned = loader.align_weekly_to_daily(weekly, daily_index)
    before = aligned.loc[: pd.Timestamp("2024-01-09")]
    assert before.isna().all()
    assert aligned.loc[pd.Timestamp("2024-01-10")] == pytest.approx(9.0)


def test_fetch_fred_backs_up_start_by_publication_lag(tmp_path: Path) -> None:
    """NFCI dated T is not public until T+L; fetch must request [start-L, end]."""
    weekly = pd.Series(
        [0.1, 0.2, 0.3],
        index=pd.to_datetime(["2019-12-25", "2020-01-01", "2020-01-08"]),
        name="NFCI",
    )
    fred = FakeFredClient({"NFCI": weekly})
    loader = _loader(tmp_path, fred_client=fred)
    frame = loader.fetch_fred(["NFCI"], start="2020-01-08", end="2020-01-20")
    request = fred.requests[0]
    lag = loader.config.fred_series["NFCI"].publication_lag_days
    assert pd.Timestamp(request["observation_start"]) <= pd.Timestamp("2020-01-08") - pd.Timedelta(
        days=lag
    )
    assert frame.loc[pd.Timestamp("2020-01-08"), "NFCI"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# [D3] validate_series
# ---------------------------------------------------------------------------


def test_validate_series_rejects_negative_oas(tmp_path: Path) -> None:
    series = _daily_levels()
    series.iloc[10] = -0.15
    loader = _loader(tmp_path)
    with pytest.raises(SeriesValidationError, match="negative"):
        loader.validate_series(series, series_id="BAMLH0A0HYM2")


def test_validate_series_allows_negative_curve_slope(tmp_path: Path) -> None:
    series = _daily_levels(level=-0.4, sigma=0.02)
    series.name = "T10Y2Y"
    loader = _loader(tmp_path)
    result = loader.validate_series(series, series_id="T10Y2Y")
    assert result.is_valid is True


def test_validate_series_rejects_reversing_ten_sigma_jump(tmp_path: Path) -> None:
    series = _daily_levels(periods=200, sigma=0.02, seed=3)
    series.iloc[80] = series.iloc[79] + 5.0
    series.iloc[81] = series.iloc[79]
    loader = _loader(tmp_path)
    with pytest.raises(SeriesValidationError, match="revers"):
        loader.validate_series(series, series_id="BAMLH0A0HYM2")


def test_validate_series_flags_reversing_jump_on_otherwise_constant_series(
    tmp_path: Path,
) -> None:
    """MAD of Δy is 0 when the series is flat; a print-error reversal must still fail."""
    idx = pd.bdate_range("2020-01-02", periods=80)
    values = np.full(80, 5.0)
    values[40] = 8.0
    values[41] = 5.0
    series = pd.Series(values, index=idx, name="BAMLH0A0HYM2")
    loader = _loader(tmp_path)
    with pytest.raises(SeriesValidationError, match="revers"):
        loader.validate_series(series, series_id="BAMLH0A0HYM2")


def test_validate_series_allows_non_reversing_jump(tmp_path: Path) -> None:
    series = _daily_levels(periods=200, sigma=0.02, seed=3)
    series.iloc[80:] = series.iloc[80:] + 5.0
    loader = _loader(tmp_path)
    result = loader.validate_series(series, series_id="BAMLH0A0HYM2")
    assert result.is_valid is True


def test_validate_series_rejects_duplicate_index(tmp_path: Path) -> None:
    series = _daily_levels(periods=60)
    dup = pd.concat([series, series.iloc[[-1]]])
    loader = _loader(tmp_path)
    with pytest.raises(SeriesValidationError, match="duplicate"):
        loader.validate_series(dup, series_id="BAMLH0A0HYM2")


def test_validate_series_rejects_insufficient_observations(tmp_path: Path) -> None:
    series = _daily_levels(periods=20)
    loader = _loader(tmp_path)
    with pytest.raises(SeriesValidationError, match="observations"):
        loader.validate_series(series, series_id="BAMLH0A0HYM2")


def test_validate_series_accepts_clean_oas(tmp_path: Path) -> None:
    series = _daily_levels(periods=80, sigma=0.03)
    loader = _loader(tmp_path)
    result = loader.validate_series(series, series_id="BAMLH0A0HYM2")
    assert result.is_valid is True
    assert result.n_obs == 80
    assert result.start == series.index.min().date()
    assert result.end == series.index.max().date()


# ---------------------------------------------------------------------------
# Cache ([D4], [C3])
# ---------------------------------------------------------------------------


def test_cache_roundtrip_stores_utc_timestamp_and_hash(tmp_path: Path) -> None:
    series = _daily_levels()
    start, end = _window(series)
    fred = FakeFredClient({"BAMLH0A0HYM2": series})
    loader = _loader(tmp_path, fred_client=fred)

    fetched = loader.fetch_fred(["BAMLH0A0HYM2"], start=start, end=end)
    cached = loader.load_cached("BAMLH0A0HYM2")

    pd.testing.assert_series_equal(
        fetched["BAMLH0A0HYM2"].dropna(),
        cached.dropna(),
        check_names=False,
    )
    assert "retrieved_at_utc" in cached.attrs
    assert "content_hash" in cached.attrs
    assert cached.attrs["n_observations"] == int(series.notna().sum())
    assert str(cached.attrs["retrieved_at_utc"]).endswith("Z")


def test_cache_hit_skips_api_without_force_refresh(tmp_path: Path) -> None:
    series = _daily_levels()
    start, end = _window(series)
    fred = FakeFredClient({"BAMLH0A0HYM2": series})
    loader = _loader(tmp_path, fred_client=fred)
    loader.fetch_fred(["BAMLH0A0HYM2"], start=start, end=end)
    n_first = len(fred.calls)
    loader.fetch_fred(["BAMLH0A0HYM2"], start=start, end=end)
    assert len(fred.calls) == n_first


def test_force_refresh_redownloads(tmp_path: Path) -> None:
    series = _daily_levels()
    start, end = _window(series)
    fred = FakeFredClient({"BAMLH0A0HYM2": series})
    loader = _loader(tmp_path, fred_client=fred)
    loader.fetch_fred(["BAMLH0A0HYM2"], start=start, end=end)
    loader.fetch_fred(
        ["BAMLH0A0HYM2"],
        start=start,
        end=end,
        force_refresh=True,
    )
    assert fred.calls.count("BAMLH0A0HYM2") == 2


def test_cache_miss_when_requested_window_exceeds_cached_range(tmp_path: Path) -> None:
    short = _daily_levels(periods=20)
    long = _daily_levels(periods=80)
    fred = FakeFredClient({"BAMLH0A0HYM2": short})
    loader = _loader(tmp_path, fred_client=fred)
    short_start, short_end = _window(short)
    loader.fetch_fred(["BAMLH0A0HYM2"], start=short_start, end=short_end)
    n_first = len(fred.calls)
    fred.store["BAMLH0A0HYM2"] = long
    _, long_end = _window(long)
    loader.fetch_fred(["BAMLH0A0HYM2"], start=short_start, end=long_end)
    assert len(fred.calls) == n_first + 1


# ---------------------------------------------------------------------------
# ETF total-return adjustment
# ---------------------------------------------------------------------------


def _etf_ohlcv(prices: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": 1_000_000,
        }
    )


def test_fetch_etf_passes_auto_adjust_true(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-02", periods=80)
    prices = pd.Series(np.linspace(80.0, 85.0, len(idx)), index=idx)
    downloader = FakeEtfDownloader({"HYG": _etf_ohlcv(prices)})
    loader = _loader(tmp_path, etf_downloader=downloader)

    frame = loader.fetch_etf(["HYG"], start="2020-01-02", end="2020-02-28")

    assert downloader.calls[0]["auto_adjust"] is True
    assert "HYG" in frame.columns
    assert (frame["HYG"] > 0).all()


def test_fetch_etf_includes_requested_end_date(tmp_path: Path) -> None:
    """yfinance end is exclusive; the loader must still return the requested last session."""
    idx = pd.bdate_range("2020-01-02", periods=80)
    prices = pd.Series(np.linspace(80.0, 85.0, len(idx)), index=idx)
    downloader = FakeEtfDownloader({"HYG": _etf_ohlcv(prices)})
    loader = _loader(tmp_path, etf_downloader=downloader)
    last = idx[20]
    frame = loader.fetch_etf(
        ["HYG"], start="2020-01-02", end=last.strftime("%Y-%m-%d")
    )
    assert last in frame.index
    assert last.strftime("%Y-%m-%d") < downloader.calls[0]["end"]


def test_unadjusted_bond_etf_returns_exhibit_monthly_coupon_drop_pattern(
    tmp_path: Path,
) -> None:
    """Control: raw Close with monthly ~0.45% drops must be flagged."""
    loader = _loader(tmp_path)
    idx = pd.bdate_range("2020-01-02", periods=520)
    prices = [100.0]
    for i in range(1, len(idx)):
        if i % 21 == 0:
            prices.append(prices[-1] * (1.0 - 0.0045))
        else:
            prices.append(prices[-1] * (1.0 + 0.0002))
    returns = pd.Series(prices, index=idx).pct_change()
    report = detect_spurious_coupon_drops(
        returns, spec=loader.config.etf_coupon_drop_detection
    )
    assert report.flagged is True
    assert report.events_per_year >= loader.config.etf_coupon_drop_detection.min_events_per_year


def test_adjusted_total_return_series_has_no_monthly_coupon_drop_pattern(
    tmp_path: Path,
) -> None:
    """auto_adjust=True / Adj Close must not look like monthly coupon strips."""
    loader = _loader(tmp_path)
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-02", periods=520)
    rumps = rng.normal(0.0003, 0.0025, size=len(idx))
    prices = 100.0 * np.cumprod(1.0 + rumps)
    returns = pd.Series(prices, index=idx).pct_change()
    report = detect_spurious_coupon_drops(
        returns, spec=loader.config.etf_coupon_drop_detection
    )
    assert report.flagged is False


def test_fetch_etf_rejects_unadjusted_coupon_drop_pattern(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-02", periods=520)
    prices = [100.0]
    for i in range(1, len(idx)):
        if i % 21 == 0:
            prices.append(prices[-1] * (1.0 - 0.0045))
        else:
            prices.append(prices[-1] * (1.0 + 0.0002))
    ohlcv = _etf_ohlcv(pd.Series(prices, index=idx))
    downloader = FakeEtfDownloader({"HYG": ohlcv})
    loader = _loader(tmp_path, etf_downloader=downloader)
    with pytest.raises(SeriesValidationError, match="coupon"):
        loader.fetch_etf(["HYG"], start="2020-01-02", end="2022-01-31")


def test_coverage_report_records_nan_drop_share(tmp_path: Path) -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    series = pd.Series([5.0, np.nan, 5.1, 5.2], index=idx, name="BAMLH0A0HYM2")
    fred = FakeFredClient({"BAMLH0A0HYM2": series})
    loader = _loader(tmp_path, fred_client=fred)
    frame = loader.fetch_fred(
        ["BAMLH0A0HYM2"], start="2024-01-02", end="2024-01-05"
    )
    report = loader.coverage_report(frame)
    row = report.set_index("series_id").loc["BAMLH0A0HYM2"]
    assert row["n_raw"] == 4
    assert row["n_obs"] == 3
    assert row["n_nan_dropped"] == 1
    assert row["pct_nan_dropped"] == pytest.approx(25.0)
