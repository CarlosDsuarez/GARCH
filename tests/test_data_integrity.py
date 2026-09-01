"""[T4] Data integrity: OAS floors, coupon drops, zero runs, as-of merges."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import FakeFredClient, make_test_config
from data.credit_loader import (
    CreditDataLoader,
    SeriesValidationError,
    detect_spurious_coupon_drops,
)
from data.quality import assert_no_stale_zero_returns, longest_exact_zero_run
from helpers import FIXTURE_DIR, ensure_synthetic_fixtures


def _loader(tmp_path: Path, **kwargs) -> CreditDataLoader:
    config = make_test_config(tmp_path)
    return CreditDataLoader(
        config,
        project_root=tmp_path,
        fred_client=kwargs.get("fred_client"),
        etf_downloader=kwargs.get("etf_downloader"),
    )


def test_t4_1_oas_series_has_no_negatives_after_ingest(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-02", periods=60)
    oas = pd.Series(5.0 + 0.01 * np.arange(60), index=idx, name="BAMLH0A0HYM2")
    assert (oas >= 0).all()
    fred = FakeFredClient({"BAMLH0A0HYM2": oas})
    loader = _loader(tmp_path, fred_client=fred)
    frame = loader.fetch_fred(["BAMLH0A0HYM2"], start="2020-01-02", end="2020-03-31")
    assert (frame["BAMLH0A0HYM2"].dropna() >= 0).all()


def test_t4_1_negative_oas_is_rejected(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-02", periods=60)
    oas = pd.Series(5.0, index=idx, name="BAMLH0A0HYM2")
    oas.iloc[10] = -0.5
    loader = _loader(tmp_path)
    with pytest.raises(SeriesValidationError, match="negative"):
        loader.validate_series(oas, series_id="BAMLH0A0HYM2")


def test_t4_2_adjusted_etf_returns_are_not_monthly_coupon_strips(tmp_path: Path) -> None:
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


def test_t4_2_raw_close_with_monthly_drops_is_flagged(tmp_path: Path) -> None:
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


def test_t4_3_three_consecutive_zero_returns_are_the_forward_fill_signature() -> None:
    idx = pd.bdate_range("2021-01-04", periods=10)
    r = pd.Series([0.01, 0.0, 0.0, 0.0, 0.02, -0.01, 0.0, 0.0, 0.03, 0.01], index=idx)
    assert longest_exact_zero_run(r) == 3
    with pytest.raises(ValueError, match="zero"):
        assert_no_stale_zero_returns(r, max_run=2)


def test_t4_3_two_consecutive_zeros_are_allowed() -> None:
    idx = pd.bdate_range("2021-01-04", periods=8)
    r = pd.Series([0.01, 0.0, 0.0, 0.02, -0.01, 0.0, 0.03, 0.01], index=idx)
    assert longest_exact_zero_run(r) == 2
    assert_no_stale_zero_returns(r, max_run=2)


def test_t4_3_level_to_changes_rejects_a_three_zero_forward_fill(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    with pytest.raises(SeriesValidationError, match="zero"):
        loader.level_to_changes(pd.Series([5.0] * 6, index=pd.bdate_range("2024-01-02", periods=6)))


def test_t4_4_asof_merge_never_propagates_a_future_release_backward(tmp_path: Path) -> None:
    """Deliberately misaligned weekly dates: Wednesday T is not knowable on Tuesday."""
    ensure_synthetic_fixtures()
    weekly = pd.read_parquet(FIXTURE_DIR / "asof_misaligned.parquet")["NFCI"]
    loader = _loader(tmp_path)
    lagged = loader.apply_publication_lag(weekly, lag_days=7)
    daily_index = pd.bdate_range("2024-01-03", "2024-01-25")
    aligned = loader.align_weekly_to_daily(lagged, daily_index)
    # First print (2024-01-03) is public on 2024-01-10.
    assert pd.isna(aligned.loc[pd.Timestamp("2024-01-09")])
    assert aligned.loc[pd.Timestamp("2024-01-10")] == pytest.approx(1.0)
    # Second print (2024-01-17) must not leak onto 2024-01-16.
    assert aligned.loc[pd.Timestamp("2024-01-16")] == pytest.approx(1.0)
    assert aligned.loc[pd.Timestamp("2024-01-24")] == pytest.approx(9.0)
    before_second = aligned.loc[: pd.Timestamp("2024-01-23")]
    assert (before_second.dropna() == 9.0).sum() == 0
