"""Estimation-sample integrity checks that do not live on the FRED loader.

A forward-filled holiday prints as an exact zero return after differencing.
GARCH then treats a mechanical gap as a genuine shock of size zero, which
biases α up and persistence with it ([D1]). Two consecutive zeros can be a
true quiet spell; three is the fill signature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "assert_no_stale_zero_returns",
    "longest_exact_zero_run",
    "mask_stale_price_runs",
]


def longest_exact_zero_run(series: pd.Series) -> int:
    """Length of the longest consecutive exact-zero spell after dropping NaN."""
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    longest = 0
    run = 0
    for value in values:
        if value == 0.0:
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    return int(longest)


def mask_stale_price_runs(
    prices: pd.Series, *, max_zero_return_run: int, tolerance: float
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Mask vendor-repeated prints before returns are taken.

    A run of identical prices longer than ``max_zero_return_run`` returns is
    a stale feed, not a quiet market. The first print of the run is kept;
    the repeats become NaN so the next real print carries the full move.

    Returns the masked series and the dates that were masked.
    """
    valid = prices.dropna().astype("float64")
    same = valid.diff().abs() <= tolerance
    run_id = (~same).cumsum()
    run_len = same.groupby(run_id).transform("sum")
    stale = same & (run_len > max_zero_return_run)
    masked_dates = pd.DatetimeIndex(stale.index[stale.to_numpy()])
    out = prices.astype("float64").copy()
    out.loc[masked_dates] = np.nan
    return out, masked_dates


def assert_no_stale_zero_returns(series: pd.Series, *, max_run: int = 2) -> int:
    """Fail loudly if the series looks forward-filled before differencing.

    Parameters
    ----------
    series
        Estimation returns (simple, log-diff, or Δlevel).
    max_run
        Longest exact-zero spell that is still allowed. Production default is 2.
    """
    n = longest_exact_zero_run(series)
    if n > max_run:
        raise ValueError(
            f"exact-zero return run of {n} days exceeds {max_run}; "
            "signature of forward-fill before differencing"
        )
    return n
