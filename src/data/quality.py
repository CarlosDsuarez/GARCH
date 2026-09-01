"""Estimation-sample integrity checks that do not live on the FRED loader.

A forward-filled holiday prints as an exact zero return after differencing.
GARCH then treats a mechanical gap as a genuine shock of size zero, which
biases α up and persistence with it ([D1]). Two consecutive zeros can be a
true quiet spell; three is the fill signature.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "assert_no_stale_zero_returns",
    "longest_exact_zero_run",
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
