"""Estimation-sample integrity checks that do not live on the FRED loader.

A forward-filled holiday prints as an exact zero return after differencing.
GARCH then treats a mechanical gap as a genuine shock of size zero, which
biases α up and persistence with it ([D1]). Two consecutive zeros can be a
true quiet spell; three is the fill signature.

Series printed on a coarse grid (OAS to 1 bp) are different: low-spread IG
OAS is unchanged on ~40% of days, so long genuine zero runs are expected.
``stale_zero_run_bound`` widens the limit only for such quantized series.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = [
    "assert_no_stale_zero_returns",
    "longest_exact_zero_run",
    "mask_stale_price_runs",
    "stale_zero_run_bound",
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


def stale_zero_run_bound(
    series: pd.Series,
    *,
    floor: int,
    quantized_min_zero_share: float,
    sd_multiplier: float,
) -> int:
    """Longest exact-zero run a quantized, un-filled series can plausibly show.

    For :math:`n` draws with zero share :math:`p`, the longest zero run
    :math:`L` satisfies (Schilling, 1990)

    .. math::

        \\mathbb{E}L \\approx \\log_{1/p}\\bigl(n(1-p)\\bigr), \\qquad
        \\mathrm{Var}\\,L \\approx \\frac{\\pi^2}{6\\ln^2(1/p)} + \\frac{1}{12}.

    The bound is :math:`\\lceil \\mathbb{E}L + k\\,\\mathrm{sd}(L) \\rceil`, never
    below ``floor``. Below ``quantized_min_zero_share`` the series is treated
    as continuous and the bound is ``floor``: there, zeros are what a
    forward fill leaves behind, so they must not loosen their own test.

    References
    ----------
    Schilling, M. F. (1990). The Longest Run of Heads. *The College
    Mathematics Journal*, 21(3), 196–207.
    """
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    n = values.size
    if n == 0:
        return int(floor)
    p = float(np.mean(values == 0.0))
    if p < quantized_min_zero_share or p >= 1.0:
        return int(floor)
    log_inv_p = math.log(1.0 / p)
    expected = math.log(n * (1.0 - p)) / log_inv_p
    sd = math.sqrt(math.pi**2 / (6.0 * log_inv_p**2) + 1.0 / 12.0)
    return max(int(floor), int(math.ceil(expected + sd_multiplier * sd)))


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
