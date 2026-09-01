"""Ingestion, validation, and timestamped cache for credit and volatility series.

This module is the only production entry point for market data. Equity-style
shortcuts (forward-fill on levels, raw ETF closes, TEDRATE) are rejected.

TEDRATE substitution
--------------------
FRED discontinued TEDRATE in January 2022 after the LIBOR→SOFR transition.
This loader refuses TEDRATE. Funding/credit-stress substitutes are NFCI and
NFCICREDIT (see ``discontinued_series`` in ``config/data.yaml``). A hand-built
SOFR–OIS spread may be added later; it is not a FRED series in this universe.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.quality import assert_no_stale_zero_returns
from data.schema import CouponDropConfig, DataConfig, load_data_config

logger = logging.getLogger(__name__)

__all__ = [
    "CouponDropReport",
    "CreditDataLoader",
    "SeriesValidationError",
    "ValidationResult",
    "detect_spurious_coupon_drops",
    "load_data_config",
]


class SeriesValidationError(ValueError):
    """Raised when a mandatory integrity check fails ([C7], [D3])."""

    def __init__(
        self,
        message: str,
        *,
        issues: list[str] | None = None,
        result: ValidationResult | None = None,
    ) -> None:
        super().__init__(message)
        self.issues = issues or [message]
        self.result = result


@dataclass(frozen=True)
class ValidationResult:
    series_id: str
    n_obs: int
    n_dropped_nan: int
    start: date | None
    end: date | None
    is_valid: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class CouponDropReport:
    flagged: bool
    n_candidate_drops: int
    n_monthly_spaced: int
    events_per_year: float
    dates: tuple[pd.Timestamp, ...]


def detect_spurious_coupon_drops(
    returns: pd.Series,
    spec: CouponDropConfig,
) -> CouponDropReport:
    """Flag monthly coupon-sized price drops that contaminate GARCH shocks.

    Bond ETFs pay coupons monthly. Unadjusted close-to-close returns then
    contain a regular strip of negative observations in a narrow band
    :math:`[r_{\\min}, r_{\\max}]` (typically 30–60 bp) spaced about one
    calendar month apart:

    .. math::

        \\mathcal{C} = \\{ t : r_{\\min} \\le -R_t \\le r_{\\max} \\}

        \\widehat{\\lambda} = |\\mathcal{C}| / \\Delta T_{\\mathrm{years}}

    The series is flagged when :math:`\\widehat{\\lambda}` exceeds
    ``min_events_per_year`` *and* consecutive candidates are almost all
    spaced inside ``[min_spacing_calendar_days, max_spacing_calendar_days]``.
    A total-return series (yfinance ``auto_adjust=True`` / Adj Close) must
    not trigger this detector.

    Parameters
    ----------
    returns
        Simple returns :math:`R_t = P_t / P_{t-1} - 1`.
    spec
        Band and spacing thresholds from ``config/data.yaml``.

    References
    ----------
    Garbade, K. D. (1982). *Securities Markets*. McGraw-Hill. (accrued
    interest / ex-coupon price drop.)
    """
    realized = returns.dropna()
    empty = CouponDropReport(False, 0, 0, 0.0, ())
    if realized.empty:
        return empty

    candidates = realized[
        (realized <= -spec.min_abs_return) & (realized >= -spec.max_abs_return)
    ]
    n_cand = int(candidates.shape[0])
    span_days = (realized.index.max() - realized.index.min()).days
    years = span_days / spec.days_per_year if span_days > 0 else 0.0
    events_per_year = (n_cand / years) if years > 0 else 0.0

    n_monthly = 0
    flagged = False
    if n_cand >= spec.min_candidates:
        gaps = pd.Series(candidates.index).diff().dt.days.dropna()
        in_band = (gaps >= spec.min_spacing_calendar_days) & (
            gaps <= spec.max_spacing_calendar_days
        )
        n_monthly = int(in_band.sum())
        required_monthly = n_cand - spec.max_unspaced_gaps
        flagged = (
            events_per_year >= spec.min_events_per_year
            and required_monthly > 0
            and n_monthly >= required_monthly
        )
    return CouponDropReport(
        flagged=flagged,
        n_candidate_drops=n_cand,
        n_monthly_spaced=n_monthly,
        events_per_year=float(events_per_year),
        dates=tuple(pd.Timestamp(ts) for ts in candidates.index),
    )


def _hash_series(series: pd.Series) -> str:
    payload = series.to_csv(header=False, date_format="%Y-%m-%d", float_format="%.12g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_timestamp(value: date | datetime | str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value)


def _coerce_series(
    df: pd.Series | pd.DataFrame,
    series_id: str | None,
) -> pd.Series:
    if isinstance(df, pd.Series):
        series = df.copy()
        if series_id is not None:
            series.name = series_id
        return series
    if series_id is not None and series_id in df.columns:
        series = df[series_id].copy()
        series.name = series_id
        return series
    if df.shape[1] == 1:
        series = df.iloc[:, 0].copy()
        if series_id is not None:
            series.name = series_id
        return series
    raise ValueError("validate_series requires a Series or a single-column DataFrame")


class CreditDataLoader:
    """FRED + yfinance ingestion with parquet cache and integrity checks.

    Parameters
    ----------
    config
        Validated ``DataConfig`` (from ``config/data.yaml``).
    project_root
        Root used to resolve a relative cache directory.
    fred_client
        Object with ``get_series(series_id, observation_start, observation_end)``.
        Injected in tests; constructed from ``fredapi.Fred`` otherwise.
    etf_downloader
        Callable with the ``yfinance.download`` signature. Injected in tests.
    """

    def __init__(
        self,
        config: DataConfig,
        *,
        project_root: str | Path | None = None,
        fred_client: Any | None = None,
        etf_downloader: Callable[..., pd.DataFrame] | None = None,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        cache_dir = Path(config.cache.directory)
        self.cache_dir = cache_dir if cache_dir.is_absolute() else self.project_root / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fred_client = fred_client
        self._etf_downloader = etf_downloader

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> CreditDataLoader:
        config_path = Path(path)
        config = load_data_config(config_path)
        kwargs.setdefault("project_root", config_path.resolve().parent.parent)
        return cls(config, **kwargs)

    def cache_path(self, series_id: str) -> Path:
        filename = self.config.cache.filename_template.format(series_id=series_id)
        return self.cache_dir / filename

    def apply_publication_lag(self, series: pd.Series, lag_days: int) -> pd.Series:
        """Stamp a series with its first *availability* date, not the FRED reference date.

        .. math::

            y^{\\mathrm{avail}}_{t+L} = y^{\\mathrm{FRED}}_t

        where :math:`L` is ``publication_lag_days``. NFCI is released on Wednesday
        with data referring to the prior week (Chicago Fed), so using the FRED
        stamp on that Wednesday without a lag is look-ahead ([C1], [D1]).

        Parameters
        ----------
        series
            Native-frequency observations indexed by FRED's reference date.
        lag_days
            Calendar-day shift from configuration (must not be hard-coded by callers
            in production paths).
        """
        if lag_days < 0:
            raise ValueError("publication lag must be non-negative")
        out = series.copy()
        out.index = pd.DatetimeIndex(out.index) + pd.Timedelta(days=lag_days)
        return out

    def align_weekly_to_daily(
        self,
        weekly: pd.Series,
        daily_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Align a weekly series onto a daily calendar without look-ahead.

        Uses ``pandas.merge_asof(..., direction='backward')``:

        .. math::

            x^{\\mathrm{aligned}}_t = x_{\\tau(t)},
            \\qquad \\tau(t) = \\max\\{\\tau : \\tau \\le t\\}

        Forward as-of (``direction='forward'``) would leak a future release
        backward onto earlier days and is forbidden ([D2], [C1]).

        This is the *only* permitted forward-fill: it repeats a value that is
        already public, after the publication lag has been applied ([D1]
        exception for NFCI / NFCICREDIT / STLFSI4).
        """
        left = pd.DataFrame({"date": pd.DatetimeIndex(daily_index).sort_values()})
        right = (
            weekly.dropna()
            .sort_index()
            .rename("value")
            .rename_axis("date")
            .reset_index()
        )
        right["date"] = pd.to_datetime(right["date"])
        right = right.sort_values("date")
        if right.empty:
            return pd.Series(np.nan, index=daily_index, name=weekly.name)
        merged = pd.merge_asof(left, right, on="date", direction="backward")
        aligned = merged.set_index("date")["value"]
        aligned.name = weekly.name
        return aligned.reindex(daily_index)

    def level_to_changes(self, series: pd.Series) -> tuple[pd.Series, int]:
        """First-difference a level series after dropping missing observations.

        .. math::

            \\Delta y_t = y_t - y_{t-1}
            \\quad \\text{on} \\quad \\{t : y_t \\text{ is observed}\\}

        Forward-fill before differencing is forbidden. If a holiday is filled
        with :math:`y_{t-1}`, then :math:`\\Delta y_t = 0`. Those exact zeros
        bias :math:`\\mathrm{Var}(y)` downward and, in a GARCH(1,1)

        .. math::

            \\sigma_t^2 = \\omega + \\alpha \\varepsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2

        attribute a predictable mean-zero gap to the variance, inflating
        :math:`\\alpha` and producing spurious persistence ([D1]).

        The resulting calendar is irregular (weekend/holiday gaps remain as
        multi-day changes) but econometrically honest.

        References
        ----------
        Bollerslev, T. (1986). Generalized Autoregressive Conditional
        Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
        Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity
        with Estimates of the Variance of United Kingdom Inflation.
        *Econometrica*, 50(4), 987–1007.
        """
        n_dropped = int(series.isna().sum())
        cleaned = series.dropna()
        if cleaned.index.has_duplicates:
            raise SeriesValidationError("duplicate dates in level series")
        changes = cleaned.diff().dropna()
        try:
            assert_no_stale_zero_returns(changes)
        except ValueError as exc:
            raise SeriesValidationError(str(exc)) from exc
        start = cleaned.index.min()
        end = cleaned.index.max()
        logger.info(
            "level_to_changes series=%s n_obs=%s n_dropped_nan=%s window=%s/%s",
            series.name,
            int(cleaned.shape[0]),
            n_dropped,
            start,
            end,
        )
        return changes, n_dropped

    def validate_series(
        self,
        df: pd.Series | pd.DataFrame,
        *,
        series_id: str | None = None,
    ) -> ValidationResult:
        """Integrity gate for a level series ([D3]).

        Checks
        ------
        * no duplicate dates; index strictly increasing
        * no negatives when the series is specified ``non_negative``
          (OAS < 0 is impossible and indicates corruption)
        * no daily change exceeding ``max_jump_sigma`` robust standard
          deviations that fully reverses the next day (print-error pattern)
        * at least ``min_observations`` non-null points for GARCH use

        Robust scale for the jump screen (Hampel / MAD):

        .. math::

            \\hat{\\sigma} = c \\cdot \\mathrm{median}_t \\lvert \\Delta y_t
            - \\mathrm{median}_s(\\Delta y_s) \\rvert

        with :math:`c` from ``validation.robust_sigma_constant``. A jump at
        :math:`t` is a reversing error when

        .. math::

            \\lvert \\Delta y_t \\rvert > \\kappa \\hat{\\sigma}
            \\quad\\text{and}\\quad
            \\lvert \\Delta y_t + \\Delta y_{t+1} \\rvert
            \\le \\delta \\lvert \\Delta y_t \\rvert

        :math:`\\kappa` and :math:`\\delta` come from configuration. Real
        credit events that do not reverse (e.g. March 2020) pass.

        Fails loudly ([C7]): any issue raises ``SeriesValidationError``.

        References
        ----------
        Hampel, F. R. (1974). The Influence Curve and Its Role in Robust
        Estimation. *Journal of the American Statistical Association*,
        69(346), 383–393.
        """
        series = _coerce_series(df, series_id)
        sid = str(series_id or series.name or "unknown")
        issues: list[str] = []
        index = pd.DatetimeIndex(series.index)

        if index.has_duplicates:
            issues.append("duplicate dates in index")
        if not index.is_monotonic_increasing:
            issues.append("time index is not monotonic increasing")

        n_dropped = int(series.isna().sum())
        cleaned = series.dropna()
        n_obs = int(cleaned.shape[0])
        start = cleaned.index.min().date() if n_obs else None
        end = cleaned.index.max().date() if n_obs else None

        rules = self.config.validation
        if n_obs < rules.min_observations:
            issues.append(
                f"insufficient observations for GARCH: {n_obs} < {rules.min_observations}"
            )

        if self._must_be_non_negative(sid) and n_obs:
            n_neg = int((cleaned < 0).sum())
            if n_neg:
                issues.append(
                    f"negative values in non-negative series {sid}: {n_neg} points"
                )

        if n_obs >= rules.min_level_points_for_jump_check:
            issues.extend(self._reversing_jump_issues(cleaned, sid))

        is_valid = not issues
        result = ValidationResult(
            series_id=sid,
            n_obs=n_obs,
            n_dropped_nan=n_dropped,
            start=start,
            end=end,
            is_valid=is_valid,
            issues=tuple(issues),
        )
        logger.info(
            "validate_series id=%s n_obs=%s window=%s/%s valid=%s issues=%s",
            sid,
            n_obs,
            start,
            end,
            is_valid,
            list(issues),
        )
        if not is_valid:
            raise SeriesValidationError(
                "; ".join(issues),
                issues=issues,
                result=result,
            )
        return result

    def fetch_fred(
        self,
        series_ids: Sequence[str],
        start: date | datetime | str,
        end: date | datetime | str,
        *,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Download or cache FRED series. Daily series are *not* forward-filled."""
        columns: dict[str, pd.Series] = {}
        for series_id in series_ids:
            self._reject_discontinued(series_id)
            if series_id not in self.config.fred_series:
                raise ValueError(
                    f"{series_id} is not in the configured FRED universe"
                )
            series = self._get_or_download_fred(
                series_id, start, end, force_refresh=force_refresh
            )
            columns[series_id] = series
        frame = pd.DataFrame(columns)
        frame.index.name = "date"
        return frame

    def fetch_etf(
        self,
        tickers: Sequence[str],
        start: date | datetime | str,
        end: date | datetime | str,
        *,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Download or cache ETF *total-return* prices (``auto_adjust=True``).

        Using raw ``Close`` on monthly-distributing bond ETFs injects a
        ~30–60 bp drop on each ex-dividend date. GARCH reads that as a
        genuine volatility shock twelve times a year. This method always
        requests adjusted prices and runs ``detect_spurious_coupon_drops``
        on the resulting simple returns.
        """
        columns: dict[str, pd.Series] = {}
        for ticker in tickers:
            if ticker not in self.config.etf_tickers:
                raise ValueError(f"ETF {ticker} is not in the configured universe")
            spec = self.config.etf_tickers[ticker]
            series = self._get_or_download_etf(
                ticker, start, end, force_refresh=force_refresh, auto_adjust=spec.auto_adjust
            )
            prices = series.dropna()
            returns = prices.pct_change()
            coupon = detect_spurious_coupon_drops(
                returns, self.config.etf_coupon_drop_detection
            )
            if coupon.flagged:
                raise SeriesValidationError(
                    f"spurious monthly coupon-drop pattern in {ticker} returns; "
                    "series looks unadjusted (use auto_adjust=True / Adj Close)",
                    issues=[
                        f"coupon-drop pattern in {ticker}: "
                        f"events_per_year={coupon.events_per_year:.2f}"
                    ],
                )
            columns[ticker] = series
        frame = pd.DataFrame(columns)
        frame.index.name = "date"
        return frame

    def load_cached(self, series_id: str) -> pd.Series:
        """Read a cached series and restore retrieval metadata."""
        path = self.cache_path(series_id)
        if not path.exists():
            raise FileNotFoundError(f"no cache for {series_id}: {path}")
        table = pq.read_table(path)
        meta = _decode_metadata(table.schema.metadata)
        frame = table.to_pandas()
        frame["date"] = pd.to_datetime(frame["date"])
        series = frame.set_index("date")["value"].copy()
        series.name = series_id
        series.index = self._normalize_index(series.index)
        series.index.name = "date"
        series.attrs["retrieved_at_utc"] = meta["retrieved_at_utc"]
        series.attrs["content_hash"] = meta["content_hash"]
        series.attrs["n_observations"] = int(meta["n_observations"])
        series.attrs["first_obs"] = meta.get("first_obs", "")
        series.attrs["last_obs"] = meta.get("last_obs", "")
        series.attrs["n_raw"] = int(meta.get("n_raw", series.shape[0]))
        series.attrs["n_nan"] = int(meta.get("n_nan", int(series.isna().sum())))
        series.attrs["requested_start"] = meta.get("requested_start", "")
        series.attrs["requested_end"] = meta.get("requested_end", "")
        series.attrs["config_fingerprint"] = meta.get("config_fingerprint", "")
        return series

    def coverage_report(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-series range, observation count, and share of NaNs dropped ([D1])."""
        rows: list[dict[str, Any]] = []
        for column in frame.columns:
            series = frame[column]
            n_raw = int(series.shape[0])
            n_obs = int(series.notna().sum())
            n_nan = n_raw - n_obs
            valid = series.dropna()
            rows.append(
                {
                    "series_id": column,
                    "start": valid.index.min().date() if n_obs else pd.NaT,
                    "end": valid.index.max().date() if n_obs else pd.NaT,
                    "n_raw": n_raw,
                    "n_obs": n_obs,
                    "n_nan_dropped": n_nan,
                    "pct_nan_dropped": (100.0 * n_nan / n_raw) if n_raw else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def _must_be_non_negative(self, series_id: str) -> bool:
        if series_id in self.config.fred_series:
            return self.config.fred_series[series_id].non_negative
        if series_id in self.config.etf_tickers:
            return True
        return False

    def _reversing_jump_issues(self, cleaned: pd.Series, series_id: str) -> list[str]:
        rules = self.config.validation
        delta = cleaned.diff()
        valid = delta.dropna()
        if valid.empty:
            return []
        centered = np.abs(valid.to_numpy() - np.median(valid.to_numpy()))
        sigma = float(rules.robust_sigma_constant * np.median(centered))
        # Flat series ⇒ σ̂ = 0. Any non-zero reversing pair is then an
        # infinite-sigma print error and must fail ([D3]).
        threshold = rules.max_jump_sigma * sigma
        issues: list[str] = []
        values = delta.to_numpy()
        index = delta.index
        for i in range(1, len(values) - 1):
            jump = values[i]
            nxt = values[i + 1]
            if np.isnan(jump) or np.isnan(nxt):
                continue
            if abs(jump) <= threshold:
                continue
            if abs(jump + nxt) <= rules.reversal_tolerance * abs(jump):
                issues.append(
                    f"reversing jump exceeding {rules.max_jump_sigma} sigma "
                    f"in {series_id} at {pd.Timestamp(index[i]).date()}"
                )
        return issues

    def _reject_discontinued(self, series_id: str) -> None:
        if series_id not in self.config.discontinued_series:
            return
        spec = self.config.discontinued_series[series_id]
        substitutes = ", ".join(spec.substitutes)
        raise ValueError(
            f"{series_id} is discontinued as of {spec.discontinued_on}: {spec.reason} "
            f"Substitutes: {substitutes}."
        )

    def _get_or_download_fred(
        self,
        series_id: str,
        start: date | datetime | str,
        end: date | datetime | str,
        *,
        force_refresh: bool,
    ) -> pd.Series:
        cached = self._try_load_covering(series_id, start, end, force_refresh)
        if cached is not None:
            logger.info("cache hit FRED series=%s", series_id)
            return self._slice(cached, start, end)

        spec = self.config.fred_series[series_id]
        obs_start = _as_timestamp(start)
        obs_end = _as_timestamp(end)
        if spec.publication_lag_days:
            obs_start = obs_start - pd.Timedelta(days=spec.publication_lag_days)

        client = self._fred()
        raw = client.get_series(
            series_id,
            observation_start=_iso_date(obs_start),
            observation_end=_iso_date(obs_end),
        )
        series = self._normalize_series(raw, series_id)
        if spec.publication_lag_days:
            series = self.apply_publication_lag(series, spec.publication_lag_days)
            logger.info(
                "applied publication lag of %s days to %s",
                spec.publication_lag_days,
                series_id,
            )
        self._write_cache(
            series_id,
            series,
            requested_start=start,
            requested_end=end,
        )
        logger.info(
            "fetched FRED series=%s n_raw=%s n_obs=%s window=%s/%s",
            series_id,
            int(series.shape[0]),
            int(series.notna().sum()),
            series.dropna().index.min() if series.notna().any() else None,
            series.dropna().index.max() if series.notna().any() else None,
        )
        return self._slice(series, start, end)

    def _get_or_download_etf(
        self,
        ticker: str,
        start: date | datetime | str,
        end: date | datetime | str,
        *,
        force_refresh: bool,
        auto_adjust: bool,
    ) -> pd.Series:
        cached = self._try_load_covering(ticker, start, end, force_refresh)
        if cached is not None:
            logger.info("cache hit ETF ticker=%s", ticker)
            return self._slice(cached, start, end)

        download_end = _as_timestamp(end)
        if self.config.etf_download.end_date_exclusive:
            download_end = download_end + pd.Timedelta(
                days=self.config.etf_download.end_exclusive_shift_days
            )
        downloader = self._etf()
        frame = downloader(
            ticker,
            start=_iso_date(start),
            end=_iso_date(download_end),
            auto_adjust=auto_adjust,
            progress=False,
        )
        series = self._extract_adjusted_close(frame, ticker)
        self._write_cache(
            ticker,
            series,
            requested_start=start,
            requested_end=end,
        )
        logger.info(
            "fetched ETF ticker=%s auto_adjust=%s n_obs=%s window=%s/%s",
            ticker,
            auto_adjust,
            int(series.notna().sum()),
            series.index.min(),
            series.index.max(),
        )
        return self._slice(series, start, end)

    def _try_load_covering(
        self,
        series_id: str,
        start: date | datetime | str,
        end: date | datetime | str,
        force_refresh: bool,
    ) -> pd.Series | None:
        path = self.cache_path(series_id)
        if force_refresh or not path.exists():
            return None
        cached = self.load_cached(series_id)
        if cached.attrs.get("config_fingerprint") != self._cache_fingerprint(series_id):
            logger.info("cache fingerprint mismatch for %s; re-downloading", series_id)
            return None
        req_start = cached.attrs.get("requested_start")
        req_end = cached.attrs.get("requested_end")
        if not req_start or not req_end:
            logger.info("cache for %s missing requested window metadata; re-downloading", series_id)
            return None
        covers = (
            _as_timestamp(req_start) <= _as_timestamp(start)
            and _as_timestamp(req_end) >= _as_timestamp(end)
        )
        if not covers:
            logger.info(
                "cache for %s covers %s/%s, not requested %s/%s",
                series_id,
                req_start,
                req_end,
                start,
                end,
            )
            return None
        return cached

    def _cache_fingerprint(self, series_id: str) -> str:
        if series_id in self.config.fred_series:
            spec = self.config.fred_series[series_id]
            payload = f"fred|{series_id}|{spec.publication_lag_days}|{spec.frequency}"
        elif series_id in self.config.etf_tickers:
            spec = self.config.etf_tickers[series_id]
            payload = f"etf|{series_id}|{spec.auto_adjust}"
        else:
            payload = series_id
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _write_cache(
        self,
        series_id: str,
        series: pd.Series,
        *,
        requested_start: date | datetime | str,
        requested_end: date | datetime | str,
    ) -> None:
        if series.dropna().empty:
            logger.warning("refusing to cache empty series %s", series_id)
            return
        path = self.cache_path(series_id)
        if path.exists():
            try:
                previous = self.load_cached(series_id)
                new_hash = _hash_series(series)
                old_hash = previous.attrs.get("content_hash")
                if old_hash and old_hash != new_hash:
                    logger.warning(
                        "source revision for %s: hash %s -> %s",
                        series_id,
                        old_hash,
                        new_hash,
                    )
            except (OSError, KeyError, ValueError):
                logger.warning("could not compare existing cache hash for %s", series_id)

        retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        valid = series.dropna()
        n_obs = int(valid.shape[0])
        metadata = {
            "retrieved_at_utc": retrieved,
            "n_observations": str(n_obs),
            "n_raw": str(int(series.shape[0])),
            "n_nan": str(int(series.isna().sum())),
            "first_obs": valid.index.min().strftime("%Y-%m-%d") if n_obs else "",
            "last_obs": valid.index.max().strftime("%Y-%m-%d") if n_obs else "",
            "content_hash": _hash_series(series),
            "series_id": series_id,
            "requested_start": _iso_date(requested_start),
            "requested_end": _iso_date(requested_end),
            "config_fingerprint": self._cache_fingerprint(series_id),
        }
        frame = series.rename("value").to_frame()
        frame.index.name = "date"
        table = pa.Table.from_pandas(frame.reset_index(), preserve_index=False)
        table = table.replace_schema_metadata(
            {key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()}
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        logger.info("wrote cache %s retrieved_at_utc=%s n_obs=%s", path, retrieved, n_obs)

    def _normalize_series(self, raw: pd.Series, series_id: str) -> pd.Series:
        series = pd.Series(raw, copy=True)
        series.name = series_id
        series.index = self._normalize_index(series.index)
        series.index.name = "date"
        return series.astype("float64")

    def _normalize_index(self, index: pd.Index) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(index)
        if idx.tz is not None:
            idx = idx.tz_convert(self.config.timezone).tz_localize(None)
        normalized = idx.normalize()
        normalized.freq = None
        return normalized

    def _extract_adjusted_close(self, frame: pd.DataFrame, ticker: str) -> pd.Series:
        close = _adjusted_close_column(frame, ticker)
        series = pd.Series(close.to_numpy(), index=close.index, name=ticker, dtype="float64")
        series.index = self._normalize_index(series.index)
        series.index.name = "date"
        return series

    def _slice(
        self,
        series: pd.Series,
        start: date | datetime | str,
        end: date | datetime | str,
    ) -> pd.Series:
        sliced = series.loc[_as_timestamp(start) : _as_timestamp(end)].copy()
        sliced.attrs.update(dict(series.attrs))
        return sliced

    def _fred(self) -> Any:
        if self._fred_client is not None:
            return self._fred_client
        from fredapi import Fred  # lazy: unit tests inject a fake

        key = os.environ.get(self.config.fred_api_key_env)
        if not key:
            raise RuntimeError(
                f"environment variable {self.config.fred_api_key_env} is not set"
            )
        self._fred_client = Fred(api_key=key)
        return self._fred_client

    def _etf(self) -> Callable[..., pd.DataFrame]:
        if self._etf_downloader is not None:
            return self._etf_downloader
        import yfinance as yf  # lazy: unit tests inject a fake

        self._etf_downloader = yf.download
        return self._etf_downloader


def _iso_date(value: date | datetime | str | pd.Timestamp) -> str:
    return _as_timestamp(value).strftime("%Y-%m-%d")


def _decode_metadata(metadata: Mapping[bytes | str, bytes | str] | None) -> dict[str, str]:
    if not metadata:
        return {}
    decoded: dict[str, str] = {}
    for key, value in metadata.items():
        key_s = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        val_s = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        decoded[key_s] = val_s
    return decoded


def _adjusted_close_column(frame: pd.DataFrame, ticker: str) -> pd.Series:
    columns = frame.columns
    if isinstance(columns, pd.MultiIndex):
        level0 = columns.get_level_values(0)
        if "Adj Close" in level0:
            block = frame["Adj Close"]
        elif "Close" in level0:
            block = frame["Close"]
        else:
            raise KeyError(f"yfinance frame for {ticker} has no Close/Adj Close")
        if isinstance(block, pd.DataFrame) and ticker in block.columns:
            return block[ticker]
        if isinstance(block, pd.Series):
            return block
        return block.iloc[:, 0]
    if "Adj Close" in frame.columns:
        return frame["Adj Close"]
    if "Close" in frame.columns:
        return frame["Close"]
    raise KeyError(f"yfinance frame for {ticker} has no Close/Adj Close")
