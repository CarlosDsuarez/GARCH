"""Excess Bond Premium ingest, publication-lag calendar, and temporal disaggregation.

The official EBP (Gilchrist & Zakrajšek 2012; Favara–Gilchrist–Lewis–Zakrajšek
FEDS Notes) is monthly. This module is the only production entry point for
that series.

Publication lag ([C1])
----------------------
At backtest date ``t`` the loader may use month ``m`` only when

.. math::

    \\mathrm{publication\\_date}(m) \\le t

with ``publication_date(m) = month\\_end(m) + L`` and ``L`` from
``ebp.publication_lag_days`` (default 45). The Federal Reserve typically
posts around the fourth business day of the following month; 45 calendar
days after month-end is a conservative no-lookahead bound.

The historical EBP is **revised** when underlying issuer files change.
A real-time vintage archive is not published. The lag calendar therefore
controls *availability of a month*, not the revised *values*. Cached
parquets store ``retrieved_at_utc`` and a content hash so a later
download that differs can be logged as a source revision. That residual
look-ahead in the numbers is an implementation risk, not a license to
ignore the lag.

Chow-Lin (1971)
---------------
Let :math:`y` be the monthly series, :math:`X` the daily indicators, and
:math:`C` the average-of-days aggregation matrix. Residuals are AR(1)
at daily frequency with autocorrelation :math:`\\rho`. GLS gives
:math:`\\hat\\beta`, and the daily interpolator

.. math::

    \\hat y = X\\hat\\beta + \\Omega C'(C\\Omega C')^{-1}(y - CX\\hat\\beta)

satisfies :math:`C\\hat y = y` exactly (accounting consistency). If GLS
is numerically unstable, fall back to additive Denton–Cholette.

Daily EBP is a **model output**. High-frequency variation is inherited
from the anchors (VIX, HY OAS, curve). The official monthly series is
the only clean evidence; daily series refine intra-month timing after
the monthly signal has already fired.

References
----------
Chow, G. C. and Lin, A. (1971). Best Linear Unbiased Interpolation,
Distribution, and Extrapolation of Time Series by Related Series.
*The Review of Economics and Statistics*, 53(4), 372–375.
Denton, F. T. (1971). Adjustment of Monthly or Quarterly Series to
Annual Totals: An Approach Based on Quadratic Minimization.
*Journal of the American Statistical Association*, 66(333), 99–102.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import DataConfig, EbpDataConfig, load_data_config

logger = logging.getLogger(__name__)

__all__ = [
    "ChowLinResult",
    "DisaggResult",
    "EBPFrame",
    "EBPLoader",
    "SensitivityReport",
    "build_aggregation_matrix",
    "build_indicator_matrix",
    "chow_lin_disaggregate",
    "denton_cholette_disaggregate",
    "publication_date",
]


class TemporalDisaggError(ValueError):
    """Chow-Lin / Denton could not produce a consistent daily series ([C7])."""


class DisaggConfigLike(Protocol):
    include_constant: bool
    rho_grid_min: float
    rho_grid_max: float
    rho_grid_size: int
    condition_number_max: float
    consistency_atol: float
    high_freq_rho_from_monthly: bool
    fallback: str


@dataclass(frozen=True)
class ChowLinResult:
    daily: pd.Series
    beta: np.ndarray
    rho: float
    method: str
    monthly_replicated: pd.Series
    aggregation_error: float


@dataclass(frozen=True)
class DisaggResult:
    daily: pd.Series
    months_used: pd.DatetimeIndex
    look_ahead: bool
    method: str
    vix_only: bool
    chow_lin: ChowLinResult


@dataclass(frozen=True)
class SensitivityReport:
    lags: list[int]
    correlations: pd.DataFrame
    asof_levels: pd.DataFrame
    fragile: bool
    min_correlation: float
    threshold: float


@dataclass(frozen=True)
class EBPFrame:
    data: pd.DataFrame
    retrieved_at_utc: str
    content_hash: str

    @property
    def ebp(self) -> pd.Series:
        return self.data["ebp"]

    @property
    def gz_spread(self) -> pd.Series:
        return self.data["gz_spread"]


@dataclass(frozen=True)
class _DisaggKnobs:
    include_constant: bool = True
    rho_grid_min: float = -0.99
    rho_grid_max: float = 0.99
    rho_grid_size: int = 199
    condition_number_max: float = 1.0e12
    consistency_atol: float = 1.0e-10
    high_freq_rho_from_monthly: bool = True
    fallback: str = "denton_cholette"


def publication_date(
    month: pd.Timestamp | str,
    lag_days: int,
    *,
    anchor: str = "month_end",
) -> pd.Timestamp:
    """First calendar date on which month ``m`` may be used ([C1]).

    .. math::

        \\mathrm{publication\\_date}(m) = \\mathrm{month\\_end}(m) + L
    """
    if lag_days < 0:
        raise ValueError("publication lag must be non-negative")
    if anchor != "month_end":
        raise ValueError(f"unsupported publication_lag_anchor: {anchor}")
    month_end = pd.Timestamp(month) + pd.offsets.MonthEnd(0)
    return (month_end + pd.Timedelta(days=int(lag_days))).normalize()


def build_aggregation_matrix(
    daily_index: pd.DatetimeIndex,
    monthly_index: pd.DatetimeIndex,
    *,
    how: str = "average",
) -> np.ndarray:
    """Average-of-days matrix :math:`C` mapping daily observations to months."""
    if how != "average":
        raise ValueError("EBP is an index-like series; aggregation must be 'average'")
    daily_index = pd.DatetimeIndex(daily_index)
    monthly_index = pd.DatetimeIndex(monthly_index)
    n_daily = len(daily_index)
    n_month = len(monthly_index)
    matrix = np.zeros((n_month, n_daily), dtype=float)
    daily_periods = daily_index.to_period("M")
    monthly_periods = monthly_index.to_period("M")
    for i, period in enumerate(monthly_periods):
        mask = np.asarray(daily_periods == period)
        count = int(mask.sum())
        if count == 0:
            raise TemporalDisaggError(f"no daily observations for month {period}")
        matrix[i, mask] = 1.0 / count
    return matrix


def build_indicator_matrix(
    vix: pd.Series,
    hy_oas: pd.Series,
    t10y2y: pd.Series,
    *,
    vix_only: bool = False,
) -> pd.DataFrame:
    """Daily Chow-Lin anchors: :math:`\\ln\\mathrm{VIX}`, :math:`\\ln\\mathrm{HY}`, T10Y2Y."""
    vix_clean = vix.dropna().astype("float64")
    if (vix_clean <= 0).any():
        raise ValueError("VIX must be strictly positive to take logs")
    frame = pd.DataFrame({"ln_vix": np.log(vix_clean)}, index=vix_clean.index)
    if vix_only:
        return frame
    hy_clean = hy_oas.reindex(frame.index).astype("float64")
    if hy_clean.notna().any() and bool((hy_clean.dropna() <= 0).any()):
        raise ValueError("HY OAS must be strictly positive to take logs")
    frame["ln_hy"] = np.log(hy_clean)
    frame["t10y2y"] = t10y2y.reindex(frame.index).astype("float64")
    return frame.dropna()


def _knobs(config: DisaggConfigLike | _DisaggKnobs | None) -> _DisaggKnobs:
    if config is None:
        return _DisaggKnobs()
    if isinstance(config, _DisaggKnobs):
        return config
    return _DisaggKnobs(
        include_constant=config.include_constant,
        rho_grid_min=config.rho_grid_min,
        rho_grid_max=config.rho_grid_max,
        rho_grid_size=config.rho_grid_size,
        condition_number_max=config.condition_number_max,
        consistency_atol=config.consistency_atol,
        high_freq_rho_from_monthly=config.high_freq_rho_from_monthly,
        fallback=config.fallback,
    )


def _align_low_high(
    low_freq: pd.Series,
    indicators: pd.DataFrame | pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    y = low_freq.dropna().astype("float64").sort_index()
    y.index = pd.DatetimeIndex(y.index).normalize()
    x = indicators.copy()
    if isinstance(x, pd.Series):
        x = x.to_frame(name=x.name or "x")
    x = x.dropna().astype("float64").sort_index()
    x.index = pd.DatetimeIndex(x.index).normalize()
    y_periods = y.index.to_period("M")
    x = x.loc[x.index.to_period("M").isin(y_periods)]
    have = x.index.to_period("M").unique()
    y = y.loc[y.index.to_period("M").isin(have)]
    if y.empty or x.empty:
        raise TemporalDisaggError("no overlapping months between EBP and daily indicators")
    return y, x


def _denton_kkt(x: np.ndarray, y: np.ndarray, agg: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Additive Denton via KKT. ``D'D`` is singular; a ridge makes it SPD.

    The aggregation block ``C ŷ = y`` stays exact, so monthly averages
    reproduce ``y`` to solver tolerance.
    """
    n_daily = int(x.shape[0])
    n_month = int(agg.shape[0])
    gram = _difference_gram(n_daily) + ridge * np.eye(n_daily)
    kkt = np.zeros((n_daily + n_month, n_daily + n_month), dtype=float)
    kkt[:n_daily, :n_daily] = gram
    kkt[:n_daily, n_daily:] = agg.T
    kkt[n_daily:, :n_daily] = agg
    rhs = np.zeros(n_daily + n_month, dtype=float)
    rhs[n_daily:] = y - agg @ x
    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError as exc:
        raise TemporalDisaggError("Denton–Cholette linear system is singular") from exc
    return x + sol[:n_daily]


def _difference_gram(n: int) -> np.ndarray:
    gram = np.zeros((n, n), dtype=float)
    if n == 1:
        gram[0, 0] = 1.0
        return gram
    gram[0, 0] = 1.0
    gram[0, 1] = -1.0
    gram[-1, -1] = 1.0
    gram[-1, -2] = -1.0
    for i in range(1, n - 1):
        gram[i, i - 1] = -1.0
        gram[i, i] = 2.0
        gram[i, i + 1] = -1.0
    return gram


def _omega_c_transpose(rho: float, agg: np.ndarray) -> np.ndarray:
    """Columns of :math:`\\Omega C'` without forming the :math:`N\\times N` :math:`\\Omega`.

    :math:`(\\Omega C')_{t,i} = n_i^{-1}\\sum_{s\\in m_i}\\rho^{|t-s|}`.
    """
    n_daily, n_month = agg.shape[1], agg.shape[0]
    times = np.arange(n_daily, dtype=float)
    omega_ct = np.empty((n_daily, n_month), dtype=float)
    abs_rho = abs(float(rho))
    for i in range(n_month):
        weights = agg[i]
        members = np.flatnonzero(weights)
        w = weights[members]
        if abs_rho == 0.0:
            col = np.zeros(n_daily, dtype=float)
            col[members] = w
            omega_ct[:, i] = col
            continue
        for t in range(n_daily):
            omega_ct[t, i] = float(np.dot(w, rho ** np.abs(times[t] - members)))
    return omega_ct


def _monthly_ar1(resid: np.ndarray) -> float:
    if resid.shape[0] < 3:
        return 0.0
    lagged = resid[:-1]
    lead = resid[1:]
    denom = float(lagged @ lagged)
    if denom == 0.0:
        return 0.0
    return float(np.clip(lagged @ lead / denom, -0.99, 0.99))


def _gls_at_rho(
    y: np.ndarray,
    design: np.ndarray,
    agg: np.ndarray,
    rho: float,
    cond_max: float,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    omega_ct = _omega_c_transpose(rho, agg)
    omega_bar = agg @ omega_ct
    cond = float(np.linalg.cond(omega_bar))
    if not np.isfinite(cond) or cond > cond_max:
        return None
    c_design = agg @ design
    try:
        inv_x = np.linalg.solve(omega_bar, c_design)
        inv_y = np.linalg.solve(omega_bar, y)
        beta = np.linalg.lstsq(c_design.T @ inv_x, c_design.T @ inv_y, rcond=None)[0]
        resid = y - c_design @ beta
        lam = np.linalg.solve(omega_bar, resid)
    except np.linalg.LinAlgError:
        return None
    daily = design @ beta + omega_ct @ lam
    ssr = float(resid @ lam)
    return beta, daily, ssr, cond


def denton_cholette_disaggregate(
    low_freq: pd.Series,
    preliminary: pd.Series | pd.DataFrame,
    agg_matrix: np.ndarray | None = None,
) -> pd.Series:
    """Additive Denton–Cholette: smooth daily path with :math:`C\\hat y=y`.

    EBP can be negative, so proportional Denton is undefined. Minimize
    :math:`(\\hat y - x)'D'D(\\hat y - x)` subject to :math:`C\\hat y = y`.
    """
    if isinstance(preliminary, pd.DataFrame):
        preliminary = preliminary.iloc[:, 0]
    y, x_frame = _align_low_high(low_freq, preliminary)
    x = x_frame.iloc[:, 0]
    agg = agg_matrix if agg_matrix is not None else build_aggregation_matrix(x.index, y.index)
    daily = _denton_kkt(x.to_numpy(dtype=float), y.to_numpy(dtype=float), agg)
    series = pd.Series(daily, index=x.index, name=y.name or "ebp")
    replicated = agg @ series.to_numpy()
    if not np.allclose(replicated, y.to_numpy(), atol=1e-8):
        raise TemporalDisaggError("Denton–Cholette failed the accounting constraint")
    return series


def chow_lin_disaggregate(
    low_freq: pd.Series,
    indicators: pd.DataFrame,
    agg_matrix: np.ndarray | None = None,
    *,
    rho: float | None = None,
    include_constant: bool | None = None,
    config: DisaggConfigLike | None = None,
) -> ChowLinResult:
    """Chow-Lin GLS interpolator; Denton–Cholette if GLS is unstable.

    Parameters
    ----------
    low_freq
        Official monthly EBP (or any low-frequency target).
    indicators
        Daily anchors already transformed (``ln_vix``, ``ln_hy``, ``t10y2y``).
    agg_matrix
        Optional pre-built :math:`C`. Built from the aligned indexes if omitted.
    """
    knobs = _knobs(config)
    use_constant = knobs.include_constant if include_constant is None else include_constant
    y, x = _align_low_high(low_freq, indicators)
    agg = agg_matrix if agg_matrix is not None else build_aggregation_matrix(x.index, y.index)
    design = x.to_numpy(dtype=float)
    if use_constant:
        design = np.column_stack([np.ones(design.shape[0]), design])
    y_vec = y.to_numpy(dtype=float)

    candidates: list[float]
    if rho is not None:
        candidates = [float(np.clip(rho, -0.99, 0.99))]
    elif knobs.high_freq_rho_from_monthly:
        c_design = agg @ design
        ols = np.linalg.lstsq(c_design, y_vec, rcond=None)[0]
        rho_m = _monthly_ar1(y_vec - c_design @ ols)
        n_bar = max(design.shape[0] / max(y_vec.shape[0], 1), 1.0)
        rho_d = float(np.sign(rho_m) * (abs(rho_m) ** (1.0 / n_bar))) if rho_m != 0.0 else 0.0
        grid = np.linspace(knobs.rho_grid_min, knobs.rho_grid_max, knobs.rho_grid_size)
        near = grid[np.abs(grid - rho_d) <= 0.35]
        candidates = [rho_d, *near.tolist()] if near.size else [rho_d]
    else:
        candidates = list(np.linspace(knobs.rho_grid_min, knobs.rho_grid_max, knobs.rho_grid_size))

    best: tuple[np.ndarray, np.ndarray, float, float, float] | None = None
    seen: set[float] = set()
    for candidate in candidates:
        key = round(float(candidate), 8)
        if key in seen:
            continue
        seen.add(key)
        fit = _gls_at_rho(y_vec, design, agg, float(candidate), knobs.condition_number_max)
        if fit is None:
            continue
        beta, daily, ssr, _cond = fit
        if best is None or ssr < best[2]:
            best = (beta, daily, ssr, float(candidate), _cond)

    if best is None:
        logger.warning("Chow-Lin GLS unstable; falling back to Denton–Cholette")
        preliminary = x.iloc[:, 0]
        daily_series = denton_cholette_disaggregate(y, preliminary, agg)
        replicated = pd.Series(agg @ daily_series.to_numpy(), index=y.index, name=y.name)
        error = float(np.max(np.abs(replicated.to_numpy() - y_vec)))
        return ChowLinResult(
            daily=daily_series,
            beta=np.asarray([]),
            rho=float("nan"),
            method="denton_cholette",
            monthly_replicated=replicated,
            aggregation_error=error,
        )

    beta, daily, _ssr, used_rho, _cond = best
    daily_series = pd.Series(daily, index=x.index, name=y.name or "ebp")
    replicated_vals = agg @ daily_series.to_numpy()
    error = float(np.max(np.abs(replicated_vals - y_vec)))
    if error > max(knobs.consistency_atol * 100.0, 1e-8):
        logger.warning(
            "Chow-Lin aggregation error %.3e exceeds tolerance; Denton fallback",
            error,
        )
        daily_series = denton_cholette_disaggregate(y, x.iloc[:, 0], agg)
        replicated_vals = agg @ daily_series.to_numpy()
        error = float(np.max(np.abs(replicated_vals - y_vec)))
        return ChowLinResult(
            daily=daily_series,
            beta=beta,
            rho=used_rho,
            method="denton_cholette",
            monthly_replicated=pd.Series(replicated_vals, index=y.index, name=y.name),
            aggregation_error=error,
        )
    return ChowLinResult(
        daily=daily_series,
        beta=beta,
        rho=used_rho,
        method="chow_lin",
        monthly_replicated=pd.Series(replicated_vals, index=y.index, name=y.name),
        aggregation_error=error,
    )


def _hash_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(date_format="%Y-%m-%d", float_format="%.12g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode_metadata(metadata: Mapping[bytes | str, bytes | str] | None) -> dict[str, str]:
    if not metadata:
        return {}
    decoded: dict[str, str] = {}
    for key, value in metadata.items():
        key_s = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        val_s = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        decoded[key_s] = val_s
    return decoded


class EBPLoader:
    """Download, cache, and as-of filter the official monthly EBP.

    Parameters
    ----------
    config
        Validated ``DataConfig`` with a populated ``ebp`` block.
    http_get
        ``url -> bytes``. Injected in tests; ``urllib`` in production.
    """

    def __init__(
        self,
        config: DataConfig,
        *,
        project_root: str | Path | None = None,
        http_get: Callable[[str], bytes] | None = None,
    ) -> None:
        if config.ebp is None:
            raise ValueError("DataConfig.ebp is required for EBPLoader")
        self.config = config
        self.ebp_cfg: EbpDataConfig = config.ebp
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        cache_dir = Path(config.cache.directory)
        self.cache_dir = cache_dir if cache_dir.is_absolute() else self.project_root / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._http_get = http_get
        self._frame: EBPFrame | None = None

    @classmethod
    def from_yaml(cls, path: str | Path, **kwargs: Any) -> EBPLoader:
        config_path = Path(path)
        config = load_data_config(config_path)
        kwargs.setdefault("project_root", config_path.resolve().parent.parent)
        return cls(config, **kwargs)

    def cache_path(self) -> Path:
        filename = self.config.cache.filename_template.format(series_id=self.ebp_cfg.cache_id)
        return self.cache_dir / filename

    def fetch(self, *, force_refresh: bool = False) -> EBPFrame:
        cached = None if force_refresh else self._try_load_cache()
        if cached is not None:
            self._frame = cached
            logger.info(
                "cache hit EBP retrieved_at_utc=%s n_obs=%s",
                cached.retrieved_at_utc,
                int(cached.ebp.notna().sum()),
            )
            return cached
        raw = self._http()(self.ebp_cfg.url)
        data = self._parse_csv(raw)
        retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        content_hash = _hash_frame(data)
        self._write_cache(data, retrieved=retrieved, content_hash=content_hash)
        frame = EBPFrame(data=data, retrieved_at_utc=retrieved, content_hash=content_hash)
        self._frame = frame
        logger.info(
            "fetched EBP n_obs=%s window=%s/%s hash=%s",
            int(data["ebp"].notna().sum()),
            data.index.min(),
            data.index.max(),
            content_hash[:12],
        )
        return frame

    def publication_calendar(
        self,
        monthly_index: pd.DatetimeIndex | None = None,
        *,
        lag_days: int | None = None,
    ) -> pd.Series:
        frame = self._require_frame()
        index = pd.DatetimeIndex(monthly_index if monthly_index is not None else frame.data.index)
        lag = self.ebp_cfg.publication_lag_days if lag_days is None else int(lag_days)
        values = [
            publication_date(ts, lag, anchor=self.ebp_cfg.publication_lag_anchor)
            for ts in index
        ]
        return pd.Series(values, index=index, name="publication_date")

    def available_asof(
        self,
        asof: pd.Timestamp | str,
        *,
        lag_days: int | None = None,
    ) -> pd.DataFrame:
        frame = self._require_frame()
        asof_ts = pd.Timestamp(asof).normalize()
        pubs = self.publication_calendar(frame.data.index, lag_days=lag_days)
        mask = pubs <= asof_ts
        out = frame.data.loc[mask].copy()
        logger.info(
            "EBP as-of %s lag=%s n_months=%s last=%s",
            asof_ts.date(),
            lag_days if lag_days is not None else self.ebp_cfg.publication_lag_days,
            int(out.shape[0]),
            out.index.max() if not out.empty else None,
        )
        return out

    def disaggregate(
        self,
        indicators: pd.DataFrame,
        *,
        vix_only: bool = False,
        asof: pd.Timestamp | str | None = None,
        descriptive_full_sample: bool = False,
        config: DisaggConfigLike | None = None,
    ) -> DisaggResult:
        """Daily EBP. Full-sample interpolation is descriptive only (look-ahead)."""
        if asof is not None and descriptive_full_sample:
            raise ValueError("asof and descriptive_full_sample are mutually exclusive")
        frame = self._require_frame()
        x = indicators.copy()
        if vix_only:
            if "ln_vix" not in x.columns:
                raise ValueError("vix_only disaggregation requires an 'ln_vix' column")
            x = x[["ln_vix"]]
        look_ahead = bool(descriptive_full_sample or asof is None)
        if asof is not None:
            asof_ts = pd.Timestamp(asof).normalize()
            monthly = self.available_asof(asof_ts)
            x = x.loc[x.index <= asof_ts]
            look_ahead = False
        else:
            monthly = frame.data
            if descriptive_full_sample:
                logger.warning(
                    "full-sample Chow-Lin uses unpublished months and future daily "
                    "anchors; descriptive only — do not use in a backtest ([C1])"
                )
        if monthly.empty:
            raise TemporalDisaggError("no EBP months are published as-of the requested date")
        fitted = chow_lin_disaggregate(monthly["ebp"], x, config=config)
        months_used = pd.DatetimeIndex(fitted.monthly_replicated.index)
        daily = fitted.daily
        if asof is not None:
            asof_ts = pd.Timestamp(asof).normalize()
            extra = x.loc[x.index > daily.index.max()]
            extra = extra.loc[extra.index <= asof_ts]
            if not extra.empty and fitted.beta.size:
                beta = fitted.beta
                extra_design = extra.to_numpy(dtype=float)
                if extra_design.shape[1] + 1 == beta.shape[0]:
                    extra_design = np.column_stack([np.ones(extra_design.shape[0]), extra_design])
                if extra_design.shape[1] == beta.shape[0]:
                    projected = pd.Series(extra_design @ beta, index=extra.index, name=daily.name)
                    daily = pd.concat([daily, projected]).sort_index()
                    logger.info(
                        "projected %s unpublished daily EBP points from Xβ only (no official y)",
                        int(projected.shape[0]),
                    )
        return DisaggResult(
            daily=daily,
            months_used=months_used,
            look_ahead=look_ahead,
            method=fitted.method,
            vix_only=vix_only,
            chow_lin=fitted,
        )

    def lag_sensitivity(
        self,
        asof_index: pd.DatetimeIndex,
        *,
        lags: Sequence[int] | None = None,
    ) -> SensitivityReport:
        """Compare as-of EBP paths under publication lags 30 / 45 / 60.

        If first-difference correlations across lags fall below
        ``sensitivity_min_correlation``, the strategy is fragile to the
        publication calendar and must be reported as an implementation risk.
        """
        self._require_frame()
        lags_used = list(lags if lags is not None else self.ebp_cfg.sensitivity_lags_days)
        asof_index = pd.DatetimeIndex(asof_index).normalize()
        columns: dict[int, pd.Series] = {}
        for lag in lags_used:
            values = []
            for stamp in asof_index:
                available = self.available_asof(stamp, lag_days=int(lag))
                values.append(float(available["ebp"].iloc[-1]) if not available.empty else np.nan)
            columns[int(lag)] = pd.Series(values, index=asof_index, name=str(lag))
        levels = pd.DataFrame(columns)
        diffs = levels.diff()
        corr = diffs.corr()
        values = corr.to_numpy(dtype=float)
        off_diag = values[np.triu_indices_from(values, k=1)]
        min_corr = float(np.nanmin(off_diag)) if off_diag.size else 1.0
        threshold = float(self.ebp_cfg.sensitivity_min_correlation)
        fragile = bool(min_corr < threshold)
        if fragile:
            logger.warning(
                "EBP publication-lag sensitivity: min Δ-corr=%.3f < %.3f; "
                "implementation-timing risk",
                min_corr,
                threshold,
            )
        else:
            logger.info(
                "EBP publication-lag sensitivity min Δ-corr=%.3f (threshold %.3f)",
                min_corr,
                threshold,
            )
        return SensitivityReport(
            lags=lags_used,
            correlations=corr,
            asof_levels=levels,
            fragile=fragile,
            min_correlation=min_corr,
            threshold=threshold,
        )

    def _require_frame(self) -> EBPFrame:
        if self._frame is None:
            return self.fetch()
        return self._frame

    def _http(self) -> Callable[[str], bytes]:
        if self._http_get is not None:
            return self._http_get

        def _get(url: str) -> bytes:
            with urlopen(url, timeout=60) as response:  # noqa: S310 — URL comes from YAML
                return response.read()

        return _get

    def _parse_csv(self, raw: bytes) -> pd.DataFrame:
        spec = self.ebp_cfg
        parsed = pd.read_csv(BytesIO(raw))
        lower = {str(col).strip().lower(): col for col in parsed.columns}

        def _col(name: str) -> str:
            if name in parsed.columns:
                return name
            key = name.lower()
            if key in lower:
                return lower[key]
            raise ValueError(f"EBP CSV missing column {name}; got {list(parsed.columns)}")

        date_col = _col(spec.date_column)
        parsed[date_col] = pd.to_datetime(parsed[date_col], utc=False)
        parsed = parsed.set_index(date_col).sort_index()
        parsed.index = pd.DatetimeIndex(parsed.index).normalize()
        parsed.index.name = "date"
        frame = pd.DataFrame(
            {
                "ebp": pd.to_numeric(parsed[_col(spec.ebp_column)], errors="coerce"),
                "gz_spread": pd.to_numeric(parsed[_col(spec.gz_spread_column)], errors="coerce"),
            },
            index=parsed.index,
        )
        if spec.recession_prob_column:
            rec_name = spec.recession_prob_column
            if rec_name in parsed.columns or rec_name.lower() in lower:
                frame["recession_prob"] = pd.to_numeric(
                    parsed[_col(rec_name)], errors="coerce"
                )
        if frame.index.has_duplicates:
            raise ValueError("duplicate dates in EBP CSV")
        frame["publication_date"] = [
            publication_date(ts, spec.publication_lag_days, anchor=spec.publication_lag_anchor)
            for ts in frame.index
        ]
        return frame

    def _cache_fingerprint(self) -> str:
        spec = self.ebp_cfg
        payload = (
            f"ebp|{spec.url}|{spec.publication_lag_days}|{spec.ebp_column}|"
            f"{spec.gz_spread_column}|{spec.date_column}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _try_load_cache(self) -> EBPFrame | None:
        path = self.cache_path()
        if not path.exists():
            return None
        table = pq.read_table(path)
        meta = _decode_metadata(table.schema.metadata)
        if meta.get("config_fingerprint") != self._cache_fingerprint():
            logger.info("EBP cache fingerprint mismatch; re-downloading")
            return None
        frame = table.to_pandas()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        frame.index.name = "date"
        if "publication_date" in frame.columns:
            frame["publication_date"] = pd.to_datetime(frame["publication_date"])
        return EBPFrame(
            data=frame,
            retrieved_at_utc=meta.get("retrieved_at_utc", ""),
            content_hash=meta.get("content_hash", ""),
        )

    def _write_cache(self, data: pd.DataFrame, *, retrieved: str, content_hash: str) -> None:
        path = self.cache_path()
        if path.exists():
            previous = self._try_load_cache()
            if previous is not None and previous.content_hash and previous.content_hash != content_hash:
                logger.warning(
                    "EBP source revision: hash %s -> %s (history was revised)",
                    previous.content_hash,
                    content_hash,
                )
        metadata = {
            "retrieved_at_utc": retrieved,
            "content_hash": content_hash,
            "n_observations": str(int(data["ebp"].notna().sum())),
            "series_id": self.ebp_cfg.cache_id,
            "config_fingerprint": self._cache_fingerprint(),
            "vintage_note": (
                "Lag controls month availability only; values are the revised "
                "series available at retrieval. No official real-time vintage archive."
            ),
        }
        table = pa.Table.from_pandas(data.reset_index(), preserve_index=False)
        table = table.replace_schema_metadata(
            {key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()}
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        logger.info("wrote EBP cache %s retrieved_at_utc=%s", path, retrieved)
