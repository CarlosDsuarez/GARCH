"""Volatility-regime detection via Markov switching on an observable proxy.

Why this exists
---------------
A single GARCH parameter vector is asked to price both quiet expansions and
liquidity collapses. That restriction is counterfactual: crisis dynamics do
not obey the same :math:`(\\omega,\\alpha,\\beta)` as calm markets.

Haas, Mittnik and Paolella (2004) write a separate variance recursion per
regime and let an unobserved Markov chain :math:`s_t` pick the active
coefficients:

.. math::

    \\sigma_{k,t}^2
    = \\omega_k + \\alpha_k \\varepsilon_{t-1}^2 + \\beta_k \\sigma_{k,t-1}^2,
    \\qquad k=1,\\ldots,K.

    P(s_t=j\\mid s_{t-1}=i)=p_{ij}.
    \\qquad
    E[\\mathrm{duration}_i]=1/(1-p_{ii}).

Gray/Klaassen MS-GARCH is path-dependent: :math:`\\sigma_t^2` depends on the
*entire* regime history (:math:`K^t` trajectories). HMP gives each regime its
own recursion and is estimable. ``arch`` does not implement MS-GARCH.

Option A (this module, v1)
--------------------------
``statsmodels.tsa.regime_switching.MarkovRegression`` on **log realized
variance** (5-day sum of :math:`r_t^2`, or a rolling first PC of a vol panel).
No path-dependence, reliable convergence, no extra dependency. It is an
approximation — switching on an *observable* vol proxy, not a true MS-GARCH.
That degradation is acceptable for de-risking; robustness is the gain.

Option B (v2) is R ``MSGARCH`` via rpy2, behind :class:`RegimeBackend`.
Option C (hand-rolled Hamilton filter + EM) is not implemented.

[C1-bis]
--------
Filtered :math:`P(s_t=k\\mid y_{1:t})` is the only legal operational input.
Smoothed :math:`P(s_t=k\\mid y_{1:T})` uses data after :math:`t`. The default
of :meth:`RegimeDetector.get_regime_probability` is ``filtered``; ``smoothed``
logs a warning on every call.

Anti-chattering (all five layers)
---------------------------------
1. Dual-threshold hysteresis (enter 0.70 / exit 0.30).
2. Minimum dwell :math:`D` (default 5 sessions) after a *declared* change.
3. :math:`N` consecutive breaches (default 2) before a change — this is the
   detection lag that a backtest must quantify.
4. Lower-frequency exogenous confirm; otherwise de-risk at 50%.
5. Transitions/year vs an alarm of 8 for :math:`K=2`.

Default :math:`K=2` (Calm / Stress). :math:`K=3` is admitted only if every
regime has unconditional mass :math:`>10\\%` and expected duration :math:`>10`
days.

References
----------
Haas, M., Mittnik, S. and Paolella, M. S. (2004). A New Approach to
Markov-Switching GARCH Models. *Journal of Financial Econometrics*, 2(4),
493–530.
Hamilton, J. D. (1989). A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2),
357–384.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from models.schema import RegimeConfig, RegimeFitConfig, load_regime_config

logger = logging.getLogger(__name__)

__all__ = [
    "FittedRegime",
    "MSGARCHBackend",
    "MarkovLogVarianceBackend",
    "RegimeBackend",
    "RegimeDetector",
    "RegimeError",
    "RegimeEstimationReport",
    "RegimeReportPaths",
    "RegimeState",
    "TransitionStats",
    "apply_anti_chattering",
    "assert_regimes_economically_valid",
    "causal_exogenous_confirm",
    "expected_durations",
    "load_regime_config",
    "realized_log_variance",
    "rolling_first_pc",
    "unconditional_probabilities",
]

RegimeLabel = Literal["calm", "stress"]
RegimeMode = Literal["filtered", "smoothed"]


class RegimeError(ValueError):
    """Mandatory regime diagnostic failed ([C7])."""


@dataclass(frozen=True)
class FittedRegime:
    """Backend-agnostic output so Option B can replace Option A downstream."""

    filtered: pd.DataFrame
    smoothed: pd.DataFrame
    transition: np.ndarray
    params: pd.DataFrame
    log_likelihood: float
    aic: float
    bic: float
    stress_regime_id: int


@dataclass(frozen=True)
class RegimeState:
    date: pd.Timestamp
    label: RegimeLabel
    derisk_fraction: float
    p_stress_filtered: float
    exogenous_confirms: bool
    days_in_regime: int


@dataclass(frozen=True)
class RegimeEstimationReport:
    backend: str
    k_regimes: int
    n_obs: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    input_measure: str
    rv_window: int
    transition_matrix: pd.DataFrame
    expected_durations: dict[str, float]
    unconditional_probabilities: dict[str, float]
    regime_parameters: pd.DataFrame
    log_likelihood: float
    aic: float
    bic: float
    stress_regime_id: int
    k3_valid: bool | None


@dataclass(frozen=True)
class TransitionStats:
    n_transitions: int
    n_obs: int
    n_years: float
    transitions_per_year: float
    alarm: bool
    alarm_threshold: float
    round_trip_cost_bps: float
    annual_friction_bps: float
    confirmation_delay_days: int
    mean_detection_lag_days: float
    n_stress_episodes: int
    mean_stress_duration_days: float
    n_partial_derisk_days: int
    n_full_derisk_days: int


@dataclass(frozen=True)
class RegimeReportPaths:
    plot_path: Path
    report_markdown: Path


class RegimeBackend(Protocol):
    name: str

    def fit(self, series: pd.Series, *, k_regimes: int, seed: int) -> FittedRegime:
        """Estimate regime probabilities on an already-constructed observed series."""


class MSGARCHBackend:
    """Option B placeholder. R MSGARCH via rpy2 is v2 validation, not production."""

    name = "msgarch"

    def fit(self, series: pd.Series, *, k_regimes: int, seed: int) -> FittedRegime:
        raise NotImplementedError(
            "Option B (MSGARCH via rpy2) is reserved for v2 validation; "
            "v1 uses MarkovRegression on log realized variance"
        )


class MarkovLogVarianceBackend:
    """Option A: Markov switching on an observable log-variance proxy."""

    name = "markov_log_variance"

    def __init__(self, fit_cfg: RegimeFitConfig) -> None:
        self.fit_cfg = fit_cfg

    def fit(self, series: pd.Series, *, k_regimes: int, seed: int) -> FittedRegime:
        observed = _as_series(series, "observed").dropna()
        if observed.shape[0] < k_regimes * 10:
            raise RegimeError("not enough observations to estimate MarkovRegression")
        model = MarkovRegression(
            observed,
            k_regimes=k_regimes,
            trend=self.fit_cfg.trend,
            switching_variance=self.fit_cfg.switching_variance,
        )
        result = model.fit(
            disp=False,
            search_reps=self.fit_cfg.search_reps,
            maxiter=self.fit_cfg.maxiter,
            rng=np.random.default_rng(seed),
        )
        params = result.params
        if np.any(~np.isfinite(np.asarray(params, dtype=float))):
            raise RegimeError("MarkovRegression produced non-finite parameters")
        constants = np.array(
            [float(params[f"const[{k}]"]) for k in range(k_regimes)], dtype=float
        )
        sigma2 = np.array(
            [float(params[f"sigma2[{k}]"]) for k in range(k_regimes)], dtype=float
        )
        stress_id = int(np.argmax(constants))
        p_sm = np.asarray(result.regime_transition, dtype=float).reshape(k_regimes, k_regimes)
        # statsmodels: column i is from-regime i (columns sum to 1).
        # User convention: T[i, j] = P(s_t = j | s_{t-1} = i).
        transition = p_sm.T.copy()
        param_table = pd.DataFrame(
            {
                "const": constants,
                "sigma2": sigma2,
                "implied_rv": np.exp(constants),
            }
        )
        filtered = result.filtered_marginal_probabilities.copy()
        smoothed = result.smoothed_marginal_probabilities.copy()
        filtered.index = observed.index
        smoothed.index = observed.index
        return FittedRegime(
            filtered=filtered,
            smoothed=smoothed,
            transition=transition,
            params=param_table,
            log_likelihood=float(result.llf),
            aic=float(result.aic),
            bic=float(result.bic),
            stress_regime_id=stress_id,
        )


def realized_log_variance(returns: pd.Series, window: int) -> pd.Series:
    """Causal log realized variance: :math:`\\log\\sum_{i=0}^{w-1} r_{t-i}^2`."""
    if window < 2:
        raise RegimeError("realized-variance window must be >= 2")
    r = _as_series(returns, "r")
    rv = (r.pow(2)).rolling(window=window, min_periods=window).sum()
    positive = rv.dropna()
    if positive.empty:
        raise RegimeError("realized variance is empty after the warmup window")
    if (positive <= 0.0).any():
        raise RegimeError("non-positive realized variance in a window; log is undefined")
    return np.log(rv).rename("log_rv")


def rolling_first_pc(
    panel: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Causal rolling PC1. Loadings at :math:`t` use only rows up to :math:`t` ([C1])."""
    if window < 3:
        raise RegimeError("PCA window must be >= 3")
    min_p = window if min_periods is None else int(min_periods)
    frame = panel.copy()
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame = frame.sort_index().astype("float64")
    if frame.index.has_duplicates:
        raise RegimeError("duplicate dates in the PCA panel")
    cols = [str(c) for c in frame.columns]
    sign_col = 0
    for i, name in enumerate(cols):
        if name.upper() in {"VIX", "VIXCLS"}:
            sign_col = i
            break
    scores = np.full(frame.shape[0], np.nan, dtype=float)
    values = frame.to_numpy(dtype=float)
    for i in range(frame.shape[0]):
        start = max(0, i + 1 - window)
        block = values[start : i + 1]
        finite = np.isfinite(block).all(axis=1)
        block = block[finite]
        if block.shape[0] < min_p:
            continue
        mu = block.mean(axis=0)
        sd = block.std(axis=0, ddof=1)
        sd = np.where(sd < 1.0e-12, 1.0, sd)
        z = (block - mu) / sd
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        loadings = vt[0]
        if loadings[sign_col] < 0.0:
            loadings = -loadings
        scores[i] = float(z[-1] @ loadings)
    return pd.Series(scores, index=frame.index, name="pc1")


def expected_durations(transition: np.ndarray) -> np.ndarray:
    """:math:`E[\\mathrm{duration}_i]=1/(1-p_{ii})` with row-stochastic :math:`P`."""
    p = np.asarray(transition, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise RegimeError("transition matrix must be square")
    diag = np.diag(p)
    out = np.full(diag.shape, np.inf, dtype=float)
    stay = diag < 1.0
    out[stay] = 1.0 / (1.0 - diag[stay])
    return out


def unconditional_probabilities(transition: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic transition matrix."""
    p = np.asarray(transition, dtype=float)
    eigvals, eigvecs = np.linalg.eig(p.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])
    pi = np.maximum(pi, 0.0)
    total = float(pi.sum())
    if total <= 0.0:
        raise RegimeError("unconditional distribution is degenerate")
    return pi / total


def assert_regimes_economically_valid(
    *,
    unconditional: np.ndarray,
    durations: np.ndarray,
    min_unconditional: float,
    min_duration_days: float,
) -> None:
    """K=3 extra regime must have mass > 10% and duration > 10 days."""
    pi = np.asarray(unconditional, dtype=float)
    dur = np.asarray(durations, dtype=float)
    if pi.shape != dur.shape:
        raise RegimeError("unconditional probabilities and durations must align")
    for i, (mass, length) in enumerate(zip(pi, dur, strict=True)):
        if mass < min_unconditional:
            raise RegimeError(
                f"regime {i} unconditional probability {mass:.1%} is below 10%; "
                "the extra regime is not economically valid"
            )
        if not np.isfinite(length) or length < min_duration_days:
            raise RegimeError(
                f"regime {i} expected duration {length:.2f} days is below 10 days; "
                "the extra regime is not economically valid"
            )


def causal_exogenous_confirm(
    level: pd.Series,
    percentile: float,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """True when the current level is in the top trailing percentile ([C1])."""
    series = _as_series(level, "exogenous")
    min_p = window if min_periods is None else int(min_periods)
    q = float(percentile) / 100.0
    flags: list[bool] = []
    arr = series.to_numpy(dtype=float)
    for i in range(arr.size):
        start = max(0, i + 1 - window)
        window_vals = arr[start : i + 1]
        window_vals = window_vals[np.isfinite(window_vals)]
        current = arr[i]
        if (
            window_vals.size < min_p
            or not np.isfinite(current)
            or (np.nanmax(window_vals) - np.nanmin(window_vals)) < 1.0e-15
        ):
            flags.append(False)
            continue
        rank = float(np.sum(window_vals <= current) / window_vals.size)
        flags.append(rank >= q)
    return pd.Series(flags, index=series.index, dtype=bool, name="exogenous_confirms")


def apply_anti_chattering(
    p_stress: pd.Series,
    *,
    enter: float,
    exit: float,
    dwell: int,
    confirm_days: int,
    exogenous_confirmed: pd.Series | None,
    partial_fraction: float = 0.5,
) -> pd.DataFrame:
    """Layers 1–4. Dwell binds *after* a declared change; initial calm may exit immediately."""
    if enter <= exit:
        raise RegimeError("hysteresis enter must exceed exit")
    if dwell < 1 or confirm_days < 1:
        raise RegimeError("dwell and confirmation days must be >= 1")
    p = _as_series(p_stress, "p_stress")
    if np.any(~np.isfinite(p.to_numpy(dtype=float))):
        raise RegimeError("filtered stress probability contains NaN")
    confirmed = _align_bool(exogenous_confirmed, p.index)
    labels: list[str] = []
    derisk: list[float] = []
    days_held: list[int] = []
    state: RegimeLabel = "calm"
    days_in = int(dwell)
    consec_enter = 0
    consec_exit = 0
    for prob, exo in zip(p.to_numpy(dtype=float), confirmed.to_numpy(dtype=bool), strict=True):
        if prob > enter:
            consec_enter += 1
            consec_exit = 0
        elif prob < exit:
            consec_exit += 1
            consec_enter = 0
        else:
            consec_enter = 0
            consec_exit = 0
        want_stress = state == "calm" and consec_enter >= confirm_days
        want_calm = state == "stress" and consec_exit >= confirm_days
        if want_stress and days_in >= dwell:
            state = "stress"
            days_in = 1
        elif want_calm and days_in >= dwell:
            state = "calm"
            days_in = 1
        else:
            days_in += 1
        if state == "calm":
            fraction = 0.0
        elif exo:
            fraction = 1.0
        else:
            fraction = float(partial_fraction)
        labels.append(state)
        derisk.append(fraction)
        days_held.append(days_in)
    return pd.DataFrame(
        {
            "label": labels,
            "derisk_fraction": derisk,
            "exogenous_confirms": confirmed.to_numpy(dtype=bool),
            "days_in_regime": days_held,
        },
        index=p.index,
    )


class RegimeDetector:
    """Fit Option A (or a substitute backend) and apply the five-layer protocol."""

    def __init__(
        self,
        config: RegimeConfig,
        backend: RegimeBackend | None = None,
    ) -> None:
        self.config = config
        self.backend: RegimeBackend = backend or _backend_from_config(config)
        self._fitted: FittedRegime | None = None
        self.filtered_stress: pd.Series | None = None
        self.smoothed_stress: pd.Series | None = None
        self.states: pd.DataFrame | None = None
        self.estimation_report: RegimeEstimationReport | None = None
        self.observed: pd.Series | None = None
        self.returns: pd.Series | None = None

    @classmethod
    def from_probabilities(
        cls,
        p_stress_filtered: pd.Series,
        config: RegimeConfig,
        *,
        p_stress_smoothed: pd.Series | None = None,
        exogenous: pd.Series | None = None,
        exogenous_confirmed: pd.Series | None = None,
    ) -> RegimeDetector:
        detector = cls(config)
        filtered = _as_series(p_stress_filtered, "p_stress_filtered")
        smoothed = (
            _as_series(p_stress_smoothed, "p_stress_smoothed")
            if p_stress_smoothed is not None
            else filtered.copy()
        )
        confirmed = _resolve_exogenous(config, filtered.index, exogenous, exogenous_confirmed)
        states = apply_anti_chattering(
            filtered,
            enter=config.hysteresis.enter,
            exit=config.hysteresis.exit,
            dwell=config.dwell.min_days,
            confirm_days=config.confirmation.consecutive_days,
            exogenous_confirmed=confirmed,
            partial_fraction=config.exogenous.partial_derisk_fraction,
        )
        detector.filtered_stress = filtered
        detector.smoothed_stress = smoothed.reindex(filtered.index)
        detector.states = states
        detector.observed = filtered
        detector.estimation_report = _report_from_states(
            states,
            backend="injected",
            measure=config.input.measure,
            rv_window=config.input.rv_window,
            k_regimes=config.fit.k_regimes,
        )
        return detector

    def fit(
        self,
        returns: pd.Series | None = None,
        *,
        panel: pd.DataFrame | None = None,
        exogenous: pd.Series | None = None,
        exogenous_confirmed: pd.Series | None = None,
    ) -> RegimeEstimationReport:
        measure = self.config.input.measure
        if measure in {"log_rv", "log_rv_oas"}:
            if returns is None:
                raise RegimeError("returns are required for log realized variance")
            self.returns = _as_series(returns, "r")
            observed = realized_log_variance(self.returns, self.config.input.rv_window)
        elif measure == "rolling_pc1":
            if panel is None:
                raise RegimeError("panel is required for rolling_pc1")
            observed = rolling_first_pc(
                panel,
                window=self.config.input.pca_window,
                min_periods=self.config.input.pca_min_periods,
            )
            if returns is not None:
                self.returns = _as_series(returns, "r")
        else:
            raise RegimeError(f"unknown input measure {measure}")
        usable = observed.dropna()
        if usable.shape[0] < self.config.input.min_observations:
            raise RegimeError(
                f"need >= {self.config.input.min_observations} observations, "
                f"got {usable.shape[0]}"
            )
        return self.fit_observed(
            usable,
            exogenous=exogenous,
            exogenous_confirmed=exogenous_confirmed,
        )

    def fit_observed(
        self,
        observed: pd.Series,
        *,
        exogenous: pd.Series | None = None,
        exogenous_confirmed: pd.Series | None = None,
    ) -> RegimeEstimationReport:
        series = _as_series(observed, "observed").dropna()
        k = self.config.fit.k_regimes
        fitted = self.backend.fit(series, k_regimes=k, seed=self.config.fit.seed)
        self._fitted = fitted
        self.observed = series
        stress_id = fitted.stress_regime_id
        filtered = _probability_column(fitted.filtered, stress_id).reindex(series.index)
        smoothed = _probability_column(fitted.smoothed, stress_id).reindex(series.index)
        if filtered.isna().any() or smoothed.isna().any():
            raise RegimeError("backend returned NaN regime probabilities")
        self.filtered_stress = filtered.rename("p_stress_filtered")
        self.smoothed_stress = smoothed.rename("p_stress_smoothed")
        confirmed = _resolve_exogenous(
            self.config, filtered.index, exogenous, exogenous_confirmed
        )
        self.states = apply_anti_chattering(
            filtered,
            enter=self.config.hysteresis.enter,
            exit=self.config.hysteresis.exit,
            dwell=self.config.dwell.min_days,
            confirm_days=self.config.confirmation.consecutive_days,
            exogenous_confirmed=confirmed,
            partial_fraction=self.config.exogenous.partial_derisk_fraction,
        )
        report = _report_from_fitted(
            fitted,
            backend=self.backend.name,
            measure=self.config.input.measure,
            rv_window=self.config.input.rv_window,
            index=series.index,
        )
        if k == 3:
            dur = np.array(list(report.expected_durations.values()), dtype=float)
            pi = np.array(list(report.unconditional_probabilities.values()), dtype=float)
            assert_regimes_economically_valid(
                unconditional=pi,
                durations=dur,
                min_unconditional=self.config.k3.min_unconditional,
                min_duration_days=self.config.k3.min_expected_duration_days,
            )
            report = RegimeEstimationReport(**{**report.__dict__, "k3_valid": True})
        self.estimation_report = report
        stats = self.transition_stats()
        logger.info(
            "regime fit backend=%s k=%s n=%s transitions/year=%.2f alarm=%s",
            report.backend,
            report.k_regimes,
            report.n_obs,
            stats.transitions_per_year,
            stats.alarm,
        )
        if stats.alarm:
            logger.warning(
                "regime specification is over-reactive: %.2f transitions/year "
                "(alarm threshold %.1f). Do not tighten thresholds to hide this; "
                "re-examine the input series and the estimation window.",
                stats.transitions_per_year,
                stats.alarm_threshold,
            )
        return report

    def get_regime_probability(self, date: object, mode: RegimeMode = "filtered") -> float:
        self._require_fit()
        ts = self._require_date(date)
        if mode == "filtered":
            return float(self.filtered_stress.loc[ts])
        if mode == "smoothed":
            logger.warning(
                "smoothed probabilities use information after t ([C1-bis] look-ahead). "
                "Use mode='filtered' for operational signals and backtests."
            )
            return float(self.smoothed_stress.loc[ts])
        raise RegimeError(f"unknown probability mode {mode!r}; use 'filtered' or 'smoothed'")

    def get_regime_state(self, date: object) -> RegimeState:
        self._require_fit()
        ts = self._require_date(date)
        row = self.states.loc[ts]
        return RegimeState(
            date=ts,
            label=str(row["label"]),  # type: ignore[arg-type]
            derisk_fraction=float(row["derisk_fraction"]),
            p_stress_filtered=float(self.filtered_stress.loc[ts]),
            exogenous_confirms=bool(row["exogenous_confirms"]),
            days_in_regime=int(row["days_in_regime"]),
        )

    def transition_stats(self) -> TransitionStats:
        self._require_fit()
        assert self.states is not None
        labels = self.states["label"].astype(str)
        n = int(labels.shape[0])
        changes = labels.ne(labels.shift(1))
        changes.iloc[0] = False
        n_transitions = int(changes.sum())
        n_years = n / float(self.config.transitions.periods_per_year)
        tpy = n_transitions / n_years if n_years > 0.0 else float("inf")
        alarm = tpy > self.config.transitions.alarm_per_year
        cost = float(self.config.transitions.round_trip_cost_bps)
        episodes = _stress_episodes(labels)
        lags = _detection_lags(
            self.filtered_stress,
            labels,
            enter=self.config.hysteresis.enter,
        )
        durations = [end - start + 1 for start, end in episodes]
        derisk = self.states["derisk_fraction"].to_numpy(dtype=float)
        return TransitionStats(
            n_transitions=n_transitions,
            n_obs=n,
            n_years=n_years,
            transitions_per_year=float(tpy),
            alarm=bool(alarm),
            alarm_threshold=float(self.config.transitions.alarm_per_year),
            round_trip_cost_bps=cost,
            annual_friction_bps=float(tpy * cost),
            confirmation_delay_days=int(self.config.confirmation.consecutive_days),
            mean_detection_lag_days=float(np.mean(lags)) if lags else 0.0,
            n_stress_episodes=len(episodes),
            mean_stress_duration_days=float(np.mean(durations)) if durations else 0.0,
            n_partial_derisk_days=int(np.sum(np.isclose(derisk, self.config.exogenous.partial_derisk_fraction))),
            n_full_derisk_days=int(np.sum(np.isclose(derisk, 1.0))),
        )

    def plot(self, returns: pd.Series, dest: Path | None = None):
        self._require_fit()
        plot_cfg = self.config.plot
        out_dir = Path(plot_cfg.output_directory)
        dest = Path(dest) if dest is not None else out_dir / plot_cfg.filename
        os.environ.setdefault("MPLCONFIGDIR", str(out_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = _as_series(returns, "r").reindex(self.filtered_stress.index)
        fig, axes = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(plot_cfg.figsize_width, plot_cfg.figsize_height),
        )
        axes[0].plot(r.index, r.to_numpy(), color="0.35", lw=0.8)
        axes[0].axhline(0.0, color="0.7", lw=0.5)
        axes[0].set_ylabel("retorno")
        axes[0].set_title("Regímenes de volatilidad (probabilidad filtrada vs estado)")

        p = self.filtered_stress
        axes[1].plot(p.index, p.to_numpy(), color="tab:red", lw=1.0, label="P(estrés) filtrada")
        axes[1].axhline(self.config.hysteresis.enter, color="tab:red", ls="--", lw=0.8, label="entrar 0.70")
        axes[1].axhline(self.config.hysteresis.exit, color="tab:blue", ls="--", lw=0.8, label="salir 0.30")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_ylabel("P filtrada")
        axes[1].legend(loc="upper left", fontsize=8)

        step = self.states["derisk_fraction"]
        axes[2].step(step.index, step.to_numpy(), where="post", color="tab:purple", lw=1.2)
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_ylabel("estado (5 capas)")
        axes[2].set_xlabel("fecha")
        fig.tight_layout()
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=plot_cfg.dpi)
        return fig, dest

    def write_report(self, returns: pd.Series | None = None) -> RegimeReportPaths:
        self._require_fit()
        if returns is None:
            if self.returns is None:
                returns = self.filtered_stress
            else:
                returns = self.returns
        fig, plot_path = self.plot(returns)
        import matplotlib.pyplot as plt

        plt.close(fig)
        md_path = Path(self.config.output.report_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_render_markdown(self), encoding="utf-8")
        return RegimeReportPaths(plot_path=plot_path, report_markdown=md_path)

    def _require_fit(self) -> None:
        if self.filtered_stress is None or self.states is None:
            raise RegimeError("fit() or from_probabilities() is required first")

    def _require_date(self, date: object) -> pd.Timestamp:
        ts = pd.Timestamp(date).normalize()
        if ts not in self.filtered_stress.index:
            raise RegimeError(f"date {ts.date()} is not in the fitted sample")
        return ts


def _backend_from_config(config: RegimeConfig) -> RegimeBackend:
    name = config.fit.backend
    if name == "markov_log_variance":
        return MarkovLogVarianceBackend(config.fit)
    if name == "msgarch":
        return MSGARCHBackend()
    raise RegimeError(f"unknown regime backend {name!r}")


def _as_series(series: pd.Series, name: str) -> pd.Series:
    out = series.copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    out = out.sort_index().astype("float64")
    out.name = name
    if out.index.has_duplicates:
        raise RegimeError(f"duplicate dates in {name}")
    return out


def _align_bool(flag: pd.Series | None, index: pd.Index) -> pd.Series:
    if flag is None:
        logger.warning(
            "no exogenous confirmation series supplied; layer 4 treats stress as "
            "unconfirmed and applies partial de-risking only"
        )
        return pd.Series(False, index=index, dtype=bool)
    aligned = flag.reindex(index)
    values = []
    for value in aligned.to_numpy():
        if pd.isna(value):
            values.append(False)
        else:
            values.append(bool(value))
    return pd.Series(values, index=index, dtype=bool)


def _resolve_exogenous(
    config: RegimeConfig,
    index: pd.Index,
    exogenous: pd.Series | None,
    exogenous_confirmed: pd.Series | None,
) -> pd.Series:
    if exogenous is not None and exogenous_confirmed is not None:
        raise RegimeError("pass exogenous levels or a boolean mask, not both")
    if exogenous_confirmed is not None:
        return _align_bool(exogenous_confirmed, index)
    if exogenous is not None:
        flag = causal_exogenous_confirm(
            exogenous,
            percentile=config.exogenous.percentile,
            window=config.exogenous.window,
            min_periods=config.exogenous.min_periods,
        )
        return _align_bool(flag, index)
    return _align_bool(None, index)


def _probability_column(frame: pd.DataFrame, regime_id: int) -> pd.Series:
    if regime_id in frame.columns:
        col = frame[regime_id]
    else:
        col = frame.iloc[:, regime_id]
    return col.astype("float64")


def _named_order(stress_id: int, k: int) -> list[int]:
    others = [i for i in range(k) if i != stress_id]
    others_sorted = sorted(others, key=lambda i: i)
    if k == 2:
        return [others_sorted[0], stress_id]
    return [*others_sorted, stress_id]


def _regime_names(k: int) -> list[str]:
    if k == 2:
        return ["calm", "stress"]
    if k == 3:
        return ["calm", "mid", "stress"]
    raise RegimeError(f"unsupported k_regimes={k}")


def _report_from_fitted(
    fitted: FittedRegime,
    *,
    backend: str,
    measure: str,
    rv_window: int,
    index: pd.Index,
) -> RegimeEstimationReport:
    k = int(fitted.transition.shape[0])
    order = _named_order(fitted.stress_regime_id, k)
    names = _regime_names(k)
    transition = fitted.transition[np.ix_(order, order)]
    matrix = pd.DataFrame(transition, index=names, columns=names)
    durations = expected_durations(transition)
    pi = unconditional_probabilities(transition)
    params = fitted.params.iloc[order].copy()
    params.index = names
    return RegimeEstimationReport(
        backend=backend,
        k_regimes=k,
        n_obs=int(len(index)),
        start=pd.Timestamp(index[0]) if len(index) else None,
        end=pd.Timestamp(index[-1]) if len(index) else None,
        input_measure=measure,
        rv_window=rv_window,
        transition_matrix=matrix,
        expected_durations={name: float(durations[i]) for i, name in enumerate(names)},
        unconditional_probabilities={name: float(pi[i]) for i, name in enumerate(names)},
        regime_parameters=params,
        log_likelihood=float(fitted.log_likelihood),
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        stress_regime_id=int(fitted.stress_regime_id),
        k3_valid=None if k < 3 else True,
    )


def _report_from_states(
    states: pd.DataFrame,
    *,
    backend: str,
    measure: str,
    rv_window: int,
    k_regimes: int,
) -> RegimeEstimationReport:
    labels = states["label"].astype(str)
    names = ["calm", "stress"]
    counts = np.zeros((2, 2), dtype=float)
    mapped = labels.map({"calm": 0, "stress": 1}).to_numpy(dtype=int)
    for a, b in zip(mapped[:-1], mapped[1:], strict=True):
        counts[a, b] += 1.0
    transition = np.zeros((2, 2), dtype=float)
    for i in range(2):
        row = counts[i].sum()
        if row <= 0.0:
            transition[i, i] = 1.0
        else:
            transition[i] = counts[i] / row
    durations = expected_durations(transition)
    pi = unconditional_probabilities(transition)
    params = pd.DataFrame(
        {"const": [np.nan, np.nan], "sigma2": [np.nan, np.nan], "implied_rv": [np.nan, np.nan]},
        index=names,
    )
    index = states.index
    return RegimeEstimationReport(
        backend=backend,
        k_regimes=k_regimes,
        n_obs=int(len(index)),
        start=pd.Timestamp(index[0]),
        end=pd.Timestamp(index[-1]),
        input_measure=measure,
        rv_window=rv_window,
        transition_matrix=pd.DataFrame(transition, index=names, columns=names),
        expected_durations={name: float(durations[i]) for i, name in enumerate(names)},
        unconditional_probabilities={name: float(pi[i]) for i, name in enumerate(names)},
        regime_parameters=params,
        log_likelihood=float("nan"),
        aic=float("nan"),
        bic=float("nan"),
        stress_regime_id=1,
        k3_valid=None,
    )


def _stress_episodes(labels: pd.Series) -> list[tuple[int, int]]:
    episodes: list[tuple[int, int]] = []
    start: int | None = None
    for i, label in enumerate(labels.astype(str).to_numpy()):
        if label == "stress" and start is None:
            start = i
        elif label != "stress" and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, int(labels.shape[0]) - 1))
    return episodes


def _detection_lags(p_stress: pd.Series, labels: pd.Series, *, enter: float) -> list[int]:
    p = p_stress.to_numpy(dtype=float)
    lags: list[int] = []
    in_episode = False
    for i, label in enumerate(labels.astype(str).to_numpy()):
        if label == "stress" and not in_episode:
            in_episode = True
            j = i
            while j >= 0 and p[j] > enter:
                j -= 1
            lags.append(i - (j + 1))
        elif label != "stress":
            in_episode = False
    return lags


def _render_markdown(detector: RegimeDetector) -> str:
    report = detector.estimation_report
    stats = detector.transition_stats()
    assert report is not None
    lines = [
        "# Volatility regime report",
        "",
        "Backend: **Option A** (`statsmodels` MarkovRegression on log realized",
        "variance), not Haas–Mittnik–Paolella MS-GARCH. `arch` has no MS-GARCH.",
        "Option B (R MSGARCH / rpy2) is the v2 cross-check and is not used here.",
        "",
        "## Look-ahead ([C1-bis])",
        "",
        "Operational signals and backtests use **filtered** probabilities",
        r"\(P(s_t\mid y_{1:t})\). **Smoothed** probabilities",
        r"\(P(s_t\mid y_{1:T})\) use information after \(t\) and are look-ahead",
        "if treated as a trading input. `get_regime_probability(date)` defaults",
        "to `filtered`; `mode='smoothed'` logs a warning on every call.",
        "",
        f"- backend: `{report.backend}`",
        f"- K: {report.k_regimes}",
        f"- n: {report.n_obs}",
        f"- measure: `{report.input_measure}` (RV window={report.rv_window})",
        f"- sample: {report.start} → {report.end}",
        f"- logL / AIC / BIC: {report.log_likelihood:.4f} / {report.aic:.4f} / {report.bic:.4f}",
        "",
        r"## Transition matrix \(p_{ij}=P(s_t=j\mid s_{t-1}=i)\)",
        "",
        report.transition_matrix.to_string(),
        "",
        "## Expected duration (days)",
        "",
    ]
    for name, value in report.expected_durations.items():
        lines.append(f"- {name}: {value:.2f}")
    lines.extend(["", "## Unconditional probability", ""])
    for name, value in report.unconditional_probabilities.items():
        lines.append(f"- {name}: {value:.4f}")
    lines.extend(
        [
            "",
            "## Volatility parameters by regime",
            "",
            report.regime_parameters.to_string(),
            "",
            "## Layer-5 transition budget",
            "",
            f"- transitions: {stats.n_transitions}",
            f"- transitions per year: {stats.transitions_per_year:.2f}",
            f"- alarm threshold: {stats.alarm_threshold:.1f} (alarm={stats.alarm})",
            f"- round-trip cost: {stats.round_trip_cost_bps:.2f} bp",
            f"- annual friction: {stats.annual_friction_bps:.2f} bp",
            f"- confirmation delay N: {stats.confirmation_delay_days} days",
            f"- mean detection lag: {stats.mean_detection_lag_days:.2f} days",
            f"- stress episodes: {stats.n_stress_episodes}",
            f"- mean stress duration: {stats.mean_stress_duration_days:.2f} days",
            f"- partial de-risk days: {stats.n_partial_derisk_days}",
            f"- full de-risk days: {stats.n_full_derisk_days}",
            "",
            "If transitions/year exceed 8 in a two-regime model, the specification",
            "is over-reactive. Do not retune the 0.30/0.70 bands to hide it.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
