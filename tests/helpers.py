"""Deterministic synthetic DGPs and causal helpers for quantitative failure-mode tests.

Fixtures live in ``tests/fixtures/synthetic/`` so recovery and look-ahead tests
replay the same series on every machine ([C4]).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "synthetic"

TRUE_GJR = {
    "omega": 0.01,
    "alpha": 0.05,
    "gamma": 0.10,
    "beta": 0.85,
    "nu": 6.0,
    "mu": 0.0,
}

TRUE_EGARCH = {
    "omega": -0.10,
    "alpha": 0.15,
    "gamma": -0.12,
    "beta": 0.94,
    "mu": 0.0,
}

TRUE_REGIME = {
    "p_stay_calm": 0.97,
    "p_stay_stress": 0.93,
    "mu_calm": -2.2,
    "mu_stress": -0.4,
    "sigma_calm": 0.15,
    "sigma_stress": 0.25,
}


def bdates(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def simulate_gjr_garch(
    n: int,
    *,
    seed: int,
    omega: float = TRUE_GJR["omega"],
    alpha: float = TRUE_GJR["alpha"],
    gamma: float = TRUE_GJR["gamma"],
    beta: float = TRUE_GJR["beta"],
    nu: float = TRUE_GJR["nu"],
    mu: float = TRUE_GJR["mu"],
    start: str = "2010-01-04",
) -> pd.Series:
    """GJR-GARCH(1,1) with unit-variance Student-t innovations."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2.0))
    pers = alpha + 0.5 * gamma + beta
    sig2 = omega / (1.0 - pers)
    r = np.empty(n)
    for t in range(n):
        eps = np.sqrt(sig2) * z[t]
        r[t] = mu + eps
        sig2 = omega + (alpha + gamma * float(eps < 0.0)) * eps * eps + beta * sig2
    return pd.Series(r, index=bdates(n, start), name="r")


def simulate_egarch(
    n: int,
    *,
    seed: int,
    omega: float = TRUE_EGARCH["omega"],
    alpha: float = TRUE_EGARCH["alpha"],
    gamma: float = TRUE_EGARCH["gamma"],
    beta: float = TRUE_EGARCH["beta"],
    mu: float = TRUE_EGARCH["mu"],
    start: str = "2012-01-03",
) -> pd.Series:
    """Gaussian EGARCH(1,1). ``gamma < 0`` is the equity-style leverage effect."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    log_sig2 = omega / (1.0 - beta)
    r = np.empty(n)
    const = np.sqrt(2.0 / np.pi)
    for t in range(n):
        sigma = float(np.exp(0.5 * log_sig2))
        r[t] = mu + sigma * z[t]
        log_sig2 = omega + alpha * (abs(z[t]) - const) + gamma * z[t] + beta * log_sig2
    return pd.Series(r, index=bdates(n, start), name="r")


def simulate_two_regime_log_rv(
    n: int,
    *,
    seed: int,
    p_stay_calm: float = TRUE_REGIME["p_stay_calm"],
    p_stay_stress: float = TRUE_REGIME["p_stay_stress"],
    mu_calm: float = TRUE_REGIME["mu_calm"],
    mu_stress: float = TRUE_REGIME["mu_stress"],
    sigma_calm: float = TRUE_REGIME["sigma_calm"],
    sigma_stress: float = TRUE_REGIME["sigma_stress"],
    start: str = "2014-01-02",
) -> tuple[pd.Series, pd.Series]:
    """Observable log-RV proxy with a known two-state Markov chain.

    Returns ``(observed, true_stress)`` where ``true_stress`` is 1 in stress.
    """
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = p_stay_calm if state[t - 1] == 0 else p_stay_stress
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    mu = np.where(state == 1, mu_stress, mu_calm)
    sig = np.where(state == 1, sigma_stress, sigma_calm)
    y = mu + sig * rng.standard_normal(n)
    idx = bdates(n, start)
    observed = pd.Series(y, index=idx, name="log_rv")
    true_stress = pd.Series(state.astype(float), index=idx, name="true_stress")
    return observed, true_stress


def simulate_gaussian_garch(
    n: int,
    *,
    seed: int,
    omega: float = 1.0e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    mu: float = 0.0002,
    start: str = "2010-01-04",
) -> tuple[pd.Series, float]:
    """Gaussian GARCH(1,1). Returns the series and true σ_{T+1}."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    sig2 = omega / (1.0 - alpha - beta)
    r = np.empty(n)
    last_eps = 0.0
    last_sig2 = sig2
    for t in range(n):
        eps = np.sqrt(sig2) * z[t]
        r[t] = mu + eps
        last_eps, last_sig2 = eps, sig2
        sig2 = omega + alpha * eps * eps + beta * sig2
    sigma_next = float(np.sqrt(omega + alpha * last_eps**2 + beta * last_sig2))
    return pd.Series(r, index=bdates(n, start), name="r"), sigma_next


def gjr_filter(
    returns: pd.Series,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    mu: float = 0.0,
) -> pd.Series:
    """Causal GJR recursion. Initial variance is the unconditional value from params."""
    r = returns.astype(float)
    pers = alpha + 0.5 * gamma + beta
    sig2 = omega / (1.0 - pers)
    out = np.empty(r.size)
    for t, value in enumerate(r.to_numpy()):
        out[t] = np.sqrt(sig2)
        eps = value - mu
        sig2 = omega + (alpha + gamma * float(eps < 0.0)) * eps * eps + beta * sig2
    return pd.Series(out, index=r.index, name="sigma")


def one_step_gjr_sigma(
    last_eps: float,
    last_sigma: float,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> float:
    sig2 = omega + (alpha + gamma * float(last_eps < 0.0)) * last_eps**2 + beta * last_sigma**2
    return float(np.sqrt(sig2))


def causal_fhs_var(
    returns: pd.Series,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    mu: float = 0.0,
    var_alpha: float = 0.99,
) -> float:
    sigma = gjr_filter(returns, omega=omega, alpha=alpha, gamma=gamma, beta=beta, mu=mu)
    aligned = pd.concat({"r": returns.astype(float), "sigma": sigma}, axis=1).dropna()
    z = ((aligned["r"] - mu) / aligned["sigma"]).to_numpy()
    last_eps = float(aligned["r"].iloc[-1] - mu)
    last_sig = float(aligned["sigma"].iloc[-1])
    sig_next = one_step_gjr_sigma(
        last_eps, last_sig, omega=omega, alpha=alpha, gamma=gamma, beta=beta
    )
    q = float(np.quantile(z, 1.0 - var_alpha))
    return float(-(mu + sig_next * q))


def dislocation_panel(n: int = 80, *, start: str = "2020-01-02") -> pd.DataFrame:
    idx = bdates(n, start)
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "sigma_ebp": 0.05 + 0.008 * t,
            "sigma_oas": 0.20 + 0.002 * t,
            "ebp_level": 0.20 + 0.015 * t,
            "oas_level": 4.0 + 0.04 * t,
            "default_proxy": np.full(n, 1.50),
        },
        index=idx,
    )


def permute_future(series: pd.Series, t: pd.Timestamp, *, seed: int) -> pd.Series:
    """Shuffle values strictly after ``t``. Dates and past values stay put."""
    out = series.copy()
    future = out.index[out.index > t]
    if future.empty:
        return out
    rng = np.random.default_rng(seed)
    values = np.array(out.loc[future].to_numpy(), copy=True)
    rng.shuffle(values)
    out.loc[future] = values
    return out


def permute_future_frame(frame: pd.DataFrame, t: pd.Timestamp, *, seed: int) -> pd.DataFrame:
    out = frame.copy()
    for i, col in enumerate(out.columns):
        out[col] = permute_future(out[col], t, seed=seed + i)
    return out


def ensure_synthetic_fixtures() -> Path:
    """Write parquet fixtures once; subsequent runs load the same bytes."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    gjr_path = FIXTURE_DIR / "gjr_5000.parquet"
    if not gjr_path.exists():
        simulate_gjr_garch(5000, seed=1).to_frame().to_parquet(gjr_path)
    egarch_path = FIXTURE_DIR / "egarch_neg_gamma.parquet"
    if not egarch_path.exists():
        simulate_egarch(2500, seed=11).to_frame().to_parquet(egarch_path)
    regime_path = FIXTURE_DIR / "two_regime_log_rv.parquet"
    if not regime_path.exists():
        observed, true_stress = simulate_two_regime_log_rv(900, seed=21)
        pd.DataFrame({"log_rv": observed, "true_stress": true_stress}).to_parquet(regime_path)
    gauss_path = FIXTURE_DIR / "gaussian_garch_fhs.parquet"
    if not gauss_path.exists():
        r, sigma_next = simulate_gaussian_garch(1600, seed=17)
        r.to_frame().to_parquet(gauss_path)
        (FIXTURE_DIR / "gaussian_garch_sigma_next.txt").write_text(
            repr(sigma_next), encoding="utf-8"
        )
    oas_path = FIXTURE_DIR / "oas_levels_lookahead.parquet"
    if not oas_path.exists():
        r = simulate_egarch(400, seed=5)
        log_oas = np.log(4.5) + np.cumsum(-r.to_numpy() / 100.0)
        pd.Series(np.exp(log_oas), index=r.index, name="oas").to_frame().to_parquet(oas_path)
    asof_path = FIXTURE_DIR / "asof_misaligned.parquet"
    if not asof_path.exists():
        weekly = pd.Series(
            [1.0, 9.0],
            index=pd.to_datetime(["2024-01-03", "2024-01-17"]),
            name="NFCI",
        )
        weekly.to_frame().to_parquet(asof_path)
    return FIXTURE_DIR


def load_gjr_fixture() -> pd.Series:
    ensure_synthetic_fixtures()
    frame = pd.read_parquet(FIXTURE_DIR / "gjr_5000.parquet")
    return frame.iloc[:, 0].astype(float).rename("r")


def load_egarch_fixture() -> pd.Series:
    ensure_synthetic_fixtures()
    frame = pd.read_parquet(FIXTURE_DIR / "egarch_neg_gamma.parquet")
    return frame.iloc[:, 0].astype(float).rename("r")


def load_regime_fixture() -> tuple[pd.Series, pd.Series]:
    ensure_synthetic_fixtures()
    frame = pd.read_parquet(FIXTURE_DIR / "two_regime_log_rv.parquet")
    return frame["log_rv"].astype(float), frame["true_stress"].astype(float)


def load_gaussian_garch_fixture() -> tuple[pd.Series, float]:
    ensure_synthetic_fixtures()
    frame = pd.read_parquet(FIXTURE_DIR / "gaussian_garch_fhs.parquet")
    sigma_next = float((FIXTURE_DIR / "gaussian_garch_sigma_next.txt").read_text())
    return frame.iloc[:, 0].astype(float).rename("r"), sigma_next
