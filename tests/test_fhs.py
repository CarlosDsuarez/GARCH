"""Filtered Historical Simulation: GJR filter, residual bootstrap, no sqrt-h.

Synthetic schema: ISO dates (YYYY-MM-DD), invented simple returns (~0.0004).
No live ETF or CCP margin files are stored here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy import stats as scipy_stats

from risk.fhs import (
    FHSEngine,
    TCopula,
    TCopulaFHSEngine,
    portfolio_returns,
)
from risk.schema import FHSConfig, load_fhs_config


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_RISK = REPO_ROOT / "config" / "risk.yaml"


def _fhs_payload(tmp_path: Path, **overrides) -> dict:
    payload = {
        "filter": {
            "min_observations": 80,
            "dist": "skewt",
            "mean": "constant",
            "ljung_box_lags": 10,
            "ljung_box_pvalue_min": 0.05,
            "p": 1,
            "o": 1,
            "q": 1,
            "rescale": False,
            "ftol": 1.0e-5,
            "maxiter": 400,
            "ar_lags": 1,
        },
        "fhs": {
            "alphas": [0.975, 0.99],
            "n_simulations": 2000,
            "seed": 7,
            "aggregation": "simple",
            "scale_sqrt_h": False,
            "default_horizon": 1,
        },
        "comparison": {"historical_window": 60, "parametric": "normal"},
        "plot": {
            "output_directory": str(tmp_path / "fhs_plots"),
            "filename": "fhs_comparison.png",
            "dpi": 80,
            "figsize_width": 8.0,
            "figsize_height": 4.0,
        },
        "output": {
            "report_markdown": str(tmp_path / "fhs_report.md"),
            "report_html": str(tmp_path / "fhs_report.html"),
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


def make_fhs_config(tmp_path: Path, **overrides) -> FHSConfig:
    return FHSConfig.model_validate(_fhs_payload(tmp_path, **overrides))


def simulate_garch(
    n: int,
    *,
    omega: float = 1.0e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    mu: float = 0.0002,
    seed: int = 3,
) -> tuple[pd.Series, float]:
    """Gaussian GARCH(1,1). Returns the series and the true σ_{T+1}."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    sig2 = np.empty(n)
    r = np.empty(n)
    sig2[0] = omega / (1.0 - alpha - beta)
    for t in range(n):
        eps = np.sqrt(sig2[t]) * z[t]
        r[t] = mu + eps
        if t + 1 < n:
            sig2[t + 1] = omega + alpha * eps * eps + beta * sig2[t]
    sig2_next = omega + alpha * (r[-1] - mu) ** 2 + beta * sig2[-1]
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.Series(r, index=idx, name="r"), float(np.sqrt(sig2_next))


def test_production_risk_config_is_fhs_not_sqrt_h() -> None:
    config = load_fhs_config(PRODUCTION_RISK)
    assert config.filter.min_observations >= 1000
    assert config.filter.dist == "skewt"
    assert 0.975 in config.fhs.alphas
    assert 0.99 in config.fhs.alphas
    assert config.fhs.n_simulations >= 10_000
    assert config.fhs.aggregation == "simple"
    assert config.fhs.scale_sqrt_h is False
    assert config.fhs.seed is not None
    assert config.comparison.historical_window == 250


def test_config_rejects_sqrt_h_scaling(tmp_path: Path) -> None:
    payload = _fhs_payload(tmp_path)
    payload["fhs"]["scale_sqrt_h"] = True
    with pytest.raises(ValidationError, match="sqrt"):
        FHSConfig.model_validate(payload)


def test_fit_rejects_short_sample(tmp_path: Path) -> None:
    cfg = make_fhs_config(tmp_path)
    r = pd.Series(np.linspace(-0.01, 0.01, 20))
    engine = FHSEngine(cfg)
    with pytest.raises(Exception, match="1000|min_observations|insufficient"):
        engine.fit(r)


def test_standardized_residuals_are_rescaled_to_unit_sample_std(tmp_path: Path) -> None:
    r, _ = simulate_garch(220, seed=5)
    engine = FHSEngine(make_fhs_config(tmp_path)).fit(r)
    assert engine.z is not None
    assert engine.z.std(ddof=1) == pytest.approx(1.0, abs=1e-10)


def test_var_and_es_are_positive_loss_magnitudes(tmp_path: Path) -> None:
    r, _ = simulate_garch(220, seed=6)
    engine = FHSEngine(make_fhs_config(tmp_path)).fit(r)
    both = engine.var_es(horizon=1)
    assert set(both) >= {0.975, 0.99}
    for alpha, report in both.items():
        assert report.var > 0.0
        assert report.es >= report.var
        assert report.alpha == alpha
        assert report.horizon == 1


def test_one_day_sigma_uses_observed_gjr_recursion(tmp_path: Path) -> None:
    r, _ = simulate_garch(220, seed=8)
    engine = FHSEngine(make_fhs_config(tmp_path)).fit(r)
    last_eps = float(engine.eps.iloc[-1])
    last_sig2 = float(engine.sigma.iloc[-1] ** 2)
    expected = engine.omega + (
        engine.alpha + engine.gamma * float(last_eps < 0.0)
    ) * last_eps**2 + engine.beta * last_sig2
    assert engine.sigma_one_step**2 == pytest.approx(expected, rel=1e-10)


def test_multi_day_var_is_not_sqrt_h_scaled(tmp_path: Path) -> None:
    r, _ = simulate_garch(260, omega=2.0e-6, alpha=0.12, beta=0.80, seed=9)
    engine = FHSEngine(make_fhs_config(tmp_path, fhs={"n_simulations": 4000})).fit(r)
    one = engine.var_es(alpha=0.99, horizon=1)
    five = engine.var_es(alpha=0.99, horizon=5)
    sqrt_scaled = one.var * np.sqrt(5.0)
    assert five.var != pytest.approx(sqrt_scaled, rel=1e-8)
    paths = engine.simulate_paths(horizon=5)
    assert paths.daily.shape == (4000, 5)
    assert paths.aggregated.shape == (4000,)
    assert paths.aggregation == "simple"
    rebuilt = np.prod(1.0 + paths.daily, axis=1) - 1.0
    np.testing.assert_allclose(paths.aggregated, rebuilt, atol=1e-12)


def test_fhs_var_matches_rescaled_residual_quantile(tmp_path: Path) -> None:
    r, _ = simulate_garch(220, seed=11)
    engine = FHSEngine(make_fhs_config(tmp_path, fhs={"n_simulations": 12_000})).fit(r)
    report = engine.var_es(alpha=0.99, horizon=1)
    z_q = float(np.quantile(engine.z.to_numpy(), 0.01))
    expected = -(engine.mu_one_step + engine.sigma_one_step * z_q)
    assert report.var == pytest.approx(expected, rel=0.08, abs=1e-4)


@pytest.mark.blocking
def test_fhs_recovers_analytical_normal_var() -> None:
    """Full-engine recovery: Gaussian GARCH → FHS VaR ≈ Φ VaR (MC + sample error)."""
    tmp = Path("/tmp")
    n = 1600
    mu = 0.0002
    r, sigma_next = simulate_garch(n, mu=mu, seed=17)
    cfg = make_fhs_config(
        tmp,
        filter={"min_observations": 1000, "maxiter": 800},
        fhs={"n_simulations": 10_000, "seed": 17},
    )
    engine = FHSEngine(cfg).fit(r)
    report = engine.var_es(alpha=0.99, horizon=1)
    analytic = -(mu + sigma_next * scipy_stats.norm.ppf(0.01))
    z_p = float(scipy_stats.norm.ppf(0.01))
    density = float(scipy_stats.norm.pdf(z_p))
    se = sigma_next * np.sqrt(0.01 * 0.99 / n) / density
    assert report.var == pytest.approx(analytic, abs=4.0 * se + 0.25 * analytic)


def test_same_seed_reproduces_var(tmp_path: Path) -> None:
    r, _ = simulate_garch(200, seed=2)
    cfg = make_fhs_config(tmp_path)
    a = FHSEngine(cfg).fit(r).var_es(alpha=0.99, horizon=1).var
    b = FHSEngine(cfg).fit(r).var_es(alpha=0.99, horizon=1).var
    assert a == pytest.approx(b, abs=1e-15)


def test_ljung_box_rejection_escalates_mean_to_ar1(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 280
    e = rng.normal(0, 0.01, size=n)
    r = np.empty(n)
    r[0] = e[0]
    for t in range(1, n):
        r[t] = 0.55 * r[t - 1] + e[t]
    series = pd.Series(r, index=pd.bdate_range("2012-01-02", periods=n))
    engine = FHSEngine(make_fhs_config(tmp_path)).fit(series)
    assert engine.mean_spec == "AR(1)"


def test_route_a_applies_current_weights_to_full_history(tmp_path: Path) -> None:
    idx = pd.bdate_range("2015-01-02", periods=180)
    rng = np.random.default_rng(1)
    assets = pd.DataFrame(
        {
            "HYG": rng.normal(0.0003, 0.007, size=180),
            "LQD": rng.normal(0.0002, 0.004, size=180),
        },
        index=idx,
    )
    weights = pd.Series({"HYG": 0.6, "LQD": 0.4})
    port = portfolio_returns(assets, weights)
    expected = 0.6 * assets["HYG"] + 0.4 * assets["LQD"]
    pd.testing.assert_series_equal(port, expected, check_names=False)
    engine = FHSEngine(make_fhs_config(tmp_path)).fit_portfolio(assets, weights)
    assert engine.returns is not None
    pd.testing.assert_series_equal(engine.returns, expected, check_names=False)


def test_t_copula_recovers_dependence() -> None:
    rng = np.random.default_rng(12)
    rho = 0.65
    nu = 6.0
    n = 800
    cov = np.array([[1.0, rho], [rho, 1.0]])
    g = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    chi = rng.chisquare(nu, size=n) / nu
    x = g / np.sqrt(chi)[:, None]
    u = scipy_stats.t.cdf(x, df=nu)
    copula = TCopula().fit(u)
    assert copula.corr[0, 1] == pytest.approx(rho, abs=0.12)
    sim = copula.simulate(4_000, rng)
    assert sim.shape == (4_000, 2)
    assert np.all((sim > 0.0) & (sim < 1.0))


def test_route_b_copula_engine_returns_positive_var(tmp_path: Path) -> None:
    idx = pd.bdate_range("2014-01-02", periods=220)
    rng = np.random.default_rng(21)
    common = rng.standard_t(8, size=220) * 0.004
    assets = pd.DataFrame(
        {
            "A": 0.0003 + common + rng.normal(0, 0.003, size=220),
            "B": 0.0002 + 0.8 * common + rng.normal(0, 0.003, size=220),
        },
        index=idx,
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})
    engine = TCopulaFHSEngine(make_fhs_config(tmp_path, fhs={"n_simulations": 1500}))
    engine.fit(assets)
    report = engine.var_es(weights, alpha=0.99, horizon=1)
    assert report.var > 0.0
    assert report.es >= report.var
    paths = engine.simulate_paths(weights, horizon=3)
    assert paths.daily.shape[1] == 3
    assert paths.aggregation == "simple"


def test_comparison_report_marks_violations_and_three_var_series(tmp_path: Path) -> None:
    r, _ = simulate_garch(240, seed=13)
    engine = FHSEngine(make_fhs_config(tmp_path)).fit(r)
    paths = engine.write_report(r)
    text = Path(paths.markdown).read_text(encoding="utf-8")
    assert "FHS" in text
    assert "250" in text or "histórico" in text.lower() or "historico" in text.lower()
    assert "normal" in text.lower()
    assert "viol" in text.lower()
    assert paths.html.exists()
    assert paths.plot.exists()
    table = engine.comparison_table
    assert table is not None
    assert {"var_fhs", "var_historical", "var_normal", "return", "violation_fhs"} <= set(
        table.columns
    )
