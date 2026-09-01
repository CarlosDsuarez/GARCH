"""[T5] Economic properties of VaR, ES, and the credit-stress leverage sign."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.oas_egarch import OASVolatilityModel, assert_leverage_sign
from risk.fhs import FHSEngine, _var_es_from_draws
from test_fhs import make_fhs_config, simulate_garch
from test_oas_egarch import _synthetic_oas, make_model_config


def test_t5_1_var_99_is_at_least_var_95(tmp_path: Path) -> None:
    r, _ = simulate_garch(220, seed=19)
    engine = FHSEngine(
        make_fhs_config(tmp_path, fhs={"alphas": [0.95, 0.99], "n_simulations": 8000, "seed": 19})
    ).fit(r)
    lo = engine.var_es(alpha=0.95, horizon=1)
    hi = engine.var_es(alpha=0.99, horizon=1)
    assert hi.var >= lo.var


def test_t5_2_es_dominates_var_at_the_same_alpha(tmp_path: Path) -> None:
    r, _ = simulate_garch(220, seed=20)
    engine = FHSEngine(make_fhs_config(tmp_path, fhs={"n_simulations": 6000})).fit(r)
    for alpha in (0.975, 0.99):
        report = engine.var_es(alpha=alpha, horizon=1)
        assert report.es >= report.var


def test_t5_3_es_is_subadditive_on_joint_gaussian_draws() -> None:
    rng = np.random.default_rng(31)
    cov = np.array([[1.0, 0.45], [0.45, 1.0]]) * (0.02**2)
    ab = rng.multivariate_normal([0.0, 0.0], cov, size=80_000)
    a, b = ab[:, 0], ab[:, 1]
    es_a = _var_es_from_draws(a, alpha=0.99, horizon=1, sigma_forecast=0.02)
    es_b = _var_es_from_draws(b, alpha=0.99, horizon=1, sigma_forecast=0.02)
    es_ab = _var_es_from_draws(a + b, alpha=0.99, horizon=1, sigma_forecast=0.02)
    assert es_ab.es <= es_a.es + es_b.es + 1e-12


def test_t5_3_var_violates_subadditivity_on_the_textbook_default_book() -> None:
    """Artzner: mutually exclusive 4% defaults; 95% VaR of the sum exceeds the sum of VaRs."""
    n = 100
    a = np.zeros(n)
    a[:4] = -1.0
    b = np.zeros(n)
    b[4:8] = -1.0
    var_a = _var_es_from_draws(a, alpha=0.95, horizon=1, sigma_forecast=1.0)
    var_b = _var_es_from_draws(b, alpha=0.95, horizon=1, sigma_forecast=1.0)
    var_ab = _var_es_from_draws(a + b, alpha=0.95, horizon=1, sigma_forecast=1.0)
    assert var_ab.var > var_a.var + var_b.var


def test_t5_4_conditional_vol_after_widening_exceeds_tightening(tmp_path: Path) -> None:
    oas = _synthetic_oas(n=400, seed=7)
    model = OASVolatilityModel(make_model_config(tmp_path, min_observations=80), series_id="BAMLH0A0HYM2")
    model.fit(oas)
    assert model.result is not None and model.stress is not None
    n_vol = len(model.result.conditional_volatility)
    vol = pd.Series(
        np.asarray(model.result.conditional_volatility),
        index=model.stress.r.index[-n_vol:],
    )
    stats = assert_leverage_sign(model.stress.r.reindex(vol.index), vol, model.config.sign_test)
    assert stats["mean_vol_after_widening"] > stats["mean_vol_after_tightening"]
