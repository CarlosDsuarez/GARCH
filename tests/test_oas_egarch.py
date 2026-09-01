"""Tests for credit-stress transform, matrix-pricing mean, EGARCH, and sign convention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from models.oas_egarch import (
    ModelInvalidError,
    OASVolatilityModel,
    SignConventionError,
    _ARMA11,
    assert_leverage_sign,
    build_credit_stress_return,
    half_life_days,
    load_model_config,
    lo_mackinlay_variance_ratio,
    pre_mean_diagnostics,
)
from models.schema import ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PARAMS = REPO_ROOT / "config" / "params.yaml"


def _model_payload(tmp_path: Path, **overrides) -> dict:
    payload = {
        "seed": 7,
        "transform": {"percent_scale": 100.0, "invert_sign": True},
        "mean": {
            "acf_lags": 10,
            "ljung_box_lags": 10,
            "ljung_box_pvalue_min": 0.05,
            "rho1_matrix_pricing_threshold": 0.15,
            "variance_ratio_horizons": [2, 5, 10],
            "ar_lags": 1,
            "ma_lags": 1,
        },
        "variance": {
            "vol": "EGARCH",
            "p": 1,
            "o": 1,
            "q": 1,
            "rescale": False,
            "ftol": 1.0e-6,
            "maxiter": 500,
            "half_life_mass": 0.5,
            "stationarity_abs_beta_max": 1.0,
            "typical_beta_min": 0.95,
            "typical_beta_max": 0.99,
            "significance_level": 0.05,
            "expected_egarch_gamma_sign": -1,
        },
        "distribution": {
            "candidates": ["t", "skewt"],
            "selection_ic": "bic",
            "forbidden": ["normal", "Normal"],
            "nu_fourth_moment_min": 4.0,
            "nu_normal_collapse": 20.0,
        },
        "forecast": {
            "horizons": [1, 5, 10],
            "method": "simulation",
            "simulations": 200,
            "reindex": False,
        },
        "sign_test": {"tail_percentile": 10.0, "min_tail_observations": 15},
        "min_observations": 80,
        "oas_universe": {
            "BAMLH0A0HYM2": {"label": "HY", "quality_rank": 4},
            "BAMLC0A0CM": {"label": "IG", "quality_rank": 1},
            "BAMLC0A4CBBB": {"label": "BBB", "quality_rank": 2},
            "BAMLH0A3HYC": {"label": "CCC", "quality_rank": 5},
            "BAMLEMCBPIOAS": {"label": "EM", "quality_rank": 3},
        },
        "plot": {
            "output_directory": str(tmp_path / "plots"),
            "dpi": 80,
            "figsize_width": 8.0,
            "figsize_height": 4.0,
            "filename_template": "{series_id}_conditional_vol.png",
        },
        "output": {"comparative_table": str(tmp_path / "panel.csv")},
    }
    payload.update(overrides)
    return payload


def make_model_config(tmp_path: Path, **overrides) -> ModelConfig:
    return ModelConfig.model_validate(_model_payload(tmp_path, **overrides))


def test_production_params_forbid_normal_and_list_five_oas() -> None:
    config = load_model_config(PRODUCTION_PARAMS)
    assert config.variance.rescale is False
    assert config.forecast.method == "simulation"
    assert set(config.oas_universe) == {
        "BAMLH0A0HYM2",
        "BAMLC0A0CM",
        "BAMLC0A4CBBB",
        "BAMLH0A3HYC",
        "BAMLEMCBPIOAS",
    }
    assert "normal" in {name.lower() for name in config.distribution.forbidden}
    assert config.min_observations >= 1000
    assert config.transform.invert_sign is True


def test_config_rejects_normal_candidate(tmp_path: Path) -> None:
    payload = _model_payload(tmp_path)
    payload["distribution"]["candidates"] = ["normal", "t"]
    with pytest.raises(ValidationError, match="normal"):
        ModelConfig.model_validate(payload)


def test_widening_spread_maps_to_negative_stress_return(tmp_path: Path) -> None:
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    oas = pd.Series([3.00, 3.30, 3.10], index=idx, name="BAMLH0A0HYM2")
    config = make_model_config(tmp_path)
    stress = build_credit_stress_return(oas, config)
    assert stress.r.loc[pd.Timestamp("2020-01-03")] < 0
    assert stress.r.loc[pd.Timestamp("2020-01-06")] > 0
    expected = -100.0 * (np.log(3.30) - np.log(3.00))
    assert stress.r.iloc[0] == pytest.approx(expected)


def test_build_drops_nan_before_log_diff(tmp_path: Path) -> None:
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    oas = pd.Series([3.00, np.nan, 3.30], index=idx)
    stress = build_credit_stress_return(oas, make_model_config(tmp_path))
    assert stress.n_dropped_nan == 1
    assert len(stress.r) == 1
    assert not np.isclose(stress.r.iloc[0], 0.0)


def test_build_rejects_nonpositive_oas(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-02", periods=5)
    oas = pd.Series([3.0, 3.1, 0.0, 3.2, 3.1], index=idx)
    with pytest.raises(ModelInvalidError, match="non-positive"):
        build_credit_stress_return(oas, make_model_config(tmp_path))


def test_leverage_sign_test_raises_when_vol_higher_after_tightening(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-02", periods=200)
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 1, size=len(idx)), index=idx)
    vol = pd.Series(1.0, index=idx)
    vol.iloc[1:] = np.where(r.iloc[:-1].to_numpy() > 0, 3.0, 1.0)
    config = make_model_config(tmp_path)
    with pytest.raises(SignConventionError, match="sign"):
        assert_leverage_sign(r, vol, config.sign_test)


def test_leverage_sign_test_passes_when_vol_higher_after_widening(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-02", periods=200)
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 1, size=len(idx)), index=idx)
    vol = pd.Series(1.0, index=idx)
    vol.iloc[1:] = np.where(r.iloc[:-1].to_numpy() < 0, 3.0, 1.0)
    assert_leverage_sign(r, vol, make_model_config(tmp_path).sign_test)


def test_lo_mackinlay_vr_exceeds_one_under_positive_ar(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    n = 2000
    e = rng.normal(0, 1, size=n)
    r = np.empty(n)
    r[0] = e[0]
    phi = 0.35
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    series = pd.Series(r, index=pd.bdate_range("2000-01-03", periods=n))
    report = lo_mackinlay_variance_ratio(
        series, horizons=make_model_config(tmp_path).mean.variance_ratio_horizons
    )
    vr2 = report.set_index("q").loc[2, "vr"]
    assert vr2 > 1.0


def test_half_life_matches_documented_formula(tmp_path: Path) -> None:
    config = make_model_config(tmp_path)
    beta = 0.97
    expected = np.log(config.variance.half_life_mass) / np.log(beta)
    assert half_life_days(beta, config.variance) == pytest.approx(expected)


def test_half_life_rejects_nonstationary_beta(tmp_path: Path) -> None:
    with pytest.raises(ModelInvalidError, match="stationar"):
        half_life_days(1.02, make_model_config(tmp_path).variance)


def test_pre_mean_diagnostics_flag_matrix_pricing(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n = 800
    e = rng.normal(0, 1, size=n)
    r = np.empty(n)
    r[0] = e[0]
    for t in range(1, n):
        r[t] = 0.25 * r[t - 1] + e[t]
    series = pd.Series(r, index=pd.bdate_range("2010-01-04", periods=n))
    diag = pre_mean_diagnostics(series, make_model_config(tmp_path).mean)
    assert diag.rho_1 > 0.15
    assert diag.matrix_pricing_contaminated is True
    assert diag.acf[1] == pytest.approx(diag.rho_1)


def _synthetic_oas(n: int = 400, seed: int = 7) -> pd.Series:
    """OAS path whose stress returns have EGARCH leverage (gamma < 0 on r)."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    log_sig2 = 0.0
    r[0] = 0.0
    for t in range(1, n):
        z = rng.standard_t(8)
        z = np.clip(z, -6, 6) / np.sqrt(8 / 6)
        log_sig2 = -0.15 + 0.12 * (abs(z) - np.sqrt(2 / np.pi)) - 0.20 * z + 0.94 * log_sig2
        sigma = float(np.exp(0.5 * log_sig2))
        r[t] = 0.15 * r[t - 1] + sigma * z
    dlog = -r / 100.0
    log_oas = np.log(4.5) + np.cumsum(dlog)
    idx = pd.bdate_range("2015-01-02", periods=n)
    return pd.Series(np.exp(log_oas), index=idx, name="BAMLH0A0HYM2")


def test_fit_egarch_on_synthetic_credit_stress(tmp_path: Path) -> None:
    oas = _synthetic_oas()
    config = make_model_config(tmp_path)
    model = OASVolatilityModel(config, series_id="BAMLH0A0HYM2")
    model.fit(oas)
    report = model.summary()
    assert report.converged is True
    assert report.dist in {"t", "skewt"}
    assert report.dist.lower() != "normal"
    assert report.half_life_days > 0
    assert report.residual_ljung_box_pvalue > config.mean.ljung_box_pvalue_min
    params = report.params.set_index("name")
    assert any("gamma" in n for n in params.index)
    fcast = model.forecast()
    assert fcast.seed == config.seed
    assert fcast.method == "simulation"
    assert 1 in fcast.volatility
    assert fcast.simulations == config.forecast.simulations


def test_fit_rejects_short_sample(tmp_path: Path) -> None:
    oas = _synthetic_oas(n=30)
    model = OASVolatilityModel(make_model_config(tmp_path), series_id="BAMLH0A0HYM2")
    with pytest.raises(ModelInvalidError, match="observations"):
        model.fit(oas)


def test_arma11_mean_bounds_match_num_params() -> None:
    """arch.fit concatenates mean bounds with vol/dist; a spare theta bound
    makes SLSQP raise 'number of bounds is not compatible with x0' (CI 3.11).
    """
    from arch.univariate import EGARCH, StudentsT

    y = pd.Series(np.linspace(-0.2, 0.2, 80), index=pd.bdate_range("2015-01-02", periods=80))
    model = _ARMA11(y, volatility=EGARCH(p=1, o=1, q=1), distribution=StudentsT(), rescale=False)
    assert len(model.bounds()) == model.num_params
    assert model.parameter_names()[-1] == "theta"
    assert model.bounds()[-1] == (-0.999, 0.999)
