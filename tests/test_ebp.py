"""EBP ingest, Chow-Lin consistency, publication lag, and GJR-GARCH tests.

Synthetic CSV schema: ISO dates (YYYY-MM-DD), invented EBP levels
(e.g. 0.45). No official Fed vintages are stored here.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from conftest import make_test_config
from data.ebp import (
    EBPLoader,
    build_aggregation_matrix,
    build_indicator_matrix,
    chow_lin_disaggregate,
    denton_cholette_disaggregate,
    publication_date,
)
from data.schema import load_data_config
from models.ebp_garch import (
    EBPVolatilityModel,
    assert_difference_stationary,
    assert_gjr_constraints,
    build_ebp_stress_return,
    gjr_half_life,
    originate_signal,
    signal_may_originate,
    write_comparative_report,
)
from models.oas_egarch import ModelInvalidError, SignConventionError, assert_leverage_sign
from models.schema import EbpModelConfig, ModelConfig, load_model_config
from test_oas_egarch import _model_payload


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DATA = REPO_ROOT / "config" / "data.yaml"
PRODUCTION_PARAMS = REPO_ROOT / "config" / "params.yaml"


def _ebp_csv(rows: list[tuple[str, float, float]]) -> bytes:
    buf = BytesIO()
    buf.write(b"date,gz_spread,ebp\n")
    for date, gz, ebp in rows:
        buf.write(f"{date},{gz:.4f},{ebp:.4f}\n".encode("utf-8"))
    return buf.getvalue()


def _monthly_rows(start: str, periods: int, *, spike_month: str | None = None) -> list[tuple[str, float, float]]:
    idx = pd.date_range(start, periods=periods, freq="MS")
    rows: list[tuple[str, float, float]] = []
    for i, ts in enumerate(idx):
        ebp = 10.0 if spike_month and ts.strftime("%Y-%m") == spike_month else 0.40 + 0.01 * i
        rows.append((ts.strftime("%Y-%m-%d"), 2.50 + 0.02 * i, ebp))
    return rows


def _ebp_data_block() -> dict:
    return {
        "url": "https://example.test/ebp_csv.csv",
        "date_column": "date",
        "ebp_column": "ebp",
        "gz_spread_column": "gz_spread",
        "recession_prob_column": None,
        "cache_id": "EBP",
        "publication_lag_days": 45,
        "publication_lag_anchor": "month_end",
        "sensitivity_lags_days": [30, 45, 60],
        "sensitivity_min_correlation": 0.85,
        "aggregation": "average",
        "vix_series_id": "VIXCLS",
        "hy_oas_series_id": "BAMLH0A0HYM2",
        "t10y2y_series_id": "T10Y2Y",
    }


def _ebp_model_block(tmp_path: Path) -> dict:
    return {
        "invert_sign": True,
        "min_observations": 40,
        "stationarity": {
            "adf_pvalue_max": 0.05,
            "kpss_pvalue_min": 0.05,
            "adf_regression": "c",
            "kpss_regression": "c",
            "adf_maxlag": 8,
        },
        "variance": {
            "vol": "GARCH",
            "p": 1,
            "o": 1,
            "q": 1,
            "rescale": False,
            "ftol": 1.0e-6,
            "maxiter": 500,
            "half_life_mass": 0.5,
            "significance_level": 0.05,
            "expected_gjr_gamma_sign": 1,
            "stationarity_alpha_half_gamma_beta_max": 1.0,
        },
        "distribution": {
            "candidates": ["t"],
            "selection_ic": "bic",
            "forbidden": ["normal", "Normal"],
            "nu_fourth_moment_min": 4.0,
            "nu_normal_collapse": 20.0,
        },
        "mean": {
            "acf_lags": 10,
            "ljung_box_lags": 10,
            "ljung_box_pvalue_min": 0.05,
            "rho1_matrix_pricing_threshold": 0.15,
            "variance_ratio_horizons": [2, 5],
            "ar_lags": 1,
            "ma_lags": 1,
        },
        "forecast": {
            "horizons": [1, 3],
            "method": "simulation",
            "simulations": 200,
            "reindex": False,
        },
        "sign_test": {"tail_percentile": 10.0, "min_tail_observations": 12},
        "disaggregation": {
            "method": "chow_lin",
            "fallback": "denton_cholette",
            "include_constant": True,
            "rho_grid_min": -0.90,
            "rho_grid_max": 0.90,
            "rho_grid_size": 19,
            "condition_number_max": 1.0e12,
            "consistency_atol": 1.0e-10,
            "high_freq_rho_from_monthly": True,
        },
        "signal": {
            "primary_frequency": "monthly",
            "daily_may_originate": False,
            "robustness_min_correlation": 0.70,
        },
        "plot": {
            "output_directory": str(tmp_path / "ebp_plots"),
            "dpi": 80,
            "figsize_width": 8.0,
            "figsize_height": 4.0,
            "filename_template": "ebp_comparative_{layer}.png",
        },
        "output": {"comparative_table": str(tmp_path / "ebp_compare.csv")},
    }


def make_ebp_data_config(tmp_path: Path, **overrides):
    fred = {
        "BAMLH0A0HYM2": {
            "description": "HY OAS",
            "frequency": "daily",
            "series_type": "oas",
            "unit": "percent",
            "non_negative": True,
            "publication_lag_days": 0,
            "primary": True,
        },
        "VIXCLS": {
            "description": "VIX",
            "frequency": "daily",
            "series_type": "index",
            "unit": "index",
            "non_negative": True,
            "publication_lag_days": 0,
            "primary": False,
        },
        "T10Y2Y": {
            "description": "curve",
            "frequency": "daily",
            "series_type": "spread",
            "unit": "percent",
            "non_negative": False,
            "publication_lag_days": 0,
            "primary": False,
        },
    }
    payload = {"ebp": _ebp_data_block(), "fred_series": fred}
    payload.update(overrides)
    return make_test_config(tmp_path, **payload)


def make_ebp_model_config(tmp_path: Path, **overrides) -> ModelConfig:
    payload = _model_payload(tmp_path)
    payload["min_observations"] = 40
    payload["ebp"] = _ebp_model_block(tmp_path)
    if "ebp" in overrides and isinstance(overrides["ebp"], dict):
        payload["ebp"].update(overrides.pop("ebp"))
    payload.update(overrides)
    return ModelConfig.model_validate(payload)


class FakeHttp:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return self.body


def _loader(tmp_path: Path, rows: list[tuple[str, float, float]], **overrides) -> EBPLoader:
    config = make_ebp_data_config(tmp_path, **overrides)
    return EBPLoader(
        config,
        project_root=tmp_path,
        http_get=FakeHttp(_ebp_csv(rows)),
    )


# ---------------------------------------------------------------------------
# Production YAML
# ---------------------------------------------------------------------------


def test_production_data_config_has_ebp_lag_and_anchors() -> None:
    config = load_data_config(PRODUCTION_DATA)
    assert config.ebp is not None
    assert config.ebp.publication_lag_days == 45
    assert config.ebp.sensitivity_lags_days == [30, 45, 60]
    assert config.ebp.vix_series_id == "VIXCLS"
    assert config.ebp.hy_oas_series_id == "BAMLH0A0HYM2"
    assert config.ebp.t10y2y_series_id == "T10Y2Y"


def test_production_params_have_gjr_ebp_block() -> None:
    config = load_model_config(PRODUCTION_PARAMS)
    assert config.ebp is not None
    assert config.ebp.invert_sign is True
    assert config.ebp.variance.vol == "GARCH"
    assert config.ebp.variance.o == 1
    assert config.ebp.variance.expected_gjr_gamma_sign == 1
    assert config.ebp.signal.daily_may_originate is False
    assert config.ebp.min_observations >= 120


def test_config_rejects_daily_originating_signal(tmp_path: Path) -> None:
    payload = _ebp_model_block(tmp_path)
    payload["signal"]["daily_may_originate"] = True
    with pytest.raises(ValidationError, match="originate"):
        EbpModelConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Publication lag [C1]
# ---------------------------------------------------------------------------


def test_january_ebp_publishes_on_march_16_with_45_day_lag() -> None:
    pub = publication_date(pd.Timestamp("2020-01-01"), lag_days=45)
    assert pub == pd.Timestamp("2020-03-16")


def test_asof_excludes_unpublished_month(tmp_path: Path) -> None:
    rows = _monthly_rows("2020-01-01", 3)
    loader = _loader(tmp_path, rows)
    frame = loader.fetch()
    early = loader.available_asof("2020-03-01")
    late = loader.available_asof("2020-03-16")
    assert frame.ebp.loc[pd.Timestamp("2020-01-01")] == pytest.approx(0.40)
    assert pd.Timestamp("2020-01-01") not in early.index
    assert pd.Timestamp("2020-01-01") in late.index
    assert late.loc[pd.Timestamp("2020-01-01"), "ebp"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Chow-Lin / Denton
# ---------------------------------------------------------------------------


def _synthetic_monthly_and_daily(months: int = 6) -> tuple[pd.Series, pd.DataFrame]:
    daily_idx = pd.bdate_range("2020-01-01", periods=months * 22)
    t = np.arange(len(daily_idx), dtype=float)
    indicators = pd.DataFrame(
        {
            "ln_vix": np.log(15.0 + 3.0 * np.sin(t / 8.0)),
            "ln_hy": np.log(4.0 + 0.4 * np.cos(t / 11.0)),
            "t10y2y": 0.5 + 0.02 * np.sin(t / 15.0),
        },
        index=daily_idx,
    )
    true_daily = (
        0.20
        + 0.15 * indicators["ln_vix"]
        + 0.25 * indicators["ln_hy"]
        + 0.05 * indicators["t10y2y"]
    )
    true_daily.name = "ebp"
    periods = true_daily.index.to_period("M")
    monthly = true_daily.groupby(periods).mean()
    monthly.index = monthly.index.to_timestamp()
    monthly.name = "ebp"
    return monthly, indicators


@pytest.mark.blocking
def test_chow_lin_reproduces_monthly_average_within_1e_10() -> None:
    monthly, indicators = _synthetic_monthly_and_daily()
    result = chow_lin_disaggregate(monthly, indicators)
    C = build_aggregation_matrix(result.daily.index, monthly.index)
    replicated = C @ result.daily.to_numpy()
    np.testing.assert_allclose(replicated, monthly.to_numpy(), atol=1e-10)
    np.testing.assert_allclose(
        result.monthly_replicated.to_numpy(), monthly.to_numpy(), atol=1e-10
    )


def test_denton_cholette_reproduces_monthly_average_within_1e_10() -> None:
    monthly, indicators = _synthetic_monthly_and_daily()
    preliminary = indicators["ln_vix"] - indicators["ln_vix"].mean()
    daily = denton_cholette_disaggregate(monthly, preliminary)
    C = build_aggregation_matrix(daily.index, monthly.index)
    np.testing.assert_allclose(C @ daily.to_numpy(), monthly.to_numpy(), atol=1e-10)


def test_vix_only_daily_differs_when_hy_moves_independently() -> None:
    monthly, indicators = _synthetic_monthly_and_daily()
    indicators = indicators.copy()
    hy_shock = indicators["ln_hy"].copy()
    hy_shock.iloc[10:20] = hy_shock.iloc[10:20] + 1.5
    indicators["ln_hy"] = hy_shock
    full = chow_lin_disaggregate(monthly, indicators)
    vix_only = chow_lin_disaggregate(monthly, indicators[["ln_vix"]])
    assert not np.allclose(full.daily.to_numpy(), vix_only.daily.to_numpy())


def test_asof_disaggregation_drops_unpublished_months_and_future_days(tmp_path: Path) -> None:
    rows = _monthly_rows("2019-10-01", 8)
    loader = _loader(tmp_path, rows)
    loader.fetch()
    daily_idx = pd.bdate_range("2019-10-01", "2020-05-29")
    rng = np.random.default_rng(4)
    indicators = pd.DataFrame(
        {
            "ln_vix": np.log(18.0 + rng.normal(0, 0.05, size=len(daily_idx))),
            "ln_hy": np.log(5.0 + rng.normal(0, 0.03, size=len(daily_idx))),
            "t10y2y": rng.normal(0.2, 0.02, size=len(daily_idx)),
        },
        index=daily_idx,
    )
    asof = pd.Timestamp("2020-03-01")
    result = loader.disaggregate(indicators, asof=asof)
    assert result.daily.index.max() <= asof
    used_months = result.months_used
    assert pd.Timestamp("2020-01-01") not in used_months
    assert pd.Timestamp("2019-12-01") in used_months
    assert result.look_ahead is False


def test_full_sample_disaggregation_is_marked_look_ahead(tmp_path: Path) -> None:
    rows = _monthly_rows("2019-10-01", 6)
    loader = _loader(tmp_path, rows)
    loader.fetch()
    daily_idx = pd.bdate_range("2019-10-01", "2020-03-31")
    indicators = pd.DataFrame(
        {
            "ln_vix": np.log(18.0 + 0.01 * np.arange(len(daily_idx))),
            "ln_hy": np.log(5.0 + 0.005 * np.arange(len(daily_idx))),
            "t10y2y": np.full(len(daily_idx), 0.25),
        },
        index=daily_idx,
    )
    result = loader.disaggregate(indicators, descriptive_full_sample=True)
    assert result.look_ahead is True


# ---------------------------------------------------------------------------
# Lag sensitivity
# ---------------------------------------------------------------------------


def test_lag_sensitivity_flags_fragile_publication_timing(tmp_path: Path) -> None:
    rows = _monthly_rows("2019-10-01", 8, spike_month="2020-01")
    loader = _loader(tmp_path, rows)
    loader.fetch()
    asof_index = pd.bdate_range("2020-02-15", "2020-04-15")
    report = loader.lag_sensitivity(asof_index)
    assert report.lags == [30, 45, 60]
    assert report.fragile is True


# ---------------------------------------------------------------------------
# Sign convention / stationarity / GJR constraints
# ---------------------------------------------------------------------------


def test_rising_ebp_maps_to_negative_stress_return(tmp_path: Path) -> None:
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    ebp = pd.Series([0.40, 0.55, 0.30], index=idx, name="ebp")
    config = make_ebp_model_config(tmp_path)
    stress = build_ebp_stress_return(ebp, config)
    assert stress.r.loc[pd.Timestamp("2020-02-29")] < 0
    assert stress.r.loc[pd.Timestamp("2020-03-31")] > 0
    assert stress.r.iloc[0] == pytest.approx(-(0.55 - 0.40))


def test_leverage_sign_raises_when_vol_higher_after_ebp_compression(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-02", periods=200)
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 1, size=len(idx)), index=idx)
    vol = pd.Series(1.0, index=idx)
    q = r.quantile(0.90)
    for t in r.index[r >= q]:
        nxt = t + pd.tseries.offsets.BDay()
        if nxt in vol.index:
            vol.loc[nxt] = 3.0
    config = make_ebp_model_config(tmp_path)
    with pytest.raises(SignConventionError):
        assert_leverage_sign(r, vol, config.ebp.sign_test)


def test_stationarity_gate_accepts_i1_rejects_i2(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2000-01-03", periods=400)
    i1 = pd.Series(np.cumsum(rng.normal(0, 1, size=400)), index=idx)
    spec = make_ebp_model_config(tmp_path).ebp.stationarity
    assert_difference_stationary(i1, spec)
    i2 = pd.Series(np.cumsum(i1.to_numpy()), index=idx)
    with pytest.raises(ModelInvalidError, match="stationary"):
        assert_difference_stationary(i2, spec)


def test_gjr_constraint_checker() -> None:
    class _Spec:
        stationarity_alpha_half_gamma_beta_max = 1.0

    assert_gjr_constraints(0.02, 0.05, 0.10, 0.80, _Spec())
    with pytest.raises(ModelInvalidError):
        assert_gjr_constraints(0.02, 0.05, 0.10, 0.95, _Spec())
    with pytest.raises(ModelInvalidError):
        assert_gjr_constraints(-0.01, 0.05, 0.10, 0.80, _Spec())
    with pytest.raises(ModelInvalidError):
        assert_gjr_constraints(0.02, 0.05, -0.10, 0.80, _Spec())


def test_gjr_half_life_uses_persistence_not_egarch_beta() -> None:
    class _Spec:
        half_life_mass = 0.5
        stationarity_alpha_half_gamma_beta_max = 1.0

    persistence = 0.05 + 0.10 / 2.0 + 0.80
    expected = float(np.log(0.5) / np.log(persistence))
    assert gjr_half_life(0.05, 0.10, 0.80, _Spec()) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Signal governance / comparative report
# ---------------------------------------------------------------------------


def test_ebp_volatility_model_requires_ebp_block(tmp_path: Path) -> None:
    config = ModelConfig.model_validate(_model_payload(tmp_path))
    with pytest.raises(ModelInvalidError, match="ebp"):
        EBPVolatilityModel(config)


def test_daily_layer_cannot_originate_signal(tmp_path: Path) -> None:
    config = make_ebp_model_config(tmp_path)
    assert signal_may_originate("primary_monthly", config) is True
    assert signal_may_originate("daily_full", config) is False
    assert signal_may_originate("daily_vix_only", config) is False
    with pytest.raises(ModelInvalidError, match="timing"):
        originate_signal("daily_full", fired=True, config=config)


def test_comparative_report_has_three_aligned_columns(tmp_path: Path) -> None:
    idx = pd.bdate_range("2020-01-02", periods=40)
    monthly = pd.Series(0.20 + 0.01 * np.arange(40), index=idx, name="monthly_official")
    daily_full = monthly * 1.02
    daily_vix = monthly * 0.97
    config = make_ebp_model_config(tmp_path)
    report = write_comparative_report(
        monthly_official=monthly,
        daily_full_anchor=daily_full,
        daily_vix_only=daily_vix,
        config=config,
    )
    assert list(report.table.columns) == [
        "monthly_official",
        "daily_full_anchor",
        "daily_vix_only",
    ]
    assert report.plot_path.exists()
    assert report.table.shape[0] == 40


def test_indicator_matrix_logs_vix_and_hy() -> None:
    idx = pd.bdate_range("2020-01-02", periods=5)
    vix = pd.Series([15.0, 16.0, 17.0, 16.5, 15.5], index=idx)
    hy = pd.Series([5.0, 5.2, 5.1, 5.4, 5.3], index=idx)
    slope = pd.Series([0.2, 0.21, 0.19, 0.18, 0.20], index=idx)
    full = build_indicator_matrix(vix, hy, slope, vix_only=False)
    only = build_indicator_matrix(vix, hy, slope, vix_only=True)
    assert list(full.columns) == ["ln_vix", "ln_hy", "t10y2y"]
    assert list(only.columns) == ["ln_vix"]
    assert full["ln_vix"].iloc[0] == pytest.approx(np.log(15.0))
    assert full["ln_hy"].iloc[0] == pytest.approx(np.log(5.0))
