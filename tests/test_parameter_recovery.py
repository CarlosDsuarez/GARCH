"""[T1] Parameter recovery on simulated data with known truth.

Dangerous GARCH bugs do not raise. They return plausible, biased parameters.
These tests check that the estimator recovers the DGP, not that it converges.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from arch.univariate import arch_model

from helpers import (
    TRUE_EGARCH,
    TRUE_GJR,
    TRUE_REGIME,
    load_egarch_fixture,
    load_gjr_fixture,
    load_regime_fixture,
    simulate_gjr_garch,
)
from models.oas_egarch import OASVolatilityModel
from models.regime import RegimeDetector
from test_oas_egarch import make_model_config
from test_regime import make_regime_config


def _gjr_estimates(series: pd.Series):
    model = arch_model(
        series.astype(float),
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="t",
        rescale=False,
    )
    start = np.array(
        [
            TRUE_GJR["mu"],
            TRUE_GJR["omega"],
            TRUE_GJR["alpha"],
            TRUE_GJR["gamma"],
            TRUE_GJR["beta"],
            TRUE_GJR["nu"],
        ],
        dtype=float,
    )
    return model.fit(disp="off", starting_values=start, show_warning=False)


def _ci_contains(result, name: str, truth: float) -> bool:
    table = result.conf_int()
    if name not in table.index:
        matches = [idx for idx in table.index if name.split("[")[0] in str(idx)]
        name = matches[0]
    if "lower" in table.columns:
        low, high = float(table.loc[name, "lower"]), float(table.loc[name, "upper"])
    else:
        low, high = float(table.iloc[table.index.get_loc(name), 0]), float(
            table.iloc[table.index.get_loc(name), 1]
        )
    return low <= truth <= high


@pytest.mark.slow
def test_t1_1_gjr_true_params_lie_in_the_95_percent_ci() -> None:
    series = load_gjr_fixture()
    assert len(series) == 5000
    fitted = _gjr_estimates(series)
    assert _ci_contains(fitted, "omega", TRUE_GJR["omega"])
    assert _ci_contains(fitted, "alpha[1]", TRUE_GJR["alpha"])
    assert _ci_contains(fitted, "gamma[1]", TRUE_GJR["gamma"])
    assert _ci_contains(fitted, "beta[1]", TRUE_GJR["beta"])


@pytest.mark.slow
def test_t1_1_gjr_coverage_rate_across_100_seeds_is_near_95_percent() -> None:
    """Coverage far below 95% means the reported standard errors are too tight."""
    names = {
        "omega": TRUE_GJR["omega"],
        "alpha[1]": TRUE_GJR["alpha"],
        "gamma[1]": TRUE_GJR["gamma"],
        "beta[1]": TRUE_GJR["beta"],
    }
    hits = {name: 0 for name in names}
    n_seeds = 100
    n_ok = 0
    for seed in range(n_seeds):
        series = simulate_gjr_garch(5000, seed=1000 + seed)
        try:
            fitted = _gjr_estimates(series)
        except Exception:
            continue
        n_ok += 1
        for name, truth in names.items():
            if _ci_contains(fitted, name, truth):
                hits[name] += 1
    assert n_ok >= 90
    for name, count in hits.items():
        rate = count / n_ok
        assert rate >= 0.80, f"{name} coverage {rate:.2f} << 95% (SEs too small?)"


def test_t1_2_egarch_recovers_negative_gamma_sign(tmp_path: Path) -> None:
    """Protects the credit-stress sign convention (Prompt 1.2)."""
    r = load_egarch_fixture()
    model = arch_model(
        r.astype(float),
        mean="Constant",
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        dist="normal",
        rescale=False,
    )
    start = np.array(
        [
            TRUE_EGARCH["mu"],
            TRUE_EGARCH["omega"],
            TRUE_EGARCH["alpha"],
            TRUE_EGARCH["gamma"],
            TRUE_EGARCH["beta"],
        ],
        dtype=float,
    )
    fitted = model.fit(disp="off", starting_values=start, show_warning=False)
    gamma = float(fitted.params["gamma[1]"])
    assert gamma < 0.0
    assert np.sign(gamma) == np.sign(TRUE_EGARCH["gamma"])


def test_t1_2_oas_model_gamma_is_negative_after_sign_inversion(tmp_path: Path) -> None:
    r = load_egarch_fixture()
    log_oas = np.log(5.0) + np.cumsum(-r.to_numpy() / 100.0)
    oas = pd.Series(np.exp(log_oas), index=r.index, name="BAMLH0A0HYM2")
    config = make_model_config(tmp_path, min_observations=80)
    model = OASVolatilityModel(config, series_id="BAMLH0A0HYM2")
    model.fit(oas)
    assert model.report is not None
    assert model.report.gamma is not None
    assert model.report.gamma < 0.0


def test_t1_3_regime_recovers_transition_matrix_and_bounded_detection_lag(
    tmp_path: Path,
) -> None:
    observed, true_stress = load_regime_fixture()
    config = make_regime_config(
        tmp_path,
        input={"min_observations": 80, "rv_window": 5},
        fit={"search_reps": 12, "maxiter": 400, "seed": 21},
    )
    detector = RegimeDetector(config)
    report = detector.fit_observed(observed)
    matrix = report.transition_matrix
    stay_calm = float(matrix.loc["calm", "calm"]) if "calm" in matrix.index else float("nan")
    stay_stress = (
        float(matrix.loc["stress", "stress"]) if "stress" in matrix.index else float("nan")
    )
    assert stay_calm == pytest.approx(TRUE_REGIME["p_stay_calm"], abs=0.12)
    assert stay_stress == pytest.approx(TRUE_REGIME["p_stay_stress"], abs=0.18)

    filtered = detector.filtered_stress
    assert filtered is not None
    declared = (filtered > 0.50).astype(int)
    true = true_stress.reindex(filtered.index).fillna(0).astype(int)
    lags: list[int] = []
    i = 1
    idx = list(filtered.index)
    while i < len(idx):
        if true.iloc[i] == 1 and true.iloc[i - 1] == 0:
            lag = None
            for k in range(0, 25):
                if i + k >= len(idx):
                    break
                if int(declared.iloc[i + k]) == 1:
                    lag = k
                    break
            if lag is not None:
                lags.append(lag)
        i += 1
    assert lags, "filtered probabilities never crossed 0.5 after a true stress entry"
    assert float(np.mean(lags)) <= 12.0
