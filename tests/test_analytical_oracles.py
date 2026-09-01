"""[T3] Analytical oracles — the suite must detect what it claims to detect."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from data.ebp import build_aggregation_matrix, chow_lin_disaggregate
from helpers import load_gaussian_garch_fixture
from risk.backtests import christoffersen_independence, kupiec_pof
from risk.fhs import FHSEngine
from test_ebp import _synthetic_monthly_and_daily
from test_fhs import make_fhs_config


def _idx(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


@pytest.mark.blocking
@pytest.mark.slow
def test_t3_1_fhs_normal_innovations_match_parametric_var_within_3_mc_se(
    tmp_path: Path,
) -> None:
    r, _true_sigma = load_gaussian_garch_fixture()
    cfg = make_fhs_config(
        tmp_path,
        filter={"min_observations": 1000, "maxiter": 800},
        fhs={"n_simulations": 100_000, "seed": 17, "alphas": [0.99]},
    )
    engine = FHSEngine(cfg).fit(r)
    report = engine.var_es(alpha=0.99, horizon=1)
    z_p = float(scipy_stats.norm.ppf(0.01))
    analytic = -(engine.mu_one_step + engine.sigma_one_step * z_p)
    p = 0.01
    density = float(scipy_stats.norm.pdf(z_p))
    n = int(engine.z.shape[0]) if engine.z is not None else 1600
    se_mc = engine.sigma_one_step * np.sqrt(p * (1.0 - p) / 100_000) / density
    se_sample = engine.sigma_one_step * np.sqrt(p * (1.0 - p) / n) / density
    se = float(np.sqrt(se_mc**2 + se_sample**2))
    assert report.var == pytest.approx(analytic, abs=3.0 * se)


@pytest.mark.blocking
def test_t3_2_uniform_expected_hits_pass_kupiec_and_christoffersen() -> None:
    n, p = 1000, 0.01
    hits = np.zeros(n, dtype=int)
    hits[50:1000:100] = 1
    series = pd.Series(hits, index=_idx(n))
    assert int(hits.sum()) == 10
    assert kupiec_pof(n=n, x=10, p=p).reject is False
    assert christoffersen_independence(series).reject is False


@pytest.mark.blocking
def test_t3_2_clustered_expected_hits_pass_kupiec_reject_christoffersen() -> None:
    n, p = 1000, 0.01
    hits = np.zeros(n, dtype=int)
    hits[200:210] = 1
    series = pd.Series(hits, index=_idx(n))
    kupiec = kupiec_pof(n=n, x=int(hits.sum()), p=p)
    independence = christoffersen_independence(series)
    assert kupiec.reject is False
    assert independence.reject is True


@pytest.mark.blocking
def test_t3_2_double_expected_rate_uniform_rejects_kupiec() -> None:
    n, p = 1000, 0.01
    hits = np.zeros(n, dtype=int)
    hits[25:1000:50] = 1
    assert int(hits.sum()) == 20
    assert kupiec_pof(n=n, x=20, p=p).reject is True


@pytest.mark.blocking
def test_t3_3_chow_lin_monthly_average_matches_within_1e_10() -> None:
    monthly, indicators = _synthetic_monthly_and_daily()
    result = chow_lin_disaggregate(monthly, indicators)
    C = build_aggregation_matrix(result.daily.index, monthly.index)
    np.testing.assert_allclose(C @ result.daily.to_numpy(), monthly.to_numpy(), atol=1e-10)
    np.testing.assert_allclose(
        result.monthly_replicated.to_numpy(), monthly.to_numpy(), atol=1e-10
    )
