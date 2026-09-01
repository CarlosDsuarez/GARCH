"""VaR/ES backtest battery: Kupiec, Christoffersen, DQ, Acerbi–Székely, Basel.

Synthetic schema: ISO dates (YYYY-MM-DD), invented simple returns and positive
VaR/ES levels. No live P&L or regulatory filings are stored here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from risk.backtests import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    basel_traffic_light,
    christoffersen_cc,
    christoffersen_independence,
    dynamic_quantile_test,
    hit_series,
    kupiec_pof,
    run_full_backtest_suite,
)
from risk.schema import load_fhs_config


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_RISK = REPO_ROOT / "config" / "risk.yaml"


def _idx(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def test_production_backtest_knobs_match_the_spec() -> None:
    config = load_fhs_config(PRODUCTION_RISK)
    assert config.backtest is not None
    assert config.backtest.rolling_window == 250
    assert config.backtest.dq_lags == 4
    assert config.backtest.acerbi_simulations >= 10_000
    assert config.backtest.seed is not None
    assert config.backtest.significance == pytest.approx(0.05)


def test_hits_flag_returns_worse_than_positive_var() -> None:
    idx = _idx(4)
    r = pd.Series([0.01, -0.02, -0.05, -0.01], index=idx)
    var = pd.Series([0.03, 0.03, 0.03, 0.03], index=idx)
    hits = hit_series(r, var)
    assert list(hits.to_numpy()) == [0, 0, 1, 0]


def test_kupiec_is_zero_when_hit_rate_equals_p() -> None:
    n, x, p = 100, 5, 0.05
    report = kupiec_pof(n=n, x=x, p=p)
    assert report.statistic == pytest.approx(0.0, abs=1e-10)
    assert report.pvalue == pytest.approx(1.0, abs=1e-6)
    assert report.reject is False
    assert report.df == 1


def test_kupiec_matches_closed_form_on_a_known_sample() -> None:
    n, x, p = 250, 10, 0.01
    expected = -2.0 * (
        (n - x) * np.log(1.0 - p)
        + x * np.log(p)
        - (n - x) * np.log(1.0 - x / n)
        - x * np.log(x / n)
    )
    report = kupiec_pof(n=n, x=x, p=p)
    assert report.statistic == pytest.approx(expected, rel=1e-10)
    assert report.pvalue == pytest.approx(float(scipy_stats.chi2.sf(expected, 1)), rel=1e-10)
    assert report.reject is True
    assert expected > 3.841


@pytest.mark.blocking
def test_clustered_hits_pass_kupiec_and_fail_christoffersen() -> None:
    """The failure mode that justifies the suite: right count, clustered week."""
    n, p = 1000, 0.01
    hits = np.zeros(n, dtype=int)
    hits[200:210] = 1
    series = pd.Series(hits, index=_idx(n))
    kupiec = kupiec_pof(n=n, x=int(hits.sum()), p=p)
    independence = christoffersen_independence(series)
    cc = christoffersen_cc(series, p=p)
    assert kupiec.reject is False
    assert kupiec.statistic == pytest.approx(0.0, abs=1e-10)
    assert independence.n_11 == 9
    assert independence.n_01 == 1
    assert independence.reject is True
    assert independence.statistic > 3.841
    assert cc.reject is True
    assert cc.statistic == pytest.approx(kupiec.statistic + independence.statistic, rel=1e-10)
    assert cc.df == 2


@pytest.mark.blocking
def test_spaced_hits_do_not_reject_independence() -> None:
    n, p = 1000, 0.01
    hits = np.zeros(n, dtype=int)
    hits[50:1000:100] = 1
    series = pd.Series(hits, index=_idx(n))
    assert int(hits.sum()) == 10
    assert kupiec_pof(n=n, x=10, p=p).reject is False
    assert christoffersen_independence(series).reject is False


def test_basel_traffic_light_zones() -> None:
    assert basel_traffic_light(exceptions=0).zone == "green"
    assert basel_traffic_light(exceptions=4).zone == "green"
    assert basel_traffic_light(exceptions=5).zone == "yellow"
    assert basel_traffic_light(exceptions=9).zone == "yellow"
    assert basel_traffic_light(exceptions=10).zone == "red"
    assert basel_traffic_light(exceptions=10).window == 250


def test_dynamic_quantile_rejects_when_hits_track_var_level() -> None:
    idx = _idx(400)
    var = pd.Series(np.linspace(0.01, 0.06, 400), index=idx)
    hits = (var > var.quantile(0.90)).astype(int)
    report = dynamic_quantile_test(hits, var, p=0.01, lags=4)
    assert report.df == 6
    assert report.lags == 4
    assert report.reject is True


def test_acerbi_z1_is_zero_when_tail_losses_match_es() -> None:
    idx = _idx(80)
    es = pd.Series(0.04, index=idx)
    r = pd.Series(0.01, index=idx)
    hits = pd.Series(0, index=idx, dtype=int)
    hits.iloc[10:15] = 1
    r.loc[hits == 1] = -0.04
    report = acerbi_szekely_z1(r, es, hits, n_simulations=200, seed=7)
    assert report.statistic == pytest.approx(0.0, abs=1e-12)
    assert report.n_simulations == 200
    assert report.seed == 7


def test_acerbi_z1_is_negative_when_tail_is_worse_than_es() -> None:
    idx = _idx(80)
    es = pd.Series(0.04, index=idx)
    r = pd.Series(0.01, index=idx)
    hits = pd.Series(0, index=idx, dtype=int)
    hits.iloc[10:18] = 1
    r.loc[hits == 1] = -0.12
    report = acerbi_szekely_z1(r, es, hits, n_simulations=400, seed=7)
    assert report.statistic < 0.0
    assert report.reject is True


def test_acerbi_z2_uses_np_scaling() -> None:
    idx = _idx(100)
    p = 0.05
    es = pd.Series(0.02, index=idx)
    r = pd.Series(0.00, index=idx)
    hits = pd.Series(0, index=idx, dtype=int)
    hits.iloc[:5] = 1
    r.iloc[:5] = -0.02
    report = acerbi_szekely_z2(r, es, hits, p=p, n_simulations=200, seed=3)
    assert report.statistic == pytest.approx(0.0, abs=1e-12)


def test_suite_returns_structured_verdict_for_three_models(tmp_path: Path) -> None:
    n = 300
    idx = _idx(n)
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.0002, 0.01, size=n), index=idx)
    var_fhs = pd.Series(0.023, index=idx)
    es_fhs = pd.Series(0.030, index=idx)
    var_hs = pd.Series(0.020, index=idx)
    es_hs = pd.Series(0.026, index=idx)
    var_n = pd.Series(0.018, index=idx)
    es_n = pd.Series(0.022, index=idx)
    suite = run_full_backtest_suite(
        r,
        var_fhs,
        es_fhs,
        alpha=0.99,
        models={
            "FHS-GJR-GARCH": (var_fhs, es_fhs),
            "histórico 250d": (var_hs, es_hs),
            "normal": (var_n, es_n),
        },
        plot_directory=tmp_path,
        seed=7,
        acerbi_simulations=150,
    )
    assert {"FHS-GJR-GARCH", "histórico 250d", "normal"} <= set(suite.models)
    assert {"full", "rolling_250"} <= set(suite.windows)
    row = suite.report("FHS-GJR-GARCH", "full")
    assert row.kupiec.pvalue is not None
    assert row.independence.pvalue is not None
    assert row.conditional_coverage.pvalue is not None
    assert row.dq.pvalue is not None
    assert row.z1.pvalue is not None
    assert row.z2.pvalue is not None
    assert row.basel.zone in {"green", "yellow", "red"}
    assert row.verdict in {"pass", "scrutiny", "reject"}
    assert suite.comparison is not None
    assert {"model", "window", "hits", "kupiec_p", "christoffersen_p", "dq_p"} <= set(
        suite.comparison.columns
    )
    assert suite.plot_path is not None
    assert suite.plot_path.exists()
    assert isinstance(suite.verdict, str)
