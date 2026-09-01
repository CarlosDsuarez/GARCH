"""Pre- and post-estimation diagnostic battery ([C7] quality gate).

Synthetic series use ISO dates and invented returns. No market data is stored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from diagnostics.econometric import (
    DiagnosticGateError,
    DiagnosticSuite,
    FittedGarchSnapshot,
    diagnostic_gate,
    joint_stationarity_verdict,
    render_markdown,
    require_post_estimation,
)
from diagnostics.schema import DiagnosticConfig, load_diagnostics_config


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = REPO_ROOT / "config" / "diagnostics.yaml"


def _diag_payload(tmp_path: Path, **overrides) -> dict:
    payload: dict = {
        "significance": 0.05,
        "adf": {"autolag": "AIC", "regression": "c"},
        "kpss": {"regression": "c", "nlags": "auto"},
        "arch_lm_lags": [5, 10],
        "ljung_box_lags": [10, 20],
        "rho1_warn": 0.15,
        "igarch_persistence": 0.999,
        "half_life_mass": 0.5,
        "optimizer_restarts": 5,
        "llf_atol": 0.01,
        "icss_critical": 1.358,
        "blocking_pre": ["A1"],
        "blocking_post": ["B1", "B2", "B3", "B5", "B6"],
        "plot": {
            "output_directory": str(tmp_path / "diag"),
            "qq_filename": "qq_standardized_residuals.png",
            "dpi": 120,
            "figsize_width": 7.0,
            "figsize_height": 5.0,
        },
    }
    payload.update(overrides)
    return payload


def _suite(tmp_path: Path, **overrides) -> DiagnosticSuite:
    return DiagnosticSuite(DiagnosticConfig.model_validate(_diag_payload(tmp_path, **overrides)))


def _series(values, *, start: str = "2018-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx, name="r")


def _white_noise(n: int = 400, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _series(rng.standard_normal(n))


def _random_walk(n: int = 500, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _series(np.cumsum(rng.standard_normal(n)))


def _garch_returns(n: int = 800, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.05, 0.15, 0.80
    eps = np.empty(n)
    sigma2 = omega / (1.0 - alpha - beta)
    for t in range(n):
        z = rng.standard_normal()
        eps[t] = np.sqrt(sigma2) * z
        sigma2 = omega + alpha * eps[t] ** 2 + beta * sigma2
    return _series(eps)


def _ar1(n: int = 400, phi: float = 0.65, seed: int = 19) -> pd.Series:
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = rng.standard_normal()
    e = rng.standard_normal(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return _series(y)


def _clean_snapshot(
    z: pd.Series,
    *,
    family: str = "GJR",
    dist: str = "normal",
    alpha: float = 0.05,
    beta: float = 0.90,
    gamma: float = 0.08,
    nu: float | None = None,
    converged: bool = True,
    restart_loglikelihoods: tuple[float, ...] | None = (10.0, 10.001, 9.999, 10.0, 10.002),
    **kwargs,
) -> FittedGarchSnapshot:
    return FittedGarchSnapshot(
        z=z,
        eps=kwargs.pop("eps", z),
        family=family,  # type: ignore[arg-type]
        dist=dist,  # type: ignore[arg-type]
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        nu=nu,
        converged=converged,
        loglikelihood=10.0,
        restart_loglikelihoods=restart_loglikelihoods,
        **kwargs,
    )


def test_production_diagnostics_yaml_loads() -> None:
    cfg = load_diagnostics_config(PRODUCTION)
    assert cfg.adf.autolag == "AIC"
    assert cfg.arch_lm_lags == [5, 10]
    assert cfg.ljung_box_lags == [10, 20]
    assert cfg.significance == pytest.approx(0.05)
    assert cfg.blocking_pre == ["A1"]
    assert cfg.blocking_post == ["B1", "B2", "B3", "B5", "B6"]


def test_diagnostics_config_forbids_unknown_keys(tmp_path: Path) -> None:
    payload = _diag_payload(tmp_path, undocumented_knob=1)
    with pytest.raises(ValidationError):
        DiagnosticConfig.model_validate(payload)


def test_joint_stationarity_verdict_covers_four_cells() -> None:
    assert joint_stationarity_verdict(0.01, 0.20, 0.05) == "PASS"
    assert joint_stationarity_verdict(0.20, 0.01, 0.05) == "FAIL"
    assert joint_stationarity_verdict(0.01, 0.01, 0.05) == "WARN"
    assert joint_stationarity_verdict(0.20, 0.20, 0.05) == "WARN"


def test_a1_white_noise_is_stationary(tmp_path: Path) -> None:
    report = _suite(tmp_path).run_pre_estimation(_white_noise())
    joint = report.by_name("A1.joint")
    assert joint.verdict == "PASS"
    assert "AIC" in report.by_name("A1.ADF").criterion
    assert "constant" in report.by_name("A1.ADF").criterion.lower()
    assert report.by_name("A1.KPSS").pvalue is not None


def test_a1_random_walk_fails_stationarity(tmp_path: Path) -> None:
    report = _suite(tmp_path).run_pre_estimation(_random_walk())
    assert report.by_name("A1.joint").verdict == "FAIL"
    assert report.has_blocking_fail is True


def test_a2_white_noise_warns_garch_unnecessary(tmp_path: Path) -> None:
    report = _suite(tmp_path).run_pre_estimation(_white_noise(n=500, seed=2))
    lm5 = report.by_name("A2.ARCH_LM_5")
    lm10 = report.by_name("A2.ARCH_LM_10")
    assert lm5.verdict == "WARN"
    assert lm10.verdict == "WARN"
    assert "unnecessary" in lm5.message.lower() or "no conditional heteroskedasticity" in lm5.message.lower()


def test_a2_garch_dgp_detects_arch_effects(tmp_path: Path) -> None:
    report = _suite(tmp_path).run_pre_estimation(_garch_returns())
    assert report.by_name("A2.ARCH_LM_5").verdict == "PASS"
    assert report.by_name("A2.ARCH_LM_5").pvalue is not None
    assert report.by_name("A2.ARCH_LM_5").pvalue < 0.05


def test_a3_reports_rho1_and_ljung_box_10_and_20(tmp_path: Path) -> None:
    report = _suite(tmp_path).run_pre_estimation(_ar1())
    rho = report.by_name("A3.rho1")
    assert rho.statistic is not None
    assert rho.statistic > 0.4
    assert report.by_name("A3.LjungBox_10").pvalue is not None
    assert report.by_name("A3.LjungBox_20").pvalue is not None
    assert "rho_1" in rho.message or "ρ" in rho.message


def test_a4_quantifies_non_normality_without_failing(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    fat = _series(rng.standard_t(4, size=400))
    result = _suite(tmp_path).run_pre_estimation(fat).by_name("A4.JarqueBera")
    assert result.verdict != "FAIL"
    assert "skew" in result.extras
    assert "kurtosis" in result.extras
    assert result.extras["kurtosis"] > 3.0


def test_a5_variance_break_reports_dates(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    first = rng.standard_normal(220)
    second = 5.0 * rng.standard_normal(220)
    report = _suite(tmp_path).run_pre_estimation(_series(np.concatenate([first, second])))
    icss = report.by_name("A5.ICSS")
    assert icss.verdict == "WARN"
    dates = icss.extras.get("break_dates")
    assert dates, "A5 must report detected break dates"
    ts = pd.Timestamp(dates[0])
    expected_mid = pd.bdate_range("2018-01-02", periods=440)[220]
    assert abs((ts - expected_mid).days) < 80


def test_pre_report_fields_are_structured(tmp_path: Path) -> None:
    report = _suite(tmp_path).run_pre_estimation(_white_noise())
    assert report.stage == "pre"
    row = report.results[0]
    assert row.name
    assert row.criterion
    assert row.verdict in {"PASS", "WARN", "FAIL"}
    assert row.message
    assert report.n_obs == 400


def test_b1_leftover_mean_autocorrelation_fails(tmp_path: Path) -> None:
    z = _ar1(phi=0.7)
    report = _suite(tmp_path).run_post_estimation(_clean_snapshot(z))
    b1 = report.by_name("B1.LjungBox_z")
    assert b1.verdict == "FAIL"
    assert b1.pvalue is not None and b1.pvalue <= 0.05
    assert "mean" in b1.message.lower()
    assert report.has_blocking_fail is True


def test_b1_iid_standardized_residuals_pass(tmp_path: Path) -> None:
    report = _suite(tmp_path).run_post_estimation(_clean_snapshot(_white_noise()))
    assert report.by_name("B1.LjungBox_z").verdict == "PASS"


def test_b2_leftover_arch_is_the_blocking_failure(tmp_path: Path) -> None:
    z = _garch_returns()
    report = _suite(tmp_path).run_post_estimation(_clean_snapshot(z))
    lb = report.by_name("B2.LjungBox_z2")
    lm = report.by_name("B2.ARCH_LM")
    assert lb.verdict == "FAIL" or lm.verdict == "FAIL"
    assert "heteroskedasticity" in (lb.message + lm.message).lower() or "ARCH" in (lb.message + lm.message)
    assert report.has_blocking_fail is True


def test_b2_iid_squares_pass(tmp_path: Path) -> None:
    report = _suite(tmp_path).run_post_estimation(_clean_snapshot(_white_noise(seed=21)))
    assert report.by_name("B2.LjungBox_z2").verdict == "PASS"
    assert report.by_name("B2.ARCH_LM").verdict == "PASS"


def test_b3_sign_bias_fails_after_gjr(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    n = 500
    eps = rng.standard_normal(n)
    z2 = np.ones(n)
    z2[1:] = 1.0 + 5.0 * (eps[:-1] < 0)
    z = np.sqrt(z2) * np.sign(rng.standard_normal(n))
    snap = _clean_snapshot(_series(z), family="GJR", eps=_series(eps))
    report = _suite(tmp_path).run_post_estimation(snap)
    result = report.by_name("B3.EngleNg")
    assert result.verdict == "FAIL"
    assert result.pvalue is not None and result.pvalue < 0.05
    assert report.has_blocking_fail is True


def test_b3_reject_on_symmetric_garch_is_warn(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    n = 500
    eps = rng.standard_normal(n)
    z2 = np.ones(n)
    z2[1:] = 1.0 + 5.0 * (eps[:-1] < 0)
    z = np.sqrt(z2) * np.sign(rng.standard_normal(n))
    snap = _clean_snapshot(_series(z), family="GARCH", gamma=0.0, eps=_series(eps))
    result = _suite(tmp_path).run_post_estimation(snap).by_name("B3.EngleNg")
    assert result.verdict == "WARN"


def test_b3_no_sign_bias_passes(tmp_path: Path) -> None:
    z = _white_noise(seed=23)
    result = _suite(tmp_path).run_post_estimation(_clean_snapshot(z, family="EGARCH")).by_name("B3.EngleNg")
    assert result.verdict == "PASS"


def test_b4_writes_qq_plot_and_ks(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    report = suite.run_post_estimation(_clean_snapshot(_white_noise(seed=29), dist="normal"))
    ks = report.by_name("B4.KS")
    assert ks.pvalue is not None
    qq = Path(tmp_path / "diag" / "qq_standardized_residuals.png")
    assert qq.is_file()
    assert qq.stat().st_size > 0


def test_b5_egarch_nonstationary_beta_fails(tmp_path: Path) -> None:
    snap = _clean_snapshot(_white_noise(), family="EGARCH", beta=1.01, alpha=0.1, gamma=-0.05)
    result = _suite(tmp_path).run_post_estimation(snap).by_name("B5.stationarity")
    assert result.verdict == "FAIL"
    assert _suite(tmp_path).run_post_estimation(snap).has_blocking_fail is True


def test_b5_egarch_uses_abs_beta(tmp_path: Path) -> None:
    fail = _clean_snapshot(_white_noise(), family="EGARCH", beta=-1.01, alpha=0.1, gamma=-0.05)
    ok = _clean_snapshot(_white_noise(), family="EGARCH", beta=-0.95, alpha=0.1, gamma=-0.05)
    assert _suite(tmp_path).run_post_estimation(fail).by_name("B5.stationarity").verdict == "FAIL"
    assert _suite(tmp_path).run_post_estimation(ok).by_name("B5.stationarity").verdict == "PASS"


def test_b5_igarch_frontier_warns_and_reports_half_life(tmp_path: Path) -> None:
    snap = _clean_snapshot(_white_noise(), family="EGARCH", beta=0.9995, alpha=0.1, gamma=-0.05)
    result = _suite(tmp_path).run_post_estimation(snap).by_name("B5.stationarity")
    assert result.verdict == "WARN"
    assert "half" in result.message.lower() or "IGARCH" in result.message
    expected = float(np.log(0.5) / np.log(0.9995))
    assert result.extras["half_life"] == pytest.approx(expected)


def test_b5_gjr_persistence_formula(tmp_path: Path) -> None:
    snap = _clean_snapshot(
        _white_noise(),
        family="GJR",
        alpha=0.10,
        gamma=0.20,
        beta=0.85,
    )
    result = _suite(tmp_path).run_post_estimation(snap).by_name("B5.stationarity")
    assert result.verdict == "FAIL"
    assert result.statistic == pytest.approx(0.10 + 0.20 / 2.0 + 0.85)
    assert _suite(tmp_path).run_post_estimation(snap).has_blocking_fail is True


def test_b5_gjr_negative_alpha_fails(tmp_path: Path) -> None:
    snap = _clean_snapshot(_white_noise(), family="GJR", alpha=-0.20, gamma=0.10, beta=0.80)
    result = _suite(tmp_path).run_post_estimation(snap).by_name("B5.stationarity")
    assert result.verdict == "FAIL"


def test_b6_not_converged_fails(tmp_path: Path) -> None:
    snap = _clean_snapshot(_white_noise(), converged=False)
    report = _suite(tmp_path).run_post_estimation(snap)
    result = report.by_name("B6.convergence")
    assert result.verdict == "FAIL"
    assert report.has_blocking_fail is True


def test_b6_restart_disagreement_fails(tmp_path: Path) -> None:
    snap = _clean_snapshot(
        _white_noise(),
        restart_loglikelihoods=(10.0, 5.0, 10.0, 10.0, 10.0),
    )
    report = _suite(tmp_path).run_post_estimation(snap)
    result = report.by_name("B6.restarts")
    assert result.verdict == "FAIL"
    assert "local" in result.message.lower() or "optima" in result.message.lower()
    assert report.has_blocking_fail is True


def test_b6_nan_restarts_fail(tmp_path: Path) -> None:
    snap = _clean_snapshot(
        _white_noise(),
        restart_loglikelihoods=(10.0, float("nan"), float("nan"), float("nan"), float("nan")),
    )
    result = _suite(tmp_path).run_post_estimation(snap).by_name("B6.restarts")
    assert result.verdict == "FAIL"


def test_b6_missing_restarts_fail(tmp_path: Path) -> None:
    snap = _clean_snapshot(_white_noise(), restart_loglikelihoods=None)
    report = _suite(tmp_path).run_post_estimation(snap)
    assert report.by_name("B6.restarts").verdict == "FAIL"
    assert report.has_blocking_fail is True


def test_b6_restart_agreement_passes(tmp_path: Path) -> None:
    snap = _clean_snapshot(_white_noise(), restart_loglikelihoods=(3.14, 3.141, 3.139, 3.14, 3.142))
    result = _suite(tmp_path).run_post_estimation(snap).by_name("B6.restarts")
    assert result.verdict == "PASS"


def test_decorator_raises_on_blocking_fail(tmp_path: Path) -> None:
    suite = _suite(tmp_path)

    @require_post_estimation(suite)
    def fit_misspecified() -> FittedGarchSnapshot:
        return _clean_snapshot(_ar1(phi=0.7))

    with pytest.raises(DiagnosticGateError) as exc:
        fit_misspecified()
    assert "B1" in str(exc.value)


def test_decorator_returns_fit_when_clean(tmp_path: Path) -> None:
    suite = _suite(tmp_path)

    @require_post_estimation(suite)
    def fit_ok() -> FittedGarchSnapshot:
        return _clean_snapshot(_white_noise(seed=31))

    out = fit_ok()
    assert isinstance(out, FittedGarchSnapshot)
    assert suite.last_post_report is not None
    assert suite.last_post_report.has_blocking_fail is False


def test_context_manager_raises_on_pre_estimation_fail(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    with pytest.raises(DiagnosticGateError) as exc:
        with diagnostic_gate(_random_walk(), suite):
            raise AssertionError("must not enter the fit body when A1 FAILs")
    assert "A1" in str(exc.value)


def test_b3_missing_eps_fails_instead_of_substituting_z(tmp_path: Path) -> None:
    z = _white_noise()
    snap = FittedGarchSnapshot(
        z=z,
        family="GJR",
        dist="normal",
        alpha=0.05,
        beta=0.90,
        gamma=0.08,
        eps=None,
        converged=True,
        restart_loglikelihoods=(10.0,) * 5,
    )
    result = _suite(tmp_path).run_post_estimation(snap).by_name("B3.EngleNg")
    assert result.verdict == "FAIL"
    assert "eps" in result.message.lower()


def test_b3_size_bias_rejects_only_when_eps_is_used(tmp_path: Path) -> None:
    rng = np.random.default_rng(41)
    n = 600
    eps = rng.standard_normal(n)
    z2 = np.ones(n)
    z2[1:] = 1.0 + 6.0 * np.clip(-eps[:-1], 0.0, None)
    signs = np.sign(rng.standard_normal(n))
    signs[signs == 0] = 1.0
    z = np.sqrt(z2) * signs
    snap = _clean_snapshot(_series(z), family="GJR", eps=_series(eps))
    result = _suite(tmp_path).run_post_estimation(snap).by_name("B3.EngleNg")
    assert result.verdict == "FAIL"


def test_b4_ks_passes_for_unit_variance_student_t(tmp_path: Path) -> None:
    nu = 8.0
    rng = np.random.default_rng(0)
    raw = rng.standard_t(nu, size=1200)
    z = raw / np.sqrt(nu / (nu - 2.0))
    snap = _clean_snapshot(_series(z), dist="t", nu=nu)
    ks = _suite(tmp_path).run_post_estimation(snap).by_name("B4.KS")
    assert ks.verdict == "PASS"


def test_context_manager_raises_on_post_estimation_arch_fail(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    series = _white_noise(seed=4)
    with pytest.raises(DiagnosticGateError) as exc:
        with diagnostic_gate(series, suite) as bag:
            bag["fitted"] = _clean_snapshot(_garch_returns())
    assert "B2" in str(exc.value)


def test_markdown_renderer_includes_verdicts_and_messages(tmp_path: Path) -> None:
    report = _suite(tmp_path).run_pre_estimation(_white_noise())
    md = render_markdown(report)
    assert report.to_markdown() == md
    assert "| Test |" in md or "A1.ADF" in md
    assert "PASS" in md
    assert "ADF" in md
    assert "AIC" in md
