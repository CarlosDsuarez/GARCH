"""Pure helpers and CLI contracts in ``scripts/`` (no network, no data files)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_garch_named_adds_plain_aliases_for_indexed_arch_names() -> None:
    module = _load("run_signal_backtest")
    params = pd.Series(
        {"Const": 0.1, "omega": 0.01, "alpha[1]": 0.1, "gamma[1]": -0.05, "beta[1]": 0.9, "nu": 6.0}
    )
    named = module._garch_named(params)
    assert named["alpha"] == pytest.approx(0.1)
    assert named["gamma"] == pytest.approx(-0.05)
    assert named["beta"] == pytest.approx(0.9)
    assert named["beta[1]"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "name",
    ["estimate_oas_panel", "estimate_ebp", "run_fhs", "run_signal_backtest", "plot_dislocation"],
)
def test_every_data_script_accepts_asof(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load(name)
    monkeypatch.setattr(sys, "argv", [name, "--asof", "2026-10-01"])
    assert module.parse_args().asof == "2026-10-01"


def test_vix_only_layer_is_not_truncated_by_hy_oas_history() -> None:
    module = _load("estimate_ebp")
    idx = pd.bdate_range("2000-01-03", periods=600)
    vix = pd.Series(20.0, index=idx)
    hy = pd.Series(4.0, index=idx).where(idx >= idx[400])  # HY OAS starts late
    t10y2y = pd.Series(1.0, index=idx)
    full, vix_only = module._indicator_layers(vix, hy, t10y2y)
    assert full.index.min() == idx[400]
    assert vix_only.index.min() == idx[0]
    assert list(vix_only.columns) == ["ln_vix"]
