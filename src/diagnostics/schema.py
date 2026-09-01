"""Pydantic schema for config/diagnostics.yaml — [C7] GARCH quality-gate knobs ([C2])."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AdfConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    autolag: Literal["AIC", "BIC", "t-stat"]
    regression: Literal["c", "ct", "ctt", "n"]


class KpssConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regression: Literal["c", "ct"]
    nlags: Literal["auto"] | int


class DiagnosticPlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_directory: str
    qq_filename: str
    dpi: int = Field(gt=0)
    figsize_width: float = Field(gt=0)
    figsize_height: float = Field(gt=0)


class DiagnosticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    significance: float = Field(gt=0, lt=1)
    adf: AdfConfig
    kpss: KpssConfig
    arch_lm_lags: list[int] = Field(min_length=1)
    ljung_box_lags: list[int] = Field(min_length=1)
    rho1_warn: float = Field(gt=0, lt=1)
    igarch_persistence: float = Field(gt=0, lt=1)
    half_life_mass: float = Field(gt=0, lt=1)
    optimizer_restarts: int = Field(ge=2)
    llf_atol: float = Field(gt=0)
    icss_critical: float = Field(gt=0)
    blocking_pre: list[str] = Field(min_length=1)
    blocking_post: list[str] = Field(min_length=1)
    plot: DiagnosticPlotConfig


def load_diagnostics_config(path: str | Path) -> DiagnosticConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return DiagnosticConfig.model_validate(payload)
