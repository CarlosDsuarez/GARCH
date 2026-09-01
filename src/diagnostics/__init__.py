"""Econometric quality gate run before and after every GARCH fit ([C7])."""

from diagnostics.econometric import (
    DiagnosticGateError,
    DiagnosticReport,
    DiagnosticResult,
    DiagnosticSuite,
    FittedGarchSnapshot,
    diagnostic_gate,
    joint_stationarity_verdict,
    render_markdown,
    require_post_estimation,
)
from diagnostics.schema import DiagnosticConfig, load_diagnostics_config

__all__ = [
    "DiagnosticConfig",
    "DiagnosticGateError",
    "DiagnosticReport",
    "DiagnosticResult",
    "DiagnosticSuite",
    "FittedGarchSnapshot",
    "diagnostic_gate",
    "joint_stationarity_verdict",
    "load_diagnostics_config",
    "render_markdown",
    "require_post_estimation",
]
