"""Credit volatility models."""

from models.ebp_garch import (
    EBPVolatilityModel,
    build_ebp_stress_return,
    write_comparative_report,
)
from models.oas_egarch import (
    OASVolatilityModel,
    build_credit_stress_return,
    fit_oas_universe,
    load_model_config,
)
from models.regime import (
    RegimeDetector,
    load_regime_config,
)
from models.schema import RegimeConfig

__all__ = [
    "EBPVolatilityModel",
    "OASVolatilityModel",
    "RegimeConfig",
    "RegimeDetector",
    "build_credit_stress_return",
    "build_ebp_stress_return",
    "fit_oas_universe",
    "load_model_config",
    "load_regime_config",
    "write_comparative_report",
]
