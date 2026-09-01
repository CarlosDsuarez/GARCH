"""Credit-market data ingestion package."""

from data.credit_loader import (
    CreditDataLoader,
    SeriesValidationError,
    detect_spurious_coupon_drops,
    load_data_config,
)
from data.ebp import EBPLoader, chow_lin_disaggregate, publication_date
from data.schema import DataConfig

__all__ = [
    "CreditDataLoader",
    "DataConfig",
    "EBPLoader",
    "SeriesValidationError",
    "chow_lin_disaggregate",
    "detect_spurious_coupon_drops",
    "load_data_config",
    "publication_date",
]
