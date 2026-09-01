"""Download the credit data universe and print a coverage report.

Requires FRED_API_KEY in the environment. yfinance does not need a key.

Usage
-----
    python scripts/bootstrap_data.py
    python scripts/bootstrap_data.py --force-refresh
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.credit_loader import CreditDataLoader, SeriesValidationError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the full credit/volatility universe and write a coverage report."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "data.yaml",
        help="Path to the pydantic-validated data YAML.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore parquet cache and re-download every series.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    loader = CreditDataLoader.from_yaml(args.config)
    start = loader.config.start_date.isoformat()
    end = (
        loader.config.end_date.isoformat()
        if loader.config.end_date is not None
        else date.today().isoformat()
    )

    rows = []
    for series_id in loader.config.fred_series:
        frame = loader.fetch_fred(
            [series_id], start=start, end=end, force_refresh=args.force_refresh
        )
        try:
            loader.validate_series(frame[series_id], series_id=series_id)
        except SeriesValidationError:
            logging.exception("validation failed for FRED series %s", series_id)
            raise
        rows.append(loader.coverage_report(frame))
        logging.info(
            "FRED %s: dropped %s NaN of %s raw (%.1f%%)",
            series_id,
            int(rows[-1]["n_nan_dropped"].iloc[0]),
            int(rows[-1]["n_raw"].iloc[0]),
            float(rows[-1]["pct_nan_dropped"].iloc[0]),
        )

    for ticker in loader.config.etf_tickers:
        frame = loader.fetch_etf(
            [ticker], start=start, end=end, force_refresh=args.force_refresh
        )
        try:
            loader.validate_series(frame[ticker], series_id=ticker)
        except SeriesValidationError:
            logging.exception("validation failed for ETF %s", ticker)
            raise
        rows.append(loader.coverage_report(frame))

    report = pd.concat(rows, ignore_index=True)
    out = loader.cache_dir / "coverage_report.csv"
    report.to_csv(out, index=False)
    print(report.to_string(index=False))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
