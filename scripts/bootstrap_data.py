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
    failures: dict[str, str] = {}
    universe = [("fred", sid) for sid in loader.config.fred_series] + [
        ("etf", ticker) for ticker in loader.config.etf_tickers
    ]
    for kind, series_id in universe:
        fetch = loader.fetch_fred if kind == "fred" else loader.fetch_etf
        frame = fetch([series_id], start=start, end=end, force_refresh=args.force_refresh)
        row = loader.coverage_report(frame, requested_start=start)
        try:
            loader.validate_series(frame[series_id], series_id=series_id)
            row["issues"] = ""
        except SeriesValidationError as exc:
            # Keep going: one short series must not hide the state of the rest.
            logging.error("validation failed for %s %s: %s", kind.upper(), series_id, exc)
            failures[series_id] = str(exc)
            row["issues"] = str(exc)
        rows.append(row)
        logging.info(
            "%s %s: dropped %s NaN of %s raw (%.1f%%)",
            kind.upper(),
            series_id,
            int(row["n_nan_dropped"].iloc[0]),
            int(row["n_raw"].iloc[0]),
            float(row["pct_nan_dropped"].iloc[0]),
        )

    report = pd.concat(rows, ignore_index=True)
    out = loader.cache_dir / "coverage_report.csv"
    report.to_csv(out, index=False)
    print(report.to_string(index=False))
    print(f"\nWrote {out}")
    if failures:
        print(f"\n{len(failures)} series failed validation:")
        for series_id, reason in failures.items():
            print(f"  {series_id}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
