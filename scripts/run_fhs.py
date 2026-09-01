"""Fit FHS on a credit total-return series and write the comparison report.

Usage
-----
    python scripts/run_fhs.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.credit_loader import CreditDataLoader  # noqa: E402
from risk.fhs import FHSEngine  # noqa: E402
from risk.schema import load_fhs_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filtered Historical Simulation VaR / ES.")
    parser.add_argument("--risk-config", type=Path, default=ROOT / "config" / "risk.yaml")
    parser.add_argument("--data-config", type=Path, default=ROOT / "config" / "data.yaml")
    parser.add_argument("--ticker", type=str, default="HYG")
    parser.add_argument("--asof", type=str, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = load_fhs_config(args.risk_config)
    credit = CreditDataLoader.from_yaml(args.data_config)
    asof = args.asof or date.today().isoformat()
    start = credit.config.start_date.isoformat()
    prices = credit.fetch_etf(
        [args.ticker], start=start, end=asof, force_refresh=args.force_refresh
    )
    returns = prices[args.ticker].astype("float64").pct_change().rename(args.ticker)
    engine = FHSEngine(cfg).fit(returns)
    both = engine.var_es(horizon=1)
    paths = engine.write_report()
    for alpha, report in both.items():
        print(f"alpha={alpha:.3f}  VaR={report.var:.6f}  ES={report.es:.6f}")
    print(f"Wrote {paths.markdown}")
    print(f"Wrote {paths.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
