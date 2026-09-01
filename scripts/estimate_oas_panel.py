"""Fit EGARCH on the five ICE BofA OAS series and write a comparative table.

Usage
-----
    python scripts/estimate_oas_panel.py
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
from models.oas_egarch import fit_oas_universe, load_model_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate the OAS EGARCH panel.")
    parser.add_argument("--data-config", type=Path, default=ROOT / "config" / "data.yaml")
    parser.add_argument("--model-config", type=Path, default=ROOT / "config" / "params.yaml")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    model_cfg = load_model_config(args.model_config)
    loader = CreditDataLoader.from_yaml(args.data_config)
    start = loader.config.start_date.isoformat()
    end = (
        loader.config.end_date.isoformat()
        if loader.config.end_date is not None
        else date.today().isoformat()
    )
    series_ids = list(model_cfg.oas_universe)
    frame = loader.fetch_fred(
        series_ids, start=start, end=end, force_refresh=args.force_refresh
    )
    series_map = {sid: frame[sid].dropna() for sid in series_ids}
    table = fit_oas_universe(series_map, model_cfg)
    dest = Path(model_cfg.output.comparative_table)
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
