"""Build the historical dislocation-score series and overlay known stress episodes.

Usage
-----
    python scripts/plot_dislocation.py --from-csv path/to/panel.csv
    python scripts/plot_dislocation.py --fit
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

from data.credit_loader import CreditDataLoader  # noqa: E402
from data.ebp import EBPLoader  # noqa: E402
from models.ebp_garch import EBPVolatilityModel  # noqa: E402
from models.oas_egarch import OASVolatilityModel  # noqa: E402
from models.schema import load_model_config  # noqa: E402
from signals.dislocation import (  # noqa: E402
    DislocationSignalEngine,
    SignalInputs,
    build_default_proxy,
)
from signals.schema import load_signal_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the EBP dislocation score.")
    parser.add_argument("--signal-config", type=Path, default=ROOT / "config" / "signal.yaml")
    parser.add_argument("--data-config", type=Path, default=ROOT / "config" / "data.yaml")
    parser.add_argument("--model-config", type=Path, default=ROOT / "config" / "params.yaml")
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--asof", type=str, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _inputs_from_csv(path: Path) -> SignalInputs:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    required = ["sigma_ebp", "sigma_oas", "ebp_level", "oas_level", "default_proxy"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"panel CSV missing columns {missing}")
    return SignalInputs(
        sigma_ebp=frame["sigma_ebp"],
        sigma_oas=frame["sigma_oas"],
        ebp_level=frame["ebp_level"],
        oas_level=frame["oas_level"],
        default_proxy=frame["default_proxy"],
    )


def _inputs_from_fit(args: argparse.Namespace, signal_cfg, asof: str) -> SignalInputs:
    model_cfg = load_model_config(args.model_config)
    if model_cfg.ebp is None:
        raise SystemExit("config/params.yaml is missing the ebp block")
    credit = CreditDataLoader.from_yaml(args.data_config)
    ebp_loader = EBPLoader.from_yaml(args.data_config)
    ebp_frame = ebp_loader.fetch(force_refresh=args.force_refresh)
    monthly = ebp_loader.available_asof(asof)
    start = credit.config.start_date.isoformat()
    spec = signal_cfg.default_proxy
    hy_id = credit.config.ebp.hy_oas_series_id if credit.config.ebp else "BAMLH0A0HYM2"
    ids = [hy_id, spec.ccc_series_id, spec.bbb_series_id]
    oas = credit.fetch_fred(ids, start=start, end=asof, force_refresh=args.force_refresh)
    oas_model = OASVolatilityModel(model_cfg, series_id=hy_id).fit(oas[hy_id].dropna())
    ebp_model = EBPVolatilityModel(model_cfg, series_id="EBP", layer="primary_monthly").fit(
        monthly["ebp"]
    )
    if spec.option == "A":
        proxy = build_default_proxy("A", gz_spread=monthly["gz_spread"], ebp=monthly["ebp"])
    elif spec.option == "C":
        proxy = build_default_proxy(
            "C", ccc_oas=oas[spec.ccc_series_id], bbb_oas=oas[spec.bbb_series_id]
        )
    else:
        raise SystemExit("option B requires --from-csv with a pre-built default_proxy")
    n_vol = len(oas_model.result.conditional_volatility)
    return SignalInputs(
        sigma_ebp=ebp_model.conditional_volatility(),
        sigma_oas=pd.Series(
            oas_model.result.conditional_volatility,
            index=oas_model.stress.r.index[-n_vol:],
            name="sigma_oas",
        ),
        ebp_level=ebp_frame.ebp,
        oas_level=oas[hy_id],
        default_proxy=proxy,
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    signal_cfg = load_signal_config(args.signal_config)
    asof = args.asof or date.today().isoformat()
    if args.from_csv is not None:
        inputs = _inputs_from_csv(args.from_csv)
    elif args.fit:
        inputs = _inputs_from_fit(args, signal_cfg, asof)
    else:
        raise SystemExit("pass --from-csv PANEL.csv or --fit")
    engine = DislocationSignalEngine(inputs, signal_cfg)
    dest = engine.plot_history()
    hist = engine.history()
    print(hist[["score", "stress", "fundamental", "active", "weight"]].tail().to_string())
    print(f"\nWrote {dest}")
    print(f"Wrote {signal_cfg.output.score_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
