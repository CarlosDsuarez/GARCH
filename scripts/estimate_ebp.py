"""Fit GJR-GARCH on official monthly EBP and write the three-layer report.

Usage
-----
    python scripts/estimate_ebp.py
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.credit_loader import CreditDataLoader  # noqa: E402
from data.ebp import EBPLoader, build_indicator_matrix  # noqa: E402
from models.ebp_garch import (  # noqa: E402
    EBPVolatilityModel,
    write_comparative_report,
)
from models.schema import load_model_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate EBP GJR-GARCH layers.")
    parser.add_argument("--data-config", type=Path, default=ROOT / "config" / "data.yaml")
    parser.add_argument("--model-config", type=Path, default=ROOT / "config" / "params.yaml")
    parser.add_argument("--asof", type=str, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _annualize(vol: pd.Series, periods_per_year: float) -> pd.Series:
    return vol * math.sqrt(periods_per_year)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    model_cfg = load_model_config(args.model_config)
    if model_cfg.ebp is None:
        raise SystemExit("config/params.yaml is missing the ebp block")

    ebp_loader = EBPLoader.from_yaml(args.data_config)
    credit = CreditDataLoader.from_yaml(args.data_config)
    ebp_loader.fetch(force_refresh=args.force_refresh)
    asof = args.asof or date.today().isoformat()
    monthly = ebp_loader.available_asof(asof)["ebp"]

    start = credit.config.start_date.isoformat()
    spec = ebp_loader.ebp_cfg
    anchors = credit.fetch_fred(
        [spec.vix_series_id, spec.hy_oas_series_id, spec.t10y2y_series_id],
        start=start,
        end=asof,
        force_refresh=args.force_refresh,
    )
    indicators = build_indicator_matrix(
        anchors[spec.vix_series_id],
        anchors[spec.hy_oas_series_id],
        anchors[spec.t10y2y_series_id],
        vix_only=False,
    )
    daily_full = ebp_loader.disaggregate(
        indicators, asof=asof, config=model_cfg.ebp.disaggregation
    )
    daily_vix = ebp_loader.disaggregate(
        indicators, vix_only=True, asof=asof, config=model_cfg.ebp.disaggregation
    )

    monthly_model = EBPVolatilityModel(
        model_cfg, series_id="EBP", layer="primary_monthly"
    ).fit(monthly)
    daily_full_model = EBPVolatilityModel(
        model_cfg, series_id="EBP_DAILY_FULL", layer="daily_full"
    ).fit(daily_full.daily)
    daily_vix_model = EBPVolatilityModel(
        model_cfg, series_id="EBP_DAILY_VIX", layer="daily_vix_only"
    ).fit(daily_vix.daily)

    monthly_vol = _annualize(monthly_model.conditional_volatility(), 12.0)
    daily_full_vol = _annualize(daily_full_model.conditional_volatility(), 252.0)
    daily_vix_vol = _annualize(daily_vix_model.conditional_volatility(), 252.0)
    aligned = monthly_vol.index.union(daily_full_vol.index).union(daily_vix_vol.index)
    monthly_on_daily = monthly_vol.reindex(aligned).ffill()
    report = write_comparative_report(
        monthly_official=monthly_on_daily,
        daily_full_anchor=daily_full_vol.reindex(aligned),
        daily_vix_only=daily_vix_vol.reindex(aligned),
        config=model_cfg,
    )
    sensitivity = ebp_loader.lag_sensitivity(pd.bdate_range(end=asof, periods=80))
    print(monthly_model.summary().as_text())
    print()
    print(report.table.tail().to_string())
    print(
        f"\nrobustness corr(full, vix-only)={report.robustness_correlation:.3f} "
        f"artifact={report.robustness_artifact}"
    )
    print(
        f"publication-lag sensitivity min Δ-corr={sensitivity.min_correlation:.3f} "
        f"fragile={sensitivity.fragile} lags={sensitivity.lags}"
    )
    print(f"\nWrote {report.plot_path}")
    if sensitivity.fragile:
        print(
            "IMPLEMENTATION RISK: EBP signal changes materially across "
            "publication lags 30/45/60."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
