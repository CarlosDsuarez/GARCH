"""Walk-forward backtest of the dislocation score (monthly GARCH reestimation).

Usage
-----
    python scripts/run_signal_backtest.py

GARCH is re-estimated on the first business day of each month using data
strictly before that date. Parameters stay frozen; daily sigma updates by
recursion. A single full-sample fit used for the whole backtest is forbidden.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from arch.univariate import EGARCH, SkewStudent, StudentsT, arch_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest.schema import load_backtest_config  # noqa: E402
from backtest.signal_backtest import (  # noqa: E402
    FrozenParams,
    WalkForwardBacktester,
)
from data.credit_loader import CreditDataLoader  # noqa: E402
from data.ebp import EBPLoader  # noqa: E402
from models.ebp_garch import EBPVolatilityModel, build_ebp_stress_return  # noqa: E402
from models.oas_egarch import (  # noqa: E402
    OASVolatilityModel,
    _ARMA11,
    build_credit_stress_return,
)
from models.schema import ModelConfig, load_model_config  # noqa: E402
from signals.dislocation import build_default_proxy, load_option_b_default_rate  # noqa: E402
from signals.schema import load_signal_config  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward statistical and economic validation of the dislocation score."
    )
    parser.add_argument("--backtest-config", type=Path, default=ROOT / "config" / "backtest.yaml")
    parser.add_argument("--signal-config", type=Path, default=ROOT / "config" / "signal.yaml")
    parser.add_argument("--data-config", type=Path, default=ROOT / "config" / "data.yaml")
    parser.add_argument("--model-config", type=Path, default=ROOT / "config" / "params.yaml")
    parser.add_argument("--asof", type=str, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _garch_named(params: pd.Series) -> dict[str, float]:
    values = {str(name): float(val) for name, val in params.items()}
    for raw, alias in (("omega", "omega"), ("alpha", "alpha"), ("beta", "beta"), ("gamma", "gamma")):
        for name, val in values.items():
            if raw in name.lower() and alias not in values:
                values[alias] = val
    return values


class OASWalkForwardEstimator:
    """Fit OAS EGARCH on levels strictly before tau; filter with frozen params."""

    def __init__(self, config: ModelConfig, series_id: str) -> None:
        self.config = config
        self.series_id = series_id
        self._fitted: dict[pd.Timestamp, OASVolatilityModel] = {}

    def fit(self, series: pd.Series) -> FrozenParams:
        model = OASVolatilityModel(self.config, series_id=self.series_id)
        model.fit(series)
        through = pd.Timestamp(series.dropna().index.max())
        self._fitted[through] = model
        assert model.result is not None
        values = _garch_named(model.result.params)
        return FrozenParams(fitted_through=through, values=values)

    def filter(self, params: FrozenParams, series: pd.Series) -> pd.Series:
        model = self._fitted[params.fitted_through]
        assert model.result is not None
        stress = build_credit_stress_return(series, self.config)
        vol = self.config.variance
        dist = model.dist
        if model.mean_spec == "AR(1)":
            am = arch_model(
                stress.r.astype("float64"),
                mean="ARX",
                lags=self.config.mean.ar_lags,
                vol=vol.vol,
                p=vol.p,
                o=vol.o,
                q=vol.q,
                dist=dist,
                rescale=vol.rescale,
            )
        elif model.mean_spec == "ARMA(1,1)":
            distribution = StudentsT() if dist == "t" else SkewStudent()
            am = _ARMA11(
                stress.r.astype("float64"),
                volatility=EGARCH(p=vol.p, o=vol.o, q=vol.q),
                distribution=distribution,
                rescale=vol.rescale,
            )
        else:
            raise ValueError(f"unsupported OAS mean spec {model.mean_spec}")
        fixed = am.fix(model.result.params)
        sigma = pd.Series(
            fixed.conditional_volatility.to_numpy(dtype=float),
            index=stress.r.index,
            name="sigma_oas",
        )
        return sigma.reindex(series.index).bfill()


class EBPWalkForwardEstimator:
    """Fit monthly GJR-GARCH; map frozen sigma onto the daily calendar."""

    def __init__(self, config: ModelConfig, series_id: str = "EBP") -> None:
        self.config = config
        self.series_id = series_id
        self._fitted: dict[pd.Timestamp, EBPVolatilityModel] = {}

    def fit(self, series: pd.Series) -> FrozenParams:
        monthly = _month_end_levels(series)
        model = EBPVolatilityModel(self.config, series_id=self.series_id, layer="primary_monthly")
        model.fit(monthly)
        through = pd.Timestamp(series.dropna().index.max())
        self._fitted[through] = model
        assert model.result is not None
        return FrozenParams(fitted_through=through, values=_garch_named(model.result.params))

    def filter(self, params: FrozenParams, series: pd.Series) -> pd.Series:
        model = self._fitted[params.fitted_through]
        assert model.result is not None
        monthly = _month_end_levels(series)
        stress = build_ebp_stress_return(monthly, self.config)
        vol = model.ebp.variance
        dist = model.ebp.distribution.candidates[0]
        kwargs = {
            "vol": vol.vol,
            "p": vol.p,
            "o": vol.o,
            "q": vol.q,
            "dist": dist,
            "rescale": vol.rescale,
        }
        if model.mean_spec == "AR(1)":
            am = arch_model(
                stress.r.astype("float64"),
                mean="ARX",
                lags=model.ebp.mean.ar_lags,
                **kwargs,
            )
        else:
            am = arch_model(stress.r.astype("float64"), mean="Constant", **kwargs)
        fixed = am.fix(model.result.params)
        sigma_m = pd.Series(
            fixed.conditional_volatility.to_numpy(dtype=float),
            index=stress.r.index,
            name="sigma_ebp",
        )
        return sigma_m.reindex(series.index).ffill().bfill()


def _month_end_levels(series: pd.Series) -> pd.Series:
    cleaned = series.dropna().astype("float64").sort_index()
    monthly = cleaned.groupby(cleaned.index.to_period("M")).last()
    monthly.index = monthly.index.to_timestamp("M")
    return monthly.rename(series.name or "ebp")


def _causal_monthly_daily(
    monthly: pd.DataFrame,
    publication_dates: pd.Series,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    avail = monthly.copy()
    avail["publication_date"] = pd.to_datetime(publication_dates.reindex(avail.index))
    avail = avail.dropna(subset=["publication_date"]).sort_values("publication_date")
    days = pd.DataFrame({"date": pd.DatetimeIndex(calendar).sort_values()})
    merged = pd.merge_asof(
        days,
        avail.reset_index().rename(columns={avail.index.name or "index": "month"}),
        left_on="date",
        right_on="publication_date",
        direction="backward",
    )
    return merged.set_index("date")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    bt_cfg = load_backtest_config(args.backtest_config)
    signal_cfg = load_signal_config(args.signal_config)
    model_cfg = load_model_config(args.model_config)
    if model_cfg.ebp is None:
        raise SystemExit("config/params.yaml is missing the ebp block")

    asof = args.asof or date.today().isoformat()
    credit = CreditDataLoader.from_yaml(args.data_config)
    ebp_loader = EBPLoader.from_yaml(args.data_config)
    ebp_frame = ebp_loader.fetch(force_refresh=args.force_refresh)

    start = credit.config.start_date.isoformat()
    ticker = bt_cfg.costs.instrument
    hy_id = credit.config.ebp.hy_oas_series_id if credit.config.ebp else "BAMLH0A0HYM2"
    spec = signal_cfg.default_proxy
    vix_id = credit.config.ebp.vix_series_id if credit.config.ebp else "VIXCLS"
    fred_ids = [hy_id, vix_id, spec.ccc_series_id, spec.bbb_series_id]
    oas = credit.fetch_fred(fred_ids, start=start, end=asof, force_refresh=args.force_refresh)
    prices = credit.fetch_etf([ticker], start=start, end=asof, force_refresh=args.force_refresh)
    credit_returns = prices[ticker].astype("float64").pct_change().rename("credit_return")

    calendar = (
        credit_returns.dropna().index.intersection(oas[hy_id].dropna().index).intersection(
            oas[vix_id].dropna().index
        )
    )
    pubs = ebp_loader.publication_calendar()
    monthly_all = ebp_frame.data
    causal = _causal_monthly_daily(monthly_all, pubs, pd.DatetimeIndex(calendar))
    ebp_level = causal["ebp"].astype("float64").rename("ebp_level")
    gz = causal["gz_spread"].astype("float64")

    if spec.option == "A":
        proxy = build_default_proxy("A", gz_spread=gz, ebp=ebp_level)
    elif spec.option == "C":
        proxy = build_default_proxy(
            "C", ccc_oas=oas[spec.ccc_series_id], bbb_oas=oas[spec.bbb_series_id]
        )
    else:
        proxy = load_option_b_default_rate(spec.option_b.path, asof=asof, spec=spec.option_b)

    backtester = WalkForwardBacktester(
        credit_returns=credit_returns,
        oas_level=oas[hy_id],
        ebp_level=ebp_level,
        default_proxy=proxy,
        vix=oas[vix_id],
        signal_config=signal_cfg,
        backtest_config=bt_cfg,
        oas_estimator=OASWalkForwardEstimator(model_cfg, hy_id),
        ebp_estimator=EBPWalkForwardEstimator(model_cfg),
    )
    result = backtester.run()
    paths = backtester.write_report(result)
    print(result.panel[["score", "weight", "strategy_return"]].tail().to_string())
    print(f"\nImplementable at 3x costs: {result.implementable}")
    print(f"Independent episodes: {result.n_independent_episodes}")
    print(f"Wrote {paths.markdown}")
    print(f"Wrote {paths.html}")
    for note in result.failure_conditions:
        print(f"- {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
