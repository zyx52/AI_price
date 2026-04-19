from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from data import DataLoader
from engine import PricingEngine, PricingEngineV3
from models import (
    ContinuousRLPricer,
    EnhancedShiftDetector,
    EnsembleDemandForecaster,
    FeatureService,
    IncrementalTrainingManager,
    ParkAttractionGraph,
    QuantileEnsembleForecaster,
    RLPricerV2,
)
from utils.logger import get_logger

logger = get_logger("shadow_backtest_30d")


@dataclass
class StrategySummary:
    strategy: str
    days: int
    avg_price: float
    avg_visitors: float
    avg_revenue: float
    total_revenue: float
    total_visitors: float
    fallback_days: int


def _build_competitor_prices(hist_price: float) -> Dict[str, float]:
    return {
        "A": float(hist_price * 1.03),
        "B": float(hist_price * 0.96),
        "C": float(hist_price * 1.07),
    }


def _simulate_visitors(
    historical_price: float,
    historical_visitors: float,
    new_price: float,
    elasticity: float,
) -> float:
    simulated = historical_visitors * (historical_price / max(new_price, 1.0)) ** elasticity
    return float(np.clip(simulated, 100, settings.park_capacity))


def _strategy_summary(df: pd.DataFrame, prefix: str) -> StrategySummary:
    return StrategySummary(
        strategy=prefix,
        days=len(df),
        avg_price=float(df[f"{prefix}_price"].mean()),
        avg_visitors=float(df[f"{prefix}_visitors"].mean()),
        avg_revenue=float(df[f"{prefix}_revenue"].mean()),
        total_revenue=float(df[f"{prefix}_revenue"].sum()),
        total_visitors=float(df[f"{prefix}_visitors"].sum()),
        fallback_days=int(df[f"{prefix}_fallback"].sum()),
    )


def run_shadow_backtest(
    window_days: int,
    elasticity: float,
    ppo_timesteps: int,
    old_rl_epochs: int,
    output_dir: Path,
) -> Dict[str, Any]:
    loader = DataLoader(source="mock")
    history = loader.load_history()

    if len(history) <= window_days + 60:
        raise ValueError("history too short for train/test split")

    history = history.sort_values("date").reset_index(drop=True)
    train_df = history.iloc[:-window_days].copy()
    test_df = history.iloc[-window_days:].copy()

    logger.info("training old baseline stack")
    old_forecaster = EnsembleDemandForecaster()
    old_forecaster.train(train_df)

    old_rl = RLPricerV2()
    old_rl.train(train_df, epochs=old_rl_epochs)

    old_engine = PricingEngine(
        forecaster=old_forecaster,
        rl_pricer=old_rl,
        shift_detector=None,
        weights=(0.30, 0.45, 0.25),
    )

    logger.info("training probabilistic v3 stack")
    q_forecaster = QuantileEnsembleForecaster()
    q_forecaster.train(train_df)

    feature_service = FeatureService(train_df)

    shift_detector = EnhancedShiftDetector(use_torch=False)
    shift_detector.fit(train_df)

    inc_manager = IncrementalTrainingManager(
        storage_path=str(output_dir / "anomaly_pool"),
        retrain_threshold=10**9,
    )

    graph = ParkAttractionGraph()
    graph.build_default_park()

    v3_rl = ContinuousRLPricer(
        forecaster=q_forecaster,
        history_df=train_df,
        attraction_graph=graph,
    )
    v3_rl.train(total_timesteps=ppo_timesteps)

    new_engine = PricingEngineV3(
        forecaster=q_forecaster,
        rl_pricer=v3_rl,
        feature_service=feature_service,
        shift_detector=shift_detector,
        incremental_manager=inc_manager,
    )

    rows: list[Dict[str, Any]] = []
    prev_old_price: float | None = None
    prev_new_price: float | None = None

    for row in test_df.itertuples(index=False):
        date = pd.Timestamp(row.date).strftime("%Y-%m-%d")
        competitor_prices = _build_competitor_prices(float(row.price))

        old_decision = old_engine.decide(
            date=date,
            weather=str(row.weather),
            temperature=float(row.temperature),
            rainfall=float(row.rainfall),
            competitor_prices=competitor_prices,
            day_type=str(row.day_type),
            prev_price=prev_old_price,
        )

        new_decision = new_engine.decide(
            date=date,
            weather=str(row.weather),
            temperature=float(row.temperature),
            rainfall=float(row.rainfall),
            competitor_prices=competitor_prices,
            day_type=str(row.day_type),
            prev_price=prev_new_price,
        )

        old_price = float(old_decision.recommended_price)
        new_price = float(new_decision.recommended_price)

        old_visitors = _simulate_visitors(
            historical_price=float(row.price),
            historical_visitors=float(row.visitors),
            new_price=old_price,
            elasticity=elasticity,
        )
        new_visitors = _simulate_visitors(
            historical_price=float(row.price),
            historical_visitors=float(row.visitors),
            new_price=new_price,
            elasticity=elasticity,
        )

        old_revenue = old_price * old_visitors * (1 + settings.secondary_consumption_ratio)
        new_revenue = new_price * new_visitors * (1 + settings.secondary_consumption_ratio)

        rows.append(
            {
                "date": date,
                "day_type": str(row.day_type),
                "weather": str(row.weather),
                "historical_price": float(row.price),
                "historical_visitors": float(row.visitors),
                "historical_revenue": float(row.revenue_total),
                "old_price": old_price,
                "old_visitors": old_visitors,
                "old_revenue": old_revenue,
                "old_fallback": int(False),
                "new_price": new_price,
                "new_visitors": new_visitors,
                "new_revenue": new_revenue,
                "new_fallback": int(bool(new_decision.fallback_mode)),
                "new_uncertainty_spread": float(
                    new_decision.uncertainty_spread
                    if new_decision.uncertainty_spread is not None
                    else 0.0
                ),
                "new_conservative_by_uncertainty": int(
                    bool(new_decision.conservative_by_uncertainty)
                ),
                "new_shift_level": (
                    str(new_decision.shift_detection.get("level"))
                    if isinstance(new_decision.shift_detection, dict)
                    else "normal"
                ),
            }
        )

        prev_old_price = old_price
        prev_new_price = new_price

    detail_df = pd.DataFrame(rows)
    old_summary = _strategy_summary(detail_df, "old")
    new_summary = _strategy_summary(detail_df, "new")

    revenue_lift_pct = (
        (new_summary.total_revenue - old_summary.total_revenue)
        / max(old_summary.total_revenue, 1.0)
        * 100
    )
    visitors_lift_pct = (
        (new_summary.total_visitors - old_summary.total_visitors)
        / max(old_summary.total_visitors, 1.0)
        * 100
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = output_dir / f"shadow_backtest_30d_detail_{timestamp}.csv"
    summary_path = output_dir / f"shadow_backtest_30d_summary_{timestamp}.json"

    detail_df.to_csv(detail_path, index=False)

    payload = {
        "config": {
            "window_days": window_days,
            "elasticity": elasticity,
            "ppo_timesteps": ppo_timesteps,
            "old_rl_epochs": old_rl_epochs,
            "uncertainty_spread_threshold": settings.uncertainty_spread_threshold,
        },
        "period": {
            "start_date": detail_df["date"].iloc[0],
            "end_date": detail_df["date"].iloc[-1],
            "days": len(detail_df),
        },
        "old_fixed_weight_strategy": asdict(old_summary),
        "new_probabilistic_strategy": asdict(new_summary),
        "comparison": {
            "revenue_lift_pct": revenue_lift_pct,
            "visitors_lift_pct": visitors_lift_pct,
            "new_fallback_days": int(detail_df["new_fallback"].sum()),
            "new_high_uncertainty_days": int(
                (detail_df["new_uncertainty_spread"] > settings.uncertainty_spread_threshold).sum()
            ),
            "new_conservative_days": int(detail_df["new_conservative_by_uncertainty"].sum()),
        },
        "outputs": {
            "detail_csv": str(detail_path),
            "summary_json": str(summary_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== Shadow Backtest (30 Days) ===")
    print(f"period: {payload['period']['start_date']} -> {payload['period']['end_date']}")
    print(
        "old total revenue: "
        f"{old_summary.total_revenue:,.0f} | new total revenue: {new_summary.total_revenue:,.0f}"
    )
    print(
        "revenue lift: "
        f"{payload['comparison']['revenue_lift_pct']:+.2f}% | "
        f"visitors lift: {payload['comparison']['visitors_lift_pct']:+.2f}%"
    )
    print(f"detail: {detail_path}")
    print(f"summary: {summary_path}")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 30-day shadow backtest: probabilistic v3 vs old fixed-weight strategy"
    )
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--elasticity", type=float, default=0.8)
    parser.add_argument("--ppo-timesteps", type=int, default=3000)
    parser.add_argument("--old-rl-epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="./data/backtests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_shadow_backtest(
        window_days=args.window_days,
        elasticity=args.elasticity,
        ppo_timesteps=args.ppo_timesteps,
        old_rl_epochs=args.old_rl_epochs,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
