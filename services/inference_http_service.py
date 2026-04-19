from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.inference_runtime import build_runtime
from utils.date_utils import get_day_type
from utils.logger import get_logger

logger = get_logger("inference_http_service")

_ALLOWED_WEATHERS = {"晴好", "雨", "暴雨", "酷热", "严寒"}


class PricingRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    weather: str = Field("晴好")
    temperature: float = Field(24.0, ge=-30.0, le=50.0)
    rainfall: float = Field(0.0, ge=0.0, le=500.0)
    competitor_prices: Dict[str, float] = Field(default={"A": 310, "B": 280, "C": 350})
    day_type: Optional[str] = Field(None, pattern=r"^(weekday|weekend|holiday|golden_week)$")
    prev_price: Optional[float] = Field(None, ge=1.0, le=10000.0)

    @field_validator("weather")
    @classmethod
    def _validate_weather(cls, v: str) -> str:
        if v not in _ALLOWED_WEATHERS:
            raise ValueError(f"weather must be one of {_ALLOWED_WEATHERS}")
        return v


@asynccontextmanager
async def lifespan(app: FastAPI):
    source = os.getenv("INFERENCE_DATA_SOURCE", "mock")
    ppo_timesteps = int(os.getenv("INFERENCE_PPO_TIMESTEPS", "3000"))

    runtime = await run_in_threadpool(build_runtime, source, ppo_timesteps, False)
    app.state.runtime = runtime

    logger.info("inference http service ready")
    yield
    logger.info("inference http service shutdown")


app = FastAPI(
    title="AI Pricing Inference Service",
    version="1.0.0",
    description="Standalone model inference service (HTTP)",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    runtime = app.state.runtime
    return {
        "status": "ok",
        "history_rows": len(runtime.history),
        "forecaster_trained": runtime.forecaster.is_trained,
        "rl_trained": runtime.rl_pricer.is_trained,
    }


@app.post("/pricing/decide")
async def decide(req: PricingRequest) -> dict:
    try:
        day_type = req.day_type or get_day_type(req.date)
        decision = await run_in_threadpool(
            app.state.runtime.engine.decide,
            date=req.date,
            weather=req.weather,
            temperature=req.temperature,
            rainfall=req.rainfall,
            competitor_prices=req.competitor_prices,
            day_type=day_type,
            prev_price=req.prev_price,
        )
        return decision.to_dict()
    except ValueError as e:
        raise HTTPException(400, {"code": "INVALID_INPUT", "msg": str(e)})
    except Exception as e:
        logger.exception("inference decide failed")
        raise HTTPException(500, {"code": "INTERNAL_ERROR", "msg": str(e)})


@app.post("/pricing/decide-with-quantiles")
async def decide_with_quantiles(req: PricingRequest) -> dict:
    try:
        day_type = req.day_type or get_day_type(req.date)
        decision = await run_in_threadpool(
            app.state.runtime.engine.decide,
            date=req.date,
            weather=req.weather,
            temperature=req.temperature,
            rainfall=req.rainfall,
            competitor_prices=req.competitor_prices,
            day_type=day_type,
            prev_price=req.prev_price,
        )

        feature_row = app.state.runtime.feature_service.build_for_engine(
            date=req.date,
            weather=req.weather,
            temperature=req.temperature,
            rainfall=req.rainfall,
            day_type=day_type,
            base_price=decision.recommended_price,
            competitor_avg=float(sum(req.competitor_prices.values()) / len(req.competitor_prices)),
        )

        curve = await run_in_threadpool(
            app.state.runtime.forecaster.demand_curve_with_quantiles,
            feature_row,
        )
        sample = curve.iloc[::10]

        return {
            "decision": decision.to_dict(),
            "demand_curve_quantiles": [
                {
                    "price": float(r["price"]),
                    "p10_visitors": float(r["visitors_p10"]),
                    "p50_visitors": float(r["visitors_p50"]),
                    "p90_visitors": float(r["visitors_p90"]),
                    "uncertainty_spread": float(r["uncertainty_ratio"]),
                }
                for _, r in sample.iterrows()
            ],
        }
    except ValueError as e:
        raise HTTPException(400, {"code": "INVALID_INPUT", "msg": str(e)})
    except Exception as e:
        logger.exception("inference decide-with-quantiles failed")
        raise HTTPException(500, {"code": "INTERNAL_ERROR", "msg": str(e)})
