from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import grpc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.inference_runtime import InferenceRuntime, build_runtime
from utils.date_utils import get_day_type
from utils.logger import get_logger

logger = get_logger("inference_grpc_server")

try:
    from services.grpc import inference_pb2, inference_pb2_grpc
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing generated gRPC modules. Run protoc first: \n"
        "python -m grpc_tools.protoc "
        "-I services/grpc/proto "
        "--python_out=services/grpc "
        "--grpc_python_out=services/grpc "
        "services/grpc/proto/inference.proto"
    ) from e


def _to_decision_proto(decision: dict):
    return inference_pb2.PricingDecision(
        date=str(decision.get("date", "")),
        recommended_price=float(decision.get("recommended_price", 0.0)),
        predicted_visitors=float(decision.get("predicted_visitors", 0.0)),
        predicted_revenue=float(decision.get("predicted_revenue", 0.0)),
        load_rate=float(decision.get("load_rate", 0.0)),
        confidence=float(decision.get("confidence", 0.0)),
        reasoning=str(decision.get("reasoning", "")),
        visitors_p10=float(decision.get("visitors_p10") or 0.0),
        visitors_p50=float(decision.get("visitors_p50") or 0.0),
        visitors_p90=float(decision.get("visitors_p90") or 0.0),
        uncertainty_spread=float(
            decision.get("uncertainty_spread")
            or decision.get("uncertainty_ratio")
            or 0.0
        ),
        fallback_mode=bool(decision.get("fallback_mode", False)),
        conservative_by_uncertainty=bool(
            decision.get("conservative_by_uncertainty", False)
        ),
    )


class PricingInferenceServicer(inference_pb2_grpc.PricingInferenceServicer):
    def __init__(self, runtime: InferenceRuntime):
        self.runtime = runtime

    @staticmethod
    def _extract_request(req) -> Tuple[str, str, float, float, Dict[str, float], str | None, float | None]:
        day_type = req.day_type if req.day_type else None
        prev_price = req.prev_price if req.prev_price > 0 else None
        competitor_prices = dict(req.competitor_prices) if req.competitor_prices else {"A": 310, "B": 280, "C": 350}
        return (
            req.date,
            req.weather,
            float(req.temperature),
            float(req.rainfall),
            {k: float(v) for k, v in competitor_prices.items()},
            day_type,
            prev_price,
        )

    async def Decide(self, request, context):
        try:
            date, weather, temperature, rainfall, competitor_prices, day_type, prev_price = self._extract_request(request)
            decision = await asyncio.to_thread(
                self.runtime.engine.decide,
                date=date,
                weather=weather,
                temperature=temperature,
                rainfall=rainfall,
                competitor_prices=competitor_prices,
                day_type=day_type or get_day_type(date),
                prev_price=prev_price,
            )
            return _to_decision_proto(decision.to_dict())
        except Exception as e:
            logger.exception("grpc Decide failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def DecideWithQuantiles(self, request, context):
        try:
            date, weather, temperature, rainfall, competitor_prices, day_type, prev_price = self._extract_request(request)
            resolved_day_type = day_type or get_day_type(date)

            decision = await asyncio.to_thread(
                self.runtime.engine.decide,
                date=date,
                weather=weather,
                temperature=temperature,
                rainfall=rainfall,
                competitor_prices=competitor_prices,
                day_type=resolved_day_type,
                prev_price=prev_price,
            )
            decision_dict = decision.to_dict()

            feature_row = self.runtime.feature_service.build_for_engine(
                date=date,
                weather=weather,
                temperature=temperature,
                rainfall=rainfall,
                day_type=resolved_day_type,
                base_price=float(decision_dict.get("recommended_price", 0.0)),
                competitor_avg=float(sum(competitor_prices.values()) / len(competitor_prices)),
            )
            curve = await asyncio.to_thread(
                self.runtime.forecaster.demand_curve_with_quantiles,
                feature_row,
            )
            sample = curve.iloc[::10]

            points = [
                inference_pb2.QuantilePoint(
                    price=float(r["price"]),
                    p10_visitors=float(r["visitors_p10"]),
                    p50_visitors=float(r["visitors_p50"]),
                    p90_visitors=float(r["visitors_p90"]),
                    uncertainty_spread=float(r["uncertainty_ratio"]),
                )
                for _, r in sample.iterrows()
            ]

            return inference_pb2.DecideWithQuantilesResponse(
                decision=_to_decision_proto(decision_dict),
                demand_curve_quantiles=points,
            )
        except Exception as e:
            logger.exception("grpc DecideWithQuantiles failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


async def serve(host: str, port: int, source: str, ppo_timesteps: int) -> None:
    runtime = await asyncio.to_thread(build_runtime, source, ppo_timesteps, False)

    server = grpc.aio.server()
    inference_pb2_grpc.add_PricingInferenceServicer_to_server(
        PricingInferenceServicer(runtime),
        server,
    )

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("grpc inference service started at %s", listen_addr)
    await server.wait_for_termination()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone gRPC inference server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--source", default=os.getenv("INFERENCE_DATA_SOURCE", "mock"))
    parser.add_argument(
        "--ppo-timesteps",
        type=int,
        default=int(os.getenv("INFERENCE_PPO_TIMESTEPS", "3000")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(serve(args.host, args.port, args.source, args.ppo_timesteps))


if __name__ == "__main__":
    main()
