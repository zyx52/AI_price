from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import settings
from data import DataLoader
from engine import PricingEngineV3
from models import (
    ContinuousRLPricer,
    EnhancedShiftDetector,
    FeatureService,
    IncrementalTrainingManager,
    ParkAttractionGraph,
    QuantileEnsembleForecaster,
)
from utils.logger import get_logger

logger = get_logger("inference_runtime")


@dataclass
class InferenceRuntime:
    history: pd.DataFrame
    forecaster: QuantileEnsembleForecaster
    feature_service: FeatureService
    shift_detector: EnhancedShiftDetector
    incremental_manager: IncrementalTrainingManager
    rl_pricer: ContinuousRLPricer
    engine: PricingEngineV3


def build_runtime(
    source: str = "mock",
    ppo_timesteps: int = 3000,
    use_torch_ae: bool = False,
) -> InferenceRuntime:
    """Build a standalone runtime for external inference service."""
    logger.info(
        "building inference runtime | source=%s | ppo_timesteps=%s",
        source,
        ppo_timesteps,
    )

    loader = DataLoader(source=source)
    history = loader.load_history()

    forecaster = QuantileEnsembleForecaster()
    forecaster.train(history)

    feature_service = FeatureService(history)

    shift_detector = EnhancedShiftDetector(use_torch=use_torch_ae)
    shift_detector.fit(history)

    incremental_manager = IncrementalTrainingManager(
        storage_path="./data/anomalies",
        retrain_threshold=20,
        retrain_callback=lambda records: logger.warning(
            "incremental retrain callback called | records=%s", len(records)
        ),
    )

    graph = ParkAttractionGraph()
    graph.build_default_park()

    rl_pricer = ContinuousRLPricer(
        forecaster=forecaster,
        history_df=history,
        attraction_graph=graph,
    )
    rl_pricer.train(total_timesteps=ppo_timesteps)

    engine = PricingEngineV3(
        forecaster=forecaster,
        rl_pricer=rl_pricer,
        feature_service=feature_service,
        shift_detector=shift_detector,
        incremental_manager=incremental_manager,
    )

    logger.info(
        "inference runtime ready | rows=%s | uncertainty_threshold=%.2f",
        len(history),
        settings.uncertainty_spread_threshold,
    )

    return InferenceRuntime(
        history=history,
        forecaster=forecaster,
        feature_service=feature_service,
        shift_detector=shift_detector,
        incremental_manager=incremental_manager,
        rl_pricer=rl_pricer,
        engine=engine,
    )
