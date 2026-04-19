from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings

try:
    import redis
except ImportError:
    redis = None


def _ok(name: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "ok": True, "details": details}


def _fail(name: str, message: str, details: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"name": name, "ok": False, "error": message}
    if details:
        payload["details"] = details
    return payload


def _decode_map(raw: Dict[Any, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        dk = k.decode() if isinstance(k, bytes) else str(k)
        dv = v.decode() if isinstance(v, bytes) else v
        out[dk] = dv
    return out


def _build_shift_info():
    from models.shift_detector import ShiftDetection, ShiftLevel

    return ShiftDetection(
        level=ShiftLevel.CRITICAL,
        reasons=["redis_e2e_probe"],
        feature_z_scores={},
        unseen_combinations=[],
        model_prediction_variance=0.5,
        adjusted_weights=(1.0, 0.0, 0.0),
        original_weights=(0.3, 0.45, 0.25),
        fallback_triggered=True,
    )


def run_validation(redis_url_override: str | None = None) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    if redis_url_override:
        os.environ["REDIS_URL"] = redis_url_override
        settings.reload()

    checks.append(
        _ok(
            "config",
            {
                "redis_enabled": bool(settings.redis_enabled),
                "redis_url": settings.redis_url,
                "redis_stream_name": settings.redis_stream_name,
            },
        )
    )

    if not settings.redis_enabled:
        checks.append(
            _fail(
                "config.redis_enabled",
                "REDIS_ENABLED=false，未启用 Redis 路径，无法进行 E2E 验收。",
            )
        )
        return {"ok": False, "checks": checks}

    if redis is None:
        checks.append(
            _fail(
                "dependency.redis",
                "未安装 redis 包，请执行: pip install redis",
            )
        )
        return {"ok": False, "checks": checks}

    try:
        client = redis.from_url(settings.redis_url, decode_responses=True)  # type: ignore[union-attr]
    except Exception as e:
        checks.append(_fail("redis.client", f"Redis 客户端初始化失败: {e}"))
        return {"ok": False, "checks": checks}

    try:
        ping_ok = bool(client.ping())
        checks.append(_ok("redis.ping", {"ping": ping_ok}))
    except Exception as e:
        checks.append(_fail("redis.ping", f"PING 失败: {e}"))
        return {"ok": False, "checks": checks}

    connectivity_stream = f"{settings.redis_stream_name}.connectivity_test"
    try:
        msg_id = client.xadd(
            connectivity_stream,
            {
                "event_type": "connectivity_test",
                "source": "ai_pricing_platform",
                "ts": str(int(time.time() * 1000)),
            },
        )
        latest = client.xrevrange(connectivity_stream, count=1)
        has_latest = bool(latest)
        checks.append(
            _ok(
                "redis.stream_connectivity",
                {
                    "stream": connectivity_stream,
                    "xadd_msg_id": msg_id,
                    "latest_found": has_latest,
                },
            )
        )
    except Exception as e:
        checks.append(_fail("redis.stream_connectivity", f"XADD/XREVRANGE 失败: {e}"))
        return {"ok": False, "checks": checks}

    try:
        from data.data_loader import DataLoader
        from utils.feature_cache import feature_cache

        history = DataLoader(source="mock").load_history().tail(120)
        feature_cache.preload_baseline(history)
        health = feature_cache.health()

        backend_ok = health.get("backend") == "redis"
        connected_ok = bool(health.get("redis_connected"))
        baseline_ok = bool(health.get("baseline_loaded"))
        if backend_ok and connected_ok and baseline_ok:
            checks.append(_ok("project.feature_cache", health))
        else:
            checks.append(
                _fail(
                    "project.feature_cache",
                    "FeatureCache 未命中 Redis 后端或基线未加载。",
                    health,
                )
            )
    except Exception as e:
        checks.append(
            _fail(
                "project.feature_cache",
                f"FeatureCache 验证异常: {e}",
                {"traceback": traceback.format_exc()},
            )
        )

    try:
        from models.shift_detector_v2 import AnomalyEventPublisher, IncrementalTrainingManager

        publisher = AnomalyEventPublisher(enabled=True)
        manager = IncrementalTrainingManager(event_publisher=publisher, retrain_threshold=999999)

        params: Dict[str, Any] = {
            "features": {"temperature": 50.0, "rainfall": 300.0},
            "shift_info": _build_shift_info(),
        }
        sig = inspect.signature(manager.label_anomaly)
        if "date" in sig.parameters:
            params["date"] = "2026-04-18"
        else:
            params["timestamp"] = "2026-04-18"
        manager.label_anomaly(**params)

        status = manager.get_status()
        stream_name = status.get("event_stream") or settings.redis_stream_name
        latest = client.xrevrange(stream_name, count=1)
        latest_event_type = None
        payload_fallback = None

        if latest:
            _, raw_data = latest[0]
            decoded = _decode_map(raw_data)
            latest_event_type = decoded.get("event_type")
            payload = decoded.get("payload")
            if isinstance(payload, str):
                try:
                    payload_obj = json.loads(payload)
                    payload_fallback = payload_obj.get("fallback_triggered")
                except json.JSONDecodeError:
                    payload_fallback = None

        event_ok = (
            bool(status.get("event_stream_enabled"))
            and int(status.get("published_events", 0)) >= 1
            and latest_event_type == "ood_fallback_critical"
            and bool(payload_fallback) is True
        )
        details = {
            "manager_status": status,
            "latest_event_type": latest_event_type,
            "payload_fallback_triggered": payload_fallback,
            "stream_name": stream_name,
        }
        if event_ok:
            checks.append(_ok("project.ood_event_pipeline", details))
        else:
            checks.append(
                _fail(
                    "project.ood_event_pipeline",
                    "OOD 事件流未满足验收条件（enabled/published/type/payload）。",
                    details,
                )
            )
    except Exception as e:
        checks.append(
            _fail(
                "project.ood_event_pipeline",
                f"OOD 事件流验证异常: {e}",
                {"traceback": traceback.format_exc()},
            )
        )

    all_ok = all(bool(c.get("ok")) for c in checks)
    return {
        "ok": all_ok,
        "checked_at": int(time.time()),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Redis E2E 一键验收脚本")
    parser.add_argument("--redis-url", default=None, help="可选，覆盖 REDIS_URL")
    args = parser.parse_args()

    result = run_validation(redis_url_override=args.redis_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("ok"):
        print("REDIS_E2E_RESULT=PASS")
        return 0
    print("REDIS_E2E_RESULT=FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
