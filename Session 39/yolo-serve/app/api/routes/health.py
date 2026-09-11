"""Health and metadata endpoints.

Liveness and readiness are different questions and need different answers:

  /health/live   "is the process up?"     -> restart the container if no
  /health/ready  "can it serve traffic?"  -> send traffic here if yes

A model server takes seconds to load its weights. If the load balancer used
liveness to decide routing, it would send requests into a process that has
no model yet. Splitting the two is what makes a rolling deploy seamless.
"""

import time

from fastapi import APIRouter

from app.api.deps import DetectorDep, SettingsDep
from app.schemas.detection import (
    LivenessResponse,
    ModelMetadataResponse,
    ReadinessResponse,
)

router = APIRouter(tags=["health"])

_PROCESS_START = time.time()


@router.get("/health/live", response_model=LivenessResponse, summary="Liveness probe")
async def liveness(settings: SettingsDep) -> LivenessResponse:
    return LivenessResponse(service=settings.app_name, version=settings.version)


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness(detector: DetectorDep, settings: SettingsDep) -> ReadinessResponse:
    return ReadinessResponse(
        status="ready" if detector.is_ready else "loading",
        model_loaded=detector.is_ready,
        model_name=detector.model_name,
        device=settings.device,
        uptime_seconds=round(time.time() - _PROCESS_START, 1),
    )


@router.get("/metadata", response_model=ModelMetadataResponse, summary="Model metadata")
async def metadata(detector: DetectorDep, settings: SettingsDep) -> ModelMetadataResponse:
    names = detector.names
    return ModelMetadataResponse(
        model_name=detector.model_name,
        device=settings.device,
        imgsz=settings.imgsz,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
        max_batch_size=settings.max_batch_size,
        class_count=len(names),
        classes=names,
    )
