"""The async bridge between the API layer and the blocking model.

The single most common bug in an ML API looks like this:

    @router.post("/predict")
    async def predict(...):
        return model.predict(image)      # <-- blocks the event loop

`async def` does not make CPU work concurrent. A 300 ms forward pass inside
an async handler freezes the whole server for 300 ms: health checks time
out, other requests queue, and the load balancer starts pulling the box out
of service. The fix is to run blocking work on a worker thread, which is
what `anyio.to_thread.run_sync` does below.

The CapacityLimiter caps how many inferences can be in flight at once so a
traffic spike queues politely instead of thrashing the CPU.
"""

from __future__ import annotations

import time
import uuid

import anyio
from PIL import Image

from app.config import Settings
from app.schemas.detection import PredictionResponse
from app.services.detector import Detector


class InferenceService:
    def __init__(self, detector: Detector, settings: Settings):
        self.detector = detector
        self.settings = settings
        self.limiter = anyio.CapacityLimiter(settings.inference_concurrency)

    async def predict(
        self,
        images: list[Image.Image],
        *,
        filenames: list[str | None] | None = None,
        conf: float | None = None,
        iou: float | None = None,
        request_id: str | None = None,
        started: float | None = None,
    ) -> PredictionResponse:
        started = started if started is not None else time.perf_counter()

        predictions, inference_ms = await anyio.to_thread.run_sync(
            lambda: self.detector.predict(images, conf=conf, iou=iou, filenames=filenames),
            limiter=self.limiter,
        )

        return PredictionResponse(
            request_id=request_id or str(uuid.uuid4()),
            model_name=self.detector.model_name,
            device=self.settings.device,
            image_count=len(images),
            inference_ms=round(inference_ms, 2),
            total_ms=round((time.perf_counter() - started) * 1000, 2),
            results=predictions,
        )
