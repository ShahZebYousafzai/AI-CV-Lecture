"""Teaching endpoints. Never enable these in production.

`/demo/predict-blocking` is the wrong way to write the handler, kept on
purpose so the failure can be demonstrated rather than described. Set
APP_ENABLE_DEMO_ROUTES=true, then run:

    # terminal 1
    curl -X POST -F "files=@samples/bus.jpg" ... http://localhost:8000/demo/predict-blocking
    # terminal 2, at the same time
    time curl http://localhost:8000/health/live

With the blocking route in flight, that health check waits for the whole
inference to finish. Against the real /predict/batch it answers in
milliseconds. Same model, same machine, one line of difference.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, File, Request, UploadFile

from app.api.deps import DetectorDep, SettingsDep
from app.schemas.detection import PredictionResponse
from app.services.image_io import decode_image

router = APIRouter(prefix="/demo", tags=["demo"])


@router.post(
    "/predict-blocking",
    response_model=PredictionResponse,
    summary="The WRONG way: CPU work directly inside an async handler",
)
async def predict_blocking(
    request: Request,
    detector: DetectorDep,
    settings: SettingsDep,
    files: list[UploadFile] = File(...),
) -> PredictionResponse:
    started = time.perf_counter()

    payloads = [(await f.read(), f.filename) for f in files]
    images = [
        decode_image(
            data,
            filename=name,
            max_bytes=settings.max_upload_bytes,
            max_pixels=settings.max_image_pixels,
        )
        for data, name in payloads
    ]

    # The bug. `async def` does not make this concurrent. This call holds
    # the event loop for its entire duration, so nothing else on this
    # worker, including /health/live, gets served until it returns.
    predictions, inference_ms = detector.predict(
        images, filenames=[name for _, name in payloads]
    )

    return PredictionResponse(
        request_id=request.state.request_id,
        model_name=detector.model_name,
        device=settings.device,
        image_count=len(images),
        inference_ms=round(inference_ms, 2),
        total_ms=round((time.perf_counter() - started) * 1000, 2),
        results=predictions,
    )
