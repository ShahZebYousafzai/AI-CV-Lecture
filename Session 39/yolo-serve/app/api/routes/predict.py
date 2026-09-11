"""Inference endpoints.

Four ways to ask the same model the same question, each for a different
kind of client:

  POST /predict            one image, multipart      interactive apps
  POST /predict/batch      many images, multipart    offline processing
  POST /predict/base64     many images, JSON         clients stuck on JSON
  POST /predict/annotated  one image, returns PNG    demos and debugging
  POST /jobs               many images, returns id   long running work

Notice how thin these handlers are. Decode, delegate, return. All the real
work lives in the services layer, which is what makes it testable without
an HTTP client.
"""

from __future__ import annotations

import time

import anyio
from fastapi import APIRouter, File, Query, Request, UploadFile, status
from fastapi.responses import Response
from PIL import Image

from app.api.deps import InferenceDep, JobsDep, SettingsDep
from app.config import Settings
from app.core.errors import BatchTooLargeError
from app.schemas.detection import (
    Base64BatchRequest,
    ErrorResponse,
    JobStatusResponse,
    JobSubmitResponse,
    PredictionResponse,
)
from app.services.image_io import (
    decode_base64_image,
    decode_image,
    draw_predictions,
    encode_png,
)

router = APIRouter(
    tags=["inference"],
    responses={
        413: {"model": ErrorResponse, "description": "Payload or batch too large"},
        422: {"model": ErrorResponse, "description": "Image could not be decoded"},
        503: {"model": ErrorResponse, "description": "Model not loaded yet"},
    },
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
async def _read_and_decode(
    files: list[UploadFile], settings: Settings
) -> tuple[list[Image.Image], list[str | None]]:
    """Read uploads off the wire and decode them on a worker thread.

    JPEG decoding is C code that holds the GIL only part of the time, and a
    4K image costs real milliseconds. Same rule as inference: keep it off
    the event loop.
    """
    if len(files) > settings.max_batch_size:
        raise BatchTooLargeError(
            f"Batch of {len(files)} exceeds max_batch_size={settings.max_batch_size}"
        )

    payloads = [(await f.read(), f.filename) for f in files]

    def _decode_all() -> list[Image.Image]:
        return [
            decode_image(
                data,
                filename=name,
                max_bytes=settings.max_upload_bytes,
                max_pixels=settings.max_image_pixels,
            )
            for data, name in payloads
        ]

    images = await anyio.to_thread.run_sync(_decode_all)
    return images, [name for _, name in payloads]


# --------------------------------------------------------------------------
# single image
# --------------------------------------------------------------------------
@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Detect objects in one image",
)
async def predict(
    request: Request,
    inference: InferenceDep,
    settings: SettingsDep,
    file: UploadFile = File(..., description="JPEG or PNG image"),
    conf: float | None = Query(None, ge=0.0, le=1.0, description="Override confidence threshold"),
    iou: float | None = Query(None, ge=0.0, le=1.0, description="Override NMS IoU threshold"),
) -> PredictionResponse:
    started = time.perf_counter()
    images, filenames = await _read_and_decode([file], settings)
    return await inference.predict(
        images,
        filenames=filenames,
        conf=conf,
        iou=iou,
        request_id=request.state.request_id,
        started=started,
    )


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------
@router.post(
    "/predict/batch",
    response_model=PredictionResponse,
    summary="Detect objects in several images in one forward pass",
)
async def predict_batch(
    request: Request,
    inference: InferenceDep,
    settings: SettingsDep,
    files: list[UploadFile] = File(..., description="Two or more images"),
    conf: float | None = Query(None, ge=0.0, le=1.0),
    iou: float | None = Query(None, ge=0.0, le=1.0),
) -> PredictionResponse:
    started = time.perf_counter()
    images, filenames = await _read_and_decode(files, settings)
    return await inference.predict(
        images,
        filenames=filenames,
        conf=conf,
        iou=iou,
        request_id=request.state.request_id,
        started=started,
    )


# --------------------------------------------------------------------------
# base64 JSON
# --------------------------------------------------------------------------
@router.post(
    "/predict/base64",
    response_model=PredictionResponse,
    summary="Batch inference from a JSON body",
)
async def predict_base64(
    request: Request,
    payload: Base64BatchRequest,
    inference: InferenceDep,
    settings: SettingsDep,
) -> PredictionResponse:
    started = time.perf_counter()

    if len(payload.images) > settings.max_batch_size:
        raise BatchTooLargeError(
            f"Batch of {len(payload.images)} exceeds max_batch_size={settings.max_batch_size}"
        )

    def _decode_all() -> list[Image.Image]:
        return [
            decode_base64_image(
                item.content,
                filename=item.filename,
                max_bytes=settings.max_upload_bytes,
                max_pixels=settings.max_image_pixels,
            )
            for item in payload.images
        ]

    images = await anyio.to_thread.run_sync(_decode_all)
    return await inference.predict(
        images,
        filenames=[item.filename for item in payload.images],
        conf=payload.conf_threshold,
        iou=payload.iou_threshold,
        request_id=request.state.request_id,
        started=started,
    )


# --------------------------------------------------------------------------
# annotated image, for demos
# --------------------------------------------------------------------------
@router.post(
    "/predict/annotated",
    summary="Return the image with boxes drawn on it",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Annotated PNG"}},
)
async def predict_annotated(
    request: Request,
    inference: InferenceDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
    conf: float | None = Query(None, ge=0.0, le=1.0),
) -> Response:
    images, filenames = await _read_and_decode([file], settings)
    result = await inference.predict(
        images, filenames=filenames, conf=conf, request_id=request.state.request_id
    )

    def _render() -> bytes:
        return encode_png(draw_predictions(images[0], result.results[0]))

    png = await anyio.to_thread.run_sync(_render)
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-Detection-Count": str(result.results[0].detection_count),
            "X-Inference-Ms": str(result.inference_ms),
        },
    )


# --------------------------------------------------------------------------
# async jobs
# --------------------------------------------------------------------------
@router.post(
    "/jobs",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a batch and get a job id back immediately",
)
async def submit_job(
    request: Request,
    jobs: JobsDep,
    settings: SettingsDep,
    files: list[UploadFile] = File(...),
    conf: float | None = Query(None, ge=0.0, le=1.0),
) -> JobSubmitResponse:
    images, filenames = await _read_and_decode(files, settings)
    job = await jobs.submit(images, filenames=filenames, conf=conf)
    return JobSubmitResponse(
        job_id=job.job_id,
        status=job.status,
        image_count=job.image_count,
        poll_url=str(request.url_for("get_job", job_id=job.job_id)),
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    name="get_job",
    summary="Poll a job",
    responses={404: {"model": ErrorResponse}},
)
async def get_job(job_id: str, jobs: JobsDep) -> JobStatusResponse:
    job = await jobs.get(job_id)
    return job.to_response()
