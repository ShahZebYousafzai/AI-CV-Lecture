"""Request and response models.

These classes ARE the API contract. FastAPI validates against them and
generates the OpenAPI docs from them, so a change here is a visible,
documented change to the API.
"""

from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Core prediction shapes
# --------------------------------------------------------------------------
class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left edge in pixels")
    y1: float = Field(..., description="Top edge in pixels")
    x2: float = Field(..., description="Right edge in pixels")
    y2: float = Field(..., description="Bottom edge in pixels")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class Detection(BaseModel):
    class_id: int = Field(..., description="COCO class index")
    class_name: str = Field(..., description="Human readable label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox


class ImagePrediction(BaseModel):
    filename: str | None = None
    width: int
    height: int
    detection_count: int
    detections: list[Detection]


class PredictionResponse(BaseModel):
    request_id: str
    model_name: str
    device: str
    image_count: int
    inference_ms: float = Field(..., description="Time inside the model only")
    total_ms: float = Field(..., description="Decode plus inference plus serialise")
    results: list[ImagePrediction]


# --------------------------------------------------------------------------
# JSON (base64) input, for clients that cannot post multipart
# --------------------------------------------------------------------------
class Base64Image(BaseModel):
    filename: str | None = None
    content: str = Field(..., description="Base64 encoded image bytes, data URI prefix allowed")


class Base64BatchRequest(BaseModel):
    images: list[Base64Image] = Field(..., min_length=1)
    conf_threshold: float | None = Field(None, ge=0.0, le=1.0)
    iou_threshold: float | None = Field(None, ge=0.0, le=1.0)


# --------------------------------------------------------------------------
# Async jobs
# --------------------------------------------------------------------------
JobState = Literal["queued", "running", "succeeded", "failed"]


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobState
    image_count: int
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobState
    image_count: int
    submitted_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: PredictionResponse | None = None


# --------------------------------------------------------------------------
# Health and metadata
# --------------------------------------------------------------------------
class LivenessResponse(BaseModel):
    status: Literal["alive"] = "alive"
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "loading"]
    model_loaded: bool
    model_name: str
    device: str
    uptime_seconds: float


class ModelMetadataResponse(BaseModel):
    model_name: str
    device: str
    imgsz: int
    conf_threshold: float
    iou_threshold: float
    max_batch_size: int
    class_count: int
    classes: dict[int, str]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
