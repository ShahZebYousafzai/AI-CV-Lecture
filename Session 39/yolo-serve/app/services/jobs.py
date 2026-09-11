"""A tiny in-process job store for fire-and-forget batch inference.

Why this exists: a 200 image batch takes minutes. Holding an HTTP
connection open for minutes is how you collect gateway timeouts, so the
client submits work, gets a job id back immediately, and polls.

Why this is NOT the production answer: the jobs live in this process's
memory. Scale to two containers and a poll can land on the wrong one; a
restart loses everything. In production this store is Redis, and the worker
is a separate consumer (Celery, RQ, or an SQS consumer). The interface
below is deliberately the same shape, so swapping the backend touches only
this file.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from PIL import Image

from app.config import Settings
from app.core.errors import JobNotFoundError
from app.schemas.detection import JobState, JobStatusResponse, PredictionResponse
from app.services.inference import InferenceService

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    image_count: int
    status: JobState = "queued"
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: PredictionResponse | None = None

    def to_response(self) -> JobStatusResponse:
        return JobStatusResponse(
            job_id=self.job_id,
            status=self.status,
            image_count=self.image_count,
            submitted_at=self.submitted_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            error=self.error,
            result=self.result,
        )


class JobStore:
    def __init__(self, inference: InferenceService, settings: Settings):
        self.inference = inference
        self.settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def submit(
        self,
        images: list[Image.Image],
        *,
        filenames: list[str | None],
        conf: float | None = None,
        iou: float | None = None,
    ) -> Job:
        job = Job(job_id=str(uuid.uuid4()), image_count=len(images))

        async with self._lock:
            self._evict_expired()
            self._jobs[job.job_id] = job

        # Runs on the event loop, but the heavy call inside inference.predict
        # is still pushed to a worker thread, so nothing here blocks.
        asyncio.create_task(self._run(job, images, filenames, conf, iou))
        return job

    async def _run(
        self,
        job: Job,
        images: list[Image.Image],
        filenames: list[str | None],
        conf: float | None,
        iou: float | None,
    ) -> None:
        job.status = "running"
        job.started_at = time.time()
        try:
            job.result = await self.inference.predict(
                images,
                filenames=filenames,
                conf=conf,
                iou=iou,
                request_id=job.job_id,
            )
            job.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 - the job records its own failure
            logger.exception("Job %s failed", job.job_id)
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = time.time()

    async def get(self, job_id: str) -> Job:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"No job with id {job_id}")
        return job

    def _evict_expired(self) -> None:
        cutoff = time.time() - self.settings.job_ttl_seconds
        stale = [
            jid
            for jid, job in self._jobs.items()
            if job.finished_at is not None and job.finished_at < cutoff
        ]
        for jid in stale:
            del self._jobs[jid]

        # Hard cap so a runaway client cannot exhaust memory.
        overflow = len(self._jobs) - self.settings.max_jobs
        if overflow > 0:
            oldest = sorted(self._jobs.values(), key=lambda j: j.submitted_at)[:overflow]
            for job in oldest:
                self._jobs.pop(job.job_id, None)
