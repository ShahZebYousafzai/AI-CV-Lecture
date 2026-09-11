"""Dependency providers.

Everything a route needs is pulled off `request.app.state`, which is where
the lifespan handler put it at startup. Routes therefore never construct a
model, and tests can swap the state for a fake in one line.
"""

from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.services.detector import Detector
from app.services.inference import InferenceService
from app.services.jobs import JobStore


def get_detector(request: Request) -> Detector:
    return request.app.state.detector


def get_inference(request: Request) -> InferenceService:
    return request.app.state.inference


def get_jobs(request: Request) -> JobStore:
    return request.app.state.jobs


SettingsDep = Annotated[Settings, Depends(get_settings)]
DetectorDep = Annotated[Detector, Depends(get_detector)]
InferenceDep = Annotated[InferenceService, Depends(get_inference)]
JobsDep = Annotated[JobStore, Depends(get_jobs)]
