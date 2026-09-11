"""Test fixtures.

`TestClient` as a context manager runs the real lifespan, so these tests
exercise the actual startup path: weights load, warmup runs, state is wired
up. Slower than mocking, and worth it, because startup is exactly where a
model server breaks.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app


@pytest.fixture(scope="session")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    """A synthetic image. Detections may be zero, which is a valid response."""
    image = Image.new("RGB", (640, 480), (40, 60, 70))
    return _png_bytes(image)


@pytest.fixture(scope="session")
def tiny_image_bytes() -> bytes:
    return _png_bytes(Image.new("RGB", (64, 64), (200, 200, 200)))
