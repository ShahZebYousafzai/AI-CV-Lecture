import base64
import time

import pytest

from app.config import get_settings


def _file(name, blob):
    return {"file": (name, blob, "image/png")}


def test_single_prediction_shape(client, sample_image_bytes):
    response = client.post("/predict", files=_file("a.png", sample_image_bytes))
    assert response.status_code == 200

    body = response.json()
    assert body["image_count"] == 1
    assert body["model_name"] == "yolov8n.pt"
    assert body["inference_ms"] > 0
    assert len(body["results"]) == 1

    result = body["results"][0]
    assert result["filename"] == "a.png"
    assert (result["width"], result["height"]) == (640, 480)
    assert result["detection_count"] == len(result["detections"])


def test_batch_returns_one_result_per_image(client, sample_image_bytes, tiny_image_bytes):
    files = [
        ("files", ("a.png", sample_image_bytes, "image/png")),
        ("files", ("b.png", tiny_image_bytes, "image/png")),
        ("files", ("c.png", sample_image_bytes, "image/png")),
    ]
    body = client.post("/predict/batch", files=files).json()

    assert body["image_count"] == 3
    assert [r["filename"] for r in body["results"]] == ["a.png", "b.png", "c.png"]
    assert body["results"][1]["width"] == 64


def test_batch_pays_the_request_overhead_once(client, sample_image_bytes):
    """One batch call does the same work with far less non-model overhead.

    We deliberately do NOT assert that batch inference is faster: on a small
    CPU box it is roughly break even, and a test that asserts a speedup would
    fail on exactly the hardware most students run. What is always true is
    that the per image overhead outside the model shrinks.
    """
    n = 6
    files = [("files", (f"{i}.png", sample_image_bytes, "image/png")) for i in range(n)]

    body = client.post("/predict/batch", files=files).json()
    batch_overhead = body["total_ms"] - body["inference_ms"]

    single_overhead = 0.0
    for i in range(n):
        single = client.post("/predict", files=_file(f"{i}.png", sample_image_bytes)).json()
        single_overhead += single["total_ms"] - single["inference_ms"]

    assert batch_overhead < single_overhead
    assert body["image_count"] == n


def test_base64_endpoint(client, sample_image_bytes):
    payload = {
        "images": [
            {"filename": "a.png", "content": base64.b64encode(sample_image_bytes).decode()},
            {
                "filename": "b.png",
                "content": "data:image/png;base64,"
                + base64.b64encode(sample_image_bytes).decode(),
            },
        ],
        "conf_threshold": 0.5,
    }
    body = client.post("/predict/base64", json=payload).json()
    assert body["image_count"] == 2


def test_annotated_endpoint_returns_png(client, sample_image_bytes):
    response = client.post("/predict/annotated", files=_file("a.png", sample_image_bytes))
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert "X-Detection-Count" in response.headers


def test_confidence_override_is_applied(client, sample_image_bytes):
    low = client.post("/predict?conf=0.01", files=_file("a.png", sample_image_bytes)).json()
    high = client.post("/predict?conf=0.99", files=_file("a.png", sample_image_bytes)).json()
    assert high["results"][0]["detection_count"] <= low["results"][0]["detection_count"]


# --------------------------------------------------------------------------
# failure paths
# --------------------------------------------------------------------------
def test_garbage_bytes_give_422_not_500(client):
    response = client.post("/predict", files=_file("bad.png", b"this is not an image"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"


def test_empty_file_is_rejected(client):
    response = client.post("/predict", files=_file("empty.png", b""))
    assert response.status_code == 422


def test_missing_file_is_a_validation_error(client):
    assert client.post("/predict").status_code == 422


def test_oversized_batch_is_rejected(client, tiny_image_bytes):
    limit = get_settings().max_batch_size
    files = [
        ("files", (f"{i}.png", tiny_image_bytes, "image/png")) for i in range(limit + 1)
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "batch_too_large"


def test_conf_outside_range_is_rejected(client, sample_image_bytes):
    response = client.post("/predict?conf=1.5", files=_file("a.png", sample_image_bytes))
    assert response.status_code == 422


# --------------------------------------------------------------------------
# async jobs
# --------------------------------------------------------------------------
def test_job_lifecycle(client, sample_image_bytes):
    files = [("files", (f"{i}.png", sample_image_bytes, "image/png")) for i in range(4)]
    submit = client.post("/jobs", files=files)

    assert submit.status_code == 202
    job = submit.json()
    assert job["status"] in {"queued", "running"}
    assert job["image_count"] == 4

    for _ in range(120):
        status = client.get(f"/jobs/{job['job_id']}").json()
        if status["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.25)
    else:
        pytest.fail("job never finished")

    assert status["status"] == "succeeded"
    assert status["result"]["image_count"] == 4
    assert status["finished_at"] >= status["started_at"]


def test_unknown_job_is_404(client):
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"
