def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_liveness(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_reports_a_loaded_model(client):
    body = client.get("/health/ready").json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True


def test_metadata_lists_coco_classes(client):
    body = client.get("/metadata").json()
    assert body["class_count"] == 80
    assert body["classes"]["0"] == "person"


def test_every_response_carries_a_request_id(client):
    response = client.get("/health/live")
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_client_supplied_request_id_is_echoed(client):
    response = client.get("/health/live", headers={"X-Request-ID": "trace-123"})
    assert response.headers["X-Request-ID"] == "trace-123"


def test_openapi_schema_builds(client):
    schema = client.get("/openapi.json").json()
    assert "/predict/batch" in schema["paths"]
