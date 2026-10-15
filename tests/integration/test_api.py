import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def _image_bytes() -> bytes:
    image = np.full((180, 240, 3), 245, dtype=np.uint8)
    cv2.ellipse(image, (120, 90), (70, 45), -20, 0, 360, (35, 145, 45), -1)
    cv2.circle(image, (115, 85), 15, (35, 90, 150), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_health_and_capabilities() -> None:
    health = client.get("/api/v1/health")
    capabilities = client.get("/api/v1/system/capabilities")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert capabilities.status_code == 200
    assert capabilities.json()["model_mode"] == "demo"
    assert capabilities.json()["model_loaded"] is False


def test_mock_diagnosis_is_explicit_and_has_artifacts() -> None:
    response = client.post(
        "/api/v1/diagnose",
        files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["is_mock"] is True
    assert body["advisory"]["is_mock"] is True
    assert body["severity"]["is_experimental"] is True
    assert body["artifacts"]["preprocessed_image"].startswith("data:image/jpeg;base64,")
    assert body["artifacts"]["leaf_isolated"].startswith("data:image/jpeg;base64,")
    assert body["artifacts"]["lesion_overlay"].startswith("data:image/jpeg;base64,")
    assert any("DEMO MODE" in warning for warning in body["warnings"])


def test_upload_rejects_non_image_mime() -> None:
    response = client.post(
        "/api/v1/diagnose",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_chat_uses_diagnosis_session_and_preserves_history() -> None:
    diagnosis = client.post(
        "/api/v1/diagnose",
        files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
    ).json()
    response = client.post(
        "/api/v1/chat",
        json={"session_id": diagnosis["session_id"], "message": "How reliable is this?"},
    )

    assert response.status_code == 200
    assert response.json()["is_mock"] is True
    history = client.get(f"/api/v1/chat/{diagnosis['session_id']}/history")
    assert [item["role"] for item in history.json()["messages"]] == ["user", "assistant"]


def test_chat_rejects_unknown_session() -> None:
    response = client.post(
        "/api/v1/chat",
        json={"session_id": "missing", "message": "What was detected?"},
    )
    assert response.status_code == 404


def test_websocket_mock_inference() -> None:
    with client.websocket_connect("/api/v1/realtime/ws") as websocket:
        websocket.send_bytes(_image_bytes())
        result = websocket.receive_json()

    assert result["is_mock"] is True
    assert 0 <= result["confidence"] <= 1
    assert result["inference_ms"] >= 0
