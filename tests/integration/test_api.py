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


def test_general_chat_session_crud_and_status_routes() -> None:
    created = client.post(
        "/api/v1/chat/sessions",
        json={"mode": "general", "preferred_language": "en"},
    )
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    answer = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        json={"message": "What evidence is available?"},
    )
    assert answer.status_code == 200
    assert answer.json()["is_mock"] is True
    assert answer.json()["insufficient_evidence"] is True
    assert client.get("/api/v1/llm/status").status_code == 200
    assert client.get("/api/v1/rag/status").status_code == 200
    assert client.delete(f"/api/v1/chat/sessions/{session_id}").status_code == 204


def test_diagnosis_reports_are_archived_and_downloadable() -> None:
    diagnosis = client.post(
        "/api/v1/diagnose",
        files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
    ).json()
    diagnosis_id = diagnosis["session_id"]

    html = client.get(f"/api/v1/reports/diagnosis/{diagnosis_id}.html")
    evidence = client.get(f"/api/v1/reports/diagnosis/{diagnosis_id}.json")

    assert html.status_code == 200
    assert "Crop Disease Detection and Diagnosis Using Multimodal AI" in html.text
    assert "Numerical severity withheld" in html.text
    assert evidence.status_code == 200
    assert evidence.json()["session_id"] == diagnosis_id


def test_missing_diagnosis_report_returns_not_found() -> None:
    response = client.get("/api/v1/reports/diagnosis/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.html")

    assert response.status_code == 404


def test_websocket_mock_inference() -> None:
    with client.websocket_connect("/api/v1/realtime/ws") as websocket:
        websocket.send_bytes(_image_bytes())
        result = websocket.receive_json()

    assert result["is_mock"] is True
    assert 0 <= result["confidence"] <= 1
    assert result["inference_ms"] >= 0
