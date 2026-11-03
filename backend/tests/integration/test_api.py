import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

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


def _diagnose() -> dict:
    return client.post(
        "/api/v1/diagnose",
        files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
    ).json()


def test_diagnosis_chat_runs_agent_team_in_demo_mode() -> None:
    diagnosis = _diagnose()
    response = client.post(
        f"/api/v1/chat/sessions/{diagnosis['session_id']}/messages",
        json={"message": "What should I do?"},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["is_mock"] is True
    actions = [step["action"] for step in body["agent_trace"]]
    assert actions[:2] == ["retrieve_evidence", "diagnose"]
    assert {step["decided_by"] for step in body["agent_trace"]} == {"rule"}
    assert "DEMO placeholder" in body["answer"] or body["questions"]
    # the trace is persisted with the assistant message
    history = client.get(f"/api/v1/chat/{diagnosis['session_id']}/history").json()["messages"]
    assert history[-1]["metadata"]["agent_trace"]


def test_reassess_links_new_photo_and_compares() -> None:
    first, second = _diagnose(), _diagnose()
    session_id = first["session_id"]
    client.post(f"/api/v1/chat/sessions/{session_id}/messages", json={"message": "Help"})
    response = client.post(
        f"/api/v1/chat/sessions/{session_id}/reassess",
        json={"diagnosis_id": second["session_id"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert "plan_followup" in [step["action"] for step in body["agent_trace"]]
    assert body["follow_up"]["due_date"]
    same = client.post(
        f"/api/v1/chat/sessions/{session_id}/reassess", json={"diagnosis_id": session_id}
    )
    assert same.status_code == 422


def test_single_agent_runs_prerequisites_and_remembers_results() -> None:
    session_id = _diagnose()["session_id"]
    first = client.post("/api/v1/agents/assess_risk/run", json={"session_id": session_id})
    body = first.json()
    assert first.status_code == 200 and body["is_mock"] is True
    assert [step["action"] for step in body["steps"]] == [
        "retrieve_evidence",
        "diagnose",
        "assess_risk",
    ]
    assert body["steps"][-1]["decided_by"] == "user"
    assert body["output"]["level"] and body["memory"]["has_diagnosis"]

    # the diagnosis is remembered, so the next agent does not redo it (tests have no knowledge
    # base, so the cheap evidence search is simply retried)
    second = client.post("/api/v1/agents/plan_followup/run", json={"session_id": session_id})
    actions = [step["action"] for step in second.json()["steps"]]
    assert "diagnose" not in actions and actions[-1] == "plan_followup"
    assert second.json()["output"]["due_date"]

    assert client.delete(f"/api/v1/agents/memory/{session_id}").status_code == 204
    memory = client.get(f"/api/v1/agents/memory/{session_id}").json()
    assert memory["has_diagnosis"] is False and memory["passages"] == 0


def test_single_agent_validation() -> None:
    session_id = _diagnose()["session_id"]
    general = client.post("/api/v1/chat/sessions", json={"mode": "general"}).json()
    run = "/api/v1/agents/{}/run"
    assert client.post(run.format("finish"), json={"session_id": session_id}).status_code == 422
    assert (
        client.post(run.format("answer_question"), json={"session_id": session_id}).status_code
        == 422
    )
    assert client.post(run.format("diagnose"), json={"session_id": "missing"}).status_code == 404
    assert (
        client.post(run.format("diagnose"), json={"session_id": general["session_id"]}).status_code
        == 409
    )


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
