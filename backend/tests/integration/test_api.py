import itertools

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
_photo_counter = itertools.count()


def _image_bytes() -> bytes:
    """A synthetic leaf; each call is numbered so it is a new photo, not a saved one."""
    image = np.full((180, 240, 3), 245, dtype=np.uint8)
    number = str(next(_photo_counter))
    cv2.putText(image, number, (5, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
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


def test_uploaded_images_are_stored_in_local_database() -> None:
    data = _image_bytes()
    diagnosis = client.post("/api/v1/diagnose", files={"file": ("my leaf.jpg", data, "image/jpeg")})
    diagnosis_id = diagnosis.json()["session_id"]
    assert diagnosis.json()["from_saved"] is False

    listing = client.get("/api/v1/images").json()
    stored = next(item for item in listing["images"] if item["diagnosis_id"] == diagnosis_id)
    assert listing["enabled"] is True
    assert stored["filename"] == "my leaf.jpg"
    assert stored["size_bytes"] == len(data)
    assert stored["is_mock"] is True

    image = client.get(stored["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert image.content == data

    assert client.delete(f"/api/v1/images/{diagnosis_id}").status_code == 204
    assert client.get(stored["image_url"]).status_code == 404
    assert client.delete(f"/api/v1/images/{diagnosis_id}").status_code == 404


def test_same_photo_returns_saved_diagnosis_unless_reanalysed() -> None:
    data = _image_bytes()
    first = client.post("/api/v1/diagnose", files={"file": ("a.jpg", data, "image/jpeg")}).json()
    again = client.post("/api/v1/diagnose", files={"file": ("b.jpg", data, "image/jpeg")}).json()

    assert again["from_saved"] is True and again["saved_at"]
    assert again["session_id"] == first["session_id"]
    assert again["prediction"] == first["prediction"]

    fresh = client.post(
        "/api/v1/diagnose?reanalyse=true", files={"file": ("c.jpg", data, "image/jpeg")}
    ).json()
    assert fresh["from_saved"] is False
    assert fresh["session_id"] != first["session_id"]
    images = client.get("/api/v1/images").json()["images"]
    rows = [item for item in images if item["sha256"] == images[0]["sha256"]]
    # One row per photo: the newest diagnosis, keeping the original filename.
    assert [(row["diagnosis_id"], row["filename"]) for row in rows] == [
        (fresh["session_id"], "a.jpg")
    ]


def test_projects_group_photos_with_their_diagnoses() -> None:
    project = client.post("/api/v1/projects", json={"name": "North field tomatoes"}).json()
    project_id = project["project_id"]
    assert project["image_count"] == 0

    ids = []
    for plant in ("Plant 1", "Plant 2"):
        response = client.post(
            "/api/v1/diagnose",
            files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
            data={"project_id": project_id, "note": plant},
        )
        ids.append(response.json()["session_id"])

    detail = client.get(f"/api/v1/projects/{project_id}").json()
    assert detail["image_count"] == 2
    assert sorted(item["note"] for item in detail["images"]) == ["Plant 1", "Plant 2"]
    assert sum(item["count"] for item in detail["summary"]) == 2
    assert [p["project_id"] for p in client.get("/api/v1/projects").json()].count(project_id) == 1

    relabel = client.patch(f"/api/v1/images/{ids[0]}", json={"note": "Plant 1 (row 3)"})
    assert relabel.json()["note"] == "Plant 1 (row 3)"
    removed = client.patch(f"/api/v1/images/{ids[1]}", json={"project_id": None})
    assert removed.json()["project_id"] is None
    assert client.get(f"/api/v1/projects/{project_id}").json()["image_count"] == 1

    unknown = client.post(
        "/api/v1/diagnose",
        files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
        data={"project_id": "missing"},
    )
    assert unknown.status_code == 404

    assert client.delete(f"/api/v1/projects/{project_id}").status_code == 204
    assert client.get(f"/api/v1/projects/{project_id}").status_code == 404
    # Deleting a project keeps its photos.
    assert client.get(f"/api/v1/images/{ids[0]}/file").status_code == 200


def test_project_chat_and_project_agent_dashboard() -> None:
    from app.services.chat_service import project_evidence
    from app.services.session_store import session_store

    project_id = client.post("/api/v1/projects", json={"name": "Greenhouse 2"}).json()[
        "project_id"
    ]
    ids = [
        client.post(
            "/api/v1/diagnose",
            files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
            data={"project_id": project_id, "note": f"Plant {n}"},
        ).json()["session_id"]
        for n in (1, 2)
    ]

    evidence = project_evidence(project_id)
    assert evidence["context_type"] == "project" and evidence["photo_count"] == 2
    assert sorted(photo["label"] for photo in evidence["photos"]) == ["Plant 1", "Plant 2"]

    session = client.post(
        "/api/v1/chat/sessions", json={"mode": "general", "project_id": project_id}
    ).json()
    assert session["project_id"] == project_id
    reply = client.post(
        f"/api/v1/chat/sessions/{session['session_id']}/messages",
        json={"message": "Which plants need attention first?"},
    )
    assert reply.status_code == 200
    assert client.post(
        "/api/v1/chat/sessions", json={"mode": "general", "project_id": "missing"}
    ).status_code == 404

    client.post("/api/v1/agents/diagnose/run", json={"session_id": ids[0]})
    dashboard = {
        row["diagnosis_id"]: row["memory"]
        for row in client.get(f"/api/v1/agents/projects/{project_id}/memory").json()
    }
    assert dashboard[ids[0]]["has_diagnosis"] is True
    assert dashboard[ids[1]]["has_diagnosis"] is False

    again = client.post("/api/v1/agents/diagnose/run", json={"session_id": ids[0]}).json()
    assert again["reused"] is True and again["llm_calls"] == 0
    assert again["output"]["candidates"]
    forced = client.post(
        "/api/v1/agents/diagnose/run", json={"session_id": ids[0], "rerun": True}
    ).json()
    assert forced["reused"] is False
    # The project chat sees the agents' saved findings for each photo.
    photos = {photo["label"]: photo for photo in project_evidence(project_id)["photos"]}
    assert photos["Plant 1"]["agent_findings"]["doctor"]["most_likely"]
    assert photos["Plant 2"]["agent_findings"] is None

    # An expired chat session is rebuilt from the diagnosis archive for the agents.
    session_store.delete(ids[1])
    run = client.post("/api/v1/agents/plan_followup/run", json={"session_id": ids[1]})
    assert run.status_code == 200
    assert run.json()["output"]["reinspect_in_days"] >= 1
