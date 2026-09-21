"""HTTP surface: job lifecycle, kind routing, and file-serving safety."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from image_gen.api.app import create_app


@pytest.fixture
def client(config):
    app = create_app(config)
    with TestClient(app) as client:
        client.app.state.runner.start()
        yield client


def _generate(client, kind="skybox", **body):
    payload = {"prompt": "a calm forest", "seed": 1234, "scene_id": "p07"}
    payload.update(body)
    return client.post(f"/generate/{kind}", json=payload)


def _await_job(client, job_id):
    client.app.state.runner.wait_idle()
    return client.get(f"/jobs/{job_id}").json()


# -- lifecycle ------------------------------------------------------------


def test_generate_then_poll_then_fetch_the_file(client):
    response = _generate(client)
    assert response.status_code == 202

    job = response.json()
    assert job["status"] == "queued"
    assert job["scene_id"] == "p07"

    done = _await_job(client, job["job_id"])
    assert done["status"] == "done"
    assert done["file"] == "p07/p07_skybox_1234.png"

    served = client.get(f"/files/{done['file']}")
    assert served.status_code == 200
    assert served.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_scene_id_defaults_to_a_timestamp_when_omitted(client):
    """A bare curl with no scene_id still has to work -- the sketched API had
    no scene_id field at all."""
    job = _generate(client, scene_id=None).json()
    assert job["scene_id"].startswith("scene_")


def test_failed_generation_surfaces_on_the_job(client, monkeypatch):
    from image_gen.backends.stub import StubGenerator

    monkeypatch.setattr(
        StubGenerator,
        "_render",
        lambda self, request, params: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    job = _generate(client).json()
    done = _await_job(client, job["job_id"])

    assert done["status"] == "failed"
    assert done["file"] is None
    assert "boom" in done["error"]


def test_unknown_job_is_404(client):
    assert client.get("/jobs/doesnotexist").status_code == 404


# -- kind routing ---------------------------------------------------------


def test_every_configured_kind_is_routable(client):
    assert _generate(client, kind="skybox").status_code == 202
    assert _generate(client, kind="image").status_code == 202


def test_unconfigured_kind_is_404(client):
    response = _generate(client, kind="texture")
    assert response.status_code == 404
    assert "unknown asset kind" in response.json()["detail"]


# -- validation -----------------------------------------------------------


def test_bad_aspect_ratio_is_rejected_before_queueing(client):
    """A 422 now beats a job that fails a minute later on the GPU."""
    response = _generate(client, width=512, height=512)
    assert response.status_code == 422
    assert "aspect ratio" in response.json()["detail"]


def test_unsafe_scene_id_is_rejected(client):
    assert _generate(client, scene_id="../escape").status_code == 422


def test_seed_is_required(client):
    response = client.post("/generate/skybox", json={"prompt": "x"})
    assert response.status_code == 422


def test_unknown_field_is_rejected(client):
    response = client.post(
        "/generate/skybox", json={"prompt": "x", "seed": 1, "stpes": 30}
    )
    assert response.status_code == 422


# -- file serving safety --------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../../../etc/passwd",
        "..%2F..%2Fsecret.txt",
        "p07/../../escape.png",
    ],
)
def test_path_traversal_is_refused(client, path):
    response = client.get(f"/files/{path}")
    assert response.status_code in (403, 404)


def test_missing_file_is_404(client):
    assert client.get("/files/p07/nope.png").status_code == 404


# -- health ---------------------------------------------------------------


def test_health_reports_backend_and_kinds(client):
    body = client.get("/health").json()
    assert body["backend"] == "stub"
    assert body["kinds_configured"] == ["image", "skybox"]
    assert "image_gen" in body["versions"]
