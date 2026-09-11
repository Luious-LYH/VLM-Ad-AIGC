import time


def wait_terminal(client, job_id):
    seen = []
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        seen.append(job["status"])
        if job["status"] in {"succeeded", "failed"}:
            return job, seen
        time.sleep(0.01)
    raise AssertionError(f"job did not finish; states={seen}")


def test_image_job_idempotency_and_transitions(client):
    payload = {"prompt": "premium bottle studio shot", "num_candidates": 3, "seed": 7}
    first = client.post("/v1/image/generations", json=payload, headers={"Idempotency-Key": "img-7"})
    assert first.status_code == 202
    assert first.json()["status"] == "queued"
    repeated = client.post("/v1/image/generations", json=payload, headers={"Idempotency-Key": "img-7"})
    assert repeated.status_code == 202
    assert repeated.json()["id"] == first.json()["id"]
    assert repeated.json()["reused"] is True

    conflict = client.post("/v1/image/generations", json={**payload, "seed": 8}, headers={"Idempotency-Key": "img-7"})
    assert conflict.status_code == 409

    job, seen = wait_terminal(client, first.json()["id"])
    assert job["status"] == "succeeded"
    assert "queued" in seen and "running" in seen and seen[-1] == "succeeded"
    assert len(job["result"]["artifacts"]) == 3
    assert job["run"]["seed"] == 7


def test_video_job_and_failure_lifecycle(client):
    success = client.post("/v1/video/generations", json={
        "prompt": "slow product turntable", "image_url": "https://assets.example.test/keyframe.png"
    })
    assert success.status_code == 202
    job, _ = wait_terminal(client, success.json()["id"])
    assert job["status"] == "succeeded"
    assert job["result"]["artifacts"][0]["url"].endswith(".mp4")

    failure = client.post("/v1/video/generations", json={"prompt": "__fake_fail__"})
    failed, states = wait_terminal(client, failure.json()["id"])
    assert failed["status"] == "failed"
    assert "running" in states
    assert "deterministic fake generation failure" in failed["error"]
    assert client.get("/v1/jobs/does-not-exist").status_code == 404

