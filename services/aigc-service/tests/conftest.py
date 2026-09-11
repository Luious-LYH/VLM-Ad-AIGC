import os

os.environ.setdefault("AIGC_BACKEND", "fake")
os.environ.setdefault("AIGC_URL_ALLOWLIST", "assets.example.test,localhost,127.0.0.1")
os.environ.setdefault("AIGC_ASSET_ROOTS", "/data/assets,/data/uploads")
# Keep queued observable across platforms: TestClient may yield between submit
# and its first poll, especially now that real inference is offloaded to a
# worker thread.
os.environ.setdefault("AIGC_FAKE_JOB_START_DELAY_MS", "80")
os.environ.setdefault("AIGC_FAKE_JOB_RUN_DELAY_MS", "40")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
