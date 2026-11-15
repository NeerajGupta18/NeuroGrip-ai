import os
import tempfile

# Point profile storage at a throwaway directory before the app is imported.
os.environ["PROFILES_DIR"] = tempfile.mkdtemp(prefix="neurogrip-test-")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


def fetch(client, user_id, **extra):
    return client.post("/api/signal/next", json={"user_id": user_id, **extra}).json()


def signal_of(recording):
    return {"ch1": recording["ch1"], "ch2": recording["ch2"]}
