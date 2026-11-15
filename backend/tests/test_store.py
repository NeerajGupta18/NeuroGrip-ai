"""Profile storage: the file backend always, the Postgres backend when TEST_DATABASE_URL is set."""

import os

import pytest

from app import store


def _roundtrip(s):
    s.save({"user_id": "alpha", "source": "wearer_a", "n": 1})
    s.save({"user_id": "alpha", "source": "wearer_a", "n": 2})      # overwrite, not duplicate
    s.save({"user_id": "beta", "source": "wearer_b", "n": 1})
    assert s.get("alpha")["n"] == 2
    assert s.get("missing") is None
    assert s.list_ids().count("alpha") == 1 and "beta" in s.list_ids()
    s.delete("alpha")
    assert s.get("alpha") is None and "beta" in s.list_ids()
    s.delete("beta")


def test_ids_are_sanitised():
    assert store.safe_id("../../etc/passwd") == "etcpasswd"
    assert store.safe_id("x" * 100) == "x" * 40
    assert store.safe_id("!!!") == "default"


def test_file_backend_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATABASE_URL", "")
    monkeypatch.setattr(store, "PROFILES_DIR", str(tmp_path))
    assert store.backend() == "files"
    _roundtrip(store)


def test_oldest_profiles_are_dropped_at_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATABASE_URL", "")
    monkeypatch.setattr(store, "PROFILES_DIR", str(tmp_path))
    monkeypatch.setattr(store, "MAX_PROFILES", 3)
    for i in range(3):
        store.save({"user_id": f"p{i}"})
        os.utime(tmp_path / f"p{i}.json", (i, i))
    store.make_room()                       # room for one more -> at most 2 remain
    assert len(store.list_ids()) == 2


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_postgres_backend_roundtrip(monkeypatch):
    monkeypatch.setattr(store, "DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setattr(store, "_pool", None)
    assert store.backend() == "postgres"
    _roundtrip(store)
