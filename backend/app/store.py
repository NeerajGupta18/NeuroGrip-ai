"""
store.py

Where wearer profiles live. One small interface (get / save / delete /
list_ids) with two backends:

  * Postgres, when DATABASE_URL is set. One row per profile, the profile
    itself stored as JSONB. This is what a hosted deployment should use,
    because free hosting wipes the local disk whenever the service restarts.
  * JSON files under PROFILES_DIR otherwise. Zero setup for local runs.

The rest of the app never knows which one is in use.
"""

import atexit
import json
import os
from typing import Optional

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
PROFILES_DIR = os.environ.get(
    "PROFILES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "profiles")
)
MAX_PROFILES = int(os.environ.get("MAX_PROFILES", "500"))   # the public demo is open; cap its growth


def safe_id(user_id: str) -> str:
    cleaned = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))[:40]
    return cleaned or "default"


# ---------------------------------------------------------------- file backend

def _path(user_id: str) -> str:
    os.makedirs(PROFILES_DIR, exist_ok=True)
    return os.path.join(PROFILES_DIR, f"{safe_id(user_id)}.json")


def _file_get(user_id: str) -> Optional[dict]:
    path = _path(user_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _file_save(profile: dict) -> None:
    path = _path(profile["user_id"])
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile, f)
    os.replace(tmp, path)          # atomic: never leaves a half-written profile


def _file_delete(user_id: str) -> None:
    path = _path(user_id)
    if os.path.exists(path):
        os.remove(path)


def _file_list() -> list[str]:
    """Profile ids, oldest first."""
    os.makedirs(PROFILES_DIR, exist_ok=True)
    names = [n for n in os.listdir(PROFILES_DIR) if n.endswith(".json")]
    names.sort(key=lambda n: os.path.getmtime(os.path.join(PROFILES_DIR, n)))
    return [n[:-5] for n in names]


# ------------------------------------------------------------ Postgres backend

_pool = None


def _db():
    """Lazily open a small connection pool and make sure the table exists."""
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool   # imported only when a database is configured

        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=4, open=True)
        atexit.register(_pool.close)
        with _pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id    TEXT PRIMARY KEY,
                    data       JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
    return _pool


def _pg_get(user_id: str) -> Optional[dict]:
    with _db().connection() as conn:
        row = conn.execute("SELECT data FROM profiles WHERE user_id = %s", (safe_id(user_id),)).fetchone()
    return row[0] if row else None


def _pg_save(profile: dict) -> None:
    with _db().connection() as conn:
        conn.execute(
            """
            INSERT INTO profiles (user_id, data) VALUES (%s, %s::jsonb)
            ON CONFLICT (user_id) DO UPDATE SET data = EXCLUDED.data, updated_at = now()
            """,
            (safe_id(profile["user_id"]), json.dumps(profile)),
        )


def _pg_delete(user_id: str) -> None:
    with _db().connection() as conn:
        conn.execute("DELETE FROM profiles WHERE user_id = %s", (safe_id(user_id),))


def _pg_list() -> list[str]:
    with _db().connection() as conn:
        return [r[0] for r in conn.execute("SELECT user_id FROM profiles ORDER BY created_at").fetchall()]


# ------------------------------------------------------------------ public API

def backend() -> str:
    return "postgres" if DATABASE_URL else "files"


def get(user_id: str) -> Optional[dict]:
    return _pg_get(user_id) if DATABASE_URL else _file_get(user_id)


def save(profile: dict) -> None:
    _pg_save(profile) if DATABASE_URL else _file_save(profile)


def delete(user_id: str) -> None:
    _pg_delete(user_id) if DATABASE_URL else _file_delete(user_id)


def list_ids() -> list[str]:
    """Profile ids, oldest first."""
    return _pg_list() if DATABASE_URL else _file_list()


def make_room() -> None:
    """Before a new profile is created: drop the oldest ones beyond the cap."""
    ids = list_ids()
    for old in ids[: max(0, len(ids) - MAX_PROFILES + 1)]:
        delete(old)
