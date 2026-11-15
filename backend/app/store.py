"""
store.py

Where wearer profiles live. One small interface (get / save / list / delete)
so the storage backend can be swapped without touching the rest of the app.
Currently: one JSON file per profile under PROFILES_DIR.
"""

import json
import os
from typing import Optional

PROFILES_DIR = os.environ.get(
    "PROFILES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "profiles")
)


def safe_id(user_id: str) -> str:
    cleaned = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))[:40]
    return cleaned or "default"


def _path(user_id: str) -> str:
    os.makedirs(PROFILES_DIR, exist_ok=True)
    return os.path.join(PROFILES_DIR, f"{safe_id(user_id)}.json")


def get(user_id: str) -> Optional[dict]:
    path = _path(user_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save(profile: dict) -> None:
    path = _path(profile["user_id"])
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile, f)
    os.replace(tmp, path)          # atomic: never leaves a half-written profile


def delete(user_id: str) -> None:
    path = _path(user_id)
    if os.path.exists(path):
        os.remove(path)


def list_ids() -> list[str]:
    os.makedirs(PROFILES_DIR, exist_ok=True)
    return sorted(name[:-5] for name in os.listdir(PROFILES_DIR) if name.endswith(".json"))
