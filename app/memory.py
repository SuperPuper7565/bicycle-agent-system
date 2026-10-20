"""Долговременная память: профиль клиента в SQLite (между разными диалогами).

Краткосрочная память (история текущего диалога) хранится чекпоинтером LangGraph
по thread_id. Здесь хранится то, что должно пережить новый диалог.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from . import config


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.STORE_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS profiles (user_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated TEXT NOT NULL)"
    )
    return conn


def load_profile(user_id: str, db_path: Path | None = None) -> dict:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT data FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    return json.loads(row[0]) if row else {}


def update_profile(user_id: str, req: dict, db_path: Path | None = None) -> dict:
    """Запомнить предпочтения после успешно подготовленного КП."""
    profile = load_profile(user_id, db_path)
    profile["offers"] = int(profile.get("offers", 0)) + 1
    for key, field in (("last_category", "category"), ("last_max_price", "max_price"),
                       ("last_wheel_size", "wheel_size"), ("last_quantity", "quantity")):
        if req.get(field):
            profile[key] = req[field]
    wishes = list(profile.get("wishes", []))
    for item in (req.get("wishes") or []) + (req.get("must_have") or []):
        if item not in wishes:
            wishes.append(item)
    profile["wishes"] = wishes[-10:]
    profile["updated"] = datetime.now().isoformat(timespec="seconds")
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO profiles (user_id, data, updated) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET data = excluded.data, updated = excluded.updated",
                (user_id, json.dumps(profile, ensure_ascii=False), profile["updated"]),
            )
    finally:
        conn.close()
    return profile


def profile_to_text(profile: dict) -> str:
    if not profile:
        return "нет данных (новый клиент)"
    parts = [f"подготовлено предложений: {profile.get('offers', 0)}"]
    if profile.get("last_category"):
        parts.append(f"последний тип: {profile['last_category']}")
    if profile.get("last_max_price"):
        parts.append(f"последний бюджет: {profile['last_max_price']} руб.")
    if profile.get("last_wheel_size"):
        parts.append(f"колёса: {profile['last_wheel_size']}")
    if profile.get("last_quantity"):
        parts.append(f"количество: {profile['last_quantity']}")
    if profile.get("wishes"):
        parts.append("прошлые пожелания: " + ", ".join(profile["wishes"]))
    return "; ".join(parts)
