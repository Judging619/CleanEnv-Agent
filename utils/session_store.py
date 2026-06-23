import json
import uuid
from datetime import datetime
from pathlib import Path

from utils.path_tool import get_abs_path


SESSION_STORE_PATH = Path(get_abs_path("data/sessions.json"))


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _default_payload() -> dict:
    return {"sessions": []}


def _read_payload() -> dict:
    if not SESSION_STORE_PATH.exists():
        SESSION_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SESSION_STORE_PATH.write_text(json.dumps(_default_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        return _default_payload()

    text = SESSION_STORE_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return _default_payload()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _default_payload()
    payload.setdefault("sessions", [])
    return payload


def _write_payload(payload: dict):
    SESSION_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_STORE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def list_sessions() -> list[dict]:
    payload = _read_payload()
    sessions = payload.get("sessions", [])
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions


def create_session(title: str = "新会话") -> dict:
    payload = _read_payload()
    now = _utc_now()
    session = {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    payload["sessions"].append(session)
    _write_payload(payload)
    return session


def get_session(session_id: str) -> dict | None:
    for session in list_sessions():
        if session.get("id") == session_id:
            return session
    return None


def _normalize_title(messages: list[dict], fallback: str = "新会话") -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = str(msg.get("content", "")).strip()
            if content:
                return content[:20] + ("..." if len(content) > 20 else "")
    return fallback


def save_session_messages(session_id: str, messages: list[dict]):
    payload = _read_payload()
    now = _utc_now()
    for session in payload.get("sessions", []):
        if session.get("id") == session_id:
            session["messages"] = messages
            session["updated_at"] = now
            session["title"] = _normalize_title(messages, fallback=session.get("title", "新会话"))
            _write_payload(payload)
            return
    # 如果找不到，自动创建兜底
    session = create_session()
    save_session_messages(session["id"], messages)


def delete_session(session_id: str):
    payload = _read_payload()
    payload["sessions"] = [s for s in payload.get("sessions", []) if s.get("id") != session_id]
    _write_payload(payload)
