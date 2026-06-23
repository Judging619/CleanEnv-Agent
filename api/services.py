import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models import ChatJob, ChatMessage, ChatSession


class SessionNotFoundError(ValueError):
    pass


class ChatJobNotFoundError(ValueError):
    pass


def _to_iso(dt: datetime | None) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _trim_title(text: str, max_len: int = 20) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return "新会话"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len] + "..."


def _parse_references(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        return []
    return []


def serialize_session(session: ChatSession) -> dict:
    return {
        "id": session.id,
        "title": session.title,
        "created_at": _to_iso(session.created_at),
        "updated_at": _to_iso(session.updated_at),
    }


def serialize_message(message: ChatMessage) -> dict:
    return {
        "id": message.id,
        "session_id": message.session_id,
        "role": message.role,
        "content": message.content,
        "references": _parse_references(message.references_json),
        "created_at": _to_iso(message.created_at),
    }


def serialize_chat_job(job: ChatJob) -> dict:
    return {
        "id": job.id,
        "session_id": job.session_id,
        "status": job.status,
        "prompt": job.prompt,
        "response_content": job.response_content or "",
        "error_message": job.error_message or "",
        "references": _parse_references(job.references_json),
        "user_message_id": job.user_message_id,
        "assistant_message_id": job.assistant_message_id,
        "created_at": _to_iso(job.created_at),
        "updated_at": _to_iso(job.updated_at),
        "started_at": _to_iso(job.started_at),
        "completed_at": _to_iso(job.completed_at),
    }


def list_sessions(db: Session, limit: int = 200) -> list[ChatSession]:
    stmt = select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)
    return list(db.execute(stmt).scalars().all())


def create_session(db: Session, title: str = "新会话") -> ChatSession:
    session = ChatSession(title=(title or "新会话").strip() or "新会话")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, session_id: str) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if not session:
        raise SessionNotFoundError(f"session not found: {session_id}")
    return session


def delete_session(db: Session, session_id: str):
    session = get_session(db, session_id)
    db.delete(session)
    db.commit()


def list_messages(db: Session, session_id: str) -> list[ChatMessage]:
    get_session(db, session_id)
    stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id.asc())
    return list(db.execute(stmt).scalars().all())


def clear_messages(db: Session, session_id: str):
    session = get_session(db, session_id)
    for msg in list(session.messages):
        db.delete(msg)
    session.updated_at = datetime.now(timezone.utc)
    db.commit()


def append_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    references: list[dict] | None = None,
) -> ChatMessage:
    session = get_session(db, session_id)
    cleaned_content = (content or "").strip()
    if not cleaned_content:
        cleaned_content = ""

    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=cleaned_content,
        references_json=json.dumps(references or [], ensure_ascii=False),
    )
    db.add(message)

    if role == "user" and (not session.title or session.title == "新会话"):
        session.title = _trim_title(cleaned_content)

    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(message)
    return message


def build_chat_history(db: Session, session_id: str) -> list[dict]:
    messages = list_messages(db, session_id)
    history = []
    for msg in messages:
        if msg.role in {"user", "assistant"} and msg.content:
            history.append({"role": msg.role, "content": msg.content})
    return history


def create_chat_job(db: Session, session_id: str, prompt: str) -> ChatJob:
    user_message = append_message(db, session_id, role="user", content=prompt, references=[])
    job = ChatJob(
        session_id=session_id,
        status="queued",
        prompt=(prompt or "").strip(),
        user_message_id=user_message.id,
        references_json=json.dumps([], ensure_ascii=False),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_chat_job(db: Session, job_id: str) -> ChatJob:
    job = db.get(ChatJob, job_id)
    if not job:
        raise ChatJobNotFoundError(f"chat job not found: {job_id}")
    return job


def list_active_chat_jobs(db: Session, limit: int = 200) -> list[ChatJob]:
    stmt = (
        select(ChatJob)
        .where(ChatJob.status.in_(["queued", "running"]))
        .order_by(ChatJob.created_at.desc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


def mark_chat_job_running(db: Session, job_id: str) -> ChatJob:
    job = get_chat_job(db, job_id)
    now = datetime.now(timezone.utc)
    job.status = "running"
    job.started_at = now
    job.updated_at = now
    db.commit()
    db.refresh(job)
    return job


def complete_chat_job(db: Session, job_id: str, content: str, references: list[dict] | None = None) -> ChatJob:
    job = get_chat_job(db, job_id)
    assistant = append_message(
        db,
        job.session_id,
        role="assistant",
        content=content,
        references=references or [],
    )
    now = datetime.now(timezone.utc)
    job.status = "completed"
    job.response_content = content or ""
    job.references_json = json.dumps(references or [], ensure_ascii=False)
    job.assistant_message_id = assistant.id
    job.updated_at = now
    job.completed_at = now
    db.commit()
    db.refresh(job)
    return job


def fail_chat_job(db: Session, job_id: str, message: str) -> ChatJob:
    job = get_chat_job(db, job_id)
    fallback = message or "抱歉，我暂时无法完成这次请求。请稍后重试，或换一种问法。"
    assistant = append_message(
        db,
        job.session_id,
        role="assistant",
        content=fallback,
        references=[],
    )
    now = datetime.now(timezone.utc)
    job.status = "failed"
    job.response_content = fallback
    job.error_message = fallback
    job.references_json = json.dumps([], ensure_ascii=False)
    job.assistant_message_id = assistant.id
    job.updated_at = now
    job.completed_at = now
    db.commit()
    db.refresh(job)
    return job
