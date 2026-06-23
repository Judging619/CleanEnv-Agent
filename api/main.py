import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Generator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import clear_last_rag_references, get_last_rag_references, reload_rag_service
from api.database import SessionLocal, get_db, init_db
from api.rate_limit import RateLimitMiddleware
from api.schemas import ChatJobOut, ChatJobRequest, ChatStreamRequest, MessageOut, OperationOut, SessionCreateRequest, SessionDetailOut, SessionOut
from api.services import (
    ChatJobNotFoundError,
    SessionNotFoundError,
    append_message,
    build_chat_history,
    clear_messages,
    complete_chat_job,
    create_chat_job,
    create_session,
    delete_session,
    fail_chat_job,
    get_chat_job,
    get_session,
    list_active_chat_jobs,
    list_messages,
    list_sessions,
    mark_chat_job_running,
    serialize_chat_job,
    serialize_message,
    serialize_session,
)
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger, request_context


class AgentRuntime:
    def __init__(self):
        self._agent = None
        self._lock = threading.Lock()

    def get_agent(self) -> ReactAgent:
        with self._lock:
            if self._agent is None:
                self._agent = ReactAgent()
            return self._agent

    def reload(self):
        with self._lock:
            self._agent = ReactAgent()


agent_runtime = AgentRuntime()


class ChatJobRunner:
    def __init__(self):
        self._max_workers = int(os.getenv("CHAT_JOB_MAX_WORKERS", "4"))
        self._lock = threading.Lock()
        self._shutdown = False
        self._executor = self._new_executor()

    def _new_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="chat-job")

    def submit(self, job_id: str):
        with self._lock:
            if self._shutdown:
                self._executor = self._new_executor()
                self._shutdown = False
            self._executor.submit(self._run_job, job_id)

    def shutdown(self):
        with self._lock:
            self._executor.shutdown(wait=False, cancel_futures=False)
            self._shutdown = True

    def _run_job(self, job_id: str):
        db = SessionLocal()
        request_id = "-"
        try:
            job = mark_chat_job_running(db, job_id)
            history = build_chat_history(db, job.session_id)

            with request_context() as request_id:
                logger.info(f"[api.chat_job] start job_id={job_id} request_id={request_id}")
                clear_last_rag_references()
                response_chunks: list[str] = []

                agent = ReactAgent()
                for chunk in agent.execute_stream(history):
                    text = str(chunk or "")
                    if text:
                        response_chunks.append(text)

                full_response = "".join(response_chunks).strip()
                if not full_response:
                    full_response = "抱歉，我暂时没有检索到有效信息，请换一种问法试试。"

                references = get_last_rag_references()
                complete_chat_job(db, job_id, full_response, references=references)
                logger.info(f"[api.chat_job] done job_id={job_id} request_id={request_id}")
        except Exception as e:
            logger.error(f"[api.chat_job] failed job_id={job_id} request_id={request_id}: {str(e)}", exc_info=True)
            try:
                fail_chat_job(db, job_id, "抱歉，我暂时无法完成这次请求。请稍后重试，或换一种问法。")
            except Exception:
                logger.error(f"[api.chat_job] mark failed error job_id={job_id}", exc_info=True)
        finally:
            db.close()


chat_job_runner = ChatJobRunner()


def _sse_pack(event: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        init_db()
        logger.info("[api] startup complete")
        try:
            yield
        finally:
            chat_job_runner.shutdown()
            logger.info("[api] shutdown complete")

    app = FastAPI(title="洁境智顾 Agent API", version="1.0.0", lifespan=lifespan)

    allow_origins = os.getenv("API_CORS_ALLOW_ORIGINS", "*").strip()
    origins = [x.strip() for x in allow_origins.split(",") if x.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rate_limit_per_minute = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60"))
    app.add_middleware(RateLimitMiddleware, limit_per_minute=rate_limit_per_minute)

    @app.get("/api/v1/health")
    def health():
        return {"ok": True, "service": "cleanenv-agent-api"}

    @app.get("/")
    def root():
        return {
            "ok": True,
            "service": "cleanenv-agent-api",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/api/v1/sessions", response_model=list[SessionOut])
    def api_list_sessions(db: Session = Depends(get_db)):
        sessions = list_sessions(db)
        return [serialize_session(s) for s in sessions]

    @app.post("/api/v1/sessions", response_model=SessionOut)
    def api_create_session(payload: SessionCreateRequest, db: Session = Depends(get_db)):
        session = create_session(db, title=payload.title)
        return serialize_session(session)

    @app.get("/api/v1/sessions/{session_id}", response_model=SessionDetailOut)
    def api_get_session_detail(session_id: str, db: Session = Depends(get_db)):
        try:
            session = get_session(db, session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="会话不存在")

        messages = list_messages(db, session_id)
        return {
            "session": serialize_session(session),
            "messages": [serialize_message(m) for m in messages],
        }

    @app.get("/api/v1/sessions/{session_id}/messages", response_model=list[MessageOut])
    def api_list_session_messages(session_id: str, db: Session = Depends(get_db)):
        try:
            messages = list_messages(db, session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="会话不存在")
        return [serialize_message(m) for m in messages]

    @app.post("/api/v1/chat/jobs", response_model=ChatJobOut)
    def api_create_chat_job(payload: ChatJobRequest, db: Session = Depends(get_db)):
        try:
            job = create_chat_job(db, payload.session_id, payload.message)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="会话不存在")

        chat_job_runner.submit(job.id)
        return serialize_chat_job(job)

    @app.get("/api/v1/chat/jobs/active", response_model=list[ChatJobOut])
    def api_list_active_chat_jobs(db: Session = Depends(get_db)):
        jobs = list_active_chat_jobs(db)
        return [serialize_chat_job(job) for job in jobs]

    @app.get("/api/v1/chat/jobs/{job_id}", response_model=ChatJobOut)
    def api_get_chat_job(job_id: str, db: Session = Depends(get_db)):
        try:
            job = get_chat_job(db, job_id)
        except ChatJobNotFoundError:
            raise HTTPException(status_code=404, detail="任务不存在")
        return serialize_chat_job(job)

    @app.delete("/api/v1/sessions/{session_id}", response_model=OperationOut)
    def api_delete_session(session_id: str, db: Session = Depends(get_db)):
        try:
            delete_session(db, session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"ok": True, "message": "会话已删除"}

    @app.delete("/api/v1/sessions/{session_id}/messages", response_model=OperationOut)
    def api_clear_session_messages(session_id: str, db: Session = Depends(get_db)):
        try:
            clear_messages(db, session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"ok": True, "message": "会话消息已清空"}

    @app.post("/api/v1/chat/stream")
    def api_chat_stream(payload: ChatStreamRequest, db: Session = Depends(get_db)):
        try:
            get_session(db, payload.session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="会话不存在")

        append_message(db, payload.session_id, role="user", content=payload.message, references=[])
        history = build_chat_history(db, payload.session_id)

        def event_generator() -> Generator[str, None, None]:
            with request_context() as request_id:
                logger.info(f"[api.chat] start request_id={request_id}")
                clear_last_rag_references()
                response_chunks: list[str] = []

                yield _sse_pack("meta", {"request_id": request_id, "session_id": payload.session_id})

                try:
                    agent = agent_runtime.get_agent()
                    for chunk in agent.execute_stream(history):
                        text = str(chunk or "")
                        if not text:
                            continue
                        response_chunks.append(text)
                        yield _sse_pack("chunk", {"text": text})

                    full_response = "".join(response_chunks).strip()
                    if not full_response:
                        full_response = "抱歉，我暂时没有检索到有效信息，请换一种问法试试。"

                    references = get_last_rag_references()
                    assistant = append_message(
                        db,
                        payload.session_id,
                        role="assistant",
                        content=full_response,
                        references=references,
                    )
                    done_payload = {
                        "session_id": payload.session_id,
                        "assistant_message": serialize_message(assistant),
                    }
                    yield _sse_pack("done", done_payload)
                    logger.info(f"[api.chat] done request_id={request_id}")
                except Exception as e:
                    logger.error(f"[api.chat] failed request_id={request_id}: {str(e)}", exc_info=True)
                    fallback = "抱歉，我暂时无法完成这次请求。请稍后重试，或换一种问法。"
                    assistant = append_message(
                        db,
                        payload.session_id,
                        role="assistant",
                        content=fallback,
                        references=[],
                    )
                    yield _sse_pack("error", {"message": fallback})
                    yield _sse_pack(
                        "done",
                        {
                            "session_id": payload.session_id,
                            "assistant_message": serialize_message(assistant),
                        },
                    )

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/api/v1/admin/knowledge/rebuild", response_model=OperationOut)
    def api_rebuild_knowledge_base():
        try:
            vs = VectorStoreService()
            vs.rebuild_document()
            reload_rag_service()
            agent_runtime.reload()
            return {"ok": True, "message": "知识库重建完成"}
        except Exception as e:
            logger.error(f"[api.admin] rebuild failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="知识库重建失败")

    @app.post("/api/v1/admin/agent/reload", response_model=OperationOut)
    def api_reload_agent():
        try:
            agent_runtime.reload()
            return {"ok": True, "message": "Agent已重载"}
        except Exception as e:
            logger.error(f"[api.admin] agent reload failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Agent重载失败")

    return app


app = create_app()
