import json
from dataclasses import dataclass
from typing import Generator

import requests


class BackendClientError(RuntimeError):
    pass


@dataclass
class StreamState:
    done_payload: dict | None = None
    error_message: str | None = None


class BackendClient:
    def __init__(self, base_url: str, timeout_seconds: int = 30):
        cleaned = (base_url or "").strip().rstrip("/")
        if not cleaned:
            raise BackendClientError("后端地址为空")
        self.base_url = cleaned
        self.timeout_seconds = timeout_seconds

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, json_body: dict | None = None) -> requests.Response:
        try:
            resp = requests.request(
                method=method,
                url=self._url(path),
                json=json_body,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as e:
            raise BackendClientError(f"请求后端失败：{str(e)}") from e

        if resp.status_code >= 400:
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = resp.text
            raise BackendClientError(f"后端错误({resp.status_code})：{detail}")

        return resp

    def health(self) -> dict:
        try:
            resp = requests.get(self._url("/api/v1/health"), timeout=2)
        except requests.RequestException as e:
            raise BackendClientError(f"后端健康检查失败：{str(e)}") from e

        if resp.status_code >= 400:
            raise BackendClientError(f"后端健康检查失败({resp.status_code})：{resp.text}")

        return resp.json()

    def list_sessions(self) -> list[dict]:
        resp = self._request("GET", "/api/v1/sessions")
        return resp.json()

    def create_session(self, title: str = "新会话") -> dict:
        resp = self._request("POST", "/api/v1/sessions", {"title": title})
        return resp.json()

    def get_session_detail(self, session_id: str) -> dict:
        resp = self._request("GET", f"/api/v1/sessions/{session_id}")
        return resp.json()

    def delete_session(self, session_id: str) -> dict:
        resp = self._request("DELETE", f"/api/v1/sessions/{session_id}")
        return resp.json()

    def clear_session_messages(self, session_id: str) -> dict:
        resp = self._request("DELETE", f"/api/v1/sessions/{session_id}/messages")
        return resp.json()

    def create_chat_job(self, session_id: str, message: str) -> dict:
        resp = self._request("POST", "/api/v1/chat/jobs", {"session_id": session_id, "message": message})
        return resp.json()

    def get_chat_job(self, job_id: str) -> dict:
        resp = self._request("GET", f"/api/v1/chat/jobs/{job_id}")
        return resp.json()

    def list_active_chat_jobs(self) -> list[dict]:
        resp = self._request("GET", "/api/v1/chat/jobs/active")
        return resp.json()

    def rebuild_knowledge_base(self) -> dict:
        resp = self._request("POST", "/api/v1/admin/knowledge/rebuild")
        return resp.json()

    def reload_agent(self) -> dict:
        resp = self._request("POST", "/api/v1/admin/agent/reload")
        return resp.json()

    def stream_chat(self, session_id: str, message: str, state: StreamState | None = None) -> Generator[str, None, None]:
        state = state or StreamState()
        url = self._url("/api/v1/chat/stream")
        payload = {"session_id": session_id, "message": message}

        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=(self.timeout_seconds, 600),
                stream=True,
                headers={"Accept": "text/event-stream"},
            )
        except requests.RequestException as e:
            raise BackendClientError(f"请求后端流失败：{str(e)}") from e

        if resp.status_code >= 400:
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = resp.text
            resp.close()
            raise BackendClientError(f"后端流错误({resp.status_code})：{detail}")

        def parse_sse_lines() -> Generator[tuple[str, str], None, None]:
            event_name = "message"
            data_lines: list[str] = []

            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()

                if not line:
                    if data_lines:
                        yield event_name, "\n".join(data_lines)
                    event_name = "message"
                    data_lines = []
                    continue

                if line.startswith("event:"):
                    event_name = line[6:].strip() or "message"
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())

            if data_lines:
                yield event_name, "\n".join(data_lines)

        try:
            for event_name, data_str in parse_sse_lines():
                try:
                    payload_obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if event_name == "chunk":
                    text = str(payload_obj.get("text", ""))
                    if text:
                        yield text
                elif event_name == "error":
                    state.error_message = str(payload_obj.get("message", ""))
                elif event_name == "done":
                    state.done_payload = payload_obj
        finally:
            resp.close()
