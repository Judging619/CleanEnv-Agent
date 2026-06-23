import os
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from utils.logger_handler import logger


class _MemoryWindowLimiter:
    def __init__(self, limit_per_minute: int):
        self.limit = max(limit_per_minute, 1)
        self.window_seconds = 60
        self._store: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            queue = self._store[key]
            while queue and now - queue[0] >= self.window_seconds:
                queue.popleft()

            if len(queue) >= self.limit:
                retry_after = int(max(1, self.window_seconds - (now - queue[0])))
                return False, retry_after

            queue.append(now)
            return True, 0


class _RedisWindowLimiter:
    def __init__(self, redis_url: str, limit_per_minute: int):
        import redis

        self.limit = max(limit_per_minute, 1)
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.window_seconds = 60

    def allow(self, key: str) -> tuple[bool, int]:
        window = int(time.time() // self.window_seconds)
        redis_key = f"zst:ratelimit:{window}:{key}"
        count = int(self.redis.incr(redis_key))
        if count == 1:
            self.redis.expire(redis_key, self.window_seconds + 1)

        if count > self.limit:
            ttl = int(self.redis.ttl(redis_key))
            retry_after = max(ttl, 1)
            return False, retry_after

        return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_minute: int = 60):
        super().__init__(app)
        self.limit_per_minute = max(limit_per_minute, 1)
        self._backend_switch_lock = Lock()
        self.enabled = os.getenv("API_ENABLE_RATE_LIMIT", "1").strip().lower() not in {"0", "false", "off"}
        self.excluded_paths = {
            "/api/v1/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        }

        redis_url = os.getenv("REDIS_URL", "").strip()
        self.backend = None

        if self.enabled and redis_url:
            try:
                self.backend = _RedisWindowLimiter(redis_url=redis_url, limit_per_minute=self.limit_per_minute)
            except Exception:
                self.backend = None

        if self.enabled and self.backend is None:
            self.backend = _MemoryWindowLimiter(limit_per_minute=self.limit_per_minute)

    def _switch_to_memory_backend(self, reason: str):
        with self._backend_switch_lock:
            if isinstance(self.backend, _MemoryWindowLimiter):
                return
            self.backend = _MemoryWindowLimiter(limit_per_minute=self.limit_per_minute)
            logger.warning(f"[rate_limit]Redis限流不可用，已降级为内存限流：{reason}")

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "").strip()
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    async def dispatch(self, request: Request, call_next):
        if not self.enabled or self.backend is None or request.url.path in self.excluded_paths:
            return await call_next(request)

        key = f"{self._client_ip(request)}:{request.method}:{request.url.path}"
        try:
            allowed, retry_after = self.backend.allow(key)
        except Exception as e:
            self._switch_to_memory_backend(str(e))
            try:
                allowed, retry_after = self.backend.allow(key)
            except Exception as inner_e:
                logger.error(f"[rate_limit]内存限流兜底失败，放行请求：{str(inner_e)}", exc_info=True)
                return await call_next(request)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后重试。"},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
