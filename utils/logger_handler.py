import logging
import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from time import perf_counter

from utils.path_tool import get_abs_path

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 确保日志的目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的格式配置  error info debug
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [request_id=%(request_id)s elapsed_ms=%(elapsed_ms)d] - %(filename)s:%(lineno)d - %(message)s'
)

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
_request_start_ctx: ContextVar[float | None] = ContextVar("request_start", default=None)


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        start = _request_start_ctx.get()
        if start is None:
            record.elapsed_ms = 0
        else:
            record.elapsed_ms = int((perf_counter() - start) * 1000)
        return True


def set_request_context(request_id: str | None = None):
    rid = request_id or str(uuid.uuid4())
    _request_id_ctx.set(rid)
    _request_start_ctx.set(perf_counter())
    return rid


def clear_request_context():
    _request_id_ctx.set("-")
    _request_start_ctx.set(None)


def get_request_id() -> str:
    return _request_id_ctx.get()


@contextmanager
def request_context(request_id: str | None = None):
    rid = set_request_context(request_id)
    try:
        yield rid
    finally:
        clear_request_context()


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加Handler
    if logger.handlers:
        return logger

    request_filter = RequestContextFilter()

    # 控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    console_handler.addFilter(request_filter)

    logger.addHandler(console_handler)

    # 文件Handler
    if not log_file:        # 日志文件的存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    file_handler.addFilter(request_filter)

    logger.addHandler(file_handler)

    return logger


# 快捷获取日志器
logger = get_logger()


if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
