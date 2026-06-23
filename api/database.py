import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils.path_tool import get_abs_path


def _default_sqlite_path() -> str:
    file_path = Path(get_abs_path("data/zst_enterprise.db"))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return str(file_path)


def resolve_database_url() -> str:
    env_url = os.getenv("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    return f"sqlite:///{_default_sqlite_path()}"


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


DATABASE_URL = resolve_database_url()
connect_args = {"check_same_thread": False} if _is_sqlite(DATABASE_URL) else {}
engine = create_engine(DATABASE_URL, echo=False, future=True, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db():
    from api.models import Base

    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
