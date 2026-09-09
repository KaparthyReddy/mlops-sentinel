"""
SQLAlchemy engine/session setup. Kept intentionally simple (sync engine,
not async) since prediction logging and drift queries here don't need
high-concurrency async DB access — the actual prediction serving path is
async at the FastAPI layer, but DB writes are fire-and-forget logging,
not something the client waits heavily on.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://mlops_user:change_me_locally@localhost:5433/mlops_sentinel"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
