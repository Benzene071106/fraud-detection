"""Database models: flagged events, incidents, analyst feedback. SQLite by default; set DATABASE_URL for PostgreSQL."""
import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, inspect, text, String, Integer, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

ROOT = Path(__file__).resolve().parents[1]
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{(ROOT / 'data' / 'soc.db').as_posix()}")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(primary_key=True)
    family: Mapped[str] = mapped_column(String(40), index=True)
    dst_ip: Mapped[str] = mapped_column(String(64), index=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime)
    last_seen: Mapped[datetime] = mapped_column(DateTime, index=True)
    event_count: Mapped[int] = mapped_column(Integer, default=0)
    alert_count: Mapped[int] = mapped_column(Integer, default=0)
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    max_risk: Mapped[int] = mapped_column(Integer, default=0)
    severity: Mapped[str] = mapped_column(String(10), default="LOW", index=True)
    status: Mapped[str] = mapped_column(String(12), default="OPEN", index=True)
    analyst_verdict: Mapped[str | None] = mapped_column(String(4), nullable=True)
    analyst_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    techniques: Mapped[str | None] = mapped_column(Text, nullable=True)   # JSON {technique_id: {...counts}}
    notified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)   # Slack/email sent
    events: Mapped[list["Event"]] = relationship(back_populates="incident")


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    src_ip: Mapped[str] = mapped_column(String(64))
    dst_ip: Mapped[str] = mapped_column(String(64))
    dst_port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verdict: Mapped[str] = mapped_column(String(8))
    family: Mapped[str] = mapped_column(String(40))
    risk: Mapped[int] = mapped_column(Integer)
    xgb_prob: Mapped[float] = mapped_column(Float)
    rule_match: Mapped[bool] = mapped_column(Boolean)
    if_pct: Mapped[float] = mapped_column(Float)
    reasons: Mapped[str] = mapped_column(Text)        # JSON list of SHAP reasons
    features: Mapped[str] = mapped_column(Text)       # JSON features, kept for feedback-driven retraining
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"), index=True)
    incident: Mapped[Incident] = relationship(back_populates="events")


# lightweight migrations for databases created by older versions of the API
NEW_COLUMNS = {"techniques": "TEXT", "notified_at": "TIMESTAMP"}


def init_db():
    Base.metadata.create_all(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("incidents")}
    for name, sql_type in NEW_COLUMNS.items():
        if name not in cols:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE incidents ADD COLUMN {name} {sql_type}"))
