from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from typing import Any, Dict, List, Optional, Protocol, Tuple

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, joinedload, mapped_column, relationship
from sqlalchemy.types import JSON

UTC = dt.timezone.utc

BASE_INTERVAL_SECS = 60  # 1 minute starting interval


class RandomizedMultiplierScheduler:
    """Per-card randomized scheduler: base interval and multiplier are sampled
    once per card and stored in the state dict."""
    name = "randomized_multiplier"
    version = 1

    def init_state(self, item: Item) -> dict:
        import random
        base = random.uniform(60, 180)       # 1-3 min
        mult = random.uniform(2.0, 5.0)      # 2x-5x
        return {"interval_secs": base, "base_interval_secs": base, "multiplier": mult}

    def update(self, *, item: Item, prev_state: dict, event: ReviewEvent) -> ScheduleUpdate:
        now = event.reviewed_at
        state = prev_state or {}
        base = state.get("base_interval_secs", BASE_INTERVAL_SECS)
        mult = state.get("multiplier", 3.0)
        interval = state.get("interval_secs", base)

        if event.correct:
            due_at = now + dt.timedelta(seconds=interval)
            new_interval = interval * mult
        else:
            due_at = now + dt.timedelta(seconds=base)
            new_interval = base

        return ScheduleUpdate(
            due_at=due_at,
            state={"interval_secs": new_interval, "base_interval_secs": base, "multiplier": mult},
            scheduler_name=self.name,
            scheduler_version=self.version,
        )

SCHEDULER = RandomizedMultiplierScheduler()


def now_utc() -> dt.datetime:
    return dt.datetime.now(UTC)


class Base(DeclarativeBase):
    pass


from collections import defaultdict
from typing import Dict, List, Optional, TypedDict


# ============================================================
# Lexeme tracking
# ============================================================

class LexemeGroup(TypedDict):
    lexeme: str
    items: List[Item]


def parse_external_id(external_id: str) -> Optional[Tuple[str, str]]:
    """
    Expected format:
        lexeme:<WORD>:<skill_type>

    Returns (lexeme, skill_type) or None.

    Note: <WORD> is the *lexeme key*, which for a sense-split homograph carries
    a sense discriminator (e.g. "분기#quarter"). Use headword_of() to get the
    bare Korean surface for display / LLM lookups.
    """
    if not external_id:
        return None
    parts = external_id.split(":")
    if len(parts) < 3:
        return None
    if parts[0] != "lexeme":
        return None
    return parts[1], parts[2]


# Sense discriminator: homographs that are genuinely different words (different
# meaning, usually different hanja) are split into separate lexeme keys that
# share a surface spelling — "분기#quarter" (分期) vs "분기#branch" (分岐). The
# key is the scheduling *identity*; the part before SENSE_SEP is the displayed
# *headword*. ("~" is reserved for diff pairs, so we use "#".)
SENSE_SEP = "#"


def headword_of(lexeme: str) -> str:
    """The bare Korean surface for a (possibly sense-tagged) lexeme key.
    '분기#quarter' -> '분기'; '분기' -> '분기'. Use this anywhere the actual word
    is needed (display, occlusion, hanja lookup) rather than the schedule key."""
    return (lexeme or "").split(SENSE_SEP, 1)[0]


def sense_of(lexeme: str) -> Optional[str]:
    """The sense slug of a sense-tagged lexeme key, or None if untagged.
    '분기#quarter' -> 'quarter'; '분기' -> None."""
    parts = (lexeme or "").split(SENSE_SEP, 1)
    return parts[1] if len(parts) > 1 else None


def extract_lexeme_from_external_id(external_id: str) -> Optional[str]:
    result = parse_external_id(external_id)
    return result[0] if result else None


def group_items_by_lexeme(session: Session) -> Dict[str, LexemeGroup]:
    """
    Returns:
        lexeme -> LexemeGroup
    """
    items = session.scalars(
        select(Item)
        .options(joinedload(Item.srs_state))
        .where(Item.item_type == "card")
        .where(Item.suspended == False)  # noqa: E712
        .where(Item.deleted_at.is_(None))
    ).unique().all()

    groups: Dict[str, LexemeGroup] = {}

    for item in items:
        lexeme = extract_lexeme_from_external_id(item.external_id)
        if lexeme is None:
            continue

        if lexeme not in groups:
            groups[lexeme] = {
                "lexeme": lexeme,
                "items": [],
            }

        groups[lexeme]["items"].append(item)

    return groups


def classify_lexeme_groups(
    session: Session,
) -> Tuple[Dict[str, LexemeGroup], Dict[str, LexemeGroup]]:
    """
    Split all lexeme groups into seen vs unseen.

    "seen" = any item in the group has srs_state.last_reviewed_at IS NOT NULL.

    Returns:
        (seen_groups, unseen_groups)
    """
    all_groups = group_items_by_lexeme(session)

    seen: Dict[str, LexemeGroup] = {}
    unseen: Dict[str, LexemeGroup] = {}

    for lexeme, group in all_groups.items():
        has_review = any(
            item.srs_state is not None and item.srs_state.last_reviewed_at is not None
            for item in group["items"]
        )
        if has_review:
            seen[lexeme] = group
        else:
            unseen[lexeme] = group

    return seen, unseen


# ============================================================
# LLM Templates (used by dynamic/LLM items; optional for static)
# ============================================================

class LLMTemplate(Base):
    __tablename__ = "llm_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)

    system_prompt: Mapped[str] = mapped_column(Text, default="")
    user_prompt_template: Mapped[str] = mapped_column(Text)

    # Optional defaults (safe to ignore)
    model: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


# ============================================================
# Media (future-proofing: audio/images/etc.)
# ============================================================

class MediaAsset(Base):
    """
    Prefer storing media as a file path (keeps DB small).
    If you really want, you can store bytes in `data`.
    """
    __tablename__ = "media_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    kind: Mapped[str] = mapped_column(String(50), index=True)  # "audio", "image", "video", ...
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    file_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    data: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    meta: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class ItemMedia(Base):
    """
    Links items to media with a role like:
      - "front_audio"
      - "back_audio"
      - "image"
      - "hint_audio"
    """
    __tablename__ = "item_media"
    __table_args__ = (
        UniqueConstraint("item_id", "media_id", "role", name="uq_item_media_role"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), index=True)
    media_id: Mapped[int] = mapped_column(ForeignKey("media_assets.id", ondelete="CASCADE"), index=True)

    role: Mapped[str] = mapped_column(String(50), default="attachment", index=True)

    item: Mapped["Item"] = relationship(back_populates="media_links")
    media: Mapped[MediaAsset] = relationship()


# ============================================================
# Items (single polymorphic container)
# ============================================================

class Item(Base):
    """
    One row = one reviewable "thing".

    item_type examples:
      - "card"      : static card (Anki-like) where content holds the full card payload
      - "llm"       : dynamic LLM-generated question using template + variables (content holds vars)
      - future types: "passage", "dictation", "image_label", ...

    Canonical item data lives in `content` JSON.
    front/back are optional cached/rendered text for UI convenience.
    """
    __tablename__ = "items"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_items_external_id"),
        Index("ix_items_active", "item_type", "suspended", "deleted_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    item_type: Mapped[str] = mapped_column(String(50), index=True)  # e.g. "card", "llm"

    # Upsert key for static items (e.g., card_id from payload). Null for many dynamic items.
    external_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, index=True)

    # Optional cached UI fields (derive from content if you want)
    front: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    back: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Canonical data for the item (full payload / variables / anything)
    content: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Optional LLM template reference (used by item_type == "llm")
    llm_template_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("llm_templates.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    llm_template: Mapped[Optional[LLMTemplate]] = relationship()

    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)

    suspended: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    deleted_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    srs_state: Mapped["SRSState"] = relationship(
        back_populates="item", uselist=False, cascade="all, delete-orphan"
    )
    reviews: Mapped[List["ReviewLog"]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )
    media_links: Mapped[List[ItemMedia]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )


# ============================================================
# Scheduler state (decoupled; Bayesian-ready)
# ============================================================

class SRSState(Base):
    """
    Generic scheduling row:
      - due_at is the only thing your "due query" needs.
      - state is opaque blob owned by the scheduler implementation.

    This keeps you free to switch from SM-2 to Bayesian later
    without schema changes.
    """
    __tablename__ = "srs_state"

    item_id: Mapped[int] = mapped_column(
        ForeignKey("items.id", ondelete="CASCADE"),
        primary_key=True,
    )
    item: Mapped[Item] = relationship(back_populates="srs_state")

    due_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True, default=now_utc)
    last_reviewed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    scheduler_name: Mapped[str] = mapped_column(String(50), default="sm2")
    scheduler_version: Mapped[int] = mapped_column(Integer, default=1)

    # opaque scheduler-owned state (e.g. Bayesian params later)
    state: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


Index("ix_srs_due_active", SRSState.due_at, SRSState.item_id)


# ============================================================
# Review history (append-only; flexible payload)
# ============================================================

class ReviewLog(Base):
    __tablename__ = "review_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id", ondelete="CASCADE"), index=True)
    item: Mapped[Item] = relationship(back_populates="reviews")

    reviewed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now_utc, index=True)

    # Basic scoring signals (optional; keep generic)
    grade: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)     # e.g. 0..5
    correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True) # if you do binary

    # presentation mode / UI route
    mode: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # "card", "llm", "dictation", ...

    # All “interesting” attempt data goes here:
    # - prompt snapshot
    # - generated question
    # - user answer
    # - LLM rubric evaluation
    # - latency/model metadata
    payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Snapshot of scheduling result (helps debugging + analytics)
    new_due_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    new_scheduler_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    new_scheduler_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    new_state: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


Index("ix_review_item_time", ReviewLog.item_id, ReviewLog.reviewed_at)


# ============================================================
# Scheduler interface (swap SM-2 -> Bayesian later)
# ============================================================

@dataclass(frozen=True)
class ReviewEvent:
    reviewed_at: dt.datetime
    grade: Optional[int] = None
    correct: Optional[bool] = None
    response_text: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ScheduleUpdate:
    due_at: dt.datetime
    state: Dict[str, Any]
    scheduler_name: str
    scheduler_version: int


class Scheduler(Protocol):
    name: str
    version: int

    def init_state(self, item: Item) -> Dict[str, Any]:
        ...

    def update(self, *, item: Item, prev_state: Dict[str, Any], event: ReviewEvent) -> ScheduleUpdate:
        ...


# ============================================================
# DB helpers
# ============================================================

def make_engine(sqlite_path: str):
    url = f"sqlite:///{sqlite_path}"
    # timeout=30s: wait for a busy lock instead of immediately raising
    # "database is locked" (a concurrent reader colliding with a commit would
    # otherwise fail the commit and poison the session). Default is only 5s.
    return create_engine(url, future=True, connect_args={"timeout": 30.0})


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


# ============================================================
# Query helpers
# ============================================================

def get_due_items(session: Session, *, limit: int = 20, item_types: Optional[List[str]] = None) -> List[Item]:
    stmt = (
        select(Item)
        .join(SRSState, Item.id == SRSState.item_id)
        .where(Item.suspended == False)  # noqa: E712
        .where(Item.deleted_at.is_(None))
        .where(SRSState.due_at <= now_utc())
        .order_by(SRSState.due_at.asc())
        .limit(limit)
    )
    if item_types:
        stmt = stmt.where(Item.item_type.in_(item_types))
    return list(session.execute(stmt).scalars().all())


# ============================================================
# Ingest helpers
# ============================================================

def upsert_items_from_payload_cards(
    session: Session,
    payload: Dict[str, Any],
    *,
    item_type: str = "card",
    default_tags: Optional[List[str]] = None,
    initial_due_at: Optional[dt.datetime] = None,
) -> List[int]:
    """
    Takes one lexeme payload of the form:
      { "lexeme": {...}, "cards": [...], "diagnostics": {...} }

    Upserts each card into items.external_id == card["card_id"].
    Stores the full card JSON in Item.content (canonical) and front/back as cached.
    Ensures SRSState exists for each item.

    Returns list of Item ids.
    """
    if initial_due_at is None:
        initial_due_at = now_utc()
    if default_tags is None:
        default_tags = []

    out_ids: List[int] = []

    cards = payload.get("cards") or []
    for card in cards:
        card_id = card.get("card_id")
        if not card_id:
            raise ValueError("Card missing card_id; cannot upsert safely.")

        existing = session.execute(select(Item).where(Item.external_id == card_id)).scalar_one_or_none()

        tags = card.get("tags") or []
        merged_tags = list(dict.fromkeys(default_tags + tags))  # stable unique order

        front = card.get("front")
        back = card.get("back")

        if existing is None:
            item = Item(
                item_type=item_type,
                external_id=card_id,
                front=front,
                back=back,
                content=card,   # canonical
                tags=merged_tags,
            )
            session.add(item)
            session.flush()

            session.add(SRSState(
                item_id=item.id,
                due_at=initial_due_at,
                scheduler_name="sm2",
                scheduler_version=1,
                state={},  # scheduler init can overwrite later
            ))
            out_ids.append(item.id)
        else:
            existing.item_type = item_type
            existing.front = front
            existing.back = back
            existing.content = card
            existing.tags = merged_tags
            existing.updated_at = now_utc()

            if existing.srs_state is None:
                session.add(SRSState(
                    item_id=existing.id,
                    due_at=initial_due_at,
                    scheduler_name="sm2",
                    scheduler_version=1,
                    state={},
                ))
            out_ids.append(existing.id)

    return out_ids


# ============================================================
# Review application (single choke point for scheduler changes)
# ============================================================

def ensure_srs_state(session: Session, *, item: Item, scheduler: Scheduler, due_at: Optional[dt.datetime] = None) -> SRSState:
    if due_at is None:
        due_at = now_utc()

    if item.srs_state is not None:
        return item.srs_state

    state_row = SRSState(
        item_id=item.id,
        due_at=due_at,
        scheduler_name=scheduler.name,
        scheduler_version=scheduler.version,
        state=scheduler.init_state(item),
    )
    session.add(state_row)
    session.flush()
    return state_row


def apply_review(
    session: Session,
    *,
    item: Item,
    event: ReviewEvent,
    scheduler: Scheduler,
) -> ScheduleUpdate:
    """
    Central update function:
    - applies scheduler.update()
    - writes SRSState
    - appends ReviewLog

    When you switch to Bayesian scheduling later, you replace the scheduler implementation,
    not this function and not your DB schema.
    """
    state_row = ensure_srs_state(session, item=item, scheduler=scheduler, due_at=now_utc())
    prev_state = state_row.state or {}

    upd = scheduler.update(item=item, prev_state=prev_state, event=event)

    state_row.due_at = upd.due_at
    state_row.last_reviewed_at = event.reviewed_at
    state_row.scheduler_name = upd.scheduler_name
    state_row.scheduler_version = upd.scheduler_version
    state_row.state = upd.state

    session.add(ReviewLog(
        item_id=item.id,
        reviewed_at=event.reviewed_at,
        grade=event.grade,
        correct=event.correct,
        mode=item.item_type,
        payload=event.payload,
        new_due_at=upd.due_at,
        new_scheduler_name=upd.scheduler_name,
        new_scheduler_version=upd.scheduler_version,
        new_state=upd.state,
    ))

    return upd

