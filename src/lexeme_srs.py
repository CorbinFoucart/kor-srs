#!/usr/bin/env python3
"""
lexeme_srs.py

SRS experiment layer implementing the two-phase learning model:
    Day 0 -> Acquiring (daily gates) -> Maintenance (half-life SRS)
    with Repair detours on failure.

Implements the SRSProvider protocol from review_cli so it can be wired
directly into the review loop.

Two-pool model: previously-seen lexemes load as reviews (skip intro),
unseen lexemes go through Day 0. Per-lexeme state persists across
sessions via SRSState.state JSON.
"""

from __future__ import annotations

import datetime as dt
import random
import re
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.orm import Session

import srs_db
from rank_difficulty import compute_lexeme_difficulties
from review_cli import Grade, SessionCounts
from incremental_model import (
    GRADE_WEIGHT,
    TARGET_RECALL,
    recall_probability,
    update_half_life,
    next_interval,
    effective_eta,
    MAX_ODDS,
)
from acquisition_model import (
    LexemeState,
    lexeme_state_from_dict,
    lexeme_state_to_dict,
    make_initial_lexeme_state,
    compute_due_at as acquisition_compute_due_at,
    process_day0_touch_a_complete,
    process_day0_touch_b_complete,
    process_gate_review,
    graduate_to_maintenance,
    get_repair_strategy,
    day0_touch_a_cards,
    day0_touch_b_cards,
    reset_daily_flags,
    PHASE_DAY0,
    PHASE_ACQUIRING,
    PHASE_MAINTENANCE,
    PHASE_REPAIRING,
    PHASE_LEARNED,
    DAY0_TOUCH_A,
    DAY0_TOUCH_B,
    DAY0_DONE,
    GRADUATION_H,
    REPAIR_RETRY_DELAY,
)


# ── strategy enum ────────────────────────────────────────────────────────

class LearningStrategy(Enum):
    PRODUCTION_ONLY        = "production_only"
    RECOGNITION_ONLY       = "recognition_only"
    MIXED                  = "mixed"
    INCREMENTAL_PRODUCTION = "incremental_production"


SKILL_PRODUCTION      = "cloze_prod_bundle"
SKILL_RECOGNITION     = "cloze_recog_bundle"
SKILL_DIFFERENTIATION = "diff_bundle"

GRADUATION_STREAK = 2  # correct in a row to graduate (legacy strategies)
BASE_INTERVAL = dt.timedelta(seconds=60)   # 1 min starting interval (legacy default)
INTRO_COOLDOWN = dt.timedelta(seconds=10)

# REPAIR_RETRY_DELAY is imported from acquisition_model (single source of truth,
# shared with compute_due_at so the due count and the review countdown agree).
REPAIR_FINAL_DELAY = 300      # seconds — delay before final production card (5 min)
REPAIR_STALE_SECS  = 3600     # seconds — repairs older than this jump ahead of reviews

# per-card randomized scheduling ranges (legacy)
BASE_INTERVAL_RANGE = (60, 180)   # seconds, uniform [1, 3] minutes
MULTIPLIER_RANGE    = (2.0, 5.0)  # uniform [2, 5]

# grade-dependent multiplier scaling (legacy)
EASY_CORRECT_MULTIPLIER = 5.0
HARD_CORRECT_MULTIPLIER_FACTOR = 0.6


def _sample_base_interval() -> dt.timedelta:
    return dt.timedelta(seconds=random.uniform(*BASE_INTERVAL_RANGE))


def _sample_multiplier() -> float:
    return random.uniform(*MULTIPLIER_RANGE)


# ── Korean syllable occlusion utilities ──────────────────────────────────

def _occludable_syllables(headword: str) -> list[int]:
    """Return indices of stem syllables eligible for occlusion."""
    if len(headword) >= 3 and headword.endswith("하다"):
        stem = headword[:-2]
    elif len(headword) >= 3 and headword.endswith("되다"):
        stem = headword[:-2]
    elif len(headword) >= 2 and headword.endswith("다"):
        stem = headword[:-1]
    else:
        stem = headword
    return list(range(len(stem)))


def _occlude_headword(headword: str) -> str | None:
    """Replace one random stem syllable with '_'. Returns None if impossible."""
    indices = _occludable_syllables(headword)
    if not indices:
        return None
    idx = random.choice(indices)
    chars = list(headword)
    chars[idx] = "_"
    return "".join(chars)


def _apply_hint_occlusion(front: str, headword: str) -> str | None:
    """Replace [hint: ...] in a prod front string with an occluded Korean form."""
    occluded = _occlude_headword(headword)
    if occluded is None:
        return None
    return re.sub(r"\[hint:\s*[^\]]*\]", f"[hint: {occluded}]", front, count=1)


# ── card wrapper (satisfies ReviewItem protocol) ─────────────────────────

@dataclass(frozen=True)
class CardWrapper:
    item_id: int
    front: str
    back: str
    requires_input: bool = False
    expected_input: str | None = None
    review_mode: str = "review"  # "intro", "review", "gate", or "repair"
    translation_en: str = ""
    skill_type: str = ""
    difficulty: float = 0.0
    hanja: dict | None = None  # per-lexeme hanja breakdown (recog cards only)
    headword: str = ""  # dictionary lemma (sense tag stripped) for the answer entry


# ── variant: one example extracted from the content JSON ─────────────────

@dataclass
class Variant:
    item_id: int       # parent DB Item id
    skill: str         # e.g. "cloze_prod_bundle"
    variant_index: int # index into content["variants"]
    front: str
    back: str
    translation_en: str = ""


def _extract_variants(item: srs_db.Item, skill: str) -> list[Variant]:
    """Pull all variants out of item.content['variants']."""
    content = item.content or {}
    raw = content.get("variants", [])
    return [
        Variant(
            item_id=item.id,
            skill=skill,
            variant_index=i,
            front=v.get("front", ""),
            back=v.get("back", ""),
            translation_en=v.get("translation_en", ""),
        )
        for i, v in enumerate(raw)
        if v.get("front") or v.get("back")
    ]


# ── helpers for building entries from DB items ────────────────────────────

def _build_skill_map(items: list[srs_db.Item]) -> dict[str, list[Variant]]:
    """Extract skill -> variants mapping from a list of DB items."""
    by_skill: dict[str, list[Variant]] = {}
    for item in items:
        parsed = srs_db.parse_external_id(item.external_id)
        skill = parsed[1] if parsed else "unknown"
        by_skill.setdefault(skill, []).extend(
            _extract_variants(item, skill)
        )
    return by_skill


def _canonical_item(items: list[srs_db.Item]) -> srs_db.Item | None:
    """Pick the prod-bundle item as the canonical item for per-lexeme state storage."""
    for item in items:
        parsed = srs_db.parse_external_id(item.external_id)
        if parsed and parsed[1] == SKILL_PRODUCTION:
            return item
    return items[0] if items else None


def _extract_hanja(items: list[srs_db.Item]) -> dict | None:
    """Pull the cached per-lexeme hanja sentinel from the recog bundle item's
    content["hanja"]. Returns the stored dict verbatim — {"has_hanja": true,
    ...} or {"has_hanja": false} — so the UI can distinguish "looked up, none"
    from "never looked up" (None)."""
    for item in items:
        parsed = srs_db.parse_external_id(item.external_id)
        if parsed and parsed[1] == SKILL_RECOGNITION:
            h = (item.content or {}).get("hanja")
            return h if isinstance(h, dict) else None
    return None


# ── intro queue ordering (unseen / Day-0 words) ──────────────────────────
#
# "Upcoming introductions" are the unseen lexemes; only the first `target_new`
# of them enter a session's new pool. The user can pin a manual order via
# set_intro_order(), stored as an integer `intro_order` on the rank-holder
# item's SRSState.state. Ordering rule:
#   · UNRANKED words sort FIRST, newest-first (LIFO) — so a freshly added word
#     always jumps to the front of the learning queue, the documented design.
#   · RANKED (pinned) words follow, in their saved order.
# A pin therefore arranges the *existing* backlog without ever demoting new
# words below it: add words → they lead; re-pin / spread to fold them in. The
# rank lives on the canonical item, or any group item that has an SRSState. (A
# lexeme with NO schedule row anywhere can't hold a rank — keep every unseen
# lexeme schedule-backed, or it sticks in the unranked/new band; see
# fix_orphan_states.py.)

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=srs_db.UTC)


def _group_created_at(items: list[srs_db.Item]) -> dt.datetime:
    """Newest created_at across a lexeme group's items (UTC-aware)."""
    dates = []
    for item in items:
        d = item.created_at
        if d is not None:
            if d.tzinfo is None:
                d = d.replace(tzinfo=srs_db.UTC)
            dates.append(d)
    return max(dates) if dates else _EPOCH


def _rank_holder(items: list[srs_db.Item]) -> srs_db.Item | None:
    """The item whose SRSState.state holds this group's intro_order: the
    canonical (prod-bundle) item if it has a schedule row, else any group item
    that does."""
    canonical = _canonical_item(items)
    if canonical is not None and canonical.srs_state is not None:
        return canonical
    for it in items:
        if it.srs_state is not None:
            return it
    return canonical


def _intro_order_of(items: list[srs_db.Item]) -> float | None:
    """Manual intro rank for a lexeme group, or None if never pinned."""
    holder = _rank_holder(items)
    if holder is None or holder.srs_state is None:
        return None
    st = holder.srs_state.state
    if isinstance(st, dict):
        v = st.get("intro_order")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _intro_sort_key(items: list[srs_db.Item]):
    """Sort key: unranked words first (newest-first, LIFO), then pinned words in
    saved order. New/unplaced words always lead the queue; a pin orders the rest."""
    order = _intro_order_of(items)
    if order is None:
        return (0, -_group_created_at(items).timestamp())
    return (1, order)


def sorted_unseen_lexemes(unseen_groups: dict) -> list[str]:
    """Unseen lexemes (from classify_lexeme_groups) in upcoming-intro order."""
    return sorted(unseen_groups.keys(),
                  key=lambda lex: _intro_sort_key(unseen_groups[lex]["items"]))


def _recog_gloss(items: list[srs_db.Item]) -> tuple[str, int]:
    """(first non-empty recog variant back, recog variant count) for display."""
    for item in items:
        parsed = srs_db.parse_external_id(item.external_id)
        if parsed and parsed[1] == SKILL_RECOGNITION:
            variants = (item.content or {}).get("variants", [])
            gloss = next((((v.get("back") or "").strip()) for v in variants
                          if (v.get("back") or "").strip()), "")
            return gloss, len(variants)
    return "", 0


def intro_queue_listing(session: Session, target_new: int) -> list[dict]:
    """Full ordered list of upcoming word introductions (unseen lexemes).

    Reads the DB directly, so it works with or without a live session. The
    first `target_new` entries are flagged in_session (they enter the new pool
    when a session starts / restarts)."""
    _, unseen_groups = srs_db.classify_lexeme_groups(session)
    ordered = sorted_unseen_lexemes(unseen_groups)
    out: list[dict] = []
    for i, lex in enumerate(ordered):
        items = unseen_groups[lex]["items"]
        gloss, nvar = _recog_gloss(items)
        out.append({
            "lexeme": lex,
            "headword": srs_db.headword_of(lex),
            "sense": srs_db.sense_of(lex),
            "gloss": gloss,
            "num_variants": nvar,
            "ranked": _intro_order_of(items) is not None,
            "in_session": i < target_new,
            "created_at": _group_created_at(items).isoformat(),
        })
    return out


def set_intro_order(session: Session, ordered_lexemes: list[str]) -> int:
    """Pin a manual intro order: assign intro_order = position (from 0) to each
    given unseen lexeme. Unknown / already-seen lexemes are skipped. Pass the
    FULL current order so every entry is re-ranked. Returns count updated;
    caller commits."""
    _, unseen_groups = srs_db.classify_lexeme_groups(session)
    updated = 0
    for rank, lex in enumerate(ordered_lexemes):
        group = unseen_groups.get(lex)
        if group is None:
            continue
        holder = _rank_holder(group["items"])
        if holder is None or holder.srs_state is None:
            continue
        st = holder.srs_state.state
        new_state = dict(st) if isinstance(st, dict) else {}
        new_state["intro_order"] = rank
        holder.srs_state.state = new_state   # reassign → JSON column dirties
        updated += 1
    return updated


def spread_homonyms(ordered: list[str]) -> list[str]:
    """Reorder so senses that share a headword (homonyms — e.g. 사기#fraud /
    사기#morale / 사기#ceramics, or a tagged sense alongside a plain spelling)
    are spread evenly across the whole list rather than clustered together.

    It's a well-established principle that confusable homonyms shouldn't be
    learned at the same time, so each headword's k members are distributed at
    ~N/k intervals. Singletons keep their slot; multi-sense members are anchored
    near their group's current centre and fanned out across [0, N). Stable:
    ties (and the singleton stream) preserve the incoming order."""
    n = len(ordered)
    if n <= 1:
        return list(ordered)
    groups: dict[str, list[tuple[int, str]]] = {}
    for i, lex in enumerate(ordered):
        groups.setdefault(srs_db.headword_of(lex), []).append((i, lex))

    placed: list[tuple[float, int, str]] = []  # (target_slot, orig_index, lexeme)
    for members in groups.values():
        k = len(members)
        if k == 1:
            i0, lex = members[0]
            placed.append((float(i0), i0, lex))          # singleton: stay put
            continue
        anchor = (sum(i for i, _ in members) / k) / n    # group centre in [0,1)
        for j, (i0, lex) in enumerate(members):
            placed.append((((j + anchor) / k) * n, i0, lex))
    placed.sort(key=lambda t: (t[0], t[1]))
    return [lex for _, _, lex in placed]


def intros_today_count(session: Session, now: dt.datetime) -> int:
    """Distinct lexemes first introduced (Day-0 intro) so far today (UTC).

    Makes `target_new` a per-DAY budget instead of per-session: without this,
    every restart / new session starts a fresh batch of target_new new words,
    so new-word introductions accumulate without bound across a day."""
    from sqlalchemy import select
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = session.execute(
        select(srs_db.Item.external_id)
        .join(srs_db.ReviewLog, srs_db.ReviewLog.item_id == srs_db.Item.id)
        .where(srs_db.ReviewLog.mode == "intro")
        .where(srs_db.ReviewLog.reviewed_at >= day_start)
    ).all()
    lexemes = {srs_db.extract_lexeme_from_external_id(r[0]) for r in rows}
    lexemes.discard(None)
    return len(lexemes)


# ── queue entry (one lexeme being actively learned) ──────────────────────

@dataclass
class LexemeQueueEntry:
    lexeme: str
    variants_by_skill: dict[str, list[Variant]]
    strategy: LearningStrategy
    canonical_item_id: int | None = None

    # New phase-based state (replaces skill_states for INCREMENTAL_PRODUCTION)
    state: LexemeState | None = None

    # Queues for multi-card sequences (Day 0 and repair)
    day0_queue: list[str] = field(default_factory=list)
    repair_queue: list[str] = field(default_factory=list)
    # Pre-shuffled, unique recognition variants for the Day-0 intro
    # sequence — popped one per intro so the user never sees the same
    # sentence twice during Touch A.
    day0_variant_pool: list[Variant] = field(default_factory=list)

    # A/B intro strategy: "interleaved" (multiple words at once) or "sequential" (one at a time)
    intro_mode: str = ""

    # Per-lexeme hanja breakdown (from the recog bundle's content["hanja"]),
    # or None for native / no-hanja words. Surfaced on recog cards only.
    hanja: dict | None = None

    # Legacy fields (kept for non-INCREMENTAL strategies)
    correct_streak: int = 0
    total_reviews: int = 0
    base_interval: dt.timedelta = field(default_factory=_sample_base_interval)
    multiplier: float = field(default_factory=_sample_multiplier)
    current_interval: dt.timedelta | None = None
    intro_queue: list[Variant] = field(default_factory=list)
    intro_done: bool = False

    due_at: dt.datetime = field(default_factory=srs_db.now_utc)

    def __post_init__(self):
        if self.current_interval is None:
            self.current_interval = self.base_interval


def _format_half_life(secs: float) -> str:
    """Human-readable half-life."""
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}m"
    if secs < 86400:
        return f"{secs / 3600:.1f}h"
    return f"{secs / 86400:.1f}d"


def _log_phase_state(label: str, lexeme: str, state: LexemeState, now: dt.datetime) -> None:
    """Print a compact summary of lexeme phase state to console."""
    parts = [f"phase={state.phase}"]
    if state.phase == PHASE_DAY0:
        parts.append(f"step={state.day0_step}")
    elif state.phase == PHASE_ACQUIRING:
        parts.append(f"passes={state.passes_24h}")
        parts.append(f"hard={state.is_hard}")
    elif state.phase == PHASE_MAINTENANCE:
        hl = _format_half_life(state.half_life_secs)
        if state.last_reviewed_at:
            delta_t = (now - state.last_reviewed_at).total_seconds()
            p = recall_probability(state.half_life_secs, delta_t)
            parts.append(f"H={hl} p={p:.2f}")
        else:
            parts.append(f"H={hl}")
        ivl = _format_half_life(next_interval(state.half_life_secs, "recognition"))
        parts.append(f"ivl={ivl}")
    elif state.phase == PHASE_REPAIRING:
        parts.append(f"strategy={state.repair_strategy}")
        parts.append(f"return={state.return_phase}")
    print(f"        [{label}] {lexeme}  {' | '.join(parts)}")


# ── main SRS class ──────────────────────────────────────────────────────

class LexemeSRS:
    """
    Manages a learning session with two concurrent queues:

      _learning_queue -- capped at queue_size; new words go through Day 0
                         then acquiring (daily gates) before graduating.
      _review_queue   -- uncapped; all previously-seen lexemes live here
                         permanently, scheduled by the half-life memory model.
    """

    def __init__(self, session: Session, *, target_new: int, queue_size: int, intro_examples: int = 2):
        self._intro_examples = intro_examples
        seen_groups, unseen_groups = srs_db.classify_lexeme_groups(session)
        now = srs_db.now_utc()

        # ── review queue: all previously-seen lexemes (due or not) ──
        self._review_queue: list[LexemeQueueEntry] = []
        self._repair_pool: list[LexemeQueueEntry] = []    # deferred maintenance repairs
        self._repair_active: list[LexemeQueueEntry] = []  # currently repairing, capped
        self.in_repair_mode: bool = False
        for lex, group in seen_groups.items():
            items = group["items"]
            by_skill = _build_skill_map(items)
            canonical = _canonical_item(items)
            saved = self._load_lexeme_state(canonical)

            # load persisted per-card schedule params; fall back to legacy defaults
            base_iv = dt.timedelta(seconds=saved.get("base_interval_secs", BASE_INTERVAL.total_seconds()))
            mult = saved.get("multiplier", 3.0)

            entry = LexemeQueueEntry(
                lexeme=lex,
                variants_by_skill=by_skill,
                strategy=saved.get("strategy", LearningStrategy.INCREMENTAL_PRODUCTION),
                canonical_item_id=canonical.id if canonical else None,
                correct_streak=saved.get("correct_streak", 0),
                current_interval=dt.timedelta(seconds=saved.get("interval_secs", base_iv.total_seconds())),
                base_interval=base_iv,
                multiplier=mult,
                intro_done=True,
                due_at=self._earliest_due(items),
                hanja=_extract_hanja(items),
            )

            # Load LexemeState for INCREMENTAL_PRODUCTION
            if entry.strategy == LearningStrategy.INCREMENTAL_PRODUCTION:
                raw_state = saved.get("lexeme_state") or saved.get("skill_states") or saved
                entry.state = lexeme_state_from_dict(raw_state)

                # Reset daily flags only if a new gate window has arrived
                reset_daily_flags(entry.state, now)

                entry.due_at = acquisition_compute_due_at(entry.state, now)

                # If in day0 with pending touch, populate day0_queue
                if entry.state.phase == PHASE_DAY0:
                    if entry.state.day0_step == DAY0_TOUCH_A:
                        entry.day0_queue = day0_touch_a_cards(self._intro_examples)
                    elif entry.state.day0_step == DAY0_TOUCH_B:
                        entry.day0_queue = day0_touch_b_cards()

                # If in repairing, rebuild repair_queue
                if entry.state.phase == PHASE_REPAIRING and entry.state.repair_strategy:
                    strategy = get_repair_strategy(entry.state.repair_strategy)
                    card_type = strategy.next_card_type(entry.state.repair_state)
                    if card_type is not None:
                        # rebuild remaining sequence
                        remaining = []
                        temp_state = dict(entry.state.repair_state)
                        while card_type is not None:
                            remaining.append(card_type)
                            temp_state = strategy.process_review(temp_state, 1)  # dummy grade to advance
                            card_type = strategy.next_card_type(temp_state)
                        entry.repair_queue = remaining

            # Route ALL repairing entries to repair pool
            if entry.state and entry.state.phase == PHASE_REPAIRING:
                self._repair_pool.append(entry)
            else:
                self._review_queue.append(entry)

        self._review_queue.sort(key=lambda e: e.due_at)

        # ── new pool: target_new is a per-DAY budget. Subtract words already
        # introduced today so restarts/new sessions don't each dump another
        # target_new batch (the cause of new words piling up with nothing due).
        # Ordering: manual intro order if pinned, else LIFO (newest first).
        self._intros_today = intros_today_count(session, now)
        remaining_new = max(0, target_new - self._intros_today)
        unseen_lexemes = sorted_unseen_lexemes(unseen_groups)
        selected = unseen_lexemes[:remaining_new]

        self._new_pool: list[LexemeQueueEntry] = []
        for lex in selected:
            items = unseen_groups[lex]["items"]
            by_skill = _build_skill_map(items)
            canonical = _canonical_item(items)

            new_entry = LexemeQueueEntry(
                lexeme=lex,
                variants_by_skill=by_skill,
                strategy=LearningStrategy.INCREMENTAL_PRODUCTION,
                canonical_item_id=canonical.id if canonical else None,
                hanja=_extract_hanja(items),
            )
            new_entry.state = make_initial_lexeme_state()
            self._new_pool.append(new_entry)

        # ── intro strategy A/B test (alternates by day of year) ──
        day_of_year = now.timetuple().tm_yday
        self._intro_mode = "interleaved" if day_of_year % 2 == 0 else "sequential"

        # ── learning queue (capped) ──
        self._queue_size = queue_size
        self._learning_queue: list[LexemeQueueEntry] = []
        self._current_entry: LexemeQueueEntry | None = None
        self._current_variant: Variant | None = None
        self._current_card_type: str | None = None  # "recognition", "occlusion", "production"
        self._session = session

        # ── difficulty scores (computed once at session start) ──
        self._difficulty_by_lexeme: dict[str, float] = compute_lexeme_difficulties(session)

        # ── progress tracking ──
        self._initial_target_count = len(self._new_pool)
        self._target_graduated = 0
        self._total_graded = 0
        self._correct_count = 0
        self.last_review_log: str = ""  # one-line summary of last review for UI

        self._fill_learning_queue()
        print(f"Intro mode: {self._intro_mode} (day {day_of_year})")

    # ── helpers for loading / persisting per-lexeme state ─────────────

    @staticmethod
    def _earliest_due(items: list[srs_db.Item]) -> dt.datetime:
        """Return the earliest due_at across a lexeme's items (always tz-aware)."""
        dues: list[dt.datetime] = []
        for item in items:
            if item.srs_state is not None and item.srs_state.due_at is not None:
                d = item.srs_state.due_at
                if d.tzinfo is None:
                    d = d.replace(tzinfo=srs_db.UTC)
                dues.append(d)
        return min(dues) if dues else srs_db.now_utc()

    @staticmethod
    def _load_lexeme_state(canonical: srs_db.Item | None) -> dict:
        """Read persisted lexeme-level state from the canonical item's SRSState.state."""
        if canonical is None or canonical.srs_state is None:
            return {}
        state = canonical.srs_state.state or {}
        strategy_val = state.get("strategy")
        if strategy_val is not None:
            try:
                state = dict(state)
                state["strategy"] = LearningStrategy(strategy_val)
            except ValueError:
                pass
        return state

    def _persist_lexeme_state(self, entry: LexemeQueueEntry) -> None:
        """Write lexeme state to the canonical item's SRSState.state."""
        if entry.canonical_item_id is None:
            return
        srs_state = self._session.get(srs_db.SRSState, entry.canonical_item_id)
        if srs_state is None:
            # First review of an unseen word — create the SRSState row
            now = srs_db.now_utc()
            srs_state = srs_db.SRSState(
                item_id=entry.canonical_item_id,
                due_at=entry.due_at or now,
                scheduler_name="incremental",
                scheduler_version=1,
                state={},
            )
            self._session.add(srs_state)

        if entry.strategy == LearningStrategy.INCREMENTAL_PRODUCTION and entry.state:
            saved = {
                "strategy": entry.strategy.value,
                "lexeme_state": lexeme_state_to_dict(entry.state),
            }
        else:
            # Legacy strategies
            saved = dict(srs_state.state or {})
            saved["correct_streak"] = entry.correct_streak
            saved["interval_secs"] = entry.current_interval.total_seconds()
            saved["strategy"] = entry.strategy.value
            saved["base_interval_secs"] = entry.base_interval.total_seconds()
            saved["multiplier"] = entry.multiplier

        srs_state.state = saved

    # ── queue management ─────────────────────────────────────────────

    def _fill_repair_active(self, now: dt.datetime | None = None) -> None:
        """Pull from _repair_pool into _repair_active up to queue_size.
        Preserves any future due_at the pool entry already carries (e.g. the
        post-failure REPAIR_RETRY_DELAY); only past-due entries are bumped
        to now."""
        if now is None:
            now = srs_db.now_utc()
        # Pull stale repairs (failed >1h ago) before fresh ones so they
        # aren't stuck behind a full active batch of recent failures.
        self._repair_pool.sort(
            key=lambda e: not self._repair_is_stale(e, now)
        )
        while len(self._repair_active) < self._queue_size and self._repair_pool:
            entry = self._repair_pool.pop(0)
            if entry.due_at < now:
                entry.due_at = now
            self._repair_active.append(entry)

    def _fill_learning_queue(self) -> None:
        """Pull from _new_pool into _learning_queue up to queue_size.

        In sequential intro mode, only allow 1 word in the learning queue
        at a time (finish Touch A before starting the next word).
        """
        cap = 1 if self._intro_mode == "sequential" else self._queue_size
        while len(self._learning_queue) < cap and self._new_pool:
            entry = self._new_pool.pop(0)
            entry.intro_mode = self._intro_mode
            if entry.state:
                entry.state.intro_mode = self._intro_mode

            if entry.strategy == LearningStrategy.INCREMENTAL_PRODUCTION and entry.state:
                # Pre-shuffle a unique-variant pool capped at intro_examples,
                # so Touch A serves min(intro_examples, n_variants) distinct
                # example sentences with no repeats.
                recog_variants = list(
                    entry.variants_by_skill.get(SKILL_RECOGNITION, [])
                )
                random.shuffle(recog_variants)
                n_intros = min(self._intro_examples, len(recog_variants))
                entry.day0_variant_pool = recog_variants[:n_intros]
                entry.day0_queue = ["recognition"] * n_intros
                entry.intro_done = False  # Day 0 uses intro-style display for Touch A
            else:
                # Legacy: use old intro system
                all_variants = [
                    v
                    for variants in entry.variants_by_skill.values()
                    for v in variants
                ]
                entry.intro_queue = all_variants[:self._intro_examples]

            self._learning_queue.append(entry)

    def hot_add_word(self, lexeme: str) -> bool:
        """Hot-load a newly added word into the live session from DB.

        Returns True if the word was added, False if already present or not found.
        """
        # Skip if already in any queue
        all_lexemes = set()
        for e in (self._new_pool + self._learning_queue + self._review_queue
                  + self._repair_pool + self._repair_active):
            all_lexemes.add(e.lexeme)
        if lexeme in all_lexemes:
            return False

        # Query DB for this lexeme's items
        from sqlalchemy import select
        from sqlalchemy.orm import joinedload
        items = self._session.scalars(
            select(srs_db.Item)
            .options(joinedload(srs_db.Item.srs_state))
            .where(srs_db.Item.item_type == "card")
            .where(srs_db.Item.suspended == False)  # noqa: E712
            .where(srs_db.Item.deleted_at.is_(None))
            .where(srs_db.Item.external_id.like(f"lexeme:{lexeme}:%"))
        ).unique().all()

        if not items:
            return False

        by_skill = _build_skill_map(items)
        canonical = _canonical_item(items)

        # Check if it's already "seen" (has SRS state with last_reviewed_at)
        is_seen = any(
            item.srs_state is not None and item.srs_state.last_reviewed_at is not None
            for item in items
        )

        entry = LexemeQueueEntry(
            lexeme=lexeme,
            variants_by_skill=by_skill,
            strategy=LearningStrategy.INCREMENTAL_PRODUCTION,
            canonical_item_id=canonical.id if canonical else None,
            hanja=_extract_hanja(items),
        )

        if is_seen and canonical and canonical.srs_state:
            # Load persisted state (maintenance words, diff pairs)
            st = canonical.srs_state.state or {}
            raw_state = st.get("lexeme_state") or st
            entry.state = lexeme_state_from_dict(raw_state)
            entry.due_at = acquisition_compute_due_at(entry.state, srs_db.now_utc())
            self._review_queue.append(entry)
            self._review_queue.sort(key=lambda e: e.due_at)
        else:
            # Unseen word — add to new pool (LIFO)
            entry.state = make_initial_lexeme_state()
            self._new_pool.insert(0, entry)
            self._fill_learning_queue()

        print(f"[HOT-ADD] {lexeme} added to {'review queue' if is_seen else 'new pool'}")
        return True

    def lookup_hanja(self, item_id: int) -> dict:
        """Return the Hanja breakdown for the lexeme owning `item_id`.

        If a result is already cached on the recog item's content, returns it
        verbatim with no LLM call. Otherwise performs an LLM lookup, caches the
        sentinel on content["hanja"], patches matching live queue entries so
        the rest of the session sees the cache, and returns it.

        `item_id` is the recognition bundle item id (CardWrapper.item_id for a
        recog card). On a transient/lookup failure returns an UNCACHED
        {"error": ...} sentinel (NOT {"has_hanja": false}) so the UI offers a
        retry instead of mislabeling the word as having no Hanja — a real Sino
        word like 의류=衣類 must never look 'native' because of a network blip."""
        item = self._session.get(srs_db.Item, item_id)
        if item is None:
            return {"error": "no_item"}

        content = dict(item.content or {})
        cached = content.get("hanja")
        if isinstance(cached, dict):
            return cached  # already looked up — no LLM call

        parsed = srs_db.parse_external_id(item.external_id)
        lexeme = parsed[0] if parsed else None
        if not lexeme:
            return {"error": "no_lexeme"}

        # Use the bare surface for the lookup; pass the sense gloss (the recog
        # back, identical across variants) so homographs resolve to the right
        # Hanja (분기#quarter -> 分期, 분기#branch -> 分岐).
        headword = srs_db.headword_of(lexeme)
        variants = content.get("variants", [])
        gloss = next((v.get("back") for v in variants if (v.get("back") or "").strip()), None)

        from hanja import lookup_hanja as _llm_lookup
        try:
            result = _llm_lookup(headword, gloss=gloss)
        except Exception as e:
            print(f"[HANJA] lookup failed for {lexeme}: {e}")
            return {"error": "lookup_failed"}  # uncached → retryable, not 'no hanja'

        content["hanja"] = result
        item.content = content
        self._session.commit()

        # Patch any live entries for this lexeme so re-displays (e.g. repair)
        # this session render the cache without another lookup.
        for e in (self._review_queue + self._learning_queue + self._new_pool
                  + self._repair_pool + self._repair_active):
            if e.lexeme == lexeme:
                e.hanja = result

        print(f"[HANJA] {lexeme}: {'has hanja' if result.get('has_hanja') else 'none'}")
        return result

    # ── helpers ──────────────────────────────────────────────────────

    def _persist_review(
        self,
        variant: Variant,
        *,
        mode: str,
        grade: Grade | None = None,
        new_due_at: dt.datetime,
        entry: LexemeQueueEntry | None = None,
        prev_interval: dt.timedelta | None = None,
        new_interval: dt.timedelta | None = None,
        H_before: float | None = None,
        phase_override: str | None = None,
    ) -> None:
        """Write a ReviewLog row, update SRSState.due_at on all lexeme items, and persist lexeme state."""
        now = srs_db.now_utc()

        # update due_at on ALL items in this lexeme (not just the reviewed variant)
        if entry is not None:
            all_item_ids = {v.item_id for vs in entry.variants_by_skill.values() for v in vs}
        else:
            all_item_ids = {variant.item_id}

        for item_id in all_item_ids:
            srs_state = self._session.get(srs_db.SRSState, item_id)
            if srs_state is None:
                srs_state = srs_db.SRSState(
                    item_id=item_id,
                    due_at=new_due_at,
                    scheduler_name="incremental",
                    scheduler_version=1,
                    state={},
                )
                self._session.add(srs_state)
            srs_state.due_at = new_due_at
            if item_id == variant.item_id:
                    srs_state.last_reviewed_at = now

        payload = {
            "variant_index": variant.variant_index,
            "skill": variant.skill,
            "front": variant.front,
            "back": variant.back,
        }
        if prev_interval is not None:
            payload["prev_interval_secs"] = prev_interval.total_seconds()
        if new_interval is not None:
            payload["new_interval_secs"] = new_interval.total_seconds()

        # Phase metadata for new model
        # Use phase_override (captured before handler dispatch) so that
        # mutations during the handler don't mis-label the review.
        if entry is not None and entry.state:
            logged_phase = phase_override if phase_override is not None else entry.state.phase
            payload["phase"] = logged_phase
            payload["card_type"] = self._current_card_type
            if entry.intro_mode:
                payload["intro_mode"] = entry.intro_mode
            if logged_phase == PHASE_DAY0:
                payload["day0_step"] = entry.state.day0_step
            elif logged_phase == PHASE_REPAIRING:
                payload["repair_strategy"] = entry.state.repair_strategy
                payload["repair_step"] = entry.state.repair_state.get("step", 0)
            elif logged_phase == PHASE_MAINTENANCE:
                payload["half_life_secs"] = entry.state.half_life_secs
                if H_before is not None:
                    payload["half_life_secs_before"] = H_before
                payload["incremental_skill"] = "recognition"

        self._session.add(srs_db.ReviewLog(
            item_id=variant.item_id,
            reviewed_at=now,
            grade=grade.value if grade is not None else None,
            correct=grade.is_correct if grade is not None else None,
            mode=mode,
            payload=payload,
            new_due_at=new_due_at,
        ))

        if entry is not None:
            self._persist_lexeme_state(entry)

        self._session.commit()

    @staticmethod
    def _variant_to_card(variant: Variant, *, review_mode: str, difficulty: float = 0.0, front_override: str | None = None, hanja: dict | None = None, headword: str = "") -> CardWrapper:
        # Hanja is a recognition-card affordance only — never decorate a
        # production card (the headword is the answer the learner produces).
        show_hanja = hanja if variant.skill == SKILL_RECOGNITION else None
        return CardWrapper(
            item_id=variant.item_id,
            front=front_override if front_override is not None else variant.front,
            back=variant.back,
            review_mode=review_mode,
            translation_en=variant.translation_en,
            skill_type=variant.skill,
            difficulty=difficulty,
            hanja=show_hanja,
            headword=headword,
        )

    # ── Card selection for phase-based model ─────────────────────────

    def _select_card_for_phase(self, entry: LexemeQueueEntry) -> tuple[Variant | None, str]:
        """Select a variant and determine review_mode based on the current phase.

        Returns (variant, review_mode) where review_mode is "intro", "review", "gate", or "repair".
        """
        state = entry.state
        skills = entry.variants_by_skill

        if state.phase == PHASE_DAY0:
            # Pop from day0_queue
            if entry.day0_queue:
                card_type = entry.day0_queue[0]
                self._current_card_type = card_type
                # Touch A: pop a pre-shuffled unique variant from the pool.
                # Legacy Touch B (only for words mid-pipeline before the
                # simplification) falls back to the random-choice path.
                if state.day0_step == DAY0_TOUCH_A and entry.day0_variant_pool:
                    variant = entry.day0_variant_pool.pop(0)
                else:
                    if card_type == "recognition":
                        candidates = skills.get(SKILL_RECOGNITION, [])
                    else:  # "occlusion" or "production" — legacy
                        candidates = skills.get(SKILL_PRODUCTION, [])
                    if not candidates:
                        candidates = [v for vs in skills.values() for v in vs]
                    variant = random.choice(candidates) if candidates else None

                # Touch A is all intro-style. Legacy Touch B: the last card
                # is the graded recognition test.
                if state.day0_step == DAY0_TOUCH_A:
                    mode = "intro"
                else:
                    mode = "review" if len(entry.day0_queue) == 1 else "intro"
                return variant, mode
            # Empty queue — auto-advance phase to avoid stuck state.
            # NEW: Touch A done → go straight to maintenance (skip Touch B + acquiring).
            now = srs_db.now_utc()
            if state.day0_step == DAY0_TOUCH_A:
                process_day0_touch_a_complete(state, now)
            elif state.day0_step == DAY0_TOUCH_B:
                # Legacy: only fires for words already mid-Touch-B at the
                # time of the simplification — they finish their old path.
                process_day0_touch_b_complete(state, now)
            else:
                # day0 done but phase not advanced — fall to maintenance.
                state.phase = PHASE_MAINTENANCE
                if state.half_life_secs <= 0:
                    state.half_life_secs = GRADUATION_H
                if state.last_reviewed_at is None:
                    state.last_reviewed_at = now
            # Recurse once with updated state
            return self._select_card_for_phase(entry)

        elif state.phase == PHASE_ACQUIRING:
            # Gate test: recognition
            self._current_card_type = "recognition"
            candidates = skills.get(SKILL_RECOGNITION, [])
            if not candidates:
                candidates = [v for vs in skills.values() for v in vs]
            variant = random.choice(candidates) if candidates else None
            return variant, "gate"

        elif state.phase == PHASE_MAINTENANCE:
            # Normal recognition review (or diff review)
            candidates = skills.get(SKILL_RECOGNITION, [])
            if candidates:
                self._current_card_type = "recognition"
            else:
                candidates = [v for vs in skills.values() for v in vs]
                self._current_card_type = "differentiation" if SKILL_DIFFERENTIATION in skills else "recognition"
            variant = random.choice(candidates) if candidates else None
            return variant, "review"

        elif state.phase == PHASE_REPAIRING:
            # Pop from repair_queue
            if entry.repair_queue:
                card_type = entry.repair_queue[0]
                self._current_card_type = card_type
                if card_type == "recognition":
                    candidates = skills.get(SKILL_RECOGNITION, [])
                else:
                    candidates = skills.get(SKILL_PRODUCTION, [])
                if not candidates:
                    candidates = [v for vs in skills.values() for v in vs]
                variant = random.choice(candidates) if candidates else None

                # Apply hint occlusion for occlusion cards
                return variant, "repair"
            # Empty repair queue — repair is complete, return to triggering phase
            if state.repair_strategy:
                strategy = get_repair_strategy(state.repair_strategy)
                state.half_life_secs = strategy.post_repair_H(
                    state.repair_state, state.half_life_secs)
            state.phase = state.return_phase or PHASE_MAINTENANCE
            state.repair_strategy = ""
            state.repair_state = {}
            state.return_phase = ""
            entry.repair_queue = []
            # Recurse once with updated state
            return self._select_card_for_phase(entry)

        return None, "review"

    # ── SRSProvider interface ────────────────────────────────────────

    def is_session_complete(self) -> bool:
        return (not self._learning_queue and not self._review_queue
                and not self._new_pool and not self._repair_pool and not self._repair_active)

    def seconds_until_next_due(self) -> float:
        """Seconds until the next entry becomes due across all queues."""
        self._fill_repair_active()
        all_entries = self._learning_queue + self._review_queue + self._repair_active
        if not all_entries:
            return 0
        now = srs_db.now_utc()
        nearest = min(e.due_at for e in all_entries)
        gap = (nearest - now).total_seconds()
        return max(0.0, gap)

    def session_counts(self) -> SessionCounts:
        now = srs_db.now_utc()
        due_count = 0
        locked_count = 0
        due_maintenance = 0
        due_acquiring = 0
        due_repair = 0
        for e in self._review_queue:
            if e.due_at <= now:
                due_count += 1
                if e.state:
                    if e.state.phase == PHASE_MAINTENANCE:
                        due_maintenance += 1
                    elif e.state.phase in (PHASE_ACQUIRING, PHASE_DAY0):
                        due_acquiring += 1
                    elif e.state.phase == PHASE_REPAIRING:
                        due_repair += 1
                    else:
                        due_maintenance += 1  # fallback
                else:
                    due_maintenance += 1  # legacy
        # Also count learning queue entries as due (Day 0 intros)
        for e in self._learning_queue:
            if e.due_at <= now:
                due_count += 1
                due_acquiring += 1
        # Count due repair_active entries (some may have timers pending)
        for e in self._repair_active:
            if e.due_at <= now:
                due_count += 1
                due_repair += 1
        # Count pool entries waiting to be activated
        due_count += len(self._repair_pool)
        due_repair += len(self._repair_pool)
        return SessionCounts(
            unseen=len(self._new_pool),
            learning=len(self._learning_queue),
            learning_capacity=self._queue_size,
            reviews_due=due_count,
            reviews_locked=locked_count,
            due_maintenance=due_maintenance,
            due_acquiring=due_acquiring,
            due_repair=due_repair,
            target_done=self._target_graduated,
            target_total=self._initial_target_count,
            total_graded=self._total_graded,
            correct_count=self._correct_count,
        )

    def set_queue_size(self, queue_size: int) -> None:
        """Update the learning queue capacity. Immediately fills new slots if available."""
        self._queue_size = queue_size
        self._fill_learning_queue()

    def add_new_words(self, count: int) -> int:
        """Load `count` additional unseen words into the pool. Returns number actually added."""
        existing_lexemes = {e.lexeme for e in self._review_queue}
        existing_lexemes |= {e.lexeme for e in self._learning_queue}
        existing_lexemes |= {e.lexeme for e in self._new_pool}

        _, unseen_groups = srs_db.classify_lexeme_groups(self._session)
        candidates = {k: v for k, v in unseen_groups.items() if k not in existing_lexemes}

        # honor any pinned intro order; falls back to LIFO for unranked words
        ordered = [lex for lex in sorted_unseen_lexemes(unseen_groups) if lex in candidates]
        added = 0
        for lex in ordered[:count]:
            items = candidates[lex]["items"]
            by_skill = _build_skill_map(items)
            canonical = _canonical_item(items)
            new_entry = LexemeQueueEntry(
                lexeme=lex,
                variants_by_skill=by_skill,
                strategy=LearningStrategy.INCREMENTAL_PRODUCTION,
                canonical_item_id=canonical.id if canonical else None,
                hanja=_extract_hanja(items),
            )
            new_entry.state = make_initial_lexeme_state()
            self._new_pool.append(new_entry)
            added += 1

        self._initial_target_count += added
        self._fill_learning_queue()
        return added

    def _repair_failed_at(self, entry: LexemeQueueEntry) -> dt.datetime | None:
        """Timestamp of the failure that triggered this entry's repair."""
        if not entry.state or not entry.state.repair_state:
            return None
        raw = entry.state.repair_state.get("failed_at")
        if raw:
            try:
                parsed = dt.datetime.fromisoformat(raw)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=srs_db.UTC)
                return parsed
            except ValueError:
                pass
        # Pre-failed_at repairs: last_reviewed_at is set at failure time and
        # never touched during repair, so it's an accurate fallback.
        return entry.state.last_reviewed_at

    def _repair_is_stale(self, entry: LexemeQueueEntry, now: dt.datetime) -> bool:
        """True if the failure happened more than REPAIR_STALE_SECS ago."""
        failed_at = self._repair_failed_at(entry)
        if failed_at is None:
            return True  # unknown age — likely carried over from a past session
        return (now - failed_at).total_seconds() > REPAIR_STALE_SECS

    def next_due_item(self) -> CardWrapper | None:
        now = srs_db.now_utc()
        self._fill_repair_active(now)

        # Priority: stale repairs (failed >1h ago, e.g. carried over from an
        # earlier session) come first; fresh repairs stay deferred behind the
        # normal review backlog so a just-missed word isn't re-shown
        # immediately. Intros last; also served while repair timers pend.
        due_review   = [e for e in self._review_queue   if e.due_at <= now]
        due_repair   = [e for e in self._repair_active  if e.due_at <= now]
        due_learning = [e for e in self._learning_queue  if e.due_at <= now]

        stale_repair = [e for e in due_repair if self._repair_is_stale(e, now)]

        if stale_repair:
            self.in_repair_mode = False
            entry = min(stale_repair, key=lambda e: e.due_at)
        elif due_review:
            self.in_repair_mode = False
            entry = min(due_review, key=lambda e: e.due_at)
        elif due_repair:
            self.in_repair_mode = True
            entry = min(due_repair, key=lambda e: e.due_at)
        elif due_learning:
            self.in_repair_mode = False
            entry = min(due_learning, key=lambda e: e.due_at)
        else:
            return None
        self._current_entry = entry

        difficulty = self._difficulty_by_lexeme.get(entry.lexeme, 0.0)

        # ── INCREMENTAL_PRODUCTION: phase-based card selection ──
        if entry.strategy == LearningStrategy.INCREMENTAL_PRODUCTION and entry.state:
            variant, review_mode = self._select_card_for_phase(entry)
            if variant is None:
                return None
            self._current_variant = variant

            _log_phase_state("CARD", entry.lexeme, entry.state, now)

            # Apply hint occlusion for occlusion card types
            front_override = None
            if self._current_card_type == "occlusion":
                occluded = _apply_hint_occlusion(variant.front, entry.lexeme)
                if occluded is not None:
                    front_override = occluded

            return self._variant_to_card(variant, review_mode=review_mode, difficulty=difficulty, front_override=front_override, hanja=entry.hanja, headword=srs_db.headword_of(entry.lexeme))

        # ── Legacy strategies ──
        # learning entry that still needs intro
        if not entry.intro_done:
            variant = entry.intro_queue[0]
            self._current_variant = variant
            return self._variant_to_card(variant, review_mode="intro", difficulty=difficulty, hanja=entry.hanja, headword=srs_db.headword_of(entry.lexeme))

        # normal review (learning or review queue)
        variant = self._select_variant_legacy(entry)
        if variant is None:
            return None

        self._current_variant = variant
        return self._variant_to_card(variant, review_mode="review", difficulty=difficulty, hanja=entry.hanja, headword=srs_db.headword_of(entry.lexeme))

    def check_answer(self, item: object, answer: str) -> bool:
        return False

    def submit_intro(self, item: object) -> None:
        if self._current_entry is None:
            return
        entry = self._current_entry
        variant = self._current_variant
        now = srs_db.now_utc()

        # ── INCREMENTAL_PRODUCTION Day 0: process intro as Day 0 card ──
        if entry.strategy == LearningStrategy.INCREMENTAL_PRODUCTION and entry.state:
            state = entry.state

            # Capture phase before mutation for logging
            phase_before = state.phase

            # Pop from day0_queue
            if entry.day0_queue:
                entry.day0_queue.pop(0)

            if not entry.day0_queue:
                # Current touch is done
                if state.day0_step == DAY0_TOUCH_A:
                    # NEW: Touch A done → straight to maintenance
                    # (process_day0_touch_a_complete handles the transition).
                    process_day0_touch_a_complete(state, now)
                elif state.day0_step == DAY0_TOUCH_B:
                    process_day0_touch_b_complete(state, now)

                entry.intro_done = True  # No more intro-style cards for now

            new_due = acquisition_compute_due_at(state, now)
            entry.due_at = new_due

            if variant is not None:
                self._persist_review(variant, mode="intro", new_due_at=new_due, entry=entry,
                                     phase_override=phase_before)

            # After Touch A, move to review queue (frees learning slot).
            # Phase stays DAY0 until Touch B completes.
            if state.day0_step in (DAY0_TOUCH_B, DAY0_DONE) and entry in self._learning_queue:
                self._learning_queue.remove(entry)
                self._review_queue.append(entry)
                self._target_graduated += 1
                self._fill_learning_queue()

            self._current_entry = None
            self._current_variant = None
            self._current_card_type = None
            return

        # ── Legacy intro handling ──
        new_due = now + INTRO_COOLDOWN

        if entry.intro_queue:
            entry.intro_queue.pop(0)
        if not entry.intro_queue:
            entry.intro_done = True
        entry.due_at = new_due

        if variant is not None:
            self._persist_review(variant, mode="intro", new_due_at=new_due, entry=entry)

        self._current_entry = None
        self._current_variant = None
        self._current_card_type = None

    def submit_review(self, item: object, grade: Grade) -> None:
        if self._current_entry is None:
            return

        entry = self._current_entry
        variant = self._current_variant
        entry.total_reviews += 1
        is_learning = entry in self._learning_queue

        # ── INCREMENTAL_PRODUCTION: four-way phase dispatch ──
        if entry.strategy == LearningStrategy.INCREMENTAL_PRODUCTION and entry.state:
            state = entry.state
            now = srs_db.now_utc()

            # Capture phase and H before handler dispatch (handlers mutate state)
            phase_before = state.phase
            H_before = state.half_life_secs if state.phase == PHASE_MAINTENANCE else None

            # Accuracy counts ONLY genuine maintenance reviews — not repair
            # re-tests, Day-0 intros, or acquisition gates. A maintenance review
            # that fails still counts here (as incorrect); its subsequent repair
            # re-tests do not.
            if phase_before == PHASE_MAINTENANCE:
                self._total_graded += 1
                if grade.is_correct:
                    self._correct_count += 1

            if state.phase == PHASE_DAY0:
                self._handle_day0_review(entry, grade, now)

            elif state.phase == PHASE_ACQUIRING:
                self._handle_gate_review(entry, grade, now)

            elif state.phase == PHASE_MAINTENANCE:
                self._handle_maintenance_review(entry, grade, now)

            elif state.phase == PHASE_REPAIRING:
                self._handle_repair_review(entry, grade, now)

            # Persist (pass phase_before so the log reflects the pre-mutation phase)
            # Repair handler sets due_at directly with timing delays; don't overwrite
            if state.phase == PHASE_REPAIRING:
                new_due = entry.due_at
            else:
                new_due = acquisition_compute_due_at(state, now)
                entry.due_at = new_due

            if variant is not None:
                self._persist_review(variant, mode="review", grade=grade, new_due_at=new_due, entry=entry,
                                     H_before=H_before, phase_override=phase_before)

            _log_phase_state("AFTER", entry.lexeme, state, now)

            # Move from learning to review queue after first graded review
            if is_learning:
                self._learning_queue.remove(entry)
                self._review_queue.append(entry)
                self._target_graduated += 1
                self._fill_learning_queue()

            self._current_entry = None
            self._current_variant = None
            self._current_card_type = None
            return

        # ── Legacy strategies: streak-based scheduling ──
        self._total_graded += 1
        if grade.is_correct:
            self._correct_count += 1
        self._handle_legacy_review(entry, grade, is_learning)
        self._current_entry = None
        self._current_variant = None
        self._current_card_type = None

    # ── Phase-specific review handlers ───────────────────────────────

    def _handle_day0_review(self, entry: LexemeQueueEntry, grade: Grade, now: dt.datetime) -> None:
        """Handle a graded review during Day 0 (Touch B gets graded).

        If a Touch B card is failed, inject a mini R→O→P repair into the
        day0_queue before completing Touch B, since Day 0 memory is fragile.
        """
        state = entry.state

        # Pop from day0_queue
        if entry.day0_queue:
            entry.day0_queue.pop(0)

        # Touch B failure: inject repair cards before remaining Touch B cards
        if state.day0_step == DAY0_TOUCH_B and not grade.is_correct:
            repair = ["recognition"]
            entry.day0_queue = repair + entry.day0_queue
            print(f"        [DAY0] {entry.lexeme} / touch_b FAIL / {grade.label} -> repair {repair}")
            self.last_review_log = f"{entry.lexeme}: day0 touch_b fail · repair"
            return

        if not entry.day0_queue:
            if state.day0_step == DAY0_TOUCH_A:
                process_day0_touch_a_complete(state, now)
                entry.day0_queue = day0_touch_b_cards()
            elif state.day0_step == DAY0_TOUCH_B:
                process_day0_touch_b_complete(state, now)

        print(f"        [DAY0] {entry.lexeme} / {state.day0_step} / {grade.label}")
        self.last_review_log = f"{entry.lexeme}: day0 {state.day0_step}"

    def _handle_gate_review(self, entry: LexemeQueueEntry, grade: Grade, now: dt.datetime) -> None:
        """Handle an acquiring-phase gate review."""
        state = entry.state
        old_passes = state.passes_24h

        result = process_gate_review(state, grade.value, now)

        print(f"        [GATE] {entry.lexeme} / {grade.label}")
        print(f"                passes: {old_passes} -> {state.passes_24h}")

        if result["trigger_repair"]:
            # Enter repair
            state.phase = PHASE_REPAIRING
            state.return_phase = PHASE_ACQUIRING
            strategy = get_repair_strategy("scaffolded")
            state.repair_strategy = strategy.name
            state.repair_state = strategy.init_repair(grade.value)
            state.repair_state["failed_at"] = now.isoformat()
            # Build repair card queue
            repair_cards = []
            temp_state = dict(state.repair_state)
            card_type = strategy.next_card_type(temp_state)
            while card_type is not None:
                repair_cards.append(card_type)
                temp_state = strategy.process_review(temp_state, 1)
                card_type = strategy.next_card_type(temp_state)
            entry.repair_queue = repair_cards
            # Route to repair queues
            if entry in self._review_queue:
                self._review_queue.remove(entry)
            if len(self._repair_active) < self._queue_size:
                entry.due_at = now
                self._repair_active.append(entry)
            else:
                self._repair_pool.append(entry)
            print(f"                -> REPAIR triggered (grade {grade.value}), "
                  f"cards: {entry.repair_queue}")
            self.last_review_log = f"{entry.lexeme}: gate {state.passes_24h}/3 · repair"

        elif result["graduated"]:
            graduate_to_maintenance(state, now)
            print(f"                -> GRADUATED to maintenance! H={_format_half_life(state.half_life_secs)}")
            self.last_review_log = f"{entry.lexeme}: graduated!"

        else:
            req = 4 if state.is_hard else 3
            self.last_review_log = f"{entry.lexeme}: gate {state.passes_24h}/{req}"

    def _handle_maintenance_review(self, entry: LexemeQueueEntry, grade: Grade, now: dt.datetime) -> None:
        """Handle a maintenance-phase review using the half-life model."""
        state = entry.state
        old_H = state.half_life_secs

        # Compute delta_t
        delta_t = (now - state.last_reviewed_at).total_seconds() if state.last_reviewed_at else 120.0
        p_before = recall_probability(old_H, delta_t)

        # Update half-life
        state.half_life_secs = update_half_life(old_H, delta_t, grade.value, skill="recognition")
        state.last_reviewed_at = now

        new_H = state.half_life_secs
        w = GRADE_WEIGHT.get(grade.value, 0.5)
        eta = effective_eta(old_H)
        p_star = TARGET_RECALL.get("recognition", 0.80)

        if grade.is_correct:
            delta_logH = eta * w
        else:
            odds = min(p_before / (1.0 - p_before + 1e-10), MAX_ODDS)
            delta_logH = -eta * w * odds

        old_ivl = next_interval(old_H, "recognition")
        new_ivl = next_interval(new_H, "recognition")
        new_due = now + dt.timedelta(seconds=new_ivl)
        wait = new_ivl
        abs_time = new_due.astimezone().strftime("%H:%M:%S")

        print(f"        [MAINT] {entry.lexeme} / recognition / {grade.label}")
        print(f"                dt={_format_half_life(delta_t)}  "
              f"p(recall)={p_before:.3f}  target={p_star:.2f}")
        print(f"                w={w}  dlogH={delta_logH:+.3f}")
        print(f"                H: {_format_half_life(old_H)} -> {_format_half_life(new_H)}  "
              f"interval: {_format_half_life(old_ivl)} -> {_format_half_life(new_ivl)}")
        print(f"                next due: +{_format_half_life(wait)}  {abs_time}")

        # Build one-line summary for UI
        suffix = ""

        # Retire as "learned" if the word was already at a ~1 year interval
        # (old_H at cap) and the user passed it — they've waited the full year
        from incremental_model import MAX_HALF_LIFE
        if grade.is_correct and old_H >= MAX_HALF_LIFE:
            state.phase = PHASE_LEARNED
            self._retire_learned(entry)
            print(f"                -> LEARNED! Word retired (passed 1-year interval)")
            self.last_review_log = f"{entry.lexeme}: learned!"
            return

        # Trigger repair on wrong answers → route to repair queues
        # (skip repair for diff cards — just reschedule sooner)
        is_diff = SKILL_DIFFERENTIATION in entry.variants_by_skill
        if not grade.is_correct and not is_diff:
            state.phase = PHASE_REPAIRING
            state.return_phase = PHASE_MAINTENANCE
            strategy = get_repair_strategy("scaffolded")
            state.repair_strategy = strategy.name
            state.repair_state = strategy.init_repair(grade.value)
            state.repair_state["failed_at"] = now.isoformat()
            # Build repair card queue
            repair_cards = []
            temp_state = dict(state.repair_state)
            card_type = strategy.next_card_type(temp_state)
            while card_type is not None:
                repair_cards.append(card_type)
                temp_state = strategy.process_review(temp_state, 1)
                card_type = strategy.next_card_type(temp_state)
            entry.repair_queue = repair_cards
            # Route to repair queues. The repair card is given a short
            # delay (REPAIR_RETRY_DELAY) so other due reviews come between
            # the failure and the rebuild card — otherwise the user would
            # see the same word twice in a row whenever no other review
            # happened to be due at that moment.
            if entry in self._review_queue:
                self._review_queue.remove(entry)
            entry.due_at = now + dt.timedelta(seconds=REPAIR_RETRY_DELAY)
            if len(self._repair_active) < self._queue_size:
                self._repair_active.append(entry)
            else:
                self._repair_pool.append(entry)
            print(f"                -> REPAIR queued, cards: {entry.repair_queue}")
            suffix = " · repair queued"

        self.last_review_log = f"{entry.lexeme}: {_format_half_life(new_ivl)}{suffix}"

    def _handle_repair_review(self, entry: LexemeQueueEntry, grade: Grade, now: dt.datetime) -> None:
        """Handle a repair-phase review with spaced timing.

        Timing rules:
        - Wrong answer: repeat same card after REPAIR_RETRY_DELAY (90s)
        - Correct non-final: advance immediately
        - Correct penultimate (next is final): delay REPAIR_FINAL_DELAY (5min) before final card
        - Correct final: repair complete
        """
        state = entry.state
        strategy = get_repair_strategy(state.repair_strategy)
        state.repair_state = strategy.process_review(state.repair_state, grade.value)

        print(f"        [REPAIR] {entry.lexeme} / {self._current_card_type} / {grade.label}")
        print(f"                 step {state.repair_state.get('step', 0)} of "
              f"{len(state.repair_state.get('sequence', []))}")

        if not grade.is_correct:
            # Wrong — repeat same card after retry delay
            # (process_review didn't advance step since grade > 3)
            entry.due_at = now + dt.timedelta(seconds=REPAIR_RETRY_DELAY)
            print(f"                 -> retry in {REPAIR_RETRY_DELAY}s")
            self.last_review_log = f"{entry.lexeme}: repair retry in {REPAIR_RETRY_DELAY}s"
            return

        # Correct — pop the completed card from repair_queue
        if entry.repair_queue:
            entry.repair_queue.pop(0)

        if strategy.is_complete(state.repair_state):
            # Repair done — return to original phase
            state.half_life_secs = strategy.post_repair_H(state.repair_state, state.half_life_secs)
            state.phase = state.return_phase
            entry.repair_queue = []
            state.repair_strategy = ""
            state.repair_state = {}
            state.return_phase = ""
            print(f"                 -> REPAIR complete, returning to {state.phase}")

            # Move completed repair back to review queue
            if entry in self._repair_active:
                self._repair_active.remove(entry)
                self._review_queue.append(entry)
                self._fill_repair_active(now)

            self.last_review_log = f"{entry.lexeme}: repair done"
        else:
            # More cards remain — check if next is the final card
            seq = state.repair_state.get("sequence", [])
            step = state.repair_state.get("step", 0)
            is_next_final = (step == len(seq) - 1)

            entry.due_at = now  # immediate next card
            self.last_review_log = f"{entry.lexeme}: repair {step}/{len(seq)}"

    def _retire_learned(self, entry: LexemeQueueEntry) -> None:
        """Retire a word as completely learned: remove from all queues and suspend items."""
        if entry in self._review_queue:
            self._review_queue.remove(entry)
        elif entry in self._repair_active:
            self._repair_active.remove(entry)
        elif entry in self._repair_pool:
            self._repair_pool.remove(entry)

        # Suspend all items in the lexeme so they won't load in future sessions
        all_item_ids = {v.item_id for vs in entry.variants_by_skill.values() for v in vs}
        for item_id in all_item_ids:
            item = self._session.get(srs_db.Item, item_id)
            if item is not None:
                item.suspended = True

        self._session.commit()

    def _handle_legacy_review(self, entry: LexemeQueueEntry, grade: Grade, is_learning: bool) -> None:
        """Handle review for legacy (non-incremental) strategies."""
        variant = self._current_variant
        prev_interval = entry.current_interval

        if grade.is_correct:
            entry.correct_streak += 1
            new_due = srs_db.now_utc() + entry.current_interval
            entry.due_at = new_due
            if grade == Grade.FLUENT_CORRECT:
                effective_mult = EASY_CORRECT_MULTIPLIER
            else:
                effective_mult = entry.multiplier * HARD_CORRECT_MULTIPLIER_FACTOR
            entry.current_interval *= effective_mult
        else:
            entry.correct_streak = 0
            entry.current_interval = entry.base_interval
            new_due = srs_db.now_utc() + entry.base_interval
            entry.due_at = new_due
            effective_mult = None

        wait = (new_due - srs_db.now_utc()).total_seconds()
        abs_time = new_due.astimezone().strftime("%H:%M:%S")
        print(f"        [GRADE] {entry.lexeme}  {grade.label}")
        print(f"                streak: {entry.correct_streak}  "
              f"strategy: {entry.strategy.name}")
        mult_label = f"x{effective_mult:.1f}" if effective_mult else "reset"
        print(f"                interval: {_format_half_life(prev_interval.total_seconds())} -> "
              f"{_format_half_life(entry.current_interval.total_seconds())}  ({mult_label})")
        print(f"                due: +{_format_half_life(wait)} ({abs_time})")

        if variant is not None:
            self._persist_review(variant, mode="review", grade=grade, new_due_at=new_due, entry=entry,
                                 prev_interval=prev_interval, new_interval=entry.current_interval)

        grad_ok = entry.correct_streak >= GRADUATION_STREAK

        if is_learning and grad_ok:
            self._learning_queue.remove(entry)
            self._review_queue.append(entry)
            self._target_graduated += 1
            self._fill_learning_queue()

    # ── variant management (edit, delete, add, quarantine) ───────────

    def edit_and_skip(self, item: object, new_front: str, new_back: str) -> None:
        """Update the variant text in memory and in the DB, then skip (no grade)."""
        variant = self._current_variant
        if variant is None:
            self._current_entry = None
            return

        variant.front = new_front
        variant.back = new_back

        db_item = self._session.get(srs_db.Item, variant.item_id)
        if db_item is not None:
            content = dict(db_item.content or {})
            variants_list = list(content.get("variants", []))
            if variant.variant_index < len(variants_list):
                variants_list[variant.variant_index] = dict(variants_list[variant.variant_index])
                variants_list[variant.variant_index]["front"] = new_front
                variants_list[variant.variant_index]["back"] = new_back
            content["variants"] = variants_list
            db_item.content = content
            self._session.commit()

        self._current_entry = None
        self._current_variant = None

    def delete_variant_and_skip(self, item: object) -> None:
        """Remove the current variant from the DB item and in-memory lists, then skip."""
        variant = self._current_variant
        entry = self._current_entry
        if variant is None:
            self._current_entry = None
            return

        db_item = self._session.get(srs_db.Item, variant.item_id)
        if db_item is not None:
            content = dict(db_item.content or {})
            variants_list = list(content.get("variants", []))
            if variant.variant_index < len(variants_list):
                variants_list.pop(variant.variant_index)
            content["variants"] = variants_list
            db_item.content = content
            self._session.commit()

        if entry is not None:
            skill_variants = entry.variants_by_skill.get(variant.skill, [])
            if variant in skill_variants:
                skill_variants.remove(variant)
            if variant in entry.intro_queue:
                entry.intro_queue.remove(variant)
                if not entry.intro_queue:
                    entry.intro_done = True
            for v in skill_variants:
                if v.item_id == variant.item_id and v.variant_index > variant.variant_index:
                    v.variant_index -= 1

        self._current_entry = None
        self._current_variant = None

    def quarantine_and_skip(self, item: object) -> None:
        """Suspend all DB items in the current lexeme and remove from queues."""
        entry = self._current_entry
        if entry is None:
            return

        all_item_ids = {v.item_id for vs in entry.variants_by_skill.values() for v in vs}
        for item_id in all_item_ids:
            db_item = self._session.get(srs_db.Item, item_id)
            if db_item is not None:
                db_item.suspended = True

        is_learning = entry in self._learning_queue
        if is_learning:
            self._learning_queue.remove(entry)
            self._initial_target_count = max(0, self._initial_target_count - 1)
        elif entry in self._review_queue:
            self._review_queue.remove(entry)
        elif entry in self._repair_active:
            self._repair_active.remove(entry)
        elif entry in self._repair_pool:
            self._repair_pool.remove(entry)

        self._session.commit()

        if is_learning:
            self._fill_learning_queue()

        self._current_entry = None
        self._current_variant = None

    def add_variant_and_skip(self, item: object, new_front: str, new_back: str) -> None:
        """Add a new variant to the current variant's DB item and in-memory lists, then skip."""
        variant = self._current_variant
        entry = self._current_entry
        if variant is None:
            self._current_entry = None
            return

        db_item = self._session.get(srs_db.Item, variant.item_id)
        new_index = 0
        if db_item is not None:
            content = dict(db_item.content or {})
            variants_list = list(content.get("variants", []))
            new_index = len(variants_list)
            variants_list.append({"front": new_front, "back": new_back})
            content["variants"] = variants_list
            db_item.content = content
            self._session.commit()

        if entry is not None:
            new_variant = Variant(
                item_id=variant.item_id,
                skill=variant.skill,
                variant_index=new_index,
                front=new_front,
                back=new_back,
            )
            entry.variants_by_skill.setdefault(variant.skill, []).append(new_variant)

        self._current_entry = None
        self._current_variant = None

    # ── variant selection for legacy strategies ──────────────────────

    def _select_variant_legacy(self, entry: LexemeQueueEntry) -> Variant | None:
        skills = entry.variants_by_skill

        if entry.strategy == LearningStrategy.PRODUCTION_ONLY:
            candidates = skills.get(SKILL_PRODUCTION, [])
        elif entry.strategy == LearningStrategy.RECOGNITION_ONLY:
            candidates = skills.get(SKILL_RECOGNITION, [])
        else:  # MIXED
            skill_key = random.choice([SKILL_PRODUCTION, SKILL_RECOGNITION])
            candidates = skills.get(skill_key, [])
            if not candidates:
                other = SKILL_RECOGNITION if skill_key == SKILL_PRODUCTION else SKILL_PRODUCTION
                candidates = skills.get(other, [])

        if not candidates:
            candidates = [v for vs in skills.values() for v in vs]

        if not candidates:
            return None

        return random.choice(candidates)
