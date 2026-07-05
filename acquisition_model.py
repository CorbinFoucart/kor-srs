"""
acquisition_model.py

Two-phase learning model: Acquisition -> Maintenance, with Repair.

Phase model:
    NEW -> DAY0 -> ACQUIRING -> MAINTENANCE
                     |              |
                   REPAIR <------  REPAIR
                     |              |
                  (return)       (return)

Pure functions (no DB dependency), parallel to incremental_model.py.

Occlusion is a scaffolding tool, not a separately tracked skill.
Only recognition is scheduled; occlusion cards are used exclusively
during Day 0 and repair.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from incremental_model import (
    INITIAL_HALF_LIFE,
    MAX_HALF_LIFE,
    MIN_HALF_LIFE,
    TARGET_RECALL,
    next_interval,
    recall_probability,
    update_half_life,
)

# ── constants ──────────────────────────────────────────────────────────

# Graduation: half-life that gives an initial interval > 1 day
# With target_recall=0.80 for recognition: interval = -H * log2(0.80)
# We want interval >= 86400s (1 day), so H >= 86400 / 0.3219 ~ 268400s
GRADUATION_H = 270_000.0  # ~3.1 days, gives interval ~1.0 day at p*=0.80

GRADUATION_PASSES = 3       # passes_24h required to graduate
GRADUATION_PASSES_HARD = 4  # passes_24h required to graduate hard words

HARD_FAIL_STREAK = 3        # consecutive gate failures to mark as hard

# Seconds a freshly-failed repair waits before it's servable again (so other
# reviews come between the miss and the re-test). compute_due_at() and the live
# session (lexeme_srs) must agree on this, or the due count and the review
# page's countdown disagree during this window.
REPAIR_RETRY_DELAY = 90.0

# Day 0 Touch B delay range (seconds)
TOUCH_B_DELAY_MIN = 20 * 60   # 20 minutes
TOUCH_B_DELAY_MAX = 60 * 60   # 60 minutes

# Phases
PHASE_DAY0 = "day0"
PHASE_ACQUIRING = "acquiring"
PHASE_MAINTENANCE = "maintenance"
PHASE_REPAIRING = "repairing"
PHASE_LEARNED = "learned"

# Day 0 steps
DAY0_TOUCH_A = "touch_a"
DAY0_TOUCH_B = "touch_b"
DAY0_DONE = "done"


# ── LexemeState dataclass ──────────────────────────────────────────────

@dataclass
class LexemeState:
    """Per-lexeme state replacing multi-skill SkillState dict.

    Tracks a single lexeme through day0 -> acquiring -> maintenance,
    with optional repair detours.
    """
    # Phase: "day0", "acquiring", "maintenance", "repairing"
    phase: str = PHASE_DAY0

    # Maintenance fields
    half_life_secs: float = GRADUATION_H
    last_reviewed_at: dt.datetime | None = None

    # Acquisition fields (spec section 1.3)
    passes_24h: int = 0
    fail_streak: int = 0
    is_hard: bool = False
    next_gate_due: dt.datetime | None = None
    day0_step: str = DAY0_TOUCH_A  # "touch_a", "touch_b", "done"
    day0_touch_b_due: dt.datetime | None = None
    gate_done_today: bool = False
    remedial_done_today: bool = False

    # Repair fields
    repair_strategy: str = ""
    repair_state: dict = field(default_factory=dict)
    return_phase: str = ""  # phase to return to after repair

    # A/B test: intro mode used when this word was introduced
    intro_mode: str = ""  # "interleaved" or "sequential"


# ── RepairProtocol interface ───────────────────────────────────────────

class RepairProtocol(Protocol):
    """Interface for pluggable repair strategies."""
    name: str

    def init_repair(self, failure_grade: int) -> dict:
        """Initialize repair state for a given failure grade."""
        ...

    def next_card_type(self, state: dict) -> str | None:
        """Returns "recognition", "occlusion", "production", or None if done."""
        ...

    def process_review(self, state: dict, grade: int) -> dict:
        """Process a repair review and return updated state."""
        ...

    def is_complete(self, state: dict) -> bool:
        """Check if the repair sequence is complete."""
        ...

    def post_repair_H(self, state: dict, current_H: float) -> float:
        """Configurable H adjustment after repair completes."""
        ...


# ── ScaffoldedRepair ───────────────────────────────────────────────────

class ScaffoldedRepair:
    """Recognition-only re-test after a maintenance failure. With occlusion
    retired, repair degenerates to a single graded recognition re-test for
    every failure grade — pass and you return to maintenance; fail and you
    stay in repair until you pass.
    """
    name = "scaffolded"

    def init_repair(self, failure_grade: int) -> dict:
        # All failure grades: a single graded recognition re-test.
        sequence = ["recognition"]
        return {
            "failure_grade": failure_grade,
            "sequence": sequence,
            "step": 0,
            "grades": [],
        }

    def next_card_type(self, state: dict) -> str | None:
        seq = state.get("sequence", [])
        step = state.get("step", 0)
        if step >= len(seq):
            return None
        return seq[step]

    def process_review(self, state: dict, grade: int) -> dict:
        state = dict(state)
        state["grades"] = list(state.get("grades", []))
        state["grades"].append(grade)
        if grade <= 3:  # correct — advance to next card
            state["step"] = state.get("step", 0) + 1
        return state

    def is_complete(self, state: dict) -> bool:
        seq = state.get("sequence", [])
        step = state.get("step", 0)
        return step >= len(seq)

    def post_repair_H(self, state: dict, current_H: float) -> float:
        """Default: keep current H (the maintenance model already adjusted H
        on the failure that triggered repair)."""
        return current_H


# ── Repair strategy registry ──────────────────────────────────────────

_REPAIR_STRATEGIES: dict[str, RepairProtocol] = {
    "scaffolded": ScaffoldedRepair(),
}


def get_repair_strategy(name: str = "scaffolded") -> RepairProtocol:
    """Look up a repair strategy by name."""
    return _REPAIR_STRATEGIES.get(name, ScaffoldedRepair())


def register_repair_strategy(strategy: RepairProtocol) -> None:
    """Register a custom repair strategy for A/B testing."""
    _REPAIR_STRATEGIES[strategy.name] = strategy


# ── Day 0 functions ───────────────────────────────────────────────────

def day0_touch_a_cards(n_examples: int = 5) -> list[str]:
    """Card types for Touch A: recognition intros across multiple example
    sentences (occlusion scaffolding has been retired in the recognition-only
    pivot — every card is now a recognition intro)."""
    return ["recognition"] * (2 * n_examples)


def day0_touch_b_cards() -> list[str]:
    """Card types for Touch B: a single graded recognition card."""
    return ["recognition"]


def process_day0_touch_a_complete(state: LexemeState, now: dt.datetime) -> LexemeState:
    """Called when all Touch A intro cards are exhausted.

    Skips Touch B and the acquiring/gate phase entirely — the word becomes
    a regular maintenance-phase review with half-life GRADUATION_H (~3.1d),
    so the first scheduled recognition review is ~1 day out. This is the
    simplified post-recognition-pivot path; Touch B / acquiring machinery
    is retained only for backward compatibility with words that were
    mid-pipeline before the change.
    """
    state.day0_step = DAY0_DONE
    state.day0_touch_b_due = None
    state.phase = PHASE_MAINTENANCE
    state.half_life_secs = GRADUATION_H
    state.last_reviewed_at = now
    return state


def process_day0_touch_b_complete(state: LexemeState, now: dt.datetime) -> LexemeState:
    """LEGACY: only invoked for words that were already mid-Touch-B at the
    time of the Day-0 simplification. They finish the old gate path."""
    state.day0_step = DAY0_DONE
    state.phase = PHASE_ACQUIRING
    state.next_gate_due = now + dt.timedelta(hours=24)
    state.passes_24h = 0
    return state


# ── Gate functions ────────────────────────────────────────────────────

def process_gate_review(
    state: LexemeState, grade: int, now: dt.datetime,
) -> dict:
    """Process a daily gate review. Returns result dict with action flags.

    Returns:
        {
            "trigger_repair": bool,
            "graduated": bool,
            "repair_grade": int | None,
        }
    """
    result = {"trigger_repair": False, "graduated": False, "repair_grade": None}
    is_correct = grade in (1, 2, 3)

    if is_correct:
        # Gate passes
        state.passes_24h += 1
        state.fail_streak = 0
        state.gate_done_today = True

        # Check graduation
        required = GRADUATION_PASSES_HARD if state.is_hard else GRADUATION_PASSES
        if state.passes_24h >= required:
            result["graduated"] = True
    else:
        # Gate fails — apply regression
        if grade == 4:
            # easy wrong: decrement by 1
            state.passes_24h = max(state.passes_24h - 1, 0)
        else:
            # grades 5, 6: full reset
            state.passes_24h = 0

        state.fail_streak += 1
        state.gate_done_today = True

        # Check hard-word flag
        if state.fail_streak >= HARD_FAIL_STREAK:
            state.is_hard = True

        # Trigger remedial repair (only if not already done today)
        if not state.remedial_done_today:
            result["trigger_repair"] = True
            result["repair_grade"] = grade
            state.remedial_done_today = True

    # Schedule next gate
    state.next_gate_due = now + dt.timedelta(hours=24)

    return result


def graduate_to_maintenance(state: LexemeState, now: dt.datetime) -> LexemeState:
    """Transition from acquiring to maintenance phase."""
    state.phase = PHASE_MAINTENANCE
    state.half_life_secs = GRADUATION_H
    state.last_reviewed_at = now
    return state


# ── Maintenance helper ────────────────────────────────────────────────

def compute_due_at(state: LexemeState, now: dt.datetime) -> dt.datetime:
    """Compute next due time based on current phase."""
    if state.phase == PHASE_DAY0:
        if state.day0_step == DAY0_TOUCH_A:
            return now  # Touch A is immediate
        elif state.day0_step == DAY0_TOUCH_B:
            if state.day0_touch_b_due is not None:
                return state.day0_touch_b_due
            return now
        else:
            # day0 done, waiting for gate
            if state.next_gate_due is not None:
                return state.next_gate_due
            return now + dt.timedelta(hours=24)

    elif state.phase == PHASE_ACQUIRING:
        if state.next_gate_due is not None:
            return state.next_gate_due
        return now + dt.timedelta(hours=24)

    elif state.phase == PHASE_MAINTENANCE:
        if state.last_reviewed_at is None:
            return now
        interval = next_interval(state.half_life_secs, "recognition")
        return state.last_reviewed_at + dt.timedelta(seconds=interval)

    elif state.phase == PHASE_REPAIRING:
        # A freshly-failed repair only becomes servable REPAIR_RETRY_DELAY after
        # the failure (the live session sets due_at = fail + delay). Mirror that
        # here so the due count doesn't show it as due while the review page is
        # still counting down to the retry.
        raw = (state.repair_state or {}).get("failed_at")
        if raw:
            try:
                failed = dt.datetime.fromisoformat(raw)
                if failed.tzinfo is None:
                    failed = failed.replace(tzinfo=dt.timezone.utc)
                return failed + dt.timedelta(seconds=REPAIR_RETRY_DELAY)
            except (ValueError, TypeError):
                pass
        return now

    elif state.phase == PHASE_LEARNED:
        # Learned words are never due; return far future
        return now + dt.timedelta(days=36500)

    return now


# ── Serialization ─────────────────────────────────────────────────────

def lexeme_state_to_dict(state: LexemeState) -> dict:
    """Serialize LexemeState for SRSState.state JSON."""
    return {
        "phase": state.phase,
        "half_life_secs": state.half_life_secs,
        "last_reviewed_at": state.last_reviewed_at.isoformat() if state.last_reviewed_at else None,
        "passes_24h": state.passes_24h,
        "fail_streak": state.fail_streak,
        "is_hard": state.is_hard,
        "next_gate_due": state.next_gate_due.isoformat() if state.next_gate_due else None,
        "day0_step": state.day0_step,
        "day0_touch_b_due": state.day0_touch_b_due.isoformat() if state.day0_touch_b_due else None,
        "gate_done_today": state.gate_done_today,
        "remedial_done_today": state.remedial_done_today,
        "repair_strategy": state.repair_strategy,
        "repair_state": state.repair_state,
        "return_phase": state.return_phase,
        "intro_mode": state.intro_mode,
    }


def _parse_dt(raw: str | None) -> dt.datetime | None:
    """Parse an ISO datetime string, ensuring UTC timezone."""
    if raw is None:
        return None
    parsed = dt.datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def lexeme_state_from_dict(data: dict) -> LexemeState:
    """Deserialize LexemeState from SRSState.state JSON.

    Handles both new format (has "phase" key) and old multi-skill format
    (has "recognition"/"occlusion"/"production" keys).
    """
    # Detect old multi-skill format
    if "phase" not in data and ("recognition" in data or "skill_states" in data):
        return _migrate_from_old_format(data)

    return LexemeState(
        phase=data.get("phase", PHASE_DAY0),
        half_life_secs=data.get("half_life_secs", GRADUATION_H),
        last_reviewed_at=_parse_dt(data.get("last_reviewed_at")),
        passes_24h=data.get("passes_24h", 0),
        fail_streak=data.get("fail_streak", 0),
        is_hard=data.get("is_hard", False),
        next_gate_due=_parse_dt(data.get("next_gate_due")),
        day0_step=data.get("day0_step", DAY0_TOUCH_A),
        day0_touch_b_due=_parse_dt(data.get("day0_touch_b_due")),
        gate_done_today=data.get("gate_done_today", False),
        remedial_done_today=data.get("remedial_done_today", False),
        repair_strategy=data.get("repair_strategy", ""),
        repair_state=data.get("repair_state", {}),
        return_phase=data.get("return_phase", ""),
        intro_mode=data.get("intro_mode", ""),
    )


# ── Migration from old format ─────────────────────────────────────────

def _migrate_from_old_format(data: dict) -> LexemeState:
    """Migrate old multi-skill JSON format to LexemeState.

    Policy:
    1. Recognition-only (occlusion not unlocked): phase=day0, restart acquisition
    2. Occlusion or production unlocked: phase=maintenance, preserve production H
       (use GRADUATION_H as minimum floor)
    """
    # Check if data has skill_states nested dict
    skill_data = data.get("skill_states", data)

    # Determine most advanced unlocked skill and gather H values
    occlusion_unlocked = False
    production_unlocked = False
    most_recent_reviewed = None
    production_H = None

    for skill_name in ("recognition", "occlusion", "production"):
        entry = skill_data.get(skill_name, {})
        if entry.get("unlocked", False):
            if skill_name == "occlusion":
                occlusion_unlocked = True
            elif skill_name == "production":
                production_unlocked = True

        if skill_name == "production" and entry.get("half_life_secs") is not None:
            production_H = entry["half_life_secs"]

        lr_raw = entry.get("last_reviewed_at")
        if lr_raw is not None:
            lr = _parse_dt(lr_raw)
            if lr is not None and (most_recent_reviewed is None or lr > most_recent_reviewed):
                most_recent_reviewed = lr

    if occlusion_unlocked or production_unlocked:
        # Policy 2: enter maintenance, preserving the old production H
        # Use GRADUATION_H as floor (don't start lower than a fresh graduate)
        H = max(production_H or GRADUATION_H, GRADUATION_H)
        return LexemeState(
            phase=PHASE_MAINTENANCE,
            half_life_secs=H,
            last_reviewed_at=most_recent_reviewed,
        )
    else:
        # Policy 1: recognition-only words skip to Touch B
        # (they've already seen the word via recognition reviews)
        return LexemeState(
            phase=PHASE_DAY0,
            day0_step=DAY0_TOUCH_B,
        )


def make_initial_lexeme_state() -> LexemeState:
    """Create a fresh LexemeState for a new word."""
    return LexemeState()


# ── Daily gate reset ──────────────────────────────────────────────────

def reset_daily_flags(state: LexemeState, now: dt.datetime) -> None:
    """Reset per-day flags only if a new gate window has arrived.

    If next_gate_due is in the past (or None), a new gate is allowed.
    Otherwise the word already gated in the current window — don't reset.
    """
    if state.next_gate_due is None or state.next_gate_due <= now:
        state.gate_done_today = False
        state.remedial_done_today = False
