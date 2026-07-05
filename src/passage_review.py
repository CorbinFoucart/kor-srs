"""
passage_review.py

Phase 2 review mode: an LLM-generated 1-2 paragraph Korean passage that
embeds several of the user's due maintenance words. The user reads the
passage, taps any word they don't recognize. Each target word's
pass/fail (untapped/tapped) flows through the normal half-life SRS
machinery; tapped non-target words become add-word candidates.

This module is pure (no FastAPI), so it can be exercised standalone:

  - select_passage_targets(session, n) -> [{lexeme, item_id}]
  - generate_passage(target_lexemes, model) -> raw passage with markers
  - extract_target_spans(passage) -> {lexeme: [surface_forms]}
  - strip_markers(passage) -> passage with [[a|b]] -> a
  - apply_passage_grade(session, lexeme, item_id, grade, now) -> result dict

The passage carries `[[surface_form|dictionary_form]]` markers around each
target occurrence — surface_form is the inflected form actually used in the
sentence, dictionary_form is the input lexeme. Frontend strips the dict
half for display but keeps it on the span so a tap is attributable to the
right lexeme.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import re
from typing import Optional

from sqlalchemy.orm import Session

import llm_query
import srs_db
from acquisition_model import (
    PHASE_MAINTENANCE,
    PHASE_REPAIRING,
    get_repair_strategy,
    lexeme_state_from_dict,
    lexeme_state_to_dict,
)
from incremental_model import (
    next_interval,
    recall_probability,
    update_half_life,
)

logger = logging.getLogger(__name__)

MARKER_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")

# gpt-4.1 produces grammatically clean, cohesive Korean in one pass and is fast
# (~5s) — both better grammar AND lower latency than gpt-5-mini here, so no
# separate grammar-check pass is needed.
PASSAGE_MODEL = "gpt-4.1"


# ── target selection ────────────────────────────────────────────────────

def select_passage_targets(session: Session, n: int) -> list[dict]:
    """Pick up to N oldest-due maintenance-phase lexemes (recognition mode).

    Returns a list of {lexeme, item_id}. Only maintenance-phase words are
    used — day0/acquiring/repairing words have different review semantics
    and don't fit passage-grading.
    """
    if n <= 0:
        return []
    now = srs_db.now_utc()
    rows = (
        session.query(srs_db.Item, srs_db.SRSState)
        .join(srs_db.SRSState, srs_db.SRSState.item_id == srs_db.Item.id)
        .filter(srs_db.Item.external_id.like("%cloze_prod_bundle"))
        .filter(srs_db.Item.deleted_at.is_(None))
        .filter(srs_db.SRSState.due_at <= now)
        .order_by(srs_db.SRSState.due_at)
        .all()
    )
    targets: list[dict] = []
    for item, state in rows:
        try:
            blob = state.state if isinstance(state.state, dict) else json.loads(state.state)
        except (TypeError, json.JSONDecodeError):
            continue
        ls = (blob or {}).get("lexeme_state") or {}
        if ls.get("phase") != PHASE_MAINTENANCE:
            continue
        parts = item.external_id.split(":")
        if len(parts) < 3:
            continue
        targets.append({"lexeme": parts[1], "item_id": item.id})
        if len(targets) >= n:
            break
    return targets


# ── LLM passage generation ──────────────────────────────────────────────

def generate_passage(target_lexemes: list[str], model: str = PASSAGE_MODEL) -> str:
    """Ask the LLM for a 1-2 paragraph Korean passage that includes every
    target, with each target wrapped as `[[surface_form|dictionary_form]]`.
    """
    if not target_lexemes:
        return ""
    instruction = (
        "You are a Korean tutor who writes short, vivid STORIES for reading "
        "practice. The user gives a list of target Korean words/phrases in "
        "dictionary form. Write ONE cohesive short story in Korean (2-3 short "
        "paragraphs) in which EVERY target appears naturally.\n"
        "\n"
        "COHESION IS THE POINT (this is contextual reading practice): the story "
        "must have a single thread — one main character or situation with a "
        "clear beginning, middle, and end — and each target must arise "
        "ORGANICALLY from the plot, never shoehorned into an unrelated sentence. "
        "First choose a setting/scenario in which THESE particular words would "
        "plausibly occur together, then tell that little story. Sentences must "
        "connect and flow (cause and effect, time order, the same characters "
        "recurring) so the whole thing reads smoothly aloud as one narrative — "
        "not a list of separate sentences that each happen to contain a word.\n"
        "\n"
        "Requirements:\n"
        "- Grammar/complexity: LOW to MID TOPIK level 2 (한국어능력시험 2급). "
        "Common connectives and everyday grammar (-고, -아서/어서, -지만, -는데, "
        "-(으)면, -(으)니까, -(으)ㄹ 때, past/present/future). Short, clear "
        "sentences; no literary or advanced grammar; no rare vocabulary other "
        "than the targets themselves.\n"
        "- The Korean must be GRAMMATICALLY PERFECT and natural. Use ONE "
        "consistent speech style for narration (plain written style 해라체 — "
        "평서문 ending in -다/-ㄴ다/-았다 — is best for a story); any dialogue may "
        "use the speaker's appropriate level (반말/해요체) but must be correct. "
        "Use correct connectives to join clauses (-고, -아서/어서, -지만, -는데), "
        "NOT the reported-speech ending -다고 (e.g. '그 소식을 듣고' is correct; "
        "'듣다고' is wrong).\n"
        "- EVERY target appears at least once, conjugated/inflected as the "
        "sentence requires. You MAY reuse a target if it helps the story flow.\n"
        "- Wrap each target occurrence as [[surface_form|dictionary_form]] — "
        "surface_form is the form actually used (with conjugation/particles), "
        "dictionary_form is the target EXACTLY as given. Leave all other words "
        "unwrapped.\n"
        "- Output ONLY the Korean story with the [[ ]] markers — no title, no "
        "English, no preamble, no explanation, no bullet points, no code fences.\n"
        "\n"
        "Example:\n"
        "Targets: 미신, 들어서다, 약속\n"
        "Output: 우리 동네에는 오래된 [[미신|미신]]이 하나 있다. 봄이 "
        "[[들어서면|들어서다]] 사람들은 더 조심한다. 어느 봄날, 민수는 친구와 한 "
        "[[약속|약속]]을 깜빡 잊어버렸다. 그는 미신 때문에 나쁜 일이 생겼다고 "
        "걱정했지만, 사실은 그냥 잊은 것이었다. 그 뒤로 민수는 [[약속|약속]]을 꼭 "
        "메모하기로 했다."
    )
    targets_block = "\n".join(f"  - {t}" for t in target_lexemes)
    user = (
        f"Target words/phrases:\n{targets_block}\n\n"
        f"Write ONE cohesive 2-3 paragraph Korean short story that weaves in "
        f"EVERY target naturally (each wrapped as [[surface|dictionary]]). Pick a "
        f"scenario where these words fit together, and make it read as a single "
        f"flowing narrative — not separate sentences."
    )
    out = llm_query.query_api(user, instruction, model=model, verbose=False)
    return (out or "").strip()


def translate_passage(passage_plain: str, model: str = PASSAGE_MODEL) -> str:
    """Natural English translation of a (marker-stripped) Korean passage. Shown
    only after the learner's first read, for the second tapping pass."""
    if not (passage_plain or "").strip():
        return ""
    instruction = (
        "You are a Korean-to-English translator. Translate the Korean passage "
        "into natural, fluent English. Preserve sentence/paragraph breaks. "
        "Output ONLY the English translation — no preamble, notes, or romanization."
    )
    out = llm_query.query_api(passage_plain, instruction, model=model, verbose=False)
    return (out or "").strip()


def extract_target_spans(passage: str) -> dict[str, list[str]]:
    """Return {lexeme -> [surface_forms]} for every [[surface|lexeme]] marker."""
    spans: dict[str, list[str]] = {}
    for m in MARKER_RE.finditer(passage or ""):
        surface, lex = m.group(1), m.group(2)
        spans.setdefault(lex, []).append(surface)
    return spans


def strip_markers(passage: str) -> str:
    """Replace [[surface|dictionary]] with surface (for plain-text context)."""
    return MARKER_RE.sub(lambda m: m.group(1), passage or "")


# ── grade application (bypasses the live LexemeSRS) ─────────────────────

def apply_passage_grade(
    session: Session,
    lexeme: str,
    item_id: int,
    grade: int,
    now: Optional[dt.datetime] = None,
) -> dict:
    """Apply a passage-review grade (2=pass, 5=fail) to a lexeme's stored
    state. Runs through the same update_half_life + repair machinery as a
    normal sentence-card maintenance review, and writes a ReviewLog row
    tagged with source='passage_review'.

    Returns a result dict the frontend can display.
    """
    now = now or srs_db.now_utc()
    srs_state = session.get(srs_db.SRSState, item_id)
    if srs_state is None:
        return {"lexeme": lexeme, "ok": False, "error": "no srs_state"}

    # Deep-copy so reassigning srs_state.state is a NEW object — SRSState.state
    # is a plain JSON column (not MutableDict), so in-place mutation of the same
    # dict is NOT flagged dirty and would silently fail to persist.
    raw = (
        srs_state.state if isinstance(srs_state.state, dict)
        else json.loads(srs_state.state or "{}")
    )
    blob = copy.deepcopy(raw or {})
    ls = lexeme_state_from_dict((blob or {}).get("lexeme_state") or {})

    if ls.phase != PHASE_MAINTENANCE:
        return {"lexeme": lexeme, "ok": False,
                "error": f"phase={ls.phase}, not maintenance"}

    old_H = ls.half_life_secs
    delta_t = (
        (now - ls.last_reviewed_at).total_seconds()
        if ls.last_reviewed_at else 120.0
    )
    p_before = recall_probability(old_H, delta_t) if old_H > 0 else 0.0

    new_H = update_half_life(old_H, delta_t, grade, skill="recognition")
    ls.half_life_secs = new_H
    ls.last_reviewed_at = now

    correct = grade <= 3
    if correct:
        new_ivl = next_interval(new_H, "recognition")
        new_due = now + dt.timedelta(seconds=new_ivl)
    else:
        # fail → enter repair (single-card recognition re-test, post-occlusion)
        ls.return_phase = PHASE_MAINTENANCE
        ls.phase = PHASE_REPAIRING
        strategy = get_repair_strategy("scaffolded")
        ls.repair_strategy = strategy.name
        ls.repair_state = strategy.init_repair(grade)
        new_due = now  # immediately due for repair

    blob["lexeme_state"] = lexeme_state_to_dict(ls)
    srs_state.state = blob
    srs_state.last_reviewed_at = now
    srs_state.due_at = new_due

    payload = {
        "phase": ls.phase,  # post-mutation
        "card_type": "recognition",
        "incremental_skill": "recognition",
        "half_life_secs": new_H,
        "half_life_secs_before": old_H,
        "source": "passage_review",
    }
    session.add(srs_db.ReviewLog(
        item_id=item_id,
        reviewed_at=now,
        grade=grade,
        correct=correct,
        mode="review",
        payload=payload,
        new_due_at=new_due,
        new_scheduler_name=srs_state.scheduler_name,
        new_scheduler_version=srs_state.scheduler_version,
    ))

    return {
        "lexeme": lexeme,
        "ok": True,
        "passed": correct,
        "grade": grade,
        "old_H_days": round(old_H / 86400, 2),
        "new_H_days": round(new_H / 86400, 2),
        "delta_t_days": round(delta_t / 86400, 2),
        "p_before": round(p_before, 3),
        "new_due_at": new_due.isoformat(),
    }
