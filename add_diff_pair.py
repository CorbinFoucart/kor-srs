#!/usr/bin/env python3
"""
add_diff_pair.py

Pipeline for adding differentiation card pairs: define both words -> generate diff bundle -> DB upsert.

Used by:
  - web_server.py (web UI endpoints)
"""

from __future__ import annotations

import logging
from typing import Tuple

from sqlalchemy.orm import Session

import llm_query
import srs_db
from add_word import upsert_static_bundle
from acquisition_model import (
    LexemeState,
    lexeme_state_to_dict,
    PHASE_MAINTENANCE,
    GRADUATION_H,
)

logger = logging.getLogger(__name__)


def diff_pair_exists(session: Session, word_a: str, word_b: str) -> bool:
    """Check if a diff pair already exists in either a~b or b~a ordering."""
    eid_ab = f"lexeme:{word_a}~{word_b}:diff_bundle"
    eid_ba = f"lexeme:{word_b}~{word_a}:diff_bundle"
    existing = (
        session.query(srs_db.Item)
        .filter(
            srs_db.Item.external_id.in_([eid_ab, eid_ba]),
            srs_db.Item.deleted_at.is_(None),
        )
        .first()
    )
    return existing is not None


def generate_and_insert_diff(
    session: Session,
    *,
    prompts: dict,
    word_a: str,
    def_block_a: str,
    word_b: str,
    def_block_b: str,
    n_variants: int = 10,
    model: str = llm_query.BUNDLE_MODEL,
) -> Tuple[bool, str]:
    """Generate a diff bundle and insert into DB with maintenance-ready SRS state.

    Returns (success, message).
    """
    # 1. Generate diff bundle via LLM
    try:
        payload = llm_query._run_diff_bundle(
            prompts,
            word_a=word_a,
            def_a=def_block_a,
            word_b=word_b,
            def_b=def_block_b,
            n_variants=n_variants,
            model=model,
        )
    except Exception as e:
        return (False, f"Diff bundle generation failed: {e}")

    # 2. Extract card, map correct_choice -> correct_word, rewrite fronts
    card = payload["cards"][0]
    for v in card["variants"]:
        v["correct_word"] = word_a if v["correct_choice"] == "a" else word_b
        # Rewrite front: English translation + Korean sentence with blank + choices
        eng = v.get("translation_en", "")
        korean_with_blank = v.get("front", "")
        # Strip any existing choice line the LLM appended (① ...)
        lines = korean_with_blank.split("\n")
        korean_line = "\n".join(l for l in lines if not l.strip().startswith("\u2460"))
        if eng:
            v["front"] = f"{eng}\n\n{korean_line.strip()}\n\n\u2460 {word_a}  \u2461 {word_b}"

    # 3. Build content with word_a/word_b metadata
    card["word_a"] = word_a
    card["word_b"] = word_b

    # 4. Upsert to DB
    external_id = f"lexeme:{word_a}~{word_b}:diff_bundle"
    item_id = upsert_static_bundle(
        session,
        external_id=external_id,
        card=card,
        tags=["static", "bundle", "diff", word_a, word_b],
    )

    # 5. Override SRSState with maintenance-ready state
    srs_state = session.get(srs_db.SRSState, item_id)
    now = srs_db.now_utc()
    now_iso = now.isoformat()

    if srs_state is not None:
        srs_state.last_reviewed_at = now
        srs_state.due_at = now
        srs_state.state = {
            "strategy": "incremental_production",
            "lexeme_state": lexeme_state_to_dict(LexemeState(
                phase=PHASE_MAINTENANCE,
                half_life_secs=GRADUATION_H,
                last_reviewed_at=now,
            )),
        }

    session.commit()
    return (True, f"Added diff pair '{word_a}~{word_b}' (id={item_id})")
