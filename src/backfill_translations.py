#!/usr/bin/env python3
"""
backfill_translations.py

Async script to backfill English translations for card variants.
Adds a "translation_en" key to each variant dict in Item.content["variants"].
Follows the seed_static_cards.py concurrency pattern (asyncio + Semaphore).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sqlalchemy.orm import sessionmaker

import llm_query
import srs_db
from seed_static_cards import log_latency, run_with_queue_timing


MODEL = "gpt-5-mini"


def _build_prompt(front: str, back: str, skill_type: str) -> str:
    """Build the translation prompt for a single variant."""
    card_type = "production" if "prod" in skill_type else "recognition"
    return (
        f"The following is a {card_type} flashcard designed to test Korean "
        f"vocabulary in context. Please provide only the natural English "
        f"translation of the complete Korean sentence. Do not send any text "
        f"other than the translation.\n\n"
        f"{front}\n{back}"
    )


def translate_variant(front: str, back: str, skill_type: str) -> str:
    """Sync OpenAI call to translate one variant."""
    prompt = _build_prompt(front, back, skill_type)
    resp = llm_query.client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


async def translate_item(
    item_id: int,
    external_id: str,
    content: dict,
    *,
    sem: asyncio.Semaphore,
    SessionLocal,
    dry_run: bool,
    progress: dict,
) -> Tuple[str, bool, str]:
    """
    Translate all variants in one item. Returns (external_id, ok, message).
    """
    parsed = srs_db.parse_external_id(external_id)
    skill_type = parsed[1] if parsed else "unknown"

    variants = content.get("variants", [])
    if not variants:
        return (external_id, True, "no variants")

    # track per-variant results
    translations: Dict[int, str] = {}
    skipped = 0
    errors = 0

    for i, v in enumerate(variants):
        # skip if already translated
        existing = v.get("translation_en")
        if existing and existing.strip():
            skipped += 1
            progress["done"] += 1
            continue

        front = v.get("front", "")
        back = v.get("back", "")
        if not front and not back:
            skipped += 1
            progress["done"] += 1
            continue

        try:
            translation = await run_with_queue_timing(
                request_type="translate_variant",
                api_sem=sem,
                fn=translate_variant,
                fn_kwargs={"front": front, "back": back, "skill_type": skill_type},
                model=MODEL,
                log_extra={"external_id": external_id, "variant_index": i},
            )
            translations[i] = translation
        except Exception as e:
            errors += 1
            print(f"  [WARN] {external_id} variant {i}: {e}")

        progress["done"] += 1
        print(f"\r  {progress['done']}/{progress['total']} variants", end="", flush=True)

    if dry_run:
        for i, t in translations.items():
            print(f"  [DRY] {external_id} v{i}: {t}")
        msg = f"dry-run: {len(translations)} translated, {skipped} already done, {errors} errors"
        return (external_id, True, msg)

    if not translations and skipped == len(variants):
        return (external_id, True, f"all {skipped} variants already translated")

    # write to DB
    try:
        with SessionLocal() as sess:
            db_item = sess.get(srs_db.Item, item_id)
            if db_item is None:
                return (external_id, False, "item not found in DB")

            updated_content = copy.deepcopy(db_item.content or {})
            updated_variants = updated_content.get("variants", [])

            for i, t in translations.items():
                if i < len(updated_variants):
                    updated_variants[i]["translation_en"] = t

            updated_content["variants"] = updated_variants
            db_item.content = updated_content
            sess.commit()

        msg = f"{len(translations)} translated, {skipped} already done, {errors} errors"
        return (external_id, True, msg)

    except Exception as e:
        return (external_id, False, f"DB write failed: {e}")


async def main_async(args):
    engine = srs_db.make_engine(args.db)
    srs_db.init_db(engine)
    SessionLocal = sessionmaker(bind=engine, future=True)

    # load all card items
    with SessionLocal() as sess:
        items = sess.query(srs_db.Item).filter(
            srs_db.Item.item_type == "card",
            srs_db.Item.deleted_at.is_(None),
        ).all()

        # snapshot into plain data (detach from session)
        item_data = [
            (item.id, item.external_id, copy.deepcopy(item.content or {}))
            for item in items
        ]

    if not item_data:
        print("No card items found.")
        return

    # count total variants and already-translated
    total_variants = 0
    already_translated = 0
    for _, _, content in item_data:
        for v in content.get("variants", []):
            total_variants += 1
            if v.get("translation_en", "").strip():
                already_translated += 1

    print(f"Found {len(item_data)} items with {total_variants} total variants")
    print(f"  Already translated: {already_translated}")
    print(f"  Need translation:   {total_variants - already_translated}")
    if args.dry_run:
        print("  (DRY RUN — no DB writes)")
    print()

    sem = asyncio.Semaphore(args.max_inflight)
    progress = {"done": 0, "total": total_variants}

    tasks = [
        asyncio.create_task(
            translate_item(
                item_id, external_id, content,
                sem=sem,
                SessionLocal=SessionLocal,
                dry_run=args.dry_run,
                progress=progress,
            )
        )
        for item_id, external_id, content in item_data
    ]

    ok_count = 0
    err_count = 0
    for coro in asyncio.as_completed(tasks):
        eid, success, msg = await coro
        if success:
            ok_count += 1
            print(f"\n[OK]  {eid}: {msg}")
        else:
            err_count += 1
            print(f"\n[ERR] {eid}: {msg}")

    print(f"\nDone: {ok_count} OK, {err_count} errors out of {len(item_data)} items.")


def main():
    ap = argparse.ArgumentParser(description="Backfill English translations for card variants")
    ap.add_argument("--db", default="test_srs.sqlite", help="Path to SQLite DB")
    ap.add_argument("--max-inflight", type=int, default=5, help="Max concurrent OpenAI calls")
    ap.add_argument("--dry-run", action="store_true", help="Print translations without writing to DB")
    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
