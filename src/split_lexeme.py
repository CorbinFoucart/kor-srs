#!/usr/bin/env python3
"""
split_lexeme.py

Split one conflated lexeme (a homograph whose senses share a spelling but are
genuinely different words) into separate sense-tagged lexemes that schedule
independently.

Why: the half-life model tracks ONE H per lexeme. For 분기 = 分期 "quarter" /
分岐 "branch", perfect recall of one sense and zero recall of the other
averages to a meaningless ~50%. Splitting gives each sense its own H, so they
diverge correctly (the known sense climbs, the unknown one fails → repair).

What it does, per sense slug:
  - new recog bundle  lexeme:<headword>#<slug>:cloze_recog_bundle  with the
    variants whose recog `back` matches that sense
  - a fresh UNSEEN (Day-0) schedule — the sense is a word the learner was never
    introduced to *as that sense*, so it enters the new-word pipeline (Touch-A
    intros first, then maintenance). The old conflated H is discarded.
  - (optional, default on --apply) per-sense Hanja cached on content["hanja"],
    looked up WITH the sense gloss so 分期 vs 分岐 resolve correctly
The old recog + prod items are soft-deleted (ReviewLog history is preserved).

Production bundles are not recreated per sense — production cards are unused
substrate in the recognition-scheduled app, and _canonical_item falls back to
the recog item to hold per-lexeme state.

Usage:
  # dry run — show the partition, no writes, no LLM
  python split_lexeme.py --db test_srs.sqlite --lexeme 분기 \
      --sense quarter "quarter (three-month" \
      --sense branch  "branching/division"

  # apply (creates a DB backup first; pre-fetches per-sense hanja)
  python split_lexeme.py --db test_srs.sqlite --lexeme 분기 \
      --sense quarter "quarter (three-month" \
      --sense branch  "branching/division" --apply
"""

from __future__ import annotations

import argparse
import copy
import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

import acquisition_model as am
import srs_db
from add_word import upsert_static_bundle

SKILL_RECOG = "cloze_recog_bundle"
SKILL_PROD = "cloze_prod_bundle"


def _load_items(sess, lexeme: str):
    """Return (recog_item, prod_item|None) for a plain (untagged) lexeme."""
    items = sess.scalars(
        select(srs_db.Item)
        .where(srs_db.Item.external_id.like(f"lexeme:{lexeme}:%"))
        .where(srs_db.Item.deleted_at.is_(None))
    ).all()
    recog = prod = None
    for it in items:
        parsed = srs_db.parse_external_id(it.external_id)
        if not parsed:
            continue
        if parsed[1] == SKILL_RECOG:
            recog = it
        elif parsed[1] == SKILL_PROD:
            prod = it
    return recog, prod


def _partition(variants: list[dict], senses: list[tuple[str, str]]) -> dict[str, list[dict]]:
    """Assign each variant to exactly one sense by case-insensitive back match.
    Raises on any unmatched or multiply-matched variant."""
    buckets: dict[str, list[dict]] = {slug: [] for slug, _ in senses}
    for i, v in enumerate(variants):
        back = (v.get("back") or "").lower()
        hits = [slug for slug, sub in senses if sub.lower() in back]
        if len(hits) == 0:
            raise SystemExit(f"  variant v{i} back={v.get('back')!r} matched NO sense — fix --sense substrings")
        if len(hits) > 1:
            raise SystemExit(f"  variant v{i} back={v.get('back')!r} matched MULTIPLE senses {hits} — make substrings unambiguous")
        buckets[hits[0]].append(v)
    return buckets


def _fresh_day0_state() -> dict:
    """A fresh, UNSEEN LexemeState dict — phase=day0, day0_step=touch_a,
    last_reviewed_at=None.

    A split sense is a word the learner has never been introduced to *as that
    sense*; dropping it into maintenance quizzes it before any Touch-A
    introduction (the learner gets tested on a meaning they were never shown).
    So a split enters the new-word pipeline like any freshly added word: Day 0
    Touch-A intros first, then maintenance once it graduates. The caller must
    leave srs_state.last_reviewed_at = None so classify_lexeme_groups() treats
    the sense as unseen."""
    st = am.make_initial_lexeme_state()  # phase=day0, day0_step=touch_a
    return {"strategy": "incremental_production", "lexeme_state": am.lexeme_state_to_dict(st)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="test_srs.sqlite")
    ap.add_argument("--lexeme", required=True, help="the conflated headword, e.g. 분기")
    ap.add_argument("--sense", nargs=2, action="append", metavar=("SLUG", "BACK_SUBSTR"),
                    required=True, help="sense slug + a substring of the recog back that identifies it")
    ap.add_argument("--no-hanja", action="store_true", help="skip per-sense hanja pre-fetch on --apply")
    ap.add_argument("--apply", action="store_true", help="write to DB (otherwise dry run)")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")
    headword = srs_db.headword_of(args.lexeme)
    senses = [(slug, sub) for slug, sub in args.sense]

    eng = srs_db.make_engine(str(db))
    with Session(eng) as sess:
        recog, prod = _load_items(sess, headword)
        if recog is None:
            raise SystemExit(f"no recog bundle for lexeme:{headword}")
        variants = (recog.content or {}).get("variants", [])
        if not variants:
            raise SystemExit("recog bundle has no variants")

        buckets = _partition(variants, senses)

        print(f"\n  splitting {headword} ({len(variants)} recog variants) into {len(senses)} senses:")
        for slug, sub in senses:
            vs = buckets[slug]
            back = (vs[0].get("back") if vs else "?")
            print(f"\n  ── {headword}#{slug}  ({len(vs)} variants) ──")
            print(f"     back: {back}")
            for v in vs:
                print(f"       · {(v.get('front') or '').splitlines()[0][:62]}")

        if not args.apply:
            print("\n  dry-run — no DB writes. Re-run with --apply to commit.")
            return

        # ── apply ──
        archive = db.parent / "archive" / f"{db.stem}_presplit_{datetime.now():%Y%m%d_%H%M%S}.sqlite"
        archive.parent.mkdir(exist_ok=True)
        shutil.copy2(db, archive)
        print(f"\n  backed up DB -> {archive}")

        now = srs_db.now_utc()

        for slug, _ in senses:
            vs = copy.deepcopy(buckets[slug])
            card = {"type": "cloze_recog", "variants": vs}

            # per-sense hanja, disambiguated by the sense gloss
            if not args.no_hanja:
                from hanja import lookup_hanja
                gloss = next((v.get("back") for v in vs if (v.get("back") or "").strip()), None)
                try:
                    card["hanja"] = lookup_hanja(headword, gloss=gloss)
                    h = card["hanja"]
                    print(f"  hanja {headword}#{slug}: "
                          + (h.get("hanja") if h.get("has_hanja") else "none"))
                except Exception as e:
                    print(f"  hanja {headword}#{slug}: lookup failed ({e})")

            ext = f"lexeme:{headword}{srs_db.SENSE_SEP}{slug}:{SKILL_RECOG}"
            item_id = upsert_static_bundle(
                sess, external_id=ext, card=card,
                tags=["static", "bundle", "cloze_recog", headword, f"sense:{slug}"],
            )
            # fresh UNSEEN (Day-0) schedule on the recog item (canonical, no
            # prod): last_reviewed_at=None → classify_lexeme_groups() treats the
            # sense as a new word, so it gets Touch-A intros before any quiz.
            srs_state = sess.get(srs_db.SRSState, item_id)
            srs_state.state = _fresh_day0_state()
            srs_state.last_reviewed_at = None   # unseen → enters Day-0 pipeline
            srs_state.due_at = now
            print(f"  created {ext}  (Day 0 / unseen — will be introduced before review)")

        # retire the old conflated lexeme (history preserved via append-only log)
        for it in (recog, prod):
            if it is not None:
                it.deleted_at = now
                if it.srs_state is not None:
                    it.srs_state.due_at = now
                print(f"  retired {it.external_id} (soft-deleted)")

        sess.commit()
        print("\n  done.")


if __name__ == "__main__":
    main()
