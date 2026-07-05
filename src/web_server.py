#!/usr/bin/env python3
"""
web_server.py

Thin FastAPI wrapper around LexemeSRS.  Single-user, single-session.
Run on your Mac and connect from your phone via Tailscale.

Usage:
    python web_server.py --db test_srs.sqlite --target-new 5 --queue-size 3
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from sqlalchemy.orm import Session as SASession

import srs_db
import llm_query
from lexeme_srs import (
    LexemeSRS, _canonical_item, intro_queue_listing, set_intro_order, spread_homonyms,
)
from review_cli import Grade
from add_word import (
    word_exists, generate_and_insert, define_vocab_raw, build_definition_block,
    normalize_korean_word, check_sense_usage,
    inventory_senses, existing_senses, generate_and_insert_senses,
)
from add_diff_pair import diff_pair_exists, generate_and_insert_diff
from acquisition_model import (
    lexeme_state_from_dict,
    compute_due_at as acquisition_compute_due_at,
    PHASE_MAINTENANCE,
    PHASE_LEARNED,
)
from incremental_model import next_interval

# A recognition word is still "acquiring" until its scheduled review interval
# reaches a week; at/after a weekly cadence it counts as mature maintenance.
ACQUIRING_MAX_INTERVAL_SECS = 7 * 86400
from daily_analysis import compute_daily_summary
from grammar_srs import (
    GrammarSRS, GRAMMAR_DEFINE_PROMPT, pattern_exists, parse_grammar_definition,
)

# ── global state (single-user) ──────────────────────────────────────────

_lock = threading.Lock()
_srs: LexemeSRS | None = None
_sa_session: SASession | None = None
_current_card: dict | None = None  # card JSON sent to browser, awaiting action

_engine = None          # SQLAlchemy engine, created once at startup
_srs_args: dict = {}    # CLI args (target_new, queue_size, etc.)
_prompts: dict = {}     # loaded once from prompts.yaml at startup

# background word generation
_word_gen_tasks: dict[str, dict] = {}  # task_id -> {"status": "pending"|"done"|"error", "word": str, "message": str}

# grammar mode globals
_grammar_lock = threading.Lock()
_grammar_srs: GrammarSRS | None = None
_grammar_sa_session: SASession | None = None
_grammar_entry = None       # current GrammarEntry being reviewed
_grammar_question: str = "" # current question text
_grammar_question_data: dict | None = None  # full question data dict (includes eval_prompt)
_grammar_eval: dict | None = None  # evaluation result awaiting grade

app = FastAPI()


@app.exception_handler(Exception)
async def _heal_session_on_error(request, exc):
    """Roll back the live SQLAlchemy sessions on any unhandled error.

    A failed commit (e.g. a transient SQLite 'database is locked') leaves the
    session in a 'needs rollback' state, after which EVERY subsequent query
    raises until the process restarts — i.e. one blip would 500 all future
    reviews. Rolling back here lets the session self-heal so the next request
    works. Committed data is unaffected (rollback only discards the failed txn).
    """
    import traceback
    traceback.print_exc()
    for sess in (_sa_session, _grammar_sa_session):
        try:
            if sess is not None:
                sess.rollback()
        except Exception:
            pass
    return JSONResponse({"error": "internal error", "detail": str(exc)}, status_code=500)

# ── request models ───────────────────────────────────────────────────────

class SubmitReviewBody(BaseModel):
    grade: int  # 1-6

class EditBody(BaseModel):
    new_front: str
    new_back: str

class AddVariantBody(BaseModel):
    new_front: str
    new_back: str

class DefineWordBody(BaseModel):
    word: str
    context: str = ""  # sentence the word was clicked in, for disambiguation
    skip_normalize: bool = False  # treat input as a fixed phrase / idiom (no lemmatization)

class SenseSelection(BaseModel):
    slug: str
    definition_en: str
    hanja: dict | None = None

class ConfirmAddWordBody(BaseModel):
    word: str
    senses: list[SenseSelection] | None = None  # new: per-sense import
    tagged: bool = False                          # key as lexeme:word#slug when True
    def_block: str = ""                           # legacy single-sense fallback
    hot_add: bool = True  # inject into the live session; False = unseen pool only

class ExtendWordSenseBody(BaseModel):
    word: str
    extended_definition: str
    sentence: str = ""  # example sentence for the new sense

class PassageReviewGenerateBody(BaseModel):
    n_targets: int = 5

class PassageReviewTargetTap(BaseModel):
    lexeme: str
    item_id: int
    tapped: bool

class PassageReviewSubmitBody(BaseModel):
    targets: list[PassageReviewTargetTap]

class GrammarSubmitAnswerBody(BaseModel):
    answer: str

class GrammarSubmitGradeBody(BaseModel):
    grade: int  # 1-6

class GrammarDefinePatternBody(BaseModel):
    pattern: str

class GrammarConfirmAddBody(BaseModel):
    pattern: str
    meaning_en: str
    meaning_ko: str = ""
    example: str = ""
    notes: str = ""

class GrammarAddQuestionTypeBody(BaseModel):
    item_id: int
    qtype_config: dict

class GrammarRemoveQuestionTypeBody(BaseModel):
    item_id: int
    index: int

class GrammarSetQuestionTypesBody(BaseModel):
    item_id: int
    question_types: list

class GrammarDeletePatternBody(BaseModel):
    item_id: int

class UpdateSessionParamsBody(BaseModel):
    target_new: int | None = None
    queue_size: int | None = None

class DefineDiffPairBody(BaseModel):
    word_a: str
    word_b: str

class ConfirmDiffPairBody(BaseModel):
    word_a: str
    word_b: str
    def_block_a: str
    def_block_b: str

class AddNewWordsBody(BaseModel):
    count: int

class IntroReorderBody(BaseModel):
    order: list[str]  # full ordered list of unseen lexeme keys (first = introduced soonest)

# ── helpers ──────────────────────────────────────────────────────────────

def _card_to_dict(card: Any) -> dict:
    return {
        "item_id": card.item_id,
        "front": card.front,
        "back": card.back,
        "review_mode": card.review_mode,
        "translation_en": card.translation_en,
        "skill_type": card.skill_type,
        "difficulty": card.difficulty,
        "hanja": getattr(card, "hanja", None),
        "headword": getattr(card, "headword", ""),
    }


def _counts_dict() -> dict:
    c = _srs.session_counts()
    return asdict(c)


def _require_session():
    """Return a JSONResponse error if SRS session is not active, else None."""
    if _srs is None:
        return JSONResponse({"error": "no active session"}, status_code=400)
    return None


def _compute_lexeme_due_at(items, now):
    """Compute due_at for a seen lexeme group, mirroring LexemeSRS.__init__ logic."""
    canonical = _canonical_item(items)

    if canonical and canonical.srs_state:
        st = canonical.srs_state.state or {}
        # Try new format first (lexeme_state key)
        raw_state = st.get("lexeme_state") or st.get("skill_states") or st
        ls = lexeme_state_from_dict(raw_state)
        return acquisition_compute_due_at(ls, now)

    # Fallback to earliest raw DB due_at across group items
    earliest = None
    for item in items:
        if item.srs_state is not None and item.srs_state.due_at is not None:
            d = item.srs_state.due_at
            if d.tzinfo is None:
                d = d.replace(tzinfo=srs_db.UTC)
            if earliest is None or d < earliest:
                earliest = d
    if earliest is not None:
        return earliest

    return now

# ── routes ───────────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# ── pre-session endpoints ────────────────────────────────────────────────

def _get_lexeme_state(items):
    """Return the LexemeState for a lexeme group, or None."""
    canonical = _canonical_item(items)
    if not canonical or not canonical.srs_state:
        return None
    st = canonical.srs_state.state or {}
    raw_state = st.get("lexeme_state") or st.get("skill_states") or st
    return lexeme_state_from_dict(raw_state)


def _get_skill_due_times(items, now):
    """Return [(skill_name, due_at, locked)] for a lexeme group.

    With the new phase model there is at most one due time per lexeme
    (production only), so this returns a single-element list.
    """
    ls = _get_lexeme_state(items)
    if ls is None:
        return []
    due_at = acquisition_compute_due_at(ls, now)
    # No sibling locking in the new model
    return [("production", due_at, False)]


@app.get("/api/pre-session-status")
def api_pre_session_status():
    from datetime import timedelta

    with SASession(_engine) as tmp:
        seen, unseen = srs_db.classify_lexeme_groups(tmp)

        now = srs_db.now_utc()
        tomorrow = now + timedelta(hours=24)
        day_after = now + timedelta(hours=48)

        # Phase breakdown — maturity-based: "acquiring" = any day0/repairing word
        # OR a maintenance word whose review interval is still under a week;
        # "maintenance" = mature words on a weekly-or-longer cadence.
        acquiring_count = 0
        maintenance_count = 0
        learned_count = 0

        # Due-time counts across all phases
        due_count = 0        # due now (overdue or exactly due)
        due_tomorrow = 0     # due in (now, +24h]
        due_day_after = 0    # due in (+24h, +48h]
        nearest_due = None

        for group in seen.values():
            items = group["items"]
            ls = _get_lexeme_state(items)
            if ls is None:
                continue

            if ls.phase == PHASE_LEARNED:
                learned_count += 1
            elif (ls.phase == PHASE_MAINTENANCE
                  and next_interval(ls.half_life_secs, "recognition") >= ACQUIRING_MAX_INTERVAL_SECS):
                maintenance_count += 1
            else:
                acquiring_count += 1

            due_at = acquisition_compute_due_at(ls, now)
            if due_at <= now:
                due_count += 1
            elif due_at <= tomorrow:
                due_tomorrow += 1
            elif due_at <= day_after:
                due_day_after += 1
            if nearest_due is None or due_at < nearest_due:
                nearest_due = due_at

        next_due_seconds = None
        if nearest_due is not None and nearest_due > now:
            next_due_seconds = max(0.0, (nearest_due - now).total_seconds())

    return {
        "total": len(seen) + len(unseen),
        "seen": len(seen),
        "unseen": len(unseen),
        "acquiring": acquiring_count,
        "maintenance": maintenance_count,
        "learned": learned_count,
        "due_now": due_count,
        "due_tomorrow": due_tomorrow,
        "due_day_after": due_day_after,
        "next_due_seconds": next_due_seconds,
        "in_session": _srs is not None,
        "target_new": _srs_args["target_new"],
        "queue_size": _srs_args["queue_size"],
    }


@app.get("/api/daily-analysis")
def api_daily_analysis():
    return compute_daily_summary(_engine)


@app.post("/api/start-session")
def api_start_session():
    with _lock:
        global _srs, _sa_session, _current_card
        if _srs is not None:
            return JSONResponse({"error": "session already active"}, status_code=400)

        _sa_session = SASession(_engine)
        _srs = LexemeSRS(
            _sa_session,
            target_new=_srs_args["target_new"],
            queue_size=_srs_args["queue_size"],
            intro_examples=_srs_args["intro_examples"],
        )
        _current_card = None

        c = _srs.session_counts()
        print(f"SRS ready — unseen: {c.unseen}  learning: {c.learning}/{c.learning_capacity}  due: {c.reviews_due}")
        return {"ok": True, "counts": asdict(c)}


@app.post("/api/end-session")
def api_end_session():
    with _lock:
        global _srs, _sa_session, _current_card
        if _sa_session is not None:
            _sa_session.close()
        _srs = None
        _sa_session = None
        _current_card = None
        return {"ok": True}


@app.get("/api/intro-queue")
def api_intro_queue():
    """Upcoming word introductions (unseen lexemes) in the order they'll be
    introduced. The first `target_new` are flagged in_session."""
    target_new = _srs_args["target_new"]
    with SASession(_engine) as tmp:
        items = intro_queue_listing(tmp, target_new)
    return {
        "ok": True,
        "target_new": target_new,
        "queue_size": _srs_args["queue_size"],
        "in_session": _srs is not None,
        "total": len(items),
        "items": items,
    }


def _persist_intro_order(order: list[str]) -> dict:
    """Pin `order` as the intro order, rebuild any live session off it, and
    return the refreshed listing. Caller holds _lock."""
    global _srs, _sa_session, _current_card
    target_new = _srs_args["target_new"]
    with SASession(_engine) as tmp:
        updated = set_intro_order(tmp, order)
        tmp.commit()

    # rebuild the live session (if any) off the new order — its in-memory new
    # pool was built from the old order and would otherwise be stale
    if _srs is not None:
        if _sa_session is not None:
            _sa_session.close()
        _sa_session = SASession(_engine)
        _srs = LexemeSRS(
            _sa_session,
            target_new=_srs_args["target_new"],
            queue_size=_srs_args["queue_size"],
            intro_examples=_srs_args["intro_examples"],
        )
        _current_card = None

    with SASession(_engine) as tmp:
        items = intro_queue_listing(tmp, target_new)
    return {"ok": True, "updated": updated, "target_new": target_new,
            "in_session": _srs is not None, "total": len(items), "items": items}


@app.post("/api/intro-queue/reorder")
def api_intro_queue_reorder(body: IntroReorderBody):
    """Pin a manual intro order, then rebuild any live session so the new
    top-`target_new` take effect immediately."""
    with _lock:
        return _persist_intro_order(body.order)


@app.post("/api/intro-queue/spread-homonyms")
def api_intro_queue_spread_homonyms():
    """Reorder the current intro queue so homonyms (senses sharing a headword)
    are spread evenly across the list — confusable senses shouldn't be learned
    together — then persist and apply it."""
    with _lock:
        with SASession(_engine) as tmp:
            current = [x["lexeme"] for x in intro_queue_listing(tmp, _srs_args["target_new"])]
        spread = spread_homonyms(current)
        result = _persist_intro_order(spread)
        # count headwords that had >1 sense (how many were spread)
        from collections import Counter
        c = Counter(srs_db.headword_of(l) for l in current)
        result["spread_groups"] = sum(1 for v in c.values() if v > 1)
        return result


@app.post("/api/update-session-params")
def api_update_session_params(body: UpdateSessionParamsBody):
    if body.target_new is not None:
        _srs_args["target_new"] = max(0, min(25, body.target_new))
    if body.queue_size is not None:
        _srs_args["queue_size"] = max(1, min(10, body.queue_size))
        with _lock:
            if _srs is not None:
                _srs.set_queue_size(_srs_args["queue_size"])
    return {
        "ok": True,
        "target_new": _srs_args["target_new"],
        "queue_size": _srs_args["queue_size"],
    }


@app.post("/api/add-new-words")
def api_add_new_words(body: AddNewWordsBody):
    with _lock:
        if _srs is None:
            return JSONResponse({"error": "no active session"}, status_code=400)
        n = max(0, min(25, body.count))
        added = _srs.add_new_words(n)
        return {"ok": True, "added": added, "counts": _counts_dict()}


# ── word addition endpoints ──────────────────────────────────────────────

@app.post("/api/define-word")
def api_define_word(body: DefineWordBody):
    """Enumerate the distinct senses of a word (sense-inventory first pass).

    Returns {ok, word, senses: [{slug, definition_en, hanja, exists}], tagged,
    context_slug, all_exist}. `tagged` is True when the word is polysemous, so
    each imported sense will be keyed lexeme:word#slug; monosemous words key
    plain. Senses already in the deck are flagged `exists`. Falls back to a
    single synthetic sense if the inventory LLM call fails."""
    word = body.word.strip()
    if not word:
        return JSONResponse({"ok": False, "error": "empty word"}, status_code=400)

    context = (body.context or "").strip()
    # Strip particles / deconjugate to the dictionary form before lookup,
    # unless the user explicitly flagged this as a fixed phrase / idiom
    # (e.g. 이래 봬도, 알다시피) whose canonical form is the form as typed.
    if not body.skip_normalize:
        word = normalize_korean_word(word, context=context)

    with SASession(_engine) as tmp:
        existing = existing_senses(tmp, word)

    context_slug = None
    try:
        inv = inventory_senses(word, context=context)
        senses = inv["senses"]
        context_slug = inv["context_slug"]
    except Exception:
        # Fallback: legacy single define so the add flow still works.
        try:
            raw_def = define_vocab_raw(_prompts, word, context=context)
            parsed = llm_query.parse_vocab_block(raw_def)
            if not parsed.get("ok"):
                return {"ok": False, "error": f"parse failed: {parsed.get('error')}"}
            senses = [{
                "slug": "main",
                "definition_en": (parsed.get("def_en") or "").strip(),
                "hanja": {"has_hanja": False},
            }]
        except Exception as e:
            return {"ok": False, "error": f"definition failed: {e}"}

    # Mark senses already in the deck (by Hanja, then slug, then plain key).
    existing_hanja = {
        e["hanja"]["hanja"] for e in existing
        if e.get("hanja") and e["hanja"].get("has_hanja") and e["hanja"].get("hanja")
    }
    existing_slugs = {e["slug"] for e in existing if e.get("slug")}
    has_plain = any(e.get("slug") is None for e in existing)
    tagged = len(senses) > 1
    for s in senses:
        h = s["hanja"].get("hanja") if isinstance(s.get("hanja"), dict) and s["hanja"].get("has_hanja") else None
        s["exists"] = bool(
            (h and h in existing_hanja)
            or (s["slug"] in existing_slugs)
            or (not tagged and has_plain)
        )

    all_exist = bool(senses) and all(s["exists"] for s in senses)
    return {
        "ok": True, "word": word, "senses": senses,
        "tagged": tagged, "context_slug": context_slug, "all_exist": all_exist,
    }


@app.post("/api/confirm-add-word")
def api_confirm_add_word(body: ConfirmAddWordBody):
    word = body.word.strip()
    sense_dicts = (
        [{"slug": s.slug, "definition_en": s.definition_en, "hanja": s.hanja}
         for s in body.senses]
        if body.senses else None
    )

    if not word or (not sense_dicts and not body.def_block):
        return JSONResponse({"ok": False, "error": "missing word or senses"}, status_code=400)

    import uuid
    task_id = uuid.uuid4().hex[:8]
    _word_gen_tasks[task_id] = {"status": "pending", "word": word, "message": ""}
    tagged = body.tagged
    hot_add = body.hot_add

    def _bg_generate():
        try:
            with SASession(_engine) as tmp:
                if sense_dicts:
                    ok, msg, created = generate_and_insert_senses(
                        tmp, prompts=_prompts, word=word, senses=sense_dicts,
                        tagged=tagged, n_variants=_srs_args["n_variants"],
                        model=_srs_args["model"],
                    )
                else:
                    # legacy single-sense path (def_block)
                    ok, msg = generate_and_insert(
                        tmp, prompts=_prompts, word=word, def_block=body.def_block,
                        n_variants=_srs_args["n_variants"], model=_srs_args["model"],
                    )
                    created = [word]
            if ok:
                _word_gen_tasks[task_id]["status"] = "done"
                _word_gen_tasks[task_id]["message"] = msg
                if hot_add and _srs is not None:
                    with _lock:
                        for key in created:
                            _srs.hot_add_word(key)
            else:
                _word_gen_tasks[task_id]["status"] = "error"
                _word_gen_tasks[task_id]["message"] = msg
        except Exception as e:
            _word_gen_tasks[task_id]["status"] = "error"
            _word_gen_tasks[task_id]["message"] = str(e)

    threading.Thread(target=_bg_generate, daemon=True).start()
    return {"ok": True, "task_id": task_id}


@app.get("/api/word-gen-status/{task_id}")
def api_word_gen_status(task_id: str):
    task = _word_gen_tasks.get(task_id)
    if task is None:
        return JSONResponse({"error": "unknown task"}, status_code=404)
    return task


@app.post("/api/extend-word-sense")
def api_extend_word_sense(body: ExtendWordSenseBody):
    """Apply a sense extension to an existing recognition-bundle item:
    rewrite every variant's `back` to the new (combined) definition, and
    append a new variant whose example is the sentence the new sense was
    observed in.
    """
    word = body.word.strip()
    extended = (body.extended_definition or "").strip()
    sentence = (body.sentence or "").strip()
    if not word or not extended:
        return JSONResponse(
            {"ok": False, "error": "missing word or extended_definition"},
            status_code=400,
        )

    with SASession(_engine) as tmp:
        recog = tmp.query(srs_db.Item).filter(
            srs_db.Item.external_id == f"lexeme:{word}:cloze_recog_bundle",
            srs_db.Item.deleted_at.is_(None),
        ).first()
        if recog is None or not isinstance(recog.content, dict):
            return {"ok": False, "error": f"'{word}' not found"}

        content = dict(recog.content)
        variants = list(content.get("variants", []))
        for v in variants:
            v["back"] = extended

        # Append a new variant for the new sense if we have a sentence.
        if sentence:
            bracketed = sentence
            if word in sentence and "[[" not in sentence:
                bracketed = sentence.replace(word, f"[[{word}]]", 1)
            new_front = bracketed + "\n\n질문: 밑줄 친 표현의 뜻은?"
            new_variant = {"front": new_front, "back": extended, "translation_en": ""}
            try:
                from backfill_translations import translate_variant
                new_variant["translation_en"] = translate_variant(
                    new_variant["front"], new_variant["back"], "cloze_recog_bundle"
                ) or ""
            except Exception as e:
                # translation is non-critical; continue without it
                pass
            variants.append(new_variant)

        content["variants"] = variants
        recog.content = content
        recog.back = extended
        recog.updated_at = srs_db.now_utc()
        tmp.commit()

    return {"ok": True, "word": word, "n_variants": len(variants)}


@app.post("/api/hanja-lookup")
def api_hanja_lookup():
    """On-demand Hanja breakdown for the in-flight recognition card's lexeme.

    Cached results return instantly (no LLM call); the first lookup for a
    lexeme calls the LLM and caches the result on the recog item. Returns the
    stored sentinel: {"has_hanja": true, "hanja", "chars", "gloss"} or
    {"has_hanja": false}."""
    with _lock:
        err = _require_session()
        if err:
            return err
        if _current_card is None or _current_card.get("card") is None:
            return JSONResponse({"error": "no card in flight"}, status_code=400)
        card = _current_card["card"]
        if "recog" not in (card.get("skill_type") or ""):
            return {"has_hanja": False}
        result = _srs.lookup_hanja(card.get("item_id"))
        # Cache onto the in-flight card so an /api/next refresh shows it too.
        card["hanja"] = result
        return result


# ── passage review (Phase 2) ─────────────────────────────────────────────

@app.post("/api/passage-review/generate")
def api_passage_review_generate(body: PassageReviewGenerateBody):
    """Pick N oldest-due maintenance lexemes and ask the LLM for a 1-2
    paragraph passage that embeds each one (marked as `[[surface|lexeme]]`).
    """
    import passage_review as pr

    n = max(1, min(20, body.n_targets))
    with SASession(_engine) as tmp:
        targets = pr.select_passage_targets(tmp, n)
    if not targets:
        return {"ok": False, "error": "no maintenance words are due"}

    lexemes = [t["lexeme"] for t in targets]
    try:
        passage = pr.generate_passage(lexemes)
    except Exception as e:
        return {"ok": False, "error": f"passage generation failed: {e}"}
    if not passage:
        return {"ok": False, "error": "LLM returned an empty passage"}

    found = pr.extract_target_spans(passage)
    # only return targets the LLM actually wove in (frontend can't grade missing)
    present_targets = [t for t in targets if t["lexeme"] in found]
    missing = [t["lexeme"] for t in targets if t["lexeme"] not in found]

    # English translation, revealed only after the learner's first read pass
    try:
        translation = pr.translate_passage(pr.strip_markers(passage))
    except Exception:
        translation = ""

    return {
        "ok": True,
        "passage": passage,
        "targets": present_targets,
        "missing": missing,
        "translation": translation,
    }


@app.post("/api/passage-review/submit")
def api_passage_review_submit(body: PassageReviewSubmitBody):
    """Grade each target: untapped → grade 2 (pass), tapped → grade 5 (fail).
    Each grade runs through the same half-life + repair machinery as a
    normal sentence-card maintenance review.
    """
    import passage_review as pr

    now = srs_db.now_utc()
    results: list[dict] = []
    with SASession(_engine) as tmp:
        for t in body.targets:
            lex = (t.lexeme or "").strip()
            if not lex or not t.item_id:
                continue
            grade = 5 if t.tapped else 2
            res = pr.apply_passage_grade(tmp, lex, t.item_id, grade, now=now)
            results.append(res)
        tmp.commit()

    # If a vocab session is live, rebuild it so the passage grades (repairs /
    # reschedules) are reflected immediately — its in-memory queues and cached
    # DB session are otherwise stale and could even clobber these writes.
    with _lock:
        global _srs, _sa_session, _current_card
        if _srs is not None:
            if _sa_session is not None:
                _sa_session.close()
            _sa_session = SASession(_engine)
            _srs = LexemeSRS(
                _sa_session,
                target_new=_srs_args["target_new"],
                queue_size=_srs_args["queue_size"],
                intro_examples=_srs_args["intro_examples"],
            )
            _current_card = None
    return {"ok": True, "target_results": results}


# ── diff pair endpoints ──────────────────────────────────────────────────

@app.post("/api/define-diff-pair")
def api_define_diff_pair(body: DefineDiffPairBody):
    word_a = body.word_a.strip()
    word_b = body.word_b.strip()
    if not word_a or not word_b:
        return JSONResponse({"ok": False, "error": "both words required"}, status_code=400)

    with SASession(_engine) as tmp:
        if diff_pair_exists(tmp, word_a, word_b):
            return {"ok": False, "error": f"diff pair '{word_a}~{word_b}' already exists"}

    # Define both words in parallel
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(define_vocab_raw, _prompts, word_a)
        fut_b = pool.submit(define_vocab_raw, _prompts, word_b)
        try:
            raw_a = fut_a.result()
            raw_b = fut_b.result()
        except Exception as e:
            return {"ok": False, "error": f"definition failed: {e}"}

    parsed_a = llm_query.parse_vocab_block(raw_a)
    parsed_b = llm_query.parse_vocab_block(raw_b)
    if not parsed_a.get("ok"):
        return {"ok": False, "error": f"parse failed for '{word_a}': {parsed_a.get('error')}"}
    if not parsed_b.get("ok"):
        return {"ok": False, "error": f"parse failed for '{word_b}': {parsed_b.get('error')}"}

    def_block_a = build_definition_block(parsed_a)
    def_block_b = build_definition_block(parsed_b)

    return {
        "ok": True,
        "parsed_a": parsed_a,
        "parsed_b": parsed_b,
        "def_block_a": def_block_a,
        "def_block_b": def_block_b,
    }


@app.post("/api/confirm-add-diff-pair")
def api_confirm_add_diff_pair(body: ConfirmDiffPairBody):
    word_a = body.word_a.strip()
    word_b = body.word_b.strip()
    def_block_a = body.def_block_a
    def_block_b = body.def_block_b

    if not word_a or not word_b or not def_block_a or not def_block_b:
        return JSONResponse({"ok": False, "error": "missing fields"}, status_code=400)

    import uuid
    task_id = uuid.uuid4().hex[:8]
    _word_gen_tasks[task_id] = {"status": "pending", "word": f"{word_a}~{word_b}", "message": ""}

    def _bg_generate():
        try:
            with SASession(_engine) as tmp:
                ok, msg = generate_and_insert_diff(
                    tmp,
                    prompts=_prompts,
                    word_a=word_a,
                    def_block_a=def_block_a,
                    word_b=word_b,
                    def_block_b=def_block_b,
                    n_variants=_srs_args["n_variants"],
                    model=_srs_args["model"],
                )
            if ok:
                _word_gen_tasks[task_id]["status"] = "done"
                _word_gen_tasks[task_id]["message"] = msg
                with _lock:
                    if _srs is not None:
                        _srs.hot_add_word(f"{word_a}~{word_b}")
            else:
                _word_gen_tasks[task_id]["status"] = "error"
                _word_gen_tasks[task_id]["message"] = msg
        except Exception as e:
            _word_gen_tasks[task_id]["status"] = "error"
            _word_gen_tasks[task_id]["message"] = str(e)

    threading.Thread(target=_bg_generate, daemon=True).start()
    return {"ok": True, "task_id": task_id}


# ── review endpoints (require active session) ────────────────────────────

@app.get("/api/status")
def api_status():
    with _lock:
        err = _require_session()
        if err:
            return err
        counts = _counts_dict()
        seconds = _srs.seconds_until_next_due()
    return {"counts": counts, "seconds_until_next_due": seconds}


@app.get("/api/next")
def api_next():
    with _lock:
        global _current_card

        err = _require_session()
        if err:
            return err

        # if a card is already in flight, return it again (refresh counts)
        if _current_card is not None:
            _current_card["counts"] = _counts_dict()
            return _current_card

        if _srs.is_session_complete():
            resp = {"state": "complete", "counts": _counts_dict()}
            return resp

        card = _srs.next_due_item()
        if card is None:
            seconds = _srs.seconds_until_next_due()
            return {
                "state": "waiting",
                "seconds_until_next_due": seconds,
                "counts": _counts_dict(),
            }

        card_dict = _card_to_dict(card)
        _current_card = {
            "state": "card",
            "card": card_dict,
            "counts": _counts_dict(),
            "in_repair_mode": _srs.in_repair_mode,
            "repair_remaining": len(_srs._repair_pool) + len(_srs._repair_active),
        }
        return _current_card


@app.post("/api/submit-intro")
def api_submit_intro():
    with _lock:
        global _current_card
        err = _require_session()
        if err:
            return err
        if _current_card is None:
            return JSONResponse({"error": "no card in flight"}, status_code=400)
        _srs.submit_intro(None)
        _current_card = None
        return {"ok": True, "counts": _counts_dict()}


@app.post("/api/submit-review")
def api_submit_review(body: SubmitReviewBody):
    with _lock:
        global _current_card
        err = _require_session()
        if err:
            return err
        if _current_card is None:
            return JSONResponse({"error": "no card in flight"}, status_code=400)
        try:
            grade = Grade(body.grade)
        except ValueError:
            return JSONResponse({"error": f"invalid grade: {body.grade}"}, status_code=400)
        _srs.submit_review(None, grade)
        _current_card = None
        return {
            "ok": True,
            "counts": _counts_dict(),
            "review_log": _srs.last_review_log,
            "in_repair_mode": _srs.in_repair_mode,
            "repair_remaining": len(_srs._repair_pool) + len(_srs._repair_active),
        }


@app.post("/api/edit")
def api_edit(body: EditBody):
    with _lock:
        global _current_card
        err = _require_session()
        if err:
            return err
        if _current_card is None:
            return JSONResponse({"error": "no card in flight"}, status_code=400)
        _srs.edit_and_skip(None, body.new_front, body.new_back)
        _current_card = None
        return {"ok": True, "counts": _counts_dict()}


@app.post("/api/delete-variant")
def api_delete_variant():
    with _lock:
        global _current_card
        err = _require_session()
        if err:
            return err
        if _current_card is None:
            return JSONResponse({"error": "no card in flight"}, status_code=400)
        _srs.delete_variant_and_skip(None)
        _current_card = None
        return {"ok": True, "counts": _counts_dict()}


@app.post("/api/quarantine")
def api_quarantine():
    with _lock:
        global _current_card
        err = _require_session()
        if err:
            return err
        if _current_card is None:
            return JSONResponse({"error": "no card in flight"}, status_code=400)
        _srs.quarantine_and_skip(None)
        _current_card = None
        return {"ok": True, "counts": _counts_dict()}


@app.post("/api/add-variant")
def api_add_variant(body: AddVariantBody):
    with _lock:
        global _current_card
        err = _require_session()
        if err:
            return err
        if _current_card is None:
            return JSONResponse({"error": "no card in flight"}, status_code=400)
        _srs.add_variant_and_skip(None, body.new_front, body.new_back)
        _current_card = None
        return {"ok": True, "counts": _counts_dict()}


# ── grammar endpoints ────────────────────────────────────────────────

@app.get("/api/grammar/pre-session-status")
def api_grammar_pre_session_status():
    with SASession(_engine) as tmp:
        srs = GrammarSRS(tmp)
        # Count suspended (unseen) grammar patterns
        from sqlalchemy import select, func as sa_func
        suspended_count = tmp.scalar(
            select(sa_func.count(srs_db.Item.id)).where(
                srs_db.Item.item_type == "grammar",
                srs_db.Item.deleted_at.is_(None),
                srs_db.Item.suspended == True,  # noqa: E712
            )
        ) or 0
        return {
            "total_patterns": srs.total_patterns(),
            "suspended_patterns": suspended_count,
            "due_now": srs.due_count(),
            "in_session": _grammar_srs is not None,
        }


class ActivateGrammarBody(BaseModel):
    count: int = 1


@app.post("/api/grammar/activate-patterns")
def api_grammar_activate_patterns(body: ActivateGrammarBody):
    """Unsuspend N grammar patterns (oldest first), create SRSState if needed."""
    n = max(0, min(25, body.count))
    with SASession(_engine) as tmp:
        from sqlalchemy import select
        items = tmp.scalars(
            select(srs_db.Item)
            .where(
                srs_db.Item.item_type == "grammar",
                srs_db.Item.deleted_at.is_(None),
                srs_db.Item.suspended == True,  # noqa: E712
            )
            .order_by(srs_db.Item.created_at.asc())
            .limit(n)
        ).all()

        now = srs_db.now_utc()
        activated = 0
        for item in items:
            item.suspended = False
            # Ensure SRSState exists
            existing = tmp.get(srs_db.SRSState, item.id)
            if not existing:
                tmp.add(srs_db.SRSState(
                    item_id=item.id,
                    due_at=now,
                    scheduler_name="incremental",
                    scheduler_version=1,
                    state={"skill_states": {}},
                ))
            activated += 1

        tmp.commit()

        # Reload in-session grammar if active
        with _grammar_lock:
            if _grammar_srs is not None:
                _grammar_srs.reload_items()

        return {"ok": True, "activated": activated}


@app.post("/api/grammar/start-session")
def api_grammar_start_session():
    with _grammar_lock:
        global _grammar_srs, _grammar_sa_session, _grammar_entry, _grammar_question, _grammar_question_data, _grammar_eval
        if _grammar_srs is not None:
            return JSONResponse({"error": "grammar session already active"}, status_code=400)
        _grammar_sa_session = SASession(_engine)
        _grammar_srs = GrammarSRS(_grammar_sa_session)
        _grammar_entry = None
        _grammar_question = ""
        _grammar_question_data = None
        _grammar_eval = None
        return {
            "ok": True,
            "total_patterns": _grammar_srs.total_patterns(),
            "due_now": _grammar_srs.due_count(),
        }


@app.post("/api/grammar/end-session")
def api_grammar_end_session():
    with _grammar_lock:
        global _grammar_srs, _grammar_sa_session, _grammar_entry, _grammar_question, _grammar_question_data, _grammar_eval
        if _grammar_srs is not None:
            _grammar_srs.shutdown()
        if _grammar_sa_session is not None:
            _grammar_sa_session.close()
        total = _grammar_srs._total_graded if _grammar_srs else 0
        correct = _grammar_srs._correct_count if _grammar_srs else 0
        _grammar_srs = None
        _grammar_sa_session = None
        _grammar_entry = None
        _grammar_question = ""
        _grammar_question_data = None
        _grammar_eval = None
        return {"ok": True, "total_graded": total, "correct_count": correct}


@app.get("/api/grammar/next")
def api_grammar_next():
    with _grammar_lock:
        global _grammar_entry, _grammar_question, _grammar_question_data, _grammar_eval
        if _grammar_srs is None:
            return JSONResponse({"error": "no grammar session"}, status_code=400)

        # if we already have an entry awaiting grade, return it
        if _grammar_eval is not None:
            return {
                "state": "feedback",
                "question": _grammar_question,
                "evaluation": _grammar_eval,
                "counts": _grammar_counts(),
            }

        if _grammar_srs.is_empty():
            return {"state": "empty", "counts": _grammar_counts()}

        entry = _grammar_srs.next_due_entry()
        if entry is None:
            wait = _grammar_srs.seconds_until_next_due()
            return {
                "state": "waiting",
                "seconds_until_next_due": wait,
                "counts": _grammar_counts(),
            }

        _grammar_entry = entry
        _grammar_eval = None
        qdata = _grammar_srs.get_question(entry)
        _grammar_question = qdata["question"]
        _grammar_question_data = qdata
        _grammar_srs.prefetch_next(current_entry=entry)

        return {
            "state": "question",
            "question": _grammar_question,
            "counts": _grammar_counts(),
        }


@app.post("/api/grammar/submit-answer")
def api_grammar_submit_answer(body: GrammarSubmitAnswerBody):
    with _grammar_lock:
        global _grammar_eval
        if _grammar_srs is None or _grammar_entry is None or _grammar_question_data is None:
            return JSONResponse({"error": "no active grammar question"}, status_code=400)
        evaluation = _grammar_srs.evaluate_answer(
            _grammar_entry, _grammar_question_data, body.answer.strip()
        )
        _grammar_eval = {
            "correct": evaluation.get("correct", False),
            "feedback": evaluation.get("feedback", ""),
            "corrected": evaluation.get("corrected", ""),
            "answer": body.answer.strip(),
        }
        return {
            "ok": True,
            "evaluation": _grammar_eval,
            "counts": _grammar_counts(),
        }


@app.post("/api/grammar/submit-grade")
def api_grammar_submit_grade(body: GrammarSubmitGradeBody):
    with _grammar_lock:
        global _grammar_entry, _grammar_question, _grammar_question_data, _grammar_eval
        if _grammar_srs is None or _grammar_entry is None or _grammar_eval is None:
            return JSONResponse({"error": "no grammar card awaiting grade"}, status_code=400)
        if body.grade not in (1, 2, 3, 4, 5, 6):
            return JSONResponse({"error": f"invalid grade: {body.grade}"}, status_code=400)
        llm_score = 3  # default
        question_type = _grammar_question_data.get("question_type", "sentence_completion") if _grammar_question_data else "sentence_completion"
        _grammar_srs.submit_grade(
            _grammar_entry, body.grade,
            _grammar_question, _grammar_eval["answer"], llm_score,
            question_type=question_type,
        )
        _grammar_srs.prefetch_next()
        _grammar_entry = None
        _grammar_question = ""
        _grammar_question_data = None
        _grammar_eval = None
        return {"ok": True, "counts": _grammar_counts()}


@app.post("/api/grammar/define-pattern")
def api_grammar_define_pattern(body: GrammarDefinePatternBody):
    pattern = body.pattern.strip()
    if not pattern:
        return JSONResponse({"ok": False, "error": "empty pattern"}, status_code=400)
    with SASession(_engine) as tmp:
        if pattern_exists(tmp, pattern):
            return {"ok": False, "error": f"'{pattern}' already exists"}
    try:
        raw_def = llm_query.query_api(
            pattern, system_prompt=GRAMMAR_DEFINE_PROMPT, verbose=False,
        )
    except Exception as e:
        return {"ok": False, "error": f"definition failed: {e}"}
    parsed = parse_grammar_definition(raw_def)
    return {"ok": True, "parsed": parsed}


@app.post("/api/grammar/confirm-add-pattern")
def api_grammar_confirm_add_pattern(body: GrammarConfirmAddBody):
    pattern = body.pattern.strip()
    if not pattern:
        return JSONResponse({"ok": False, "error": "empty pattern"}, status_code=400)
    with SASession(_engine) as tmp:
        if pattern_exists(tmp, pattern):
            return {"ok": False, "error": f"'{pattern}' already exists"}
        srs = GrammarSRS(tmp)
        item_id = srs.add_pattern_to_db(
            pattern=pattern,
            meaning_en=body.meaning_en,
            meaning_ko=body.meaning_ko,
            example=body.example,
            notes=body.notes,
        )
        srs.shutdown()
    # reload active session if running
    if _grammar_srs is not None:
        _grammar_srs.reload_items()
    return {"ok": True, "message": f"Added '{pattern}' (id={item_id})"}


@app.post("/api/grammar/add-question-type")
def api_grammar_add_question_type(body: GrammarAddQuestionTypeBody):
    """Add a question type config to an existing grammar pattern."""
    with SASession(_engine) as tmp:
        srs = GrammarSRS(tmp)
        item = tmp.get(srs_db.Item, body.item_id)
        if item is None or item.item_type != "grammar":
            return JSONResponse({"ok": False, "error": "item not found"}, status_code=404)
        srs.add_question_type(body.item_id, body.qtype_config)
        srs.shutdown()
    # reload active session if running
    if _grammar_srs is not None:
        _grammar_srs.reload_items()
    return {"ok": True}


@app.get("/config")
def serve_config():
    return FileResponse(Path(__file__).parent / "static" / "grammar_config.html")


@app.get("/api/grammar/patterns")
def api_grammar_patterns():
    with SASession(_engine) as tmp:
        srs = GrammarSRS(tmp)
        patterns = [{
            "item_id": e.item_id,
            "pattern": e.pattern,
            "meaning_en": e.meaning_en,
            "question_types": e.content.get("question_types", []),
        } for e in srs._entries]
        srs.shutdown()
    return {"patterns": patterns}


@app.post("/api/grammar/set-question-types")
def api_grammar_set_question_types(body: GrammarSetQuestionTypesBody):
    with SASession(_engine) as tmp:
        srs = GrammarSRS(tmp)
        item = tmp.get(srs_db.Item, body.item_id)
        if item is None or item.item_type != "grammar":
            srs.shutdown()
            return JSONResponse({"ok": False, "error": "item not found"}, status_code=404)
        srs.set_question_types(body.item_id, body.question_types)
        srs.shutdown()
    if _grammar_srs is not None:
        _grammar_srs.reload_items()
    return {"ok": True}


@app.post("/api/grammar/remove-question-type")
def api_grammar_remove_question_type(body: GrammarRemoveQuestionTypeBody):
    with SASession(_engine) as tmp:
        srs = GrammarSRS(tmp)
        item = tmp.get(srs_db.Item, body.item_id)
        if item is None or item.item_type != "grammar":
            srs.shutdown()
            return JSONResponse({"ok": False, "error": "item not found"}, status_code=404)
        srs.remove_question_type(body.item_id, body.index)
        srs.shutdown()
    if _grammar_srs is not None:
        _grammar_srs.reload_items()
    return {"ok": True}


@app.post("/api/grammar/delete-pattern")
def api_grammar_delete_pattern(body: GrammarDeletePatternBody):
    with SASession(_engine) as tmp:
        srs = GrammarSRS(tmp)
        ok = srs.delete_pattern(body.item_id)
        srs.shutdown()
    if not ok:
        return JSONResponse({"ok": False, "error": "item not found"}, status_code=404)
    if _grammar_srs is not None:
        _grammar_srs.reload_items()
    return {"ok": True}


def _grammar_counts() -> dict:
    return {
        "due": _grammar_srs.due_count() if _grammar_srs else 0,
        "total_graded": _grammar_srs._total_graded if _grammar_srs else 0,
        "correct_count": _grammar_srs._correct_count if _grammar_srs else 0,
        "total_patterns": _grammar_srs.total_patterns() if _grammar_srs else 0,
    }


# ── startup ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Korean SRS web server")
    ap.add_argument("--db", default="test_srs.sqlite")
    ap.add_argument("--target-new", type=int, default=0)
    ap.add_argument("--queue-size", type=int, default=3)
    ap.add_argument("--intro-examples", type=int, default=2)
    ap.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.yaml"))
    ap.add_argument("--n-variants", type=int, default=8)
    ap.add_argument("--model", default=llm_query.BUNDLE_MODEL, help="Model for bundle generation")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    # Store args for deferred SRS init
    _srs_args = {
        "target_new": args.target_new,
        "queue_size": args.queue_size,
        "intro_examples": args.intro_examples,
        "n_variants": args.n_variants,
        "model": args.model,
    }

    # Create engine + load prompts once at startup
    _engine = srs_db.make_engine(args.db)
    srs_db.init_db(_engine)
    _prompts = llm_query.load_prompts(Path(args.prompts))

    print(f"Server ready (pre-session mode). DB: {args.db}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
