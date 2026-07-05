#!/usr/bin/env python3
"""
add_word.py

Canonical word-addition pipeline: define -> generate bundles -> translate -> DB upsert.

Used by:
  - seed_static_cards.py (batch CLI)
  - web_server.py (web UI)
  - review_cli.py (interactive CLI, via show_definition_and_confirm)
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from sqlalchemy.orm import Session

import llm_query
import srs_db

logger = logging.getLogger(__name__)


# ── shared helpers (canonical home) ─────────────────────────────────────


def define_vocab_raw(
    prompts: dict, word: str, model: str = "gpt-5-mini", context: str = ""
) -> str:
    """
    Reuse your existing define_vocab prompt via llm_query.query_api (streaming).
    Returns the raw TERM/DEF_EN/DEF_KO/EXAMPLES block.

    If `context` (the sentence the word was clicked in) is given, it is
    appended so the LLM defines the sense actually used there.
    """
    instruction = prompts["system_prompts"]["define_vocab"]
    query = prompts["queries"]["define_vocab"]
    user = f"{query} {word}"
    if context:
        user += (
            f"\n\nThe word appears in this sentence — define the sense used "
            f"here: {context}"
        )
    return llm_query.query_api(user, instruction, model=model, verbose=False)


def normalize_korean_word(word: str, context: str = "", model: str = "gpt-5-mini") -> str:
    """Reduce a clicked/typed Korean word to its dictionary (lemma) form.

    Strips particles (josa) from nouns — 을/를, 은/는, 이/가, 의, 에/에서,
    (으)로, 게, etc. — and reduces conjugated verbs/adjectives to the base
    form ending in 다. The LLM decides whether trailing syllables are a
    particle/ending or part of the word itself (가게, 시계, 마을, 가을 stay
    whole). The optional `context` sentence disambiguates that call.
    Falls back to the original word on any failure.
    """
    word = (word or "").strip()
    if not word:
        return word
    instruction = (
        "You normalize a Korean expression to its dictionary form. The input "
        "may be a single word OR a fixed multi-word collocation/phrase/idiom.\n"
        "\n"
        "SINGLE WORD inputs:\n"
        "- Noun + attached particle/josa (을/를, 은/는, 이/가, 의, 에, 에서, "
        "에게, 한테, 도, 만, (으)로, 와/과, 까지, 부터, 게 ...): return the "
        "noun WITHOUT the particle.\n"
        "- Conjugated verb or adjective: return the base form ending in 다.\n"
        "- If the trailing syllables are part of the word itself (가게, 시계, "
        "마을, 가을, 그릇), return the word unchanged.\n"
        "\n"
        "MULTI-WORD inputs (any input containing a space — e.g. 영향을 미치다, "
        "손을 잡다, 고개를 떨구다, 꼼짝 없이, 힘이 나다, 숨을 고르다): the "
        "phrase is itself the lexeme. PRESERVE the full phrase. Do NOT strip "
        "the internal particles — they are part of the collocation. Only "
        "normalize the FINAL verb/adjective (if any) to its dictionary form.\n"
        "\n"
        "CRITICAL CARDINALITY RULE: your output's space-separated word count "
        "MUST match the input's. A single-token input ALWAYS produces a "
        "single-token output, even if the surrounding sentence shows the "
        "word as part of a collocation. The sentence is only for sense "
        "disambiguation — never use it to expand the input into a phrase.\n"
        "\n"
        "Examples:\n"
        "  확대를                                       ->  확대\n"
        "  들어섰다                                     ->  들어서다\n"
        "  마을                                         ->  마을\n"
        "  가게                                         ->  가게\n"
        "  영향을 미치다                                ->  영향을 미치다\n"
        "  영향을 미쳤다                                ->  영향을 미치다\n"
        "  손을 잡았다                                  ->  손을 잡다\n"
        "  꼼짝 없이                                    ->  꼼짝 없이\n"
        "  미치다  (sentence: 사회에 큰 영향을 미쳤다)  ->  미치다       "
        "(NOT 영향을 미치다 — single token in, single token out)\n"
        "\n"
        "Reply with ONLY the normalized expression — no quotes, no explanation."
    )
    user = f"Word: {word}"
    if context:
        user += f"\nSentence it appears in: {context}"
    try:
        out = llm_query.query_api(user, instruction, model=model, verbose=False)
    except Exception as e:
        logger.warning("normalize_korean_word failed for '%s': %s", word, e)
        return word
    out = (out or "").strip().strip('"').strip("'").strip()
    # Accept only a plausible result: short, Korean syllables (spaces allowed
    # for multi-word lexemes). Otherwise keep the original.
    if out and len(out) <= 20 and all(
        "가" <= c <= "힣" or c == " " for c in out
    ):
        return out
    return word


def check_sense_usage(
    word: str,
    existing_definitions: List[str],
    sentence: str,
    model: str = "gpt-5-mini",
) -> Dict[str, Any]:
    """Decide whether `word` as used in `sentence` is a sense already
    covered by `existing_definitions`, or an additional sense.

    Returns:
        {"same_sense": bool,
         "new_sense_definition": str,     # empty if same_sense
         "extended_definition": str}      # empty if same_sense

    Falls back conservatively (same_sense=True) on any failure or empty
    inputs, so we never wrongly flag a new sense.
    """
    if not existing_definitions or not sentence:
        return {"same_sense": True, "new_sense_definition": "", "extended_definition": ""}
    defs_block = "\n".join(f"  - {d}" for d in existing_definitions)
    instruction = (
        "You analyze Korean word usage. You will be given a Korean word, "
        "every English definition currently recorded for it, and a new "
        "sentence using the word. Decide whether the use in the new "
        "sentence is covered by any existing definition (same sense), or "
        "represents a sense not yet recorded (new/additional sense).\n\n"
        "Reply with ONLY a JSON object, no markdown fences, no other text. "
        "Schema:\n"
        '{"same_sense": true | false, '
        '"new_sense_definition": "<short English definition of how the word '
        'is used in the new sentence; empty string if same_sense is true>", '
        '"extended_definition": "<a combined definition covering every '
        'existing sense plus the new one, formatted as a numbered list; '
        'empty string if same_sense is true>"}'
    )
    user = (
        f"Korean word: {word}\n"
        f"Existing definition(s) recorded for this word:\n{defs_block}\n"
        f"New sentence using the word: {sentence}"
    )
    try:
        raw = llm_query.query_api(user, instruction, model=model, verbose=False)
    except Exception as e:
        logger.warning("check_sense_usage failed for '%s': %s", word, e)
        return {"same_sense": True, "new_sense_definition": "", "extended_definition": ""}
    raw = (raw or "").strip()
    # strip optional code fences
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip()
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    try:
        result = json.loads(raw)
    except Exception as e:
        logger.warning("check_sense_usage parse failed for '%s': %s; raw=%r",
                       word, e, raw[:200])
        return {"same_sense": True, "new_sense_definition": "", "extended_definition": ""}
    return {
        "same_sense": bool(result.get("same_sense", True)),
        "new_sense_definition": str(result.get("new_sense_definition", "") or ""),
        "extended_definition": str(result.get("extended_definition", "") or ""),
    }


_SENSE_SYSTEM = """You are a Korean lexicography assistant for English-speaking \
learners. Given ONE Korean lexeme, enumerate the DISTINCT senses a learner \
should study as SEPARATE cards.

SPLIT genuine homographs / unrelated meanings — distinct words that happen to \
share a spelling. The strongest signal is DIFFERENT HANJA: 분기 = 分期 "quarter" \
vs 分岐 "branch"; 이상 = 異常 "abnormal" vs 以上 "at least" vs 理想 "ideal".

Be CONSERVATIVE. MERGE into a single sense any nuances, figurative extensions, \
or shades of one core meaning. Most words have exactly ONE sense.
- Sino-Korean: split only when the senses have different Hanja.
- Native (no Hanja): split ONLY a true homograph that a dictionary lists as \
separate entries (e.g. 쓰다 = write / use / bitter / wear a hat). Do NOT split \
mere extensions of one verb (먹다 "eat / use up / take damage" is ONE sense; \
들다 "to hold / to enter / to cost" — judge by whether a learner would call \
them different words).

Return ONLY a JSON object, no markdown:
{"senses": [
  {"slug": "<short lowercase English handle, a-z0-9 + underscores, unique in list>",
   "definition_en": "<concise English definition of THIS sense, <=18 words>",
   "hanja": {"has_hanja": true, "hanja": "<chars>",
             "chars": [{"char": "<one>", "reading": "<Korean 음>", "meaning": "<gloss>",
                        "examples": ["<other common Korean word with this Hanja>", ...]}],
             "gloss": "<short literal gloss>"}
            | {"has_hanja": false}}
 ],
 "context_slug": "<slug of the sense used in the provided sentence, or null>"}

Rules:
- slug: a memorable English tag for the sense ("quarter", "branch"), unique.
- hanja: same rules as a Hanja breakdown — native/loanword senses → {"has_hanja": false};
  Sino-Korean senses → the character breakdown for THAT sense specifically.
- hanja reading/character consistency (CRITICAL): each character's standard
  Korean reading (음) MUST equal the corresponding syllable of the lexeme. Never
  use a similar-looking character whose reading differs (e.g. 자극 → 戟 read 극,
  NOT 激 read 격). Use the 표준국어대사전 dictionary hanja.
- hanja chars[].examples: 1-2 OTHER common Korean words genuinely WRITTEN with
  that exact Hanja character — not just same-sounding homophones (Korean has many
  Hanja per syllable). If unsure a word uses this exact character, omit it; []
  if none. Korean only, not the lexeme itself. e.g. 分 → ["분리","구분"].
- context_slug: only when a sentence is given — the slug whose sense matches that
  usage; otherwise null.
- Output ONLY the JSON object."""


def _slugify(s: str, fallback: str) -> str:
    """Lowercase ascii handle safe for an external_id sense tag."""
    out = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return out or fallback


def inventory_senses(
    word: str, context: str = "", model: str = "gpt-5-mini"
) -> Dict[str, Any]:
    """Enumerate the distinct senses of `word` (one LLM call), each with its own
    Hanja. Returns {"senses": [{slug, definition_en, hanja}], "context_slug": str|None}.

    Slugs are sanitized + de-duplicated. Raises on a hard failure so callers can
    fall back to the legacy single-sense define path."""
    from hanja import _parse_json, _normalize  # local: hanja imports llm_query only

    user = f"Lexeme: {word}"
    if context and context.strip():
        user += f"\nSentence the word appears in: {context.strip()}"
    resp = llm_query.client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SENSE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    data = _parse_json(resp.choices[0].message.content or "")
    raw_senses = data.get("senses") if isinstance(data, dict) else None
    if not isinstance(raw_senses, list) or not raw_senses:
        raise ValueError("no senses returned")

    senses: List[Dict[str, Any]] = []
    seen_slugs: set[str] = set()
    for i, s in enumerate(raw_senses):
        if not isinstance(s, dict):
            continue
        definition = str(s.get("definition_en", "") or "").strip()
        if not definition:
            continue
        slug = _slugify(str(s.get("slug", "")), f"sense{i+1}")
        while slug in seen_slugs:
            slug = f"{slug}_{i+1}"
        seen_slugs.add(slug)
        senses.append({
            "slug": slug,
            "definition_en": definition,
            "hanja": _normalize(s.get("hanja") if isinstance(s.get("hanja"), dict) else {}),
        })
    if not senses:
        raise ValueError("no usable senses after normalization")

    # Gate each sense's hanja through the deterministic reading check (Unihan);
    # repair any reading-inconsistent breakdown via the gated lookup_hanja
    # (which re-prompts and fails closed to no-hanja).
    import hanja_validate
    from hanja import lookup_hanja
    for s in senses:
        h = s.get("hanja")
        if isinstance(h, dict) and h.get("has_hanja") and not hanja_validate.validate_breakdown(h)["ok"]:
            s["hanja"] = lookup_hanja(word, gloss=s["definition_en"])

    context_slug = data.get("context_slug")
    if context_slug:
        context_slug = _slugify(str(context_slug), "")
        if context_slug not in seen_slugs:
            context_slug = None
    else:
        context_slug = None
    return {"senses": senses, "context_slug": context_slug}


def existing_senses(session: Session, headword: str) -> List[Dict[str, Any]]:
    """Recog bundles already in the deck for this headword — plain `lexeme:word`
    and every sense-tagged `lexeme:word#slug`. Returns
    [{key, slug, definition_en, hanja}] so the picker can mark/skip duplicates."""
    items = session.query(srs_db.Item).filter(
        srs_db.Item.deleted_at.is_(None),
        srs_db.Item.external_id.like(f"lexeme:{headword}:cloze_recog_bundle")
        | srs_db.Item.external_id.like(f"lexeme:{headword}{srs_db.SENSE_SEP}%:cloze_recog_bundle"),
    ).all()
    out: List[Dict[str, Any]] = []
    for it in items:
        parsed = srs_db.parse_external_id(it.external_id)
        if not parsed:
            continue
        key = parsed[0]
        content = it.content or {}
        variants = content.get("variants", [])
        definition = next((v.get("back") for v in variants if (v.get("back") or "").strip()), "")
        out.append({
            "key": key,
            "slug": srs_db.sense_of(key),
            "definition_en": (definition or "").strip(),
            "hanja": content.get("hanja") if isinstance(content.get("hanja"), dict) else None,
        })
    return out


def _sense_def_block(headword: str, definition_en: str, hanja: Dict[str, Any] | None) -> str:
    """A definition block scoped to ONE sense, so bundle generation stays on-sense."""
    lines = [headword, f"English: {definition_en}"]
    if hanja and hanja.get("has_hanja") and hanja.get("hanja"):
        lines.append(f"Hanja: {hanja['hanja']} ({hanja.get('gloss', '')})")
    lines.append(
        "SENSE CONSTRAINT: Generate example sentences using ONLY this exact "
        "sense of the word. Do NOT use the word in any other meaning."
    )
    return "\n".join(lines).strip() + "\n"


def generate_and_insert_senses(
    session: Session,
    *,
    prompts: dict,
    word: str,
    senses: List[Dict[str, Any]],
    tagged: bool,
    n_variants: int = 8,
    model: str = llm_query.BUNDLE_MODEL,
) -> tuple[bool, str, List[str]]:
    """Generate a recog-only bundle per selected sense and upsert.

    `senses`: [{slug, definition_en, hanja}]. `tagged`: if True the lexeme is
    polysemous so keys are `lexeme:word#slug`; if False (monosemous) the single
    sense is keyed plain `lexeme:word`. Each sense's Hanja is cached on content.
    New senses are inserted as fresh UNSEEN words (normal Day-0 path).

    Returns (ok, message, created_lexeme_keys)."""
    from backfill_translations import translate_variant

    created: List[str] = []
    for s in senses:
        slug, definition, hanja = s["slug"], s["definition_en"], s.get("hanja")
        def_block = _sense_def_block(word, definition, hanja)
        try:
            payload = llm_query._run_bundle(
                prompts, kind="create_static_cloze_recog_bundle",
                def_block=def_block, n_variants=n_variants, model=model,
            )
        except Exception as e:
            return (False, f"bundle generation failed for sense '{slug}': {e}", created)

        card = payload["cards"][0]
        # Use the inventory's per-sense hanja: it's assigned with full multi-sense
        # context, which disambiguates homographs more reliably than re-picking a
        # single sense via Tier-2's gloss (that mis-picks, e.g. 부채#fan -> 負債).
        # Single-form (non-homograph) words still gain dict authority via the
        # on-demand lookup_hanja path.
        card["hanja"] = hanja if isinstance(hanja, dict) else {"has_hanja": False}

        variants = card.get("variants", [])
        if variants:
            with ThreadPoolExecutor(max_workers=min(len(variants), 8)) as pool:
                futs = {
                    pool.submit(translate_variant, v.get("front", ""), v.get("back", ""),
                                "cloze_recog_bundle"): v
                    for v in variants
                }
                for fut in as_completed(futs):
                    v = futs[fut]
                    try:
                        v["translation_en"] = fut.result() or ""
                    except Exception:
                        v["translation_en"] = ""

        key = f"{word}{srs_db.SENSE_SEP}{slug}" if tagged else word
        ext = f"lexeme:{key}:cloze_recog_bundle"
        upsert_static_bundle(
            session, external_id=ext, card=card,
            tags=["static", "bundle", "cloze_recog", word] + ([f"sense:{slug}"] if tagged else []),
        )
        created.append(key)

    session.commit()
    label = ", ".join(created)
    return (True, f"Added {len(created)} sense(s): {label}", created)


def build_definition_block(parsed: Dict[str, Any]) -> str:
    """
    Convert llm_query.parse_vocab_block() output to the "definition block" format
    expected by your create_static_cloze_*_bundle prompts.
    """
    term = parsed["term"].strip()
    def_en = parsed["def_en"].strip()
    def_ko = parsed["def_ko"].strip()
    examples = parsed["examples"]

    lines = [
        term,
        f"English: {def_en}",
        f"Korean: {def_ko}",
        "Examples:",
    ]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"{i}. {ex.strip()}")
    return "\n".join(lines).strip() + "\n"


def upsert_static_bundle(
    sess,
    *,
    external_id: str,
    card: dict,
    tags: List[str],
) -> int:
    """
    Upsert one bundled static card as one Item row:
      - Item.item_type = "card"
      - Item.external_id = stable key
      - Item.content = the card object (type + variants)
    """
    existing = (
        sess.query(srs_db.Item)
        .filter(srs_db.Item.external_id == external_id)
        .one_or_none()
    )

    variants = card.get("variants") or []
    cached_front = variants[0].get("front") if variants else None
    cached_back = variants[0].get("back") if variants else None

    if existing is None:
        item = srs_db.Item(
            item_type="card",
            external_id=external_id,
            front=cached_front,
            back=cached_back,
            content=card,   # canonical
            tags=tags,
            suspended=False,
        )
        sess.add(item)
        sess.flush()

        sess.add(
            srs_db.SRSState(
                item_id=item.id,
                due_at=srs_db.now_utc(),
                scheduler_name=srs_db.SCHEDULER.name,
                scheduler_version=srs_db.SCHEDULER.version,
                state=srs_db.SCHEDULER.init_state(item),
            )
        )
        sess.flush()
        return item.id

    # update
    existing.front = cached_front
    existing.back = cached_back
    existing.content = card
    existing.tags = tags
    existing.updated_at = srs_db.now_utc()

    if existing.srs_state is None:
        sess.add(
            srs_db.SRSState(
                item_id=existing.id,
                due_at=srs_db.now_utc(),
                scheduler_name=srs_db.SCHEDULER.name,
                scheduler_version=srs_db.SCHEDULER.version,
                state=srs_db.SCHEDULER.init_state(existing),
            )
        )

    return existing.id


# ── query functions ─────────────────────────────────────────────────────


def word_exists(session: Session, word: str) -> bool:
    """Check if a lexeme already has cards in the DB."""
    existing = session.query(srs_db.Item).filter(
        srs_db.Item.external_id == f"lexeme:{word}:cloze_prod_bundle",
        srs_db.Item.deleted_at.is_(None),
    ).first()
    return existing is not None


def show_definition_and_confirm(
    prompts: dict,
    word: str,
) -> tuple[bool, dict | None, str | None]:
    """
    Step 1: generate definition via LLM (streams to stdout), parse it,
    and ask user to confirm.

    Returns (confirmed, parsed_dict, def_block) or (False, None, None).
    """
    print(f"\n  Generating definition for '{word}'...\n")
    try:
        raw_def = define_vocab_raw(prompts, word)
    except Exception as e:
        print(f"\n  Error generating definition: {e}")
        return (False, None, None)

    parsed = llm_query.parse_vocab_block(raw_def)
    if not parsed.get("ok"):
        print(f"\n  Error: definition parse failed: {parsed.get('error')}")
        return (False, None, None)

    def_block = build_definition_block(parsed)

    print(f"\n  Track this word? [y/n] ", end="", flush=True)
    answer = input().strip().lower()
    if answer != "y":
        return (False, None, None)

    return (True, parsed, def_block)


# ── canonical pipeline ──────────────────────────────────────────────────


def generate_and_insert(
    session: Session,
    *,
    prompts: dict,
    word: str,
    def_block: str,
    n_variants: int = 8,
    model: str = llm_query.BUNDLE_MODEL,
) -> tuple[bool, str]:
    """
    Generate prod+recog bundles, translate all variants, and upsert to DB.

    Single commit with translations included. Returns (success, message).
    """
    # 1. Generate prod + recog bundles in parallel
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_prod = pool.submit(
                llm_query._run_bundle,
                prompts,
                kind="create_static_cloze_prod_bundle",
                def_block=def_block,
                n_variants=n_variants,
                model=model,
            )
            fut_recog = pool.submit(
                llm_query._run_bundle,
                prompts,
                kind="create_static_cloze_recog_bundle",
                def_block=def_block,
                n_variants=n_variants,
                model=model,
            )
            payload_prod = fut_prod.result()
            payload_recog = fut_recog.result()
    except Exception as e:
        return (False, f"Bundle generation failed: {e}")

    card_prod = payload_prod["cards"][0]
    card_recog = payload_recog["cards"][0]
    headword = payload_prod["lexeme"]["headword"].strip() or word

    # 2. Translate ALL variants in parallel (in-memory, before DB insert)
    from backfill_translations import translate_variant

    all_variants = [
        (v, "cloze_prod_bundle") for v in card_prod.get("variants", [])
    ] + [
        (v, "cloze_recog_bundle") for v in card_recog.get("variants", [])
    ]

    if all_variants:
        with ThreadPoolExecutor(max_workers=len(all_variants)) as pool:
            future_to_variant = {
                pool.submit(
                    translate_variant,
                    v.get("front", ""), v.get("back", ""), skill_type,
                ): v
                for v, skill_type in all_variants
            }
            for fut in as_completed(future_to_variant):
                v = future_to_variant[fut]
                try:
                    v["translation_en"] = fut.result()
                except Exception as e:
                    logger.warning("translation failed for '%s': %s", word, e)

    # 3. Upsert to DB — single commit, translations already in variant dicts
    prod_eid = f"lexeme:{headword}:cloze_prod_bundle"
    recog_eid = f"lexeme:{headword}:cloze_recog_bundle"

    id_prod = upsert_static_bundle(
        session,
        external_id=prod_eid,
        card=card_prod,
        tags=["static", "bundle", "cloze_prod", headword],
    )
    id_recog = upsert_static_bundle(
        session,
        external_id=recog_eid,
        card=card_recog,
        tags=["static", "bundle", "cloze_recog", headword],
    )
    session.commit()

    return (True, f"Added '{headword}' (prod={id_prod}, recog={id_recog})")
