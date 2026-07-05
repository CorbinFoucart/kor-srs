#!/usr/bin/env python3
"""
hanja.py

On-demand Hanja (Sino-Korean) breakdown for a single lexeme.

`lookup_hanja(lexeme)` asks the LLM whether the word has a meaningful
Sino-Korean origin and, if so, returns its character-by-character
breakdown. Native Korean words (고유어), loanwords (외래어), and anything
without a useful Hanja interpretation return {"has_hanja": false}.

The returned dict is the exact sentinel stored in
Item.content["hanja"], so a cached lookup can be returned verbatim with
no LLM call. Shape:

  has hanja:
    {"has_hanja": true,
     "hanja": "受講料",
     "chars": [{"char": "受", "reading": "수", "meaning": "receive"},
               {"char": "講", "reading": "강", "meaning": "lecture"},
               {"char": "料", "reading": "료", "meaning": "fee, materials"}],
     "gloss": "fee for receiving lectures"}

  no hanja:
    {"has_hanja": false}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import llm_query


MODEL = "gpt-5-mini"

# ── Tier-2 data: authoritative word→hanja + frequency-ranked example words ──
_DICT_PATH = Path(__file__).parent / "hanja_dict.json"
_EXAMPLES_PATH = Path(__file__).parent / "hanja_examples.json"
_DICT: dict | None = None
_EXAMPLES: dict | None = None


def _dict_forms(word: str) -> list[str]:
    """Authoritative hanja spelling(s) for a hangul word, from Wiktionary."""
    global _DICT
    if _DICT is None:
        _DICT = json.loads(_DICT_PATH.read_text(encoding="utf-8")) if _DICT_PATH.exists() else {}
    return _DICT.get(word, [])


def _char_examples(char: str, exclude: str, n: int = 2) -> list[str]:
    """Top-frequency common words containing `char` (excluding `exclude`)."""
    global _EXAMPLES
    if _EXAMPLES is None:
        _EXAMPLES = json.loads(_EXAMPLES_PATH.read_text(encoding="utf-8")) if _EXAMPLES_PATH.exists() else {}
    return [w for w in _EXAMPLES.get(char, []) if w != exclude][:n]


_TIER2_SYSTEM = """You annotate a Korean Sino-Korean word whose Hanja is already \
known. You will be given the word, its intended sense, and one or more candidate \
Hanja spellings. Return ONLY a JSON object:
{"hanja": "<the correct Hanja spelling, characters only>",
 "chars": [{"char": "<one>", "meaning": "<concise English, 1-4 words>",
            "examples": ["<other common Korean word written with this char>", ...]}],
 "gloss": "<short literal gloss tying the characters together>"}

Rules:
- If several candidate spellings are given, CHOOSE the one matching the word's
  intended sense; if only one is given, use it.
- The chosen "hanja" MUST be one of the given candidates, unchanged.
- "chars" must be exactly the characters of the chosen hanja, in order — do not
  add, drop, or substitute characters. Only annotate them.
- examples: 1-2 OTHER common words genuinely WRITTEN with that exact character
  (not same-sounding homophones written with a different Hanja); [] if unsure.
- Output ONLY the JSON object."""

_SYSTEM = """You are a Korean–Hanja etymology assistant for English-speaking \
learners. Given ONE Korean lexeme, return its Sino-Korean (Hanja) breakdown \
as a single strict JSON object and nothing else. No markdown, no prose.

Decide first whether the word has a genuine, useful Hanja origin:
- Native Korean words (고유어) such as 먹다, 자다, 예쁘다, 뚜렷하다 → no hanja.
- Loanwords (외래어) such as 커피, 버스, 컴퓨터 → no hanja.
- Words with no meaningful or commonly-cited Hanja → no hanja.
For any of these, return exactly: {"has_hanja": false}

Otherwise (Sino-Korean / 한자어), return:
{"has_hanja": true,
 "hanja": "<the Hanja string, characters only>",
 "chars": [{"char": "<one Hanja>", "reading": "<Korean 음, one syllable>",
            "meaning": "<concise English gloss, 1-4 words>",
            "examples": ["<other common Korean word using this same Hanja>", ...]}, ...],
 "gloss": "<short literal gloss tying the characters together>"}

Rules:
- "chars" MUST align 1:1 with the characters in "hanja", in order.
- "reading" is the Korean reading (음) of that character AS USED in this word.
- CRITICAL — reading/character consistency: every character's STANDARD Korean
  reading (음) MUST equal the corresponding syllable of the lexeme as actually
  pronounced. Do NOT substitute a similar-looking or similar-meaning character
  whose reading differs from the syllable. If two characters are plausible,
  pick the one whose reading matches the syllable.
    e.g. 자극 is read 자·극 → the 극 syllable is 戟 (read 극). NEVER 激, which is
    read 격 (자극 ≠ 자격). Use the 표준국어대사전 dictionary hanja for the word.
  Self-check before answering: read your chosen hanja aloud character-by-
  character; the syllables must spell the lexeme exactly.
- "examples": 1-2 OTHER common Korean words that contain THIS EXACT Hanja
  character, to ground it to vocabulary the learner will meet. Korean word only
  (no English).
    CRITICAL: the example must genuinely be WRITTEN with this character — not
    merely a word that shares the syllable's SOUND. Korean has many homophones
    written with different Hanja (e.g. 상 = 常/想/上/商/賞…), so a same-sounding
    word usually does NOT contain this character. If you are not certain a word
    is written with this exact Hanja, omit it. Prefer [] over a guess. Do not
    repeat the lexeme itself.
    e.g. 不 (불상사) → ["불행", "불편"]; 事 (불상사) → ["사고", "행사"];
         改 (개선) → ["개혁", "개정"]; 善 (개선) → ["최선", "친선"].
- Verbs/adjectives ending in 하다 (e.g. 通하다 = 通하다): give Hanja only for the
  Sino-Korean morpheme(s); the native 하다 is not a Hanja character.
- Multi-word collocations: include only the Hanja-bearing morphemes; set
  has_hanja true if any meaningful Hanja exists; let "gloss" explain the whole.
- Output ONLY the JSON object."""


def _parse_json(raw: str) -> dict:
    """Extract the JSON object from a model reply (tolerates code fences)."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    # Fall back to the first {...} span if there's stray text.
    if not s.startswith("{"):
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            s = m.group(0)
    return json.loads(s)


def _normalize(data: dict) -> dict:
    """Coerce a model reply into the stored sentinel shape; be conservative —
    any malformed 'has hanja' payload degrades to {"has_hanja": false}."""
    if not isinstance(data, dict) or not data.get("has_hanja"):
        return {"has_hanja": False}
    hanja = (data.get("hanja") or "").strip()
    chars = data.get("chars") or []
    if not hanja or not isinstance(chars, list) or not chars:
        return {"has_hanja": False}
    clean_chars = []
    for c in chars:
        if not isinstance(c, dict):
            continue
        raw_ex = c.get("examples") or []
        examples = [str(e).strip() for e in raw_ex if isinstance(e, (str,)) and str(e).strip()] \
            if isinstance(raw_ex, list) else []
        char = (c.get("char") or "").strip()
        # Drop examples that can't actually contain this character (gross
        # homophone/wrong-word mismatches); keep up to 2.
        import hanja_validate
        examples = hanja_validate.filter_examples(char, examples)[:2]
        clean_chars.append({
            "char": char,
            "reading": (c.get("reading") or "").strip(),
            "meaning": (c.get("meaning") or "").strip(),
            "examples": examples,
        })
    clean_chars = [c for c in clean_chars if c["char"]]
    if not clean_chars:
        return {"has_hanja": False}
    return {
        "has_hanja": True,
        "hanja": hanja,
        "chars": clean_chars,
        "gloss": (data.get("gloss") or "").strip(),
    }


# Sino-derivation suffixes. 적(的) appends a character; the others are native
# endings, so the hanja aligns to the stem only.
_SUFFIX_APPEND = {"적": "的"}
_SUFFIX_STEM = ("하다", "되다", "시키다", "당하다", "스럽다", "롭다", "히", "이")


def _resolve_forms(lexeme: str) -> tuple[list[str], str]:
    """Find authoritative hanja candidates that align 1:1 with a base hangul
    string. Returns (forms, base). Handles derived words by looking up the stem:
    자극적 → 자극(刺戟) + 的; 통하다 → 통(通)."""
    forms = [f for f in _dict_forms(lexeme) if len(f) == len(lexeme)]
    if forms:
        return forms, lexeme
    for suf, hch in _SUFFIX_APPEND.items():
        if lexeme.endswith(suf) and len(lexeme) > len(suf):
            stem = lexeme[: -len(suf)]
            sf = [f + hch for f in _dict_forms(stem) if len(f) == len(stem)]
            if sf:
                return sf, lexeme  # base = full word (X적), aligns to X的
    for suf in _SUFFIX_STEM:
        if lexeme.endswith(suf) and len(lexeme) > len(suf):
            stem = lexeme[: -len(suf)]
            # only multi-syllable stems: a bare monosyllabic stem is often a
            # different word than the verb's morpheme (통하다 = 通하다, but the
            # standalone noun 통 = 桶). Multi-syllable Sino stems are reliable.
            if len(stem) < 2:
                continue
            sf = [f for f in _dict_forms(stem) if len(f) == len(stem)]
            if sf:
                return sf, stem  # base = stem; the native ending has no hanja
    return [], lexeme


def _tier2_lookup(lexeme: str, gloss: str | None, model: str) -> dict | None:
    """Authoritative dictionary path: take the characters from Wiktionary (never
    let the model invent them), the readings from the word's own syllables, and
    example words from the frequency-ranked index (LLM-supplemented only when the
    index is thin). The LLM only picks among real candidate spellings and writes
    English meanings. Returns a validated sentinel, or None to fall back."""
    import hanja_validate

    forms, base = _resolve_forms(lexeme)
    if not forms:
        return None

    user = (
        f"Word: {lexeme}\n"
        f"Intended sense: {gloss.strip() if gloss and gloss.strip() else '(main/most common sense)'}\n"
        f"Candidate Hanja: {', '.join(forms)}"
    )
    try:
        resp = llm_query.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _TIER2_SYSTEM},
                      {"role": "user", "content": user}],
        )
        data = _parse_json(resp.choices[0].message.content or "")
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None

    chosen = (data.get("hanja") or "").strip()
    if chosen not in forms:
        if len(forms) == 1:
            chosen = forms[0]
        else:
            return None
    if len(chosen) != len(base):
        return None

    by_char = {c.get("char"): c for c in (data.get("chars") or []) if c.get("char")}
    chars = []
    for i, ch in enumerate(chosen):
        c = by_char.get(ch, {})
        # examples: high-frequency index first; top up with LLM suggestions
        # (deterministically filtered) only if the index is thin.
        ex = _char_examples(ch, lexeme)
        if len(ex) < 2:
            extra = hanja_validate.filter_examples(ch, c.get("examples") or [])
            for e in extra:
                if e != lexeme and e not in ex:
                    ex.append(e)
                if len(ex) >= 2:
                    break
        chars.append({
            "char": ch,
            "reading": base[i],                    # the word's own syllable
            "meaning": (c.get("meaning") or "").strip(),
            "examples": ex[:2],
        })

    result = {"has_hanja": True, "hanja": chosen, "chars": chars,
              "gloss": (data.get("gloss") or "").strip()}
    # backstop: readings must still pass the deterministic, word-anchored check
    if not hanja_validate.validate_breakdown(result, lexeme)["ok"]:
        return None
    return _normalize(result)


def _upgrade_examples(result: dict, lexeme: str) -> None:
    """Replace each char's example words with high-frequency ones from the index
    (topped up with the model's own validated suggestions), so the LLM path gets
    the same frequency-ranked examples as the dictionary (Tier-2) path."""
    import hanja_validate
    for c in result.get("chars", []):
        ch = (c.get("char") or "").strip()
        if not ch:
            continue
        ex = _char_examples(ch, lexeme)                       # freq-ranked index
        for e in hanja_validate.filter_examples(ch, c.get("examples") or []):
            if e != lexeme and e not in ex:
                ex.append(e)
        c["examples"] = ex[:2]


def lookup_hanja(lexeme: str, model: str = MODEL, gloss: str | None = None,
                 max_retries: int = 2) -> dict:
    """LLM lookup of the Hanja breakdown for one lexeme, GATED by a deterministic
    reading check (hanja_validate / Unihan). Returns the stored sentinel dict.

    Every candidate is validated: each character's claimed reading must be a real
    Korean reading of that character (allowing 두음법칙). On a mismatch the model
    is re-prompted with the specific correction; if it still can't produce a
    reading-consistent breakdown after `max_retries`, we **fail closed** and
    return {"has_hanja": false} — better to show no Hanja than wrong Hanja.

    `gloss` disambiguates homographs (분기 = 分期 quarter / 分岐 branch).

    Tier 2 first: if the word is in the authoritative dictionary, take the
    characters from there (correct by construction) with frequency-ranked example
    words. Only words absent from the dictionary fall through to the gated LLM."""
    import hanja_validate

    # Tier-2 (authoritative word→hanja dict) is sense-BLIND: it picks a form for
    # the spelling, so for a glossed (sense-specific) lookup of a homograph it
    # mis-picks — 사기 "ceramic" → 士氣 (morale), 부채 "fan" (a native word) →
    # 負債 (debt). When a gloss is given we therefore go straight to the gated
    # LLM, which can read the sense (and word-anchored validation keeps it
    # honest). Tier-2 stays the fast, authoritative path for unglossed lookups.
    if not (gloss and gloss.strip()):
        t2 = _tier2_lookup(lexeme, gloss, model)
        if t2 is not None:
            return t2

    user = f"Lexeme: {lexeme}"
    if gloss and gloss.strip():
        user += (
            f"\nIntended sense (English meaning): {gloss.strip()}\n"
            "Return the Hanja for THIS sense specifically — this word may be a "
            "homograph with other senses that use different Hanja."
        )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]

    for attempt in range(max_retries + 1):
        try:
            resp = llm_query.client.chat.completions.create(model=model, messages=messages)
            raw = resp.choices[0].message.content or ""
            result = _normalize(_parse_json(raw))
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            return {"has_hanja": False}
        if not result.get("has_hanja"):
            return result  # native / no-hanja — nothing to validate
        check = hanja_validate.validate_breakdown(result, lexeme)
        if check["ok"]:
            _upgrade_examples(result, lexeme)
            return result
        if attempt == max_retries:
            break
        # re-prompt with the specific reading correction(s)
        fixes = "; ".join(
            f"'{i['char']}' is read {i['actual']}, NOT '{i['claimed']}' — replace it with "
            f"the character that is genuinely read '{i['claimed']}' for this word/meaning"
            for i in check["issues"]
        )
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                f"That breakdown has reading errors: {fixes}. Every character's "
                "standard Korean reading must match the word's syllables. "
                "Re-answer with the correct dictionary characters. Output ONLY the JSON object."
            )},
        ]

    print(f"[HANJA] {lexeme!r}: no reading-consistent breakdown after "
          f"{max_retries + 1} attempts — returning no-hanja")
    return {"has_hanja": False}
