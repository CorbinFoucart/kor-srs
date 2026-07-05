#!/usr/bin/env python3
"""
hanja_validate.py

Deterministic reading-consistency check for Hanja breakdowns, grounded in the
Unicode Unihan kHangul table (see build_hanja_readings.py). No LLM.

The check: for each character in a breakdown, the reading the card claims for it
must be one of that character's real Korean readings (Unihan), allowing the
초성 두음법칙 word-initial alternations (ㄹ→ㄴ/ㅇ, ㄴ→ㅇ). This catches the
dangerous error class — a wrong character whose reading doesn't match its
syllable (e.g. 激 claimed 극, but 激 reads 격 → reject).

It does NOT判断 whether a (validly-read) character is the right one for the
word's meaning (e.g. 假設 vs 假說 both read 가설) — that needs a dictionary.

Used both as a gate inside hanja.lookup_hanja and as a standalone DB audit:
    python hanja_validate.py --db test_srs.sqlite
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_HANGUL_RUN = re.compile(r"[가-힣]+")

_READINGS_PATH = Path(__file__).parent / "hanja_readings.json"
_READINGS: dict[str, list[str]] | None = None

# Hangul syllable composition constants
_CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")  # 19 initials
_L, _N, _NG = 5, 2, 11   # ㄹ, ㄴ, ㅇ indices in _CHO
# medials that trigger ㄹ/ㄴ → ㅇ (i/y onglides): ㅑㅒㅕㅖㅛㅠㅣ
_YI = {2, 3, 6, 7, 12, 17, 20}


def _load() -> dict[str, list[str]]:
    global _READINGS
    if _READINGS is None:
        _READINGS = json.loads(_READINGS_PATH.read_text(encoding="utf-8"))
    return _READINGS


def _decompose(syl: str):
    code = ord(syl) - 0xAC00
    if code < 0 or code > 11171:
        return None
    return code // (21 * 28), (code % (21 * 28)) // 28, code % 28


def _compose(cho: int, jung: int, jong: int) -> str:
    return chr(0xAC00 + (cho * 21 + jung) * 28 + jong)


def _dueum_variants(syl: str) -> set[str]:
    """Word-initial 두음법칙 variant(s) of a Sino reading."""
    d = _decompose(syl)
    if d is None:
        return set()
    cho, jung, jong = d
    out: set[str] = set()
    if cho == _L:                       # ㄹ-initial
        out.add(_compose(_NG if jung in _YI else _N, jung, jong))
    elif cho == _N and jung in _YI:     # ㄴ + i/y → ㅇ
        out.add(_compose(_NG, jung, jong))
    return out


def acceptable_readings(char: str) -> set[str] | None:
    """All readings we'll accept for `char` (Unihan + 두음 variants), or None if
    the character isn't in the Unihan Korean table (unverifiable)."""
    base = _load().get(char)
    if not base:
        return None
    acc = set(base)
    for r in base:
        acc |= _dueum_variants(r)
    return acc


def filter_examples(char: str, examples: list[str]) -> list[str]:
    """Drop example words that can't contain `char`: a word genuinely written
    with `char` must include a syllable read as one of its readings (incl. 두음).
    This catches gross mismatches (a word with no matching syllable at all). It
    cannot catch homophones — a same-sound word written with a *different* Hanja
    passes this check (that needs a word->hanja dictionary)."""
    # sanitize each example to its leading pure-hangul run — drops stray
    # annotations the LLM sometimes appends, e.g. "자상 (刺傷)" -> "자상".
    cleaned = []
    for e in (examples or []):
        m = _HANGUL_RUN.match((e or "").strip())
        if m:
            cleaned.append(m.group(0))
    acc = acceptable_readings(char)
    if acc is None:
        return cleaned  # unverifiable char — keep sanitized as-is
    return [e for e in cleaned if any(syl in acc for syl in e)]


def _hangul_syllables(word: str) -> list[str]:
    """The hangul syllables of `word`, in order (non-hangul chars dropped)."""
    return [ch for ch in (word or "") if _decompose(ch) is not None]


def validate_breakdown(hanja: dict, lexeme: str | None = None) -> dict:
    """Check a stored hanja sentinel. Returns
    {ok: bool, issues: [{char, claimed, actual}], unverifiable: [chars]}.
    `ok` is True only when there are zero reading mismatches (unverifiable
    characters do not make it False — we just can't check them).

    When `lexeme` is given, validation is WORD-ANCHORED: each Hanja character is
    aligned to the corresponding leading syllable of the word and must have that
    syllable among its acceptable readings (Unihan + 두음). This is what catches
    a self-consistent but wrong-word breakdown — e.g. 사기 → 磁器, which reads
    자기 (磁 is genuinely 자): every char is a valid reading of itself, but the
    word isn't 자기. Without `lexeme` we fall back to the weaker per-character
    check (claimed reading is *a* real reading of that character).

    Alignment is applied when the Hanja is a prefix of the word's syllables
    (pure-Sino words: equal counts; mixed words like 응시하다 = 凝視 + native
    하다: Hanja is the leading run). If there are more Hanja chars than syllables
    (shouldn't happen) we skip alignment and use the per-character check."""
    issues, unverifiable = [], []
    if not (isinstance(hanja, dict) and hanja.get("has_hanja")):
        return {"ok": True, "issues": [], "unverifiable": []}
    chars = hanja.get("chars", [])
    syllables = _hangul_syllables(lexeme) if lexeme else None
    anchored = bool(syllables) and 0 < len([c for c in chars if (c.get("char") or "").strip()]) <= len(syllables)
    for i, c in enumerate(chars):
        char = (c.get("char") or "").strip()
        claimed = (c.get("reading") or "").strip()
        if not char:
            continue
        acc = acceptable_readings(char)
        if acc is None:
            unverifiable.append(char)
            continue
        if anchored:
            want = syllables[i]                 # the word's syllable at this position
            if want not in acc:
                # claimed=want makes the re-prompt say "use the char read 'want'"
                issues.append({"char": char, "claimed": want, "actual": sorted(acc)})
        elif claimed not in acc:
            issues.append({"char": char, "claimed": claimed, "actual": sorted(acc)})
    return {"ok": not issues, "issues": issues, "unverifiable": unverifiable}


# ── standalone DB audit ─────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="test_srs.sqlite")
    args = ap.parse_args()

    import srs_db
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    sess = Session(srs_db.make_engine(args.db))
    rows = sess.scalars(
        select(srs_db.Item)
        .where(srs_db.Item.external_id.like("lexeme:%:cloze_recog_bundle"))
        .where(srs_db.Item.deleted_at.is_(None))
    ).all()

    n_checked = n_bad = n_unverif = 0
    bad = []
    for it in rows:
        h = (it.content or {}).get("hanja")
        if not (isinstance(h, dict) and h.get("has_hanja") and h.get("chars")):
            continue
        n_checked += 1
        res = validate_breakdown(h)
        if res["unverifiable"]:
            n_unverif += 1
        if not res["ok"]:
            n_bad += 1
            key = srs_db.parse_external_id(it.external_id)[0]
            bad.append((key, h.get("hanja"), res["issues"]))

    print(f"  checked {n_checked} cached hanja breakdown(s)")
    print(f"  reading-consistent: {n_checked - n_bad}    mismatches: {n_bad}"
          f"    (with unverifiable chars: {n_unverif})\n")
    if bad:
        print("  ══ READING MISMATCHES (deterministic, Unihan) ══")
        for key, hstr, issues in sorted(bad):
            detail = "; ".join(f"{i['char']} claimed '{i['claimed']}' but reads {i['actual']}" for i in issues)
            print(f"  {key:26} {hstr}   {detail}")
    else:
        print("  ✓ no reading mismatches.")
    sess.close()


if __name__ == "__main__":
    main()
