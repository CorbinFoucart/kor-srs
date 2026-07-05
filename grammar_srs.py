#!/usr/bin/env python3
"""
grammar_srs.py

SRS provider for Korean grammar points.  Handles item loading, scheduling,
LLM question generation/evaluation, prefetch threading, and grade persistence.

Grammar items use item_type="grammar" and store per-card prompts in content
JSON.  Questions are generated dynamically by the LLM on each review.
"""

from __future__ import annotations

import json
import random
import re
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

import srs_db
import llm_query
from incremental_model import (
    INITIAL_HALF_LIFE,
    SkillState,
    update_half_life,
    next_interval,
    recall_probability,
    GRADE_WEIGHT,
)

# ── helpers ──────────────────────────────────────────────────────────────

def _fmt(secs: float) -> str:
    """Human-readable duration."""
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}m"
    if secs < 86400:
        return f"{secs / 3600:.1f}h"
    return f"{secs / 86400:.1f}d"

# ── constants ────────────────────────────────────────────────────────────

GRAMMAR_SKILL = "grammar_production"
GRAMMAR_INITIAL_H = 368_500.0  # ~4.3 days → gives ~1 day interval at p*=0.85

DEFAULT_QUESTION_PROMPT = (
    "Generate a sentence completion question for the grammar pattern {pattern} "
    "({meaning_en}). Example: {example}\n\n"
    "Rules:\n"
    "- Give a Korean sentence with ___ where the pattern goes.\n"
    "- The blank ___ must replace the ENTIRE conjugated expression that uses the "
    "pattern, including the verb/adjective stem it attaches to. For example, if the "
    "pattern is -아/어 보이다 and the answer is 재미있어 보여요, the blank must replace "
    "ALL of 재미있어 보여요 — do NOT write 재미있어 ___ with part of the answer visible.\n"
    "- Do NOT include the pattern name or any hint after the blank. Just write ___.\n"
    "- CRITICAL: The sentence context must make {pattern} the ONLY natural choice. "
    "Avoid situations where a similar pattern (e.g., -아/어서 vs -(으)니까, -겠- vs "
    "-(으)ㄹ 거예요) would also be acceptable. The English translation and surrounding "
    "context should unambiguously require {pattern}.\n"
    "- Use only simple TOPIK I level vocabulary. Sentences can be complex.\n"
    "- Vary scenarios across questions.\n\n"
    "Output exactly two lines:\n"
    "<English translation of the full sentence>\n"
    "<Korean sentence with ___>"
)

DEFAULT_EVAL_PROMPT = (
    "Evaluate this Korean grammar answer. "
    "Pattern: {pattern}. Question: {question}. Answer: {answer}. "
    'Respond in JSON: {{"correct": bool, "feedback": "...", "corrected": "..."}}.\n'
    "Rules:\n"
    "- correct=true if the target grammar pattern was applied correctly. Ignore minor "
    "spelling or spacing errors.\n"
    '- If correct, set feedback to "" and corrected to "". Say nothing.\n'
    "- If the user used a DIFFERENT but grammatically valid pattern (e.g., -아/어서 "
    "instead of -(으)니까), mark correct=true but set feedback to a brief explanation "
    "of the target pattern {pattern} and why it fits this context. Leave corrected "
    'to "".\n'
    "- If wrong, set feedback to a brief English explanation of the error, and corrected "
    "to the corrected Korean sentence."
)

# ── question type prompts ────────────────────────────────────────────────

CONTEXT_CHOICE_PROMPT = (
    "Generate a multiple-choice question testing which Korean grammar pattern fits a "
    "given context.\n\n"
    "Patterns:\n{pattern_list}\n\n"
    "Meanings:\n{meaning_list}\n\n"
    "Rules:\n"
    "- Write a Korean sentence with ___ where ONE of these patterns fits naturally.\n"
    "- The blank ___ must replace the ENTIRE conjugated expression that uses the "
    "pattern, including the verb/adjective stem it attaches to. For example, if the "
    "pattern is -아/어 보이다 and the answer is 재미있어 보여요, the blank must replace "
    "ALL of 재미있어 보여요 — do NOT write 재미있어 ___ with part of the answer visible.\n"
    "- Use only simple TOPIK I level vocabulary.\n"
    "- The correct answer should be {pattern} ({meaning_en}).\n"
    "- Do NOT reveal which pattern is correct.\n\n"
    "Output exactly:\n"
    "<English translation of the full sentence>\n"
    "<Korean sentence with ___>\n"
    "Choices: <comma-separated list of the patterns>"
)

CHOICE_EVAL_PROMPT = (
    "Evaluate this Korean grammar multiple-choice answer.\n"
    "The correct pattern is: {pattern}.\n"
    "Question: {question}\n"
    "User's answer: {answer}\n"
    'Respond in JSON: {{"correct": bool, "feedback": "...", "corrected": "..."}}.\n'
    "Rules:\n"
    "- correct=true if the user selected or wrote the correct pattern ({pattern}). "
    "Accept the pattern itself or a full sentence using it.\n"
    '- If correct, set feedback to "" and corrected to "".\n'
    "- If wrong, explain briefly in English why the correct pattern fits better."
)

MINIMAL_PAIR_PROMPT = (
    "Generate a question that tests the difference between two similar Korean grammar "
    "patterns.\n\n"
    "Pattern A: {pattern_A} ({meaning_A})\n"
    "Pattern B: {pattern_B} ({meaning_B})\n\n"
    "Rules:\n"
    "- Describe a specific situation in English where ONLY Pattern A ({pattern_A}) is "
    "appropriate, not Pattern B.\n"
    "- Do NOT reveal which pattern is correct. The learner must choose.\n"
    "- Use simple, clear English for the situation description.\n\n"
    "Output exactly:\n"
    "Situation: <description of the situation>\n"
    "Which pattern fits here — {pattern_A} or {pattern_B}? Explain why."
)

MINIMAL_PAIR_EVAL_PROMPT = (
    "Evaluate this answer about the difference between two Korean grammar patterns.\n"
    "The correct answer is Pattern A: {pattern_A} ({meaning_A}).\n"
    "Pattern B: {pattern_B} ({meaning_B})\n"
    "Question: {question}\n"
    "User's answer: {answer}\n"
    'Respond in JSON: {{"correct": bool, "feedback": "...", "corrected": "..."}}.\n'
    "Rules:\n"
    "- The student is expected to answer in English. Accept English explanations.\n"
    "- correct=true ONLY if the user (1) chose Pattern A ({pattern_A}) as the correct "
    "pattern AND (2) demonstrated understanding of WHY it fits and Pattern B does not, "
    "even if the explanation is imperfect.\n"
    "- If the user chose Pattern B, mark correct=false regardless of their explanation.\n"
    '- If correct, set feedback to "" and corrected to "".\n'
    "- If wrong, give a brief English explanation of the actual difference."
)

REWRITE_PROMPT = (
    "Generate a sentence rewriting exercise for the grammar pattern {pattern} "
    "({meaning_en}).\n\n"
    "Rules:\n"
    "- Think of a Korean sentence that does NOT use {pattern}, but whose meaning "
    "can be naturally expressed using {pattern}.\n"
    "- The sentence must express a meaning where rewriting with {pattern} is natural "
    "and does not change what is being said.\n"
    "- Use only simple TOPIK I level vocabulary.\n"
    "- Do NOT include the answer or rewritten sentence.\n\n"
    "Output exactly two lines:\n"
    "<Korean sentence WITHOUT the pattern>\n"
    "Rewrite to mean: <English translation of the sentence rewritten with {pattern}>"
)

REWRITE_EVAL_PROMPT = (
    "Evaluate this Korean sentence rewrite.\n"
    "Target pattern: {pattern} ({meaning_en}).\n"
    "Question: {question}\n"
    "User's rewrite: {answer}\n"
    'Respond in JSON: {{"correct": bool, "feedback": "...", "corrected": "..."}}.\n'
    "Rules:\n"
    "- correct=true if the user correctly used {pattern} and the rewrite preserves the "
    "meaning of the original sentence. Ignore minor spelling or spacing errors.\n"
    "- Accept any natural rewrite that uses the pattern correctly, even if it differs "
    "from the reference answer.\n"
    "- If the user rewrote with a DIFFERENT but grammatically valid pattern (e.g., "
    "-아/어서 instead of -(으)니까), mark correct=true but set feedback to a brief "
    "explanation of the target pattern {pattern} and why it fits this context. Leave "
    'corrected to "".\n'
    '- If correct, set feedback to "" and corrected to "".\n'
    "- If wrong, explain the error briefly in English and provide the corrected rewrite."
)

ERROR_CORRECTION_PROMPT = (
    "Generate an error correction question for the grammar pattern {pattern} "
    "({meaning_en}).\n\n"
    "Rules:\n"
    "- Write a Korean sentence that INCORRECTLY uses {pattern}.\n"
    "- The error should be a plausible mistake a learner would make (wrong conjugation, "
    "wrong attachment, wrong context, etc.).\n"
    "- Use only simple TOPIK I level vocabulary.\n"
    "- Include an English translation of what the sentence SHOULD mean (the correct "
    "intended meaning), so the learner can compare the broken Korean against the "
    "intended meaning.\n\n"
    "Output exactly two lines:\n"
    "<English translation of what the sentence should mean>\n"
    "Find and correct the error: <Korean sentence with a grammar error involving {pattern}>"
)

ERROR_CORRECTION_EVAL_PROMPT = (
    "Evaluate this Korean grammar error correction.\n"
    "Pattern: {pattern} ({meaning_en}).\n"
    "Question: {question}\n"
    "User's correction: {answer}\n"
    'Respond in JSON: {{"correct": bool, "feedback": "...", "corrected": "..."}}.\n'
    "Rules:\n"
    "- correct=true if the user identified and fixed the grammar error correctly. "
    "Ignore minor spelling or spacing errors unrelated to the target pattern.\n"
    '- If correct, set feedback to "" and corrected to "".\n'
    "- If wrong, explain the actual error and provide the corrected sentence."
)

EVIDENCE_IDENTIFICATION_PROMPT = (
    "Generate an evidence identification question for the grammar pattern {pattern} "
    "({meaning_en}).\n\n"
    "Rules:\n"
    "- Write a Korean sentence using {pattern} that clearly implies some observable "
    "evidence (something the speaker saw, heard, or noticed).\n"
    "- Do NOT explicitly state the evidence in the sentence.\n"
    "- The student must infer what evidence the speaker is probably relying on.\n"
    "- Use only simple TOPIK I level vocabulary.\n\n"
    "Output exactly two lines:\n"
    "<Korean sentence using {pattern}>\n"
    "What evidence is the speaker probably relying on?"
)

EVIDENCE_IDENTIFICATION_EVAL_PROMPT = (
    "Evaluate this answer about inferred evidence for a Korean conjecture pattern.\n"
    "Pattern: {pattern} ({meaning_en}).\n"
    "Question: {question}\n"
    "User's answer: {answer}\n"
    'Respond in JSON: {{"correct": bool, "feedback": "...", "corrected": "..."}}.\n'
    "Rules:\n"
    "- The student is expected to answer in English. Accept English explanations.\n"
    "- correct=true if the user identified plausible observable evidence that logically "
    "supports the conjecture expressed in the sentence.\n"
    "- Accept any reasonable evidence, even if it differs from what you had in mind, as "
    "long as it logically connects to the conjecture.\n"
    '- If correct, set feedback to "" and corrected to "".\n'
    "- If wrong, explain the missing logical connection and suggest what evidence "
    "the speaker was likely relying on."
)

USAGE_CONSTRAINT_PROMPT = (
    "Generate a usage constraint question for the grammar pattern {pattern} "
    "({meaning_en}).\n\n"
    "Rules:\n"
    "- Write a Korean sentence that uses {pattern} in a grammatically INCORRECT way.\n"
    "- The error must relate to a usage restriction or speaker perspective constraint "
    "(e.g., wrong subject person, wrong tense context, wrong speech act type), "
    "NOT a simple conjugation or spelling error.\n"
    "- Use only simple TOPIK I level vocabulary.\n"
    "- Ask the student to explain why the sentence is incorrect.\n\n"
    "Output exactly two lines:\n"
    "<Incorrect Korean sentence using {pattern}>\n"
    "Why is this sentence incorrect?"
)

USAGE_CONSTRAINT_EVAL_PROMPT = (
    "Evaluate this explanation of why a Korean grammar usage is incorrect.\n"
    "Pattern: {pattern} ({meaning_en}).\n"
    "Question: {question}\n"
    "User's explanation: {answer}\n"
    'Respond in JSON: {{"correct": bool, "feedback": "...", "corrected": "..."}}.\n'
    "Rules:\n"
    "- The student is expected to answer in English. Accept English explanations.\n"
    "- correct=true if the user correctly identified the specific usage constraint or "
    "perspective rule that was violated, even if the explanation is imperfect.\n"
    '- If correct, set feedback to "" and corrected to "".\n'
    "- If wrong, explain the specific constraint that was violated and provide "
    "the corrected sentence."
)

# ── question type registry ───────────────────────────────────────────────

QUESTION_TYPE_DEFAULTS = {
    "sentence_completion": {
        "question_prompt": DEFAULT_QUESTION_PROMPT,
        "eval_prompt": DEFAULT_EVAL_PROMPT,
    },
    "context_choice": {
        "question_prompt": CONTEXT_CHOICE_PROMPT,
        "eval_prompt": CHOICE_EVAL_PROMPT,
    },
    "minimal_pair": {
        "question_prompt": MINIMAL_PAIR_PROMPT,
        "eval_prompt": MINIMAL_PAIR_EVAL_PROMPT,
    },
    "rewrite": {
        "question_prompt": REWRITE_PROMPT,
        "eval_prompt": REWRITE_EVAL_PROMPT,
    },
    "error_correction": {
        "question_prompt": ERROR_CORRECTION_PROMPT,
        "eval_prompt": ERROR_CORRECTION_EVAL_PROMPT,
    },
    "evidence_identification": {
        "question_prompt": EVIDENCE_IDENTIFICATION_PROMPT,
        "eval_prompt": EVIDENCE_IDENTIFICATION_EVAL_PROMPT,
    },
    "usage_constraint": {
        "question_prompt": USAGE_CONSTRAINT_PROMPT,
        "eval_prompt": USAGE_CONSTRAINT_EVAL_PROMPT,
    },
}

SCENARIO_POOL = [
    "daily routine or morning habits",
    "ordering food at a restaurant",
    "making weekend plans with a friend",
    "talking about the weather",
    "describing a family member",
    "shopping for clothes",
    "visiting a doctor",
    "asking for directions",
    "talking about a movie or TV show",
    "describing your hobby",
    "making a phone call",
    "talking about school or studying",
    "planning a trip or vacation",
    "cooking or making food at home",
    "exercising or playing sports",
    "talking about work or a meeting",
    "birthday party or celebration",
    "moving to a new house",
    "riding public transportation",
    "talking about a pet",
    "cleaning or doing housework",
    "sending a package at the post office",
    "meeting someone for the first time",
    "giving advice to a friend",
    "talking about yesterday's events",
    "talking about future goals or dreams",
    "describing a problem or complaint",
    "comparing two things",
    "apologizing or making excuses",
    "expressing surprise or disappointment",
    "preparing for a job interview",
    "asking your boss for time off",
    "explaining a project to a coworker",
    "attending a company dinner or 회식",
    "emailing a colleague about a deadline",
    "giving a presentation at work",
    "dealing with a difficult client",
    "onboarding at a new job",
    "discussing a promotion or raise",
    "scheduling a meeting with your team",
]

GRAMMAR_DEFINE_PROMPT = (
    "You are a Korean grammar instructor. The user will give you a Korean grammar pattern. "
    "Provide a clear, concise explanation in this exact format:\n\n"
    "PATTERN: <the pattern>\n"
    "MEANING_EN: <English meaning, 1 line>\n"
    "MEANING_KO: <Korean explanation, 1 line>\n"
    "EXAMPLE: <one natural example sentence using the pattern>\n"
    "NOTES: <brief usage notes, e.g., what it attaches to, formality level>"
)


# ── helpers ──────────────────────────────────────────────────────────────

def pattern_exists(session: Session, pattern: str) -> bool:
    """Check if a grammar pattern already has an item in the DB."""
    normalized = pattern.replace(" ", "_")
    existing = session.query(srs_db.Item).filter(
        srs_db.Item.external_id == f"grammar:{normalized}",
        srs_db.Item.deleted_at.is_(None),
    ).first()
    return existing is not None


def parse_grammar_definition(text: str) -> dict:
    """Parse structured LLM grammar definition output."""
    result = {}
    for key in ("PATTERN", "MEANING_EN", "MEANING_KO", "EXAMPLE", "NOTES"):
        m = re.search(rf"(?im)^\s*{key}\s*:\s*(.+?)\s*$", text)
        if m:
            result[key.lower()] = m.group(1).strip()
    return result


# ── grammar skill helpers ────────────────────────────────────────────────

def compute_grammar_due_at(
    skill_states: dict[str, SkillState], now: dt.datetime
) -> dt.datetime:
    """Earliest due time across all question-type skills.

    Never-reviewed skills are immediately due (return ``now``).
    """
    earliest = None
    for ss in skill_states.values():
        if ss.last_reviewed_at is None:
            return now  # never reviewed → due immediately
        interval = next_interval(ss.half_life_secs, GRAMMAR_SKILL)
        due = ss.last_reviewed_at + dt.timedelta(seconds=interval)
        if earliest is None or due < earliest:
            earliest = due
    return earliest if earliest is not None else now


def pick_question_type(
    skill_states: dict[str, SkillState], now: dt.datetime,
    *, skip: str | None = None,
) -> str:
    """Return the question type with the lowest current recall probability.

    Never-reviewed types (last_reviewed_at is None) are returned immediately.
    If *skip* is given, that question type is excluded from consideration
    (used by prefetch to target the *next* question type).
    Fallback: ``"sentence_completion"``.
    """
    best_qtype = None
    best_recall = 2.0  # > 1.0 so any real value wins

    for qtype, ss in skill_states.items():
        if qtype == skip:
            continue
        if ss.last_reviewed_at is None:
            return qtype
        delta_t = (now - ss.last_reviewed_at).total_seconds()
        p = recall_probability(ss.half_life_secs, delta_t)
        if p < best_recall:
            best_recall = p
            best_qtype = qtype

    return best_qtype or "sentence_completion"


# ── data class ───────────────────────────────────────────────────────────

@dataclass
class GrammarEntry:
    item_id: int
    pattern: str
    meaning_en: str
    content: dict
    skill_states: dict[str, SkillState]    # keyed by question type name
    due_at: dt.datetime
    current_question_type: str | None = None


# ── main class ───────────────────────────────────────────────────────────

class GrammarSRS:
    """
    Manages grammar review sessions.

    Each grammar item stores its own question-generation and evaluation
    prompts in content JSON.  Questions are generated via LLM on each
    review, with background prefetching to eliminate wait time.
    """

    def __init__(self, session: Session):
        self._session = session
        self._entries: list[GrammarEntry] = []
        self._total_graded = 0
        self._correct_count = 0

        # prefetch
        self._prefetch_pool = ThreadPoolExecutor(max_workers=1)
        self._prefetch_future: Future | None = None
        self._prefetch_item_id: int | None = None

        self._load_items()

    # ── item loading ─────────────────────────────────────────────────

    def _load_items(self) -> None:
        """Load all grammar items from DB and build entry list."""
        items = self._session.scalars(
            select(srs_db.Item)
            .options(joinedload(srs_db.Item.srs_state))
            .where(srs_db.Item.item_type == "grammar")
            .where(srs_db.Item.suspended == False)  # noqa: E712
            .where(srs_db.Item.deleted_at.is_(None))
        ).unique().all()

        now = srs_db.now_utc()
        for item in items:
            content = item.content or {}
            pattern = content.get("pattern", item.front or "")
            meaning_en = content.get("meaning_en", item.back or "")

            # load per-question-type skill states
            skill_states: dict[str, SkillState] = {}
            if item.srs_state and item.srs_state.state:
                ss_data = item.srs_state.state.get("skill_states")
                if ss_data and isinstance(ss_data, dict):
                    for qtype, qdata in ss_data.items():
                        lr_raw = qdata.get("last_reviewed_at")
                        lr = None
                        if lr_raw:
                            lr = dt.datetime.fromisoformat(lr_raw)
                            if lr.tzinfo is None:
                                lr = lr.replace(tzinfo=dt.timezone.utc)
                        skill_states[qtype] = SkillState(
                            half_life_secs=qdata.get(
                                "half_life_secs", INITIAL_HALF_LIFE
                            ),
                            last_reviewed_at=lr,
                            unlocked=True,
                        )
                # else: old format or empty → fresh start (skill_states stays {})

            # ensure every configured question type has a SkillState entry
            for qt_cfg in content.get("question_types", []):
                qt_name = qt_cfg.get("type", "sentence_completion")
                if qt_name not in skill_states:
                    skill_states[qt_name] = SkillState(
                        half_life_secs=GRAMMAR_INITIAL_H, unlocked=True
                    )

            # fallback: if no question types configured at all, seed sentence_completion
            if not skill_states:
                skill_states["sentence_completion"] = SkillState(
                    half_life_secs=GRAMMAR_INITIAL_H, unlocked=True
                )

            # compute due_at across all question-type skills
            due_at = compute_grammar_due_at(skill_states, now)

            self._entries.append(
                GrammarEntry(
                    item_id=item.id,
                    pattern=pattern,
                    meaning_en=meaning_en,
                    content=content,
                    skill_states=skill_states,
                    due_at=due_at,
                )
            )

        self._entries.sort(key=lambda e: e.due_at)

    def reload_items(self) -> None:
        """Clear and reload all grammar items from DB."""
        self._entries.clear()
        self._load_items()

    # ── queries ──────────────────────────────────────────────────────

    def total_patterns(self) -> int:
        return len(self._entries)

    def due_count(self) -> int:
        """Count individual question-type skills that are due across all patterns."""
        now = srs_db.now_utc()
        count = 0
        for e in self._entries:
            for ss in e.skill_states.values():
                if ss.last_reviewed_at is None:
                    count += 1
                else:
                    interval = next_interval(ss.half_life_secs, GRAMMAR_SKILL)
                    due = ss.last_reviewed_at + dt.timedelta(seconds=interval)
                    if due <= now:
                        count += 1
        return count

    def seconds_until_next_due(self) -> float:
        if not self._entries:
            return 0
        now = srs_db.now_utc()
        nearest = min(e.due_at for e in self._entries)
        gap = (nearest - now).total_seconds()
        return max(0.0, gap)

    def is_empty(self) -> bool:
        return not self._entries

    def next_due_entry(self) -> GrammarEntry | None:
        now = srs_db.now_utc()
        due = [e for e in self._entries if e.due_at <= now]
        if not due:
            return None
        due.sort(key=lambda e: e.due_at)
        return due[0]

    # ── helpers ───────────────────────────────────────────────────────

    def _find_entry_by_pattern(self, pattern: str) -> GrammarEntry | None:
        """Look up a GrammarEntry by pattern string."""
        for e in self._entries:
            if e.pattern == pattern:
                return e
        return None

    def _resolve_template_vars(
        self, entry: GrammarEntry, qtype_config: dict
    ) -> dict:
        """Build template variables dict based on question type."""
        content = entry.content
        base_vars = {
            "pattern": entry.pattern,
            "meaning_en": entry.meaning_en,
            "meaning_ko": content.get("meaning_ko", ""),
            "example": content.get("example", ""),
        }

        qtype = qtype_config.get("type", "sentence_completion")

        if qtype == "context_choice":
            related = qtype_config.get("related_patterns", [])
            all_patterns = [entry.pattern] + related
            pattern_lines = []
            meaning_lines = []
            for p in all_patterns:
                other = self._find_entry_by_pattern(p)
                if other:
                    pattern_lines.append(other.pattern)
                    meaning_lines.append(f"{other.pattern}: {other.meaning_en}")
                else:
                    pattern_lines.append(p)
                    meaning_lines.append(f"{p}: (unknown)")
            random.shuffle(pattern_lines)
            base_vars["pattern_list"] = "\n".join(f"- {p}" for p in pattern_lines)
            base_vars["meaning_list"] = "\n".join(f"- {m}" for m in meaning_lines)

        elif qtype == "minimal_pair":
            partner_name = qtype_config.get("partner_pattern", "")
            partner = self._find_entry_by_pattern(partner_name)
            base_vars["pattern_A"] = entry.pattern
            base_vars["meaning_A"] = entry.meaning_en
            base_vars["pattern_B"] = partner_name
            base_vars["meaning_B"] = partner.meaning_en if partner else "(unknown)"

        elif qtype == "rewrite":
            base_vars["target_pattern"] = entry.pattern

        # error_correction and sentence_completion use base_vars only

        return base_vars

    # ── LLM question generation ─────────────────────────────────────

    def _generate_question(self, entry: GrammarEntry) -> dict:
        """Generate a practice question via LLM (non-streaming).

        Picks a random question type from the item's ``question_types`` list
        (falling back to sentence_completion).  Returns a self-contained dict
        that includes the resolved ``eval_prompt`` for later evaluation.
        """
        content = entry.content
        now = srs_db.now_utc()

        # pick question type by lowest recall probability
        # skip= avoids regenerating the same type during prefetch
        qtype_name = pick_question_type(entry.skill_states, now, skip=entry.current_question_type)
        entry.current_question_type = qtype_name

        # find matching config from content
        qtypes = content.get("question_types", [])
        qtype_config = {"type": qtype_name}
        for qt in qtypes:
            if qt.get("type") == qtype_name:
                qtype_config = qt
                break

        # resolve prompts: per-type override > registry > legacy content field
        registry = QUESTION_TYPE_DEFAULTS.get(qtype_name, QUESTION_TYPE_DEFAULTS["sentence_completion"])
        question_prompt_template = qtype_config.get(
            "question_prompt",
            registry["question_prompt"],
        )
        eval_prompt_template = qtype_config.get(
            "eval_prompt",
            registry["eval_prompt"],
        )

        # for legacy patterns without question_types, honour the content-level overrides
        if not qtypes and qtype_name == "sentence_completion":
            question_prompt_template = content.get("question_prompt", question_prompt_template)
            eval_prompt_template = content.get("eval_prompt", eval_prompt_template)

        # build template vars for this question type
        tvars = self._resolve_template_vars(entry, qtype_config)

        try:
            prompt = question_prompt_template.format(**tvars)
        except KeyError:
            prompt = question_prompt_template

        # append learner context if present
        ctx = content.get("context", {})
        if ctx:
            prompt += f"\n\nLearner context: {json.dumps(ctx, ensure_ascii=False)}"

        scenario = random.choice(SCENARIO_POOL)
        resp = llm_query.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Generate a practice question for: {entry.pattern}\n"
                        f"Scenario: {scenario}"
                    ),
                },
            ],
        )
        return {
            "item_id": entry.item_id,
            "question": resp.choices[0].message.content,
            "question_type": qtype_name,
            "eval_prompt_template": eval_prompt_template,
            "eval_template_vars": tvars,
            "pattern": entry.pattern,
            "meaning_en": entry.meaning_en,
        }

    def _evaluate_answer(
        self, entry: GrammarEntry, question_data: dict, answer: str
    ) -> dict:
        """Evaluate user answer via LLM (non-streaming, JSON response).

        ``question_data`` is the dict returned by ``_generate_question``,
        which carries the eval prompt template and template vars separately
        so we can do a single ``.format()`` call (avoiding double-brace issues).
        """
        question = question_data.get("question", "")
        eval_template = question_data.get("eval_prompt_template", DEFAULT_EVAL_PROMPT)
        tvars = question_data.get("eval_template_vars", {})

        try:
            prompt = eval_template.format(question=question, answer=answer, **tvars)
        except KeyError:
            prompt = eval_template

        resp = llm_query.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Evaluate the answer."},
            ],
            response_format={"type": "json_object"},
        )
        try:
            result = json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, AttributeError):
            result = {
                "correct": False,
                "score": 5,
                "feedback": "Could not parse evaluation.",
                "corrected": "",
                "pattern_used": False,
            }
        return result

    # ── prefetch ─────────────────────────────────────────────────────

    def _start_prefetch(self, entry: GrammarEntry) -> None:
        """Start background generation of next question."""
        self._prefetch_item_id = entry.item_id
        self._prefetch_future = self._prefetch_pool.submit(
            self._generate_question, entry
        )

    def get_question(self, entry: GrammarEntry) -> dict:
        """Get question from prefetch if available, else generate sync."""
        if (
            self._prefetch_future is not None
            and self._prefetch_item_id == entry.item_id
        ):
            result = self._prefetch_future.result()
            self._prefetch_future = None
            self._prefetch_item_id = None
            return result
        # discard stale prefetch
        self._prefetch_future = None
        self._prefetch_item_id = None
        return self._generate_question(entry)

    def evaluate_answer(
        self, entry: GrammarEntry, question_data: dict, answer: str
    ) -> dict:
        """Evaluate answer via LLM (sync)."""
        return self._evaluate_answer(entry, question_data, answer)

    def prefetch_next(self, *, current_entry: GrammarEntry | None = None) -> None:
        """Start prefetch for whatever entry is coming next.

        Called after showing a question (with ``current_entry`` set so the
        prefetch can target the same entry's *next* question type) or after
        grading (no args, targets whatever is next due).  Skips if a
        matching prefetch is already in flight.
        """
        if current_entry is not None:
            # check if the current entry has more unreviewed question types
            has_more = any(
                ss.last_reviewed_at is None
                for qtype, ss in current_entry.skill_states.items()
                if qtype != current_entry.current_question_type
            )
            if has_more:
                target = current_entry
            else:
                target = self._peek_next(exclude_id=current_entry.item_id)
        else:
            target = self._peek_next()

        if target is None:
            return
        # don't clobber an existing prefetch for the same item
        if (
            self._prefetch_future is not None
            and self._prefetch_item_id == target.item_id
        ):
            return
        self._start_prefetch(target)

    def _peek_next(
        self, *, exclude_id: int | None = None
    ) -> GrammarEntry | None:
        """Find the next entry to review: first due item, or soonest upcoming."""
        candidates = [
            e for e in self._entries
            if exclude_id is None or e.item_id != exclude_id
        ]
        if not candidates:
            return None
        now = srs_db.now_utc()
        due = [e for e in candidates if e.due_at <= now]
        if due:
            due.sort(key=lambda e: e.due_at)
            return due[0]
        # nothing due — prefetch for the soonest upcoming
        return min(candidates, key=lambda e: e.due_at)

    # ── grading / persistence ────────────────────────────────────────

    def submit_grade(
        self,
        entry: GrammarEntry,
        grade: int,
        question: str,
        answer: str,
        llm_score: int,
        question_type: str = "sentence_completion",
    ) -> None:
        """Update half-life for the specific question type, persist SRSState + ReviewLog."""
        now = srs_db.now_utc()

        # look up the skill state for this specific question type
        if question_type not in entry.skill_states:
            entry.skill_states[question_type] = SkillState(
                half_life_secs=GRAMMAR_INITIAL_H, unlocked=True
            )
        ss = entry.skill_states[question_type]

        delta_t = (
            (now - ss.last_reviewed_at).total_seconds()
            if ss.last_reviewed_at
            else 120.0
        )

        old_H = ss.half_life_secs
        ss.half_life_secs = update_half_life(ss.half_life_secs, delta_t, grade, skill="recognition")
        ss.last_reviewed_at = now

        # recompute pattern-level due_at across all question-type skills
        entry.due_at = compute_grammar_due_at(entry.skill_states, now)

        # persist SRSState with per-question-type skill_states
        srs_state = self._session.get(srs_db.SRSState, entry.item_id)
        if srs_state:
            srs_state.due_at = entry.due_at
            srs_state.last_reviewed_at = now
            srs_state.state = {
                "skill_states": {
                    qtype: {
                        "half_life_secs": s.half_life_secs,
                        "last_reviewed_at": s.last_reviewed_at.isoformat() if s.last_reviewed_at else None,
                        "unlocked": True,
                    }
                    for qtype, s in entry.skill_states.items()
                }
            }

        # track stats
        self._total_graded += 1
        if grade in (1, 2, 3):
            self._correct_count += 1

        # console audit log
        new_H = ss.half_life_secs
        interval = next_interval(ss.half_life_secs, GRAMMAR_SKILL)
        p_before = recall_probability(old_H, delta_t)
        due_abs = (now + dt.timedelta(seconds=interval)).strftime("%H:%M")
        print(
            f"        [GRADE] {entry.pattern} / {question_type} / grade={grade}"
        )
        print(
            f"                Δt={_fmt(delta_t)}  p(recall)={p_before:.3f}"
        )
        print(
            f"                H: {_fmt(old_H)} -> {_fmt(new_H)}  "
            f"next due: +{_fmt(interval)} ({due_abs})"
        )

        # append ReviewLog
        self._session.add(
            srs_db.ReviewLog(
                item_id=entry.item_id,
                reviewed_at=now,
                grade=grade,
                correct=grade in (1, 2, 3),
                mode="grammar",
                payload={
                    "grammar_pattern": entry.pattern,
                    "question": question,
                    "user_answer": answer,
                    "llm_score": llm_score,
                    "question_type": question_type,
                },
                new_due_at=entry.due_at,
            )
        )

        self._session.commit()

    # ── pattern management ───────────────────────────────────────────

    def add_pattern_to_db(
        self,
        *,
        pattern: str,
        meaning_en: str,
        meaning_ko: str = "",
        example: str = "",
        notes: str = "",
    ) -> int:
        """Insert a grammar item into the DB.  Returns the item ID."""
        normalized = pattern.replace(" ", "_")
        external_id = f"grammar:{normalized}"

        content = {
            "pattern": pattern,
            "meaning_en": meaning_en,
            "meaning_ko": meaning_ko,
            "example": example,
            "notes": notes,
            "question_prompt": DEFAULT_QUESTION_PROMPT,
            "eval_prompt": DEFAULT_EVAL_PROMPT,
            "context": {},
        }

        now = srs_db.now_utc()
        item = srs_db.Item(
            item_type="grammar",
            external_id=external_id,
            front=pattern,
            back=meaning_en,
            content=content,
        )
        self._session.add(item)
        self._session.flush()

        self._session.add(
            srs_db.SRSState(
                item_id=item.id,
                due_at=now,
                scheduler_name="incremental",
                scheduler_version=1,
                state={"skill_states": {}},
            )
        )
        self._session.commit()
        return item.id

    # ── question type management ────────────────────────────────────

    def add_question_type(self, item_id: int, qtype_config: dict) -> None:
        """Append a question type config to an existing pattern's content."""
        item = self._session.get(srs_db.Item, item_id)
        if item is None:
            return
        content = dict(item.content or {})
        qtypes = list(content.get("question_types", []))
        qtypes.append(qtype_config)
        content["question_types"] = qtypes
        item.content = content
        self._session.commit()

        # update in-memory entry
        qt_name = qtype_config.get("type", "sentence_completion")
        for e in self._entries:
            if e.item_id == item_id:
                e.content = content
                if qt_name not in e.skill_states:
                    e.skill_states[qt_name] = SkillState(
                        half_life_secs=GRAMMAR_INITIAL_H, unlocked=True
                    )
                break

    def remove_question_type(self, item_id: int, index: int) -> None:
        """Remove a question type config by index from a pattern's content."""
        item = self._session.get(srs_db.Item, item_id)
        if item is None:
            return
        content = dict(item.content or {})
        qtypes = list(content.get("question_types", []))
        if 0 <= index < len(qtypes):
            qtypes.pop(index)
            content["question_types"] = qtypes
            item.content = content
            self._session.commit()

            for e in self._entries:
                if e.item_id == item_id:
                    e.content = content
                    break

    def set_question_types(self, item_id: int, question_types: list[dict]) -> None:
        """Replace the entire question_types list for a pattern."""
        item = self._session.get(srs_db.Item, item_id)
        if item is None:
            return
        content = dict(item.content or {})
        content["question_types"] = list(question_types)
        item.content = content
        self._session.commit()
        for e in self._entries:
            if e.item_id == item_id:
                e.content = content
                # ensure newly added question types get a SkillState entry
                for qt_cfg in question_types:
                    qt_name = qt_cfg.get("type", "sentence_completion")
                    if qt_name not in e.skill_states:
                        e.skill_states[qt_name] = SkillState(
                            half_life_secs=GRAMMAR_INITIAL_H, unlocked=True
                        )
                break

    def delete_pattern(self, item_id: int) -> bool:
        """Soft-delete a grammar pattern by setting deleted_at."""
        item = self._session.get(srs_db.Item, item_id)
        if item is None or item.item_type != "grammar":
            return False
        item.deleted_at = srs_db.now_utc()
        self._session.commit()
        self._entries = [e for e in self._entries if e.item_id != item_id]
        return True

    # ── cleanup ──────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Shut down the prefetch thread pool."""
        self._prefetch_pool.shutdown(wait=False)
