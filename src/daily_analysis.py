#!/usr/bin/env python3
"""Daily review analysis: per-day stats, H trajectories, difficulty ranking, time prediction.

Usage:
    python daily_analysis.py --db test_srs.sqlite [--days 7] [--date 2026-02-28]
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from collections import defaultdict
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

import srs_db
from incremental_model import (
    INITIAL_HALF_LIFE,
    SKILL_ORDER,
    SOLID_HALF_LIFE,
    TARGET_RECALL,
    next_interval,
    recall_probability,
    skill_states_from_dict,
    update_half_life,
)
from acquisition_model import (
    LexemeState,
    lexeme_state_from_dict,
    GRADUATION_H,
    PHASE_DAY0,
    PHASE_ACQUIRING,
    PHASE_MAINTENANCE,
    PHASE_REPAIRING,
    PHASE_LEARNED,
)

UTC = dt.timezone.utc

GRADE_LABELS = {
    1: "F|C", 2: "N|C", 3: "H|C",
    4: "Easy|W", 5: "Hard|W", 6: "No Idea",
}


# ── helpers ──────────────────────────────────────────────────────────

def _ensure_utc(t: dt.datetime) -> dt.datetime:
    if t.tzinfo is None:
        return t.replace(tzinfo=UTC)
    return t


def _fmt_duration(secs: float) -> str:
    """Human-readable duration."""
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}m"
    if secs < 86400:
        return f"{secs / 3600:.1f}h"
    return f"{secs / 86400:.1f}d"


def _fmt_H(secs: float) -> str:
    """Format half-life compactly."""
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.0f}m"
    if secs < 86400:
        return f"{secs / 3600:.1f}h"
    return f"{secs / 86400:.1f}d"


def _active_skill(skill_data: dict) -> Optional[str]:
    """Return the currently active skill name from skill_states dict, or None."""
    skills = skill_states_from_dict(skill_data)
    for sk in SKILL_ORDER:
        ss = skills.get(sk)
        if ss and ss.unlocked and not ss.graduated:
            return sk
    return None


def _active_H(skill_data: dict) -> Optional[float]:
    """Return the H of the currently active skill."""
    skills = skill_states_from_dict(skill_data)
    for sk in SKILL_ORDER:
        ss = skills.get(sk)
        if ss and ss.unlocked and not ss.graduated:
            return ss.half_life_secs
    return None


# ── data loading ─────────────────────────────────────────────────────

def _load_graded_reviews(session: Session) -> list[dict]:
    """Load all graded non-intro reviews, chronologically."""
    logs = (
        session.execute(
            select(srs_db.ReviewLog)
            .where(srs_db.ReviewLog.grade.isnot(None))
            .where(srs_db.ReviewLog.mode != "intro")
            .order_by(srs_db.ReviewLog.reviewed_at)
        )
        .scalars()
        .all()
    )
    results = []
    for l in logs:
        ext_id = l.item.external_id if l.item else None
        lexeme = srs_db.extract_lexeme_from_external_id(ext_id) if ext_id else None
        if not lexeme:
            continue
        payload = l.payload or {}
        results.append({
            "lexeme": lexeme,
            "skill": payload.get("incremental_skill"),
            "grade": l.grade,
            "correct": l.correct,
            "reviewed_at": _ensure_utc(l.reviewed_at),
            "H_after": payload.get("half_life_secs"),
            "H_before": payload.get("half_life_secs_before"),
            "phase": payload.get("phase"),
        })
    return results


def _load_intro_reviews(session: Session) -> list[dict]:
    """Load all intro reviews."""
    logs = (
        session.execute(
            select(srs_db.ReviewLog)
            .where(srs_db.ReviewLog.mode == "intro")
            .order_by(srs_db.ReviewLog.reviewed_at)
        )
        .scalars()
        .all()
    )
    results = []
    for l in logs:
        ext_id = l.item.external_id if l.item else None
        lexeme = srs_db.extract_lexeme_from_external_id(ext_id) if ext_id else None
        if not lexeme:
            continue
        results.append({
            "lexeme": lexeme,
            "reviewed_at": _ensure_utc(l.reviewed_at),
        })
    return results


def _load_current_states(session: Session) -> dict[str, dict]:
    """Load current SRS state for all active (non-suspended) lexemes.

    Returns {lexeme: {"phase": ..., "active_skill": ..., "active_H": ..., "due_at": ...,
                       "reviewed": bool, "passes_24h": int, "is_hard": bool,
                       "skill_data": ... (legacy compat)}}
    """
    items = (
        session.execute(
            select(srs_db.Item)
            .where(srs_db.Item.suspended == False)
            .where(srs_db.Item.deleted_at.is_(None))
            .where(srs_db.Item.external_id.like("%cloze_prod_bundle%"))
        )
        .scalars()
        .all()
    )
    states = {}
    for item in items:
        lexeme = srs_db.extract_lexeme_from_external_id(item.external_id)
        if not lexeme:
            continue
        if not item.srs_state or not item.srs_state.state:
            continue
        state = item.srs_state.state
        due = _ensure_utc(item.srs_state.due_at) if item.srs_state.due_at else None
        reviewed = item.srs_state.last_reviewed_at is not None

        # Try new format first (has "lexeme_state" key or "phase" key)
        lexeme_state_data = state.get("lexeme_state")
        if lexeme_state_data and "phase" in lexeme_state_data:
            ls = lexeme_state_from_dict(lexeme_state_data)
            phase = ls.phase
            if phase == PHASE_MAINTENANCE:
                active_H = ls.half_life_secs
                active_skill = "production"
            else:
                active_H = None
                active_skill = None
            states[lexeme] = {
                "phase": phase,
                "active_skill": active_skill,
                "active_H": active_H,
                "due_at": due,
                "reviewed": reviewed,
                "passes_24h": ls.passes_24h,
                "is_hard": ls.is_hard,
                "skill_data": None,
            }
            continue

        # Fall back to old multi-skill format
        skill_data = state.get("skill_states")
        if not skill_data:
            continue
        states[lexeme] = {
            "phase": PHASE_MAINTENANCE,  # legacy words are in maintenance
            "skill_data": skill_data,
            "active_skill": _active_skill(skill_data),
            "active_H": _active_H(skill_data),
            "due_at": due,
            "reviewed": reviewed,
            "passes_24h": 0,
            "is_hard": False,
        }
    return states


# ── analysis functions ───────────────────────────────────────────────

def _day_key(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%d")


def _compute_mean_review_time(reviews: list[dict]) -> float:
    """Estimate mean seconds per review from inter-review gaps within sessions."""
    gaps = []
    for i in range(1, len(reviews)):
        gap = (reviews[i]["reviewed_at"] - reviews[i - 1]["reviewed_at"]).total_seconds()
        if gap < 600:  # filter out session breaks only
            gaps.append(gap)
    return sum(gaps) / len(gaps) if gaps else 30.0


def _compute_difficulty(reviews: list[dict]) -> dict[str, dict]:
    """Compute per-lexeme difficulty from historical success rate.

    Returns {lexeme: {"total": N, "correct": N, "rate": float, "grades": list}}
    """
    by_lex = defaultdict(lambda: {"total": 0, "correct": 0, "grades": []})
    for r in reviews:
        entry = by_lex[r["lexeme"]]
        entry["total"] += 1
        if r["correct"]:
            entry["correct"] += 1
        entry["grades"].append(r["grade"])
    for lex, d in by_lex.items():
        d["rate"] = d["correct"] / d["total"] if d["total"] > 0 else 0.0
    return dict(by_lex)


def _analyze_day(
    day: str,
    day_reviews: list[dict],
    all_reviews: list[dict],
    intro_reviews: list[dict],
) -> dict:
    """Analyze a single day's reviews."""
    n_total = len(day_reviews)
    n_correct = sum(1 for r in day_reviews if r["correct"])
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    # unique lexemes studied
    lexemes_studied = set(r["lexeme"] for r in day_reviews)

    # grade distribution
    grade_dist = defaultdict(int)
    for r in day_reviews:
        grade_dist[r["grade"]] += 1

    # session time (first to last review)
    times = [r["reviewed_at"] for r in day_reviews]
    session_secs = (max(times) - min(times)).total_seconds() if len(times) > 1 else 0

    # mean time per review (from this day's gaps)
    mean_time = _compute_mean_review_time(day_reviews)

    # intros on this day
    day_intros = [i for i in intro_reviews if _day_key(i["reviewed_at"]) == day]
    intro_lexemes = set(i["lexeme"] for i in day_intros)

    # new words: first-ever graded review on this day
    all_prior = [r for r in all_reviews if _day_key(r["reviewed_at"]) < day]
    prior_lexemes = set(r["lexeme"] for r in all_prior)
    new_lexemes = lexemes_studied - prior_lexemes

    # H trajectory: track per (lexeme, skill) changes within this day
    # Build H_before from previous reviews, H_after from this day's last review
    by_key = defaultdict(list)
    for r in all_reviews:
        if r["H_after"] is None or r["skill"] is None:
            continue
        by_key[(r["lexeme"], r["skill"])].append(r)

    increased = []  # (lexeme, skill, H_before, H_after)
    decreased = []
    unlocks = []    # (lexeme, new_skill)

    for (lex, skill), history in by_key.items():
        day_entries = [r for r in history if _day_key(r["reviewed_at"]) == day]
        if not day_entries:
            continue
        pre_entries = [r for r in history if _day_key(r["reviewed_at"]) < day]

        # Determine H_before: prefer the explicit H_before field from the first
        # review today (accounts for migration from old model), then fall back to
        # the last pre-day H_after, then INITIAL_HALF_LIFE for brand new words.
        #
        # Migration edge case: if the previous review was under the old multi-skill
        # model (no "phase" field) but today's review is under the new model
        # (has "phase"), the old H is stale — use GRADUATION_H as the baseline.
        first_today = day_entries[0]
        if first_today.get("H_before") is not None:
            H_before = first_today["H_before"]
        elif pre_entries:
            last_pre = pre_entries[-1]
            is_migration_boundary = (
                last_pre.get("phase") is None
                and first_today.get("phase") == "maintenance"
            )
            if is_migration_boundary:
                H_before = GRADUATION_H
            else:
                H_before = last_pre["H_after"]
        else:
            H_before = INITIAL_HALF_LIFE

        H_after = day_entries[-1]["H_after"]
        if H_after > H_before * 1.01:
            increased.append((lex, skill, H_before, H_after))
        elif H_after < H_before * 0.99:
            decreased.append((lex, skill, H_before, H_after))

    # detect skill unlocks: first review of a non-recognition skill on this day
    for (lex, skill), history in by_key.items():
        if skill == "recognition":
            continue
        day_entries = [r for r in history if _day_key(r["reviewed_at"]) == day]
        pre_entries = [r for r in history if _day_key(r["reviewed_at"]) < day]
        if day_entries and not pre_entries:
            unlocks.append((lex, skill))

    return {
        "day": day,
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "lexemes_studied": lexemes_studied,
        "grade_dist": dict(grade_dist),
        "session_secs": session_secs,
        "mean_time": mean_time,
        "intro_lexemes": intro_lexemes,
        "new_lexemes": new_lexemes,
        "increased": increased,
        "decreased": decreased,
        "unlocks": unlocks,
    }


def _count_due_by(states: dict[str, dict], deadline: dt.datetime, reviewed_only: bool = True) -> int:
    """Count lexemes due by the given deadline."""
    count = 0
    for lex, st in states.items():
        if reviewed_only and not st.get("reviewed"):
            continue
        if st["due_at"] is not None and st["due_at"] <= deadline:
            count += 1
    return count


def _estimate_reviews_due(
    states: dict[str, dict],
    deadline: dt.datetime,
    mean_time: float,
) -> tuple[int, float]:
    """Estimate number of reviews and time needed to clear all due by deadline.

    Each due lexeme gets ~1.2 reviews on average (some will come due again within
    the session if their interval is short).
    """
    due = _count_due_by(states, deadline)
    # rough multiplier: short-interval words get re-reviewed in session
    estimated_reviews = int(due * 1.2)
    estimated_secs = estimated_reviews * mean_time
    return estimated_reviews, estimated_secs


# ── display ──────────────────────────────────────────────────────────

def _print_day_report(analysis: dict, difficulty: dict[str, dict]) -> None:
    day = analysis["day"]
    n = analysis["n_total"]
    acc = analysis["accuracy"]
    n_lex = len(analysis["lexemes_studied"])

    print(f"\n{'=' * 60}")
    print(f"  {day}  —  {n} reviews, {n_lex} words, {acc:.0%} accuracy")
    print(f"{'=' * 60}")

    # session time + review pace
    if analysis["session_secs"] > 0:
        print(f"  Session: {_fmt_duration(analysis['session_secs'])}  "
              f"({analysis['mean_time']:.0f}s/review avg)")

    # grade distribution
    gd = analysis["grade_dist"]
    parts = []
    for g in sorted(gd):
        parts.append(f"{GRADE_LABELS.get(g, f'g{g}')}: {gd[g]}")
    print(f"  Grades: {', '.join(parts)}")

    # new words
    if analysis["new_lexemes"]:
        print(f"  New words ({len(analysis['new_lexemes'])}): "
              f"{', '.join(sorted(analysis['new_lexemes']))}")
    if analysis["intro_lexemes"] - analysis["new_lexemes"]:
        extra = analysis["intro_lexemes"] - analysis["new_lexemes"]
        print(f"  Re-introduced ({len(extra)}): {', '.join(sorted(extra))}")

    # ── memory trace changes ──
    inc = analysis["increased"]
    dec = analysis["decreased"]

    # aggregate by lexeme: net strongest/weakest skill change
    inc_by_lex: dict[str, list] = defaultdict(list)
    dec_by_lex: dict[str, list] = defaultdict(list)
    for lex, skill, h0, h1 in inc:
        inc_by_lex[lex].append((skill, h0, h1))
    for lex, skill, h0, h1 in dec:
        dec_by_lex[lex].append((skill, h0, h1))

    # words that ONLY got stronger (no regressions on any skill)
    pure_stronger = set(inc_by_lex.keys()) - set(dec_by_lex.keys())
    # words that had at least one regression
    had_regression = set(dec_by_lex.keys())
    # words that ONLY got weaker (no gains)
    pure_weaker = had_regression - set(inc_by_lex.keys())
    # mixed: both gains and losses across skills
    mixed = had_regression & set(inc_by_lex.keys())

    print(f"\n  Memory trace changes:")
    print(f"    Strengthened (only gains):  {len(pure_stronger)} words")
    print(f"    Weakened (only losses):     {len(pure_weaker)} words")
    print(f"    Mixed (gains + losses):     {len(mixed)} words")

    # show weakened words (sorted by worst regression ratio)
    if had_regression:
        print(f"\n  Weakened words:")
        reg_summary = []
        for lex in sorted(had_regression):
            for skill, h0, h1 in dec_by_lex[lex]:
                ratio = h1 / h0
                reg_summary.append((lex, skill, h0, h1, ratio))
        reg_summary.sort(key=lambda x: x[4])
        for lex, skill, h0, h1, ratio in reg_summary[:15]:
            sk_short = skill[:4]
            marker = " (mixed)" if lex in mixed else ""
            print(f"    {lex:14s} [{sk_short}]  {_fmt_H(h0):>6s} -> {_fmt_H(h1):>6s}  "
                  f"({ratio:.2f}x){marker}")

    # show biggest gains (exclude freshly unlocked skills — those are shown under unlocks)
    unlock_keys = set((lex, skill) for lex, skill in analysis["unlocks"])
    non_unlock_inc = [(l, s, h0, h1) for l, s, h0, h1 in inc if (l, s) not in unlock_keys]
    if non_unlock_inc:
        print(f"\n  Biggest gains (existing skills):")
        gain_summary = []
        for lex, skill, h0, h1 in non_unlock_inc:
            ratio = h1 / h0
            gain_summary.append((lex, skill, h0, h1, ratio))
        gain_summary.sort(key=lambda x: x[4], reverse=True)
        for lex, skill, h0, h1, ratio in gain_summary[:10]:
            sk_short = skill[:4]
            print(f"    {lex:14s} [{sk_short}]  {_fmt_H(h0):>6s} -> {_fmt_H(h1):>6s}  "
                  f"({ratio:.1f}x)")

    # skill unlocks
    if analysis["unlocks"]:
        print(f"\n  Skill unlocks ({len(analysis['unlocks'])}):")
        for lex, skill in sorted(analysis["unlocks"]):
            print(f"    {lex} -> {skill}")

    # hardest words studied today (by historical success rate)
    studied = analysis["lexemes_studied"]
    studied_diff = [(lex, difficulty[lex]) for lex in studied if lex in difficulty]
    studied_diff.sort(key=lambda x: x[1]["rate"])
    if studied_diff:
        print(f"  Hardest words studied (by historical success rate):")
        for lex, d in studied_diff[:10]:
            print(f"    {lex:12s}  {d['rate']:.0%} ({d['correct']}/{d['total']})")


def _print_current_state(states: dict[str, dict], now: dt.datetime) -> None:
    reviewed_states = {k: v for k, v in states.items() if v.get("reviewed")}
    never_reviewed = {k: v for k, v in states.items() if not v.get("reviewed")}

    print(f"\n{'=' * 60}")
    print(f"  Current state  ({len(reviewed_states)} reviewed, "
          f"{len(never_reviewed)} never-reviewed, {len(states)} total)")
    print(f"{'=' * 60}")

    # Phase breakdown
    by_phase = defaultdict(list)
    for lex, st in states.items():
        phase = st.get("phase", PHASE_MAINTENANCE)
        by_phase[phase].append((lex, st))

    # Day 0
    day0_words = by_phase.get(PHASE_DAY0, [])
    if day0_words:
        print(f"\n  DAY 0 ({len(day0_words)} words)")
        for lex, st in sorted(day0_words, key=lambda x: x[0]):
            print(f"    {lex}")

    # Acquiring
    acq_words = by_phase.get(PHASE_ACQUIRING, [])
    if acq_words:
        print(f"\n  ACQUIRING ({len(acq_words)} words)")
        for lex, st in sorted(acq_words, key=lambda x: x[1].get("passes_24h", 0), reverse=True):
            passes = st.get("passes_24h", 0)
            hard = " [hard]" if st.get("is_hard") else ""
            print(f"    {lex:14s}  passes={passes}{hard}")

    # Repairing
    rep_words = by_phase.get(PHASE_REPAIRING, [])
    if rep_words:
        print(f"\n  REPAIRING ({len(rep_words)} words)")
        for lex, st in sorted(rep_words, key=lambda x: x[0]):
            print(f"    {lex}")

    # Maintenance
    maint_words = by_phase.get(PHASE_MAINTENANCE, [])
    if maint_words:
        print(f"\n  MAINTENANCE ({len(maint_words)} words)")

        # Sub-categorize by maturity
        mature = [(l, s) for l, s in maint_words if s["active_H"] and next_interval(s["active_H"], "production") > 2 * 86400]
        in_learning = [(l, s) for l, s in maint_words if s["active_H"] and s["active_H"] < SOLID_HALF_LIFE]
        solid = [(l, s) for l, s in maint_words if s["active_H"] and s["active_H"] >= SOLID_HALF_LIFE]

        print(f"    Mature (interval > 2d): {len(mature)}")
        print(f"    Solid (H >= 4.3d):      {len(solid)}")
        print(f"    In-learning:            {len(in_learning)}")

    # Learned
    learned_words = by_phase.get(PHASE_LEARNED, [])
    if learned_words:
        print(f"\n  LEARNED ({len(learned_words)} words)")
        for lex, st in sorted(learned_words, key=lambda x: x[0]):
            print(f"    {lex}")

    # Legacy: words with old skill_data that weren't migrated
    legacy_by_skill = defaultdict(list)
    for lex, st in maint_words:
        if st.get("skill_data"):
            sk = st["active_skill"]
            if sk and sk != "production":
                legacy_by_skill[sk].append((lex, st))

    if legacy_by_skill:
        print(f"\n  Legacy skill breakdown (within maintenance):")
        for sk in SKILL_ORDER:
            entries = legacy_by_skill.get(sk, [])
            if entries:
                print(f"    {sk.capitalize()}: {len(entries)}")

    print(f"\n  Phase summary:")
    print(f"    Day 0:          {len(day0_words)}")
    print(f"    Acquiring:      {len(acq_words)}")
    print(f"    Repairing:      {len(rep_words)}")
    print(f"    Maintenance:    {len(maint_words)}")
    print(f"    Learned:        {len(learned_words)}")
    print(f"    Never reviewed: {len(never_reviewed)}")


def _print_tomorrow_forecast(
    states: dict[str, dict],
    mean_time: float,
    now: dt.datetime,
) -> None:
    tomorrow_start = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_end = tomorrow_start + dt.timedelta(hours=23, minutes=59, seconds=59)

    due_now = _count_due_by(states, now)
    due_tomorrow_start = _count_due_by(states, tomorrow_start)
    due_tomorrow_end = _count_due_by(states, tomorrow_end)
    new_tomorrow = due_tomorrow_end - due_tomorrow_start

    est_reviews, est_secs = _estimate_reviews_due(states, tomorrow_end, mean_time)

    print(f"\n{'=' * 60}")
    print(f"  Tomorrow's forecast")
    print(f"{'=' * 60}")
    print(f"  Due right now:                {due_now}")
    print(f"  Due by start of tomorrow:     {due_tomorrow_start}")
    print(f"  New reviews coming due:       {new_tomorrow}")
    print(f"  Total due by end of tomorrow: {due_tomorrow_end}")
    print(f"  Mean time per review:         {mean_time:.0f}s")
    print(f"  Estimated reviews:            ~{est_reviews}")
    print(f"  Estimated session time:       ~{_fmt_duration(est_secs)}")


def _print_difficulty_ranking(difficulty: dict[str, dict], top_n: int = 20) -> None:
    # filter to words with >= 3 reviews for meaningful stats
    eligible = {k: v for k, v in difficulty.items() if v["total"] >= 3}
    if not eligible:
        return

    ranked = sorted(eligible.items(), key=lambda x: x[1]["rate"])

    print(f"\n{'=' * 60}")
    print(f"  Hardest words (>= 3 reviews, by success rate)")
    print(f"{'=' * 60}")
    print(f"  {'Word':14s} {'Rate':>6s} {'Correct':>8s} {'Total':>6s}  {'Avg Grade':>10s}")
    print(f"  {'-' * 50}")
    for lex, d in ranked[:top_n]:
        avg_grade = sum(d["grades"]) / len(d["grades"])
        print(f"  {lex:14s} {d['rate']:>5.0%}  {d['correct']:>7d}  {d['total']:>5d}  {avg_grade:>10.1f}")


# ── forward simulation ────────────────────────────────────────────────

def _compute_grade_distributions(reviews: list[dict]) -> tuple[dict[int, float], dict[int, float]]:
    """Compute historical grade distributions for correct and wrong reviews.

    Returns (correct_dist, wrong_dist) where each is {grade: probability}.
    """
    correct_grades = defaultdict(int)
    wrong_grades = defaultdict(int)
    for r in reviews:
        if r["correct"]:
            correct_grades[r["grade"]] += 1
        else:
            wrong_grades[r["grade"]] += 1

    total_correct = sum(correct_grades.values()) or 1
    total_wrong = sum(wrong_grades.values()) or 1

    correct_dist = {g: n / total_correct for g, n in correct_grades.items()}
    wrong_dist = {g: n / total_wrong for g, n in wrong_grades.items()}

    # fallback: if no data, use grade 2 for correct, grade 5 for wrong
    if not correct_dist:
        correct_dist = {2: 1.0}
    if not wrong_dist:
        wrong_dist = {5: 1.0}

    return correct_dist, wrong_dist


def _expected_grade(dist: dict[int, float]) -> float:
    """Weighted average grade from a distribution."""
    return sum(g * p for g, p in dist.items())


def _simulate_review(
    H: float, skill: str, delta_t: float,
    correct_dist: dict[int, float], wrong_dist: dict[int, float],
) -> tuple[float, float, float]:
    """Simulate one review and return (expected_new_H, new_interval, p_recall).

    Uses the model's p(recall) as the probability of getting it correct,
    then computes expected H as: p * H_if_correct + (1-p) * H_if_wrong.
    """
    p = recall_probability(H, delta_t)

    # expected H if correct: weighted average over correct grade distribution
    H_correct = 0.0
    for g, prob in correct_dist.items():
        H_correct += prob * update_half_life(H, delta_t, g, skill)

    # expected H if wrong: weighted average over wrong grade distribution
    H_wrong = 0.0
    for g, prob in wrong_dist.items():
        H_wrong += prob * update_half_life(H, delta_t, g, skill)

    expected_H = p * H_correct + (1.0 - p) * H_wrong
    new_interval = next_interval(expected_H, skill)
    return expected_H, new_interval, p


def _simulate_forward(
    states: dict[str, dict],
    all_reviews: list[dict],
    now: dt.datetime,
    n_days: int = 14,
) -> list[dict]:
    """Simulate reviews forward n_days assuming no new words.

    Returns list of per-day dicts:
        {"day": int, "date": str, "reviews": int, "expected_correct": float,
         "expected_accuracy": float, "strengthened": int, "regressed": int}
    """
    correct_dist, wrong_dist = _compute_grade_distributions(all_reviews)

    # build simulation state: only maintenance words (half-life model applies)
    sim_words = []
    for lex, st in states.items():
        if not st.get("reviewed"):
            continue
        if st.get("phase", PHASE_MAINTENANCE) != PHASE_MAINTENANCE:
            continue  # skip day0, acquiring, repairing — not driven by H
        skill = st["active_skill"]
        H = st["active_H"]
        due = st["due_at"]
        if skill is None or H is None or due is None:
            continue
        sim_words.append({"lexeme": lex, "skill": skill, "H": H, "due_at": due})

    results = []
    for day_offset in range(1, n_days + 1):
        day_start = (now + dt.timedelta(days=day_offset)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        day_end = day_start + dt.timedelta(hours=23, minutes=59, seconds=59)

        # find words due during this day
        due_today = [w for w in sim_words if w["due_at"] <= day_end]

        n_reviews = len(due_today)
        total_p = 0.0
        # fractional counts: each review contributes p to strengthened,
        # (1-p) to regressed, since correct always grows H and wrong shrinks
        expected_strengthened = 0.0
        expected_regressed = 0.0

        for w in due_today:
            # delta_t = scheduled interval (the time between reviews when
            # the user reviews promptly at the due time)
            delta_t = next_interval(w["H"], w["skill"])

            old_H = w["H"]
            new_H, new_interval, p = _simulate_review(
                old_H, w["skill"], delta_t, correct_dist, wrong_dist,
            )
            total_p += p
            expected_strengthened += p
            expected_regressed += (1.0 - p)

            # update simulation state
            review_time = min(day_end, max(day_start, w["due_at"]))
            w["H"] = new_H
            w["due_at"] = review_time + dt.timedelta(seconds=new_interval)

        expected_accuracy = total_p / n_reviews if n_reviews > 0 else 0.0

        results.append({
            "day": day_offset,
            "date": day_start.strftime("%Y-%m-%d"),
            "reviews": n_reviews,
            "expected_correct": round(total_p, 1),
            "expected_accuracy": round(expected_accuracy, 3),
            "strengthened": round(expected_strengthened),
            "regressed": round(expected_regressed),
        })

    return results


def _print_forward_simulation(
    states: dict[str, dict],
    all_reviews: list[dict],
    now: dt.datetime,
    n_days: int = 14,
) -> None:
    results = _simulate_forward(states, all_reviews, now, n_days)
    if not results:
        return

    print(f"\n{'=' * 60}")
    print(f"  Forward simulation ({n_days} days, no new words)")
    print(f"{'=' * 60}")
    print(f"  {'Day':>4s}  {'Date':10s}  {'Reviews':>7s}  {'Acc':>5s}  "
          f"{'Str':>4s}  {'Reg':>4s}")
    print(f"  {'-' * 46}")
    for r in results:
        print(f"  {r['day']:>4d}  {r['date']}  {r['reviews']:>7d}  "
              f"{r['expected_accuracy']:>4.0%}  "
              f"{r['strengthened']:>4d}  {r['regressed']:>4d}")


def _plot_forward_simulation(
    states: dict[str, dict],
    all_reviews: list[dict],
    now: dt.datetime,
    mean_time: float = 30.0,
    n_days: int = 21,
    save: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    results = _simulate_forward(states, all_reviews, now, n_days)
    if not results:
        return

    days = [r["day"] for r in results]
    dates = [r["date"][5:] for r in results]  # MM-DD
    reviews = [r["reviews"] for r in results]
    strengthened = [r["strengthened"] for r in results]
    regressed = [r["regressed"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={"hspace": 0.4})

    # top: review count line plot with time annotations
    ax1.plot(days, reviews, "o-", color="#2a69e8", linewidth=2, markersize=5)
    ax1.fill_between(days, reviews, alpha=0.15, color="#2a69e8")
    ax1.set_ylabel("Reviews")
    ax1.set_title(f"Forward simulation ({n_days} days, no new words)")
    ax1.set_ylim(0, max(reviews) * 1.3)
    ax1.grid(axis="y", alpha=0.3)

    ax1.set_xticks(days)
    ax1.set_xticklabels(dates, rotation=45, ha="center", fontsize=8)

    # annotate each point with estimated time
    for i, n in enumerate(reviews):
        est_secs = n * mean_time
        label = _fmt_duration(est_secs)
        ax1.annotate(label, (days[i], n), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color="#555")

    # bottom: strengthened and regressed line plots
    ax2.plot(days, strengthened, "o-", color="#4caf50", linewidth=2,
             markersize=5, label="Strengthened")
    ax2.plot(days, regressed, "o-", color="#e74c3c", linewidth=2,
             markersize=5, label="Regressed")
    ax2.fill_between(days, strengthened, alpha=0.15, color="#4caf50")
    ax2.fill_between(days, regressed, alpha=0.15, color="#e74c3c")
    ax2.set_ylabel("Words")
    ax2.set_xlabel("Date")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    ax2.set_xticks(days)
    ax2.set_xticklabels(dates, rotation=45, ha="center", fontsize=8)

    plt.tight_layout()
    if save:
        for ext in ("png", "pdf"):
            path = f"img/forward_simulation.{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved {path}")
    else:
        plt.show()


# ── public API ────────────────────────────────────────────────────────

def compute_daily_summary(engine, date: Optional[str] = None) -> dict:
    """Compute a summary dict for the web dashboard.

    Args:
        engine: SQLAlchemy engine
        date: optional YYYY-MM-DD string; defaults to today (UTC)

    Returns dict with:
        reviews, words_studied, accuracy,
        strengthened, weakened, weakened_words (top 5),
        mean_time_per_review,
        tomorrow_due, tomorrow_estimated_time
    """
    import srs_db as _srs_db
    now = _srs_db.now_utc()
    target_date = date or _day_key(now)

    with Session(engine) as session:
        all_reviews = _load_graded_reviews(session)
        intro_reviews = _load_intro_reviews(session)
        current_states = _load_current_states(session)

    # group by day
    by_day = defaultdict(list)
    for r in all_reviews:
        by_day[_day_key(r["reviewed_at"])].append(r)

    day_reviews = by_day.get(target_date, [])

    # today's stats
    if day_reviews:
        analysis = _analyze_day(target_date, day_reviews, all_reviews, intro_reviews)
        reviews = analysis["n_total"]
        words_studied = len(analysis["lexemes_studied"])
        accuracy = analysis["accuracy"]
        mean_time = analysis["mean_time"]

        # strengthened / weakened counts
        inc_lexemes = set(lex for lex, _sk, _h0, _h1 in analysis["increased"])
        dec_lexemes = set(lex for lex, _sk, _h0, _h1 in analysis["decreased"])
        pure_stronger = inc_lexemes - dec_lexemes
        had_regression = dec_lexemes

        strengthened = len(pure_stronger)
        weakened = len(had_regression)

        # top 5 weakened words (sorted by worst regression ratio)
        dec_by_lex = defaultdict(list)
        for lex, skill, h0, h1 in analysis["decreased"]:
            dec_by_lex[lex].append((skill, h0, h1))
        reg_summary = []
        for lex in had_regression:
            worst_ratio = min(h1 / h0 for _sk, h0, h1 in dec_by_lex[lex])
            reg_summary.append((lex, worst_ratio))
        reg_summary.sort(key=lambda x: x[1])
        weakened_words = [lex for lex, _r in reg_summary[:5]]
    else:
        reviews = 0
        words_studied = 0
        accuracy = 0.0
        mean_time = 0.0
        strengthened = 0
        weakened = 0
        weakened_words = []

    # global mean time (fallback if today has too few reviews)
    global_mean_time = _compute_mean_review_time(all_reviews) if all_reviews else 30.0
    effective_mean_time = mean_time if reviews >= 5 else global_mean_time

    # tomorrow forecast
    tomorrow_end = (now + dt.timedelta(days=1)).replace(
        hour=23, minute=59, second=59, microsecond=0,
    )
    tomorrow_due = _count_due_by(current_states, tomorrow_end)
    _est_reviews, est_secs = _estimate_reviews_due(
        current_states, tomorrow_end, effective_mean_time,
    )

    # Phase breakdown
    phase_counts = defaultdict(int)
    for _lex, st in current_states.items():
        phase_counts[st.get("phase", PHASE_MAINTENANCE)] += 1

    # Acquired today: words first INTRODUCED (Day-0 Touch-A) today that are now
    # in maintenance. The legacy ACQUIRING gate phase is retired — new words go
    # from Day-0 intros straight to maintenance and never log a PHASE_ACQUIRING
    # review — so key off the intro reviews (each new word's first intro day),
    # not a gate that no longer fires.
    first_intro_day: dict[str, str] = {}
    for r in intro_reviews:
        lex = r["lexeme"]
        day = _day_key(r["reviewed_at"])
        if lex not in first_intro_day or day < first_intro_day[lex]:
            first_intro_day[lex] = day
    acquired_today = sum(
        1
        for lex, day in first_intro_day.items()
        if day == target_date
        and current_states.get(lex, {}).get("phase", PHASE_MAINTENANCE) == PHASE_MAINTENANCE
    )

    return {
        "reviews": reviews,
        "words_studied": words_studied,
        "accuracy": round(accuracy, 3),
        "strengthened": strengthened,
        "weakened": weakened,
        "weakened_words": weakened_words,
        "acquired_today": acquired_today,
        "mean_time_per_review": round(effective_mean_time, 1),
        "tomorrow_due": tomorrow_due,
        "tomorrow_estimated_time_secs": round(est_secs, 0),
        "tomorrow_estimated_time": _fmt_duration(est_secs),
        "phase_counts": dict(phase_counts),
    }


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Daily review analysis")
    parser.add_argument("--db", default="test_srs.sqlite", help="SQLite DB path")
    parser.add_argument("--days", type=int, default=1, help="Number of recent days to show")
    parser.add_argument("--date", type=str, default=None, help="Show specific date (YYYY-MM-DD)")
    parser.add_argument("--all-days", action="store_true", help="Show all days")
    parser.add_argument("--top-hard", type=int, default=20, help="Number of hardest words to show")
    parser.add_argument("--sim-days", type=int, default=14, help="Days to simulate forward")
    parser.add_argument("--plot", action="store_true", help="Plot forward simulation")
    parser.add_argument("--save", action="store_true", help="Save plot to img/")
    args = parser.parse_args()

    engine = srs_db.make_engine(args.db)
    now = srs_db.now_utc()

    with Session(engine) as session:
        print("Loading review data...")
        all_reviews = _load_graded_reviews(session)
        intro_reviews = _load_intro_reviews(session)
        current_states = _load_current_states(session)

    if not all_reviews:
        print("No graded reviews found.")
        return

    # group reviews by day
    by_day = defaultdict(list)
    for r in all_reviews:
        by_day[_day_key(r["reviewed_at"])].append(r)

    all_days = sorted(by_day.keys())

    # select days to display
    if args.date:
        display_days = [args.date] if args.date in by_day else []
        if not display_days:
            print(f"No reviews found for {args.date}")
            print(f"Available days: {', '.join(all_days)}")
            return
    elif args.all_days:
        display_days = all_days
    else:
        display_days = all_days[-args.days:]

    # compute difficulty across all history
    difficulty = _compute_difficulty(all_reviews)

    # compute global mean review time
    global_mean_time = _compute_mean_review_time(all_reviews)

    # header
    total_reviews = len(all_reviews)
    total_days = len(all_days)
    total_lexemes = len(set(r["lexeme"] for r in all_reviews))
    overall_accuracy = sum(1 for r in all_reviews if r["correct"]) / total_reviews
    print(f"\n  All-time: {total_reviews} reviews across {total_days} days, "
          f"{total_lexemes} words, {overall_accuracy:.0%} accuracy")
    print(f"  Global mean review time: {global_mean_time:.0f}s/review")

    # per-day reports
    for day in display_days:
        day_reviews = by_day[day]
        analysis = _analyze_day(day, day_reviews, all_reviews, intro_reviews)
        _print_day_report(analysis, difficulty)

    # current state
    _print_current_state(current_states, now)

    # tomorrow forecast
    _print_tomorrow_forecast(current_states, global_mean_time, now)

    # forward simulation
    _print_forward_simulation(current_states, all_reviews, now, n_days=args.sim_days)

    if args.plot or args.save:
        _plot_forward_simulation(
            current_states, all_reviews, now,
            mean_time=global_mean_time,
            n_days=args.sim_days, save=args.save,
        )

    # difficulty ranking
    _print_difficulty_ranking(difficulty, top_n=args.top_hard)


if __name__ == "__main__":
    main()
