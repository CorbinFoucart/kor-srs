#!/usr/bin/env python3
"""Print all review log entries from the SRS database, with optional scatter plot."""

import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_KOREAN_FONT = fm.FontProperties(
    fname=str(Path(__file__).resolve().parent / "assets" / "Noto_Sans_KR" / "static" / "NotoSansKR-Regular.ttf")
)
from sqlalchemy import select
from sqlalchemy.orm import Session

import srs_db

GRADE_LABELS = {1: "Easy|Correct", 2: "Hard|Correct", 3: "Easy|Wrong", 4: "Hard|Wrong", 5: "No Idea"}


def print_reviews(db: str = "test_srs.sqlite") -> None:
    engine = srs_db.make_engine(db)

    with Session(engine) as s:
        logs = s.execute(
            select(srs_db.ReviewLog).order_by(srs_db.ReviewLog.reviewed_at)
        ).scalars().all()

        print(f"Total reviews: {len(logs)}\n")

        for log in logs:
            item = log.item
            lexeme = srs_db.extract_lexeme_from_external_id(item.external_id) or "?"
            payload = log.payload or {}
            skill = payload.get("skill", "?")
            vi = payload.get("variant_index", "?")
            front = payload.get("front", "")

            grade_str = f"grade={log.grade}" if log.grade is not None else "no grade"
            correct_str = f"correct={log.correct}" if log.correct is not None else ""

            print(f"  {log.reviewed_at}  [{log.mode}]  {lexeme} ({skill} v{vi})")
            print(f"    {grade_str}  {correct_str}  due_at={log.new_due_at}")
            print(f"    front: {front[:80]}")
            print()


def plot_random_word(db: str = "test_srs.sqlite") -> None:
    engine = srs_db.make_engine(db)

    with Session(engine) as s:
        logs = s.execute(
            select(srs_db.ReviewLog).order_by(srs_db.ReviewLog.reviewed_at)
        ).scalars().all()

        # group graded reviews by lexeme: (grade, reviewed_at)
        by_lexeme: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for log in logs:
            if log.grade is None or log.mode == "intro":
                continue
            lexeme = srs_db.extract_lexeme_from_external_id(log.item.external_id)
            if lexeme:
                by_lexeme[lexeme].append((log.grade, log.reviewed_at.timestamp()))

    if not by_lexeme:
        print("No graded reviews found.")
        return

    word = random.choice(list(by_lexeme.keys()))
    entries = by_lexeme[word]
    grades = [g for g, _ in entries]
    timestamps = [t for _, t in entries]
    review_nums = list(range(1, len(grades) + 1))

    # compute log10(seconds since previous review) for reviews 2+
    log_intervals = []
    for i in range(1, len(timestamps)):
        dt = max(timestamps[i] - timestamps[i - 1], 1e-3)  # clamp to avoid log(0)
        log_intervals.append(math.log10(dt))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    colors = ["green" if g <= 2 else "red" for g in grades]

    # top: grade scatter
    ax1.scatter(review_nums, grades, c=colors, s=60, zorder=3)
    ax1.set_ylabel("Grade")
    ax1.set_title(f"Review history: {word}", fontproperties=_KOREAN_FONT, fontsize=14)
    ax1.set_yticks(list(GRADE_LABELS.keys()))
    ax1.set_yticklabels(list(GRADE_LABELS.values()), fontproperties=_KOREAN_FONT)
    ax1.set_ylim(0.5, 5.5)
    ax1.invert_yaxis()
    ax1.grid(axis="y", alpha=0.3)

    # bottom: log10 inter-review interval
    ax2.scatter(review_nums[1:], log_intervals, c=colors[1:], s=60, zorder=3)
    ax2.set_xlabel("Review #")
    ax2.set_ylabel("log₁₀(seconds since prev)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_interval_audit(n: int = 5, db: str = "test_srs.sqlite") -> None:
    engine = srs_db.make_engine(db)

    with Session(engine) as s:
        logs = s.execute(
            select(srs_db.ReviewLog).order_by(srs_db.ReviewLog.reviewed_at)
        ).scalars().all()

        # group graded reviews by lexeme: (reviewed_at, new_due_at, grade, correct)
        by_lexeme: dict[str, list[tuple]] = defaultdict(list)
        for log in logs:
            if log.grade is None or log.mode == "intro":
                continue
            if log.new_due_at is None:
                continue
            lexeme = srs_db.extract_lexeme_from_external_id(log.item.external_id)
            if lexeme:
                by_lexeme[lexeme].append((
                    log.reviewed_at,
                    log.new_due_at,
                    log.grade,
                    log.correct,
                ))

    # filter to lexemes with >= 2 graded reviews
    eligible = {k: v for k, v in by_lexeme.items() if len(v) >= 2}
    if not eligible:
        print("No lexemes with >= 2 graded reviews found.")
        return

    pick = random.sample(list(eligible.keys()), min(n, len(eligible)))

    fig, axes = plt.subplots(len(pick), 1, figsize=(10, 3 * len(pick)), squeeze=False)

    # reference lines: label, seconds
    ref_lines = [
        ("1 min", 60),
        ("10 min", 600),
        ("1 hr", 3600),
        ("1 day", 86400),
    ]

    for idx, word in enumerate(pick):
        ax = axes[idx, 0]
        entries = eligible[word]

        times = [e[0] for e in entries]
        intervals = [(e[1] - e[0]).total_seconds() for e in entries]
        colors = ["green" if e[2] <= 2 else "red" for e in entries]

        # grey connecting line
        ax.plot(times, intervals, color="grey", linewidth=0.8, alpha=0.5, zorder=1)
        # colored scatter points
        ax.scatter(times, intervals, c=colors, s=50, zorder=3, edgecolors="none")

        ax.set_yscale("log")
        ax.set_ylabel("Interval (s)")
        ax.set_title(word, fontproperties=_KOREAN_FONT, fontsize=14)

        # human-readable reference lines
        for label, secs in ref_lines:
            ax.axhline(y=secs, color="blue", linewidth=0.5, alpha=0.3, linestyle="--")
            ax.text(times[0], secs * 1.15, label, fontsize=8, color="blue", alpha=0.5)

        ax.grid(axis="y", alpha=0.15)
        fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="test_srs.sqlite")
    args = ap.parse_args()
    print_reviews(db=args.db)
    plot_interval_audit(db=args.db)
