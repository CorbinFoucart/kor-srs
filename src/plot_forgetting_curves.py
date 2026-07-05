#!/usr/bin/env python3
"""Plot forgetting curves per review for a randomly-selected word.

For each review event, shows the estimated recall probability curve both
before and after the half-life update, making it easy to see how the
memory model adapts over time.
"""

import argparse
import datetime as dt
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

import srs_db
from incremental_model import (
    GRADE_WEIGHT,
    INITIAL_HALF_LIFE,
    SOLID_RECALL,
    TARGET_RECALL,
    SKILL_ORDER,
    next_interval,
    recall_probability,
)

_KOREAN_FONT = fm.FontProperties(
    fname=str(
        Path(__file__).resolve().parent
        / "assets"
        / "Noto_Sans_KR"
        / "static"
        / "NotoSansKR-Regular.ttf"
    )
)

GRADE_COLORS = {
    1: "#2ecc71",  # easy correct — green
    2: "#27ae60",  # hard correct — darker green
    3: "#e67e22",  # easy wrong — orange
    4: "#e74c3c",  # hard wrong — red
    5: "#8e44ad",  # no idea — purple
}

SKILL_COLORS = {
    "recognition": "#3498db",
    "occlusion": "#e67e22",
    "production": "#e74c3c",
}


def _format_half_life(secs: float) -> str:
    """Human-readable half-life."""
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}m"
    if secs < 86400:
        return f"{secs / 3600:.1f}h"
    return f"{secs / 86400:.1f}d"


def _gather_reviews(db_path: str) -> dict[str, dict[str, list[dict]]]:
    """Query all graded non-intro reviews, grouped by lexeme then skill.

    Returns: {lexeme: {skill: [{"reviewed_at": datetime, "grade": int, "H_after": float}, ...]}}
    """
    engine = srs_db.make_engine(db_path)
    by_lexeme: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    with Session(engine) as s:
        logs = (
            s.execute(
                select(srs_db.ReviewLog).order_by(srs_db.ReviewLog.reviewed_at)
            )
            .scalars()
            .all()
        )

        for log in logs:
            if log.grade is None or log.mode == "intro":
                continue
            payload = log.payload or {}
            skill = payload.get("incremental_skill")
            H_after = payload.get("half_life_secs")
            if skill is None or H_after is None:
                continue

            lexeme = srs_db.extract_lexeme_from_external_id(log.item.external_id)
            if not lexeme:
                continue

            by_lexeme[lexeme][skill].append(
                {
                    "reviewed_at": log.reviewed_at,
                    "grade": log.grade,
                    "H_after": H_after,
                }
            )

    return dict(by_lexeme)


def _pick_word(by_lexeme: dict, word: str | None, min_reviews: int) -> str | None:
    """Pick a word to plot. If word is specified, use it; otherwise pick randomly."""
    if word:
        if word not in by_lexeme:
            print(f"Word '{word}' not found in reviews. Available: {sorted(by_lexeme.keys())}")
            return None
        return word

    eligible = {
        k: v
        for k, v in by_lexeme.items()
        if sum(len(revs) for revs in v.values()) >= min_reviews
    }
    if not eligible:
        print(f"No words with >= {min_reviews} graded reviews found.")
        return None

    return random.choice(list(eligible.keys()))


def plot_forgetting_curves(
    db_path: str = "test_srs.sqlite",
    word: str | None = None,
    min_reviews: int = 3,
    save: bool = False,
) -> None:
    by_lexeme = _gather_reviews(db_path)
    if not by_lexeme:
        print("No graded incremental reviews found.")
        return

    chosen = _pick_word(by_lexeme, word, min_reviews)
    if chosen is None:
        return

    skill_reviews = by_lexeme[chosen]
    # only plot skills that have reviews, in canonical order
    skills_to_plot = [sk for sk in SKILL_ORDER if sk in skill_reviews and skill_reviews[sk]]
    if not skills_to_plot:
        print(f"No skill reviews found for '{chosen}'.")
        return

    n_skills = len(skills_to_plot)
    fig, axes = plt.subplots(n_skills, 1, figsize=(12, 4 * n_skills), squeeze=False)

    for idx, skill in enumerate(skills_to_plot):
        ax = axes[idx, 0]
        reviews = skill_reviews[skill]
        color = SKILL_COLORS.get(skill, "#555555")
        target_p = TARGET_RECALL.get(skill, 0.85)

        # build (reviewed_at, grade, H_before, H_after) list
        segments = []
        for i, rev in enumerate(reviews):
            H_before = INITIAL_HALF_LIFE if i == 0 else reviews[i - 1]["H_after"]
            segments.append(
                {
                    "reviewed_at": rev["reviewed_at"],
                    "grade": rev["grade"],
                    "H_before": H_before,
                    "H_after": rev["H_after"],
                }
            )

        # --- Phase 1: Draw forward decay curves ---
        # Each curve starts at p=1.0 at a review time and decays with
        # H_after[i].  Solid for observed intervals (next review exists);
        # dashed for the last (projected into the future).
        # The END of each solid curve is the next review's empty dot.
        for i, seg in enumerate(segments):
            t_start = seg["reviewed_at"]
            H = seg["H_after"]
            is_last = i + 1 >= len(segments)

            if not is_last:
                t_end = segments[i + 1]["reviewed_at"]
            else:
                extend_secs = max(next_interval(H, skill) * 2, 300)
                t_end = t_start + dt.timedelta(seconds=extend_secs)

            t0 = t_start.timestamp()
            t1 = t_end.timestamp()
            if t1 <= t0:
                continue
            ts = np.linspace(t0, t1, 300)
            delta_ts = ts - t0
            ps = np.array([recall_probability(H, d) for d in delta_ts])
            dates = [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc) for t in ts]

            if is_last:
                ax.plot(dates, ps, color=color, linewidth=1.2, linestyle="--", alpha=0.5)
            else:
                ax.plot(dates, ps, color=color, linewidth=1.5, alpha=0.8)

            ax.annotate(
                f"H={_format_half_life(H)}",
                xy=(dates[0], 1.0),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
                alpha=0.7,
                ha="left",
                va="bottom",
            )

        # --- Phase 2: Draw revealed curves and review markers ---
        # At each review (i >= 1) the model updates H.  We show:
        #   - empty dot: predicted recall on the OLD curve (H_before)
        #   - filled dot: revised recall on the REVEALED curve (H_after)
        #     from the same origin — higher if correct, lower if wrong
        #   - thin dashed revealed curve from previous origin with H_after
        #   - vertical connector between the two dots
        for i, seg in enumerate(segments):
            t_review = seg["reviewed_at"]
            grade = seg["grade"]
            grade_color = GRADE_COLORS.get(grade, "#999999")

            if i == 0:
                # First review — single filled dot at p=1.0
                ax.plot(
                    t_review, 1.0, "o",
                    color=grade_color, markersize=8,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5,
                )
                ax.annotate(
                    str(grade), xy=(t_review, 1.0),
                    xytext=(4, -12), textcoords="offset points",
                    fontsize=7, fontweight="bold", color=grade_color, alpha=0.9,
                )
                continue

            prev_t = segments[i - 1]["reviewed_at"]
            delta_secs = (t_review - prev_t).total_seconds()
            H_before = seg["H_before"]   # = H_after[i-1]
            H_after = seg["H_after"]

            # empty dot: model's predicted recall (on old curve)
            p_predicted = recall_probability(H_before, delta_secs)
            ax.plot(
                t_review, p_predicted, "o", markersize=8,
                markerfacecolor="none", markeredgecolor="grey",
                markeredgewidth=1.5, zorder=5,
            )

            # filled dot: grade evidence (what the review observed)
            p_evidence = GRADE_WEIGHT.get(grade, 0.5)
            ax.plot(
                t_review, p_evidence, "o",
                color=grade_color, markersize=8,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5,
            )

            # grade label next to filled dot
            ax.annotate(
                str(grade), xy=(t_review, p_evidence),
                xytext=(4, -12), textcoords="offset points",
                fontsize=7, fontweight="bold", color=grade_color, alpha=0.9,
            )

            # vertical connector between empty and filled dots
            if abs(p_predicted - p_evidence) > 0.01:
                ax.plot(
                    [t_review, t_review], [p_predicted, p_evidence],
                    color="grey", linewidth=0.8, linestyle=":",
                    alpha=0.5, zorder=2,
                )

            # revealed curve: H_after from previous origin, extending
            # past the review point to show the continuation
            if abs(H_before - H_after) / max(H_before, 1) > 0.01:
                t0 = prev_t.timestamp()
                t1 = t_review.timestamp()
                t1_ext = t1 + (t1 - t0) * 0.3
                ts = np.linspace(t0, t1_ext, 300)
                d_ts = ts - t0
                ps_rev = np.array([recall_probability(H_after, d) for d in d_ts])
                dates_rev = [dt.datetime.fromtimestamp(t, tz=dt.timezone.utc) for t in ts]
                ax.plot(dates_rev, ps_rev, color=color, linewidth=0.8,
                        alpha=0.25, linestyle="--")

        # --- reference lines ---
        ax.axhline(y=target_p, color=color, linewidth=0.8, linestyle="--", alpha=0.4)
        ax.text(
            ax.get_xlim()[0],
            target_p + 0.02,
            f"target p*={target_p}",
            fontsize=7,
            color=color,
            alpha=0.5,
        )
        ax.axhline(y=SOLID_RECALL, color="grey", linewidth=0.8, linestyle=":", alpha=0.3)
        ax.text(
            ax.get_xlim()[0],
            SOLID_RECALL + 0.02,
            f"solid={SOLID_RECALL}",
            fontsize=7,
            color="grey",
            alpha=0.5,
        )

        # --- legend (outside plot, right side) ---
        legend_handles = [
            mlines.Line2D([], [], color=color, linewidth=1.5, label="Decay (active H)"),
            mlines.Line2D([], [], color=color, linewidth=1.2, linestyle="--", alpha=0.5, label="Projected"),
            mlines.Line2D([], [], color=color, linewidth=0.8, linestyle="--", alpha=0.3, label="Revealed (new H)"),
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor="none",
                          markeredgecolor="grey", markeredgewidth=1.5, markersize=7, label="Predicted p"),
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#2ecc71", markersize=7, label="Correct (1-2)"),
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#e74c3c", markersize=7, label="Wrong (3-5)"),
        ]
        ax.legend(handles=legend_handles, fontsize=7,
                  loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.7, borderaxespad=0)

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Recall probability", fontsize=10)
        ax.set_title(f"{skill}", fontsize=11, fontweight="bold")
        ax.grid(axis="both", alpha=0.15)

    fig.autofmt_xdate(rotation=30)
    fig.suptitle(
        f"Forgetting curves: {chosen}",
        fontproperties=_KOREAN_FONT,
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    if save:
        img_dir = Path(__file__).resolve().parent / "img"
        for ext in ("png", "pdf"):
            out_path = img_dir / f"forgetting_curves.{ext}"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot forgetting curves for a word")
    parser.add_argument("--db", default="test_srs.sqlite", help="SQLite DB path")
    parser.add_argument("--word", default=None, help="Specific word to plot (random if omitted)")
    parser.add_argument("--min-reviews", type=int, default=3, help="Minimum reviews to be eligible for random pick")
    parser.add_argument("--save", action="store_true", help="Save plot to img/")
    args = parser.parse_args()
    plot_forgetting_curves(db_path=args.db, word=args.word, min_reviews=args.min_reviews, save=args.save)
