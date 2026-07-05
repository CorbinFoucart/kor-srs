#!/usr/bin/env python3
"""
rank_difficulty.py

Rank all reviewed lexemes from easiest to hardest based on review history.

Algorithm
─────────
1. Grade penalty: each grade maps to a difficulty score in [0, 1]:
     F|C          (1) → 0.0     (fluent, effortless recall)
     N|C          (2) → 0.15    (correct but had to think)
     H|C          (3) → 0.3     (correct but hard to retrieve)
     Easy|W       (4) → 0.7     (wrong but close)
     Hard|W       (5) → 0.9     (wrong, far off)
     No Idea      (6) → 1.0     (complete miss)

2. Exponential time-decay weighting (half-life = 7 days):
   Each review's weight = 2^(-age_days / 7).
   Recent reviews count much more than old ones so the score reflects
   *current* mastery rather than early learning stumbles.

3. Bayesian shrinkage toward the global mean:
   raw_score  = Σ(weight_i * penalty_i) / Σ(weight_i)
   final_score = (n * raw_score + k * global_mean) / (n + k)

   where n = number of reviews and k = 5 (prior strength).
   Words with very few reviews are pulled toward the population average,
   preventing a single lucky/unlucky grade from dominating.

4. Lexeme-level aggregation: reviews from both the prod and recog items
   of a lexeme are merged into one score, since we care about how well
   the *word* is known overall.

Output: table sorted from easiest (score ≈ 0) to hardest (score ≈ 1).
"""

import argparse
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

import srs_db

# ── grade → difficulty penalty ────────────────────────────────────────────
GRADE_PENALTY = {
    1: 0.0,   # F|C — fluent correct
    2: 0.15,  # N|C — non-fluent correct
    3: 0.3,   # H|C — hard correct
    4: 0.7,   # Easy|W
    5: 0.9,   # Hard|W
    6: 1.0,   # No Idea
}

HALF_LIFE_DAYS = 7.0
PRIOR_STRENGTH = 5     # Bayesian shrinkage pseudo-count


STRATEGY_SHORT = {
    "production_only":        "prod",
    "recognition_only":       "recog",
    "mixed":                  "mixed",
    "incremental_production": "incr",
}


def compute_lexeme_difficulties(session: Session) -> dict[str, float]:
    """Return {lexeme: difficulty_score} using review history from the given session.

    Scores range from 0.0 (easiest) to ~1.0 (hardest).  Lexemes with no
    review history are not included in the result.
    """
    now = srs_db.now_utc()

    logs = session.execute(
        select(srs_db.ReviewLog).order_by(srs_db.ReviewLog.reviewed_at)
    ).scalars().all()

    # collect (penalty, weight) per lexeme
    reviews_by_lex: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for log in logs:
        if log.grade is None or log.mode != "review":
            continue
        lexeme = srs_db.extract_lexeme_from_external_id(log.item.external_id)
        if lexeme is None:
            continue

        penalty = GRADE_PENALTY.get(log.grade, 0.5)
        reviewed = log.reviewed_at
        if reviewed.tzinfo is None:
            reviewed = reviewed.replace(tzinfo=srs_db.UTC)
        age_days = (now - reviewed).total_seconds() / 86400.0
        weight = 2.0 ** (-age_days / HALF_LIFE_DAYS)

        reviews_by_lex[lexeme].append((penalty, weight))

    if not reviews_by_lex:
        return {}

    # global weighted mean (shrinkage target)
    total_wp, total_w = 0.0, 0.0
    for pairs in reviews_by_lex.values():
        for p, w in pairs:
            total_wp += w * p
            total_w += w
    global_mean = total_wp / total_w if total_w > 0 else 0.5

    # per-lexeme scoring with Bayesian shrinkage
    result: dict[str, float] = {}
    for lex, pairs in reviews_by_lex.items():
        n = len(pairs)
        sum_wp = sum(w * p for p, w in pairs)
        sum_w = sum(w for _, w in pairs)
        raw = sum_wp / sum_w if sum_w > 0 else 0.5
        result[lex] = (n * raw + PRIOR_STRENGTH * global_mean) / (n + PRIOR_STRENGTH)

    return result


def rank_lexemes(db_path: str = "test_srs.sqlite") -> list[dict]:
    engine = srs_db.make_engine(db_path)

    with Session(engine) as s:
        scores = compute_lexeme_difficulties(s)
        if not scores:
            return []

        # collect per-lexeme review stats for the detailed table
        now = srs_db.now_utc()
        logs = s.execute(
            select(srs_db.ReviewLog).order_by(srs_db.ReviewLog.reviewed_at)
        ).scalars().all()

        reviews_by_lex: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for log in logs:
            if log.grade is None or log.mode != "review":
                continue
            lexeme = srs_db.extract_lexeme_from_external_id(log.item.external_id)
            if lexeme is None:
                continue
            penalty = GRADE_PENALTY.get(log.grade, 0.5)
            reviewed = log.reviewed_at
            if reviewed.tzinfo is None:
                reviewed = reviewed.replace(tzinfo=srs_db.UTC)
            age_days = (now - reviewed).total_seconds() / 86400.0
            weight = 2.0 ** (-age_days / HALF_LIFE_DAYS)
            reviews_by_lex[lexeme].append((penalty, weight))

        # load per-lexeme scheduling params from canonical (prod) item
        srs_params: dict[str, dict] = {}
        seen, _ = srs_db.classify_lexeme_groups(s)
        for lex, group in seen.items():
            for item in group["items"]:
                parsed = srs_db.parse_external_id(item.external_id)
                if parsed and parsed[1] == "cloze_prod_bundle" and item.srs_state:
                    st = item.srs_state.state or {}
                    srs_params[lex] = {
                        "base_interval": st.get("base_interval_secs"),
                        "multiplier": st.get("multiplier"),
                        "strategy": st.get("strategy"),
                    }
                    break

    results = []
    for lex, score in scores.items():
        pairs = reviews_by_lex.get(lex, [])
        n = len(pairs)
        sum_wp = sum(w * p for p, w in pairs)
        sum_w = sum(w for _, w in pairs)
        raw = sum_wp / sum_w if sum_w > 0 else 0.5

        correct = sum(1 for p, _ in pairs if p < 0.5)
        wrong = n - correct

        params = srs_params.get(lex, {})
        base_iv = params.get("base_interval")
        mult = params.get("multiplier")
        strategy = params.get("strategy")

        results.append({
            "lexeme": lex,
            "score": score,
            "raw": raw,
            "reviews": n,
            "correct": correct,
            "wrong": wrong,
            "base_interval": base_iv,
            "multiplier": mult,
            "strategy": STRATEGY_SHORT.get(strategy, "?") if strategy else "?",
        })

    results.sort(key=lambda r: r["score"])
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank words from easiest to hardest")
    ap.add_argument("--db", default="test_srs.sqlite")
    args = ap.parse_args()

    results = rank_lexemes(args.db)
    if not results:
        print("No review data found.")
        return

    # header
    print(f"\n  {'Rank':<5} {'Score':>6}  {'Raw':>5}  {'Rev':>4} {'C':>3}/{'W':>3}  {'Base':>5} {'Mult':>5} {'Strat':<5}  Lexeme")
    print(f"  {'─'*5} {'─'*6}  {'─'*5}  {'─'*4} {'─'*3} {'─'*3}  {'─'*5} {'─'*5} {'─'*5}  {'─'*20}")

    for i, r in enumerate(results, 1):
        base = f"{r['base_interval']:.0f}s" if r['base_interval'] is not None else "—"
        mult = f"{r['multiplier']:.1f}x" if r['multiplier'] is not None else "—"
        print(
            f"  {i:<5} {r['score']:6.3f}  {r['raw']:5.3f}  "
            f"{r['reviews']:4d} {r['correct']:3d}/{r['wrong']:3d}  "
            f"{base:>5} {mult:>5} {r['strategy']:<5}  {r['lexeme']}"
        )

    print(f"\n  {len(results)} lexemes ranked  |  prior strength k={PRIOR_STRENGTH}  |  half-life={HALF_LIFE_DAYS:.0f}d")
    print()

    plot_score_vs_mult(results)


STRATEGY_COLORS = {
    "prod":  "#e74c3c",
    "recog": "#2a69e8",
    "mixed": "#ff9800",
    "incr":  "#2ecc71",
    "?":     "#888888",
}


def plot_score_vs_mult(results: list[dict]) -> None:
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    _KOREAN_FONT = fm.FontProperties(
        fname=str(Path(__file__).resolve().parent / "assets" / "Noto_Sans_KR" / "static" / "NotoSansKR-Regular.ttf")
    )
    # filter to entries that have both values
    pts = [(r["score"], r["multiplier"], r["lexeme"], r["strategy"])
           for r in results if r["multiplier"] is not None]
    if not pts:
        print("No data with multiplier values to plot.")
        return

    scores     = [p[0] for p in pts]
    mults      = [p[1] for p in pts]
    labels     = [p[2] for p in pts]
    strategies = [p[3] for p in pts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # ── left: plain scatter with labels ──
    ax1.scatter(mults, scores, s=50, alpha=0.8, edgecolors="k", linewidths=0.5)
    for s, m, lbl, _ in pts:
        ax1.annotate(lbl, (m, s), fontsize=8, fontproperties=_KOREAN_FONT,
                     textcoords="offset points", xytext=(5, 4), alpha=0.7)
    ax1.set_xlabel("Multiplier", fontsize=12)
    ax1.set_ylabel("Difficulty Score", fontsize=12)
    ax1.set_title("Difficulty Score vs. Multiplier", fontsize=14)
    ax1.grid(alpha=0.3)

    # ── right: colored by strategy ──
    for strat, color in STRATEGY_COLORS.items():
        idx = [i for i, st in enumerate(strategies) if st == strat]
        if not idx:
            continue
        ax2.scatter(
            [mults[i] for i in idx], [scores[i] for i in idx],
            s=50, alpha=0.8, edgecolors="k", linewidths=0.5,
            color=color, label=strat,
        )
    for s, m, lbl, _ in pts:
        ax2.annotate(lbl, (m, s), fontsize=8, fontproperties=_KOREAN_FONT,
                     textcoords="offset points", xytext=(5, 4), alpha=0.7)
    ax2.set_xlabel("Multiplier", fontsize=12)
    ax2.set_title("Colored by Review Strategy", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
