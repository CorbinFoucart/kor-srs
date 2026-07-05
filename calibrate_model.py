#!/usr/bin/env python3
"""Replay-based parameter tuning for the Bernoulli NLL gradient SRS model.

Replays the entire review history under candidate parameters and finds
optimal values via scipy L-BFGS-B minimisation of negative log-likelihood.

Usage:
    python calibrate_model.py --db test_srs.sqlite [--save] [--method optimize|grid]
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sqlalchemy import select
from sqlalchemy.orm import Session

import srs_db
from incremental_model import (
    ETA_BASE,
    ETA_BOOST,
    GRADE_WEIGHT,
    H_SCALE,
    INITIAL_HALF_LIFE,
    MAX_HALF_LIFE,
    MAX_ODDS,
    MIN_HALF_LIFE,
    SKILL_ORDER,
    recall_probability,
    next_interval,
)
from acquisition_model import GRADUATION_H

_KOREAN_FONT = fm.FontProperties(
    fname=str(
        Path(__file__).resolve().parent
        / "assets"
        / "Noto_Sans_KR"
        / "static"
        / "NotoSansKR-Regular.ttf"
    )
)

SKILL_COLORS = {
    "recognition": "#3498db",
    "occlusion": "#e67e22",
    "production": "#e74c3c",
}

EPSILON = 1e-10


# ── Data extraction ───────────────────────────────────────────────────

def _extract_reviews(db_path: str, *, maintenance_only: bool = True,
                     recognition_only: bool = False) -> list[dict]:
    """Load graded reviews as flat records.

    Args:
        maintenance_only: If True (default), only include maintenance-phase reviews
            for calibration. Day 0, acquisition, and repair reviews are excluded
            since the half-life model only applies to maintenance.
        recognition_only: If True, keep only recognition-skill reviews and DROP
            legacy (phase-less) reviews — the deck is recognition-scheduled now,
            and legacy/production reviews are a different regime that would
            contaminate a recognition fit.
    """
    engine = srs_db.make_engine(db_path)
    reviews = []
    phase_counts = {"maintenance": 0, "day0": 0, "acquiring": 0, "repairing": 0, "legacy": 0}

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

            # Determine phase from payload (new format) or fall back to legacy
            phase = payload.get("phase")
            skill = payload.get("incremental_skill")

            if phase is not None:
                # New format: phase is explicit
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                if maintenance_only and phase != "maintenance":
                    continue
                skill = skill or "production"
            else:
                # Legacy format: no phase field, treat as maintenance
                phase_counts["legacy"] += 1
                if skill is None:
                    continue
                if recognition_only:
                    continue  # drop pre-phase legacy reviews (regime unknown)

            if recognition_only and skill != "recognition":
                continue

            lexeme = srs_db.extract_lexeme_from_external_id(log.item.external_id)
            if not lexeme:
                continue

            reviews.append({
                "lexeme": lexeme,
                "skill": skill,
                "grade": log.grade,
                "reviewed_at": log.reviewed_at,
                "correct": log.grade in (1, 2, 3),
            })

    # Print phase breakdown for diagnostics
    total_excluded = sum(v for k, v in phase_counts.items() if k not in ("maintenance", "legacy"))
    if total_excluded > 0 and maintenance_only:
        print(f"  Phase breakdown: {dict(phase_counts)}")
        print(f"  Excluded {total_excluded} non-maintenance reviews from calibration")

    return reviews


# ── Parametric model ──────────────────────────────────────────────────

def _default_params() -> dict:
    """Return current production parameter values."""
    params = {
        "eta_base": ETA_BASE,
        "eta_boost": ETA_BOOST,
        "h_scale": H_SCALE,
        # seed = the H a word actually enters maintenance at (graduation), NOT the
        # legacy INITIAL_HALF_LIFE; this is what the replay's first review uses.
        "initial_half_life": GRADUATION_H,
        "max_odds": MAX_ODDS,
    }
    for g in range(1, 7):
        params[f"w_{g}"] = GRADE_WEIGHT[g]
    return params


def _update_half_life_parametric(
    H: float, delta_t: float, grade: int, params: dict, skill: str = "recognition"
) -> float:
    """Parametric version of update_half_life for replay."""
    eta = params["eta_base"] + params["eta_boost"] / (1.0 + H / params["h_scale"])
    w = params.get(f"w_{grade}", 0.5)
    max_odds = params.get("max_odds", MAX_ODDS)

    if grade in (1, 2, 3):
        delta_logH = eta * w
    else:
        p_hat = recall_probability(H, delta_t)
        odds = min(p_hat / (1.0 - p_hat + 1e-10), max_odds)
        delta_logH = -eta * w * odds

    new_H = H * math.exp(delta_logH)
    return max(MIN_HALF_LIFE, min(MAX_HALF_LIFE, new_H))


# ── Forward replay ────────────────────────────────────────────────────

def _replay_history(
    reviews: list[dict], params: dict
) -> list[tuple[float, bool]]:
    """Replay review history under candidate params.

    Returns list of (p_predicted, correct) for each review.
    """
    sim_state: dict[tuple[str, str], dict] = {}
    init_H = params["initial_half_life"]
    results = []

    for r in reviews:
        key = (r["lexeme"], r["skill"])
        prev = sim_state.get(key)

        if prev is not None:
            H = prev["H"]
            delta_t = (r["reviewed_at"] - prev["reviewed_at"]).total_seconds()
            delta_t = max(delta_t, 1.0)
        else:
            H = init_H
            # first maintenance review lands ~at its scheduled interval, not 2min
            delta_t = next_interval(H, r["skill"])

        p_hat = recall_probability(H, delta_t)
        results.append((p_hat, r["correct"]))

        new_H = _update_half_life_parametric(H, delta_t, r["grade"], params, skill=r["skill"])
        sim_state[key] = {"H": new_H, "reviewed_at": r["reviewed_at"]}

    return results


def _replay_history_detailed(
    reviews: list[dict], params: dict
) -> list[dict]:
    """Like _replay_history but returns full detail dicts for diagnostics."""
    sim_state: dict[tuple[str, str], dict] = {}
    init_H = params["initial_half_life"]
    results = []

    for r in reviews:
        key = (r["lexeme"], r["skill"])
        prev = sim_state.get(key)

        if prev is not None:
            H = prev["H"]
            delta_t = (r["reviewed_at"] - prev["reviewed_at"]).total_seconds()
            delta_t = max(delta_t, 1.0)
        else:
            H = init_H
            # first maintenance review lands ~at its scheduled interval, not 2min
            delta_t = next_interval(H, r["skill"])

        p_hat = recall_probability(H, delta_t)
        new_H = _update_half_life_parametric(H, delta_t, r["grade"], params, skill=r["skill"])

        results.append({
            "lexeme": r["lexeme"],
            "skill": r["skill"],
            "grade": r["grade"],
            "correct": r["correct"],
            "p_predicted": p_hat,
            "H_before": H,
            "H_after": new_H,
            "delta_t": delta_t,
        })

        sim_state[key] = {"H": new_H, "reviewed_at": r["reviewed_at"]}

    return results


# ── Scoring functions ─────────────────────────────────────────────────

def _nll(results: list[tuple[float, bool]]) -> float:
    """Mean negative log-likelihood (binary cross-entropy)."""
    total = 0.0
    for p_hat, correct in results:
        p = max(EPSILON, min(1 - EPSILON, p_hat))
        if correct:
            total -= math.log(p)
        else:
            total -= math.log(1 - p)
    return total / max(len(results), 1)


def _brier_score(results: list[tuple[float, bool]]) -> float:
    """Mean Brier score: mean((p - y)^2)."""
    total = 0.0
    for p_hat, correct in results:
        y = 1.0 if correct else 0.0
        total += (p_hat - y) ** 2
    return total / max(len(results), 1)


def _calibration_error(results: list[tuple[float, bool]], n_bins: int = 10) -> float:
    """Weighted mean calibration error across bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    total_error = 0.0
    total_n = 0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [(p, c) for p, c in results
                  if lo <= p < hi or (i == n_bins - 1 and p == hi)]
        n = len(in_bin)
        if n == 0:
            continue
        obs = sum(1 for _, c in in_bin if c) / n
        mean_p = sum(p for p, _ in in_bin) / n
        total_error += abs(obs - mean_p) * n
        total_n += n

    return total_error / max(total_n, 1)


# ── Optimizer ─────────────────────────────────────────────────────────

# Free parameters: eta_base, eta_boost, h_scale, w_1..w_6, initial_half_life, max_odds
_PARAM_NAMES = [
    "eta_base", "eta_boost", "h_scale",
    "w_1", "w_2", "w_3", "w_4", "w_5", "w_6",
    "initial_half_life",
    "max_odds",
]
_BOUNDS = [
    (0.1, 3.0),      # eta_base
    (0.1, 5.0),      # eta_boost
    (100.0, 30000.0), # h_scale
    (0.1, 3.0),      # w_1
    (0.05, 2.0),     # w_2
    (0.01, 1.5),     # w_3
    (0.01, 1.5),     # w_4
    (0.05, 2.0),     # w_5
    (0.1, 3.0),      # w_6
    (3600.0, 2_592_000.0),  # initial_half_life (graduation seed: 1h … 30d)
    (2.0, 100.0),    # max_odds
]


def _vec_to_params(x: np.ndarray) -> dict:
    """Convert optimizer vector to full params dict."""
    return dict(zip(_PARAM_NAMES, x))


def _params_to_vec(params: dict) -> np.ndarray:
    return np.array([params[k] for k in _PARAM_NAMES])


def _objective(x: np.ndarray, reviews: list[dict]) -> float:
    """NLL objective for scipy.optimize.minimize."""
    params = _vec_to_params(x)
    results = _replay_history(reviews, params)
    return _nll(results)


def run_optimize(reviews: list[dict]) -> dict:
    """Run L-BFGS-B optimization, return best params dict."""
    x0 = _params_to_vec(_default_params())
    res = minimize(
        _objective,
        x0,
        args=(reviews,),
        method="L-BFGS-B",
        bounds=_BOUNDS,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    print(f"Optimizer converged: {res.success}  (message: {res.message})")
    print(f"Function evaluations: {res.nfev}")
    return _vec_to_params(res.x)


def run_grid(reviews: list[dict]) -> dict:
    """Coarse grid search over eta_base, eta_boost, and initial_half_life."""
    grid = {
        "eta_base": np.linspace(0.3, 1.5, 5),
        "eta_boost": np.linspace(0.5, 3.0, 5),
        "initial_half_life": np.array([90, 120, 180, 240, 360, 496, 1800, 3600]),
    }
    best_nll = float("inf")
    best_params = _default_params()
    n_combos = np.prod([len(v) for v in grid.values()])
    print(f"Grid search: {int(n_combos)} combinations...")

    defaults = _default_params()
    for eb in grid["eta_base"]:
        for ebo in grid["eta_boost"]:
            for ih in grid["initial_half_life"]:
                params = dict(defaults)
                params["eta_base"] = float(eb)
                params["eta_boost"] = float(ebo)
                params["initial_half_life"] = float(ih)
                results = _replay_history(reviews, params)
                score = _nll(results)
                if score < best_nll:
                    best_nll = score
                    best_params = params.copy()

    print(f"Grid search best NLL: {best_nll:.6f}")
    return best_params


# ── Output / display ──────────────────────────────────────────────────

def _print_comparison(current: dict, optimized: dict) -> None:
    """Print current vs optimized parameters side by side."""
    print("\n" + "=" * 60)
    print("PARAMETER COMPARISON")
    print("=" * 60)
    print(f"{'Parameter':<22} {'Current':>12} {'Optimized':>12} {'Change':>12}")
    print("-" * 60)
    for key in _PARAM_NAMES:
        cur = current[key]
        opt = optimized[key]
        diff = opt - cur
        sign = "+" if diff > 0 else ""
        print(f"  {key:<20} {cur:>12.4f} {opt:>12.4f} {sign}{diff:>11.4f}")
    print()


def _print_scores(label: str, results: list[tuple[float, bool]]) -> None:
    """Print NLL, Brier, and calibration error."""
    nll = _nll(results)
    brier = _brier_score(results)
    cal = _calibration_error(results)
    print(f"  {label:<12}  NLL={nll:.6f}  Brier={brier:.6f}  CalError={cal:.6f}")


def _print_calibration_table(label: str, detailed: list[dict]) -> None:
    """Print calibration table from detailed replay results."""
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    print(f"\n  Calibration table ({label}):")
    print(f"  {'p_predicted bin':>18} {'n':>6} {'obs_correct':>12} {'mean_p':>10} {'error':>8}")
    print("  " + "-" * 58)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [r for r in detailed
                  if lo <= r["p_predicted"] < hi or (i == n_bins - 1 and r["p_predicted"] == hi)]
        n = len(in_bin)
        if n == 0:
            continue
        obs = sum(1 for r in in_bin if r["correct"]) / n
        mean_p = sum(r["p_predicted"] for r in in_bin) / n
        err = abs(obs - mean_p)
        print(f"  [{lo:.1f}, {hi:.1f}){' ' * 4}{n:>6} {obs:>12.3f} {mean_p:>10.3f} {err:>8.3f}")
    print()


def _print_interval_diagnostics(detailed: list[dict]) -> None:
    """Print max H and implied max interval to sanity-check growth."""
    if not detailed:
        return
    max_H = max(r["H_after"] for r in detailed)
    max_interval_secs = max_H * 0.322
    print(f"  Max simulated H: {_format_half_life(max_H)}")
    print(f"  Max implied interval (p*=0.80): {_format_half_life(max_interval_secs)}")
    print(f"  (Capped by MAX_HALF_LIFE={_format_half_life(MAX_HALF_LIFE)})")
    print()


def _format_half_life(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}m"
    if secs < 86400:
        return f"{secs / 3600:.1f}h"
    return f"{secs / 86400:.1f}d"


# ── Plotting ──────────────────────────────────────────────────────────

def _plot_calibration_comparison(
    before_detailed: list[dict],
    after_detailed: list[dict],
    save: bool,
) -> None:
    """Calibration plot: overall + per-skill, before vs after."""
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    skills_present = sorted(
        set(r["skill"] for r in before_detailed),
        key=lambda s: SKILL_ORDER.index(s) if s in SKILL_ORDER else 99,
    )
    n_cols = 1 + len(skills_present)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)

    def _bin_data(detailed):
        mean_ps, obs_rs, ns = [], [], []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = [r for r in detailed
                      if lo <= r["p_predicted"] < hi or (i == n_bins - 1 and r["p_predicted"] == hi)]
            n = len(in_bin)
            if n == 0:
                continue
            mp = sum(r["p_predicted"] for r in in_bin) / n
            obs = sum(1 for r in in_bin if r["correct"]) / n
            mean_ps.append(mp)
            obs_rs.append(obs)
            ns.append(n)
        return np.array(mean_ps), np.array(obs_rs), np.array(ns)

    def _plot_one(ax, before_data, after_data, title):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Perfect")

        bmp, bor, bns = _bin_data(before_data)
        if len(bmp) > 0:
            ses = np.sqrt(bor * (1 - bor) / np.maximum(bns, 1))
            ax.errorbar(bmp, bor, yerr=1.96 * ses, fmt="s", color="#cc4444",
                        markersize=5, capsize=3, alpha=0.6, label="Before")

        amp, aor, ans = _bin_data(after_data)
        if len(amp) > 0:
            ses = np.sqrt(aor * (1 - aor) / np.maximum(ans, 1))
            ax.errorbar(amp, aor, yerr=1.96 * ses, fmt="o", color="#2266cc",
                        markersize=5, capsize=3, label="After")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Mean predicted p(recall)")
        ax.set_ylabel("Observed correct fraction")
        ax.set_title(title, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)

    _plot_one(axes[0, 0], before_detailed, after_detailed, "Overall")
    for i, skill in enumerate(skills_present):
        bd = [r for r in before_detailed if r["skill"] == skill]
        ad = [r for r in after_detailed if r["skill"] == skill]
        _plot_one(axes[0, i + 1], bd, ad, skill.capitalize())

    fig.suptitle("Calibration: Before vs After Tuning", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        img_dir = Path(__file__).resolve().parent / "img"
        img_dir.mkdir(exist_ok=True)
        for ext in ("png", "pdf"):
            out_path = img_dir / f"calibration_tuned.{ext}"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Replay-based SRS parameter tuning")
    parser.add_argument("--db", default="test_srs.sqlite", help="SQLite DB path")
    parser.add_argument("--save", action="store_true", help="Save plots to img/")
    parser.add_argument(
        "--method", choices=["optimize", "grid"], default="optimize",
        help="Optimization method (default: optimize)",
    )
    parser.add_argument(
        "--recognition-only", action="store_true",
        help="Calibrate on recognition+maintenance reviews only (drop legacy/production)",
    )
    args = parser.parse_args()

    print(f"Loading reviews from {args.db}...")
    reviews = _extract_reviews(args.db, recognition_only=args.recognition_only)
    if not reviews:
        print("No graded incremental reviews found.")
        return

    n_correct = sum(1 for r in reviews if r["correct"])
    print(f"Found {len(reviews)} graded reviews ({n_correct} correct, "
          f"{len(reviews) - n_correct} wrong) across "
          f"{len(set(r['lexeme'] for r in reviews))} lexemes.")

    from collections import Counter
    gc = Counter(r["grade"] for r in reviews)
    print(f"Grade distribution: {dict(sorted(gc.items()))}")
    print()

    # ── Current params (before) ───────────────────────────────────────
    current_params = _default_params()
    before_results = _replay_history(reviews, current_params)
    before_detailed = _replay_history_detailed(reviews, current_params)

    print("=" * 60)
    print("CURRENT PARAMETERS")
    print("=" * 60)
    _print_scores("Current", before_results)
    _print_calibration_table("Current", before_detailed)
    _print_interval_diagnostics(before_detailed)

    # ── Optimize ──────────────────────────────────────────────────────
    print("=" * 60)
    print(f"RUNNING {'GRID SEARCH' if args.method == 'grid' else 'L-BFGS-B OPTIMIZATION'}...")
    print("=" * 60)

    if args.method == "grid":
        best_params = run_grid(reviews)
    else:
        best_params = run_optimize(reviews)

    # ── Optimized results (after) ─────────────────────────────────────
    after_results = _replay_history(reviews, best_params)
    after_detailed = _replay_history_detailed(reviews, best_params)

    _print_comparison(current_params, best_params)

    print("=" * 60)
    print("OPTIMIZED PARAMETERS")
    print("=" * 60)
    _print_scores("Optimized", after_results)
    _print_calibration_table("Optimized", after_detailed)
    _print_interval_diagnostics(after_detailed)

    # ── Side-by-side scores ───────────────────────────────────────────
    print("=" * 60)
    print("BEFORE vs AFTER")
    print("=" * 60)
    _print_scores("Before", before_results)
    _print_scores("After ", after_results)
    print()

    # ── Plot ──────────────────────────────────────────────────────────
    _plot_calibration_comparison(before_detailed, after_detailed, args.save)
    plt.show()


if __name__ == "__main__":
    main()
