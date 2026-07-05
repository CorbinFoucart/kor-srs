#!/usr/bin/env python3
"""
plot_word_trajectory.py

Plot the recognition review history of a single acquired word across two
side-by-side panels, revealed together over time (static PNG or animated GIF).

  Left  — recall between reviews. p = 2^(-Δt/H) resets to 1 at each review and
          decays with the half-life the model held then, so the trace is a
          sawtooth whose teeth widen as H grows. Review events sit on the curve
          at the recall the model predicted going in, marked Korean-style
          (blue O recalled, red X missed).
  Right — the half-life (blue) climbing over the same timeline.

    python plot_word_trajectory.py --lexeme 수송
    python plot_word_trajectory.py --lexeme 친환경 --out img/traj_친환경.png
    python plot_word_trajectory.py --lexeme 통하다 --animate      # -> img/traj_통하다.gif

Read-only: copies the DB to /tmp first, so it is safe while the server runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
sns.set_style("whitegrid")

# register the bundled Korean font so the headword renders (not tofu boxes);
# resolved relative to this file so it works from any working directory
_FONT = Path(__file__).resolve().parent / "assets/Noto_Sans_KR/static/NotoSansKR-Regular.ttf"
if _FONT.exists():
    font_manager.fontManager.addfont(str(_FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_FONT)).get_name()

# palette: grey forgetting curves in the background, blue half-life as the hero
# line, Korean-style O (blue, recalled) / X (red, missed) review marks.
C_CURVE = "#9AA6B2"  # forgetting (decay) curves — grey
C_HL = "#3B57E6"     # half-life step line — blue
C_OK = "#1E33CC"     # recalled marker — blue O
C_WRONG = "#D1495B"  # missed marker — red X

TARGET_RECALL = 0.80
LOG2P = -math.log2(TARGET_RECALL)  # interval = H * LOG2P


def _parse(s: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(s.strip().replace(" ", "T"))
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="test_srs.sqlite")
    ap.add_argument("--lexeme", required=True)
    ap.add_argument("--gloss", default="", help="short English gloss for the title")
    ap.add_argument("--out", default=None)
    ap.add_argument("--animate", action="store_true", help="save an animated GIF")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    src = Path(args.db)
    if not src.exists():
        raise SystemExit(f"DB not found: {src}")
    tmp = "/tmp/_word_traj.db"
    shutil.copy2(src, tmp)
    con = sqlite3.connect(tmp)
    rows = con.execute(
        """SELECT r.reviewed_at, r.grade, r.payload
           FROM review_log r JOIN items i ON i.id = r.item_id
           WHERE i.external_id LIKE ? AND r.payload LIKE '%maintenance%'
           ORDER BY r.reviewed_at""",
        (f"lexeme:{args.lexeme}:%",),
    ).fetchall()
    con.close()
    gloss = args.gloss

    # keep only maintenance-phase reviews
    revs = []  # (t, grade, H_post)
    for ra, g, pl in rows:
        p = json.loads(pl) if pl else {}
        if p.get("phase") != "maintenance":
            continue
        revs.append((_parse(ra), g, p.get("half_life_secs")))
    if len(revs) < 2:
        raise SystemExit(f"not enough maintenance reviews for {args.lexeme}")

    t0 = revs[0][0]
    days = [(t - t0).total_seconds() / 86400 for t, _, _ in revs]
    grades = [g for _, g, _ in revs]
    H_days = [(H / 86400 if H else np.nan) for _, _, H in revs]
    n = len(revs)

    # layout constants (fixed across animation frames)
    xmax = days[-1] + (H_days[-1] * LOG2P) * 1.05
    ymax2 = max(H_days) * 1.15
    title = f"{args.lexeme}" + (f"  ·  {gloss}" if gloss else "")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], color=C_CURVE, lw=1.8, label="forgetting curve"),
        Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=C_OK,
               markeredgewidth=1.8, ls="", label="recalled (O)"),
        Line2D([], [], marker="x", color=C_WRONG, markeredgewidth=2.2,
               ls="", label="missed (X)"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=120,
                                   constrained_layout=True)

    def draw(t_cursor: float) -> None:
        ax1.clear(); ax2.clear()
        idx = sum(1 for d in days if d <= t_cursor)   # reviews so far

        # ── left panel: recall between reviews (forgetting curves + O/X) ──
        # each review resets recall to 1 and decays with that review's
        # post-update H; the last segment runs to the scheduled next due.
        for i in range(n):
            H = H_days[i]
            if not H or math.isnan(H) or days[i] > t_cursor:
                continue
            t_start = days[i]
            t_end = days[i + 1] if i + 1 < n else t_start + H * LOG2P
            t_draw = min(t_end, t_cursor)
            npts = max(2, min(120, int((t_draw - t_start) / (xmax / 400)) + 2))
            xs = np.linspace(t_start, t_draw, npts)
            ax1.plot(xs, 2.0 ** (-(xs - t_start) / H), color=C_CURVE,
                     lw=1.8, zorder=3)
            ax1.scatter([t_start], [1.0], s=20, color=C_CURVE, zorder=4)  # start
        for i in range(1, n):
            if days[i] > t_cursor:
                continue
            H_prev = H_days[i - 1]
            if not H_prev or math.isnan(H_prev):
                continue
            p_hat = 2.0 ** (-(days[i] - days[i - 1]) / H_prev)
            if grades[i] is not None and grades[i] <= 3:
                ax1.scatter([days[i]], [p_hat], s=70, marker="o",
                            facecolors="none", edgecolors=C_OK,
                            linewidths=1.8, zorder=6)
            else:
                ax1.scatter([days[i]], [p_hat], s=85, marker="x",
                            color=C_WRONG, linewidths=2.2, zorder=6)
        ax1.axhline(TARGET_RECALL, color="grey", ls="--", lw=1.1, alpha=0.8)
        ax1.text(xmax, TARGET_RECALL + 0.02, "target recall 0.80",
                 ha="right", va="bottom", fontsize=8, color="grey")
        ax1.set_ylim(0, 1.2); ax1.set_xlim(0, xmax)
        ax1.set_xlabel("days since first maintenance review")
        ax1.set_ylabel("predicted recall  p = 2^(−Δt/H)")
        ax1.set_title("Recall between reviews", fontsize=12)
        ax1.set_box_aspect(1)
        ax1.legend(handles=legend_handles, loc="upper right", fontsize=8,
                   framealpha=0.9)
        sns.despine(ax=ax1)

        # ── right panel: half-life over time (blue step, revealed to cursor) ──
        if idx >= 1:
            xs_hl = days[:idx] + [min(t_cursor, xmax)]
            ys_hl = H_days[:idx] + [H_days[idx - 1]]
            ax2.step(xs_hl, ys_hl, where="post", color=C_HL, lw=2.6, zorder=3)
            ax2.scatter([min(t_cursor, xmax)], [H_days[idx - 1]], s=45,
                        color=C_OK, edgecolor="white", zorder=5)
        ax2.set_ylim(0, ymax2); ax2.set_xlim(0, xmax)
        ax2.set_xlabel("days since first maintenance review")
        ax2.set_ylabel("half-life (days)")
        ax2.set_title("Half-life over time", fontsize=12)
        ax2.set_box_aspect(1)
        sns.despine(ax=ax2)

        n_wrong = sum(1 for g in grades[:idx] if g is not None and g >= 4)
        cur_H = H_days[idx - 1] if idx >= 1 else H_days[0]
        fig.suptitle(f"{title}      {idx} reviews · {n_wrong} missed · "
                     f"half-life {cur_H:.0f} days", fontsize=13)

    if args.animate:
        from matplotlib.animation import FuncAnimation, PillowWriter
        cursors = list(np.linspace(days[0] + 1e-3, xmax, 200))
        cursors += [xmax] * 20   # hold the final frame
        anim = FuncAnimation(fig, lambda c: draw(c), frames=cursors, interval=50)
        out = args.out or f"img/traj_{args.lexeme}.gif"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        anim.save(out, writer=PillowWriter(fps=args.fps))
        print(f"saved {out} ({len(cursors)} frames)")
    else:
        draw(xmax)
        out = args.out or f"img/traj_{args.lexeme}.png"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
