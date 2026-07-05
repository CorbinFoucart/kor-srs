#!/usr/bin/env python3
"""
render_cards_due_gif.py

Animate the "days-until-due" distribution across the deck, one frame per day,
over the recognition-scheduled era. Reconstructs each lexeme's due date as of
each day from the half_life_secs logged on every maintenance ReviewLog:
    due = last_review + next_interval(H)   (interval = -H*log2(0.80))

  python render_cards_due_gif.py                 # -> cards_due.gif here
  python render_cards_due_gif.py --db X --out Y --start 2026-05-24

Read-only: copies the DB to /tmp first, so it's safe while the server runs.
"""

from __future__ import annotations

import argparse
import bisect
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
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style("whitegrid")  # white background (not grey), faint gridlines
from matplotlib.animation import FuncAnimation, PillowWriter

# brighter blue palette (periwinkle bars, vivid line, deep dot)
C_FILL = "#7E8CF2"   # histogram bars + rug
C_LINE = "#3B57E6"   # KDE curve + acquired-over-time line
C_DOT = "#1E33CC"    # center-of-mass dot

LOG2 = -math.log2(0.80)  # interval = H * 0.3219 at recognition target recall


def _parse(s: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(s.strip().replace(" ", "T"))
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="test_srs.sqlite")
    ap.add_argument("--out", default="cards_due.gif")
    ap.add_argument("--start", default="2026-05-24", help="first frame date (YYYY-MM-DD)")
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    src = Path(args.db)
    if not src.exists():
        raise SystemExit(f"DB not found: {src}")
    tmp = "/tmp/_cards_due_hist.db"
    shutil.copy2(src, tmp)
    con = sqlite3.connect(tmp)
    rows = con.execute(
        """SELECT r.reviewed_at, r.payload, i.external_id
           FROM review_log r JOIN items i ON i.id = r.item_id
           WHERE r.payload LIKE '%half_life_secs%' ORDER BY r.reviewed_at"""
    ).fetchall()
    # first review (any mode) per lexeme → running "total words studied"
    first_seen: dict[str, dt.datetime] = {}
    for ra, ext in con.execute(
        """SELECT r.reviewed_at, i.external_id FROM review_log r
           JOIN items i ON i.id = r.item_id WHERE r.reviewed_at IS NOT NULL"""):
        lex = ext.split(":")[1]
        t = _parse(ra)
        if lex not in first_seen or t < first_seen[lex]:
            first_seen[lex] = t
    con.close()
    first_seen_sorted = sorted(first_seen.values())

    tl: dict[str, tuple[list, list]] = {}
    for ra, pl, ext in rows:
        p = json.loads(pl) if pl else {}
        H = p.get("half_life_secs")
        if H and p.get("phase") == "maintenance":
            lex = ext.split(":")[1]
            tl.setdefault(lex, ([], []))
            tl[lex][0].append(_parse(ra))
            tl[lex][1].append(H)
    if not tl:
        raise SystemExit("no maintenance half-life data to animate")

    end = max(t for ts, _ in tl.values() for t in ts)
    start = _parse(args.start + "T12:00:00")
    days, d = [], start
    while d <= end:
        days.append(d); d += dt.timedelta(days=1)

    bins = np.arange(-7, 93, 3.0)

    def frame(D):
        vals = []
        for ts, Hs in tl.values():
            i = bisect.bisect_right(ts, D) - 1
            if i < 0:
                continue
            due = ts[i] + dt.timedelta(seconds=Hs[i] * LOG2)
            vals.append((due - D).total_seconds() / 86400)
        return vals

    from scipy.stats import gaussian_kde
    frames = [frame(D) for D in days]
    clipped = [np.clip(np.array(v, float), bins[0], bins[-1] - 1e-6) if v else np.array([])
               for v in frames]
    dens = [np.histogram(cv, bins=bins, density=True)[0] if len(cv) else np.zeros(len(bins) - 1)
            for cv in clipped]
    total_by = [bisect.bisect_right(first_seen_sorted, D) for D in days]   # words studied
    acquired_by = [len(v) for v in frames]                                 # in maintenance

    # fixed y-limit from histogram-density + KDE peaks so the flattening reads
    grid = np.linspace(bins[0], bins[-1], 240)
    def _kdepeak(cv):
        if len(cv) < 3 or np.ptp(cv) == 0:
            return 0.0
        try:
            return float(gaussian_kde(cv)(grid).max())
        except Exception:
            return 0.0
    ymax = max([max(d.max() if len(d) else 0.0, _kdepeak(cv))
                for d, cv in zip(dens, clipped)] or [1.0]) * 1.15

    ymaxline = max(acquired_by) * 1.08   # fixed y for the line plot (acquired only)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=110,
                                   constrained_layout=True)

    def draw(k):
        # ── left: due-by-time distribution ──
        ax1.clear()
        cv = clipped[k]
        if len(cv):
            sns.histplot(x=cv, bins=list(bins), stat="density", kde=True, ax=ax1,
                         color=C_FILL, edgecolor="white", alpha=0.60,
                         line_kws={"color": C_LINE, "lw": 2})
            sns.rugplot(x=cv, ax=ax1, height=0.035, color=C_FILL, lw=0.6, alpha=0.5)
            # center of mass: a dot sitting on top of the bar that contains it
            com = float(np.mean(cv))
            idx = int(np.clip(np.digitize(com, bins) - 1, 0, len(dens[k]) - 1))
            ax1.scatter([com], [dens[k][idx]], s=95, color=C_DOT, edgecolor="white",
                        linewidth=1.2, zorder=6)
        ax1.axvline(0, color="grey", ls="--", lw=1.2, alpha=0.85)          # due-now
        ax1.set_xlim(bins[0], bins[-1]); ax1.set_ylim(0, ymax)
        ax1.set_xticks(np.arange(0, 91, 20))                               # uniform gridlines
        ax1.set_xlabel("days until due  (←overdue | future→)"); ax1.set_ylabel("density")
        ax1.set_title(f"Cards due by time — {days[k].date()}", fontsize=12)
        ax1.set_box_aspect(1)                                              # square panel
        sns.despine(ax=ax1)
        ax1.text(0.98, 0.96, f"total: {total_by[k]}\nacquired: {acquired_by[k]}",
                 transform=ax1.transAxes, ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.8))

        # ── right: acquired words over time (fixed axes, revealed to date k) ──
        ax2.clear()
        ax2.plot(days[:k + 1], acquired_by[:k + 1], color=C_LINE, lw=2.4)
        ax2.scatter([days[k]], [acquired_by[k]], s=45, color=C_DOT,
                    edgecolor="white", zorder=6)
        ax2.set_xlim(days[0], days[-1]); ax2.set_ylim(0, ymaxline)         # fixed to max
        ax2.set_ylabel("acquired words"); ax2.set_title("Acquired words over time", fontsize=12)
        # uniform (weekly) date gridlines — avoids the ragged month-boundary ticks
        wk = []
        _d = days[0]
        while _d <= days[-1]:
            wk.append(_d); _d += dt.timedelta(days=7)
        ax2.set_xticks(wk)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax2.tick_params(axis="x", rotation=45)
        ax2.set_box_aspect(1)                                              # square panel
        sns.despine(ax=ax2)

    anim = FuncAnimation(fig, draw, frames=len(days), interval=200)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"saved {args.out} ({len(days)} frames)")


if __name__ == "__main__":
    main()
