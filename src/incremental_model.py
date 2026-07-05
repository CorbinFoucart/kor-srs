"""
incremental_model.py

Pure memory-model functions for the INCREMENTAL_PRODUCTION strategy.

Half-life update uses the gradient of the Bernoulli negative log-likelihood
with respect to log(H), with per-grade fluency weights.

No DB or queue dependencies — all functions operate on plain data
structures so they are easy to unit-test.
"""

from __future__ import annotations

import math
import datetime as dt
from dataclasses import dataclass, field

# ── constants ──────────────────────────────────────────────────────────

# Per-grade fluency weights: how much each grade affects the H update.
# Correct grades (1-3): higher w → bigger H growth.
# Wrong grades (4-6): higher w → bigger H penalty.
# Recognition-calibrated (tempered fit, 2026-06): recognition slips are rare and
# weakly predictive, so wrong-grade weights (w_4/w_5) are small; correct grades
# keep a strong gradient.
GRADE_WEIGHT = {1: 3.0, 2: 2.0, 3: 1.5, 4: 0.05, 5: 0.10, 6: 1.0}

TARGET_RECALL = {
    "recognition":        0.80,
    "occlusion":          0.85,
    "production":         0.80,
    "grammar_production": 0.80,
}

ETA_BASE  = 0.50     # asymptotic η for large H (recognition-calibrated 2026-06)
ETA_BOOST = 5.0      # additional η for small H
H_SCALE   = 3000.0   # transition scale (seconds)
ETA = ETA_BASE       # backward-compat alias (rough approximation for large H)

INITIAL_HALF_LIFE = 883.0           # seconds (~15 min), for newly unlocked skills
MIN_HALF_LIFE = 5.0                 # clamp floor

# MAX_HALF_LIFE: H where interval = 1 year at target recall — the RETIREMENT
# ("learned") threshold. interval = -H * log2(p*), so H = interval / -log2(p*)
_LEARNED_INTERVAL_SECS = 365.0 * 86400.0  # 1 year
MAX_HALF_LIFE = _LEARNED_INTERVAL_SECS / (-math.log2(TARGET_RECALL["production"]))

# MAX_SCHED_HALF_LIFE: the H clamp applied on every update — caps the scheduled
# review interval at ~90 days. Recognition recall stays high far past the model's
# prediction, and we only have ~1 month of recognition data, so we cap intervals
# here rather than extrapolate to year-long gaps. Because this clamp (≈280d) sits
# below MAX_HALF_LIFE (≈1133d), capped words never reach the retirement threshold
# — they keep getting reviewed ~quarterly instead of being retired. Raise the
# 90-day figure once longer-horizon recognition retention data exists.
_MAX_SCHED_INTERVAL_SECS = 90.0 * 86400.0  # 90 days
MAX_SCHED_HALF_LIFE = _MAX_SCHED_INTERVAL_SECS / (-math.log2(TARGET_RECALL["recognition"]))
MAX_ODDS = 5.0                      # clamp on p̂/(1-p̂); recognition-calibrated (gentle wrongs)

GRADE_EVIDENCE = GRADE_WEIGHT       # backward-compat alias

SOLID_HORIZON = 86400.0             # 24 hours in seconds
SOLID_RECALL = 0.85                 # p(recall) at 24h threshold
SOLID_HALF_LIFE = 368_500.0         # minimum H where recall@24h >= 0.85 (for migration)

SKILL_ORDER = ["recognition", "occlusion", "production"]
SIBLING_COOLDOWN_SECS = 21_600.0    # 6 hours

# ── skill state ────────────────────────────────────────────────────────

@dataclass
class SkillState:
    half_life_secs: float = INITIAL_HALF_LIFE
    last_reviewed_at: dt.datetime | None = None
    unlocked: bool = False
    graduated: bool = False


def is_active(ss: SkillState) -> bool:
    """A skill is active if it is unlocked and not yet graduated."""
    return ss.unlocked and not ss.graduated


# ── core model functions ───────────────────────────────────────────────

def effective_eta(
    H: float,
    eta_base: float = ETA_BASE,
    eta_boost: float = ETA_BOOST,
    h_scale: float = H_SCALE,
) -> float:
    """H-dependent learning rate: large at small H, decays to eta_base."""
    return eta_base + eta_boost / (1.0 + H / h_scale)


def recall_probability(H: float, delta_t: float) -> float:
    """p(recall) = 2^(-Δt / H)"""
    if H <= 0:
        return 0.0
    return 2.0 ** (-delta_t / H)


def update_half_life(
    H: float, delta_t: float, grade: int,
    skill: str = "recognition",
) -> float:
    """Bernoulli NLL gradient update with H-dependent learning rate.

    η_eff(H) = ETA_BASE + ETA_BOOST / (1 + H/H_SCALE)
    Large at small H (fast early learning), decays to ETA_BASE at large H.

    Correct (grades 1-3):
        δlogH = +η_eff · w[grade]

    Wrong (grades 4-5-6):
        δlogH = -η_eff · w[grade] · min(p̂/(1-p̂), MAX_ODDS)

    where:
        w = GRADE_WEIGHT[grade]            — fluency multiplier
        p̂ = 2^(-Δt/H)                     — predicted recall

    Correct always grows H. Wrong always shrinks H.
    The grade modulates magnitude, not direction.
    """
    eta = effective_eta(H)
    w = GRADE_WEIGHT.get(grade, 0.5)

    if grade in (1, 2, 3):
        delta_logH = eta * w
    else:
        p_hat = recall_probability(H, delta_t)
        odds = min(p_hat / (1.0 - p_hat + 1e-10), MAX_ODDS)
        delta_logH = -eta * w * odds

    new_H = H * math.exp(delta_logH)
    return max(MIN_HALF_LIFE, min(MAX_SCHED_HALF_LIFE, new_H))


def next_interval(H: float, skill: str) -> float:
    """Seconds until target recall: Δt = -H · log2(p*)"""
    p_star = TARGET_RECALL.get(skill, 0.85)
    if p_star <= 0 or p_star >= 1:
        return H
    return -H * math.log2(p_star)


def is_solid(H: float) -> bool:
    """A skill is solid if recall_probability(H, 24h) >= 0.85"""
    return recall_probability(H, SOLID_HORIZON) >= SOLID_RECALL


def check_unlocks(states: dict[str, SkillState]) -> list[str]:
    """Unlock next skill if previous is solid. Returns list of newly unlocked skill names.

    Also graduates the predecessor skill when the next one unlocks.

    H inheritance:
      - recognition → occlusion: reset to INITIAL_HALF_LIFE (big skill jump)
      - occlusion → production: inherit half of predecessor's H (similar skills)
    """
    newly_unlocked = []
    for i in range(1, len(SKILL_ORDER)):
        prev = SKILL_ORDER[i - 1]
        curr = SKILL_ORDER[i]
        if (prev in states and curr in states
                and states[prev].unlocked
                and is_solid(states[prev].half_life_secs)
                and not states[curr].unlocked):
            states[curr].unlocked = True
            if curr == "production":
                # occlusion → production: inherit half of predecessor H
                states[curr].half_life_secs = states[prev].half_life_secs / 2.0
            else:
                # recognition → occlusion: start from scratch
                states[curr].half_life_secs = INITIAL_HALF_LIFE
            states[curr].last_reviewed_at = None
            states[prev].graduated = True
            newly_unlocked.append(curr)
    return newly_unlocked


def ensure_graduations(states: dict[str, SkillState]) -> None:
    """Retroactively graduate predecessors where successor is already unlocked.

    Idempotent — safe to call on every load. Handles existing data where
    graduated was never set because the field didn't exist yet.
    """
    for i in range(len(SKILL_ORDER) - 1):
        curr = SKILL_ORDER[i]
        nxt = SKILL_ORDER[i + 1]
        if (curr in states and nxt in states
                and states[curr].unlocked
                and states[nxt].unlocked
                and not states[curr].graduated):
            states[curr].graduated = True


def _sibling_lock_until(
    skill_name: str,
    states: dict[str, SkillState],
) -> dt.datetime | None:
    """Return the time at which the sibling lock expires for *skill_name*,
    or None if no lock applies.

    A skill is sibling-locked when any OTHER unlocked skill was reviewed
    within the past SIBLING_COOLDOWN_SECS.
    """
    latest_other = None
    for other, ss in states.items():
        if other == skill_name:
            continue
        if not is_active(ss) or ss.last_reviewed_at is None:
            continue
        if latest_other is None or ss.last_reviewed_at > latest_other:
            latest_other = ss.last_reviewed_at
    if latest_other is None:
        return None
    lock_end = latest_other + dt.timedelta(seconds=SIBLING_COOLDOWN_SECS)
    return lock_end


def pick_review_skill(
    states: dict[str, SkillState],
    now: dt.datetime,
) -> str:
    """Return the single active (unlocked + not graduated) skill, respecting sibling locks."""
    best_skill = None
    best_recall = 2.0  # > 1.0 so any real value wins

    for skill_name in SKILL_ORDER:
        ss = states.get(skill_name)
        if ss is None or not is_active(ss):
            continue
        # skip sibling-locked skills
        lock_end = _sibling_lock_until(skill_name, states)
        if lock_end is not None and now < lock_end:
            continue
        if ss.last_reviewed_at is None:
            # never reviewed → recall is effectively 0
            return skill_name
        delta_t = (now - ss.last_reviewed_at).total_seconds()
        p = recall_probability(ss.half_life_secs, delta_t)
        if p < best_recall:
            best_recall = p
            best_skill = skill_name

    # fallback: if all skills are locked, pick the one whose lock expires soonest
    if best_skill is None:
        soonest_skill = None
        soonest_end = None
        for skill_name in SKILL_ORDER:
            ss = states.get(skill_name)
            if ss is None or not is_active(ss):
                continue
            lock_end = _sibling_lock_until(skill_name, states)
            if lock_end is not None and (soonest_end is None or lock_end < soonest_end):
                soonest_end = lock_end
                soonest_skill = skill_name
        best_skill = soonest_skill or "recognition"

    return best_skill


def compute_due_at(states: dict[str, SkillState], now: dt.datetime) -> dt.datetime:
    """Earliest due time across all unlocked skills, respecting sibling locks."""
    earliest = None
    for _skill_name, due, _locked in iter_skill_due_times(states, now):
        if earliest is None or due < earliest:
            earliest = due
    return earliest if earliest is not None else now


def iter_skill_due_times(
    states: dict[str, SkillState], now: dt.datetime,
) -> list[tuple[str, dt.datetime, bool]]:
    """Return [(skill_name, due_at, sibling_locked), ...] for each unlocked skill.

    sibling_locked is True when the skill's raw due_at has passed but
    the sibling cooldown is pushing it into the future.
    """
    result = []
    for skill_name in SKILL_ORDER:
        ss = states.get(skill_name)
        if ss is None or not is_active(ss):
            continue
        interval = next_interval(ss.half_life_secs, skill_name)
        ref = ss.last_reviewed_at if ss.last_reviewed_at is not None else now
        raw_due = ref + dt.timedelta(seconds=interval)
        lock_end = _sibling_lock_until(skill_name, states)
        if lock_end is not None and lock_end > raw_due:
            due = lock_end
            locked = raw_due <= now < lock_end
        else:
            due = raw_due
            locked = False
        result.append((skill_name, due, locked))
    return result


# ── serialization helpers ──────────────────────────────────────────────

def skill_states_to_dict(states: dict[str, SkillState]) -> dict:
    """Serialize skill states for SRSState.state JSON."""
    result = {}
    for name, ss in states.items():
        result[name] = {
            "half_life_secs": ss.half_life_secs,
            "last_reviewed_at": ss.last_reviewed_at.isoformat() if ss.last_reviewed_at else None,
            "unlocked": ss.unlocked,
            "graduated": ss.graduated,
        }
    return result


def skill_states_from_dict(data: dict) -> dict[str, SkillState]:
    """Deserialize skill states from SRSState.state JSON."""
    states = {}
    for name in SKILL_ORDER:
        entry = data.get(name, {})
        lr_raw = entry.get("last_reviewed_at")
        lr = None
        if lr_raw is not None:
            lr = dt.datetime.fromisoformat(lr_raw)
            if lr.tzinfo is None:
                lr = lr.replace(tzinfo=dt.timezone.utc)
        states[name] = SkillState(
            half_life_secs=entry.get("half_life_secs", INITIAL_HALF_LIFE),
            last_reviewed_at=lr,
            unlocked=entry.get("unlocked", False),
            graduated=entry.get("graduated", False),
        )
    return states


def make_initial_skill_states() -> dict[str, SkillState]:
    """Create default skill states: R unlocked, O/P locked, all at initial H."""
    return {
        "recognition": SkillState(unlocked=True),
        "occlusion":   SkillState(unlocked=False),
        "production":  SkillState(unlocked=False),
    }
