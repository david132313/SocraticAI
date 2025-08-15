"""
Rule-based learner state detection for Adaptive Socratic Prompts.
States: confident, hesitant, guessing, silent (with universal fallback).
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import re
import statistics

# ----------------------------
# Public API
# ----------------------------

def detect_state(recent_turns: List[Dict[str, Any]],
                 *,
                 latency_slow: float = 25.0,
                 min_history: int = 1,
                 hedge_terms: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Classify learner state from the last ~3–5 turns.
    Inputs (per turn dict; missing fields are tolerated):
      - 'student_response': str (may be empty)
      - 'response_latency_sec': float
      - 'understanding': float in [0,1]
      - 'clarification_count': int (per concept, cumulative or turn-local)
    Returns:
      { 'state': 'confident'|'hesitant'|'guessing'|'silent',
        'confidence': float in [0,1],
        'signals': {summary metrics and rule flags} }
    """
    if hedge_terms is None:
        hedge_terms = ["maybe", "i think", "i guess", "not sure", "idk", "i'm not sure"]

    window = recent_turns[-5:] if recent_turns else []
    if len(window) < min_history:
        # With no history, prefer a gentle start (silent → universal fallbacks later)
        return {"state": "silent", "confidence": 0.4, "signals": {"reason": "insufficient_history"}}

    # --------- Aggregate signals ----------
    texts = [str(t.get("student_response", "") or "").strip() for t in window]
    latencies = [float(t.get("response_latency_sec", 0.0) or 0.0) for t in window if t.get("response_latency_sec") is not None]
    understands = [clamp01(t.get("understanding", 0.0)) for t in window if t.get("understanding") is not None]
    clarifs = [int(t.get("clarification_count", 0) or 0) for t in window]

    empty_count = sum(1 for x in texts if len(x) == 0)
    very_short_count = sum(1 for x in texts if 0 < len(x) < 15)
    hedge_count = sum(count_hedges(x, hedge_terms) for x in texts)

    avg_under = mean_safe(understands, default=0.0)
    std_under = stdev_safe(understands, default=0.0)
    avg_latency = mean_safe(latencies, default=0.0)
    total_clarifs = sum(clarifs)

    # --------- Rule checks (boolean signals) ----------
    rules_met: List[tuple[str, bool]] = []

    # SILENT: empty / explicit "I don't know" pattern dominates
    silent_like = (empty_count >= max(1, len(window)//2)) or any(is_idk(x) for x in texts)
    rules_met.append(("silent_like", silent_like))

    # CONFIDENT: good understanding, consistent, normal latency, few clarifs, few hedges
    confident_like = (
        avg_under >= 0.70 and
        std_under <= 0.15 and
        avg_latency < latency_slow and
        total_clarifs <= len(window) and
        hedge_count <= 1 and
        very_short_count == 0
    )
    rules_met.append(("confident_like", confident_like))

    # HESITANT: longer latency OR repeated clarifications or partials (very short answers)
    hesitant_like = (
        (avg_latency >= latency_slow) or
        (total_clarifs >= max(2, len(window)//2)) or
        (very_short_count >= max(1, len(window)//3))
    )
    rules_met.append(("hesitant_like", hesitant_like))

    # GUESSING: hedge language + mid understanding band
    guessing_like = (
        (hedge_count >= 1) and
        (0.30 <= avg_under <= 0.60)
    )
    rules_met.append(("guessing_like", guessing_like))

    # --------- Choose label with simple precedence ----------
    # Priority: silent > confident > hesitant > guessing
    if silent_like:
        detected = "silent"
    elif confident_like:
        detected = "confident"
    elif hesitant_like:
        detected = "hesitant"
    elif guessing_like:
        detected = "guessing"
    else:
        # fallback when no strong signal → treat as hesitant scaffolding start
        detected = "hesitant"

    # Confidence as fraction of rules supporting the chosen label
    support = sum(1 for name, ok in rules_met if ok and name.startswith(detected[:3]))
    total_true = sum(1 for _, ok in rules_met if ok)
    confidence = round((support / max(1, total_true)), 2) if total_true > 0 else 0.5

    signals = {
        "avg_understanding": round(avg_under, 3),
        "std_understanding": round(std_under, 3),
        "avg_latency_sec": round(avg_latency, 2),
        "total_clarifications": total_clarifs,
        "hedge_count": hedge_count,
        "empty_count": empty_count,
        "very_short_count": very_short_count,
        "rules": dict(rules_met),
    }
    return {"state": detected, "confidence": float(confidence), "signals": signals}


def transition_policy(prev_state: Optional[str], detected_state: str) -> str:
    """
    Smooths state evolution following scaffolding flow:
      Hesitant → Confident
      Guessing → Hesitant/Confident (prefer Hesitant unless confidence later improves)
      Silent → Universal → escalate (map to Hesitant next if activity resumes)
    """
    p = (prev_state or "").lower()
    d = (detected_state or "").lower()

    if p == "":
        return d

    if p == "silent":
        # After silence, use a gentle on-ramp unless the learner jumps to confident
        return "hesitant" if d in {"hesitant", "guessing"} else d

    if p == "guessing":
        # Step up through Hesitant before Confident to avoid abrupt jumps
        if d == "confident":
            return "hesitant"
        # Allow move to hesitant or stay guessing/silent
        return d

    if p == "hesitant":
        # Do not step down to guessing immediately
        if d == "guessing":
            return "hesitant"
        return d

    if p == "confident":
        # Soften large drops
        if d == "guessing":
            return "hesitant"
        return d

    # default
    return d


# ----------------------------
# Helpers (module-internal)
# ----------------------------

def clamp01(x: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, v))

def mean_safe(xs, default=0.0):
    xs = [float(x) for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else default

def stdev_safe(xs, default=0.0):
    xs = [float(x) for x in xs if x is not None]
    if len(xs) < 2:
        return default
    try:
        return float(statistics.pstdev(xs))
    except Exception:
        return default

def count_hedges(text: str, hedge_terms: List[str]) -> int:
    t = (text or "").lower()
    n = 0
    for h in hedge_terms:
        if re.search(rf"\b{re.escape(h)}\b", t):
            n += 1
    return n

def is_idk(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"", "idk", "i don't know", "dont know", "don't know"}

__all__ = ["detect_state", "transition_policy"]
