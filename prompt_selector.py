"""
Prompt selection and rotation for Adaptive Socratic Prompts.
Chooses a prompt type within a learner state, avoids immediate repeats,
and provides a seed instruction string for the generator.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import random

from adaptive_prompts import PROMPTS, get_prompts_for_state, fill_placeholders

# Friendly labels for generator seeding / UI tags
MOVE_LABELS: Dict[str, str] = {
    # confident
    "counterexample": "Counterexample Challenge",
    "assumption": "Assumption Probe",
    "dual_path": "Dual-Path Reasoning",
    # hesitant
    "clarify_terms": "Clarify Terms",
    "chunk_build": "Chunk & Build",
    "option_compare": "Option Comparison",
    # guessing
    "justify": "Justify Before Reveal",
    "confidence_rate": "Confidence Rating",
    "link_known": "Link to Known",
    # silent
    "micro_prompt": "Micro-Prompt",
    "this_or_that": "This-or-That",
    "real_life": "Real-Life Hook",
    # universal
    "prediction": "Prediction",
    "evidence": "Evidence",
    "analogy": "Analogy",
    "reflection": "Reflection",
    "teach_back": "Teach-Back",
}


def list_prompt_types(state: str) -> List[str]:
    """List available prompt types for a given state (or universal fallback)."""
    return list(get_prompts_for_state(state).keys())


def _recent_types_from_history(history: List[Dict[str, Any]], state: Optional[str], k: int = 5) -> List[str]:
    """
    Extract last-used adaptive prompt types (most recent first) for the same state from history.
    History items may contain 'adaptive_prompt': {'state','prompt_type',...}.
    """
    recent: List[str] = []
    if not history:
        return recent
    state_norm = (state or "").strip().lower()
    for turn in reversed(history[-k:]):
        ap = turn.get("adaptive_prompt") or {}
        if not isinstance(ap, dict) or not ap:
            continue
        if state_norm and (ap.get("state", "").strip().lower() != state_norm):
            continue
        pt = ap.get("prompt_type")
        if isinstance(pt, str) and pt and pt not in recent:
            recent.append(pt)
    return recent


def _choose_type_with_rotation(
    all_types: List[str],
    recent_types: List[str],
    *,
    avoid_repeat: bool = True,
    last_choice: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Rotation policy:
      1) Prefer types NOT seen in recent_types.
      2) If all seen, prefer those least-recently seen (i.e., not at index 0).
      3) Avoid immediate repeat of last_choice when possible.
      4) Choose randomly among candidates.
    """
    rng = rng or random
    if not all_types:
        return ""

    # 1) Prefer unseen types
    unseen = [t for t in all_types if t not in recent_types]
    candidates = unseen if unseen else list(all_types)

    # 2) If all seen, demote the most recent (recent_types[0]) to last
    if not unseen and recent_types:
        most_recent = recent_types[0]
        candidates = [t for t in all_types if t != most_recent] or [most_recent]

    # 3) Avoid immediate repeat of last_choice, if possible
    if avoid_repeat and last_choice in candidates and len(candidates) > 1:
        candidates = [t for t in candidates if t != last_choice] or candidates

    # 4) Random choice (stable if rng seeded by caller)
    return rng.choice(candidates)


def _infer_last_choice(history: List[Dict[str, Any]], state: Optional[str]) -> Optional[str]:
    """Infer last prompt_type used for this state from history (most recent first)."""
    state_norm = (state or "").strip().lower()
    for turn in reversed(history or []):
        ap = turn.get("adaptive_prompt") or {}
        if not isinstance(ap, dict):
            continue
        if state_norm and (ap.get("state", "").strip().lower() != state_norm):
            continue
        t = ap.get("prompt_type")
        if isinstance(t, str) and t:
            return t
    return None


def select_prompt(
    state: str,
    history: List[Dict[str, Any]],
    bloom_level: Optional[str] = None,
    *,
    avoid_repeat: bool = True,
    last_choice: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Pick a prompt template for a learner state with light rotation.
    Args:
      state: detected learner state string
      history: list of prior turn dicts (may include 'adaptive_prompt')
      bloom_level: optional Bloom level name (e.g., 'understand', 'apply')
      avoid_repeat: avoid choosing the last prompt type again if alternatives exist
      last_choice: explicit last prompt type to avoid (optional; otherwise inferred from history)
      rng: optional Random instance for deterministic tests
    Returns:
      {
        "state": str,
        "prompt_type": str,
        "template": str,
        "meta": {
           "label": str,
           "bloom_level": Optional[str],
           "fallback_used": bool,
           "candidates": List[str],
           "recent_types": List[str]
        }
      }
    """
    rng = rng or random

    # Normalize state and pull templates (falls back to 'universal' internally)
    state_norm = (state or "").strip().lower()
    mapping = get_prompts_for_state(state_norm)
    all_types = list(mapping.keys())
    fallback_used = state_norm not in PROMPTS

    # Rotation inputs
    recents = _recent_types_from_history(history, state_norm, k=5)
    last = last_choice if last_choice else _infer_last_choice(history, state_norm)

    chosen_type = _choose_type_with_rotation(
        all_types,
        recents,
        avoid_repeat=avoid_repeat,
        last_choice=last,
        rng=rng,
    )

    # Choose a template for the chosen type
    templates = mapping.get(chosen_type, []) or []
    if not templates and all_types:
        # Extreme edge: mapping exists but chosen_type missing/empty; fallback to first type's first template
        chosen_type = all_types[0]
        templates = mapping.get(chosen_type, []) or []

    template = rng.choice(templates) if templates else ""

    return {
        "state": state_norm or "universal",
        "prompt_type": chosen_type,
        "template": template,
        "meta": {
            "label": MOVE_LABELS.get(chosen_type, "Socratic Move"),
            "bloom_level": bloom_level,
            "fallback_used": bool(fallback_used),
            "candidates": all_types,
            "recent_types": recents,
        },
    }


def build_seed_instruction(selection: Dict[str, Any], *, concept: Optional[str] = None) -> str:
    """
    Make a short instruction string for the generator based on a selection dict.
      selection: output of select_prompt()
      concept: optional concept name to fill placeholders (best-effort)
    """
    state = selection.get("state", "universal")
    ptype = selection.get("prompt_type", "")
    template = selection.get("template", "")
    label = selection.get("meta", {}).get("label", "Socratic Move")

    # Fill placeholders with very safe defaults if concept is known
    filled = fill_placeholders(template, concept=(concept or "this idea"))

    return (
        f"Use the following Socratic move ({label}, state={state}). "
        f"Phrase a follow-up that reflects: \"{filled}\". "
        f"Stay concise, and do not reveal the final answer."
    )


__all__ = ["select_prompt", "build_seed_instruction", "list_prompt_types"]
