from typing import Dict, List

"""
Canonical Adaptive Socratic Prompt Set.
Organized by learner state and prompt type. Pure data + tiny helpers.
No side effects, no external deps. Safe to import anywhere.
"""

# Quick references
CANONICAL_STATES: List[str] = ["confident", "hesitant", "guessing", "silent", "universal"]
UNIVERSAL_TYPES: List[str] = ["prediction", "evidence", "analogy", "reflection", "teach_back"]

# Canonical templates
PROMPTS: Dict[str, Dict[str, List[str]]] = {
    "confident": {
        "counterexample": [
            "You seem sure—can you think of a case where {concept} wouldn’t hold?",
        ],
        "assumption": [
            "What assumption is your answer based on? What if that assumption changes?",
        ],
        "dual_path": [
            "Is there another way to reach the same conclusion? Which feels more reliable, and why?",
        ],
    },
    "hesitant": {
        "clarify_terms": [
            "Which part is unclear? Let’s define {term} first.",
        ],
        "chunk_build": [
            "Let’s do the first half only—what’s the very first check you’d make?",
        ],
        "option_compare": [
            "Would you start with {A} or {B}? Why?",
        ],
    },
    "guessing": {
        "justify": [
            "Explain how you got that—what were the steps?",
        ],
        "confidence_rate": [
            "Rate your confidence 1–5. Why not higher or lower?",
        ],
        "link_known": [
            "What does this remind you of that you already know?",
        ],
    },
    "silent": {
        "micro_prompt": [
            "If you had to guess, what word would come first?",
        ],
        "this_or_that": [
            "Is it more like {X} or {Y}? Why?",
        ],
        "real_life": [
            "Where might you see this in real life?",
        ],
    },
    "universal": {
        "prediction": ["Before checking, what do you predict will happen? Why?"],
        "evidence": ["What evidence supports your answer?"],
        "analogy": ["Can you make a simple analogy for this idea?"],
        "reflection": ["What was hardest here, and what would you try next time?"],
        "teach_back": ["Explain it as if teaching a friend in one minute."],
    },
}


def get_prompts_for_state(state: str) -> Dict[str, List[str]]:
    """
    Return a mapping of prompt_type -> templates[] for the given state.
    Falls back to 'universal' if state not recognized.
    """
    s = (state or "").strip().lower()
    return PROMPTS.get(s, PROMPTS["universal"])


def fill_placeholders(
    template: str,
    *,
    concept: str = "this idea",
    term: str = "the key term",
    A: str = "Approach A",
    B: str = "Approach B",
    X: str = "Option X",
    Y: str = "Option Y",
) -> str:
    """
    Minimal, safe placeholder substitution for templates.
    Keeps defaults so callers can pass only what they know.
    """
    try:
        return template.format(concept=concept, term=term, A=A, B=B, X=X, Y=Y)
    except Exception:
        # In case a template has no placeholders or mismatched keys
        return template


__all__ = [
    "PROMPTS",
    "CANONICAL_STATES",
    "UNIVERSAL_TYPES",
    "get_prompts_for_state",
    "fill_placeholders",
]
