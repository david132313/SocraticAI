import json
import logging
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from tenacity import retry, wait_exponential, stop_after_attempt

from extractors import _get_openai_client, OPENAI_MODEL_CHAT
from spaced_repetition import create_sm2_scheduler


class GraspPredictor:
    """
    Tracks and predicts student grasp levels for individual concepts.
    Grasp scores range from 0.0 (no understanding) to 1.0 (complete mastery).
    """

    def __init__(self):
        self.concept_data: Dict[str, Dict[str, Any]] = {}
        self.sm2_scheduler = create_sm2_scheduler()
        self.client = _get_openai_client()
        self.model = OPENAI_MODEL_CHAT
        self.logger = logging.getLogger(__name__)
        self._ref_cache: Dict[tuple, str] = {}  # (concept, question, context[:256]) -> reference answer

        # Confidence tuning
        self.success_threshold = 0.60  # performance >= threshold counts as success
        self.confidence_z = 1.96       # ~95% Wilson interval; use 1.64 for ~90%

    # ---------- Core ----------

    def initialize_concept(self, concept_name: str, difficulty: float) -> None:
        """Initialize tracking for a new concept."""
        self.concept_data[concept_name] = {
            "grasp_score": 0.0,   # Start with no knowledge
            "confidence": 0.0,    # Low confidence initially
            "difficulty": float(difficulty),
            "interactions": [],   # List of {timestamp, performance, quality, ...}
            "last_seen": None,    # ISO timestamp (UTC)
            "created": datetime.now(timezone.utc).isoformat(),
        }
        # Initialize SM-2 card for this concept
        self.sm2_scheduler.initialize_concept_card(concept_name, float(difficulty))

    def get_grasp_score(self, concept_name: str) -> float:
        """Get current grasp prediction for a concept."""
        return float(self.concept_data.get(concept_name, {}).get("grasp_score", 0.0))

    def get_confidence(self, concept_name: str) -> float:
        """Get confidence level in the grasp prediction."""
        return float(self.concept_data.get(concept_name, {}).get("confidence", 0.0))

    def update_grasp(
        self,
        concept_name: str,
        performance_score: float,
        response_quality: str = "medium",
    ) -> None:
        """
        Update grasp prediction based on student interaction.

        Args:
            concept_name: Name of the concept
            performance_score: 0.0–1.0 score representing correctness/understanding
            response_quality: "low" | "medium" | "high"
        """
        if concept_name not in self.concept_data:
            # Initialize with default difficulty if unseen
            self.initialize_concept(concept_name, 3.0)

        concept = self.concept_data[concept_name]

        # Clamp performance and normalize quality
        perf = max(0.0, min(1.0, float(performance_score)))
        rq = str(response_quality).lower()
        if rq not in {"low", "medium", "high"}:
            rq = "medium"

        # Record interaction
        interaction = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": perf,
            "quality": rq,
        }
        concept["interactions"].append(interaction)
        concept["last_seen"] = interaction["timestamp"]

        # Recompute grasp & confidence
        concept["grasp_score"] = self._calculate_weighted_grasp(concept)
        concept["confidence"] = self._calculate_confidence(concept)

        # Update SM-2 schedule (use UTC review time)
        sm2_result = self.sm2_scheduler.review_concept(
            concept_name, perf, rq, review_datetime=datetime.now(timezone.utc)
        )
        interaction["sm2_next_due"] = sm2_result.get("next_due")
        interaction["sm2_rating"] = sm2_result.get("rating_name")

    # ---------- LLM helpers: reference & grading (robust) ----------
#David Zhu
    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def get_reference_answer(
        self,
        self_concept: str = None,  # kept for backward compatibility if called positionally
        *,
        concept: Optional[str] = None,
        question: str = "",
        context_text: str = "",
        **kwargs,
    ) -> str:
        """
        Generate a high-quality model answer to compare against a student response.
        Accepts either `context_text` or legacy `context` via kwargs.

        Notes:
        - Uses a small in-memory cache keyed by (concept, question, context[:256]).
        """
        # Positional compat: if first arg used positionally, interpret as concept
        if concept is None and isinstance(self_concept, str):
            concept = self_concept

        try:
            if not context_text:
                context_text = kwargs.get("context", "") or ""

            if not question:
                question = f"Explain the core idea of {concept} clearly and concisely."

            cache_key = (str(concept), question.strip(), context_text.strip()[:256])
            if cache_key in self._ref_cache:
                return self._ref_cache[cache_key]

            prompt = (
                "You are a subject-matter expert writing a model answer.\n\n"
                f"CONCEPT: {concept}\n"
                f"QUESTION: {question}\n"
                + (f"CONTEXT:\n{context_text}\n" if context_text else "")
                + "\nWrite a clear, correct model answer in 3–6 sentences. "
                  "If CONTEXT is provided, rely on it and avoid adding outside facts. "
                  "Do not include preambles or markdown fences. "
                  "Return only the answer text."
            )

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=350,
            )
            answer = (resp.choices[0].message.content or "").strip()
            self._ref_cache[cache_key] = answer
            return answer

        except Exception as e:
            self.logger.error(f"Reference answer generation failed: {e}")
            return "Unable to generate reference answer."
#David Zhu
    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def rate_student_answer(
        self,
        *,
        question: str,
        student_response: Optional[str] = None,
        concept: str,
        context_text: str = "",
        reference_answer: Optional[str] = None,
        current_grasp: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Rate a student's answer with detailed analysis.

        Flexible args:
          - `student_response` (preferred) OR legacy `student_answer` via kwargs.
          - `context_text` OR legacy `context` via kwargs.
          - If `reference_answer` is missing, it will be generated.
        """
        try:
            # Compat shims
            if student_response is None:
                student_response = kwargs.get("student_answer", "") or ""
            if not context_text:
                context_text = kwargs.get("context", "") or ""

            if reference_answer is None:
                reference_answer = self.get_reference_answer(
                    concept=concept,
                    question=question,
                    context_text=context_text,
                )

            if current_grasp is None:
                try:
                    current_grasp = float(self.get_concept_summary(concept)["grasp_score"])
                except Exception:
                    current_grasp = 0.5

            prompt = f"""
You are an experienced tutor evaluating student responses.

CONCEPT: {concept}
CURRENT GRASP LEVEL: {current_grasp:.2f} (0.0 = beginner, 1.0 = expert)
QUESTION: {question}

REFERENCE ANSWER:
{reference_answer}

STUDENT ANSWER:
{student_response}

Evaluate accuracy, completeness, and reasoning quality.

Grading rubric for understanding_level (0.0–1.0):
- 0.0–0.2: irrelevant, incorrect, or extremely brief (e.g., < 15 chars).
- 0.2–0.4: mentions 1 minor idea but misses core points.
- 0.4–0.6: covers some core points, incomplete or shallow.
- 0.6–0.8: mostly correct and covers most core points, minor gaps.
- 0.8–1.0: correct, complete, clear explanation.

Return a JSON object only (no prose, no backticks) with these keys:
{{
  "understanding_level": <number 0..1>,
  "response_quality": "<low|medium|high>",
  "key_points_covered": ["<short phrase>", "..."],
  "missing_elements": ["<short phrase>", "..."],
  "suggested_follow_up": "<one short question to push thinking>",
  "feedback": "<one-line constructive feedback>",
  "grasp_adjustment": <number -0.2..0.2>
}}
""".strip()

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )

            raw = (resp.choices[0].message.content or "").strip()

            # Strip code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()

            # Parse JSON (rescue embedded JSON if needed)
            try:
                data = json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                if not m:
                    raise TypeError("LLM did not return JSON.")
                data = json.loads(m.group(0))

            # ------- Guardrails & normalization -------
            # Score clamped
            score = float(data.get("understanding_level", 0.5))
            score = max(0.0, min(1.0, score))

            # Very short/empty answers can't score high
            if len((student_response or "").strip()) < 15:
                score = min(score, 0.2)

            # Lexical overlap with reference (cheap proxy)
            def _overlap(a: str, b: str) -> float:
                A = set(re.findall(r"\b[a-z]{4,}\b", (a or "").lower()))
                B = set(re.findall(r"\b[a-z]{4,}\b", (b or "").lower()))
                return 0.0 if not A else len(A & B) / len(A)

            # Refine low-overlap guard: apply only when the reference is substantive AND the student response is short
            ref_words = len((reference_answer or "").split())
            is_short = len((student_response or "").strip()) < 60
            ov = _overlap(reference_answer, student_response)
            if ref_words >= 12 and is_short and ov < 0.10:
                score = min(score, 0.3)

            # Normalize response_quality
            rq = str(data.get("response_quality", "medium")).lower()
            if rq not in {"low", "medium", "high"}:
                rq = "medium"

            # Clamp external adjustment and blend with a small internal adjustment
            model_adj = max(-0.2, min(0.2, float(data.get("grasp_adjustment", 0.0))))
            delta = score - float(current_grasp or 0.5)
            internal_adj = max(-0.15, min(0.15, 0.6 * delta))
            blended_adj = round(0.5 * model_adj + 0.5 * internal_adj, 3)

            # Put back sanitized score and defaults
            data_out: Dict[str, Any] = {
                "understanding_level": score,
                "response_quality": rq,
                "key_points_covered": list(data.get("key_points_covered", [])),
                "missing_elements": list(data.get("missing_elements", [])),
                "suggested_follow_up": str(data.get("suggested_follow_up", "")),
                "feedback": str(data.get("feedback", "")),
                "grasp_adjustment": blended_adj,
                "reference_answer": reference_answer,
            }
            return data_out

        except Exception as e:
            self.logger.error(f"Student answer rating failed: {e}")
            return {
                "understanding_level": 0.5,
                "response_quality": "medium",
                "key_points_covered": [],
                "missing_elements": ["Analysis unavailable"],
                "suggested_follow_up": "Could you elaborate on your answer?",
                "feedback": "Please try to be more specific.",
                "grasp_adjustment": 0.0,
                "reference_answer": reference_answer or "",
            }

    # ---------- Time/decay & summaries ----------

    def apply_temporal_decay(self, concept_name: str) -> None:
        """Apply forgetting-curve decay based on time since last interaction."""
        concept = self.concept_data.get(concept_name)
        if not concept or not concept.get("last_seen"):
            return

        # last_seen stored as ISO UTC
        last_seen = datetime.fromisoformat(concept["last_seen"])
        days_elapsed = (datetime.now(timezone.utc) - last_seen).total_seconds() / 86400.0
        if days_elapsed < 1.0:
            return  # No decay for very recent interactions

        current_grasp = float(concept["grasp_score"])
        decay_rate = self._calculate_decay_rate(concept)

        decayed_grasp = current_grasp * math.exp(-decay_rate * days_elapsed)
        concept["grasp_score"] = round(max(0.0, decayed_grasp), 3)

    def _calculate_decay_rate(self, concept: Dict[str, Any]) -> float:
        """Calculate concept-specific decay rate based on difficulty and mastery."""
        difficulty = float(concept.get("difficulty", 3.0))
        current_grasp = float(concept.get("grasp_score", 0.0))

        # Base decay rate (higher difficulty = faster forgetting)
        base_rate = 0.1 + (difficulty - 1.0) * 0.05  # ~0.1 to ~0.3 for difficulty 1..5

        # Well-learned concepts decay slower
        mastery_factor = 1.0 - (current_grasp * 0.5)  # 0.5..1.0 as grasp increases 0..1

        return float(base_rate * mastery_factor)

    def _calculate_weighted_grasp(self, concept: Dict[str, Any]) -> float:
        """Calculate grasp score using weighted recent performance."""
        interactions = concept.get("interactions", [])
        if not interactions:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        # Weight recent interactions more heavily (exponential)
        for i, interaction in enumerate(interactions):
            weight = 0.5 ** (len(interactions) - i - 1)  # newer → higher weight
            performance = float(interaction.get("performance", 0.0))

            # Adjust for response quality
            quality_multiplier = {"low": 0.8, "medium": 1.0, "high": 1.2}
            adjusted = performance * quality_multiplier.get(str(interaction.get("quality", "medium")), 1.0)

            weighted_sum += adjusted * weight
            total_weight += weight

        grasp = min(1.0, weighted_sum / max(total_weight, 1e-9))
        return round(float(grasp), 3)

    def _calculate_confidence(self, concept: Dict[str, Any]) -> float:
        """
        Wilson score *lower bound* as a conservative confidence estimate.
        Treat each interaction as "success" if performance >= self.success_threshold.
        """
        interactions = concept.get("interactions", [])
        n = len(interactions)
        if n == 0:
            return 0.0

        # Count successes
        s = 0
        for it in interactions:
            try:
                perf = float(it.get("performance", 0.0))
            except Exception:
                perf = 0.0
            if perf >= self.success_threshold:
                s += 1

        phat = s / n
        z = float(getattr(self, "confidence_z", 1.96))

        # Wilson lower bound:
        denom = 1.0 + (z * z) / n
        centre = phat + (z * z) / (2.0 * n)
        margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)
        lower = (centre - margin) / denom

        lower = max(0.0, min(1.0, lower))
        return round(float(lower), 3)

    def update_all_decay(self) -> None:
        """Apply temporal decay to all concepts."""
        for concept_name in list(self.concept_data.keys()):
            self.apply_temporal_decay(concept_name)

    def get_concept_summary(self, concept_name: str) -> Dict[str, Any]:
        """Get comprehensive summary of concept state."""
        if concept_name not in self.concept_data:
            return {"error": "Concept not found"}

        concept = self.concept_data[concept_name]
        return {
            "concept": concept_name,
            "grasp_score": float(concept.get("grasp_score", 0.0)),
            "confidence": float(concept.get("confidence", 0.0)),
            "difficulty": float(concept.get("difficulty", 3.0)),
            "total_interactions": len(concept.get("interactions", [])),
            "last_seen": concept.get("last_seen"),
            "status": self._get_mastery_status(float(concept.get("grasp_score", 0.0))),
        }

    def _get_mastery_status(self, grasp_score: float) -> str:
        """Convert grasp score to readable status."""
        if grasp_score < 0.3:
            return "Needs Review"
        elif grasp_score < 0.6:
            return "Learning"
        elif grasp_score < 0.8:
            return "Good Understanding"
        else:
            return "Mastered"

    def save_to_file(self, filepath: str) -> None:
        """Save grasp data to JSON file atomically, plus SM-2 data next to it."""
        try:
            # Ensure directory exists
            dir_ = os.path.dirname(filepath) or "."
            os.makedirs(dir_, exist_ok=True)

            # Atomic write for concept_data
            with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_, suffix=".tmp", encoding="utf-8") as tf:
                json.dump(self.concept_data, tf, indent=2, ensure_ascii=False)
                tmpname = tf.name
            os.replace(tmpname, filepath)

            # Save SM-2 data as before (best-effort)
            sm2_filepath = filepath.replace(".json", "_sm2.json")
            self.sm2_scheduler.save_to_file(sm2_filepath)
        except Exception as e:
            self.logger.error(f"Error saving grasp data: {e}")

    def load_from_file(self, filepath: str) -> None:
        """Load grasp data from JSON file + SM-2 data if present."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self.concept_data = json.load(f)

            sm2_filepath = filepath.replace(".json", "_sm2.json")
            self.sm2_scheduler.load_from_file(sm2_filepath)
        except FileNotFoundError:
            self.logger.warning(f"Grasp data file not found: {filepath}")
            self.concept_data = {}
        except Exception as e:
            self.logger.error(f"Error loading grasp data: {e}")
            self.concept_data = {}

    def initialize_from_concepts(self, concepts: List[Dict[str, Any]]) -> None:
        """Initialize tracking for a list of extracted concepts."""
        for c in concepts:
            name = c.get("concept", "")
            difficulty = float(c.get("difficulty", 3.0))
            if name and name not in self.concept_data:
                self.initialize_concept(name, difficulty)

    def get_all_concepts_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all tracked concepts (after applying decay)."""
        self.update_all_decay()
        summaries = [self.get_concept_summary(name) for name in self.concept_data.keys()]
        summaries.sort(key=lambda x: x["grasp_score"])
        return summaries

    # Legacy grasp-based “needs review”; prefer SM-2 due list.
    def get_concepts_needing_review(self, threshold: float = 0.5) -> List[str]:
        self.update_all_decay()
        return [name for name, data in self.concept_data.items() if float(data.get("grasp_score", 0.0)) < threshold]

    def get_concepts_due_for_review(self) -> List[Dict[str, Any]]:
        """Get concepts due via SM-2 scheduling, enriched with grasp/confidence/difficulty."""
        due_concepts = self.sm2_scheduler.get_due_concepts()
        enhanced: List[Dict[str, Any]] = []
        for info in due_concepts:
            name = info.get("concept")
            gsum = self.get_concept_summary(name)
            difficulty = float(self.concept_data.get(name, {}).get("difficulty", 3.0))
            enhanced.append(
                {
                    **info,
                    "grasp_score": gsum["grasp_score"],
                    "confidence": gsum["confidence"],
                    "status": gsum["status"],
                    "total_interactions": gsum["total_interactions"],
                    "difficulty": difficulty,
                }
            )
        return enhanced


# Convenience factory
def create_grasp_predictor(concepts: Optional[List[Dict[str, Any]]] = None) -> GraspPredictor:
    predictor = GraspPredictor()
    if concepts:
        predictor.initialize_from_concepts(concepts)
    return predictor
