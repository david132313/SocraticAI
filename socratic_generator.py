# socratic_generator.py
import re
import json
from typing import Dict, List, Optional, Any
from tenacity import retry, wait_exponential, stop_after_attempt
from datetime import datetime
import uuid

from extractors import _get_openai_client, OPENAI_MODEL_CHAT


class SocraticQuestionGenerator:
    """
    Generates adaptive Socratic questions based on student grasp levels and Bloom's taxonomy.
    """

    def __init__(self):
        self.client = _get_openai_client()
        self.model = OPENAI_MODEL_CHAT
        # Bloom's taxonomy bands (inclusive of lower bound, exclusive of upper)
        self.bloom_levels = {
            "remember":   {"range": (0.0, 0.3),  "verbs": ["define", "identify", "recall", "list"]},
            "understand": {"range": (0.3, 0.5),  "verbs": ["explain", "describe", "summarize", "interpret"]},
            "apply":      {"range": (0.5, 0.7),  "verbs": ["demonstrate", "solve", "use", "implement"]},
            "analyze":    {"range": (0.7, 0.85), "verbs": ["compare", "examine", "categorize", "differentiate"]},
            "evaluate":   {"range": (0.85, 0.95),"verbs": ["assess", "critique", "judge", "validate"]},
            "create":     {"range": (0.95, 1.01),"verbs": ["design", "construct", "develop", "formulate"]},
        }

    # ---------------- Hints helpers ----------------

    def extract_key_points(self, reference_answer: str, k: int = 5) -> List[str]:
        prompt = (
            "Extract key points from the following reference answer.\n"
            "Return a JSON array of strings only, no prose.\n\n"
            f"REFERENCE ANSWER:\n{reference_answer}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
            )
            text = (resp.choices[0].message.content or "").strip()
            # tolerate code-fences
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
            points = json.loads(text)
            points = [p.strip("- • ").strip() for p in points if isinstance(p, str)]
            return points[:k]
        except Exception:
            # Fallback: first k non-trivial sentences
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reference_answer) if len(s.strip()) > 20]
            return sents[:k]

    def missing_points(self, ref_points: List[str], student_answer: str) -> List[str]:
        """Which reference points are missing or incorrect in the student's answer?"""
        try:
            prompt = (
                "Given the reference key points and a student's answer, list which points are missing or incorrect. "
                "Return JSON array of strings (each a missing/incorrect point). No prose.\n\n"
                f"REFERENCE POINTS:\n{json.dumps(ref_points)}\n\n"
                f"STUDENT ANSWER:\n{student_answer}"
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=250,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
            misses = json.loads(text)
            misses = [m.strip("- • ").strip() for m in misses if isinstance(m, str)]
        except Exception:
            # Heuristic fallback: token overlap
            student_lo = set(re.findall(r"\b\w+\b", (student_answer or "").lower()))
            misses = []
            for p in ref_points:
                p_tokens = set(re.findall(r"\b\w+\b", p.lower()))
                if not p_tokens:
                    continue
                overlap = len(student_lo & p_tokens) / len(p_tokens)
                if overlap < 0.4:
                    misses.append(p)
        return misses[:2]

    def build_hint_ladder(self, missing_points: List[str], context_text: str, concept: str) -> List[Dict[str, str]]:
        """Construct up to 3 escalating hints; grounded with context_text if available."""
        hints: List[Dict[str, str]] = []
        if not missing_points:
            return hints
        focus = "; ".join(missing_points[:2])

        hints.append({
            "level": 1,
            "text": (
                f"Think about *{concept}*. What role does **{missing_points[0]}** play in it? "
                "Which definition, property, or example from your notes would connect to this?"
            ),
        })
        hints.append({
            "level": 2,
            "text": (
                f"A key piece you might be missing: **{missing_points[0]}**. "
                f"Try to state it in one sentence and relate it to the main idea: *{concept}*."
            ),
        })
        quote = context_text.strip().split("\n")[0][:240] if context_text else ""
        grounded = f"\n\nFrom the material:\n> {quote}" if quote else ""
        hints.append({
            "level": 3,
            "text": f"Concrete cue: include **{focus}** when explaining *{concept}*." + grounded,
        })
        return hints

    # ---------------- Socratic question generation ----------------

    def determine_bloom_level(self, grasp_score: float) -> str:
        """Determine appropriate Bloom's level based on grasp score."""
        for level, data in self.bloom_levels.items():
            lo, hi = data["range"]
            if lo <= grasp_score < hi:
                return level
        return "create"

    def get_question_template(self, bloom_level: str, concept: str) -> str:
        """Get question template for specific Bloom's level."""
        templates = {
            "remember":   f"What is {concept}? Can you define or identify the key characteristics of {concept}?",
            "understand": f"How would you explain {concept} in your own words? What does {concept} mean in this context?",
            "apply":      f"How could you use {concept} to solve a real-world problem? Give an example of {concept} in practice.",
            "analyze":    f"How does {concept} relate to other concepts we've studied? What are the key components of {concept}?",
            "evaluate":   f"What are the strengths and weaknesses of {concept}? How would you assess the effectiveness of {concept}?",
            "create":     f"How would you design a new approach using {concept}? What improvements could you make to {concept}?",
        }
        return templates.get(bloom_level, templates["understand"])

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def generate_socratic_question(
        self,
        concept: str,
        grasp_score: float,
        context_text: str = "",
        previous_responses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a Socratic question tailored to the student's grasp level.
        Returns dict with keys: question, expected_response_type, bloom_level, hints, teaching_goal
        """
        bloom_level = self.determine_bloom_level(grasp_score)
        base_template = self.get_question_template(bloom_level, concept)
        prompt = self._build_question_prompt(concept, bloom_level, context_text, base_template, previous_responses)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400,
            )
            raw = (response.choices[0].message.content or "").strip()
            # tolerate code fences or accidental prose
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            try:
                question_data = json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                question_data = json.loads(m.group(0)) if m else {}

            # Minimal normalization
            if "question" not in question_data or not question_data["question"]:
                question_data["question"] = base_template
            question_data.setdefault("expected_response_type", "explanation")
            # normalize hints to 2–3 strings
            hints = question_data.get("hints", [])
            if not isinstance(hints, list):
                hints = []
            hints = [str(h).strip() for h in hints if isinstance(h, (str, int, float))][:3]
            if len(hints) < 2:
                hints += [f"Think about the key aspects of {concept}"] * (2 - len(hints))
            question_data["hints"] = hints

            question_data["bloom_level"] = question_data.get("bloom_level", bloom_level)
            question_data["teaching_goal"] = question_data.get("teaching_goal", question_data["bloom_level"])
            question_data["grasp_score_used"] = float(grasp_score)
            return question_data

        except Exception:
            # Fallback to template question
            return {
                "question": base_template,
                "expected_response_type": "explanation",
                "bloom_level": bloom_level,
                "hints": [f"Think about the key aspects of {concept}"],
                "teaching_goal": bloom_level,
                "grasp_score_used": float(grasp_score),
            }

    def _build_question_prompt(
        self,
        concept: str,
        bloom_level: str,
        context_text: str,
        base_template: str,
        previous_responses: Optional[List[str]] = None,
    ) -> str:
        """Build the GPT prompt for question generation."""
        bloom_verbs = ", ".join(self.bloom_levels[bloom_level]["verbs"])
        prev = f'PREVIOUS STUDENT RESPONSES (use to avoid repetition and to scaffold): {previous_responses}' if previous_responses else ''
        prompt = f"""
You are a Socratic tutor. Generate a thoughtful question about "{concept}" that encourages discovery learning.

REQUIREMENTS:
- Target Bloom's taxonomy level: {bloom_level} (use verbs like: {bloom_verbs})
- Question should guide student thinking, not just test recall
- Include a follow-up question and 2–3 hints that scaffold learning
- Make it engaging and thought-provoking
- If CONTEXT is provided, ground the question in it and avoid introducing outside facts.

CONTEXT FROM COURSE MATERIAL:
{(context_text or "No specific context provided")[:500]}

BASE TEMPLATE: {base_template}

{prev}

Return EXACTLY this JSON format (no prose, no backticks):
{{
  "question": "Your main Socratic question here",
  "follow_up": "A follow-up question to deepen thinking",
  "hints": ["hint1", "hint2", "hint3"],
  "expected_response_type": "explanation|analysis|application|evaluation",
  "teaching_goal": "What understanding you're trying to develop"
}}
""".strip()
        return prompt

    # ---------------- Session orchestration ----------------

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def analyze_student_response(self, question: str, student_response: str, concept: str, expected_type: str) -> Dict:
        """
        (Kept for backward compatibility — current pipeline prefers GraspPredictor.rate_student_answer)
        """
        prompt = f"""
Analyze this student response to a Socratic question about "{concept}".

QUESTION: {question}
STUDENT RESPONSE: {student_response}
EXPECTED RESPONSE TYPE: {expected_type}

Evaluate and return EXACTLY this JSON:
{{
  "understanding_level": 0.0,
  "response_quality": "high|medium|low",
  "key_points_covered": ["point1", "point2"],
  "missing_elements": ["element1", "element2"],
  "suggested_follow_up": "next question to ask",
  "feedback": "brief encouraging feedback",
  "grasp_adjustment": 0.0
}}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            return json.loads(raw)
        except Exception:
            # Fallback analysis
            return {
                "understanding_level": 0.5,
                "response_quality": "medium",
                "key_points_covered": [],
                "missing_elements": [],
                "suggested_follow_up": f"Can you elaborate more on {concept}?",
                "feedback": "Thanks for your response. Let's explore this further.",
                "grasp_adjustment": 0.0,
            }

    def generate_follow_up_question(self, original_question: str, student_response: str,
                                    analysis: Dict, concept: str, new_grasp_score: float) -> Dict:
        """Generate adaptive follow-up question based on response analysis."""
        if analysis.get("response_quality") == "high":
            # Challenge with harder question
            return self.generate_socratic_question(
                concept=concept,
                grasp_score=min(1.0, float(new_grasp_score) + 0.1),
                previous_responses=[student_response],
            )
        elif analysis.get("response_quality") == "low":
            # Provide scaffolding question
            return self.generate_scaffolding_question(concept, student_response, analysis)
        else:
            # Medium quality - clarification question
            return self.generate_clarification_question(concept, analysis)

    def generate_scaffolding_question(self, concept: str, student_response: str, analysis: Dict) -> Dict:
        """Generate a simpler, scaffolding question for struggling students."""
        missing = ", ".join(analysis.get("missing_elements", []))
        scaffolding_prompt = f"""
The student is struggling with "{concept}". Generate a simpler, more guided question.

STUDENT'S PREVIOUS RESPONSE: {student_response}
MISSING ELEMENTS: {missing}

Create a question that:
1) Breaks down the concept into smaller parts
2) Provides more guidance
3) Builds confidence while teaching

Return the same JSON format as before.
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": scaffolding_prompt}],
                temperature=0.5,
                max_tokens=300,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            result = json.loads(raw)
            result["question_type"] = "scaffolding"
            return result
        except Exception:
            return {
                "question": f"Let's break down {concept} into smaller parts. What is one key aspect of {concept} that you're familiar with?",
                "question_type": "scaffolding",
                "bloom_level": "remember",
                "hints": [f"Think about the basic definition of {concept}"],
            }

    def generate_clarification_question(self, concept: str, analysis: Dict) -> Dict:
        """Generate clarification question for medium-quality responses."""
        return {
            "question": analysis.get("suggested_follow_up", f"Can you explain more about how {concept} works?"),
            "question_type": "clarification",
            "bloom_level": "understand",
            "hints": [f"Consider the specific mechanisms of {concept}"],
        }

    def create_learning_session(self, concepts: List[Dict], grasp_predictor, use_sm2: bool = True) -> Dict:
        """Create a structured learning session, optionally using SM-2 scheduling if available."""
        if use_sm2:
            due_concepts = grasp_predictor.get_concepts_due_for_review()
            if due_concepts:
                session_type = "SM-2 Scheduled"
                concept_queue = due_concepts
            else:
                use_sm2 = False

        if not use_sm2:
            concept_priorities = []
            for concept in concepts:
                concept_name = concept["concept"]
                grasp_score = grasp_predictor.get_grasp_score(concept_name)
                confidence = grasp_predictor.get_confidence(concept_name)
                priority_score = (1 - grasp_score) + (1 - confidence)
                concept_priorities.append({
                    "concept": concept_name,
                    "difficulty": concept.get("difficulty", 3.0),
                    "grasp_score": grasp_score,
                    "priority_score": priority_score,
                    "explanation": concept.get("explanation", ""),
                })
            concept_priorities.sort(key=lambda x: x["priority_score"], reverse=True)
            concept_queue = concept_priorities
            session_type = "Practice"

        return {
            "session_id": str(uuid.uuid4())[:8],
            "created_at": datetime.now().isoformat(),
            "concept_queue": concept_queue,
            "current_concept_index": 0,
            "questions_asked": 0,
            "total_concepts": len(concept_queue),
            "session_history": [],
            "session_type": session_type,
        }

    def get_next_question_for_session(
        self,
        session: Dict[str, Any],
        grasp_predictor,
        context_text: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Get the next question for the current learning session."""
        queue = session.get("concept_queue", [])
        idx = session.get("current_concept_index", 0)
        if idx >= len(queue):
            return None  # Session complete

        current_concept = queue[idx]
        concept_name = current_concept.get("concept", "")
        current_grasp = float(grasp_predictor.get_grasp_score(concept_name))

        bloom_level = self.determine_bloom_level(current_grasp)
        teaching_goal = bloom_level

        question_text, follow_up, hints, expected_type = None, None, [], "explanation"
        try:
            payload = self.generate_socratic_question(
                concept=concept_name,
                grasp_score=current_grasp,
                context_text=context_text,
            )
            if isinstance(payload, dict):
                question_text = payload.get("question") or payload.get("prompt") or payload.get("text")
                follow_up = payload.get("follow_up")
                hints = payload.get("hints", []) or []
                expected_type = payload.get("expected_response_type", expected_type)
                bloom_level = payload.get("bloom_level", bloom_level)
                teaching_goal = payload.get("teaching_goal", teaching_goal)
            elif isinstance(payload, str):
                question_text = payload
        except Exception:
            question_text = f"In 3–5 sentences, explain: {concept_name}"

        if not question_text:
            question_text = f"In 3–5 sentences, explain: {concept_name}"

        return {
            "session_id": session.get("session_id"),
            "concept": concept_name,
            "question_number": session.get("questions_asked", 0) + 1,
            "question": question_text,
            "follow_up": follow_up,
            "hints": hints,
            "expected_response_type": expected_type,
            "bloom_level": bloom_level,
            "teaching_goal": teaching_goal,
            "context": context_text or "",
            "concept_difficulty": current_concept.get("difficulty", 3.0),
            "grasp_score_used": current_grasp,
        }

    def _get_recent_responses(self, session: Dict, concept_name: str) -> List[str]:
        """Get recent responses for this concept from session history."""
        responses = []
        for interaction in session.get("session_history", [])[-3:]:
            if interaction.get("concept") == concept_name:
                responses.append(interaction.get("student_response", ""))
        return responses

    def get_session_summary(self, session: Dict) -> Dict:
        """Generate summary of learning session."""
        total_questions = session.get("questions_asked", 0)
        concepts_covered = session.get("current_concept_index", 0)
        hist = session.get("session_history", [])
        if not hist:
            return {"error": "No interactions in session"}

        avg_understanding = sum(h["analysis"].get("understanding_level", 0.0) for h in hist) / len(hist)
        return {
            "session_id": session.get("session_id"),
            "total_questions": total_questions,
            "concepts_covered": concepts_covered,
            "total_concepts": session.get("total_concepts", 0),
            "average_understanding": round(avg_understanding, 2),
            "session_duration": len(hist),
            "completion_rate": round(100.0 * concepts_covered / max(session.get("total_concepts", 1), 1), 1),
        }

    def process_session_response(
        self,
        session: Dict,
        question_data: Dict,
        student_response: str,
        grasp_predictor,
        manual_progression: bool = False,
    ) -> Dict:
        """Process student response and update session state."""
        concept_name = question_data.get("concept", "")
        question_text = question_data.get("question", "")
        ctx = question_data.get("context", "")

        analysis = grasp_predictor.rate_student_answer(
            question=question_text,
            student_response=student_response,
            concept=concept_name,
            context_text=ctx,
        )

        # Update grasp
        grasp_predictor.update_grasp(
            concept_name,
            performance_score=analysis.get("understanding_level", 0.0),
            response_quality=analysis.get("response_quality", "low"),
        )

        # Count prior questions for this concept (before appending current interaction)
        concept_questions = sum(1 for h in session.get("session_history", [])
                                if h.get("concept") == concept_name)

        # Record interaction
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "concept": concept_name,
            "question": question_text,
            "student_response": student_response,
            "analysis": analysis,
            "grasp_before": question_data.get("grasp_score_used", 0.0),
            "grasp_after": grasp_predictor.get_grasp_score(concept_name),
        }
        session.setdefault("session_history", []).append(interaction)
        session["questions_asked"] = session.get("questions_asked", 0) + 1

        # --------- Advancement & hint lock ----------
        ul = float(analysis.get("understanding_level", 0.0) or 0.0)
        threshold = float(getattr(grasp_predictor, "success_threshold", 0.6))
        needs_hints = (ul < threshold)
        recommended_advance = (ul >= threshold)

        idx = int(session.get("current_concept_index", 0) or 0)
        total = int(len(session.get("concept_queue", [])))
        has_more = (idx < max(0, total - 1))  # safe

        did_advance = False
        # Do NOT auto-advance if hints are needed, even if manual_progression=False
        if not manual_progression and recommended_advance and has_more and not needs_hints:
            session["current_concept_index"] = idx + 1
            did_advance = True

        # if learner needs hints, explicitly *stay* on the concept
        stay_on_concept = needs_hints or (manual_progression and not did_advance)

        result = {
            "analysis": analysis,
            "session_updated": True,
            "concept_mastered": ul > 0.8,
            "recommended_advance": recommended_advance,
            "advancing_concept": did_advance,
            "ready_for_next": (did_advance and has_more) or (manual_progression and has_more),
            "has_more": has_more,
            "stay_on_concept": stay_on_concept,
            "lock_on_concept": needs_hints,  # signal for UI
            "questions_on_concept": concept_questions + 1,
        }
        return result


# Convenience factory
def create_socratic_generator() -> SocraticQuestionGenerator:
    return SocraticQuestionGenerator()


# === Adaptive Socratic integration helpers (non-breaking)
from typing import Any as _Any, Dict as _Dict, Optional as _Optional

try:
    from prompt_selector import build_seed_instruction
    from adaptive_prompts import fill_placeholders
except Exception:
    # If modules not present, provide safe fallbacks
    def build_seed_instruction(selection: dict, concept: _Optional[str] = None) -> str:
        tmpl = (selection or {}).get("template") or ""
        return fill_placeholders(tmpl, concept=(concept or "this idea")) if tmpl else ""
    def fill_placeholders(t: str, **kwargs) -> str:
        try:
            return t.format(**kwargs)
        except Exception:
            return t


def _merge_adaptive_into_result(
    result: _Dict[str, _Any],
    selection: _Optional[_Dict[str, _Any]],
    concept: _Optional[str] = None
) -> _Dict[str, _Any]:
    """Inject adaptive metadata + a seed hint (as string) into a generated question dict."""
    if not isinstance(result, dict):
        return result
    if selection:
        result.setdefault("adaptive_state", selection.get("state"))
        result.setdefault("adaptive_prompt", {
            "state": selection.get("state"),
            "prompt_type": selection.get("prompt_type"),
            "template": selection.get("template"),
            "meta": selection.get("meta", {}),
        })
        # Ensure a visible follow-up/hint that reflects the selected move
        seed = build_seed_instruction(selection, concept=concept or result.get("concept"))
        # Attach as follow_up if missing or empty-like
        if seed and not result.get("follow_up"):
            result["follow_up"] = seed
        # Also ensure a hint mirrors the move (append as STRING to avoid mixed types)
        hints = result.get("hints")
        if not isinstance(hints, list):
            hints = []
        if seed and seed not in hints:
            hints.append(seed)
        result["hints"] = hints
    return result


def get_next_question_for_session_adaptive(
    generator: "SocraticQuestionGenerator",
    *args,
    adaptive_state: _Optional[str] = None,
    adaptive_prompt: _Optional[_Dict[str, _Any]] = None,
    concept: _Optional[str] = None,
    **kwargs
) -> _Dict[str, _Any]:
    """
    Wrapper that calls the existing generator and then injects Adaptive Prompt info.
    Usage:
        res = get_next_question_for_session_adaptive(gen, session, ..., adaptive_state=state, adaptive_prompt=selection)
    This avoids modifying the class signature and stays backward-compatible.
    """
    res = generator.get_next_question_for_session(*args, **kwargs)
    # Prefer explicit adaptive_prompt; otherwise ignore.
    selection = adaptive_prompt if isinstance(adaptive_prompt, dict) else None
    return _merge_adaptive_into_result(res, selection, concept=concept)
