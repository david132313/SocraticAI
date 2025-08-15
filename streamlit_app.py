import os
import time
import streamlit as st
from dotenv import load_dotenv

# custom imports
from extractors import (
    extract_text_from_pdf,
    extract_concepts_via_embedding,
    test_openai_connection,
    build_tfidf_index,
    rag_retrieve,
)
from grasp_engine import create_grasp_predictor
from socratic_generator import create_socratic_generator
from learner_state import detect_state, transition_policy
from prompt_selector import select_prompt, build_seed_instruction

# ────────────────────────────────────────────────────────────────────────────────
# Boot & page config
# ────────────────────────────────────────────────────────────────────────────────

load_dotenv()
st.set_page_config(page_title="SocraticAI", page_icon="📚", layout="wide")
st.title("SocraticAI")
st.subheader("Your AI-Powered Socratic Learning Assistant")

# Sidebar
with st.sidebar:
    st.header("Upload Course Materials")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf",
        help="Upload your course materials for personalized learning"
    )
    mode = st.selectbox("Learning Mode", ["Socratic (Adaptive)", "Baseline (Direct Q&A)"])
    manual_progression = st.checkbox("Manual Progression", value=True)
    st.session_state["manual_progression"] = manual_progression
    st.markdown("---")
    st.caption("*Built for DS-UA 301 Final Project*")

# Ensure data dir
os.makedirs("data", exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# Session helpers
# ────────────────────────────────────────────────────────────────────────────────

def _ensure_session_defaults():
    ss = st.session_state
    ss.setdefault("current_filename", None)
    ss.setdefault("concepts_extracted", False)
    ss.setdefault("learning_session", None)
    ss.setdefault("current_question", None)     # payload for the CURRENT active question
    ss.setdefault("awaiting_response", False)
    ss.setdefault("current_hints", [])          # generated after submission
    ss.setdefault("concepts", [])
    ss.setdefault("predictor", None)
    ss.setdefault("socratic_generator", create_socratic_generator())
    ss.setdefault("rag_index", None)

    # Adaptive prompts state
    ss.setdefault("turn_log", [])
    ss.setdefault("last_state", None)
    ss.setdefault("question_shown_at", None)

    # Chat transcript & de-dupe
    ss.setdefault("use_chat_ui", True)
    ss.setdefault("chat_log", [])               # [{role: "assistant"|"user", content: str}]
    ss.setdefault("rendered_qids", set())       # de-dupe set for question/summary messages

def _reset_for_new_file(filename: str):
    ss = st.session_state
    ss.current_filename = filename
    ss.concepts_extracted = False
    ss.learning_session = None
    ss.current_question = None
    ss.awaiting_response = False
    ss.current_hints = []
    ss.concepts = []
    ss.predictor = None
    ss.rag_index = None

    # Reset adaptive/logging/chat state so sessions don't bleed across uploads
    ss.turn_log = []
    ss.last_state = None
    ss.question_shown_at = None
    ss.chat_log = []
    ss.rendered_qids = set()

def _save_uploaded(uploaded) -> str:
    safe_name = os.path.basename(uploaded.name)
    file_path = os.path.join("data", safe_name)
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())
    return file_path

def _extract_and_prepare(file_path: str, file_display_name: str):
    """Extract text, test API, extract concepts, init predictor, build RAG."""
    # 1) Extract text
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(file_path)
        st.session_state.extracted_text = extracted_text

    if isinstance(extracted_text, str) and extracted_text.startswith("Error"):
        st.error(extracted_text)
        st.stop()
    st.success("Text extraction completed!")
    st.write(f"**File size:** {os.path.getsize(file_path)} bytes")
    st.write(f"**Extracted words:** {len(extracted_text.split())}")

    # 2) API check
    if not test_openai_connection():
        st.error("OpenAI API connection failed. Please check your API key in .env file.")
        st.stop()

    # 3) Concepts (if not already done)
    if not st.session_state.concepts_extracted:
        with st.spinner("Identifying key concepts with embeddings & GPT..."):
            try:
                concepts = extract_concepts_via_embedding(extracted_text, n_clusters=12)
            except Exception as e:
                st.error(f"Concept extraction failed: {e}")
                st.stop()
        st.success(f"Found {len(concepts)} key concepts!")

        predictor = create_grasp_predictor()
        for concept in concepts:
            predictor.initialize_concept(concept['concept'], concept['difficulty'])

        st.session_state.concepts = concepts
        st.session_state.predictor = predictor
        st.session_state.concepts_extracted = True

        # Persist grasp data
        try:
            grasp_file = os.path.join("data", f"{file_display_name}_grasp.json")
            predictor.save_to_file(grasp_file)
        except Exception as e:
            st.warning(f"Could not save initial grasp data: {e}")
    else:
        # Use cached
        concepts = st.session_state.concepts
        predictor = st.session_state.predictor
        st.success(f"Using cached concepts: {len(concepts)} found!")

    # 4) Build/refresh RAG index for targeted context
    try:
        st.session_state.rag_index = build_tfidf_index(extracted_text)
    except Exception:
        st.session_state.rag_index = None

# ────────────────────────────────────────────────────────────────────────────────
# Chat helpers
# ────────────────────────────────────────────────────────────────────────────────

def _append_chat(role: str, content: str):
    """Append a message to the persistent chat log."""
    st.session_state.chat_log.append({"role": role, "content": content})

def _format_question_block(qd: dict) -> str:
    """Make a friendly markdown block for a question payload."""
    header_bits = [f"**Question about:** {qd.get('concept','?')}"]
    if isinstance(qd.get("bloom_level"), str):
        header_bits.append(f"_Bloom: {qd['bloom_level'].title()}_")
    if "grasp_score_used" in qd:
        header_bits.append(f"_Grasp: {qd['grasp_score_used']:.2f}_")
    header = " · ".join(header_bits)

    follow = ""
    if qd.get("follow_up"):
        follow = f"\n\n_{qd['follow_up']}_"

    hints_md = ""
    hints = qd.get("hints") or []
    if hints:
        hints_md = "\n\n**Starter hints (optional):**\n" + "\n".join(
            f"- {(h.get('text') if isinstance(h, dict) else str(h))}" for h in hints
        )

    return f"{header}\n\n**Q:** {qd.get('question','')}{follow}{hints_md}"

def _record_question_in_chat(qd: dict):
    """Add the question to chat once per unique qid."""
    qkey = f"{qd.get('session_id')}:{qd.get('concept')}:{qd.get('question_number')}"
    if qkey not in st.session_state.rendered_qids:
        _append_chat("assistant", _format_question_block(qd))
        st.session_state.rendered_qids.add(qkey)

# ────────────────────────────────────────────────────────────────────────────────
# UI sections
# ────────────────────────────────────────────────────────────────────────────────

def _render_concepts(concepts, predictor):
    if not concepts:
        return
    st.subheader("Identified Concepts with Grasp Tracking")
    col1, col2 = st.columns([3, 1])
    for i, c in enumerate(concepts, start=1):
        with col1:
            st.markdown(f"**{i}. {c['concept']}**")
            st.write(f"- Difficulty: {c['difficulty']}/5")
            st.write(f"- Explanation: {c['explanation']}")
            summary = predictor.get_concept_summary(c['concept'])
            st.write(f"- Grasp Level: {summary['grasp_score']} ({summary['status']})")
            st.write(f"- Confidence: {summary['confidence']}")
            st.write("---")
        with col2:
            st.metric(label=f"Concept {i}", value=f"{summary['grasp_score']:.1f}", delta=summary['status'])

def _start_session_button(concepts, predictor):
    if st.button("Start Learning Session", type="primary"):
        use_sm2 = (mode == "Socratic (Adaptive)")
        st.session_state.learning_session = (
            st.session_state.socratic_generator.create_learning_session(
                concepts, predictor, use_sm2=use_sm2
            )
        )
        st.session_state.awaiting_response = False
        st.session_state.current_hints = []
        st.session_state.chat_log = []
        st.session_state.rendered_qids = set()
        st.success(f"Session started! {len(concepts)} concepts to explore.")

def _next_question_button():
    if st.button("Get Next Question") and st.session_state.learning_session:
        sess = st.session_state.learning_session
        queue = sess.get("concept_queue", [])
        idx = sess.get("current_concept_index", 0)

        # Manual progression: move forward if we've engaged or shown hints
        if st.session_state.get("manual_progression", False) and queue:
            current_concept = queue[idx]["concept"] if idx < len(queue) else None
            had_engagement = any(h.get("concept") == current_concept for h in sess.get("session_history", []))
            if had_engagement or st.session_state.get("current_hints"):
                if idx < len(queue) - 1:
                    sess["current_concept_index"] = idx + 1
                    idx = sess["current_concept_index"]
                else:
                    _render_final_summary(sess)
                    return
                st.session_state.current_hints = []
                st.session_state.current_question = None

        # End guard
        if idx >= len(queue):
            _render_final_summary(sess)
            return

        concept_name = queue[idx]["concept"]

        # RAG context
        context_text = ""
        if st.session_state.get("rag_index"):
            context_text = rag_retrieve(st.session_state.rag_index, concept_name, top_k=6, window=1)

        # Generate question
        if mode == "Socratic (Adaptive)":
            qd = st.session_state.socratic_generator.get_next_question_for_session(
                sess, st.session_state.predictor, context_text
            )
        else:
            qd = {
                "session_id": sess["session_id"],
                "concept": concept_name,
                "question_number": sess["questions_asked"] + 1,
                "question": f"In 3–5 sentences, explain: {concept_name}",
                "follow_up": None,
                "hints": [],
                "expected_response_type": "explanation",
                "teaching_goal": "Recall/understand the core idea.",
                "context": context_text,
            }

        # --- Adaptive Socratic Prompt Integration ---
        if os.getenv("ADAPTIVE_PROMPTS", "1") != "0":
            detected = detect_state(st.session_state.turn_log) if st.session_state.turn_log else {"state": "silent"}
            next_state = transition_policy(st.session_state.get("last_state"), detected.get("state"))
            st.session_state.last_state = next_state
            sel = select_prompt(next_state, st.session_state.turn_log, bloom_level=qd.get("bloom_level"))
            seed = build_seed_instruction(sel, concept=qd.get("concept"))

            qd["adaptive_state"] = next_state
            qd["adaptive_prompt"] = sel
            if not qd.get("follow_up"):
                qd["follow_up"] = seed
            hints = qd.get("hints") or []
            # Avoid duplicate seed
            def _hint_text(h):
                return (h.get("text") or "") if isinstance(h, dict) else str(h or "")
            if seed and all(seed != _hint_text(h) for h in hints):
                hints.append({"level": 1, "text": seed})
            qd["hints"] = hints

        st.session_state.question_shown_at = time.time()
        st.session_state.current_question = qd
        st.session_state.awaiting_response = True
        st.session_state.current_hints = []

        # Put the question into chat (deduped on rerun) and immediately rerun to render it
        _record_question_in_chat(qd)
        st.rerun()  # <— ensure the new question appears right away

def _reset_session_button():
    if st.button("Reset Session"):
        st.session_state.learning_session = None
        st.session_state.current_question = None
        st.session_state.awaiting_response = False
        st.session_state.current_hints = []
        st.session_state.chat_log = []
        st.session_state.rendered_qids = set()
        st.info("Session reset. Click 'Start Learning Session' to begin.")
        st.rerun()

def _render_session_controls(concepts, predictor):
    col1, col2, col3 = st.columns(3)
    with col1:
        _start_session_button(concepts, predictor)
    with col2:
        _next_question_button()
    with col3:
        _reset_session_button()

def _render_chat_ui():
    """Always render the full transcript first, then the input."""
    for msg in st.session_state.chat_log:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

    # Chat input (returns None until user sends)
    return st.chat_input("Type your answer…")

def _handle_submission(submitted_response: str):
    # Basic guards
    if not submitted_response.strip():
        st.warning("Please type an answer before submitting.")
        return
    if not st.session_state.get("current_question"):
        st.warning("No active question. Click 'Get Next Question' first.")
        return
    if not st.session_state.get("learning_session"):
        st.warning("No active session. Click 'Start Learning Session' first.")
        return

    q = st.session_state.current_question
    sess = st.session_state.learning_session
    predictor = st.session_state.predictor

    # Echo user message to transcript
    _append_chat("user", submitted_response)

    with st.spinner("Analyzing your response..."):
        if mode == "Socratic (Adaptive)":
            result = st.session_state.socratic_generator.process_session_response(
                sess, q, submitted_response, predictor,
                manual_progression=st.session_state.get("manual_progression", False),
            )
            analysis = result["analysis"]

            # ---------- HINT LADDER ----------
            st.session_state.current_hints = []
            if analysis.get("understanding_level", 0.0) < float(getattr(predictor, "success_threshold", 0.6)):
                concept_name = q.get("concept", "")

                # RAG context for hints
                context_text = ""
                if st.session_state.get("rag_index"):
                    context_text = rag_retrieve(st.session_state.rag_index, concept_name, top_k=6, window=1)

                # Build hint ladder
                reference = predictor.get_reference_answer(
                    concept_name, question=q.get("question", ""), context_text=context_text,
                )
                ref_points = st.session_state.socratic_generator.extract_key_points(reference, k=5)
                misses = st.session_state.socratic_generator.missing_points(ref_points, submitted_response)
                st.session_state.current_hints = st.session_state.socratic_generator.build_hint_ladder(
                    misses, context_text, concept_name
                )

        else:
            # ---------- BASELINE ----------
            concept_name = q.get("concept", "")
            analysis = predictor.rate_student_answer(
                question=q.get("question", ""),
                student_response=submitted_response,
                concept=concept_name,
                context_text=q.get("context", ""),
            )
            predictor.update_grasp(
                concept_name,
                performance_score=analysis.get("understanding_level", 0.0),
                response_quality=analysis.get("response_quality", "low"),
            )
            # Advance to next concept immediately; no follow-ups or hints
            sess["questions_asked"] += 1
            sess["current_concept_index"] += 1
            st.session_state.current_hints = []

    # Prepare assistant feedback message (goes into transcript)
    ul = analysis.get("understanding_level")
    rq = analysis.get("response_quality", "")
    rq_text = rq.title() if isinstance(rq, str) else str(rq)
    fb = analysis.get("feedback") or analysis.get("explanation") or ""

    parts = ["**Response submitted!**"]
    if isinstance(ul, (int, float)):
        parts.append(f"**Understanding Level:** {ul:.2f}/1.0")
    parts.append(f"**Response Quality:** {rq_text}")
    if fb:
        parts.append(f"**Feedback:** {fb}")

    # Hints (if any)
    hints = st.session_state.get("current_hints") or []
    if mode == "Socratic (Adaptive)" and hints:
        parts.append("**Hints to improve your answer:**")
        for h in hints:
            parts.append(f"- **Hint {h['level']}:** {h['text']}")

    _append_chat("assistant", "\n\n".join(parts))

    # Turn log (for adaptive selection)
    try:
        started = st.session_state.get("question_shown_at")
        latency = float(time.time() - started) if started else 0.0
        st.session_state.turn_log.append({
            "timestamp": time.time(),
            "concept": q.get("concept"),
            "question": q.get("question"),
            "student_response": submitted_response,
            "response_latency_sec": latency,
            "quality": analysis.get("response_quality", ""),
            "understanding": analysis.get("understanding_level", 0.0),
            "clarification_count": 0,
            "adaptive_prompt": q.get("adaptive_prompt"),
        })
    except Exception:
        pass
    st.session_state.question_shown_at = None

    # We’re done processing the answer for this question
    st.session_state.awaiting_response = False

    # Save grasp data best-effort
    try:
        grasp_file = os.path.join("data", f"{st.session_state.current_filename}_grasp.json")
        st.session_state.predictor.save_to_file(grasp_file)
    except Exception as e:
        st.warning(f"Could not save grasp data: {e}")

    # Guidance block (in transcript-friendly form)
    sess = st.session_state.get("learning_session")
    if sess:
        idx = sess.get("current_concept_index", 0)
        total = len(sess.get("concept_queue", []))
        has_more = idx < max(0, total - 1)
        if has_more:
            _append_chat("assistant", "When you’re ready, click **Get Next Question** to continue!")
        else:
            _render_final_summary(sess)

    # Rerun so the newly appended messages render immediately
    st.rerun()  # <— ensure your latest answer + feedback show up right away

def _render_final_summary(sess):
    """
    Append a final session summary to the chat (chat UI) or render inline (legacy UI).
    De-duped so it won't post twice across reruns.
    """
    key = f"summary:{sess.get('session_id')}"
    if key in st.session_state.rendered_qids:
        return

    # Try to compute a summary
    lines = ["🎉 **Learning session complete! Great work!**"]
    try:
        summary = st.session_state.socratic_generator.get_session_summary(sess)
        lines.append(f"- Questions answered: {summary.get('total_questions')}")
        lines.append(f"- Concepts covered: {summary.get('concepts_covered')}/{summary.get('total_concepts')}")
        avg_u = summary.get("average_understanding")
        if isinstance(avg_u, (int, float)):
            lines.append(f"- Average understanding: {avg_u:.1f}/1.0")
    except Exception:
        pass

    if st.session_state.get("use_chat_ui", True):
        _append_chat("assistant", "\n".join(lines))
    else:
        st.success(lines[0])
        for ln in lines[1:]:
            st.write(ln)

    # mark as rendered once
    st.session_state.rendered_qids.add(key)
    # clear current question / awaiting flags
    st.session_state.current_question = None
    st.session_state.awaiting_response = False

def _render_footer():
    """Footer is shown in the sidebar during chat mode to avoid sitting above the input."""
    if not st.session_state.get("use_chat_ui", True):
        st.markdown("---")
        st.markdown("*Built for DS-UA 301 Final Project*")
        st.caption(f"Mode: {mode}")

# ────────────────────────────────────────────────────────────────────────────────
# Main flow
# ────────────────────────────────────────────────────────────────────────────────

_ensure_session_defaults()

if uploaded_file is not None:
    # Detect new file and reset state
    if st.session_state.current_filename != uploaded_file.name:
        _reset_for_new_file(uploaded_file.name)

    # Save & confirm
    file_path = _save_uploaded(uploaded_file)
    st.success(f"Uploaded: {uploaded_file.name}")
    st.success(f"Saved to: {file_path}")

    # Extract + prepare
    _extract_and_prepare(file_path, uploaded_file.name)

    # Concepts section
    _render_concepts(st.session_state.concepts, st.session_state.predictor)

    # Interactive session controls
    st.subheader("Interactive Learning Session")
    if not (st.session_state.concepts and st.session_state.predictor):
        st.info("Please upload a PDF and wait for concept extraction to complete before starting a learning session.")
    else:
        _render_session_controls(st.session_state.concepts, st.session_state.predictor)

    # ==== Chat transcript + input (always render transcript first) ====
    if st.session_state.get("use_chat_ui", True):
        user_text = _render_chat_ui()
        if user_text is not None:
            _handle_submission(user_text)
    else:
        # (Legacy) single-question UI
        qd = st.session_state.get("current_question")
        if qd and st.session_state.get("awaiting_response"):
            st.markdown("---")
            st.markdown(_format_question_block(qd))
            student_response = st.text_area("Your answer:", height=150, key="student_response_input")
            if st.button("Submit Response", type="primary"):
                _handle_submission(student_response or "")

# Footer (only appears here when not using chat UI)
_render_footer()
