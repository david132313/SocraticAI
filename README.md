# SocraticAI — Adaptive Socratic Learning Assistant

**Turns a course PDF into a personal AI tutor**  
Extracts key concepts, asks adaptive Socratic questions, and tracks learning progress for deeper understanding and better retention.

---

## 📚 Overview
SocraticAI transforms static course materials — lecture slides, readings, or PDFs — into an interactive tutoring experience.  
It uses **adaptive Socratic questioning**, **learner state detection & grasp track**, and **spaced repetition** to promote active recall, critical thinking, and long-term retention.

---

## ✨ Core Features

- **Smart PDF Concept Extraction** — Handles messy formatting with OCR fallback, slide vs. prose segmentation, clustering, and LLM-based concept naming.  
- **Adaptive Socratic Questioning** — Targets Bloom’s taxonomy levels based on learner state and grasp score, rotating styles to prevent repetition.  
- **Learner Grasp Tracking** — Continuous score blending answer quality, confidence, hesitation, and time since last review.  
- **Real-Time State Detection** — Classifies learners as confident, hesitant, or guessing to tailor responses.  
- **Hint Ladder Guidance** — Nudges from subtle hints to full explanations without giving away answers too soon.  
- **Spaced Repetition (SM-2)** — Brings back weaker concepts sooner, mastered ones later.  
- **Baseline vs. Adaptive Modes** — Compare adaptive tutoring with a static Q&A baseline.

---

## 🔄 How It Works

1. **Extract concepts** → Clean text from PDFs, cluster, name, and rate difficulty.  
2. **Track grasp** → Update scores based on performance, confidence, and recency.  
3. **Detect learner state** → Adapt tone and pacing for confident, hesitant, or guessing learners.  
4. **Select question style** → Choose Bloom’s level & retrieve relevant material.  
5. **Deliver with hints** → Provide stepwise hints and references as needed.  
6. **Reinforce** → Schedule reviews with SM-2 for optimal retention.

---

## 🚀 Getting Started

```bash
# Clone repo
git clone https://github.com/catharinelii/SocraticAI.git
cd SocraticAI

# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

---
**Core functions (V1 pipeline — currently in use):** Catharine Li  
-PDF text extraction & cleaning
-Segmentation, embedding retrieval, clustering
-LLM-based concept naming & explanations
-TF-IDF RAG context retrieval
-Learner state detection & adaptive Socratic prompts
-Grasp score tracking & SM-2 scheduling
-Full integration into streamlit_app.py

**Enhancements (V2 pipeline — not integrated):** David Zhu  
- OCR fallback for image-based PDFs  
- Slide/bullet segmentation + auto mode selection  
- Adaptive clustering (`pick_n_clusters`)  
- Anchor-based relevance filtering  
- History-aware difficulty adjustments  
- Expanded difficulty heuristics  
- extract_concepts_v2_adaptive (standalone function, not in main app)
> **Note:** To try:
> ```python
> from extractors import extract_concepts_v2_adaptive
> concepts = extract_concepts_v2_adaptive("path/to/your.pdf")
> ```

### Experimental Add-on — Similarity-Aware Difficulty Scoring
Developed and tested by David Zhu: Adjusts concept difficulty if the name is similar to a concept you’ve already learned. Not part of the main app — see experiments/test_report.pdf for details.

**Example usage:**
```python
from test_similarity import adjust_difficulty

new_concepts = ["Machine Learning", "Parameter Efficient Fine-Tuning"]
seen_concepts = ["Machine Learning", "Testing Documentation"]

adjusted = adjust_difficulty(new_concepts, seen_concepts, similarity_threshold=0.82, lower_by=1)
print(adjusted)
```
