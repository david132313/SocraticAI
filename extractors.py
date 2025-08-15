"""
V2 extractor (revised):
- OCR fallback when text is sparse
- Slide/bullet segmenter for decks (keeps short headers)
- Extended ML/DL/VLM anchors + adaptive similarity threshold
- Adaptive cluster count for short inputs
- V2 pipeline with relevance gating (tuned to accept ~3/4 on image-heavy decks)
"""
import os
import re
import io
import json
import glob
import logging
from difflib import SequenceMatcher
from typing import List, Any, Dict, Iterable, Tuple, Optional, Set
import fitz  # PyMuPDF
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#david Zhu
# Optional OCR deps (ok if missing on desktop; available in Colab after apt/pip step)
try:
    import pytesseract
    from PIL import Image
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

# ---- Config ----
OPENAI_MODEL_CHAT  = os.getenv("OPENAI_MODEL_CHAT",  "gpt-4o-mini")
OPENAI_MODEL_EMBED = os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-small")

_logger = logging.getLogger(__name__)

# We use the new OpenAI 1.x client #david Zhu
import openai
def _get_openai_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------- Connectivity ------------------
def test_openai_connection() -> bool:
    try:
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_CHAT,
            messages=[{"role": "user", "content": "Say: API test successful"}],
            max_tokens=10,
            temperature=0,
        )
        txt = (resp.choices[0].message.content or "").lower()
        return "successful" in txt
    except Exception as e:
        _logger.warning(f"API test failed: {e}")

# ----------------- PDF Text Extraction ------------------
def extract_text_from_pdf(file_path: str) -> str:
    """Native text layer only (no OCR)."""
    try:
        chunks = []
        with fitz.open(file_path) as doc:
            if doc.needs_pass:
                return "Error extracting text: PDF is encrypted/password-protected."
            for page in doc:
                t = page.get_text("text") or ""
                if t.strip():
                    chunks.append(t)
        return "\n\n".join(chunks) if chunks else "Error extracting text: No extractable text found."
    except Exception as e:
        _logger.error(f"Error extracting text from {file_path}: {e}")
        return f"Error extracting text: {e}"
        return False
#david Zhu
def extract_text_from_pdf_ocr_fallback(
    file_path: str,
    *,
    dpi: int = 300,
    lang: str = "eng",
    min_chars_to_skip_ocr: int = 1000,
) -> str:
    """
    Try native text first; if tiny, render pages and run OCR; merge both.
    Works even when OCR deps are absent (returns native text).
    """
    native = extract_text_from_pdf(file_path) or ""
    native_compact = re.sub(r"\s+", "", native)
    if len(native_compact) >= min_chars_to_skip_ocr or not _HAS_OCR:
        return native

    ocr_chunks: List[str] = []
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                txt = pytesseract.image_to_string(img, lang=lang)
                if txt.strip():
                    ocr_chunks.append(txt)
    except Exception as e:
        _logger.warning(f"OCR fallback failed ({file_path}): {e}")

    combined = (native + "\n\n" if native.strip() else "") + "\n\n".join(ocr_chunks)
    return combined

# ----------------- Segmentation ------------------
def split_into_sentences(text: str, min_len: int = 20) -> List[str]:
    if not isinstance(text, str):
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in parts:
        s = " ".join(s.split())
        if len(s) >= min_len and re.search(r"[A-Za-z0-9]", s):
            out.append(s)
    return out

#david Zhu
def split_into_segments_slide(text: str, min_len: int = 8) -> List[str]:
    """
    Slide/bullet segmenter: keeps short headers and bullet items.
    """
    chunks: List[str] = []
    for line in re.split(r"[\r\n]+", text or ""):
        # strip "Slide 12:" style labels
        line = re.sub(r"^\s*slide\s*\d+[:\-]?\s*", "", line, flags=re.I)
        # split common bullet symbols
        parts = re.split(r"[•●▪▫–—\-]\s+", line)
        for p in parts:
            p = " ".join(p.split())
            if len(p) >= min_len and re.search(r"[A-Za-z]", p):
                chunks.append(p)
    return chunks

# ----------------- Embeddings ------------------
@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def get_embeddings(texts: List[str], batch_size: int = 256) -> List[List[float]]:
    if not texts:
        return []
    client = _get_openai_client()
    all_embs: List[List[float]] = []
    bs = max(1, int(batch_size))
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        resp = client.embeddings.create(input=chunk, model=OPENAI_MODEL_EMBED)
        vecs = [obj.embedding for obj in resp.data]
        if len(vecs) != len(chunk):
            raise RuntimeError(f"Embedding count mismatch: got {len(vecs)} for {len(chunk)} inputs")
        all_embs.extend(vecs)
    return all_embs

# ----------------- Clustering ------------------
def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 12) -> List[int]:
    n = int(len(embeddings))
    if n == 0:
        return []
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    k = max(1, min(int(n_clusters), n))
    if k == 1:
        return [0] * n
    model = AgglomerativeClustering(n_clusters=k)
    return model.fit_predict(embeddings)
#david Zhu
def pick_n_clusters(n_segments: int, lo: int = 6, hi: int = 16) -> int:
    """Adaptive cluster count for short inputs (slide decks)."""
    if n_segments <= lo:
        return max(2, n_segments)
    if n_segments >= 200:
        return hi
    return max(3, min(hi, n_segments // 8))

# ----------------- Representative Selection ------------------
def pick_representatives(
    sentences: List[str], embeddings: np.ndarray, labels: List[int]
) -> List[str]:
    reps: Dict[int, str] = {}
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        cluster = embeddings[idxs]
        if len(cluster) == 1:
            reps[label] = sentences[idxs[0]]
            continue
        centroid = cluster.mean(axis=0)
        dists = np.linalg.norm(cluster - centroid, axis=1)
        best_idx = idxs[int(np.argmin(dists))]
        reps[label] = sentences[best_idx]
    return [reps[k] for k in sorted(reps.keys())]

# ----------------- Prior-card difficulty adjust (unchanged) ------------------
#David Zhu
_PRIOR_CARD_CACHE: Optional[Set[str]] = None
def _load_prior_card_names(data_dir: str = "data") -> Set[str]:
    global _PRIOR_CARD_CACHE
    if _PRIOR_CARD_CACHE is not None:
        return _PRIOR_CARD_CACHE
    names: Set[str] = set()
    for path in glob.glob(os.path.join(data_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, dict) and any(key in v for key in ("grasp_score", "confidence", "created")):
                            names.add(str(k).strip())
        except Exception:
            continue
    _PRIOR_CARD_CACHE = names
    return names
#David Zhu
def _best_name_similarity(name: str, candidates: Iterable[str]) -> Tuple[Optional[str], float]:
    name_norm = name.strip().lower()
    best_match, best_ratio = None, 0.0
    for c in candidates:
        ratio = SequenceMatcher(a=name_norm, b=str(c).strip().lower()).ratio()
        if ratio > best_ratio:
            best_match, best_ratio = c, ratio
    return best_match, best_ratio
#David Zhu
def adjust_difficulty_based_on_history(
    concept_name: str,
    base_difficulty: int,
    data_dir: str = "data",
    similarity_threshold: float = 0.82,
    lower_by: int = 1,
) -> int:
    if not concept_name:
        return base_difficulty
    prior = _load_prior_card_names(data_dir)
    if not prior:
        return base_difficulty
    _, ratio = _best_name_similarity(concept_name, prior)
    hint_text = concept_name.lower()
    based_on_hint = any(p in hint_text for p in ["based on", "extension of", "variant of", "follow-up to"])
    if ratio >= similarity_threshold or based_on_hint:
        return max(1, int(base_difficulty) - lower_by)
    return base_difficulty

# ----------------- Concept Explanation (LLM) ------------------
@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def explain_concept(sentence: str, data_dir: str = "data") -> Dict[str, Any]:
    client = _get_openai_client()
    rubric = """
Rate DIFFICULTY on a 1–5 scale using this rubric (do NOT default to 3):
1 = basic definition or everyday fact; minimal prerequisites
2 = course term or simple formula recall; 1 prerequisite idea
3 = multi-part definition OR straightforward application; 1–2 prerequisites
4 = multi-step reasoning/procedure OR heavy terminology; 2+ prerequisites
5 = derivation/proof/nuanced edge cases/complexity analysis; high expertise
If uncertain between two levels, choose the HIGHER one.
""".strip()

    prompt = f"""
Extract the core concept from the sentence, name it, rate difficulty 1–5 using the rubric, 
and write a one-sentence explanation. Also justify the difficulty briefly (1 short clause).

Return JSON only (no prose, no backticks):
{{
  "concept": "<short name>",
  "difficulty": <1–5 int>,
  "explanation": "<one concise sentence>",
  "difficulty_rationale": "<why this level>"
}}

Rubric:
{rubric}

Sentence:
{sentence}
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL_CHAT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=260,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()

    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(m.group(0)) if m else {}

    concept = str(data.get("concept", "")).strip()
    explanation = str(data.get("explanation", "")).strip()
    rationale = str(data.get("difficulty_rationale", "")).strip()

    try:
        difficulty = int(data.get("difficulty"))
    except Exception:
        difficulty = None
    if difficulty is None or not (1 <= difficulty <= 5):
        difficulty = heuristic_difficulty(sentence, data.get("explanation",""))

    difficulty_adjusted = adjust_difficulty_based_on_history(
        concept_name=concept,
        base_difficulty=difficulty,
        data_dir=data_dir,
        similarity_threshold=0.82,
        lower_by=1,
    )

    return {
        "concept": concept,
        "difficulty": difficulty_adjusted,
        "explanation": explanation,
        "difficulty_rationale": rationale,
    }
#david Zhu_modified
def heuristic_difficulty(sentence: str, explanation: str = "") -> int:
    txt = f"{sentence} {explanation}".lower()
    score = 3
    hard_markers = [
        "prove","derive","theorem","lemma","corollary","convergence","complexity","np-hard",
        "gradient","hessian","eigen","bayes","likelihood","integral","asymptotic",
        "regularization","lagrangian","optimality","time complexity","space complexity","o(","Ω(","θ("
    ]
    easy_markers = ["define","definition","means that","consists of","is called","example"]
    if any(k in txt for k in hard_markers): score += 1
    if any(k in txt for k in easy_markers): score -= 1
    if len(sentence) > 220: score += 1
    if re.search(r"[∑∫=≤≥→≈±^_]|\b(d/dx|∂/∂|\d+\s*%)\b", sentence): score += 1
    return int(max(1, min(5, score)))

# ----------------- Anchor relevance (EXTENDED + adaptive) ------------------ #David Zhu
# Base anchors
_BASE_ANCHORS = [
    "machine learning","deep learning","neural networks","gradient descent","optimization",
    "loss function","regularization","convolutional neural network","transformer","attention",
    "backpropagation","activation function","evaluation metrics","dimensionality reduction",
    "retrieval augmented generation","vector database","embeddings","semantic similarity",
    "fine-tuning","parameter-efficient fine-tuning","large language model","tokenization",
    "pretraining","transfer learning"
]

# VLM / multimodal additions
_VLM_ANCHORS = [
    "vision-language model","VLM","multimodal","Flamingo","Perceiver Resampler",
    "gated cross-attention","cross-attention","visual tokens","few-shot learning",
    "video understanding","CLIP","ALIGN","NFNet","Chinchilla"
]
_ALL_ANCHORS = _BASE_ANCHORS + _VLM_ANCHORS

_ANCHOR_EMB: Optional[np.ndarray] = None  # lazy-initialized

def _get_anchor_embeddings() -> np.ndarray:
    global _ANCHOR_EMB
    if _ANCHOR_EMB is None:
        _ANCHOR_EMB = np.array(get_embeddings(_ALL_ANCHORS), dtype=float)
    return _ANCHOR_EMB

def auto_sim_threshold(n_segments: int, base: float = 0.68) -> float:
    """Relax the gate when we have few segments (slide decks)."""
    return 0.62 if n_segments < 30 else base

def is_relevant(name: str, thresh: float) -> bool:
    if not name:
        return False
    anchors = _get_anchor_embeddings()
    v = np.array(get_embeddings([name])[0], dtype=float).reshape(1, -1)
    sim = float(cosine_similarity(v, anchors).max())
    return sim >= thresh

# ----------------- Main V2 (adaptive, slide-friendly) ------------------
#David Zhu
ADMIN_NOISE_RE = re.compile(
    r"\b(new\s+documentation|commit\s+message|meeting\s+notes|doc(ument)?\s+template)\b",
    re.I
)

def _segment_auto(text: str) -> List[str]:
    """
    Choose a segmenter automatically:
    - use prose sentences if we have enough long sentences
    - otherwise fall back to slide/bullet segmentation
    """
    sents = split_into_sentences(text, min_len=20)
    if len(sents) >= 30:
        return sents
    # slide mode
    return split_into_segments_slide(text, min_len=8)

def extract_concepts_v2_adaptive(
    full_text: str,
    *,
    sim_base: float = 0.68
) -> List[Dict[str, Any]]:
    """
    Slide-aware V2: segment -> filter -> embed/cluster -> reps -> LLM -> relevance gate.
    Tuned so image-heavy decks accept ~3 of 4 salient headings instead of 0.
    """
    segs = [s for s in _segment_auto(full_text) if not ADMIN_NOISE_RE.search(s)]
    if not segs:
        return []

    k = pick_n_clusters(len(segs))                # fewer clusters for short inputs
    sim_thr = auto_sim_threshold(len(segs), sim_base)  # e.g., 0.62 for small decks

    embs = get_embeddings(segs)
    labels = cluster_embeddings(np.array(embs, dtype=float), n_clusters=k)
    reps = pick_representatives(segs, np.array(embs, dtype=float), labels)

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for sent in reps:
        info = explain_concept(sent)
        name = (info.get("concept") or "").strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        if not is_relevant(name, sim_thr):
            # borderline: try explanation string as a backup name
            alt = (info.get("explanation") or "").split(".")[0].strip()
            if not alt or not is_relevant(alt, sim_thr):
                continue
            name = alt
            key = name.casefold()
            if key in seen:
                continue
        seen.add(key)
        d = info.get("difficulty")
        if not isinstance(d, int) or not (1 <= d <= 5):
            d = heuristic_difficulty(sent, info.get("explanation",""))
        out.append({"concept": name, "difficulty": int(d), "explanation": info.get("explanation","").strip()})
    return out

# ----------------- Old pipeline kept for compatibility ------------------
def extract_concepts_via_embedding(full_text: str, n_clusters: int = 12) -> List[Dict[str, Any]]:
    sentences = split_into_sentences(full_text)
    if not sentences:
        return []
    embs = get_embeddings(sentences)
    if not embs:
        return []
    embeddings = np.array(embs, dtype=float)
    labels = cluster_embeddings(embeddings, n_clusters=n_clusters)
    reps = pick_representatives(sentences, embeddings, labels)

    explained: List[Dict[str, Any]] = []
    seen = set()
    for sent in reps:
        info = explain_concept(sent)
        name = str(info.get("concept", "")).strip()
        key = name.casefold() if name else ""
        d_raw = info.get("difficulty", None)
        try:
            d_int = int(d_raw) if d_raw is not None else None
        except Exception:
            d_int = None
        if d_int is None or d_int < 1 or d_int > 5:
            d_int = heuristic_difficulty(sent, info.get("explanation", ""))

        if name and key not in seen:
            seen.add(key)
            explained.append({
                "concept": name,
                "difficulty": int(max(1, min(5, d_int))),
                "explanation": str(info.get("explanation", "")).strip(),
            })
    return explained

# ----------------- Lightweight RAG (unchanged) ------------------
def build_tfidf_index(full_text: str, *, max_features: int = 5000, ngram_range=(1,2)) -> Dict[str, Any]:
    sentences = [s.strip() for s in split_into_sentences(full_text) if len(s.strip()) > 30]
    if not sentences:
        sentences = [full_text.strip()] if isinstance(full_text, str) else []
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    tfidf = vectorizer.fit_transform(sentences) if sentences else None
    return {"sentences": sentences, "vectorizer": vectorizer, "tfidf_matrix": tfidf}
def rag_retrieve(index: Dict[str, Any], query: str, top_k: int = 5, window: int = 1, char_limit: int = 1200) -> str:
    try:
        if not index or not index.get("sentences") or index.get("tfidf_matrix") is None:
            return ""
        k = max(1, int(top_k))
        win = max(0, int(window))
        vec = index["vectorizer"].transform([query])
        sims = cosine_similarity(vec, index["tfidf_matrix"]).ravel()
        top_idx = sims.argsort()[::-1][:k]
        picked = set()
        spans: List[str] = []
        for i in top_idx:
            for j in range(max(0, i - win), min(len(index["sentences"]), i + win + 1)):
                if j not in picked:
                    picked.add(j)
                    spans.append(index["sentences"][j])
        context = "\n".join(spans)
        return context[:char_limit] if len(context) > char_limit else context
    except Exception:
        sentences = index.get("sentences", []) or []
        return "\n".join(sentences[:max(1, top_k)])[:char_limit]

# ----------------- Convenience: PDF -> (OCR) -> V2 adaptive ------------------
#David Zhu
def extract_concepts_from_pdf_v2(pdf_path: str) -> List[Dict[str, Any]]:
    """
    End-to-end convenience:
      PDF -> (native text or OCR fallback) -> slide-aware V2 adaptive extractor.
    """
    text = extract_text_from_pdf_ocr_fallback(pdf_path, dpi=300, lang="eng", min_chars_to_skip_ocr=1000)
    return extract_concepts_v2_adaptive(text, sim_base=0.68)

# ----------------- Public surface ------------------
__all__ = [
    # config
    "OPENAI_MODEL_CHAT", "OPENAI_MODEL_EMBED",
    # connectivity
    "_get_openai_client", "test_openai_connection",
    # extractors
    "extract_text_from_pdf", "extract_text_from_pdf_ocr_fallback",
    "split_into_sentences", "split_into_segments_slide",
    "get_embeddings", "cluster_embeddings", "pick_representatives",
    "explain_concept", "heuristic_difficulty",
    "extract_concepts_via_embedding",
    "extract_concepts_v2_adaptive", "extract_concepts_from_pdf_v2",
    # rag
    "build_tfidf_index", "rag_retrieve",
]
