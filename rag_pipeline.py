"""
RAG-Based Domain-Specific AI Assistant
Domain: College / Student Information
Documents: Fee Structure, Attendance & Academic Policy, Academic Calendar & Examination Rules

Pipeline (per assignment spec, Section 6):
Step 1  : Load PDFs
Step 2  : Extract text
Step 3  : Create chunks
Step 4  : Store metadata
Step 5  : Create chunk embeddings
Step 6  : Store/search embeddings
Step 7  : Embed question
Step 8  : Calculate similarity
Step 9  : Select top-k chunks
Step 10 : Build context
Step 11 : Ask LLM to answer from context   (OPTIONAL -- see Supported Libraries doc)
Step 12 : Display concise answer and source

Author: Sameer Shaik (Sunny)
"""

import os
import glob
import json
import numpy as np
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity

# SentenceTransformer requires internet access to huggingface.co to download
# the model weights on first run. Some sandboxed/offline environments block
# this. We try the real embedding model first; if it cannot be downloaded,
# we fall back to a TF-IDF vectorizer (scikit-learn) so the rest of the
# pipeline (chunking -> similarity -> retrieval -> grounding) can still be
# demonstrated end-to-end. In Google Colab, the SentenceTransformer path
# will work normally since Colab has unrestricted internet access.
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False



# Optional LLM generation -- only used if an OpenAI key is available.
# Per the "Supported Libraries" doc: "this should be treated as an optional
# generation component rather than a requirement for the basic retrieval
# demonstration."
try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "data"                      # folder containing the source PDFs
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_WORD_SIZE = 180                  # words per chunk
CHUNK_OVERLAP = 30                     # word overlap between consecutive chunks
TOP_K = 3                              # number of chunks retrieved per question
NOT_FOUND_THRESHOLD = 0.35             # below this cosine similarity -> "not found"
USE_LLM_GENERATION = bool(os.environ.get("OPENAI_API_KEY"))  # auto-detects key


# ---------------------------------------------------------------------------
# Step 1 & 2: Load PDFs and extract text (with page-level metadata)
# ---------------------------------------------------------------------------
def load_and_extract(data_dir):
    """
    Returns a list of dicts: {"source": filename, "page": page_num, "text": page_text}
    Extraction is done page-by-page so page metadata is preserved for traceability.
    """
    records = []
    pdf_paths = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in '{data_dir}'.")

    for path in pdf_paths:
        filename = os.path.basename(path)
        doc = fitz.open(path)
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text").strip()
            if text:
                records.append({
                    "source": filename,
                    "page": page_index + 1,
                    "text": text
                })
        doc.close()
    return records


# ---------------------------------------------------------------------------
# Step 3 & 4: Chunk text and attach metadata (source, page, chunk_id)
# ---------------------------------------------------------------------------
def chunk_text(page_records, chunk_size=CHUNK_WORD_SIZE, overlap=CHUNK_OVERLAP):
    """
    Word-based sliding-window chunking with overlap.
    chunk_size=180 words / overlap=30 words was chosen because:
      - these policy-style PDFs are short (300-500 words per page), so
        ~180-word chunks keep each logical section (e.g. one policy rule)
        intact in a single chunk rather than splitting mid-idea.
      - a 30-word overlap prevents a fact that sits on a chunk boundary
        (e.g. a rule and its example) from being separated during retrieval.
    """
    chunks = []
    chunk_id = 0
    for record in page_records:
        words = record["text"].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words)
            chunks.append({
                "chunk_id": chunk_id,
                "source": record["source"],
                "page": record["page"],
                "text": chunk_str
            })
            chunk_id += 1
            if end == len(words):
                break
            start = end - overlap  # slide window back by 'overlap' words
    return chunks


def load_embedding_model():
    """
    Returns (model, backend_name).
    Tries, in order:
      1. 'sbert'  -- sentence-transformers (required by the assignment spec;
                     used by default and in Google Colab, which has full
                     internet access).
      2. 'tfidf'  -- keyword-overlap fallback, used ONLY when sbert cannot be
                     downloaded (e.g. a network-restricted sandbox), purely
                     to keep the pipeline runnable end-to-end for validation.
    """
    if _SBERT_AVAILABLE:
        try:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            return model, "sbert"
        except Exception as e:
            print(f"  [warning] Could not load SentenceTransformer "
                  f"('{e.__class__.__name__}'). This happens only in "
                  f"network-restricted environments that block "
                  f"huggingface.co -- it will work normally in Google Colab. "
                  f"Falling back to TF-IDF for this run only.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(), "tfidf"


# ---------------------------------------------------------------------------
# Step 5 & 6: Create chunk embeddings and store them (in-memory NumPy array)
# ---------------------------------------------------------------------------
def embed_chunks(chunks, model, backend):
    texts = [c["text"] for c in chunks]
    if backend == "sbert":
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    else:  # tfidf
        embeddings = model.fit_transform(texts).toarray()
    return embeddings  # shape: (num_chunks, embedding_dim) -- acts as our vector store


# ---------------------------------------------------------------------------
# Step 7, 8, 9: Embed question, calculate similarity, select top-k chunks
# ---------------------------------------------------------------------------
def retrieve_top_k(question, model, backend, chunks, chunk_embeddings, k=TOP_K):
    if backend == "sbert":
        question_embedding = model.encode([question], convert_to_numpy=True)
    else:  # tfidf
        question_embedding = model.transform([question]).toarray()
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1][:k]

    results = []
    for idx in ranked_indices:
        results.append({
            "chunk": chunks[idx],
            "score": float(similarities[idx])
        })
    return results


# ---------------------------------------------------------------------------
# Step 10: Build context string from retrieved chunks
# ---------------------------------------------------------------------------
def build_context(retrieved):
    context_parts = []
    for r in retrieved:
        c = r["chunk"]
        context_parts.append(f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}")
    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Step 11: Ask LLM to answer from context (OPTIONAL generation step)
# ---------------------------------------------------------------------------
def generate_llm_answer(question, context):
    """Only called if OPENAI_API_KEY is set. Falls back gracefully otherwise."""
    client = OpenAI()
    system_prompt = (
        "You are a college assistant. Answer the question using ONLY the "
        "provided context. If the answer is not present in the context, "
        "reply exactly: 'Answer not found in the provided documents.' "
        "Be concise (1-3 sentences)."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def generate_extractive_answer(question, retrieved, best_score):
    """
    Fallback answer generator used when no LLM/API key is configured.
    Returns the highest-scoring chunk's text as the 'grounded answer',
    satisfying the assignment's core retrieval requirement without an LLM.
    """
    if best_score < NOT_FOUND_THRESHOLD:
        return "Answer not found in the provided documents."
    top_chunk = retrieved[0]["chunk"]
    return top_chunk["text"]


# ---------------------------------------------------------------------------
# Step 12: Full question-answering wrapper (display concise answer + source)
# ---------------------------------------------------------------------------
def answer_question(question, model, backend, chunks, chunk_embeddings, k=TOP_K, verbose=True):
    retrieved = retrieve_top_k(question, model, backend, chunks, chunk_embeddings, k=k)
    best_score = retrieved[0]["score"]
    context = build_context(retrieved)

    if best_score < NOT_FOUND_THRESHOLD:
        answer = "Answer not found in the provided documents."
        used_llm = False
    elif USE_LLM_GENERATION and OPENAI_SDK_AVAILABLE:
        try:
            answer = generate_llm_answer(question, context)
            used_llm = True
        except Exception as e:
            answer = generate_extractive_answer(question, retrieved, best_score)
            used_llm = False
    else:
        answer = generate_extractive_answer(question, retrieved, best_score)
        used_llm = False

    result = {
        "question": question,
        "answer": answer,
        "top_score": round(best_score, 4),
        "used_llm": used_llm,
        "sources": [
            {"source": r["chunk"]["source"], "page": r["chunk"]["page"],
             "score": round(r["score"], 4), "chunk_id": r["chunk"]["chunk_id"]}
            for r in retrieved
        ]
    }

    if verbose:
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"   [top similarity: {result['top_score']} | "
              f"generation: {'LLM' if used_llm else 'extractive/fallback'}]")
        for s in result["sources"]:
            print(f"   -> {s['source']} (page {s['page']}, chunk {s['chunk_id']}, "
                  f"score {s['score']})")

    return result


# ---------------------------------------------------------------------------
# Main / Demo
# ---------------------------------------------------------------------------
DEMO_QUESTIONS = [
    # 5 answerable questions (direct wording)
    "What is the minimum attendance percentage required for exams?",
    "What is the tuition fee for B.Tech Computer Science per semester?",
    "When does the Semester I end-semester examination take place?",
    "What is the refund percentage if admission is cancelled before the semester begins?",
    "What grade is awarded for marks between 70 and 79?",
    # 2 deliberately unanswerable questions
    "What is the hostel curfew time?",
    "Is there a fee waiver for sports achievements?",
    # 1 semantic-search example (wording differs from the document)
    "How many classes can I skip and still be allowed to sit for exams?",
]


def main():
    print("=" * 70)
    print("RAG-Based Domain-Specific AI Assistant -- College Info Domain")
    print("=" * 70)

    print("\n[Step 1-2] Loading PDFs and extracting text...")
    page_records = load_and_extract(DATA_DIR)
    print(f"  Extracted {len(page_records)} page(s) from "
          f"{len(set(r['source'] for r in page_records))} document(s).")

    print("\n[Step 3-4] Chunking text with metadata...")
    chunks = chunk_text(page_records)
    print(f"  Created {len(chunks)} chunks "
          f"(chunk_size={CHUNK_WORD_SIZE} words, overlap={CHUNK_OVERLAP} words).")

    print(f"\n[Step 5-6] Loading embedding model '{EMBEDDING_MODEL_NAME}' "
          f"and embedding chunks...")
    model, backend = load_embedding_model()
    chunk_embeddings = embed_chunks(chunks, model, backend)
    print(f"  Embedded {chunk_embeddings.shape[0]} chunks into "
          f"{chunk_embeddings.shape[1]}-dim vectors using backend='{backend}' "
          f"(stored in NumPy array).")

    print(f"\n[Step 7-12] Running {len(DEMO_QUESTIONS)} demo questions "
          f"(LLM generation: {'ON' if USE_LLM_GENERATION else 'OFF -- extractive fallback'})...")

    results = []
    for q in DEMO_QUESTIONS:
        results.append(answer_question(q, model, backend, chunks, chunk_embeddings))

    # Save results table for the report
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results table to outputs/results.json")


if __name__ == "__main__":
    main()
