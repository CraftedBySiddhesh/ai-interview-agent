"""
rag.py — Vectorless RAG pipeline using BM25 keyword retrieval.

Responsibilities (FR-024 to FR-030):
  - chunk_text()             — split document into overlapping character-level chunks
  - tokenize_query()         — whitespace/punctuation tokeniser (no ML model)
  - build_bm25_index()       — build BM25Okapi index from chunk list
  - retrieve_context_bm25()  — top-K chunks by BM25 score against a query

Key v1.1 change: FAISS and sentence-transformers REMOVED.
Retrieval is pure Python — rank-bm25 (~20KB), no model download.
Interface (retrieve top chunks per turn) is unchanged from the rest of the app.
"""

import logging
import re
from typing import Optional

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FR-024, FR-025 — Document chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping character-level chunks.

    Mimics LangChain's RecursiveCharacterTextSplitter behaviour:
    prefer splitting on double-newlines, then single newlines, then spaces,
    then characters — ensuring chunks respect natural paragraph boundaries
    where possible.

    Args:
        text:       Input document text.
        chunk_size: Maximum characters per chunk (default 500 — FR-024/025).
        overlap:    Characters of overlap between consecutive chunks (default 50).

    Returns:
        List of text chunk strings.
    """
    if not text or not text.strip():
        logger.warning("chunk_text received empty text — returning empty list.")
        return []

    # Recursive split: try preferred separators in order
    separators = ["\n\n", "\n", " ", ""]
    chunks = _recursive_split(text.strip(), chunk_size, overlap, separators)
    logger.info("chunk_text produced %d chunks (size=%d, overlap=%d).", len(chunks), chunk_size, overlap)
    return chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: list[str],
) -> list[str]:
    """Recursively split text using the best available separator."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0]
    remaining_separators = separators[1:]

    if sep == "":
        # Character-level fallback
        return _char_split(text, chunk_size, overlap)

    parts = text.split(sep)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).lstrip(sep) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                if len(current) > chunk_size and remaining_separators:
                    chunks.extend(_recursive_split(current, chunk_size, overlap, remaining_separators))
                else:
                    chunks.append(current)
                # Carry overlap into next chunk
                current = current[-overlap:] + sep + part if overlap > 0 else part
            else:
                current = part

    if current.strip():
        if len(current) > chunk_size and remaining_separators:
            chunks.extend(_recursive_split(current, chunk_size, overlap, remaining_separators))
        else:
            chunks.append(current)

    return chunks


def _char_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Hard character-level split with overlap as last resort."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# FR-026 — BM25 tokeniser (no ML embeddings)
# ---------------------------------------------------------------------------
def tokenize_query(text: str) -> list[str]:
    """
    Tokenise text for BM25 indexing using whitespace and punctuation splitting.

    No ML model, no embeddings — pure string operations (FR-026, v1.1).

    Args:
        text: Input string (chunk or candidate answer).

    Returns:
        List of lowercase tokens.
    """
    # Split on whitespace and common punctuation
    tokens = re.split(r"[\s\.,;:!?()\[\]{}'\"\\/<>|=+\-*&#@%$^~`]+", text.lower())
    # Remove empty strings and very short tokens (likely noise)
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


# ---------------------------------------------------------------------------
# FR-027 — Build BM25 index
# ---------------------------------------------------------------------------
def build_bm25_index(chunks: list[str]):
    """
    Build an in-memory BM25Okapi index from a list of text chunks.

    Uses rank-bm25 — pure Python, ~20KB, no vector arithmetic (FR-027, v1.1).

    Args:
        chunks: List of text chunks from chunk_text().

    Returns:
        BM25Okapi index object, or None if chunks is empty.
    """
    if not chunks:
        logger.warning("build_bm25_index received empty chunk list — returning None.")
        return None

    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [tokenize_query(chunk) for chunk in chunks]
        index = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built: %d chunks indexed.", len(chunks))
        return index
    except Exception as exc:
        logger.error("BM25 index build failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# FR-028 — Per-turn retrieval
# ---------------------------------------------------------------------------
def retrieve_context_bm25(
    query: str,
    index,
    chunks: list[str],
    top_k: int = 3,
) -> list[str]:
    """
    Retrieve the top-K most relevant chunks for a query using BM25 scoring.

    Called every conversation turn with the candidate's latest answer as query.
    Returns top chunks to be injected into the system prompt (FR-028, FR-029).

    Args:
        query:  Candidate's answer or current question (used as retrieval query).
        index:  BM25Okapi index built by build_bm25_index().
        chunks: Original chunk list (parallel to index corpus).
        top_k:  Number of chunks to return.

    Returns:
        List of up to top_k chunk strings ordered by BM25 score (best first).
        Returns empty list if index is None or retrieval fails.
    """
    if index is None:
        logger.warning("retrieve_context_bm25 called with None index — skipping.")
        return []

    if not chunks:
        return []

    try:
        query_tokens = tokenize_query(query)
        if not query_tokens:
            logger.warning("Empty query tokens — returning empty context.")
            return []

        scores = index.get_scores(query_tokens)
        # Pair scores with chunks and sort descending
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_chunks = [chunks[i] for i, score in scored[:top_k] if score > 0]

        logger.info(
            "BM25 retrieval: query '%s…' → %d chunks returned (top score=%.3f).",
            query[:40],
            len(top_chunks),
            scored[0][1] if scored else 0.0,
        )
        return top_chunks
    except Exception as exc:
        logger.error("BM25 retrieval failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Convenience: build both JD and CV indexes in one call
# ---------------------------------------------------------------------------
def build_all_indexes(
    jd_text: str,
    cv_text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[Optional[object], list[str], Optional[object], list[str]]:
    """
    Chunk JD and CV texts and build their respective BM25 indexes.

    Returns:
        (jd_index, jd_chunks, cv_index, cv_chunks)
        Any index may be None if the corresponding text was empty or build failed.
    """
    jd_chunks = chunk_text(jd_text, chunk_size, overlap)
    cv_chunks = chunk_text(cv_text, chunk_size, overlap)

    jd_index = build_bm25_index(jd_chunks)
    cv_index = build_bm25_index(cv_chunks)

    logger.info(
        "build_all_indexes complete — JD: %d chunks, CV: %d chunks.",
        len(jd_chunks),
        len(cv_chunks),
    )
    return jd_index, jd_chunks, cv_index, cv_chunks
