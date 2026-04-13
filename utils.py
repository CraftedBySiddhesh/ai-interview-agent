"""
utils.py — Text extraction, sanitisation, and helpers.

Responsibilities (FR-011, FR-012, FR-013, FR-014, FR-028, FR-062, FR-065, FR-083):
  - extract_text(file, filename) — PDF and Word extraction
  - sanitise_text(text)         — null bytes, injection patterns, whitespace, truncation
  - is_coding_question(tag, text, role_info) — role-aware coding question detection
  - validate_python_syntax(code) — ast.parse check
  - parse_tag_and_content(text)  — split [TAG] from question body
  - build_transcript(name, qa_log, scorecard) — .txt download content
"""

import ast
import logging
import re
from io import BytesIO
from typing import Optional

from config import MAX_DOC_CHARS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coding-question detection keywords (FR-062)
# ---------------------------------------------------------------------------
CODING_KEYWORDS: set[str] = {
    "code", "write", "implement", "function", "algorithm", "debug",
    "fix", "class", "script", "program", "method", "loop", "recursion",
    "sort", "search", "data structure", "complexity", "runtime",
    "output", "print", "return", "variable", "syntax",
}

# ---------------------------------------------------------------------------
# FR-011, FR-012 — Text extraction
# ---------------------------------------------------------------------------
def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from a PDF or DOCX file.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to detect format).

    Returns:
        Extracted text string (may be empty on failure — caller handles FR-015).

    Raises:
        ValueError: If the file format is not PDF or DOCX.
    """
    lower = filename.lower()

    if lower.endswith(".pdf"):
        return _extract_pdf(file_bytes, filename)
    elif lower.endswith(".docx"):
        return _extract_docx(file_bytes, filename)
    else:
        raise ValueError(
            f"Unsupported file format: '{filename}'. Only PDF and DOCX are accepted."
        )


def _extract_pdf(file_bytes: bytes, filename: str) -> str:
    """Extract text from a PDF using pypdf (FR-011)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        extracted = "\n".join(pages)
        logger.info("PDF extraction complete: '%s' — %d chars", filename, len(extracted))
        return extracted
    except Exception as exc:
        logger.error("PDF extraction failed for '%s': %s", filename, exc)
        return ""


def _extract_docx(file_bytes: bytes, filename: str) -> str:
    """Extract text from a Word document using python-docx (FR-012)."""
    try:
        import docx

        doc = docx.Document(BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        extracted = "\n".join(paragraphs)
        logger.info("DOCX extraction complete: '%s' — %d chars", filename, len(extracted))
        return extracted
    except Exception as exc:
        logger.error("DOCX extraction failed for '%s': %s", filename, exc)
        return ""


# ---------------------------------------------------------------------------
# FR-013, FR-014 — Sanitisation
# ---------------------------------------------------------------------------
def sanitise_text(text: str) -> str:
    """
    Sanitise extracted document text before prompt injection.

    Steps:
      1. Remove null bytes
      2. Redact prompt injection patterns
      3. Collapse 4+ consecutive blank lines to 2
      4. Strip leading/trailing whitespace
      5. Truncate to MAX_DOC_CHARS

    Args:
        text: Raw extracted text.

    Returns:
        Sanitised, truncated text.
    """
    if not text:
        return ""

    _injection_patterns: list[re.Pattern] = [
        re.compile(r"ignore (all )?previous instructions?", re.IGNORECASE),
        re.compile(r"disregard (all )?previous", re.IGNORECASE),
        re.compile(r"you are now", re.IGNORECASE),
        re.compile(r"act as", re.IGNORECASE),
        re.compile(r"<\|.*?\|>"),         # token boundary injection
        re.compile(r"\[INST\]|\[/INST\]"),
        re.compile(r"###\s*System"),
        re.compile(r"###\s*Instruction"),
    ]

    # 1 — Remove null bytes
    text = text.replace("\x00", "")

    # 2 — Redact injection patterns
    for pattern in _injection_patterns:
        text = pattern.sub("[REDACTED]", text)

    # 3 — Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n", text)

    # 4 — Strip
    text = text.strip()

    # 5 — Truncate (FR-014)
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS]
        logger.info("Document truncated to %d chars.", MAX_DOC_CHARS)

    return text


# ---------------------------------------------------------------------------
# FR-062 — Coding question detection (role-aware, v1.1)
# ---------------------------------------------------------------------------
def is_coding_question(tag: str, question_text: str, role_info: Optional[dict] = None) -> bool:
    """
    Determine whether a question requires a code editor.

    A question is a coding question only when:
      - The category tag is [Technical], AND
      - The question text contains coding keywords, AND
      - The role is relevant to coding (derived from JD — role-agnostic)

    For non-engineering roles, coding questions are omitted unless the JD
    specifies technical/coding skills (FR-042, FR-062).

    Args:
        tag:           Category tag string e.g. '[Technical]'
        question_text: Full question text from the LLM.
        role_info:     dict from analyze_role — contains 'core_skills', 'domain'.
                       Pass None to skip role-awareness check.

    Returns:
        True if a code editor should be shown.
    """
    if tag.strip().upper() != "[TECHNICAL]":
        return False

    lower_q = question_text.lower()
    has_coding_keyword = any(kw in lower_q for kw in CODING_KEYWORDS)
    if not has_coding_keyword:
        return False

    # Role-awareness check — if role_info available, verify coding is relevant
    if role_info:
        core_skills = " ".join(role_info.get("core_skills", [])).lower()
        domain = role_info.get("domain", "").lower()
        # Consider coding relevant if domain or skills mention programming terms
        coding_relevant_terms = {
            "engineer", "developer", "software", "data", "ml", "machine learning",
            "backend", "frontend", "fullstack", "devops", "cloud", "python",
            "javascript", "java", "typescript", "go", "sql", "coding", "programming",
        }
        role_text = core_skills + " " + domain
        if not any(term in role_text for term in coding_relevant_terms):
            logger.info("Coding question suppressed — role not coding-relevant.")
            return False

    return True


# ---------------------------------------------------------------------------
# FR-065 — Python syntax validation
# ---------------------------------------------------------------------------
def validate_python_syntax(code: str) -> tuple[bool, str]:
    """
    Validate Python code syntax using ast.parse.

    Args:
        code: Python source code string.

    Returns:
        (is_valid, error_message) — error_message is empty string if valid.
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        msg = f"Syntax error on line {exc.lineno}: {exc.msg}"
        logger.warning("Python syntax validation failed: %s", msg)
        return False, msg


# ---------------------------------------------------------------------------
# FR-034 — Parse tag and content from LLM question output
# ---------------------------------------------------------------------------
def parse_tag_and_content(text: str) -> tuple[str, str]:
    """
    Extract the category tag and question body from LLM output.

    Expected format: '[Technical] How would you implement...'

    Args:
        text: Raw LLM question output.

    Returns:
        (tag, content) — tag is e.g. '[Technical]', content is the rest.
        If no tag found, returns ('[General]', original_text).
    """
    pattern = re.compile(r"^(\[(Technical|Behavioural|Situational|General)\])\s*", re.IGNORECASE)
    match = pattern.match(text.strip())
    if match:
        tag = match.group(1)
        content = text.strip()[match.end():]
        return tag, content
    logger.warning("No category tag found in LLM output — defaulting to [General].")
    return "[General]", text.strip()


# ---------------------------------------------------------------------------
# FR-083, FR-084, FR-085 — Transcript builder
# ---------------------------------------------------------------------------
def build_transcript(
    candidate_name: str,
    qa_log: list[dict],
    scorecard=None,
) -> str:
    """
    Build a plain-text interview transcript for download.

    Structure:
      - Scorecard (all 6 fields) at the top
      - All Q&A pairs with per-answer timing and quality scores

    Args:
        candidate_name: Candidate's name.
        qa_log:         List of QA log dicts per answer.
        scorecard:      Pydantic Scorecard object (optional).

    Returns:
        Multi-line string suitable for .txt download.
    """
    lines: list[str] = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("AI INTERVIEW AGENT — TRANSCRIPT")
    lines.append(f"Candidate: {candidate_name}")
    lines.append(sep)

    # FR-084 — Scorecard at top
    if scorecard:
        lines.append("\nSCORECARD")
        lines.append("-" * 40)
        lines.append(f"Fit Rating:       {scorecard.fit_rating}/10")
        lines.append(f"Fit Justification: {scorecard.fit_justification}")
        lines.append(f"Overall Summary:  {scorecard.overall_summary}")
        lines.append(f"Hint Dependency:  {scorecard.hint_dependency}")
        lines.append("\nStrengths:")
        for s in scorecard.strengths:
            lines.append(f"  • {s}")
        lines.append("\nGaps:")
        for g in scorecard.gaps:
            lines.append(f"  • {g}")
        lines.append("")

    # FR-085 — Q&A pairs with scores and timing
    lines.append(sep)
    lines.append("INTERVIEW LOG")
    lines.append(sep)

    for i, entry in enumerate(qa_log, start=1):
        lines.append(f"\nQ{i} [{entry.get('tag', 'General')}]: {entry.get('question', '')}")
        lines.append(f"Answer: {entry.get('answer', '')}")
        lines.append(f"Time: {entry.get('time_seconds', 0):.1f}s")

        if entry.get("is_code"):
            cs = entry.get("code_scores", {})
            lines.append(
                f"Code Quality — Correctness: {cs.get('correctness', '-')}/5  "
                f"Efficiency: {cs.get('efficiency', '-')}/5  "
                f"Readability: {cs.get('readability', '-')}/5"
            )
        else:
            lines.append(
                f"Quality — Clarity: {entry.get('clarity', '-')}/5  "
                f"Depth: {entry.get('depth', '-')}/5"
            )

    lines.append(f"\n{sep}")
    lines.append("End of Transcript")
    lines.append(sep)

    return "\n".join(lines)
