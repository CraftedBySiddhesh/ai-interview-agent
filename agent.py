"""
agent.py — All AI-powered functions. No Streamlit imports.

Responsibilities (FR-017 to FR-023, FR-034, FR-044, FR-047, FR-055 to FR-059, FR-062, FR-075):
  - analyze_role()              — extract 7-field role profile from JD (FR-017)
  - detect_cv_gaps()            — compare JD vs CV, return gap dict (FR-018)
  - search_trends()             — Tavily trend search using extracted role/domain (FR-020)
  - get_quality_scores()        — score an answer on Clarity+Depth (FR-034)
  - get_code_quality_scores()   — score code on Correctness+Efficiency+Readability (FR-067)
  - generate_scorecard()        — build Pydantic Scorecard from QA log (FR-062)
  - safe_model_invoke()         — token guard + error handling wrapper (FR-075)
  - trim_messages_if_needed()   — trim oldest messages at 6000-word threshold (FR-075)
  - check_adaptive_difficulty() — evaluate last 3 scores for escalation (FR-055)

Zero Streamlit imports. All functions are pure Python, importable into FastAPI (Phase 2).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FR-017 — Role analysis
# ---------------------------------------------------------------------------
def analyze_role(jd_text: str) -> dict:
    """
    Analyse a job description and extract a structured role profile.

    Uses ChatPromptTemplate + JsonOutputParser LCEL chain (FR-070).
    All fields are derived dynamically from the JD — no hardcoded role assumptions.

    Args:
        jd_text: Sanitised JD text string.

    Returns:
        dict with keys: role_title, seniority_level, core_skills,
                        nice_to_have_skills, key_responsibilities,
                        domain, interview_focus_areas.
        Returns empty dict on failure (non-blocking — FR-023).
    """
    from prompts import get_analyze_role_chain

    try:
        chain = get_analyze_role_chain()
        result = chain.invoke({"jd_text": jd_text})
        logger.info(
            "analyze_role complete: role='%s', domain='%s', skills=%s",
            result.get("role_title", "?"),
            result.get("domain", "?"),
            result.get("core_skills", []),
        )
        return result
    except Exception as exc:
        logger.error("analyze_role failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# FR-018 — CV gap detection
# ---------------------------------------------------------------------------
def detect_cv_gaps(jd_text: str, cv_text: str, core_skills: list[str]) -> dict:
    """
    Compare JD against CV and identify skill gaps.

    Uses ChatPromptTemplate + JsonOutputParser LCEL chain (FR-071).

    Args:
        jd_text:     Sanitised JD text.
        cv_text:     Sanitised CV text.
        core_skills: List of core skills extracted by analyze_role().

    Returns:
        dict with keys: missing_skills, weak_areas, strengths_match, gap_summary.
        Returns empty dict on failure (non-blocking — FR-023).
    """
    from prompts import get_detect_cv_gaps_chain

    try:
        chain = get_detect_cv_gaps_chain()
        result = chain.invoke({
            "jd_text": jd_text,
            "cv_text": cv_text,
            "core_skills": ", ".join(core_skills) if core_skills else "not specified",
        })
        logger.info(
            "detect_cv_gaps complete: missing=%s, weak=%s",
            result.get("missing_skills", []),
            result.get("weak_areas", []),
        )
        return result
    except Exception as exc:
        logger.error("detect_cv_gaps failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# FR-020 — Tavily trend search
# ---------------------------------------------------------------------------
def search_trends(role_title: str, domain: str) -> str:
    """
    Search for latest industry trends relevant to the role using Tavily.

    Query is built dynamically from the extracted role_title and domain,
    making it fully role-agnostic (FR-020, FR-044, v1.1).

    Args:
        role_title: Role title extracted by analyze_role().
        domain:     Domain extracted by analyze_role().

    Returns:
        Concatenated trend snippet string (up to 5 results × 300 chars).
        Returns empty string on failure (non-blocking — FR-023).
    """
    from config import get_tavily, TAVILY_MAX_RESULTS, TAVILY_SNIPPET_CHARS

    if not role_title and not domain:
        logger.warning("search_trends called with empty role_title and domain — skipping.")
        return ""

    query = f"latest trends {role_title} {domain} 2024 2025".strip()

    try:
        client = get_tavily()
        response = client.search(
            query=query,
            max_results=TAVILY_MAX_RESULTS,
            search_depth="basic",
        )
        results = response.get("results", [])
        snippets = []
        for r in results:
            content = r.get("content", "")
            if content:
                snippets.append(content[:TAVILY_SNIPPET_CHARS])

        combined = "\n---\n".join(snippets)
        logger.info(
            "search_trends complete: query='%s', %d results returned.", query, len(snippets)
        )
        return combined
    except Exception as exc:
        logger.error("search_trends failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# FR-034 — Answer quality scoring
# ---------------------------------------------------------------------------
def get_quality_scores(question: str, answer: str) -> tuple[int, int]:
    """
    Score a candidate's text answer on Clarity and Depth.

    Uses ChatPromptTemplate + JsonOutputParser LCEL chain (FR-072).

    Args:
        question: The question that was asked.
        answer:   The candidate's answer.

    Returns:
        (clarity, depth) — both integers in range 1-5.
        Returns (3, 3) as neutral fallback on failure.
    """
    from prompts import get_quality_scores_chain

    try:
        chain = get_quality_scores_chain()
        result = chain.invoke({"question": question, "answer": answer})
        clarity = int(result.get("clarity", 3))
        depth = int(result.get("depth", 3))
        # Clamp to valid range
        clarity = max(1, min(5, clarity))
        depth = max(1, min(5, depth))
        logger.info("get_quality_scores: clarity=%d, depth=%d", clarity, depth)
        return clarity, depth
    except Exception as exc:
        logger.error("get_quality_scores failed: %s — returning neutral (3, 3).", exc)
        return 3, 3


# ---------------------------------------------------------------------------
# FR-067 — Code quality scoring
# ---------------------------------------------------------------------------
def get_code_quality_scores(
    question: str,
    code: str,
    language: str = "Python",
) -> dict:
    """
    Score a candidate's code submission on Correctness, Efficiency, Readability.

    Uses ChatPromptTemplate + JsonOutputParser LCEL chain (FR-073).

    Args:
        question: The coding question that was asked.
        code:     Candidate's submitted code.
        language: Programming language (default Python).

    Returns:
        dict with keys: correctness, efficiency, readability (all int 1-5).
        Returns {correctness:3, efficiency:3, readability:3} on failure.
    """
    from prompts import build_code_quality_chain

    default = {"correctness": 3, "efficiency": 3, "readability": 3}

    try:
        chain = build_code_quality_chain(language=language)
        result = chain.invoke({"question": question, "code": code, "language": language})
        scores = {
            "correctness": max(1, min(5, int(result.get("correctness", 3)))),
            "efficiency": max(1, min(5, int(result.get("efficiency", 3)))),
            "readability": max(1, min(5, int(result.get("readability", 3)))),
        }
        logger.info("get_code_quality_scores: %s", scores)
        return scores
    except Exception as exc:
        logger.error("get_code_quality_scores failed: %s — returning neutral.", exc)
        return default


# ---------------------------------------------------------------------------
# FR-062 — Scorecard generation
# ---------------------------------------------------------------------------
def generate_scorecard(
    qa_log: list[dict],
    hints_used: int,
    max_hints: int,
    candidate_name: str,
) -> Optional[object]:
    """
    Generate a validated Pydantic Scorecard from the QA log.

    Builds a structured interview summary from the QA log and invokes the
    scorecard LCEL chain with PydanticOutputParser (FR-074).

    Args:
        qa_log:         List of QA log entries, each with question, answer,
                        tag, clarity, depth, time_seconds, is_code, etc.
        hints_used:     Total hints used during the interview.
        max_hints:      Maximum hints allowed.
        candidate_name: Candidate's name.

    Returns:
        Validated Scorecard object, or None on failure (FR-081).
    """
    from prompts import get_scorecard_chain

    try:
        # Build a concise interview summary from the QA log
        summary_lines = [f"Interview for: {candidate_name}", f"Total questions answered: {len(qa_log)}"]
        for i, entry in enumerate(qa_log, start=1):
            tag = entry.get("tag", "General")
            question = entry.get("question", "")[:200]
            answer = entry.get("answer", "")[:300]
            if entry.get("is_code"):
                cs = entry.get("code_scores", {})
                scores_str = (
                    f"Code scores — Correctness:{cs.get('correctness', '-')}/5 "
                    f"Efficiency:{cs.get('efficiency', '-')}/5 "
                    f"Readability:{cs.get('readability', '-')}/5"
                )
            else:
                scores_str = (
                    f"Quality — Clarity:{entry.get('clarity', '-')}/5 "
                    f"Depth:{entry.get('depth', '-')}/5"
                )
            summary_lines.append(
                f"\nQ{i} [{tag}]: {question}\nAnswer: {answer}\n{scores_str}"
            )

        interview_summary = "\n".join(summary_lines)

        chain = get_scorecard_chain()
        scorecard = chain.invoke({
            "interview_summary": interview_summary,
            "hints_used": hints_used,
            "max_hints": max_hints,
        })
        logger.info(
            "generate_scorecard complete: fit_rating=%s, strengths=%d, gaps=%d",
            scorecard.fit_rating,
            len(scorecard.strengths),
            len(scorecard.gaps),
        )
        return scorecard
    except Exception as exc:
        logger.error("generate_scorecard failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# FR-075 — Token guard: message trimming
# ---------------------------------------------------------------------------
def trim_messages_if_needed(messages: list, max_words: int, keep_messages: int) -> list:
    """
    Trim oldest conversation messages when total word count exceeds the threshold.

    Preserves:
      - The first message (SystemMessage)
      - The last `keep_messages` messages

    Args:
        messages:     List of LangChain message objects.
        max_words:    Word count threshold triggering trim (default 6000 — FR-075).
        keep_messages: Number of recent messages to preserve after SystemMessage.

    Returns:
        Trimmed message list (or original if under threshold).
    """
    if not messages:
        return messages

    total_words = sum(len(m.content.split()) for m in messages)

    if total_words <= max_words:
        return messages

    logger.warning(
        "Token guard triggered: %d words > %d limit. Trimming to last %d messages.",
        total_words,
        max_words,
        keep_messages,
    )

    system_message = messages[0]
    recent_messages = messages[-(keep_messages):]

    trimmed = [system_message] + recent_messages
    new_total = sum(len(m.content.split()) for m in trimmed)
    logger.info("After trim: %d words in %d messages.", new_total, len(trimmed))
    return trimmed


# ---------------------------------------------------------------------------
# FR-075 — Safe model invocation (token guard + error handling)
# ---------------------------------------------------------------------------
def safe_model_invoke(messages: list) -> Optional[str]:
    """
    Invoke the Mistral model safely with token guard and full error handling.

    Steps:
      1. Trim messages if word count exceeds TOKEN_GUARD_WORDS
      2. Invoke ChatMistralAI
      3. Return response content string

    Args:
        messages: List of LangChain message objects (system + conversation history).

    Returns:
        Model response content string, or None on failure.
    """
    from config import get_model, TOKEN_GUARD_WORDS, TOKEN_GUARD_KEEP_MESSAGES

    try:
        # FR-075 — Token guard
        messages = trim_messages_if_needed(messages, TOKEN_GUARD_WORDS, TOKEN_GUARD_KEEP_MESSAGES)

        model = get_model()
        response = model.invoke(messages)
        content = response.content
        logger.info("safe_model_invoke: response length=%d chars.", len(content))
        return content
    except Exception as exc:
        logger.error("safe_model_invoke failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# FR-055, FR-056, FR-057, FR-059 — Adaptive difficulty check
# ---------------------------------------------------------------------------
def check_adaptive_difficulty(
    qa_log: list[dict],
    current_difficulty: str,
    already_escalated: bool,
) -> bool:
    """
    Evaluate whether to escalate interview difficulty.

    Triggers when:
      - The average (clarity + depth) / 2 over the last 3 answers >= 4.0 (FR-056)
      - Current difficulty is not already 'Hard' (FR-057)
      - Escalation has not already happened this session (FR-059)

    Args:
        qa_log:            List of QA log entries.
        current_difficulty: Current difficulty setting.
        already_escalated:  True if escalation already triggered this session.

    Returns:
        True if adaptive escalation should be triggered, False otherwise.
    """
    from config import ADAPTIVE_AVG_THRESHOLD, ADAPTIVE_LOOKBACK

    if already_escalated:
        return False
    if current_difficulty == "Hard":
        return False
    if len(qa_log) < ADAPTIVE_LOOKBACK:
        return False

    recent = qa_log[-ADAPTIVE_LOOKBACK:]
    scores = []
    for entry in recent:
        if entry.get("is_code"):
            cs = entry.get("code_scores", {})
            avg = (cs.get("correctness", 3) + cs.get("efficiency", 3) + cs.get("readability", 3)) / 3
        else:
            avg = (entry.get("clarity", 3) + entry.get("depth", 3)) / 2
        scores.append(avg)

    avg_score = sum(scores) / len(scores)
    should_escalate = avg_score >= ADAPTIVE_AVG_THRESHOLD

    if should_escalate:
        logger.info(
            "Adaptive difficulty triggered: avg_score=%.2f >= %.1f over last %d answers.",
            avg_score,
            ADAPTIVE_AVG_THRESHOLD,
            ADAPTIVE_LOOKBACK,
        )
    return should_escalate
