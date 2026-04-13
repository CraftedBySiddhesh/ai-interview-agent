"""
app.py — Streamlit UI. All st.* calls live here.

Responsibilities (FR-001 to FR-016, FR-071, FR-082):
  - Setup widgets: candidate name, interview type, difficulty, coding language,
    JD upload, CV upload, API key inputs, custom questions (FR-002 to FR-009)
  - Pre-interview checklist (FR-010)
  - 4-step startup sequence with spinners (FR-015, FR-016, FR-024, FR-017)
  - Session state initialisation
  - View routing: setup → interview → scorecard

Zero logic: no AI calls, no chunking, no retrieval in this file.
Zero imports of st.* outside this file (agent, rag, utils, prompts, config are pure Python).
"""

import os
import time

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — MUST be the very first Streamlit call (FR-001)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Interview Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Internal imports — unidirectional dependency (app → agent/rag/utils/prompts/config)
# No circular imports. No Streamlit in any of these modules.
# ---------------------------------------------------------------------------
from config import (
    CODING_LANGUAGES,
    DIFFICULTY_LEVELS,
    INTERVIEW_TYPES,
    MAX_HINTS,
    TOTAL_QUESTIONS,
    validate_env_keys,
)
from utils import extract_text, sanitise_text
from rag import build_all_indexes
from agent import analyze_role, detect_cv_gaps, search_trends
from prompts import build_base_system_prompt


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """
    Initialise all session state keys to safe defaults on first load.

    Groups:
      - Routing       : view
      - Setup inputs  : candidate_name, interview_type, difficulty, coding_language,
                        custom_questions
      - Documents     : jd_text, cv_text
      - AI analysis   : role_info, gap_info, trend_snippets
      - BM25 indexes  : jd_index, jd_chunks, cv_index, cv_chunks
      - Interview     : messages, qa_log, question_count, hints_used,
                        interview_start_time, current_difficulty,
                        adaptive_escalated, system_prompt
      - Scorecard     : scorecard
    """
    defaults: dict = {
        # ---- Routing ----
        "view": "setup",                          # setup | interview | scorecard

        # ---- Setup inputs (FR-002 to FR-008) ----
        "candidate_name": "",
        "interview_type": INTERVIEW_TYPES[2],     # default: Mixed
        "difficulty": DIFFICULTY_LEVELS[1],       # default: Medium
        "coding_language": CODING_LANGUAGES[0],   # default: Python
        "custom_questions": [],

        # ---- Documents ----
        "jd_text": "",
        "cv_text": "",

        # ---- AI analysis results ----
        "role_info": {},
        "gap_info": {},
        "trend_snippets": "",

        # ---- BM25 indexes (FR-027) ----
        "jd_index": None,
        "jd_chunks": [],
        "cv_index": None,
        "cv_chunks": [],

        # ---- Interview state (used in PR5) ----
        "messages": [],
        "qa_log": [],
        "question_count": 0,
        "hints_used": 0,
        "interview_start_time": None,
        "current_difficulty": DIFFICULTY_LEVELS[1],
        "adaptive_escalated": False,
        "system_prompt": "",

        # ---- Scorecard (used in PR6) ----
        "scorecard": None,
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


# ---------------------------------------------------------------------------
# Checklist helpers (FR-010)
# ---------------------------------------------------------------------------
def _checklist_item(label: str, done: bool) -> None:
    """Render a single checklist row with a tick or empty box."""
    icon = "✅" if done else "⬜"
    st.markdown(f"{icon} &nbsp; {label}")


def _render_checklist(
    has_name: bool,
    has_jd: bool,
    has_cv: bool,
    has_api_keys: bool,
) -> bool:
    """
    Render the pre-interview readiness checklist (FR-010).

    Returns True when all four items are satisfied.
    """
    st.subheader("Pre-Interview Checklist")
    _checklist_item("Candidate name entered", has_name)
    _checklist_item("Job Description uploaded", has_jd)
    _checklist_item("CV / Résumé uploaded", has_cv)
    _checklist_item("API keys configured", has_api_keys)
    return all([has_name, has_jd, has_cv, has_api_keys])


# ---------------------------------------------------------------------------
# 4-step startup sequence (FR-015, FR-016, FR-013, FR-014, FR-027, FR-017)
# ---------------------------------------------------------------------------
def _run_startup_sequence(
    jd_bytes: bytes,
    jd_filename: str,
    cv_bytes: bytes,
    cv_filename: str,
) -> bool:
    """
    Execute the 4-step pre-interview startup sequence with progress feedback.

    Step 1 — Extract text from JD and CV              (FR-015, FR-016)
    Step 2 — Sanitise both documents                  (FR-013, FR-014)
    Step 3 — Build BM25 indexes for JD and CV         (FR-024, FR-025, FR-027)
    Step 4 — AI: analyse role, detect CV gaps, trends (FR-017, FR-018, FR-020)

    All steps must complete within 15 seconds in practice (FR-performance).

    Returns:
        True  — all steps succeeded; session state is populated.
        False — a critical step failed; error message already rendered.
    """
    progress = st.progress(0, text="Initialising startup sequence…")

    # ── Step 1: Extract ───────────────────────────────────────────────────
    with st.spinner("Step 1 / 4 — Extracting document text…"):
        try:
            jd_raw = extract_text(jd_bytes, jd_filename)
            cv_raw = extract_text(cv_bytes, cv_filename)
        except ValueError as exc:
            st.error(f"❌ {exc}")
            progress.empty()
            return False

        if not jd_raw:
            st.error(
                "❌ Could not extract text from the Job Description. "
                "Please re-upload a readable PDF or DOCX file."
            )
            progress.empty()
            return False

        if not cv_raw:
            st.error(
                "❌ Could not extract text from the CV. "
                "Please re-upload a readable PDF or DOCX file."
            )
            progress.empty()
            return False

    progress.progress(25, text="Step 1 / 4 — Text extracted ✓")
    time.sleep(0.2)

    # ── Step 2: Sanitise ──────────────────────────────────────────────────
    with st.spinner("Step 2 / 4 — Sanitising documents…"):
        jd_text = sanitise_text(jd_raw)
        cv_text = sanitise_text(cv_raw)
        st.session_state["jd_text"] = jd_text
        st.session_state["cv_text"] = cv_text

    progress.progress(50, text="Step 2 / 4 — Documents sanitised ✓")
    time.sleep(0.2)

    # ── Step 3: Build BM25 indexes ────────────────────────────────────────
    with st.spinner("Step 3 / 4 — Building BM25 retrieval indexes…"):
        jd_index, jd_chunks, cv_index, cv_chunks = build_all_indexes(jd_text, cv_text)
        st.session_state["jd_index"] = jd_index
        st.session_state["jd_chunks"] = jd_chunks
        st.session_state["cv_index"] = cv_index
        st.session_state["cv_chunks"] = cv_chunks

    progress.progress(75, text="Step 3 / 4 — BM25 indexes built ✓")
    time.sleep(0.2)

    # ── Step 4: AI analysis ───────────────────────────────────────────────
    with st.spinner("Step 4 / 4 — Analysing role and candidate profile…"):
        role_info = analyze_role(jd_text)
        core_skills = role_info.get("core_skills", []) if role_info else []

        gap_info = detect_cv_gaps(jd_text, cv_text, core_skills)

        trend_snippets = search_trends(
            role_title=role_info.get("role_title", "") if role_info else "",
            domain=role_info.get("domain", "") if role_info else "",
        )

        st.session_state["role_info"] = role_info
        st.session_state["gap_info"] = gap_info
        st.session_state["trend_snippets"] = trend_snippets

    progress.progress(100, text="Step 4 / 4 — AI analysis complete ✓")
    time.sleep(0.3)
    progress.empty()
    return True


# ---------------------------------------------------------------------------
# Setup view (FR-002 to FR-016)
# ---------------------------------------------------------------------------
def _render_setup_view() -> None:
    """
    Render the full interview setup page.

    Left column  — candidate details, document upload, API keys, custom questions.
    Right column — pre-interview checklist, session summary, Start Interview button.
    """
    # ── Page header ───────────────────────────────────────────────────────
    st.title("🎯 AI Interview Agent")
    st.caption(
        "Powered by **Mistral-Small-2506** · **BM25 Vectorless RAG** · **Tavily Search** "
        "· Phase 1 v1.1"
    )
    st.divider()

    left_col, right_col = st.columns([2, 1], gap="large")

    # ════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN — inputs
    # ════════════════════════════════════════════════════════════════════════
    with left_col:

        # ── API Keys (FR-009, FR-071) ─────────────────────────────────────
        keys_configured = bool(os.getenv("MISTRAL_API_KEY")) and bool(os.getenv("TAVILY_API_KEY"))
        with st.expander(
            "🔑 API Keys" + (" ✅" if keys_configured else " — required"),
            expanded=not keys_configured,
        ):
            mistral_input = st.text_input(
                "Mistral API Key",
                value=os.getenv("MISTRAL_API_KEY", ""),
                type="password",
                key="input_mistral_key",
                placeholder="Enter your Mistral API key",
                help="Required. Get yours at console.mistral.ai",
            )
            tavily_input = st.text_input(
                "Tavily API Key",
                value=os.getenv("TAVILY_API_KEY", ""),
                type="password",
                key="input_tavily_key",
                placeholder="Enter your Tavily API key",
                help="Required. Get yours at tavily.com",
            )
            if st.button("💾 Save API Keys", type="secondary", key="btn_save_keys"):
                if mistral_input.strip():
                    os.environ["MISTRAL_API_KEY"] = mistral_input.strip()
                if tavily_input.strip():
                    os.environ["TAVILY_API_KEY"] = tavily_input.strip()
                missing = validate_env_keys()
                if not missing:
                    st.success("✅ API keys saved successfully.")
                else:
                    st.warning(f"Still missing: {', '.join(missing)}")
                st.rerun()

        # ── Candidate Details ─────────────────────────────────────────────
        st.subheader("👤 Candidate Details")

        candidate_name = st.text_input(
            "Candidate Name",
            value=st.session_state["candidate_name"],
            placeholder="e.g. Alex Johnson",
            key="input_candidate_name",
            help="The candidate's full name — used throughout the interview.",
        )
        st.session_state["candidate_name"] = candidate_name.strip()

        col_type, col_diff, col_lang = st.columns(3)

        with col_type:
            # FR-003 — Interview type
            interview_type = st.selectbox(
                "Interview Type",
                options=INTERVIEW_TYPES,
                index=INTERVIEW_TYPES.index(st.session_state["interview_type"]),
                key="input_interview_type",
                help="Technical focuses on skills; Behavioural on STAR; Mixed balances both.",
            )
            st.session_state["interview_type"] = interview_type

        with col_diff:
            # FR-004 — Difficulty
            difficulty = st.selectbox(
                "Difficulty",
                options=DIFFICULTY_LEVELS,
                index=DIFFICULTY_LEVELS.index(st.session_state["difficulty"]),
                key="input_difficulty",
                help="Controls question complexity. Adaptive escalation can raise this mid-interview.",
            )
            st.session_state["difficulty"] = difficulty
            st.session_state["current_difficulty"] = difficulty

        with col_lang:
            # FR-005 — Coding language
            coding_language = st.selectbox(
                "Coding Language",
                options=CODING_LANGUAGES,
                index=CODING_LANGUAGES.index(st.session_state["coding_language"]),
                key="input_coding_language",
                help="Used when a coding question is appropriate for the role.",
            )
            st.session_state["coding_language"] = coding_language

        # ── Document Upload ───────────────────────────────────────────────
        st.subheader("📄 Documents")

        col_jd, col_cv = st.columns(2)

        with col_jd:
            # FR-006 — JD upload
            jd_file = st.file_uploader(
                "Job Description",
                type=["pdf", "docx"],
                key="upload_jd",
                help="Upload the Job Description as a PDF or DOCX. "
                     "The AI will extract the role, skills, and domain automatically.",
            )
            if jd_file:
                st.caption(f"📎 {jd_file.name}")

        with col_cv:
            # FR-007 — CV upload
            cv_file = st.file_uploader(
                "Candidate CV / Résumé",
                type=["pdf", "docx"],
                key="upload_cv",
                help="Upload the candidate's CV as a PDF or DOCX. "
                     "Used for gap analysis and tailored questioning.",
            )
            if cv_file:
                st.caption(f"📎 {cv_file.name}")

        # ── Custom Questions (FR-008) ─────────────────────────────────────
        with st.expander("➕ Custom Must-Ask Questions (optional)"):
            st.caption(
                "Add questions that must appear in the interview. "
                "Enter one question per line."
            )
            custom_raw = st.text_area(
                "Custom questions",
                value="\n".join(st.session_state.get("custom_questions", [])),
                height=110,
                placeholder=(
                    "e.g. Describe a time you led a team under a tight deadline.\n"
                    "e.g. What is your experience with distributed systems?"
                ),
                key="input_custom_questions",
                label_visibility="collapsed",
            )
            custom_questions = [q.strip() for q in custom_raw.splitlines() if q.strip()]
            st.session_state["custom_questions"] = custom_questions
            if custom_questions:
                st.caption(f"✅ {len(custom_questions)} custom question(s) will be included.")

    # ════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN — checklist + start button
    # ════════════════════════════════════════════════════════════════════════
    with right_col:

        # ── Checklist (FR-010) ────────────────────────────────────────────
        has_name = bool(st.session_state["candidate_name"])
        has_jd = jd_file is not None
        has_cv = cv_file is not None
        has_api_keys = bool(os.getenv("MISTRAL_API_KEY")) and bool(os.getenv("TAVILY_API_KEY"))

        all_ready = _render_checklist(has_name, has_jd, has_cv, has_api_keys)

        st.divider()

        # ── Session summary ───────────────────────────────────────────────
        st.caption("**Session Settings**")
        st.caption(f"Questions: **{TOTAL_QUESTIONS}**  ·  Max hints: **{MAX_HINTS}**")
        if st.session_state["candidate_name"]:
            st.caption(f"Candidate: **{st.session_state['candidate_name']}**")
        st.caption(
            f"Type: **{st.session_state['interview_type']}**  ·  "
            f"Difficulty: **{st.session_state['difficulty']}**"
        )
        st.caption(f"Language: **{st.session_state['coding_language']}**")

        st.divider()

        # ── Start Interview button ────────────────────────────────────────
        start_clicked = st.button(
            "🚀  Start Interview",
            type="primary",
            disabled=not all_ready,
            use_container_width=True,
            help=(
                "Complete all checklist items to launch."
                if not all_ready
                else "Everything is ready — launch the interview!"
            ),
            key="btn_start_interview",
        )

        if not all_ready:
            missing_items = []
            if not has_name:
                missing_items.append("candidate name")
            if not has_jd:
                missing_items.append("Job Description")
            if not has_cv:
                missing_items.append("CV")
            if not has_api_keys:
                missing_items.append("API keys")
            if missing_items:
                st.caption(f"⚠️ Missing: {', '.join(missing_items)}")

        # ── Startup sequence (triggered on button click) ──────────────────
        if start_clicked and all_ready:
            st.divider()
            success = _run_startup_sequence(
                jd_bytes=jd_file.read(),
                jd_filename=jd_file.name,
                cv_bytes=cv_file.read(),
                cv_filename=cv_file.name,
            )

            if success:
                # Build initial system prompt from all gathered context (FR-082)
                st.session_state["system_prompt"] = build_base_system_prompt(
                    candidate_name=st.session_state["candidate_name"],
                    interview_type=st.session_state["interview_type"],
                    difficulty=st.session_state["difficulty"],
                    coding_language=st.session_state["coding_language"],
                    role_info=st.session_state["role_info"],
                    gap_info=st.session_state["gap_info"],
                    trend_snippets=st.session_state["trend_snippets"],
                    custom_questions=st.session_state["custom_questions"] or None,
                )
                st.session_state["interview_start_time"] = time.time()
                st.session_state["view"] = "interview"
                st.success("✅ Setup complete — launching interview…")
                time.sleep(0.4)
                st.rerun()


# ---------------------------------------------------------------------------
# Interview view — core loop implemented in PR5
# ---------------------------------------------------------------------------
def _render_interview_view() -> None:
    """
    Interview view placeholder.

    PR5 will implement:
      - Per-turn BM25 retrieval injected into system prompt
      - Category tag badges [Technical] / [Behavioural] / [Situational]
      - 3-hint progressive reveal system
      - Live MM:SS timer
      - Code editor for coding questions
    """
    role_info = st.session_state.get("role_info", {})
    role_title = role_info.get("role_title", "the role")
    gap_info = st.session_state.get("gap_info", {})

    # ── Header ────────────────────────────────────────────────────────────
    st.title(f"🎤 Interview: {st.session_state.get('candidate_name', 'Candidate')}")
    st.caption(
        f"Role: **{role_title}**  ·  "
        f"Type: **{st.session_state.get('interview_type', '—')}**  ·  "
        f"Difficulty: **{st.session_state.get('current_difficulty', '—')}**  ·  "
        f"Language: **{st.session_state.get('coding_language', '—')}**"
    )
    st.divider()

    st.info(
        "**Interview loop coming in PR5.** "
        "Setup is complete — role analysed, BM25 indexes built, system prompt ready.",
        icon="🔧",
    )

    # ── Role analysis summary ─────────────────────────────────────────────
    if role_info:
        with st.expander("📋 Role Analysis (extracted from JD)", expanded=True):
            r1, r2 = st.columns(2)
            with r1:
                st.write(f"**Role:** {role_info.get('role_title', '—')}")
                st.write(f"**Seniority:** {role_info.get('seniority_level', '—')}")
                st.write(f"**Domain:** {role_info.get('domain', '—')}")
            with r2:
                skills = role_info.get("core_skills", [])
                if skills:
                    st.write(f"**Core Skills:** {', '.join(skills)}")
                focus = role_info.get("interview_focus_areas", [])
                if focus:
                    st.write(f"**Focus Areas:** {', '.join(focus)}")

    # ── CV Gap analysis summary ───────────────────────────────────────────
    if gap_info:
        with st.expander("⚠️ CV Gap Analysis"):
            missing = gap_info.get("missing_skills", [])
            strengths = gap_info.get("strengths_match", [])
            summary = gap_info.get("gap_summary", "")
            if strengths:
                st.success(f"✅ Matched strengths: {', '.join(strengths)}")
            if missing:
                st.error(f"⚠️ Missing skills: {', '.join(missing)}")
            if summary:
                st.info(summary)

    # ── BM25 index stats ──────────────────────────────────────────────────
    with st.expander("🔍 BM25 Index Stats"):
        jd_chunks = st.session_state.get("jd_chunks", [])
        cv_chunks = st.session_state.get("cv_chunks", [])
        col_a, col_b = st.columns(2)
        with col_a:
            status = "✅ Ready" if st.session_state.get("jd_index") else "❌ Not built"
            st.metric("JD Index", f"{len(jd_chunks)} chunks", status)
        with col_b:
            status = "✅ Ready" if st.session_state.get("cv_index") else "❌ Not built"
            st.metric("CV Index", f"{len(cv_chunks)} chunks", status)

    st.divider()
    if st.button("← Back to Setup", type="secondary", key="btn_back_from_interview"):
        st.session_state["view"] = "setup"
        st.rerun()


# ---------------------------------------------------------------------------
# Scorecard view — implemented in PR6
# ---------------------------------------------------------------------------
def _render_scorecard_view() -> None:
    """
    Scorecard view placeholder.

    PR6 will implement:
      - Pydantic Scorecard generation via LCEL chain
      - Structured scorecard UI (fit_rating, strengths, gaps, hint_dependency)
      - Transcript download button
    """
    st.title("📊 Interview Complete")
    st.info("**Scorecard UI coming in PR6.**", icon="🔧")
    st.divider()
    if st.button("← Start New Interview", type="secondary", key="btn_back_from_scorecard"):
        # Reset interview state but keep API keys
        for key in [
            "view", "candidate_name", "jd_text", "cv_text",
            "role_info", "gap_info", "trend_snippets",
            "jd_index", "jd_chunks", "cv_index", "cv_chunks",
            "messages", "qa_log", "question_count", "hints_used",
            "interview_start_time", "current_difficulty",
            "adaptive_escalated", "system_prompt", "scorecard",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------
def main() -> None:
    """Initialise state and route to the correct view."""
    _init_session_state()

    view = st.session_state.get("view", "setup")

    if view == "setup":
        _render_setup_view()
    elif view == "interview":
        _render_interview_view()
    elif view == "scorecard":
        _render_scorecard_view()
    else:
        st.session_state["view"] = "setup"
        st.rerun()


if __name__ == "__main__":
    main()
