from __future__ import annotations

import html
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

OUTPUT_DIR = Path("outputs")
LOG_FILE = Path("careerlens.log")
UPLOAD_TYPES = ["pdf", "docx", "txt"]

SPECIALTIES = [
    "Python Backend Developer",
    "Flutter Developer",
    "Full Stack .NET Developer",
    "UI/UX Designer",
    "Machine Learning Engineer",
    "Data Analyst",
    "أخرى",
]

st.set_page_config(page_title="CareerLens", page_icon="📄", layout="wide")


# =====================================================
# 0. Logging
# =====================================================

def get_logger() -> logging.Logger:
    logger = logging.getLogger("careerlens")

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )
        )

        logger.addHandler(handler)
        logger.propagate = False

    return logger


log = get_logger()


# =====================================================
# 1. Style
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        direction: rtl;
        background: #f7f9fc;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp p,
    .stApp label {
        text-align: right;
    }

    .cl-hero {
        background: linear-gradient(135deg, #0f766e, #1e3a8a);
        color: #ffffff;
        padding: 24px 28px;
        border-radius: 16px;
        margin-bottom: 16px;
    }

    .cl-hero h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2rem;
    }

    .cl-hero p {
        color: #dbeafe;
        margin: 6px 0 0 0;
    }

    .cl-chip {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px;
        border-radius: 999px;
        background: #e0f2fe;
        color: #075985;
        font-size: 14px;
    }

    .cl-chip-ok {
        background: #dcfce7;
        color: #166534;
    }

    .cl-chip-missing {
        background: #fee2e2;
        color: #991b1b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 2. Password gate
# =====================================================

def check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True

    password = st.text_input("كلمة السر", type="password")

    if password:
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
            log.info("Login success")
            st.rerun()
        else:
            log.warning("Login failed")
            st.error("كلمة السر غلط")

    return False


if not check_password():
    st.stop()


# =====================================================
# 3. Heavy imports (after login only)
# =====================================================

with st.spinner("جاري تحميل النظام..."):
    from main import analyze_cv, _write_json_atomic

    from cv_extractor import (
        extract_text_from_cv as extract_text
    )

    from job_description_analyzer import analyze_job_description
    from matching_engine import match_cv_to_job
    from learning_advisor import build_advisor_section


# =====================================================
# 4. File helpers
# =====================================================

def save_upload_to_temp(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:
        tmp.write(uploaded.getbuffer())
        return Path(tmp.name)


def read_upload_text(uploaded) -> str:
    tmp_path = save_upload_to_temp(uploaded)

    try:
        return (extract_text(str(tmp_path)) or "").strip()
    finally:
        tmp_path.unlink(missing_ok=True)


def analyze_cv_upload(uploaded) -> dict:
    tmp_path = save_upload_to_temp(uploaded)

    try:
        return analyze_cv(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_submission(result: dict, original_name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    submission_id = uuid.uuid4().hex

    result["submission_id"] = submission_id
    result["original_file_name"] = original_name
    result["file_name"] = f"{submission_id}.pdf"
    result["uploaded_at"] = datetime.now().isoformat(
        timespec="seconds"
    )

    output_file = OUTPUT_DIR / f"{submission_id}_analysis.json"

    _write_json_atomic(output_file, result)

    return output_file


# =====================================================
# 5. Pipeline
# =====================================================

def run_pipeline(cv_file, job_file, job_text: str) -> dict:

    cv_result = analyze_cv_upload(cv_file)

    if not cv_result.get("ai_extracted_info"):
        raise ValueError(
            "فشل استخراج بيانات السيرة الذاتية. جربي تاني."
        )

    saved_path = save_submission(cv_result, cv_file.name)

    log.info(
        "CV analyzed | file=%s | status=%s | saved=%s",
        cv_file.name,
        cv_result.get("ai_extraction_status"),
        saved_path.name,
    )

    job_source = (
        read_upload_text(job_file)
        if job_file
        else job_text.strip()
    )

    job_data = analyze_job_description(job_source)

    log.info(
        "Job analyzed | title=%s | career_field=%s | provider=%s",
        job_data.get("job_title"),
        job_data.get("career_field"),
        job_data.get("provider"),
    )

    match = match_cv_to_job(cv_result, job_data)

    log.info(
        "Matching done | cv=%s | score=%s | coverage=%s | status=%s",
        cv_file.name,
        match.get("matching_score"),
        match.get("evaluation_coverage"),
        match.get("status"),
    )

    advisor = build_advisor_section(match)

    return {
        "cv": cv_result,
        "job": job_data,
        "match": match,
        "advisor": advisor,
        "saved_path": str(saved_path),
    }


def handle_submit(cv_file, job_file, job_text: str) -> None:

    if cv_file is None:
        st.warning("ارفعي السيرة الذاتية الأول.")
        return

    if job_file is None and not job_text.strip():
        st.warning("ارفعي ملف الوظيفة أو الصقي نصها.")
        return

    st.session_state.pop("analysis", None)
    st.session_state.pop("confirmed_specialty", None)
    st.session_state.pop("show_specialty_picker", None)

    with st.spinner("جاري تحليل السيرة الذاتية والوظيفة..."):

        try:
            st.session_state["analysis"] = run_pipeline(
                cv_file,
                job_file,
                job_text
            )

        except Exception as error:
            log.exception(
                "Pipeline failed | cv=%s",
                cv_file.name
            )

            st.error(f"تعذّر إجراء التحليل: {error}")


# =====================================================
# 6. Display helpers
# =====================================================

def chips(items, css_class: str = "cl-chip") -> str:

    if not items:
        return "—"

    return "".join(
        f'<span class="{css_class}">'
        f'{html.escape(str(item))}'
        f'</span>'
        for item in items
    )


def show_cv_details(result: dict) -> None:

    if result.get("ai_extraction_status") == "success":
        st.success("تم الاستخراج بنجاح")
    else:
        st.warning(
            "فشل الاستخراج بالذكاء الاصطناعي: "
            f"{result.get('ai_extraction_error')}"
        )

    contact = result.get("contact_info") or {}

    cols = st.columns(4)

    cols[0].metric("Email", contact.get("email") or "—")
    cols[1].metric("Phone", contact.get("phone") or "—")
    cols[2].metric(
        "LinkedIn",
        "✓" if contact.get("linkedin") else "—"
    )
    cols[3].metric(
        "GitHub",
        "✓" if contact.get("github") else "—"
    )

    ai = result["ai_extracted_info"]

    if ai.get("professional_summary"):
        st.write(ai["professional_summary"])

    tab_exp, tab_edu, tab_proj, tab_skills, tab_cert = st.tabs(
        [
            "الخبرة",
            "التعليم",
            "المشاريع",
            "المهارات",
            "الشهادات",
        ]
    )

    with tab_exp:

        if not ai.get("experience"):
            st.write("لا توجد خبرات.")

        for job in ai.get("experience", []):
            st.markdown(
                f"**{job.get('title') or '—'}** — "
                f"{job.get('company') or '—'}"
            )

            st.caption(job.get("dates") or "")

            for line in job.get("description", []):
                st.markdown(f"- {line}")

    with tab_edu:

        for edu in ai.get("education", []):
            st.markdown(f"**{edu.get('degree') or '—'}**")
            st.write(edu.get("institution") or "")

            st.caption(
                " | ".join(
                    x for x in [
                        edu.get("dates"),
                        edu.get("grade"),
                    ]
                    if x
                )
            )

    with tab_proj:

        for project in ai.get("projects", []):
            st.markdown(
                f"**{project.get('name') or '—'}**"
            )

            for line in project.get("description", []):
                st.markdown(f"- {line}")

            st.divider()

    with tab_skills:

        st.markdown("**المهارات التقنية**")
        st.write(" • ".join(ai.get("skills", [])) or "—")

        st.markdown("**المهارات الشخصية**")
        st.write(" • ".join(ai.get("soft_skills", [])) or "—")

        st.markdown("**اللغات**")
        st.write(" • ".join(ai.get("languages", [])) or "—")

    with tab_cert:

        if not ai.get("certifications"):
            st.write("لا توجد شهادات.")

        for cert in ai.get("certifications", []):
            st.markdown(
                f"**{cert.get('name') or '—'}**"
            )

            st.caption(
                " | ".join(
                    x for x in [
                        cert.get("issuer"),
                        cert.get("date"),
                    ]
                    if x
                )
            )


def render_cv_summary(cv_result: dict) -> None:

    ai = cv_result["ai_extracted_info"]

    st.markdown(
        f"### {html.escape(ai.get('candidate_name') or '—')}"
    )

    st.caption(
        "التخصص المقترح من النظام: "
        f"{ai.get('suggested_specialty') or '—'}"
    )

    with st.expander("تفاصيل السيرة الذاتية"):
        show_cv_details(cv_result)


def save_specialty(
    analysis: dict,
    suggested: str,
    confirmed: str
) -> None:

    cv_result = analysis["cv"]

    cv_result["suggested_specialty"] = suggested
    cv_result["confirmed_specialty"] = confirmed

    _write_json_atomic(
        Path(analysis["saved_path"]),
        cv_result
    )

    st.session_state["confirmed_specialty"] = confirmed

    log.info(
        "Specialty confirmed | file=%s | suggested=%s | confirmed=%s | corrected=%s",
        cv_result.get("original_file_name"),
        suggested,
        confirmed,
        confirmed != suggested,
    )


def render_specialty_confirmation(analysis: dict) -> None:

    suggested = (
        analysis["cv"]["ai_extracted_info"].get(
            "suggested_specialty"
        )
        or "غير محدد"
    )

    confirmed = st.session_state.get("confirmed_specialty")

    st.markdown("### تأكيد التخصص")

    if confirmed:

        st.success(f"التخصص المؤكد: {confirmed}")

        if confirmed != suggested:
            st.info(
                "ارفعي وصف وظيفة من تخصصك واضغطي "
                "'حلل وقارن' عشان نقيّم السيرة الذاتية عليها."
            )

        return

    st.write(
        f"النظام شايف إن تخصصك **{suggested}**. هو ده صح؟"
    )

    col_yes, col_no = st.columns(2)

    if col_yes.button("أيوه، ده تخصصي"):
        save_specialty(analysis, suggested, suggested)
        st.rerun()

    if col_no.button("لأ، تخصصي مختلف"):
        st.session_state["show_specialty_picker"] = True

    if st.session_state.get("show_specialty_picker"):

        choice = st.selectbox(
            "اختاري تخصصك الصح",
            SPECIALTIES,
            key="specialty_choice"
        )

        if st.button("احفظ التخصص"):
            save_specialty(analysis, suggested, choice)
            st.rerun()


# =====================================================
# 7. Matching Results
# =====================================================

def render_match(
    match: dict,
    job: dict,
    advisor: dict
) -> None:

    score = match.get(
        "matching_score",
        match.get("overall_score", 0)
    )

    score_value = float(score) if score is not None else 0.0

    status = match.get("status", "")

    if status == "Accepted for Interview":
        status_label = "مناسب للمقابلة"

    elif status == "Potentially Suitable":
        status_label = "مناسب مبدئيًا"

    elif status == "Needs Review":
        status_label = "يحتاج مراجعة"

    elif status == "Rejected":
        status_label = "غير مناسب حاليًا"

    elif status == "Not Accepted (Seniority / Skill Gap)":
        status_label = "يحتاج تطوير بسبب فجوة الخبرة أو المهارات"

    else:
        status_label = status or "غير محدد"

    st.markdown("### نتيجة المطابقة")

    st.caption(
        f"الوظيفة: {job.get('job_title') or '—'}"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "نسبة التقارب",
        f"{score_value:.1f}%"
    )

    c2.metric(
        "الحالة",
        status_label
    )

    st.progress(
        min(max(score_value / 100, 0.0), 1.0)
    )

    # Read skills using the actual structure returned
    # by matching_engine.py

    breakdown = match.get("breakdown") or {}

    skills_result = breakdown.get("skills") or {}

    matched_names = skills_result.get(
        "matched_required_skills",
        []
    )

    review_names = skills_result.get(
        "review_required_skills",
        []
    )

    missing_names = skills_result.get(
        "missing_required_skills",
        []
    )

    st.markdown("**المهارات المتطابقة**")

    st.markdown(
        chips(
            matched_names,
            "cl-chip cl-chip-ok"
        ),
        unsafe_allow_html=True
    )

    st.markdown("**المهارات التي تحتاج مراجعة**")

    st.markdown(
        chips(review_names),
        unsafe_allow_html=True
    )

    st.markdown("**المهارات الناقصة**")

    st.markdown(
        chips(
            missing_names,
            "cl-chip cl-chip-missing"
        ),
        unsafe_allow_html=True
    )

    if match.get("seniority_warning"):
        st.warning(match["seniority_warning"])

    st.markdown("### الخطوة الجاية")

    if not advisor:
        st.info("لا توجد توصيات متاحة حاليًا.")
        return

    if advisor.get("type") == "encouragement":
        st.success(
            advisor.get("message", "أداء جيد!")
        )
        return

    links = advisor.get("links") or []

    if not links:
        st.info("لا توجد مصادر متاحة حاليًا.")
        return

    for item in links:

        with st.container(border=True):

            st.markdown(
                f"**{html.escape(str(item.get('skill', '—')))}**"
            )

            if item.get("explanation"):
                st.caption(item["explanation"])

            if item.get("name") and item.get("url"):
                st.markdown(
                    f"[{item['name']}]({item['url']})"
                )


# =====================================================
# 8. Page
# =====================================================

st.markdown(
    """
    <div class="cl-hero">
        <h1>CareerLens</h1>
        <p>
            ارفعي السيرة الذاتية والوظيفة،
            واطّلعي على نسبة التقارب والخطوات اللي تقفل الفجوة
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 1. السيرة الذاتية")

cv_file = st.file_uploader(
    "ارفعي ملف السيرة الذاتية",
    type=UPLOAD_TYPES,
    key="cv_upload"
)

st.markdown("### 2. الوظيفة")

job_file = st.file_uploader(
    "ارفعي ملف الوظيفة (اختياري)",
    type=UPLOAD_TYPES,
    key="job_upload"
)

job_text = st.text_area(
    "أو الصقي نص الوظيفة هنا",
    height=160,
    key="job_text"
)

st.caption(
    "لو رفعتي ملف للوظيفة، هيتاخد بدل النص."
)

if st.button("حلل وقارن", type="primary"):
    handle_submit(
        cv_file,
        job_file,
        job_text
    )

analysis = st.session_state.get("analysis")

if analysis:

    st.divider()

    render_cv_summary(analysis["cv"])

    st.divider()

    render_specialty_confirmation(analysis)

    st.divider()

    render_match(
        analysis["match"],
        analysis["job"],
        analysis["advisor"]
    )
