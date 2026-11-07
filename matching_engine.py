import re
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

# ===================================================
# 1. Configuration
# ===================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ACCEPTANCE_THRESHOLD = 85

WEIGHTS = {
    "required_skills": 60,
    "responsibilities": 25,
    "experience": 10,
    "education": 5,
}

SKILL_MATCH_THRESHOLD = 0.72
SKILL_REVIEW_THRESHOLD = 0.55
RESPONSIBILITY_THRESHOLD = 0.48
EDUCATION_FIELD_THRESHOLD = 0.40

DEGREE_LEVELS = {
    "bachelor": 1,
    "bs": 1,
    "bsc": 1,
    "ba": 1,
    "licentiate": 1,
    "faculty of": 1,
    "undergraduate": 1,
    "major": 1,
    "master": 2,
    "ms": 2,
    "msc": 2,
    "ma": 2,
    "phd": 3,
    "doctor": 3,
    "doctorate": 3,
}

_model = None


def load_model():
    global _model
    if _model is None and SentenceTransformer is not None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_model():
    return load_model()


# ===================================================
# 2. Text Normalization & Phrase Matching Helpers
# ===================================================

def to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        expanded = []
        for item in value:
            if isinstance(item, str) and "\n" in item:
                expanded.extend(line.strip() for line in item.splitlines() if line.strip())
            else:
                expanded.append(item)
        return expanded
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [value]


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(clean_text(item) for item in value if clean_text(item))
    if isinstance(value, dict):
        ignored = {"dates", "grade", "institution", "company"}
        return " ".join(
            clean_text(v)
            for k, v in value.items()
            if k.lower() not in ignored and v is not None
        )
    return str(value).strip()


SKILL_SYNONYM_REPLACEMENTS = [
    (r"\bux\s*/\s*ui\b", "ui ux"),
    (r"\bui\s*/\s*ux\b", "ui ux"),
    (r"\bux\s+research\b", "user research"),
    (r"\busability\s+testing\b", "usability"),
    (r"\busability\s+reviews\b", "usability"),
    (r"\bresponsive\s+interfaces\b", "responsive design"),
]


def normalize_text(text):
    text = str(text).lower()
    text = text.replace("&", " and ")
    for pat, repl in SKILL_SYNONYM_REPLACEMENTS:
        text = re.sub(pat, repl, text)
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)
    return " ".join(text.split())


def contains_phrase(text, phrase):
    """
    Match whole phrases while preventing 'C' from falsely matching 'C++' or 'C#',
    and supporting '.js' variations (e.g. 'node.js' vs 'nodejs').
    """
    text_norm = normalize_text(text)
    phrase_norm = normalize_text(phrase)
    if not text_norm or not phrase_norm:
        return False

    pattern = r"(?<![a-z0-9+#])" + re.escape(phrase_norm) + r"(?![a-z0-9+#])"
    if re.search(pattern, text_norm):
        return True

    # Support Node.js <-> nodejs / React.js <-> reactjs
    if ".js" in phrase_norm or ".js" in text_norm:
        t_no_dot = text_norm.replace(".js", "js")
        p_no_dot = phrase_norm.replace(".js", "js")
        pattern_js = r"(?<![a-z0-9+#])" + re.escape(p_no_dot) + r"(?![a-z0-9+#])"
        return bool(re.search(pattern_js, t_no_dot))

    return False


def _token_overlap_similarity(text1: str, text2: str) -> float:
    """Fallback similarity when SentenceTransformer model is offline/unavailable."""
    t1 = set(normalize_text(text1).split())
    t2 = set(normalize_text(text2).split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / max(1, min(len(t1), len(t2)) * 0.5 + max(len(t1), len(t2)) * 0.5)


def compute_similarity_matrix(candidate_items, target_items):
    c_texts = [clean_text(item) for item in candidate_items]
    t_texts = [clean_text(item) for item in target_items]

    c_valid = [t for t in c_texts if t]
    t_valid = [t for t in t_texts if t]

    if not c_valid or not t_valid:
        return None, [], []

    model = load_model()
    if model is not None:
        c_embeddings = model.encode(c_valid, convert_to_numpy=True, normalize_embeddings=True)
        t_embeddings = model.encode(t_valid, convert_to_numpy=True, normalize_embeddings=True)
        sim_matrix = np.matmul(c_embeddings, t_embeddings.T)
        return sim_matrix, c_valid, t_valid

    # Fallback matrix using token overlap if model not loaded
    sim_matrix = np.zeros((len(c_valid), len(t_valid)), dtype=float)
    for i, c_str in enumerate(c_valid):
        for j, t_str in enumerate(t_valid):
            sim_matrix[i, j] = _token_overlap_similarity(c_str, t_str)
    return sim_matrix, c_valid, t_valid


def best_semantic_match(candidate_items, target_items):
    sim_matrix, c_valid, t_valid = compute_similarity_matrix(candidate_items, target_items)
    if sim_matrix is None:
        return 0.0
    best_per_target = np.max(sim_matrix, axis=0)
    return float(np.mean(best_per_target))


# ==========================================
# 3. Evaluation Component: Skills
# ==========================================

def _evaluate_compound_requirement(req_clean: str, candidate_norm_list: list[str], full_cv_text_norm: str) -> str:
    """
    Evaluate compound skill requirements like:
    'Expert in Figma: components, variants, auto layout, variables, and prototyping'
    Returns 'matched', 'review', or 'missing'.
    """
    req_norm = normalize_text(req_clean)

    # Check if requirement has colon or comma-separated sub-skills
    if ":" in req_clean and "," in req_clean:
        head, tail = req_clean.split(":", 1)
        head_norm = normalize_text(head.replace("expert in", "").replace("proficient in", ""))
        sub_skills = [
            normalize_text(re.sub(r"^\band\b\s*", "", part.strip()))
            for part in re.split(r",|\band\b", tail)
            if normalize_text(part)
        ]
        head_matched = any(
            contains_phrase(cn, head_norm) or contains_phrase(head_norm, cn)
            for cn in candidate_norm_list
        ) or contains_phrase(full_cv_text_norm, head_norm)

        if sub_skills:
            sub_hits = sum(
                1 for sub in sub_skills
                if any(contains_phrase(cn, sub) for cn in candidate_norm_list)
                or contains_phrase(full_cv_text_norm, sub)
            )
            ratio = sub_hits / len(sub_skills)
            if head_matched and ratio >= 0.6:
                return "matched"
            if head_matched or ratio >= 0.3:
                return "review"
            return "missing"

    # Standard requirement check: candidate skill must cover the requirement
    for c_norm in candidate_norm_list:
        if req_norm == c_norm or contains_phrase(c_norm, req_norm):
            return "matched"
        # Only allow reverse containment if requirement is short (<= 3 words)
        if len(req_norm.split()) <= 3 and len(c_norm.split()) >= 1 and contains_phrase(req_norm, c_norm):
            return "matched"

    if len(req_norm.split()) <= 3 and contains_phrase(full_cv_text_norm, req_norm):
        return "matched"

    return "missing"


def evaluate_skills_match(candidate_skills, job_required_skills, job_optional_skills=None, full_cv_text=""):
    candidate_skills = to_list(candidate_skills)
    job_required_skills = to_list(job_required_skills)
    job_optional_skills = to_list(job_optional_skills)

    if not job_required_skills:
        return {
            "score": 100.0,
            "matched_required_skills": [],
            "review_required_skills": [],
            "missing_required_skills": [],
            "matched_optional_skills": [],
            "missing_optional_skills": [],
            "details": "No required skills specified in job description."
        }

    candidate_clean = [clean_text(s) for s in candidate_skills if clean_text(s)]
    candidate_norm = [normalize_text(s) for s in candidate_clean if normalize_text(s)]
    full_cv_norm = normalize_text(full_cv_text or " ".join(candidate_clean))

    matched_req = []
    review_req = []
    missing_req = []

    for req_skill in job_required_skills:
        req_clean = clean_text(req_skill)
        if not normalize_text(req_clean):
            continue

        status = _evaluate_compound_requirement(req_clean, candidate_norm, full_cv_norm)
        if status == "matched":
            matched_req.append(req_clean)
        elif status == "review":
            review_req.append(req_clean)
        else:
            missing_req.append(req_clean)

    # Semantic Fallback for unmatched required skills
    if missing_req and candidate_clean:
        sim_matrix, c_valid, t_valid = compute_similarity_matrix(candidate_clean, missing_req)
        if sim_matrix is not None:
            still_missing = []
            for j, t_skill in enumerate(t_valid):
                max_sim = float(np.max(sim_matrix[:, j])) if sim_matrix.shape[0] > 0 else 0.0
                if max_sim >= SKILL_MATCH_THRESHOLD:
                    matched_req.append(t_skill)
                elif max_sim >= SKILL_REVIEW_THRESHOLD:
                    review_req.append(t_skill)
                else:
                    still_missing.append(t_skill)
            missing_req = still_missing

    matched_opt = []
    missing_opt = []
    for opt_skill in job_optional_skills:
        opt_clean = clean_text(opt_skill)
        opt_norm = normalize_text(opt_clean)
        if not opt_norm:
            continue
        direct_match = any(
            opt_norm == c_norm or contains_phrase(c_norm, opt_norm)
            for c_norm in candidate_norm
        ) or contains_phrase(full_cv_norm, opt_norm)
        if direct_match:
            matched_opt.append(opt_clean)
        else:
            missing_opt.append(opt_clean)

    total_req = len(job_required_skills)
    matched_req_count = len(matched_req)
    review_req_count = len(review_req)

    base_score = ((matched_req_count + (0.5 * review_req_count)) / total_req) * 100.0 if total_req > 0 else 100.0

    if job_optional_skills:
        opt_ratio = len(matched_opt) / len(job_optional_skills)
        bonus = opt_ratio * 10.0
        final_score = min(100.0, base_score + bonus)
    else:
        final_score = base_score

    return {
        "score": round(final_score, 2),
        "matched_required_skills": matched_req,
        "review_required_skills": review_req,
        "missing_required_skills": missing_req,
        "matched_optional_skills": matched_opt,
        "missing_optional_skills": missing_opt,
        "details": f"Matched {matched_req_count}/{total_req} required skills ({review_req_count} partial/review)."
    }


# ==========================================
# 4. Evaluation Component: Responsibilities
# ==========================================

def evaluate_responsibilities_match(
    candidate_experience_items,
    candidate_projects,
    job_responsibilities,
    job_required_exp_years=0,
):
    job_responsibilities = to_list(job_responsibilities)
    if not job_responsibilities:
        return {
            "score": 100.0,
            "matched_responsibilities": [],
            "missing_responsibilities": [],
            "details": "No responsibilities specified in job description."
        }

    exp_texts = [clean_text(exp) for exp in to_list(candidate_experience_items) if clean_text(exp)]
    proj_texts = [clean_text(proj) for proj in to_list(candidate_projects) if clean_text(proj)]

    candidate_texts = exp_texts + proj_texts
    if not candidate_texts:
        return {
            "score": 0.0,
            "matched_responsibilities": [],
            "missing_responsibilities": [clean_text(r) for r in job_responsibilities if clean_text(r)],
            "details": "Candidate has no experience or projects text for responsibility matching."
        }

    sim_matrix, c_valid, t_valid = compute_similarity_matrix(candidate_texts, job_responsibilities)
    if sim_matrix is None:
        return {
            "score": 0.0,
            "matched_responsibilities": [],
            "missing_responsibilities": [clean_text(r) for r in job_responsibilities if clean_text(r)],
            "details": "Could not compute similarity matrix."
        }

    matched_resp = []
    missing_resp = []
    scores = []

    for j, resp in enumerate(t_valid):
        max_sim = float(np.max(sim_matrix[:, j])) if sim_matrix.shape[0] > 0 else 0.0
        scores.append(max_sim)
        if max_sim >= RESPONSIBILITY_THRESHOLD:
            matched_resp.append({"responsibility": resp, "similarity": round(max_sim, 2)})
        else:
            missing_resp.append(resp)

    avg_sim = float(np.mean(scores)) if scores else 0.0
    raw_score = min(100.0, avg_sim * 100.0)

    # Discount project-only alignment when role requires >=3 years of professional experience
    project_only_multiplier = 0.65 if (not exp_texts and float(job_required_exp_years or 0) >= 3) else 1.0
    final_score = round(raw_score * project_only_multiplier, 2)

    return {
        "score": final_score,
        "matched_responsibilities": matched_resp,
        "missing_responsibilities": missing_resp,
        "details": (
            f"Average semantic responsibility alignment: {round(avg_sim, 2)}"
            + (" (discounted 35% because candidate has only projects and no formal work experience for a Senior role)." if project_only_multiplier < 1.0 else "")
        )
    }


# ==========================================
# 5. Evaluation Component: Experience Years
# ==========================================

def evaluate_experience_years_match(candidate_exp_years, job_required_exp_years, seniority_level="Not Specified"):
    try:
        c_years = float(candidate_exp_years or 0)
    except (ValueError, TypeError):
        c_years = 0.0

    try:
        r_years = float(job_required_exp_years or 0)
    except (ValueError, TypeError):
        r_years = 0.0

    if r_years == 0:
        return {
            "score": 100.0,
            "candidate_years": c_years,
            "required_years": r_years,
            "details": "No minimum experience years required."
        }

    if c_years >= r_years:
        score = 100.0
        details = f"Candidate experience ({c_years} yrs) meets or exceeds requirement ({r_years} yrs)."
    else:
        ratio = c_years / r_years
        score = ratio * 100.0
        details = f"Candidate experience ({c_years} yrs) is below requirement ({r_years} yrs)."

    return {
        "score": round(score, 2),
        "candidate_years": c_years,
        "required_years": r_years,
        "seniority_level": seniority_level,
        "details": details
    }


# ==========================================
# 6. Evaluation Component: Education
# ==========================================

def get_degree_level(degree_text):
    norm = normalize_text(degree_text)
    best_lvl = 0
    for deg, lvl in DEGREE_LEVELS.items():
        if contains_phrase(norm, deg):
            best_lvl = max(best_lvl, lvl)
    return best_lvl


def evaluate_education_match(candidate_education, job_education_reqs):
    candidate_education = to_list(candidate_education)
    job_education_reqs = to_list(job_education_reqs)

    c_texts = [clean_text(e) for e in candidate_education if clean_text(e)]
    c_max_degree = max([get_degree_level(t) for t in c_texts] + [0])

    if not job_education_reqs:
        return {
            "score": 100.0,
            "detected_candidate_degree_level": c_max_degree,
            "details": "No explicit education requirements specified in job description."
        }

    if not candidate_education:
        return {
            "score": 50.0,
            "detected_candidate_degree_level": 0,
            "details": "No education details found on candidate CV."
        }

    j_texts = [clean_text(e) for e in job_education_reqs if clean_text(e)]
    j_max_degree = max([get_degree_level(t) for t in j_texts] + [0])

    if j_max_degree == 0 or c_max_degree >= j_max_degree:
        degree_score = 100.0
    elif c_max_degree == j_max_degree - 1:
        degree_score = 75.0
    else:
        degree_score = 50.0

    sim_score = best_semantic_match(c_texts, j_texts)
    field_score = min(100.0, sim_score * 100.0)
    final_score = (degree_score * 0.6) + (field_score * 0.4)

    return {
        "score": round(final_score, 2),
        "detected_candidate_degree_level": c_max_degree,
        "degree_level_match_score": degree_score,
        "field_match_score": round(field_score, 2),
        "details": f"Degree level match: {degree_score}%, Field match: {round(field_score, 2)}%"
    }


# ==========================================
# 7. Main Function: match_cv_to_job
# ==========================================

def match_cv_to_job(cv_data, job_requirements):
    ai_info = cv_data.get("ai_extracted_info", {})
    sections = cv_data.get("sections", {})

    cand_skills = ai_info.get("skills") or cv_data.get("skills") or sections.get("skills", [])
    cand_exp_items = ai_info.get("work_experience") or sections.get("experience", [])
    cand_projects = ai_info.get("projects") or sections.get("projects", [])
    cand_edu = ai_info.get("education") or sections.get("education", [])
    full_cv_text = cv_data.get("cleaned_text") or cv_data.get("full_text") or clean_text(sections)

    cand_years = 0.0
    if "total_experience_years" in ai_info and ai_info["total_experience_years"] is not None:
        try:
            cand_years = float(ai_info["total_experience_years"])
        except (ValueError, TypeError):
            cand_years = 0.0
    elif "experience_years" in cv_data and cv_data["experience_years"] is not None:
        try:
            cand_years = float(cv_data["experience_years"])
        except (ValueError, TypeError):
            cand_years = 0.0

    job_req_skills = job_requirements.get("required_skills", [])
    job_opt_skills = job_requirements.get("optional_skills", [])
    job_resp = job_requirements.get("responsibilities", [])
    job_exp_years = job_requirements.get("required_experience_years", 0)
    job_seniority = job_requirements.get("seniority_level", "Not Specified")
    job_edu = job_requirements.get("education_requirements", [])

    skills_res = evaluate_skills_match(cand_skills, job_req_skills, job_opt_skills, full_cv_text=full_cv_text)
    resp_res = evaluate_responsibilities_match(cand_exp_items, cand_projects, job_resp, job_required_exp_years=job_exp_years)
    exp_res = evaluate_experience_years_match(cand_years, job_exp_years, job_seniority)
    edu_res = evaluate_education_match(cand_edu, job_edu)

    # Balanced weights: Skills (45%), Responsibilities (25%), Experience (20%), Education (10%)
    overall_score = (
        (skills_res["score"] * 0.45) +
        (resp_res["score"] * 0.25) +
        (exp_res["score"] * 0.20) +
        (edu_res["score"] * 0.10)
    )
    overall_score = round(overall_score, 2)

    seniority_warning = None
    if float(job_exp_years or 0) >= 3 and cand_years < (float(job_exp_years) * 0.5):
        seniority_warning = (
            f"Seniority Gap: Role requires {job_exp_years}+ years ({job_seniority}), "
            f"while candidate has {cand_years} years of professional experience."
        )

    if overall_score >= ACCEPTANCE_THRESHOLD and not seniority_warning:
        fit_level = "Strong Match"
        status = "Accepted for Interview"
    elif overall_score >= 60.0 and not seniority_warning:
        fit_level = "Good Match"
        status = "Potentially Suitable"
    elif overall_score >= 40.0:
        fit_level = "Moderate Match (Junior / Partial Alignment)" if seniority_warning else "Moderate Match"
        status = "Not Accepted (Seniority / Skill Gap)" if seniority_warning else "Needs Review"
    else:
        fit_level = "Low Match"
        status = "Rejected"

    skill_gaps = skills_res.get("missing_required_skills", []) + skills_res.get("review_required_skills", [])
    missing_skills_structured = [{"requirement": s} for s in skill_gaps]

    return {
        "job_title": job_requirements.get("job_title", "Not Specified"),
        "overall_score": overall_score,
        "matching_score": overall_score,
        "evaluation_coverage": "100%",
        "fit_level": fit_level,
        "status": status,
        "seniority_warning": seniority_warning,
        "sub_scores": {
            "skills": skills_res["score"],
            "responsibilities": resp_res["score"],
            "experience": exp_res["score"],
            "education": edu_res["score"],
        },
        "breakdown": {
            "skills": skills_res,
            "responsibilities": resp_res,
            "experience": exp_res,
            "education": edu_res,
        },
        "skill_gaps": skill_gaps,
        "missing_skills": missing_skills_structured,
        "summary": (
            f"Candidate achieved an overall match score of {overall_score}% ({fit_level}). "
            f"Skills: {skills_res['score']}%, Responsibilities: {resp_res['score']}%, "
            f"Experience: {exp_res['score']}%, Education: {edu_res['score']}%."
            + (f" Warning: {seniority_warning}" if seniority_warning else "")
        ),
    }