import re
from datetime import datetime
from sentence_transformers import SentenceTransformer, util


# =====================================================
# 1. Configuration
# =====================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ACCEPTANCE_THRESHOLD = 85

WEIGHTS = {
    "required_skills": 60,
    "responsibilities": 25,
    "experience": 10,
    "education": 5,
}

# Similarity thresholds (0 to 1). These are starting guesses.
# Tune them later using a labeled dataset of CV/job pairs.
SKILL_MATCH_THRESHOLD = 0.72
SKILL_REVIEW_THRESHOLD = 0.55
RESPONSIBILITY_THRESHOLD = 0.48
EDUCATION_FIELD_THRESHOLD = 0.40

# Degree levels used to compare education without keyword lists per field.
DEGREE_LEVELS = {
    "phd": 3,
    "doctorate": 3,
    "doctoral": 3,
    "master": 2,
    "msc": 2,
    "mba": 2,
    "bachelor": 1,
    "bsc": 1,
    "undergraduate": 1,
}

_model = None


# =====================================================
# 2. Load Model
# =====================================================

def load_model():
    global _model

    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded successfully.")

    return _model


# =====================================================
# 3. General Helpers
# =====================================================

def to_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        return [value] if value.strip() else []

    return []


def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return " ".join(
            clean_text(item)
            for item in value
            if clean_text(item)
        )

    if isinstance(value, dict):
        ignored = {"dates", "grade", "institution", "company"}

        return " ".join(
            clean_text(v)
            for k, v in value.items()
            if k.lower() not in ignored and v is not None
        )

    return str(value).strip()


def normalize_text(text):
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)
    return " ".join(text.split())


def contains_phrase(text, phrase):
    text = normalize_text(text)
    phrase = normalize_text(phrase)

    if not text or not phrase:
        return False

    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text))


def flatten_records(items):
    results = []

    for item in to_list(items):
        text = clean_text(item)
        if text:
            results.append(text)

    return results


def best_semantic_match(query, candidates):
    """Return the highest cosine similarity (0 to 1) and the best candidate text."""
    query = clean_text(query)
    candidates = [clean_text(c) for c in to_list(candidates) if clean_text(c)]

    if not query or not candidates:
        return 0.0, ""

    model = load_model()

    embeddings = model.encode(
        [query] + candidates,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarities = util.cos_sim(embeddings[0], embeddings[1:])[0]
    best_index = similarities.argmax().item()

    return max(0.0, similarities[best_index].item()), candidates[best_index]


# =====================================================
# 4. Prepare CV Profile
# =====================================================

def split_project_records(items):
    """
    Turn each project into one line per bullet, with the project name in front.
    Short, separate lines match job requirements much better than one long block.
    """
    results = []

    for item in to_list(items):
        if isinstance(item, dict):
            name = clean_text(item.get("name"))

            for desc in to_list(item.get("description")):
                text = clean_text(desc)
                if text:
                    results.append(f"{name}: {text}" if name else text)
        else:
            text = clean_text(item)
            if text:
                results.append(text)

    return results


def prepare_cv_profile(cv_data):
    ai_data = cv_data.get("ai_extracted_info") or {}

    skills = ai_data.get("skills") or cv_data.get("skills") or []

    experience = to_list(ai_data.get("experience"))
    projects = to_list(ai_data.get("projects"))
    education = to_list(ai_data.get("education"))

    summary = clean_text(ai_data.get("professional_summary"))

    skill_texts = [clean_text(skill) for skill in to_list(skills) if clean_text(skill)]

    return {
        "skills": skill_texts,
        "summary": summary,
        "experience": split_project_records(experience),
        "projects": split_project_records(projects),
        "education": flatten_records(education),
        "experience_records": experience,
    }


# =====================================================
# 5. Skill Matching
# =====================================================

def get_skill_evidence(profile):
    """All CV text where a skill could appear."""
    evidence = (
        profile["skills"]
        + [profile["summary"]]
        + profile["experience"]
        + profile["projects"]
    )
    return [item for item in evidence if item]


def match_one_skill(skill, profile):
    """
    Outcomes:
    - matched: the exact name is in the CV, or the meaning is close enough.
    - potential_match: somewhat close, needs review.
    - missing: not close enough.
    """
    evidence = get_skill_evidence(profile)

    # 1. Exact name appears in the CV. No alias list needed.
    for item in evidence:
        if contains_phrase(item, skill):
            return {
                "requirement": skill,
                "similarity": 100.0,
                "score": 100.0,
                "status": "matched",
                "evidence": item,
                "match_type": "exact",
            }

    # 2. Meaning-based similarity handles synonyms like "Postgres" vs "SQL databases".
    similarity, best_evidence = best_semantic_match(skill, evidence)
    percentage = round(similarity * 100, 2)

    if similarity >= SKILL_MATCH_THRESHOLD:
        status = "matched"
        score = 100.0
        match_type = "semantic"
    elif similarity >= SKILL_REVIEW_THRESHOLD:
        status = "potential_match"
        score = 0.0
        match_type = "semantic_review"
    else:
        status = "missing"
        score = 0.0
        match_type = "none"

    return {
        "requirement": skill,
        "similarity": percentage,
        "score": score,
        "status": status,
        "evidence": best_evidence,
        "match_type": match_type,
    }


def extract_skill_alternatives(job_data, required_skills):
    """Group skills written as alternatives, like 'FastAPI or Django' or 'FastAPI/Django'."""
    summary = clean_text(job_data.get("summary"))
    all_text = " ".join([summary] + [str(skill) for skill in required_skills])

    skills = [str(skill).strip() for skill in required_skills if str(skill).strip()]
    groups = []
    used = set()

    for i, first in enumerate(skills):
        if first in used:
            continue

        for second in skills[i + 1:]:
            if second in used:
                continue

            pattern = (
                r"(?<!\w)" + re.escape(first)
                + r"\s*(?:/|\bor\b)\s*"
                + re.escape(second) + r"(?!\w)"
            )
            reverse_pattern = (
                r"(?<!\w)" + re.escape(second)
                + r"\s*(?:/|\bor\b)\s*"
                + re.escape(first) + r"(?!\w)"
            )

            if (re.search(pattern, all_text, re.IGNORECASE)
                    or re.search(reverse_pattern, all_text, re.IGNORECASE)):
                groups.append([first, second])
                used.add(first)
                used.add(second)
                break

    result = []
    for skill in skills:
        if skill in used:
            group = next(group for group in groups if skill in group)
            if group not in result:
                result.append(group)
        else:
            result.append([skill])

    return result


def match_required_skills(job_data, profile):
    required_skills = to_list(job_data.get("required_skills"))

    if not required_skills:
        return {
            "score": 100.0,
            "matched": [],
            "potential_matches": [],
            "missing": [],
            "groups": [],
        }

    groups = extract_skill_alternatives(job_data, required_skills)

    group_results = []
    matched = []
    potential_matches = []
    missing = []

    for group in groups:
        alternatives_results = [match_one_skill(skill, profile) for skill in group]

        # Pick the best alternative: a confirmed match beats a potential match.
        best_result = max(
            alternatives_results,
            key=lambda item: (item["score"], item["similarity"]),
        )

        group_name = " OR ".join(group) if len(group) > 1 else group[0]
        group_status = best_result["status"]

        group_result = {
            "requirement": group_name,
            "alternatives": alternatives_results,
            "score": best_result["score"],
            "similarity": best_result["similarity"],
            "status": group_status,
            "matched_alternative": (
                best_result["requirement"] if group_status == "matched" else None
            ),
            "evidence": best_result["evidence"],
        }

        group_results.append(group_result)

        if group_status == "matched":
            matched.append(group_result)
        elif group_status == "potential_match":
            potential_matches.append(group_result)
        else:
            missing.append(group_result)

    score = sum(item["score"] for item in group_results) / len(group_results)

    return {
        "score": round(score, 2),
        "matched": matched,
        "potential_matches": potential_matches,
        "missing": missing,
        "groups": group_results,
    }


# =====================================================
# 6. Responsibility Matching
# =====================================================

def match_responsibilities(job_data, profile):
    responsibilities = [
        str(item) for item in to_list(job_data.get("responsibilities"))
        if clean_text(item)
    ]

    if not responsibilities:
        return {"score": 100.0, "matched": [], "missing": []}

    evidence = (
        profile["experience"]
        + profile["projects"]
        + ([profile["summary"]] if profile["summary"] else [])
    )

    matched = []
    missing = []
    scores = []

    for responsibility in responsibilities:
        similarity, best_evidence = best_semantic_match(responsibility, evidence)
        percentage = round(similarity * 100, 2)
        scores.append(percentage)

        result = {
            "requirement": responsibility,
            "similarity": percentage,
            "evidence": best_evidence,
            "match_type": "semantic",
        }

        if similarity >= RESPONSIBILITY_THRESHOLD:
            matched.append(result)
        else:
            missing.append(result)

    return {
        "score": round(sum(scores) / len(scores), 2),
        "matched": matched,
        "missing": missing,
    }


# =====================================================
# 7. Experience Evaluation
# =====================================================

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_date_parts(text):
    text = str(text).lower()

    matches = re.findall(
        r"\b(january|february|march|april|may|june|july|august|"
        r"september|sept|october|november|december|jan|feb|mar|"
        r"apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})\b",
        text,
    )

    if matches:
        return [(int(year), MONTHS[month]) for month, year in matches]

    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    return [(int(year), 1) for year in years]


def months_between(start, end):
    return max(0, (end[0] - start[0]) * 12 + end[1] - start[1] + 1)


def calculate_total_experience_months(experience_records):
    records = [record for record in to_list(experience_records) if isinstance(record, dict)]

    # No work experience section means 0 months, not "unknown".
    if not records:
        return 0

    intervals = []
    now = datetime.now()
    current = (now.year, now.month)

    for record in records:
        dates_text = str(record.get("dates") or "")

        if not dates_text:
            continue

        date_parts = parse_date_parts(dates_text)

        if not date_parts:
            continue

        start = date_parts[0]

        if re.search(r"\b(present|current|now)\b", dates_text, re.I):
            end = current
        elif len(date_parts) >= 2:
            end = date_parts[1]
        else:
            end = start

        if end < start:
            continue

        intervals.append((start, end))

    # Records exist but no dates could be read: information is missing.
    if not intervals:
        return None

    intervals.sort()

    # Merge overlapping jobs so the same months are not counted twice.
    merged = []
    for start, end in intervals:
        if not merged:
            merged.append([start, end])
        elif start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])

    return sum(months_between(start, end) for start, end in merged)


def parse_experience_requirement(requirements):
    text = " ".join(str(item) for item in to_list(requirements)).lower()

    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if not numbers:
        return None, None

    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])

    value = float(numbers[0])

    if any(word in text for word in ["less than", "under", "maximum", "up to"]):
        return 0.0, value

    return value, None


def evaluate_experience(job_data, profile):
    requirements = to_list(job_data.get("experience_requirements"))

    if not requirements:
        return {
            "score": 100.0,
            "status": "not_required",
            "candidate_experience_months": None,
            "required_min_years": None,
            "required_max_years": None,
        }

    total_months = calculate_total_experience_months(profile["experience_records"])
    minimum, maximum = parse_experience_requirement(requirements)

    if total_months is None or minimum is None:
        return {
            "score": None,
            "status": "not_provided",
            "candidate_experience_months": total_months,
            "required_min_years": minimum,
            "required_max_years": maximum,
        }

    candidate_years = total_months / 12

    if candidate_years < minimum:
        score = candidate_years / minimum * 100 if minimum > 0 else 100.0
        status = "below_requirement"
    elif maximum is not None and candidate_years > maximum:
        score = 100.0
        status = "above_required_range"
    else:
        score = 100.0
        status = "meets_requirement"

    return {
        "score": round(min(score, 100.0), 2),
        "status": status,
        "candidate_experience_months": total_months,
        "candidate_experience_years": round(candidate_years, 2),
        "required_min_years": minimum,
        "required_max_years": maximum,
    }


# =====================================================
# 8. Education Evaluation
# =====================================================

def get_degree_level(text):
    """Return the highest degree level found in the text (0 if none)."""
    best = 0

    for keyword, level in DEGREE_LEVELS.items():
        if contains_phrase(text, keyword):
            best = max(best, level)

    return best


def evaluate_education(job_data, profile):
    requirements = [
        clean_text(item) for item in to_list(job_data.get("education_requirements"))
        if clean_text(item)
    ]

    if not requirements:
        return {
            "score": 100.0,
            "status": "not_required",
            "degree_verified": None,
            "field_verified": None,
            "candidate_education": profile["education"],
        }

    education_items = profile["education"]

    if not education_items:
        return {
            "score": None,
            "status": "not_provided",
            "degree_verified": None,
            "field_verified": None,
            "candidate_education": [],
        }

    requirement_text = " ".join(requirements)
    candidate_text = " ".join(education_items)

    # Degree level check: candidate level must be at least the required level.
    required_level = get_degree_level(requirement_text)
    candidate_level = get_degree_level(candidate_text)

    degree_verified = None
    if required_level:
        degree_verified = candidate_level >= required_level

    # Field check (only when the requirement mentions a field of study).
    field_verified = None
    field_similarity = None
    if re.search(r"\bin\b|field|major|discipline", normalize_text(requirement_text)):
        similarity, _ = best_semantic_match(requirement_text, education_items)
        field_similarity = round(similarity * 100, 2)
        field_verified = similarity >= EDUCATION_FIELD_THRESHOLD

    checks = [value for value in (degree_verified, field_verified) if value is not None]

    if not checks:
        return {
            "score": None,
            "status": "not_checked",
            "degree_verified": degree_verified,
            "field_verified": field_verified,
            "candidate_education": education_items,
        }

    score = sum(100.0 if value else 0.0 for value in checks) / len(checks)

    if all(checks):
        status = "verified"
    elif any(checks):
        status = "partially_verified"
    else:
        status = "not_verified"

    return {
        "score": round(score, 2),
        "status": status,
        "degree_verified": degree_verified,
        "field_verified": field_verified,
        "field_similarity": field_similarity,
        "candidate_education": education_items,
    }


# =====================================================
# 9. Final Score Calculation
# =====================================================

def calculate_final_score(criteria):
    available_score = 0.0
    available_weight = 0.0
    total_weight = sum(WEIGHTS.values())

    for name, weight in WEIGHTS.items():
        score = criteria[name].get("score")

        # Criteria with no information are excluded from the final score.
        if score is None:
            continue

        available_score += score * weight
        available_weight += weight

    if available_weight == 0:
        return {"matching_score": None, "evaluation_coverage": 0.0}

    return {
        "matching_score": round(available_score / available_weight, 2),
        "evaluation_coverage": round(available_weight / total_weight * 100, 2),
    }


# =====================================================
# 10. Main Matching Engine
# =====================================================

def match_cv_to_job(cv_data, job_data):
    if not isinstance(cv_data, dict):
        raise TypeError("cv_data must be a dictionary.")

    if not isinstance(job_data, dict):
        raise TypeError("job_data must be a dictionary.")

    profile = prepare_cv_profile(cv_data)

    skills_result = match_required_skills(job_data, profile)
    responsibilities_result = match_responsibilities(job_data, profile)
    experience_result = evaluate_experience(job_data, profile)
    education_result = evaluate_education(job_data, profile)

    criteria = {
        "required_skills": skills_result,
        "responsibilities": responsibilities_result,
        "experience": experience_result,
        "education": education_result,
    }

    score_result = calculate_final_score(criteria)

    final_score = score_result["matching_score"]
    coverage = score_result["evaluation_coverage"]

    if final_score is None:
        status = "Insufficient Information"
    elif final_score >= ACCEPTANCE_THRESHOLD:
        status = "Suitable for Interview"
    else:
        status = "Not Suitable Yet"

    return {
        "job_title": job_data.get("job_title", ""),
        "career_field": job_data.get("career_field", ""),
        "matching_score": final_score,
        "acceptance_threshold": ACCEPTANCE_THRESHOLD,
        "evaluation_coverage": coverage,
        "score_type": "Weighted matching score, not a probability",
        "status": status,
        "weights": WEIGHTS,
        "criteria_scores": criteria,
        "missing_skills": skills_result["missing"],
        "missing_responsibilities": responsibilities_result["missing"],
        "evaluation_notes": [
            "The matching score is not a probability of being hired.",
            "Criteria with no information are excluded from the final score.",
            "Evaluation coverage shows how much of the job criteria could be evaluated.",
            "Exact skill names are checked first, then meaning-based similarity is used.",
            "Thresholds should be validated on a labeled set of CV and job pairs.",
        ],
    }