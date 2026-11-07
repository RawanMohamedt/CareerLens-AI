import re
import unicodedata

SECTION_ALIASES = {
    "summary": [
        "summary",
        "professional summary",
        "profile",
        "professional profile",
        "career profile",
        "career summary",
        "objective",
        "career objective",
        "about me",
        "about",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "work experience and internships",
        "internships",
        "internship experience",
        "professional experience and internships",
    ],
    "education": [
        "education",
        "educational background",
        "academic background",
        "academic qualifications",
        "education and qualifications",
    ],
    "skills": [
        "skills",
        "technical skills",
        "core skills",
        "key skills",
        "technical competencies",
        "technical expertise",
        "tools and technologies",
        "technologies",
        "tech stack",
    ],
    "soft_skills": [
        "soft skills",
        "personal skills",
        "interpersonal skills",
    ],
    "projects": [
        "projects",
        "personal projects",
        "academic projects",
        "selected projects",
        "key projects",
        "project experience",
        "project portfolio",
    ],
    "certifications": [
        "certifications",
        "certificates",
        "courses",
        "courses and certifications",
        "training",
        "training and certifications",
        "professional development",
        "licenses and certifications",
        "certifications and training",
        "workshops",
    ],
    "languages": [
        "languages",
        "language skills",
        "languages spoken",
    ],
    "achievements": [
        "achievements",
        "awards",
        "honors",
        "honours",
        "accomplishments",
    ],
    "volunteering": [
        "volunteering",
        "volunteer experience",
        "volunteer work",
        "community involvement",
    ],
    "interests": [
        "interests",
        "hobbies",
        "interests and hobbies",
    ],
    "references": [
        "references",
        "professional references",
    ],
}


def normalize_heading(line: str) -> str:
    """Normalize a possible CV section heading."""
    line = unicodedata.normalize("NFKC", line)
    line = line.strip().lower()

    # Remove common bullets and numbering at the start.
    line = re.sub(
        r"^\s*(?:[-*•·+]|\d{1,2}[.])\s*",
        "",
        line,
    )

    # Remove common separators at the end.
    line = re.sub(r"[: |]+\s*$", "", line)

    # Normalize whitespace.
    line = re.sub(r"\s+", " ", line)

    return line.strip(" \t---|: :")


NORMALIZED_ALIASES = {
    normalize_heading(alias): section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}


def detect_section_heading(line: str):
    """Return a section key only for an exact recognized heading."""
    heading = normalize_heading(line)

    if not heading:
        return None

    return NORMALIZED_ALIASES.get(heading)


def detect_inline_section_heading(line: str) -> tuple[str | None, str]:
    """
    Detect inline headings such as 'Skills: Python, SQL' or 'Languages: Arabic, English'.
    Returns (section_key, remainder_text).
    """
    if ":" not in line:
        return None, ""
    prefix, remainder = line.split(":", 1)
    section = detect_section_heading(prefix)
    if section and remainder.strip():
        return section, remainder.strip()
    return None, ""


def parse_cv_sections(text: str) -> dict:
    """Split CV text into recognized sections without discarding text."""
    section_names = [
        "other",
        "summary",
        "experience",
        "education",
        "skills",
        "soft_skills",
        "projects",
        "certifications",
        "languages",
        "achievements",
        "volunteering",
        "interests",
        "references",
    ]

    sections = {name: [] for name in section_names}
    current_section = "other"

    for line in text.splitlines():
        cleaned_line = line.strip()

        if not cleaned_line:
            continue

        detected_section = detect_section_heading(cleaned_line)
        if detected_section:
            current_section = detected_section
            continue

        inline_section, remainder = detect_inline_section_heading(cleaned_line)
        if inline_section:
            current_section = inline_section
            sections[current_section].append(remainder)
            continue

        sections[current_section].append(cleaned_line)

    return {
        section: "\n".join(lines).strip()
        for section, lines in sections.items()
    }
