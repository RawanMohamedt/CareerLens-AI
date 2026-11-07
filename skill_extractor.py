import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, TypedDict

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher

logger = logging.getLogger(__name__)

MODEL_NAME = "en_core_web_sm"


# =====================================================
# 1. Skills vocabulary
# =====================================================

DEFAULT_SKILLS: tuple[str, ...] = (
    # Programming languages
    "Python", "Java", "JavaScript", "TypeScript",
    "C", "C++", "C#", "PHP", "R", "Ruby", "Go",
    "GoLang", "Kotlin", "Swift", "Dart", "Scala",
    "Bash", "PowerShell",

    # Web development
    "HTML", "HTML5", "CSS", "CSS3",
    "React", "React.js", "Angular", "Vue.js",
    "Node.js", "Express.js", "Next.js",
    "ASP.NET", "ASP.NET Core", "ASP.NET Core MVC",
    ".NET", ".NET Core", "Entity Framework",
    "Entity Framework Core", "LINQ",
    "REST API", "REST APIs", "GraphQL",
    "FastAPI", "Flask", "Django",
    "Bootstrap", "Tailwind CSS",
    "Responsive Design", "Web Development",

    # Databases
    "SQL", "SQL Server", "MySQL", "PostgreSQL",
    "SQLite", "MongoDB", "Oracle Database",
    "Redis", "Database Design", "Database Management",

    # Data analysis
    "Pandas", "NumPy", "SciPy",
    "Matplotlib", "Seaborn",
    "Excel", "Microsoft Excel",
    "Power BI", "Tableau",
    "Data Analysis", "Data Visualization",
    "Statistics", "Exploratory Data Analysis",
    "ETL", "Data Cleaning",

    # AI and machine learning
    "Artificial Intelligence", "AI",
    "Machine Learning", "Deep Learning",
    "Natural Language Processing", "NLP",
    "Computer Vision",
    "Generative AI", "Large Language Models",
    "LLM", "LLMs", "RAG",
    "Retrieval-Augmented Generation",
    "Prompt Engineering", "LangChain",
    "Hugging Face", "OpenAI API",
    "Google Gemini", "Gemini API",
    "Scikit-learn", "TensorFlow", "PyTorch",
    "Keras", "XGBoost",
    "Feature Engineering", "Model Evaluation",
    "Model Training", "Neural Networks",
    "spaCy", "NLTK", "Transformers",
    "Streamlit", "Gradio",

    # UI/UX and design
    "UI/UX", "UI Design", "UX Design",
    "UI/UX Design", "User Interface Design",
    "User Experience Design", "Figma",
    "Adobe XD", "Adobe Illustrator",
    "Adobe Photoshop", "Illustrator", "Photoshop",
    "Wireframing", "Prototyping",
    "User Research", "Usability Testing",
    "Interaction Design", "Information Architecture",
    "Design Systems", "User Flows", "Visual Design",

    # Software engineering
    "Object-Oriented Programming", "OOP",
    "Data Structures", "Algorithms",
    "Software Engineering", "Software Development",
    "MVC", "Dependency Injection",
    "Unit Testing", "Debugging",
    "Clean Code", "Design Patterns",

    # DevOps, cloud, and tools
    "Git", "GitHub", "GitLab",
    "Docker", "Kubernetes",
    "AWS", "Amazon Web Services",
    "Microsoft Azure", "Azure",
    "Google Cloud", "GCP", "CI/CD", "Linux",
    "Visual Studio", "Visual Studio Code",
    "VS Code", "Jupyter Notebook",
    "Postman", "Swagger",

    # General professional skills
    "Requirements Analysis",
    "Problem Solving", "Communication",
    "Teamwork", "Time Management",
    "Critical Thinking", "Project Management",
)

# Short or common-word skills that must match with exact casing
# to avoid false positives (e.g. "go" in a sentence, "r" in a list).
CASE_SENSITIVE_SKILLS: frozenset[str] = frozenset({"C", "R", "Go", "AI"})


# =====================================================
# 2. Heuristic hint lists (matched as whole words)
# =====================================================

JOB_TITLE_HINTS = (
    "developer", "engineer", "designer", "manager", "intern",
    "trainee", "student", "analyst", "specialist", "consultant",
    "administrator", "architect",
)

LOCATION_HINTS = (
    "minufiyah", "menoufia", "egypt", "cairo", "alexandria", "governorate",
)

ORGANIZATION_HINTS = (
    "university", "college", "faculty", "institute", "school",
    "scholarship", "platform", "company", "corporation", "foundation",
    "technologies", "ltd", "inc",
)

EDUCATION_HINTS = (
    "bachelor's degree", "bachelor degree", "master's degree",
    "master degree", "phd", "b.sc", "m.sc", "computer science", "degree in",
)

CONTACT_HINTS = (
    "email:", "phone:", "linkedin", "github", "www.", "http://", "https://",
)


def _compile_hints(hints: tuple[str, ...]) -> re.Pattern:
    """Build one regex that matches any hint as a whole phrase."""
    alternatives = "|".join(re.escape(hint) for hint in hints)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


JOB_TITLE_RE = _compile_hints(JOB_TITLE_HINTS)
LOCATION_RE = _compile_hints(LOCATION_HINTS)
ORGANIZATION_RE = _compile_hints(ORGANIZATION_HINTS)
EDUCATION_RE = _compile_hints(EDUCATION_HINTS)
CONTACT_RE = _compile_hints(CONTACT_HINTS)


# =====================================================
# 3. Model loading
# =====================================================

@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """Load the spaCy model once, with a clear error if it is missing."""
    try:
        return spacy.load(MODEL_NAME)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{MODEL_NAME}' is not installed. "
            f"Run: python -m spacy download {MODEL_NAME}"
        ) from exc


# =====================================================
# 4. Skill matching
# =====================================================

@lru_cache(maxsize=8)
def get_skill_matchers(
    skills: tuple[str, ...],
) -> tuple[PhraseMatcher, PhraseMatcher]:
    """
    Return two matchers:
    - loose: case-insensitive, for most skills
    - strict: exact-casing, for ambiguous short skills
    """
    nlp = get_nlp()

    loose = PhraseMatcher(nlp.vocab, attr="LOWER")
    strict = PhraseMatcher(nlp.vocab, attr="ORTH")

    loose_patterns = []
    strict_patterns = []

    for skill in skills:
        pattern = nlp.make_doc(skill)
        if skill in CASE_SENSITIVE_SKILLS:
            strict_patterns.append(pattern)
        else:
            loose_patterns.append(pattern)

    if loose_patterns:
        loose.add("SKILL", loose_patterns)
    if strict_patterns:
        strict.add("SKILL_STRICT", strict_patterns)

    return loose, strict


def find_skills(
    doc,
    skills: tuple[str, ...],
    display_names: dict[str, str],
) -> list[str]:
    """Return the skills found in the doc, in order of appearance."""
    loose, strict = get_skill_matchers(skills)

    # Collect (start, end, canonical_name) for every match.
    candidates: list[tuple[int, int, str]] = []

    for _, start, end in loose(doc):
        key = doc[start:end].text.casefold()
        candidates.append((start, end, display_names[key]))

    for _, start, end in strict(doc):
        text = doc[start:end].text
        candidates.append((start, end, text))

    # Prefer longer matches so "ASP.NET Core MVC" beats "ASP.NET".
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))

    selected_spans: list[tuple[int, int]] = []
    found: list[tuple[int, str]] = []
    seen_names: set[str] = set()

    for start, end, name in candidates:
        overlaps = any(
            start < s_end and end > s_start
            for s_start, s_end in selected_spans
        )
        if overlaps:
            continue

        selected_spans.append((start, end))

        if name.casefold() not in seen_names:
            seen_names.add(name.casefold())
            found.append((start, name))

    found.sort(key=lambda item: item[0])
    return [name for _, name in found]


# =====================================================
# 5. Entity cleaning
# =====================================================

_FRAGMENT_SPLIT_RE = re.compile(r"[\r\n|•;]+")
_WHITESPACE_RE = re.compile(r"\s+")


def split_entity_fragments(entity_text: str) -> list[str]:
    """Split an entity span that accidentally spans several lines."""
    cleaned = []
    for fragment in _FRAGMENT_SPLIT_RE.split(entity_text):
        fragment = _WHITESPACE_RE.sub(" ", fragment).strip()
        fragment = fragment.strip(" \t-–—:|,;")
        if fragment:
            cleaned.append(fragment)
    return cleaned


def _is_contact_or_header(text: str) -> bool:
    return CONTACT_RE.search(text) is not None


def is_probably_not_person(text: str, skill_keys: set[str]) -> bool:
    words = text.split()

    if not 2 <= len(words) <= 6:
        return True
    if len(text) > 70:
        return True
    if _is_contact_or_header(text):
        return True
    if text.casefold() in skill_keys:
        return True
    return any(
        pattern.search(text) is not None
        for pattern in (JOB_TITLE_RE, LOCATION_RE, ORGANIZATION_RE, EDUCATION_RE)
    )


def is_probably_not_organization(text: str, skill_keys: set[str]) -> bool:
    if len(text) > 120:
        return True
    if _is_contact_or_header(text):
        return True
    if text.casefold() in skill_keys:
        return True
    return any(
        pattern.search(text) is not None
        for pattern in (EDUCATION_RE, JOB_TITLE_RE)
    )


# =====================================================
# 6. Entity extraction
# =====================================================

class Entities(TypedDict):
    persons: list[str]
    organizations: list[str]
    candidate_name: Optional[str]


class ExtractionResult(TypedDict):
    skills: list[str]
    entities: Entities


def extract_entities(doc, skill_keys: set[str]) -> Entities:
    persons: list[str] = []
    organizations: list[str] = []

    for ent in doc.ents:
        for fragment in split_entity_fragments(ent.text):
            if fragment.casefold() in skill_keys:
                continue

            # spaCy sometimes labels companies or platforms as PERSON.
            if ORGANIZATION_RE.search(fragment):
                if not is_probably_not_organization(fragment, skill_keys):
                    organizations.append(fragment)
                continue

            if ent.label_ == "PERSON":
                if not is_probably_not_person(fragment, skill_keys):
                    persons.append(fragment)
            elif ent.label_ == "ORG":
                if not is_probably_not_organization(fragment, skill_keys):
                    organizations.append(fragment)

    persons = list(dict.fromkeys(persons))
    organizations = list(dict.fromkeys(organizations))

    return {
        "persons": persons,
        "organizations": organizations,
        # Heuristic only. The AI extraction should be the primary source.
        "candidate_name": persons[0] if persons else None,
    }


# =====================================================
# 7. Public API
# =====================================================

def _build_skill_vocabulary(custom_skills: Optional[list[str]]) -> tuple[
    tuple[str, ...], dict[str, str]
]:
    """Merge default and custom skills, deduplicated case-insensitively."""
    display_names: dict[str, str] = {}

    for skill in [*DEFAULT_SKILLS, *(custom_skills or [])]:
        if isinstance(skill, str) and skill.strip():
            cleaned = skill.strip()
            display_names.setdefault(cleaned.casefold(), cleaned)

    skills = tuple(display_names.values())
    return skills, display_names


def extract_skills_and_entities(
    text: str,
    custom_skills_list: Optional[list[str]] = None,
) -> ExtractionResult:
    empty: ExtractionResult = {
        "skills": [],
        "entities": {
            "persons": [],
            "organizations": [],
            "candidate_name": None,
        },
    }

    if not isinstance(text, str) or not text.strip():
        return empty

    try:
        nlp = get_nlp()
    except RuntimeError:
        logger.exception("Failed to load spaCy model")
        raise

    doc = nlp(text)

    skills, display_names = _build_skill_vocabulary(custom_skills_list)
    found_skills = find_skills(doc, skills, display_names)

    skill_keys = {skill.casefold() for skill in found_skills}
    entities = extract_entities(doc, skill_keys)

    logger.debug(
        "Extracted %d skills and %d persons",
        len(found_skills),
        len(entities["persons"]),
    )

    return {"skills": found_skills, "entities": entities}