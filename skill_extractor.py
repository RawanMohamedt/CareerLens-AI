import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, TypedDict

try:
    import spacy
    from spacy.language import Language
    from spacy.matcher import PhraseMatcher
except ImportError:
    spacy = None
    Language = Any
    PhraseMatcher = Any

logger = logging.getLogger(__name__)

MODEL_NAME = "en_core_web_sm"

# =========================================================
# 1. Multi-Domain Skills Vocabulary (Software, Data/AI, UI/UX, Product, ERP)
# =========================================================

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
    "Responsive Design", "Responsive Interfaces", "Web Development",

    # UI/UX & Product Design
    "UI/UX Design", "UX/UI Design", "UX Design", "UI Design",
    "User Research", "UX Research", "User Interviews", "Stakeholder Interviews",
    "User Flows", "User Journeys", "Journey Mapping", "Personas",
    "Information Architecture", "Wireframing", "Wireframes",
    "Prototyping", "Interactive Prototypes", "High-Fidelity UI",
    "High-Fidelity Prototypes", "Low-Fidelity Prototypes",
    "Usability", "Usability Testing", "Usability Reviews",
    "Visual Hierarchy", "Layout & Spacing", "Typography", "Color Theory",
    "Design Systems", "Design System", "Component Libraries", "UI Components",
    "Figma", "Auto Layout", "Figma Variables", "Figma Components", "Dev Mode",
    "Developer Handoff", "Sketch", "Adobe XD", "Photoshop", "Illustrator",
    "InVision", "Framer", "Zeplin", "Miro", "FigJam",
    "Accessibility", "WCAG", "Arabic RTL", "Bilingual Interfaces",
    "Micro-interactions", "Motion Design", "Brand Identity",

    # Mobile Development
    "Flutter", "React Native", "Android Development",
    "iOS Development", "Mobile App Development",

    # Databases & Storage
    "SQL", "MySQL", "PostgreSQL", "SQLite",
    "Microsoft SQL Server", "SQL Server", "T-SQL",
    "MongoDB", "Redis", "Cassandra", "Oracle",
    "Firebase", "Supabase", "NoSQL",

    # DevOps, Cloud, ERP & Tools
    "Git", "GitHub", "GitLab", "Bitbucket",
    "Docker", "Kubernetes", "CI/CD",
    "AWS", "Amazon Web Services", "Azure",
    "Google Cloud Platform", "GCP",
    "Linux", "Unix", "Jira", "Agile", "Scrum",
    "Odoo", "ERP", "SaaS", "eCommerce",
    "Object-Oriented Programming", "OOP",
    "Data Structures", "Algorithms",
    "Software Architecture", "Design Patterns",

    # Data Science, AI & ML
    "Machine Learning", "Deep Learning", "Data Analysis",
    "Data Science", "Pandas", "NumPy", "Scikit-Learn",
    "TensorFlow", "PyTorch", "Keras", "OpenCV",
    "NLP", "Natural Language Processing",
    "Computer Vision", "Matplotlib", "Seaborn",
    "Power BI", "Tableau", "Excel",
)

SKILL_ALIASES: dict[str, str] = {
    "golang": "Go",
    "react.js": "React",
    "reactjs": "React",
    "vue.js": "Vue.js",
    "vuejs": "Vue.js",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "express.js": "Express.js",
    "expressjs": "Express.js",
    "next.js": "Next.js",
    "nextjs": "Next.js",
    "asp.net core": "ASP.NET Core",
    "asp.net core mvc": "ASP.NET Core MVC",
    "entity framework core": "Entity Framework Core",
    "ef core": "Entity Framework Core",
    "rest apis": "REST API",
    "restful api": "REST API",
    "restful apis": "REST API",
    "amazon web services": "AWS",
    "google cloud platform": "GCP",
    "html5": "HTML",
    "css3": "CSS",
    "microsoft sql server": "Microsoft SQL Server",
    "t-sql": "SQL Server",
    "object-oriented programming": "OOP",
    "ux/ui design": "UI/UX Design",
    "ui & ux": "UI/UX Design",
    "ui/ux": "UI/UX Design",
    "ux research": "User Research",
    "wireframes": "Wireframing",
    "interactive prototypes": "Prototyping",
    "high-fidelity prototypes": "Prototyping",
    "design system": "Design Systems",
    "responsive interfaces": "Responsive Design",
    "usability reviews": "Usability Testing",
}

PRESERVE_CASE_SKILLS: set[str] = {
    "C", "C++", "C#", "R", "Go", "Git", "AWS", "GCP", "SQL", "OOP",
    "LINQ", "NLP", "CI/CD", "UI/UX Design", "UX Design", "UI Design",
    "WCAG", "ERP", "SaaS", "Odoo", "Adobe XD"
}


class SkillOccurrence(TypedDict):
    canonical: str
    matched_text: str
    start_char: int
    end_char: int
    confidence: float


@dataclass
class SkillExtractionResult:
    raw_skills: list[str] = field(default_factory=list)
    canonical_skills: list[str] = field(default_factory=list)
    occurrences: list[SkillOccurrence] = field(default_factory=list)


# =========================================================
# 2. Spacy Pipeline Loader (with Pure-Regex Fallback if spaCy unavailable)
# =========================================================

@lru_cache(maxsize=1)
def get_nlp_model() -> Language | None:
    if spacy is None:
        return None
    try:
        return spacy.load(MODEL_NAME)
    except OSError:
        logger.info("Model '%s' not found. Falling back to spacy.blank('en').", MODEL_NAME)
        return spacy.blank("en")


# =========================================================
# 3. Matcher Construction
# =========================================================

def canonicalize_skill_name(skill_name: str) -> str:
    cleaned = skill_name.strip()
    lower_name = cleaned.lower()

    if lower_name in SKILL_ALIASES:
        return SKILL_ALIASES[lower_name]

    if cleaned in PRESERVE_CASE_SKILLS:
        return cleaned

    if cleaned.isupper() and len(cleaned) <= 5:
        return cleaned

    return cleaned.title()


@lru_cache(maxsize=1)
def get_skill_matcher() -> tuple[PhraseMatcher | None, dict[str, str]]:
    canonical_map: dict[str, str] = {}
    for skill in DEFAULT_SKILLS:
        canonical_map[skill.lower()] = canonicalize_skill_name(skill)
    for alias, canonical in SKILL_ALIASES.items():
        canonical_map[alias.lower()] = canonical

    nlp = get_nlp_model()
    if nlp is None:
        return None, canonical_map

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(k) for k in canonical_map]
    matcher.add("SKILLS", patterns)
    return matcher, canonical_map


# =========================================================
# 4. Special skills regex matcher (C, C++, C#, R, Go, UI/UX, etc.)
# =========================================================

SPECIAL_SKILLS_PATTERNS = [
    (r"(?<![A-Za-z0-9#+])C\+\+(?![A-Za-z0-9#+])", "C++"),
    (r"(?<![A-Za-z0-9#+])C#(?![A-Za-z0-9#+])", "C#"),
    (r"(?<![A-Za-z0-9#+])\.NET(?![A-Za-z0-9#+])", ".NET"),
    (r"(?<![A-Za-z0-9#+])\.NET Core(?![A-Za-z0-9#+])", ".NET Core"),
    (r"(?<![A-Za-z0-9#+])ASP\.NET(?![A-Za-z0-9#+])", "ASP.NET"),
    (r"(?<![A-Za-z0-9#+])ASP\.NET Core(?![A-Za-z0-9#+])", "ASP.NET Core"),
    (r"(?<![A-Za-z0-9#+])C(?![A-Za-z0-9#+])", "C"),
    (r"(?<![A-Za-z0-9#+])R(?![A-Za-z0-9#+])", "R"),
    (r"(?<![A-Za-z0-9#+])Go(?![A-Za-z0-9#+])", "Go"),
]


def extract_special_skills(text: str) -> list[SkillOccurrence]:
    occurrences: list[SkillOccurrence] = []

    for pattern, canonical in SPECIAL_SKILLS_PATTERNS:
        flags = re.IGNORECASE if canonical not in {"C", "R", "Go"} else 0
        for match in re.finditer(pattern, text, flags=flags):
            matched_text = match.group(0)
            occurrences.append({
                "canonical": canonical,
                "matched_text": matched_text,
                "start_char": match.start(),
                "end_char": match.end(),
                "confidence": 0.95
            })

    return occurrences


# =========================================================
# 5. Extract Skills Function
# =========================================================

def extract_skills(text: str) -> SkillExtractionResult:
    if not text or not text.strip():
        return SkillExtractionResult()

    nlp = get_nlp_model()
    matcher, canonical_map = get_skill_matcher()

    occurrences: list[SkillOccurrence] = []
    seen_spans = set()

    if nlp is not None and matcher is not None:
        doc = nlp(text)
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            span_key = (span.start_char, span.end_char)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            matched_text = span.text.strip()
            matched_lower = matched_text.lower()
            if matched_lower in {"c", "r", "go"}:
                continue
            canonical = canonical_map.get(matched_lower, canonicalize_skill_name(matched_text))
            occurrences.append({
                "canonical": canonical,
                "matched_text": matched_text,
                "start_char": span.start_char,
                "end_char": span.end_char,
                "confidence": 0.90
            })
    else:
        # Pure-regex fallback when spaCy is not installed
        for phrase_lower, canonical in sorted(canonical_map.items(), key=lambda x: -len(x[0])):
            if phrase_lower in {"c", "r", "go", "c++", "c#", ".net", ".net core", "asp.net", "asp.net core"}:
                continue
            pattern = r"(?<![A-Za-z0-9#+])" + re.escape(phrase_lower) + r"(?![A-Za-z0-9#+])"
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                span_key = (m.start(), m.end())
                if any(not (m.end() <= s or m.start() >= e) for s, e in seen_spans):
                    continue
                seen_spans.add(span_key)
                occurrences.append({
                    "canonical": canonical,
                    "matched_text": m.group(0),
                    "start_char": m.start(),
                    "end_char": m.end(),
                    "confidence": 0.90
                })

    special_occurrences = extract_special_skills(text)
    for spec in special_occurrences:
        span_key = (spec["start_char"], spec["end_char"])
        if not any(not (spec["end_char"] <= s or spec["start_char"] >= e) for s, e in seen_spans):
            seen_spans.add(span_key)
            occurrences.append(spec)

    occurrences.sort(key=lambda x: x["start_char"])

    raw_skills = [occ["matched_text"] for occ in occurrences]
    canonical_skills = list(dict.fromkeys(occ["canonical"] for occ in occurrences))

    return SkillExtractionResult(
        raw_skills=raw_skills,
        canonical_skills=canonical_skills,
        occurrences=occurrences
    )
