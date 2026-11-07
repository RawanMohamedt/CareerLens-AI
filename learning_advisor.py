import json
import os
import re
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:
    Groq = None

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"
REQUEST_TIMEOUT = 5

# Multi-specialty official documentation & learning paths (Backend, Frontend, UI/UX, Mobile, Data/AI, DevOps, ERP)
KNOWN_PATHS = {
    # UI/UX & Product Design
    "help.figma.com": "/hc/en-us/categories/360002051613-Get-started",
    "www.nngroup.com": "/articles/",
    "m3.material.io": "/foundations",
    "www.w3.org": "/WAI/fundamentals/accessibility-intro/",
    # Backend & Databases
    "docs.python.org": "/3/tutorial/",
    "fastapi.tiangolo.com": "/tutorial/",
    "docs.djangoproject.com": "/en/stable/intro/tutorial01/",
    "www.postgresql.org": "/docs/current/tutorial.html",
    "www.odoo.com": "/documentation/master/developer.html",
    # Frontend & Web
    "developer.mozilla.org": "/en-US/docs/Learn",
    "react.dev": "/learn",
    # Mobile & Cross-Platform
    "docs.flutter.dev": "/get-started/codelab",
    # DevOps & Tools
    "docs.docker.com": "/get-started/",
    "git-scm.com": "/book/en/v2/",
    # Data Science & AI
    "scikit-learn.org": "/stable/tutorial/index.html",
    "pytorch.org": "/tutorials/",
}

TRUSTED_DOMAINS = set(KNOWN_PATHS)

# Rule-based mapping for deterministic fallback or fast domain selection across specialties
SKILL_DOMAIN_KEYWORDS = [
    (("figma", "auto layout", "variables", "variants", "prototyping", "wireframing", "dev mode", "handoff", "component"), "help.figma.com",
     "إتقان خصائص Figma المتقدمة مثل Auto Layout وVariables وDev Mode أساسي لتسليم التصميمات للمبرمجين وبناء أنظمة تصميم احترافية."),
    (("design system", "ui design", "visual hierarchy", "typography", "bilingual", "rtl", "responsive"), "m3.material.io",
     "بناء Design Systems متكاملة وتصميم واجهات متجاوبة تدعم العربية (RTL) والإنجليزية يرفع جودة وتناسق المنتجات الرقمية."),
    (("ux research", "user research", "usability", "persona", "user flow", "journey", "information architecture"), "www.nngroup.com",
     "أبحاث المستخدم (UX Research) واختبارات قابلية الاستخدام (Usability Testing) ضرورية لفهم سلوك المستخدمين وحل مشاكل التجربة الفعلية."),
    (("accessibility", "wcag"), "www.w3.org",
     "تطبيق معايير إتاحة الوصول (WCAG Accessibility) يضمن سهولة استخدام الواجهات لجميع فئات المستخدمين."),
    (("odoo", "erp"), "www.odoo.com",
     "فهم بيئة Odoo وتخصيص واجهات وقوالب أنظمة الـ ERP يمنحك ميزة تنافسية قوية في شركات البرمجيات وشركاء Odoo."),
    (("html", "css", "javascript", "typescript", "web"), "developer.mozilla.org",
     "معرفة أساسيات HTML وCSS تساعد المصمم والمطور على بناء واجهات قابلة للتنفيذ البرمجي بدقة."),
    (("react", "next.js", "frontend"), "react.dev",
     "تطوير الواجهات التفاعلية باستخدام React يعتمد على بناء المكونات القابلة لإعادة الاستخدام وإدارة الحالة."),
    (("flutter", "dart", "mobile"), "docs.flutter.dev",
     "بناء تطبيقات الهواتف المتعددة المنصات باستخدام Flutter يتطلب إتقان هيكلة الـ Widgets وإدارة الحالة."),
    (("python",), "docs.python.org",
     "إتقان لغة Python هو الأساس لبناء تطبيقات الخلفية ومعالجة البيانات والذكاء الاصطناعي."),
    (("fastapi",), "fastapi.tiangolo.com",
     "إطار FastAPI يتيح بناء واجهات برمجة تطبيقات (APIs) سريعة وعالية الأداء."),
    (("django",), "docs.djangoproject.com",
     "إطار Django يوفر بيئة متكاملة وآمنة لبناء تطبيقات الويب المؤسسية."),
    (("docker", "kubernetes", "container"), "docs.docker.com",
     "استخدام Docker يضمن تشغيل التطبيقات بنفس الكفاءة عبر بيئات التطوير والإنتاج."),
    (("sql", "postgres", "database"), "www.postgresql.org",
     "إدارة قواعد البيانات العلائقية وكتابة استعلامات SQL الفعالة ركيزة أساسية لأي نظام برمجي."),
    (("git", "github"), "git-scm.com",
     "إدارة الإصدارات عبر Git ضرورية للعمل الجماعي وتتبع التعديلات البرمجية."),
]


def get_System_prompt(target_role: str = "Target Role") -> str:
    return f"""
You are the CareerLens Learning Advisor. You help job candidates close skill gaps
for a {target_role} position.

For each skill you receive, write a short explanation and choose the official
documentation domain the candidate should learn it from.

Rules:
1. Output a single valid JSON object and nothing else. No markdown, no commentary.
2. The "skill" value must copy the input skill name exactly, character for character.
3. The "domain" value must be chosen only from the allowed domains listed in the user message.
   Never invent a domain. Never add a path. Never output a full URL.
4. The "explanation" must be one sentence in Egyptian Arabic explaining why this skill
   matters for the {target_role} role. Keep technology/tool names (Figma, Design Systems, Python, Docker, etc.) in English.
5. Do not add skills that were not requested.
6. If no domain in the allowed list fits the skill, set "domain" to "search".

Output format:
{{"skills": [{{"skill": "<exact input name>", "explanation": "<one Arabic sentence>", "domain": "<one allowed domain or search>"}}]}}
"""


_client = None


def get_client():
    global _client
    if _client is None and Groq is not None and os.getenv("GROQ_API_KEY"):
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def build_user_prompt(skill_names):
    allowed = "\n".join(f"- {domain}" for domain in sorted(TRUSTED_DOMAINS))
    skills = "\n".join(f"- {name}" for name in skill_names)
    return f"""Allowed domains:
{allowed}
- search

Skills to process:
{skills}

Return the JSON object now."""


def match_domain_by_keywords(skill_name: str) -> tuple[str | None, str]:
    lower_skill = skill_name.lower()
    for keywords, domain, explanation in SKILL_DOMAIN_KEYWORDS:
        if any(kw in lower_skill for kw in keywords):
            return domain, explanation
    return None, f"مهارة {skill_name} مطلوبة في الوصف الوظيفي وتساعدك على سد الفجوة التقنية للوظيفة."


def ask_ai_for_resources(skill_names, target_role="Target Role"):
    client = get_client()
    if client is None:
        return []

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": get_System_prompt(target_role)},
            {"role": "user", "content": build_user_prompt(skill_names)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return []

    items = data.get("skills", []) if isinstance(data, dict) else []
    return items if isinstance(items, list) else []


def build_official_url(domain):
    domain = str(domain).strip().lower()
    path = KNOWN_PATHS.get(domain)
    if path is None:
        return None
    return f"https://{domain}{path}"


def is_url_reachable(url):
    headers = {"User-Agent": "CareerLens/1.0"}
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers=headers,
            allow_redirects=True,
            stream=True,
        )
        response.close()
        return response.status_code < 400
    except requests.RequestException:
        return False


def search_fallback(skill):
    query = quote_plus(f"{skill} official documentation guide")
    return {
        "name": f"Search: {skill} official guide",
        "url": f"https://www.google.com/search?q={query}",
    }


def extract_skill_names(missing_skills):
    """
    Extract clean skill names from either a list of dicts ([{'requirement': '...'}])
    or a list of strings, splitting on case-insensitive 'or' and '/'.
    """
    names = []
    for item in (missing_skills or []):
        raw = item.get("requirement", "") if isinstance(item, dict) else str(item)
        # Do not split common compound terms like UI/UX or HTML/CSS
        protected = re.sub(r"(?i)\b(ui/ux|ux/ui|html/css|ci/cd)\b", lambda m: m.group(0).replace("/", "__SLASH__"), raw)
        parts = re.split(r"\s+or\s+|/", protected, flags=re.IGNORECASE)
        for part in parts:
            name = part.replace("__SLASH__", "/").strip()
            if name and name.lower() not in {n.lower() for n in names}:
                names.append(name)
    return names


def get_learning_links(missing_skills, target_role="Target Role", verify_reachability=False):
    skill_names = extract_skill_names(missing_skills)
    if not skill_names:
        return []

    try:
        ai_items = ask_ai_for_resources(skill_names, target_role=target_role)
    except Exception:
        ai_items = []

    ai_by_skill = {
        str(item.get("skill", "")).strip().lower(): item
        for item in ai_items
        if isinstance(item, dict) and item.get("skill")
    }

    results = []
    for skill in skill_names:
        ai_item = ai_by_skill.get(skill.lower(), {})
        kw_domain, kw_explanation = match_domain_by_keywords(skill)

        domain = ai_item.get("domain")
        if not domain or domain == "search" or domain not in TRUSTED_DOMAINS:
            domain = kw_domain

        explanation = ai_item.get("explanation") or kw_explanation
        official_url = build_official_url(domain) if domain else None

        if official_url and (not verify_reachability or is_url_reachable(official_url)):
            results.append({
                "skill": skill,
                "explanation": explanation,
                "name": f"Official Guide ({domain})",
                "url": official_url,
            })
        else:
            fb = search_fallback(skill)
            results.append({
                "skill": skill,
                "explanation": explanation,
                "name": fb["name"],
                "url": fb["url"],
            })

    return results


def build_advisor_section(matching_result_or_missing_skills, target_role="Target Role"):
    if isinstance(matching_result_or_missing_skills, dict):
        missing_skills = (
            matching_result_or_missing_skills.get("missing_skills")
            or matching_result_or_missing_skills.get("skill_gaps")
            or []
        )
        target_role = matching_result_or_missing_skills.get("job_title", target_role)
    else:
        missing_skills = matching_result_or_missing_skills or []

    links = get_learning_links(missing_skills, target_role=target_role)

    if not links:
        return {
            "type": "encouragement",
            "message": (
                "أداء ممتاز! الـ CV يطابق جميع المهارات الأساسية للوظيفة. "
                "ننصحك بمراجعة المشاريع التطبيقية والتحضير للأسئلة التقنية في المقابلة."
            ),
            "links": [],
        }

    return {
        "type": "recommendations",
        "message": f"تم تحديد {len(links)} مهارات تحتاج إلى تطوير أو توضيح في الـ CV لوظيفة {target_role}:",
        "links": links,
    }
