import json
import os
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"
REQUEST_TIMEOUT = 5

# الدومين -> المسار الثابت للصفحة الرسمية.
# الكود بيبني الرابط من هنا، والـ AI مايكتبش مسارات ولا روابط كاملة.
KNOWN_PATHS = {
    "docs.python.org": "/3/tutorial/",
    "fastapi.tiangolo.com": "/tutorial/",
    "docs.djangoproject.com": "/en/stable/intro/tutorial01/",
    "docs.docker.com": "/get-started/",
    "www.postgresql.org": "/docs/current/tutorial.html",
    "git-scm.com": "/book/en/v2",
}

TRUSTED_DOMAINS = set(KNOWN_PATHS)

SYSTEM_PROMPT = """
You are the CareerLens Learning Advisor. You help job candidates close skill gaps
for a Junior Backend Developer role.

For each skill you receive, write a short explanation and choose the official
documentation domain the candidate should learn it from.

Rules:
1. Output a single valid JSON object and nothing else. No markdown, no commentary.
2. The "skill" value must copy the input skill name exactly, character for character.
3. The "domain" value must be chosen only from the allowed domains listed in the user message.
   Never invent a domain. Never add a path. Never output a full URL.
4. The "explanation" must be one sentence in Egyptian Arabic explaining why this skill
   matters for a Junior Backend Developer. Keep technology names (Python, FastAPI, Docker, etc.) in English.
5. Do not add skills that were not requested.
6. If you are unsure which domain fits, choose the closest official documentation domain
   from the list. Never leave "domain" empty.

Output format:
{"skills": [{"skill": "<exact input name>", "explanation": "<one Arabic sentence>", "domain": "<one allowed domain>"}]}
"""

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def build_user_prompt(skill_names):
    allowed = "\n".join(f"- {domain}" for domain in sorted(TRUSTED_DOMAINS))
    skills = "\n".join(f"- {name}" for name in skill_names)

    return f"""Allowed domains:
{allowed}

Skills to process:
{skills}

Return the JSON object now."""


def ask_ai_for_resources(skill_names):
    response = get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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

    except requests.RequestException as error:
        print("URL error:", url, error)
        return False


def search_fallback(skill):
    query = quote_plus(f"{skill} official documentation")
    return {
        "name": f"Search: {skill} official documentation",
        "url": f"https://www.google.com/search?q={query}",
    }


def extract_skill_names(missing_skills):
    # "FastAPI OR Django" -> ["FastAPI", "Django"]
    names = []

    for skill in missing_skills:
        for part in skill["requirement"].split(" OR "):
            name = part.strip()
            if name and name.lower() not in {n.lower() for n in names}:
                names.append(name)

    return names


def get_learning_links(missing_skills):
    skill_names = extract_skill_names(missing_skills)

    if not skill_names:
        return []

    try:
        ai_items = ask_ai_for_resources(skill_names)
        print("AI items:", ai_items)
    except Exception as error:
        print("AI error:", error)
        ai_items = []

    requested = {name.lower() for name in skill_names}
    results = []
    covered = set()

    for item in ai_items:
        if not isinstance(item, dict):
            continue

        skill = str(item.get("skill", "")).strip()

        # نتجاهل أي مهارة الـ AI رجعها من غير ما نطلبها
        if skill.lower() not in requested or skill.lower() in covered:
            continue

        url = build_official_url(item.get("domain", ""))

        if not url or not is_url_reachable(url):
            continue

        covered.add(skill.lower())
        results.append({
            "skill": skill,
            "explanation": str(item.get("explanation", "")).strip(),
            "name": f"{skill} official documentation",
            "url": url,
        })

    # أي مهارة مش لاقية مصدر صالح، تاخد رابط بحث جاهز
    for name in skill_names:
        if name.lower() not in covered:
            results.append({
                "skill": name,
                "explanation": "",
                **search_fallback(name),
            })

    return results
ENCOURAGEMENT_MESSAGE = (
    "تهنئة، الـ CV بتاعك مناسب للوظيفة دي. "
    "الخطوة الجاية إنك تجهزي نفسك للمقابلة، وتراجعي المشاريع اللي فيه وتشرحيها بالتفصيل."
)


def build_advisor_section(result):
    # لو المرشح مناسب، مش محتاج خطة تعلم
    if result["status"] == "Suitable for Interview":
        return {
            "type": "encouragement",
            "message": ENCOURAGEMENT_MESSAGE,
        }

    # لو مش مناسب، المساعد يشتغل على المهارات الناقصة بس
    links = get_learning_links(result["missing_skills"])

    return {
        "type": "learning_plan",
        "links": links,
    }