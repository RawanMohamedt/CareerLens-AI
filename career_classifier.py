
import os
import json
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)


# ==========================================
# 1. Prepare CV data for career classification
# ==========================================

def prepare_cv_profile(cv_data):

    ai_info = cv_data.get("ai_extracted_info", {})
    sections = cv_data.get("sections", {})

    profile = {
        "professional_summary": (
            ai_info.get("professional_summary")
            or sections.get("summary", "")
        ),

        "skills": (
            ai_info.get("skills")
            or cv_data.get("skills", [])
        ),

        "experience": (
            ai_info.get("experience")
            or sections.get("experience", "")
        ),

        "projects": (
            ai_info.get("projects")
            or sections.get("projects", "")
        ),

        "education": (
            ai_info.get("education")
            or sections.get("education", "")
        )
    }

    return profile


# ==========================================
# 2. Build AI prompt
# ==========================================

def build_classification_prompt(profile):

    profile_text = json.dumps(
        profile,
        ensure_ascii=False,
        indent=2
    )

    return f"""
You are an expert Career Classification AI.

Your task is to identify the most suitable professional
specialty for a candidate based on their CV.

Analyze the candidate's:
- Skills
- Work experience
- Projects
- Professional summary
- Education

IMPORTANT RULES:

1. Identify the actual profession, not just a broad field.
2. Support your decision with evidence from the CV.
3. Consider different careers across all industries.
4. Do not restrict your classification to IT.
5. Possible careers include:
   Backend Developer, Frontend Developer,
   Full Stack Developer, UI/UX Designer,
   Graphic Designer, Data Analyst,
   Machine Learning Engineer, Accountant,
   HR Specialist, Digital Marketer,
   Content Writer, and many others.
6. Do not invent skills or experience.
7. Prioritize actual skills, projects and experience
   over job titles alone.
8. If the CV does not provide enough evidence,
   return "Insufficient Information".
9. Confidence must be High, Medium, or Low.
   It is a qualitative assessment, not a statistical probability.

CV DATA:

{profile_text}

Return ONLY valid JSON using this structure:

{{
    "career_field": "General professional field",
    "specialty": "Most suitable specific profession",
    "confidence_level": "High / Medium / Low",
    "evidence": [
        "Evidence from the CV"
    ],
    "alternative_specialties": [
        "Alternative profession"
    ],
    "explanation": "Short explanation of the decision"
}}
"""


# ==========================================
# 3. Parse AI response
# ==========================================

def parse_ai_response(response_text):

    response_text = response_text.strip()

    if response_text.startswith("```"):
        response_text = response_text.replace(
            "```json", ""
        ).replace("```", "").strip()

    result = json.loads(response_text)

    required_fields = [
        "career_field",
        "specialty",
        "confidence_level",
        "evidence",
        "alternative_specialties",
        "explanation"
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(
                f"Missing field in AI response: {field}"
            )

    return result


# ==========================================
# 4. Gemini Classification
# ==========================================

def classify_with_gemini(prompt):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    client = genai.Client(api_key=api_key)

    model = os.getenv(
        "GEMINI_MODEL",
        "gemini-flash-latest"
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    )

    return parse_ai_response(response.text)


# ==========================================
# 5. Groq Classification (Fallback)
# ==========================================

def classify_with_groq(prompt):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    client = Groq(api_key=api_key)

    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional career "
                    "classification AI. Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return parse_ai_response(
        response.choices[0].message.content
    )


# ==========================================
# 6. Main Career Classification Function
# ==========================================

def classify_career(cv_data):

    profile = prepare_cv_profile(cv_data)

    prompt = build_classification_prompt(profile)

    # Try Gemini first
    try:

        result = classify_with_gemini(prompt)

        result["provider"] = "Gemini"

        return result

    except Exception as gemini_error:

        logger.warning(
            "Gemini classification failed: %s",
            gemini_error
        )

    # Use Groq if Gemini fails
    try:

        result = classify_with_groq(prompt)

        result["provider"] = "Groq"

        return result

    except Exception as groq_error:

        raise RuntimeError(
            "Both Gemini and Groq classification failed. "
            f"Groq error: {groq_error}"
        )
