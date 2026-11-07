import os
import json
import logging

from dotenv import load_dotenv
from google import genai
from groq import Groq
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


# ==========================================
# 1. Build Job Description Analysis Prompt
# ==========================================

def build_job_analysis_prompt(job_description):

    return f"""
You are a professional job description analyzer.

Analyze the following job description carefully.

The job can belong to ANY industry or career field,
not only software development.

Extract the information and return ONLY valid JSON
using this exact structure:

{{
    "job_title": "",
    "career_field": "",
    "seniority_level": "",
    "required_skills": [],
    "preferred_skills": [],
    "responsibilities": [],
    "experience_requirements": [],
    "education_requirements": [],
    "languages": [],
    "employment_type": "",
    "location": "",
    "keywords": [],
    "summary": ""
}}

Important instructions:

- Extract information only from the job description.
- Do not invent missing requirements.
- Use empty strings when information is unavailable.
- Use empty lists when no information is available.
- Separate required skills from preferred skills.
- Identify the specific career field and job title.
- Keep the output clear and structured.

JOB DESCRIPTION:

{job_description}
"""


# ==========================================
# 2. Parse AI Response
# ==========================================

def parse_json_response(response_text):

    response_text = response_text.strip()

    if response_text.startswith("```"):
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        response_text = response_text.strip()

    return json.loads(response_text)


# ==========================================
# 3. Analyze with Groq
# ==========================================

def analyze_with_groq(prompt):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    client = Groq(
        api_key=api_key,
        timeout=30.0,
        max_retries=0
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a professional job description analyzer. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )

    result = parse_json_response(
        response.choices[0].message.content
    )

    result["provider"] = "Groq"

    return result


# ==========================================
# 4. Analyze with Gemini
# ==========================================

def analyze_with_gemini(prompt):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30,
            retry_options=types.HttpRetryOptions(attempts=1)
        )
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    result = parse_json_response(response.text)
    result["provider"] = "Gemini"

    return result


# ==========================================
# 5. Main Analysis Function
# ==========================================

def analyze_job_description(job_description):

    if not isinstance(job_description, str) or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    prompt = build_job_analysis_prompt(job_description)

    # Groq أولاً لأنه أسرع وشغال عندك، وGemini احتياطي
    groq_error = None

    try:
        return analyze_with_groq(prompt)

    except Exception as error:
        groq_error = error
        logger.warning("Groq analysis failed: %s", error)

    try:
        return analyze_with_gemini(prompt)

    except Exception as gemini_error:
        logger.error("Gemini analysis failed: %s", gemini_error)

        raise RuntimeError(
            "Both Groq and Gemini failed. "
            f"Groq error: {groq_error}. "
            f"Gemini error: {gemini_error}."
        ) from gemini_error