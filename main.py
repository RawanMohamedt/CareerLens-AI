from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from cv_extractor import extract_text_from_cv as extract_text
from cv_cleaner import clean_cv_text_advanced
from contact_extractor import extract_contact_info
from skill_extractor import extract_skills_and_entities
from cv_section_parser import parse_cv_sections
from ai_resume_extractor import extract_resume_with_ai

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
AI_SUCCESS = "success"
AI_FAILED = "failed"


# =====================================================
# 1. Analysis
# =====================================================

def analyze_cv(file_path: str | Path) -> dict[str, Any]:
    """Run the full pipeline on one CV and return a result dict."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"CV file not found: {path}")

    raw_text = extract_text(str(path))
    if not raw_text or not raw_text.strip():
        raise ValueError("No text could be extracted from the CV.")

    cleaned_text = clean_cv_text_advanced(raw_text)
    if not cleaned_text or not cleaned_text.strip():
        raise ValueError("CV text is empty after cleaning.")

    contact_info = extract_contact_info(cleaned_text)

    nlp_info = extract_skills_and_entities(
        cleaned_text,
        custom_skills_list=["LangChain", "Hugging Face", "RAG", "LLM"],
    )

    sections = parse_cv_sections(cleaned_text)

    ai_info: Optional[dict] = None
    ai_error: Optional[str] = None

    try:
        ai_info = extract_resume_with_ai(cleaned_text)
        if ai_info is None:
            raise RuntimeError("AI extraction returned no result.")
    except Exception as error:
        ai_error = str(error) or type(error).__name__
        logger.warning("AI extraction failed for %s: %s", path.name, ai_error)

    return {
        "file_name": path.name,
        "contact_info": contact_info,
        "skills": nlp_info["skills"],
        "entities": nlp_info["entities"],
        "sections": sections,
        "ai_extracted_info": ai_info,
        "ai_extraction_status": AI_SUCCESS if ai_error is None else AI_FAILED,
        "ai_extraction_error": ai_error,
    }


# =====================================================
# 2. Safe file operations
# =====================================================

def _read_json(path: Path) -> Optional[dict]:
    """Return the JSON object in a file, or None if it is missing or unreadable."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read existing result %s: %s", path, error)
        return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    """
    Write JSON to a temp file in the same folder, flush it to disk, then
    replace the destination in one step. The old file stays intact if
    anything fails before the replace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=4)
            file.flush()
            os.fsync(file.fileno())

        os.replace(tmp_path, path)

    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


# =====================================================
# 3. Decision: save or keep
# =====================================================

def decide_what_to_save(
    output_file: Path,
    new_result: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """
    Returns (payload, should_write).

    - New AI extraction succeeded: write the new result (replaces old file).
    - New AI extraction failed and an old successful file exists: write nothing.
    - New AI extraction failed and no old successful file exists: write the
      failed result so the status is visible.
    """
    if new_result["ai_extraction_status"] == AI_SUCCESS:
        return new_result, True

    previous = _read_json(output_file)
    if previous and previous.get("ai_extraction_status") == AI_SUCCESS:
        return previous, False

    return new_result, True


# =====================================================
# 4. Main
# =====================================================

def process_cv_folder(cv_dir: Path, output_dir: Path) -> None:
    if not cv_dir.is_dir():
        print(f"CV folder not found: {cv_dir.resolve()}")
        return

    cv_files = sorted(
        p for p in cv_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not cv_files:
        print("No supported CV files found in the CVs folder.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for cv_file in cv_files:
        print(f"\n{'=' * 50}\nProcessing: {cv_file.name}\n{'=' * 50}")
        output_file = output_dir / f"{cv_file.stem}_analysis.json"

        try:
            result = analyze_cv(cv_file)
            payload, should_write = decide_what_to_save(output_file, result)

            if should_write:
                _write_json_atomic(output_file, payload)
                print(f"Results saved to: {output_file.resolve()}")
            else:
                print("AI failed. Previous successful result kept unchanged.")

            if result["ai_extraction_status"] == AI_FAILED:
                print(f"Note: AI extraction failed: {result['ai_extraction_error']}")

        except Exception as error:
            print(f"CV processing failed for {cv_file.name}: {error}")
            if output_file.exists():
                print("The previous result file has been preserved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process_cv_folder(Path("CVs"), Path("outputs"))
