
import json
from pathlib import Path

cv_file = Path(
    "outputs/d0b8e0a24dd04d3f8c9ab06a6e1fb69d_analysis.json"
)

cv_data = json.loads(cv_file.read_text(encoding="utf-8"))
ai = cv_data.get("ai_extracted_info", {})

print("===== CV SKILLS =====")
print(json.dumps(ai.get("skills", []), indent=4, ensure_ascii=False))

print("\n===== CV EXPERIENCE =====")
print(json.dumps(ai.get("experience", []), indent=4, ensure_ascii=False))

print("\n===== CV PROJECTS =====")
print(json.dumps(ai.get("projects", []), indent=4, ensure_ascii=False))

print("\n===== CV EDUCATION =====")
print(json.dumps(ai.get("education", []), indent=4, ensure_ascii=False))

print("\n===== CV SUMMARY =====")
print(ai.get("professional_summary", ""))
