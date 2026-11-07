import json
from pathlib import Path

from job_description_analyzer import analyze_job_description

job_file = Path("job_description.txt")

if not job_file.exists():
    print("Job description file not found.")
    raise SystemExit

job_description = job_file.read_text(encoding="utf-8").strip()

if not job_description:
    print("Job description file is empty.")
    raise SystemExit

print("===== ANALYZING JOB DESCRIPTION =====")

try:
    result = analyze_job_description(job_description)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "job_analysis.json"

    output_file.write_text(
        json.dumps(result, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n===== ANALYSIS RESULT =====")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    print(f"\nResult saved successfully: {output_file}")

except Exception as error:
    print("\nAnalysis failed:")
    print(error)