
import json
from pathlib import Path

from matching_engine import match_cv_to_job


def load_json(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():

    print("\n===== CareerLens Matching Engine =====\n")

    cv_path = input("Enter CV JSON path: ").strip().strip('"')
    job_path = input("Enter Job JSON path: ").strip().strip('"')

    try:
        cv_data = load_json(cv_path)
        job_data = load_json(job_path)

        result = match_cv_to_job(
            cv_data=cv_data,
            job_data=job_data
        )

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / "matching_result.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=4
            )

        print("\n===== Matching Result =====")
        print("Job:", result["job_title"])
        print("Score:", result["matching_score"])
        print("Coverage:", result["evaluation_coverage"])
        print("Status:", result["status"])

        print("\nMissing Skills:")
        for skill in result["missing_skills"]:
            print("-", skill["requirement"])

        print("\nResult saved to:", output_path)

    except (FileNotFoundError, json.JSONDecodeError, TypeError, KeyError) as error:
        print("\nError:", error)


if __name__ == "__main__":
    main()
