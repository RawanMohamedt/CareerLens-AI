
import json
from pathlib import Path

from career_classifier import classify_career

# Get all CV analysis files
outputs_folder = Path("outputs")
cv_files = sorted(outputs_folder.glob("*_analysis.json"))

if not cv_files:
    print("No CV analysis files found in outputs folder.")
    raise SystemExit

# Display available CV files
print("\n===== AVAILABLE CV FILES =====")

for index, file_path in enumerate(cv_files, start=1):
    print(f"{index}. {file_path.name}")

# Let the user choose a CV
try:
    choice = int(input("\nChoose CV number: "))

    if choice < 1 or choice > len(cv_files):
        print("Invalid choice.")
        raise SystemExit

except ValueError:
    print("Please enter a valid number.")
    raise SystemExit

# Load selected CV
selected_file = cv_files[choice - 1]

with open(selected_file, "r", encoding="utf-8") as file:
    cv_data = json.load(file)

print(f"\nSelected CV: {selected_file.name}")
print("\nClassifying career...")

# Run classification
result = classify_career(cv_data)

# Display results
print("\n===== CAREER CLASSIFICATION RESULT =====")
print("Career Field:", result["career_field"])
print("Specialty:", result["specialty"])
print("Confidence:", result["confidence_level"])
print("Provider:", result["provider"])

print("\nEvidence:")
for item in result["evidence"]:
    print("-", item)

print("\nAlternative Specialties:")
for item in result["alternative_specialties"]:
    print("-", item)

print("\nExplanation:")
print(result["explanation"])
