import json
from pathlib import Path

from learning_advisor import build_advisor_section

result = json.loads(Path("outputs/matching_result.json").read_text(encoding="utf-8"))

section = build_advisor_section(result)

print("Type:", section["type"])

if section["type"] == "encouragement":
    print(section["message"])
else:
    for link in section["links"]:
        print("-", link["skill"], "|", link["name"], "->", link["url"])