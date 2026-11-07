import re
from urllib.parse import urlsplit


def extract_contact_info(text: str) -> dict:
    """
    Extract email, phone, LinkedIn, GitHub, and Portfolio (Behance/Dribbble/Web)
    from CV text. Returns the first detected value for each field.
    """

    result = {
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "portfolio": None,
    }

    if not isinstance(text, str) or not text.strip():
        return result

    # 1. Email
    email_pattern = (
        r"(?<![\w.+-])"
        r"[a-zA-Z0-9_.%+-]+(?:\.[a-zA-Z0-9_.%+-]+)*"
        r"@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+"
        r"(?![\w.-])"
    )

    emails = re.findall(email_pattern, text)
    if emails:
        result["email"] = emails[0]

    # 2. LinkedIn, GitHub, and Portfolio URLs (supports dots in slugs & Behance/Dribbble)
    url_pattern = re.compile(
        r"(?i)(?<![\w@])"
        r"(?:https?://)?(?:www\.)?"
        r"(?:linkedin\.com/in/|github\.com/|behance\.net/|dribbble\.com/)"
        r"[a-zA-Z0-9_.-]+"
        r"(?:/[^\s<>]*)?"
    )

    for match in url_pattern.finditer(text):
        candidate = match.group(0).rstrip(".,;:!)]}")

        normalized = (
            candidate
            if candidate.lower().startswith(("http://", "https://"))
            else "https://" + candidate
        )

        parsed = urlsplit(normalized)
        host = (parsed.hostname or "").lower()
        path_parts = [part for part in parsed.path.split("/") if part]

        if not path_parts:
            continue

        username = path_parts[0]

        if host.endswith("linkedin.com"):
            if username.lower() == "in" and len(path_parts) >= 2 and not result["linkedin"]:
                result["linkedin"] = f"https://www.linkedin.com/in/{path_parts[1]}"

        elif host.endswith("github.com"):
            if username.lower() not in {
                "features", "topics", "collections",
                "marketplace", "login", "signup",
                "settings", "explore", "orgs"
            } and not result["github"]:
                result["github"] = f"https://github.com/{username}"

        elif host.endswith(("behance.net", "dribbble.com")):
            if username.lower() not in {"gallery", "shots", "search", "explore"} and not result["portfolio"]:
                domain = "behance.net" if host.endswith("behance.net") else "dribbble.com"
                result["portfolio"] = f"https://www.{domain}/{username}"
            elif not result["portfolio"]:
                result["portfolio"] = normalized

    # 3. Phone candidates (Supports Egyptian 010/011/012/015, +20..., and international numbers)
    phone_pattern = re.compile(
        r"(?<!\w)"
        r"(?:\+\d{1,3}[\s.-]*)?"
        r"(?:\(?\d{2,4}\)?[\s.-]*)?"
        r"\d{3,4}[\s.-]*\d{3,5}"
        r"(?!\w)"
    )

    for match in phone_pattern.finditer(text):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)

        if not 10 <= len(digits) <= 15:
            continue

        is_egyptian_mobile = (
            len(digits) == 11
            and digits.startswith(("010", "011", "012", "015"))
        ) or (
            len(digits) == 12
            and digits.startswith(("2010", "2011", "2012", "2015"))
        )

        is_international = (
            candidate.startswith("+")
            and 10 <= len(digits) <= 15
        )

        if is_egyptian_mobile or is_international:
            result["phone"] = candidate
            break

    return result
