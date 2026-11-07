import re
import unicodedata


def clean_cv_text_advanced(text: str) -> str:
    """
    Clean extracted CV text while preserving important information,
    technical terms, punctuation, and section boundaries.
    """

    # 1. Handle empty or invalid input
    if text is None:
        return ""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return ""

    # 2. Normalize Unicode without compatibility decomposition
    text = unicodedata.normalize("NFC", text)

    # 3. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Replace tabs and unusual spacing characters with spaces
    text = re.sub(r"[\t\f\v\u00a0\u202f\u2007]", " ", text)

    # 5. Remove invisible formatting characters that may
    # interfere with NLP, while preserving line breaks.
    text = re.sub(
        r"[\u200b\u200c\u200d\u2060\ufeff]",
        "",
        text
    )

    # 6. Remove control characters except newline
    text = "".join(
        char
        for char in text
        if char == "\n"
        or unicodedata.category(char) != "Cc"
    )

    # 7. Remove common decorative bullets and check marks.
    # Preserve technical punctuation such as +, #, ., /, -, and |
    text = re.sub(
        r"[•▪►■◆★✓✔☑☒]",
        " ",
        text
    )

    # 8. Normalize repeated spaces inside each line
    lines = []

    for line in text.split("\n"):
        line = re.sub(r" {2,}", " ", line).strip()
        lines.append(line)

    # 9. Collapse long blank-line sequences to one blank line.
    # Keep section separation instead of deleting every empty line.
    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    # 10. Remove unnecessary spaces around line boundaries
    cleaned_text = re.sub(r" *\n *", "\n", cleaned_text)

    return cleaned_text.strip()


# =========================
# Example usage
# =========================

# raw_text = extract_text("my_cv.pdf")
# cleaned_text = clean_cv_text_advanced(raw_text)
# print(cleaned_text)