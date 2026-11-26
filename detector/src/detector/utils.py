import re
import unicodedata

REPLACEMENTS = str.maketrans({
    "’": "`",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    "…": "...",
    "ß": "ss",
    "æ": "ae",
    "Æ": "AE",
    "œ": "oe",
    "Œ": "OE",
})

MARKDOWN = re.compile(
    r"(?m)"
    r"^[ \t]{0,3}(?:#{1,6}[ \t]+|>[ \t]+|```|~~~)"
    r"|^[ \t]*(?:[-*+]|\d+[.)])[ \t]+\S"
    r"|\*\*[^*\n]+\*\*|__[^_\n]+__"
    r"|`[^`\n]+`"
    r"|\[[^\[\]\n]+\]\([^\s)]+\)"
)

AI_PHRASE = re.compile(
    r"\b(?:delve into|in today['’]s fast-paced world|"
    r"it['’]s worth noting|a testament to|"
    r"multifaceted|tapestry|underscores)\b",
    re.IGNORECASE,
)

CONTRAST = re.compile(
    r"\bnot\s+(?:just|only)\s+[^.!?\n]{1,120}?\bbut\b[ \t]+\w",
    re.IGNORECASE,
)

AI_OPENING = re.compile(
    r"\A\s*(?:(?:certainly|absolutely|of course)[!,.]\s*)?"
    r"(?:here['’]s|here is)\b",
    re.IGNORECASE,
)


def sanitise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.translate(REPLACEMENTS))
    return "".join(
        char for char in text if " " <= char <= "~" or char in "\t\n\r"
    )


def _is_unusual_unicode(char: str) -> bool:
    if char in "\t\n\r":
        return False
    if char.isascii():
        return not char.isprintable()
    return unicodedata.category(char)[0] in "PSZC"


def common_identifiers(text: str) -> dict[str, bool]:
    """Return stylistic flags from the original text, before sanitising it."""
    return {
        "em_dash": "—" in text,
        "markdown": bool(MARKDOWN.search(text)),
        "ai_phrase": bool(AI_PHRASE.search(text)),
        "contrast_phrase": bool(CONTRAST.search(text)),
        # "unusual_unicode": any(_is_unusual_unicode(char) for char in text),
        "ai_opening": bool(AI_OPENING.search(text)),
    }


def is_ai_text(text: str) -> bool:
    return any(common_identifiers(text).values())
