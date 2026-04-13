import re


TEXT_NORMALIZATION_VERSION = "unicode_punctuation_v1"


_UNICODE_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u00ad": "",
        "\u00b4": "'",
        "\u2007": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u200b": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2026": "...",
        "\u202f": " ",
        "\u2032": "'",
        "\u2212": "-",
        "\ufeff": " ",
    }
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_unicode_punctuation(text):
    if text is None:
        return ""
    return str(text).translate(_UNICODE_PUNCTUATION_TRANSLATION)


def normalize_text_for_vectorization(text):
    normalized = normalize_unicode_punctuation(text)
    return _WHITESPACE_RE.sub(" ", normalized).strip()
