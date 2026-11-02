"""Small text-normalisation helpers shared by search and matching code.

Customers type "violao", "violão", "VIOLÕES" or "takamine gd-20" interchangeably, so every
comparison goes through the same accent-free, lower-case, lightly-stemmed form.
"""

import re
import unicodedata

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do", "das", "dos", "e", "em",
    "no", "na", "nos", "nas", "com", "para", "pra", "por", "ate", "que", "qual", "quais", "tem",
    "voces", "vcs", "quero", "queria", "gostaria", "algum", "alguma", "modelo", "modelos",
    "opcao", "opcoes", "preco", "precos", "valor", "custa", "quanto", "r", "reais", "rs",
}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Lower-case, accent-free, alphanumerics separated by single spaces."""
    text = strip_accents(text or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def compact(text: str) -> str:
    """normalize() without spaces, so "GD-20", "gd 20" and "GD20" compare equal."""
    return normalize(text).replace(" ", "")


def stem(token: str) -> str:
    """Very light Portuguese plural stemmer (violões -> violao, guitarras -> guitarra)."""
    if len(token) <= 3 or token.isdigit():
        return token
    for suffix, replacement in (("oes", "ao"), ("aes", "ao"), ("ais", "al"), ("eis", "el"), ("res", "r"), ("s", "")):
        if token.endswith(suffix):
            return token[: -len(suffix)] + replacement
    return token


def tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    out = []
    for tok in normalize(text).split():
        if drop_stopwords and tok in STOPWORDS:
            continue
        out.append(stem(tok))
    return out


def digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")
