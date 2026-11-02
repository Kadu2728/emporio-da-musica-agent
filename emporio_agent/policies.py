"""Policy-manual ingestion and retrieval.

Retrieval design: the manual is 8 pages / 10 numbered sections (~3k tokens). Each section is a
self-contained topic (hours, payment, returns, shipping...), so the section *is* the natural
chunk. Instead of embeddings + similarity search, the LLM picks one or more topics from an enum
(`get_store_policy` tool) and receives the full, verbatim section(s). For a corpus this size that
is exact, cheap, deterministic and easy to debug; see README for when this would change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

from . import DATA_DIR

POLICY_PDF = DATA_DIR / "politicas_da_loja.pdf"

# Known inconsistencies in the manual, fixed at ingestion time so the agent never repeats them.
# Each one is an assumption documented in the README ("Data treatment").
ERRATA = [
    # Contact table has an accented, invalid e-mail domain; the cover page has the correct one.
    ("contato@empóriodamusica.com.br", "contato@emporiodamusica.com.br"),
    # Section 7 lists a WhatsApp number that contradicts the company data table (1.2) and the cover,
    # which both say the phone/WhatsApp is (67) 3341-4444. We trust the official data table.
    ("(67) 3321-4500", "(67) 3341-4444"),
]

# topic id -> (section number, short description used in the tool schema)
TOPICS: dict[str, tuple[int, str]] = {
    "sobre_e_contato": (1, "about the store, what it sells (only instruments, no accessories), address, phone, e-mail, CNPJ"),
    "horario": (2, "opening hours, holidays, special dates"),
    "pagamento": (3, "payment methods, PIX discount, installments, combining payment methods"),
    "trocas_e_devolucoes": (4, "returns, regret/withdrawal right, exchanges for defect or preference, non-eligible items"),
    "frete_e_entregas": (5, "shipping, delivery fees and deadlines, carriers, tracking codes"),
    "promocoes": (6, "promotion calendar and promotion rules (non-cumulative, stock, rain check)"),
    "atendimento": (7, "service guidelines: out-of-stock, discontinued, expired promotions, complaints"),
    "garantia": (8, "legal and manufacturer warranty, what warranty does not cover"),
    "privacidade": (9, "personal data / LGPD, data deletion requests"),
}


@dataclass(frozen=True)
class PolicySection:
    number: int
    text: str


def _clean(raw: str) -> str:
    text = re.sub(r"\s+", " ", raw)  # pypdf breaks the line between almost every word
    text = text.replace("Empório da Música Manual de Políticas e Procedimentos", " ")
    text = re.sub(r"Página \d+", " ", text)
    text = re.sub(r"\(67\)\s*(\d{4})\s*-\s*(\d{4})", r"(67) \1-\2", text)
    text = re.sub(r"\s*•\s*", "\n• ", text)  # bullets
    text = re.sub(r"\s(\d{1,2}\.\d)\s(?=[A-ZÁÉÍÓÚ])", r"\n\1 ", text)  # sub-headings ("3.1 Regras ...")
    text = re.sub(r" {2,}", " ", text)
    for wrong, right in ERRATA:
        text = text.replace(wrong, right)
    return text.strip()


def parse_sections(text: str) -> list[PolicySection]:
    """Split on top-level headings ("1. Sobre ...", "2. Horário ...").

    Numbered lists inside sections ("1. Saudação: ...") are skipped because a heading is only
    accepted if it carries the *next expected* section number.
    """
    sections: list[PolicySection] = []
    expected, starts = 1, []
    for match in re.finditer(r"(?<![\d.])(\d{1,2})\.\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])", text):
        if int(match.group(1)) == expected:
            starts.append((expected, match.start(), match.end()))
            expected += 1
    for i, (number, start, body_start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        sections.append(PolicySection(number, f"{number}. {text[body_start:end].strip()}"))
    return sections


@lru_cache(maxsize=1)
def load_sections(pdf_path: Path = POLICY_PDF) -> dict[int, PolicySection]:
    reader = PdfReader(str(pdf_path))
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    return {s.number: s for s in parse_sections(_clean(raw))}


def get_sections(topics: list[str]) -> list[PolicySection]:
    sections = load_sections()
    numbers = sorted({TOPICS[t][0] for t in topics if t in TOPICS})
    return [sections[n] for n in numbers if n in sections]
