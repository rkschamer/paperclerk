"""Text comparison for content-based duplicate detection.

A document counts as already present if (1) most of its word trigrams occur in the other document and
(2) almost all of its numbers (dates, amounts, invoice numbers) appear there as well. Condition (2) keeps
two monthly invoices of the same provider with identical boilerplate apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TRANSLIT = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d[\d.,/-]*\d")
_SEARCH_WORD = re.compile(r"[^\W\d_]{7,}")

MIN_SHINGLES = 30
NUMBER_SHARE = 0.9


def words(text: str) -> list[str]:
    return _WORD.findall(text.casefold().translate(_TRANSLIT))


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def numbers(text: str) -> set[str]:
    found = set()
    for m in _NUMBER.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if len(digits) >= 4:
            found.add(digits)
    return found


@dataclass
class Similarity:
    containment: float  # share of our trigrams found in the other text
    numbers: float  # share of our numbers found in the other text

    def is_duplicate(self, threshold: float) -> bool:
        return self.containment >= threshold and self.numbers >= NUMBER_SHARE


def compare(ours: str, theirs: str) -> Similarity | None:
    """None if our text is too short for a reliable comparison."""
    s = shingles(ours)
    if len(s) < MIN_SHINGLES:
        return None
    containment = len(s & shingles(theirs)) / len(s)
    nums = numbers(ours)
    share = len(nums & numbers(theirs)) / len(nums) if nums else 1.0
    return Similarity(containment, share)


def search_terms(text: str, limit: int = 12) -> list[str]:
    """Long, preferably rare words for the full-text search in Paperless."""
    seen: set[str] = set()
    terms: list[str] = []
    for w in sorted(_SEARCH_WORD.findall(text[:6000]), key=len, reverse=True):
        k = w.casefold()
        if k not in seen and k not in ("and", "not"):
            seen.add(k)
            terms.append(w)
        if len(terms) >= limit:
            break
    return terms
