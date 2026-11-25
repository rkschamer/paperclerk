"""Turn model output into clean Paperless values."""

from __future__ import annotations

import re
from datetime import date, timedelta

from rapidfuzz import fuzz, process

_LEGAL = {
    "gmbh", "mbh", "ag", "se", "kg", "kgaa", "ohg", "gbr", "ug", "ev", "e.v", "ek", "e.k", "co", "&",
    "inc", "ltd", "llc", "plc", "sa", "s.a", "bv", "b.v", "nv", "n.v", "sarl", "s.a.r.l", "corp",
    "corporation", "limited", "haftungsbeschränkt", "eg", "e.g", "ggmbh", "gmbh&co", "gmbh&co.kg",
}
_TRANSLIT = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "&": " und "})


def _is_legal(token: str) -> bool:
    return token.lower().strip(".,;()") in _LEGAL


def strip_legal(name: str) -> str:
    tokens = name.split()
    while len(tokens) > 1 and _is_legal(tokens[-1]):
        tokens.pop()
    return " ".join(tokens).rstrip(" ,;-–")


def clean(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").replace("\n", " ")).strip().strip("\"'„“”").strip()


def key(name: str) -> str:
    s = strip_legal(clean(name)).casefold().translate(_TRANSLIT)
    return " ".join(re.sub(r"[^\w]+", " ", s).split())


# ---------------------------------------------------------------- correspondents

_GENERIC_SENDERS = {"noreply", "no-reply", "no reply", "info", "service", "kundenservice", "support", "rechnung",
                    "rechnungen", "billing", "invoice", "mail", "newsletter", "kontakt", "team"}
_EMAIL_RE = re.compile(r"^[\w.+-]+@((?:[\w-]+\.)+[a-z]{2,})$", re.I)
_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?((?:[\w-]+\.)+[a-z]{2,})/?$", re.I)
_SECOND_LEVEL = {"co", "com", "org", "net", "gv", "or", "ac"}


def domain_label(value: str) -> str | None:
    """For e-mail addresses and domains the name part ("noreply@mail.enbw.com" -> "enbw"), else None."""
    value = value.strip().strip("<>").lower()
    m = _EMAIL_RE.match(value) or _DOMAIN_RE.match(value)
    if not m:
        return None
    parts = m.group(1).split(".")
    if len(parts) >= 3 and parts[-2] in _SECOND_LEVEL:
        return parts[-3]
    return parts[-2] if len(parts) >= 2 else None


class CorrespondentNormalizer:
    """Unifies correspondent names against known names (Paperless plus names recognised so far)."""

    def __init__(self, known: list[str], aliases: dict[str, str] | None = None, cutoff: float = 94):
        self.cutoff = cutoff
        self.aliases: dict[str, str] = {}
        for src, dst in (aliases or {}).items():
            self.aliases[src.casefold()] = dst
            self.aliases.setdefault(key(src), dst)
        self.known: dict[str, str] = {}
        for name in known:
            if not domain_label(name):  # "info@o2.com" is not a name to copy
                self.add(name)

    def add(self, name: str) -> None:
        k = key(name)
        if k:
            self.known.setdefault(k, clean(name))

    def __call__(self, name: str, fallback: str = "") -> str:
        """fallback: display name of the mail sender, used if the model only returned an address."""
        name = clean(name)
        if not name:
            return ""
        label = domain_label(name)
        if label:
            name = self._from_address(label, fallback)
        alias = self.aliases.get(name.casefold()) or self.aliases.get(key(name))
        if alias:
            return alias
        k = key(name)
        if not k:
            return ""
        if k in self.known:
            return self.known[k]
        match = process.extractOne(k, list(self.known), scorer=fuzz.ratio, score_cutoff=self.cutoff)
        if match:
            return self.known[match[0]]
        result = strip_legal(name)
        self.add(result)
        return result

    def _from_address(self, label: str, fallback: str) -> str:
        k = key(label)
        if k in self.known:
            return self.known[k]
        fb = clean(fallback)
        if fb and not domain_label(fb) and fb.casefold() not in _GENERIC_SENDERS:
            return fb
        first_word = [v for kk, v in self.known.items() if kk.split()[0] == k]
        if len(first_word) == 1:
            return first_word[0]
        return label[:1].upper() + label[1:]


# ---------------------------------------------------------------- tags

class TagNormalizer:
    """Maps suggested tags onto existing ones; new tags only if allowed."""

    def __init__(self, known: list[str], allow_new: bool, blocked: set[str], cutoff: float = 92):
        self.allow_new, self.blocked, self.cutoff = allow_new, blocked, cutoff
        self.known: dict[str, str] = {}
        for t in known:
            self.add(t)

    def add(self, tag: str) -> None:
        k = key(tag)
        if k:
            self.known.setdefault(k, tag.strip())

    def __call__(self, tags: list, limit: int = 3) -> list[str]:
        result: list[str] = []
        for raw in tags if isinstance(tags, list) else []:
            tag = clean(str(raw)).lstrip("#")[:50]
            k = key(tag)
            if not k or tag.casefold() in self.blocked:
                continue
            if k in self.known:
                tag = self.known[k]
            elif match := process.extractOne(k, list(self.known), scorer=fuzz.ratio, score_cutoff=self.cutoff):
                tag = self.known[match[0]]
            elif not self.allow_new:
                continue
            if tag not in result:
                result.append(tag)
            if len(result) >= limit:
                break
        return result


# ---------------------------------------------------------------- dates

_DATE_PATTERNS = (
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})"), (1, 2, 3)),
    (re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})"), (3, 2, 1)),
    (re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})"), (1, 2, 3)),
)


def parse_date(value: str | None) -> date | None:
    value = (value or "").strip()
    for pattern, (yi, mi, di) in _DATE_PATTERNS:
        m = pattern.match(value)
        if m:
            try:
                return date(int(m.group(yi)), int(m.group(mi)), int(m.group(di)))
            except ValueError:
                return None
    return None


def pick_date(doc_date: str | None, reference: str | None, today: date | None = None,
              fallback_to_reference: bool = True) -> tuple[str, str]:
    """Check the document date for plausibility against a reference date (mail date or date added to
    Paperless): a document cannot be issued long after it arrived. Returns (YYYY-MM-DD, source)."""
    today = today or date.today()
    d = parse_date(doc_date)
    ref = parse_date((reference or "")[:10])
    limit = min(today, ref) + timedelta(days=31) if ref else today + timedelta(days=31)
    if d and d.year >= 1970 and d <= limit:
        return d.isoformat(), "document"
    if ref and fallback_to_reference:
        return ref.isoformat(), "reference"
    return "", ""


def clean_title(title: str, fallback: str) -> str:
    title = clean(title).rstrip(".")
    return (title or fallback).strip()[:128]


# ---------------------------------------------------------------- custom field values

_CURRENCY_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP", "chf": "CHF", "fr.": "CHF"}


def parse_number(text: str) -> float | None:
    """'1.234,56' / '1,234.56' / '89,45' / '89.45' -> float"""
    m = re.search(r"-?\d[\d.,' ]*", text)
    if not m:
        return None
    s = m.group().replace(" ", "").replace("'", "").rstrip(".,")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".") if s.count(",") == 1 else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_field(kind: str, raw: object) -> str | int | float | bool | None:
    """Model output -> value in Paperless' API format, or None if absent/unusable."""
    text = clean(str(raw)) if raw is not None else ""
    if not text or text.casefold() in ("null", "none", "n/a", "-", "unbekannt", "unknown"):
        return None
    if kind == "string":
        return text[:128]
    if kind == "url":
        return text if re.match(r"https?://\S+$", text) else None
    if kind == "date":
        d = parse_date(text)
        return d.isoformat() if d else None
    if kind == "boolean":
        t = text.casefold()
        return True if t in ("true", "yes", "ja", "1") else False if t in ("false", "no", "nein", "0") else None
    number = parse_number(text)
    if number is None:
        return None
    if kind == "integer":
        return int(round(number))
    if kind == "float":
        return number
    if kind == "monetary":
        code = re.search(r"\b([A-Z]{3})\b", text)
        currency = code.group(1) if code else next(
            (c for sym, c in _CURRENCY_SYMBOLS.items() if sym in text.casefold()), "EUR")
        return f"{currency}{number:.2f}"
    return None
