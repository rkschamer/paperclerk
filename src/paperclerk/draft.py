"""The form of `paperclerk ui`: the values a Paperless document will have after accepting.

A draft starts as the document's current values with the model's suggestion applied according to the
write policy of `process` (section 6.2 of docs/design.md) and, with `[process] remove_inbox_tags`, without
the inbox tags, because accepting means the user has reviewed the document. Whatever the user then types
wins: the changes are computed against the document as loaded and written with
`process.apply(overwrite=True)`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .config import Config
from .normalize import clean, parse_date, parse_field
from .paperless import Registry
from .process import plan

KEYS = ("title", "correspondent", "document_type", "date", "tags", "note")
HINTS = {
    "monetary": 'an amount such as "44,15 EUR" or "EUR44.15"',
    "date": "a date such as 2023-04-28 or 28.04.2023",
    "integer": "a whole number",
    "float": "a number",
    "boolean": "yes or no",
    "url": "a URL starting with http:// or https://",
}


@dataclass
class Draft:
    title: str = ""
    correspondent: str = ""
    document_type: str = ""
    date: str = ""  # YYYY-MM-DD as loaded; the user may type DD.MM.YYYY
    tags: list[str] = field(default_factory=list)  # without the processed tag, which is always added
    fields: dict[str, str] = field(default_factory=dict)  # every configured custom field, "" = empty
    note: str = ""  # summary, added as a note

    def copy(self) -> "Draft":
        return copy.deepcopy(self)


def split_tags(text: str) -> list[str]:
    result: list[str] = []
    for part in text.split(","):
        tag = clean(part)
        if tag and tag.casefold() not in {t.casefold() for t in result}:
            result.append(tag)
    return result


def join_tags(tags: list[str]) -> str:
    return ", ".join(tags)


def field_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def current(doc: dict, registry: Registry, cfg: Config) -> Draft:
    """The document as it is in Paperless."""
    names = registry.names
    processed = cfg.paperless.processed_tag.casefold()
    values = {cf.get("field"): cf.get("value") for cf in doc.get("custom_fields") or []}
    return Draft(
        title=doc.get("title") or "",
        correspondent=names["correspondents"].get(doc.get("correspondent"), "") if doc.get("correspondent") else "",
        document_type=names["document_types"].get(doc.get("document_type"), "") if doc.get("document_type") else "",
        date=str(doc.get("created") or doc.get("created_date") or "")[:10],
        tags=[names["tags"][t] for t in doc.get("tags") or []
              if t in names["tags"] and names["tags"][t].casefold() != processed],
        fields={f.name: field_text(values.get(registry.get("custom_fields", f.name))) for f in cfg.fields},
    )


def proposal(doc: dict, suggestion: dict | None, registry: Registry, cfg: Config, inbox: set[int]) -> Draft:
    """The document after accepting: the suggestion applied with the write policy of `process` and, if
    configured, the inbox tags (ids in `inbox`) removed."""
    d = current(doc, registry, cfg)
    if cfg.process.remove_inbox_tags:
        leaving = {registry.names["tags"].get(t, "").casefold() for t in inbox}
        d.tags = [t for t in d.tags if t.casefold() not in leaving]
    if not suggestion:
        return d
    changes = plan(doc, suggestion, registry, bool(inbox & set(doc.get("tags") or [])))
    if "title" in changes:
        d.title = changes["title"][1]
    for key in ("correspondent", "document_type"):
        if key in changes:
            setattr(d, key, changes[key][1])
    if "created" in changes:
        d.date = changes["created"][1]
    d.tags = [*d.tags, *changes.get("tags", [])]
    for name, value in (changes.get("fields") or {}).items():
        if name in d.fields:
            d.fields[name] = field_text(value)
    d.note = changes.get("note", "")
    return d


def merge(old: Draft, new: Draft, draft: Draft) -> Draft:
    """A new proposal arrived (model answer, reload): take it for every value the user has not changed."""
    result = new.copy()
    for key in KEYS:
        if getattr(draft, key) != getattr(old, key):
            setattr(result, key, copy.deepcopy(getattr(draft, key)))
    for name, value in draft.fields.items():
        if name in result.fields and value != old.fields.get(name, ""):
            result.fields[name] = value
    return result


def changes_for(cur: Draft, draft: Draft, cfg: Config) -> tuple[dict, dict[str, str]]:
    """Changes for process.apply(overwrite=True) and error messages by form key ("title", "field:Betrag")."""
    out: dict = {}
    errors: dict[str, str] = {}
    title = clean(draft.title)
    if not title:
        errors["title"] = "must not be empty"
    elif title != cur.title:
        out["title"] = [cur.title, title[:128]]
    for key in ("correspondent", "document_type"):
        new, old = clean(getattr(draft, key)), getattr(cur, key)
        if new.casefold() != old.casefold():  # Paperless names are matched ignoring case
            out[key] = [old or None, new]
    text = clean(draft.date)
    if text != cur.date:
        day = parse_date(text)
        if day is None:
            errors["date"] = "use a date such as 2023-04-28 or 28.04.2023"
        elif day.isoformat() != cur.date:
            out["created"] = [cur.date, day.isoformat()]
    have = {t.casefold() for t in cur.tags}
    want = {t.casefold() for t in draft.tags}
    if added := [t for t in draft.tags if t.casefold() not in have]:
        out["tags"] = added
    if removed := [t for t in cur.tags if t.casefold() not in want]:
        out["untag"] = removed
    types = {f.name: f.type for f in cfg.fields}
    fields: dict = {}
    for name, raw in draft.fields.items():
        text = clean(raw)
        if text == cur.fields.get(name, ""):
            continue
        kind = types.get(name, "string")
        value = parse_field(kind, text) if text else None
        if text and value is None:
            errors[f"field:{name}"] = f"expected {HINTS.get(kind, 'text')}"
        elif field_text(value) != cur.fields.get(name, ""):
            fields[name] = value  # None clears the field
    if fields:
        out["fields"] = fields
    if note := draft.note.strip():
        out["note"] = note
    return out, errors


def changed_keys(changes: dict) -> set[str]:
    """Form keys that a change set touches."""
    keys = {k for k in ("title", "correspondent", "document_type", "note") if k in changes}
    if "created" in changes:
        keys.add("date")
    if "tags" in changes or "untag" in changes:
        keys.add("tags")
    keys |= {f"field:{name}" for name in changes.get("fields") or {}}
    return keys
