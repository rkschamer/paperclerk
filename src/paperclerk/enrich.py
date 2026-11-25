"""Shared enrichment pipeline: one model call determines all metadata of a document.

Used by `import` (mail attachments) and `process` (documents already in Paperless). The vocabulary –
correspondents, document types, tags, custom fields – comes from Paperless plus the configuration.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, FieldDef
from .db import DB
from .normalize import (CorrespondentNormalizer, TagNormalizer, clean, clean_title, domain_label, parse_field,
                        pick_date)
from .ollama import ChatResult, Ollama, OllamaError
from .paperless import Paperless, PaperlessError, cached, sync_cache

log = logging.getLogger(__name__)

# Bump when prompt or schema change so much that stored suggestions should be recomputed.
VERSION = 2
MAX_TAGS = 3

FIELD_FORMATS = {
    "string": "text exactly as printed",
    "monetary": 'amount with ISO currency code and a dot as decimal separator, e.g. "EUR 1234.56"',
    "date": "YYYY-MM-DD",
    "integer": "whole number",
    "float": "number with a dot as decimal separator",
    "boolean": '"true" or "false"',
    "url": "complete URL",
}

SYSTEM_PROMPT = """You are a meticulous archivist. You receive a document (text excerpt or page image) with its \
context and determine its metadata for the document management system Paperless-ngx. Reply with JSON only, \
following the schema.

Fields:
- correspondent: who issued or sent the document (company, authority, practice, person). Never the recipient or \
customer, never an e-mail address or domain. Use the common name without legal form, e.g. "Telekom" instead of \
"Telekom Deutschland GmbH" and "EnBW" instead of "noreply@enbw.com". If the issuer is among the known \
correspondents, copy that spelling exactly. Empty string if unknown.
- document_type: exactly one of the document types listed below, or "" if none fits. Title and document type \
must describe the same document. If the file combines several documents (e.g. an invoice with an enclosed form \
or terms), describe the one that matters most for the recipient's records, usually the invoice or contract.
- date: the issue date of the document (invoice date, date of the letter, purchase date) as YYYY-MM-DD; not a \
due, delivery or service date. "" if the document shows no date.
- title: a short descriptive title in {language} with at most 8 words, without the correspondent's name. \
Examples: "Rechnung Mobilfunk April 2023", "Beitragsrechnung Kfz-Versicherung 2024", "Lohnsteuerbescheinigung 2022".
- tags: 0 to {max_tags} tags for the topic of the document. {tag_rule}
{field_rule}{summary_rule}
Known correspondents: {correspondents}

Document types: {document_types}

Tags: {tags}"""


@dataclass
class Vocabulary:
    correspondents: list[str]  # shown to the model
    all_correspondents: list[str]  # used to normalise answers
    document_types: list[str]
    tags: list[str]
    fields: list[FieldDef]


@dataclass
class Source:
    """A document as the model sees it."""
    text: str
    filename: str
    context: list[str] = field(default_factory=list)
    reference_date: str = ""  # mail date or date added to Paperless: the document cannot be much newer
    sender_name: str = ""  # mail sender, fallback if the model only returns an address
    images: list[str] | None = None
    date_fallback: bool = True  # implausible date: use the reference date (import) or leave unchanged


def _merge_names(*groups: list[str]) -> list[str]:
    seen, result = set(), []
    for group in groups:
        for name in group:
            name = (name or "").strip()
            if name and name.casefold() not in seen:
                seen.add(name.casefold())
                result.append(name)
    return result


def refresh_vocabulary(db: DB, cfg: Config) -> bool:
    """Update the local cache of Paperless objects; keep the old cache if Paperless is unreachable."""
    if not (cfg.paperless.url and cfg.paperless.token):
        return False
    try:
        with Paperless.from_config(cfg) as pl:
            sync_cache(db, pl)
        return True
    except PaperlessError as e:
        log.warning("Paperless unreachable (%s) – using the cached vocabulary", e)
        return False


def vocabulary(db: DB, cfg: Config, recent_correspondents: list[str] = ()) -> Vocabulary:
    corr = sorted(cached(db, "correspondents"), key=lambda o: -(o.get("document_count") or 0))
    names = [o["name"] for o in corr]
    shown = _merge_names([n for n in names if not domain_label(n)], list(cfg.aliases.values()),
                         list(recent_correspondents))
    blocked = cfg.system_tags()
    tags = [o["name"] for o in cached(db, "tags") if not o.get("is_inbox_tag") and o["name"].casefold() not in blocked]
    return Vocabulary(
        correspondents=shown[: cfg.enrich.known_correspondents],
        all_correspondents=names,
        document_types=_merge_names([o["name"] for o in cached(db, "document_types")], cfg.enrich.document_types),
        tags=_merge_names(tags, [t for t in cfg.enrich.tags if t.casefold() not in blocked]),
        fields=list(cfg.fields),
    )


def build_prompt(cfg: Config, vocab: Vocabulary) -> tuple[str, dict]:
    e = cfg.enrich
    tag_rule = (f'Prefer the listed tags. Only if none fits, propose a new broad topic in {e.language} (e.g. '
                '"Strom", "Auto"), never a year, amount, document type or correspondent name.'
                if e.new_tags else "Only tags from the list.")
    field_rule = ""
    if vocab.fields:
        lines = "\n".join(f'  - "{f.name}": {f.description} ({FIELD_FORMATS[f.type]})' for f in vocab.fields)
        field_rule = f"- fields: the value of each field below if the document states it, otherwise \"\".\n{lines}\n"
    summary_rule = (f"- summary: one or two sentences in {e.language}: what the document is and anything the "
                    "recipient has to act on (amount, deadline).\n" if e.summary else "")
    system = SYSTEM_PROMPT.format(
        language=e.language, max_tags=MAX_TAGS, tag_rule=tag_rule, field_rule=field_rule, summary_rule=summary_rule,
        correspondents="; ".join(vocab.correspondents) or "(none yet)",
        document_types="; ".join(vocab.document_types) or "(none)", tags="; ".join(vocab.tags) or "(none yet)")
    tag_item: dict = {"type": "string"} if e.new_tags else {"type": "string", "enum": vocab.tags or [""]}
    props: dict = {
        "correspondent": {"type": "string"},
        "document_type": {"type": "string", "enum": [*vocab.document_types, ""]},
        "date": {"type": "string"},
        "title": {"type": "string"},
        "tags": {"type": "array", "items": tag_item},
    }
    if vocab.fields:
        props["fields"] = {"type": "object", "properties": {f.name: {"type": "string"} for f in vocab.fields},
                           "required": [f.name for f in vocab.fields]}
    if e.summary:
        props["summary"] = {"type": "string"}
    return system, {"type": "object", "properties": props, "required": list(props)}


def parse_json(content: str) -> dict:
    try:
        data = json.loads(content)
    except ValueError:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise OllamaError(f"no JSON in the answer: {content[:200]!r}") from None
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise OllamaError(f"expected a JSON object: {content[:200]!r}")
    return data


class Enricher:
    def __init__(self, cfg: Config, vocab: Vocabulary, ollama: Ollama):
        self.cfg, self.vocab, self.ollama = cfg, vocab, ollama
        self.correspondents = CorrespondentNormalizer(_merge_names(vocab.correspondents, vocab.all_correspondents),
                                                      cfg.aliases)
        self.tags = TagNormalizer(vocab.tags, cfg.enrich.new_tags, cfg.system_tags())
        self.system, self.schema = build_prompt(cfg, vocab)

    def message(self, src: Source) -> dict:
        lines = [f"File name: {src.filename}", *src.context, ""]
        text = src.text.strip()
        limit = self.cfg.enrich.max_chars
        if len(text) > limit:  # beginning and end: mixed files often put the invoice on the last page
            head = int(limit * 0.75)
            lines += ["Document text (beginning and end):", f"{text[:head]}\n[…]\n{text[-(limit - head):]}"]
        elif text:
            lines += ["Document text:", text]
        elif src.images:
            lines.append("No readable text – the first page is attached as an image.")
        else:
            lines.append("No readable text.")
        msg: dict = {"role": "user", "content": "\n".join(lines)}
        if src.images:
            msg["images"] = src.images
        return msg

    def __call__(self, src: Source) -> tuple[dict, ChatResult]:
        e = self.cfg.enrich
        res = self.ollama.chat(e.model, [{"role": "system", "content": self.system}, self.message(src)],
                               schema=self.schema, think=e.think, options={"temperature": 0, "num_ctx": e.num_ctx})
        suggestion = self.normalize(parse_json(res.content), src)
        known = {t.casefold() for t in self.vocab.tags}
        new = [t for t in suggestion["tags"] if t.casefold() not in known]
        if new:  # later documents should reuse new tags instead of inventing synonyms
            self.vocab.tags.extend(new)
            for t in new:
                self.tags.add(t)
            self.system, self.schema = build_prompt(self.cfg, self.vocab)
        return suggestion, res

    def normalize(self, data: dict, src: Source) -> dict:
        date, date_source = pick_date(str(data.get("date") or ""), src.reference_date,
                                      fallback_to_reference=src.date_fallback)
        types = {t.casefold(): t for t in self.vocab.document_types}
        raw_fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
        fields = {}
        for f in self.vocab.fields:
            value = parse_field(f.type, raw_fields.get(f.name))
            if value is not None:
                fields[f.name] = value
        raw_corr = clean(str(data.get("correspondent") or ""))
        return {
            "correspondent": self.correspondents(raw_corr, fallback=src.sender_name),
            "correspondent_raw": raw_corr,
            "document_type": types.get(clean(str(data.get("document_type") or "")).casefold(), ""),
            "date": date,
            "date_source": date_source,
            "title": clean_title(str(data.get("title") or ""), Path(src.filename or "Dokument").stem),
            "tags": self.tags(data.get("tags") or [], MAX_TAGS),
            "fields": fields,
            "summary": clean(str(data.get("summary") or ""))[:600] if self.cfg.enrich.summary else "",
        }
