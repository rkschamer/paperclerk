"""`paperclerk process`: enrich documents that are already in Paperless.

Selection: documents without the processed tag (scope "inbox": only those with an inbox tag).
Write policy: on inbox documents the suggestion replaces Paperless' automatic guesses (title,
correspondent, document type, date); outside the inbox only missing values are filled in. Tags are only
ever added, custom fields only filled when empty. Every processed document gets the processed tag.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path

from .config import Config
from .db import DB, now_iso
from .enrich import VERSION, Enricher, Source, vocabulary
from .ollama import Ollama, OllamaError, OllamaUnavailable
from .paperless import Paperless, PaperlessError, Registry, sync_cache
from .store import sniff
from .textextract import render_page
from .util import Progress

log = logging.getLogger(__name__)
MAX_FAILS = 3


def _hash(doc: dict) -> str:
    return hashlib.sha256((doc.get("content") or "").encode("utf-8", "replace")).hexdigest()


def select(pl: Paperless, cfg: Config, registry: Registry, inbox: set[int], full: bool = False) -> list[dict]:
    """Documents to process; full=True returns complete documents (with text) instead of id, title, tags."""
    processed = registry.get("tags", cfg.paperless.processed_tag)
    params: dict = {"ordering": "id"} if full else {"ordering": "id", "fields": "id,title,tags"}
    if processed is not None:
        params["tags__id__none"] = processed
    if cfg.process.scope == "inbox":
        params["tags__id__in"] = ",".join(str(i) for i in sorted(inbox))
    docs = []
    for d in pl.documents(params):  # the filters are checked again here in case Paperless ignores one
        tags = set(d.get("tags") or [])
        if processed in tags or (cfg.process.scope == "inbox" and not tags & inbox):
            continue
        docs.append(d)
    return docs


def source_for(pl: Paperless, cfg: Config, doc: dict) -> Source:
    text = doc.get("content") or ""
    images = None
    if len(text.strip()) < cfg.text.min_chars_per_page:  # no OCR text: show the model the first page
        try:
            data = pl.download(doc["id"])
            kind = sniff(data)
            if kind and kind.kind in ("pdf", "image"):
                cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp = cfg.tmp_dir / f"paperless-{doc['id']}.{kind.ext}"
                tmp.write_bytes(data)
                try:
                    page = render_page(tmp, kind.kind, 0, cfg.text.vision_max_side)
                finally:
                    tmp.unlink(missing_ok=True)
                images = [page] if page else None
        except PaperlessError as e:
            log.warning("#%s: no text and no download (%s)", doc["id"], e)
    name = doc.get("original_file_name") or doc.get("title") or f"document-{doc['id']}"
    context = [f"Added to Paperless: {(doc.get('added') or '')[:10]}"]
    title = doc.get("title") or ""
    if title and title != Path(name).stem:
        context.append(f"Current title in Paperless (often the mail subject or an earlier guess): {title}")
    return Source(text=text, filename=name, context=context, reference_date=doc.get("added") or "", images=images,
                  date_fallback=False)


def plan(doc: dict, s: dict, registry: Registry, in_inbox: bool) -> dict:
    """The changes a suggestion leads to: {field: [old, new]}, tags: [added], fields: {name: value}, note."""
    changes: dict = {}
    default_title = doc.get("title") == Path(doc.get("original_file_name") or "").stem
    if s.get("title") and s["title"] != doc.get("title") and (in_inbox or default_title):
        changes["title"] = [doc.get("title"), s["title"]]
    for kind, name in (("correspondents", "correspondent"), ("document_types", "document_type")):
        current = registry.names[kind].get(doc.get(name)) if doc.get(name) else None
        new = s.get(name)
        if new and (current or "").casefold() != new.casefold() and (in_inbox or current is None):
            changes[name] = [current, new]
    current_date = str(doc.get("created") or doc.get("created_date") or "")[:10]
    if s.get("date") and in_inbox and s["date"] != current_date:
        changes["created"] = [current_date, s["date"]]
    have = {registry.names["tags"].get(t, "").casefold() for t in doc.get("tags") or []}
    added = [t for t in s.get("tags") or [] if t.casefold() not in have]
    if added:
        changes["tags"] = added
    filled = {cf.get("field"): cf.get("value") for cf in doc.get("custom_fields") or []}
    fields = {}
    for fname, value in (s.get("fields") or {}).items():
        fid = registry.get("custom_fields", fname)
        if fid is None or filled.get(fid) in (None, ""):
            fields[fname] = value
    if fields:
        changes["fields"] = fields
    if s.get("summary"):
        changes["note"] = s["summary"]
    return changes


def apply(pl: Paperless, registry: Registry, cfg: Config, doc_id: int, changes: dict,
          old_note: int | None, overwrite: bool = False) -> int | None:
    """Write planned changes. Tags in changes["untag"] are removed; custom fields are only filled when
    empty unless overwrite=True (edits the user made in `paperclerk ui`, where None clears a value)."""
    fresh = pl.document(doc_id)  # read tags and custom fields again right before writing
    payload: dict = {}
    if "title" in changes:
        payload["title"] = changes["title"][1][:128]
    if "correspondent" in changes:
        payload["correspondent"] = registry.ensure("correspondents", changes["correspondent"][1] or "")
    if "document_type" in changes:
        payload["document_type"] = registry.ensure("document_types", changes["document_type"][1] or "")
    if "created" in changes:
        payload["created"] = changes["created"][1]
    removed = {registry.get("tags", name) for name in changes.get("untag", [])}
    tags = [t for t in fresh.get("tags") or [] if t not in removed]
    for name in [*changes.get("tags", []), cfg.paperless.processed_tag]:
        tid = registry.ensure("tags", name)
        if tid is not None and tid not in tags:
            tags.append(tid)
    payload["tags"] = tags
    if changes.get("fields"):
        types = {f.name: f.type for f in cfg.fields}
        current = [{"field": cf["field"], "value": cf.get("value")} for cf in fresh.get("custom_fields") or []]
        for fname, value in changes["fields"].items():
            fid = registry.ensure("custom_fields", fname, types.get(fname, "string"))
            slot = next((c for c in current if c["field"] == fid), None)
            if slot is None:
                current.append({"field": fid, "value": value})
            elif overwrite or slot["value"] in (None, ""):
                slot["value"] = value
        payload["custom_fields"] = current
    pl.update(doc_id, payload)
    if "note" not in changes:
        return old_note
    if old_note:
        try:
            pl.delete_note(doc_id, old_note)
        except PaperlessError as e:
            log.warning("#%s: old summary note %s not deleted: %s", doc_id, old_note, e)
    return pl.add_note(doc_id, f"{changes['note']}\n\n(paperclerk, {cfg.enrich.model})")


def describe(doc: dict, changes: dict, in_inbox: bool, processed_tag: str) -> str:
    lines = [f"#{doc['id']} {doc.get('title')!r}{' (inbox)' if in_inbox else ''}"]
    labels = {"title": "title", "correspondent": "correspondent", "document_type": "document type",
              "created": "date"}
    for key, label in labels.items():
        if key in changes:
            old, new = changes[key]
            lines.append(f"    {label + ':':<15} {old or '-'} -> {new}")
    lines.append(f"    {'tags:':<15} + {', '.join([*changes.get('tags', []), processed_tag])}")
    for fname, value in (changes.get("fields") or {}).items():
        lines.append(f"    {fname + ':':<15} {value}")
    if changes.get("note"):
        lines.append(f"    {'summary:':<15} {changes['note']}")
    return "\n".join(lines)


def stored_suggestion(rec, cfg: Config, doc: dict) -> dict | None:
    """The stored suggestion, if it is not applied yet and was made for this text by the current pipeline and
    model."""
    if (rec and rec["status"] == "suggested" and rec["version"] == VERSION
            and rec["model"] == cfg.enrich.model and rec["content_hash"] == _hash(doc)):
        return json.loads(rec["suggestion"])
    return None


def open_enricher(db: DB, cfg: Config) -> tuple[Ollama, Enricher]:
    ollama = Ollama(cfg.ollama.url, cfg.ollama.keep_alive, cfg.ollama.timeout)
    try:
        ollama.require(cfg.enrich.model)
    except OllamaError:
        ollama.close()
        raise
    return ollama, Enricher(cfg, vocabulary(db, cfg), ollama)


def suggest(db: DB, cfg: Config, pl: Paperless, enricher: Enricher, doc: dict) -> dict:
    """Ask the model and store the suggestion. A failed request is recorded and re-raised."""
    t0 = time.monotonic()
    try:
        suggestion, res = enricher(source_for(pl, cfg, doc))
    except OllamaUnavailable:
        raise
    except OllamaError as e:
        db.x("""INSERT INTO processed(doc_id, title, status, fail_count, error, updated_at)
                VALUES (?, ?, 'failed', 1, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET status = 'failed', fail_count = processed.fail_count + 1,
                    error = excluded.error, updated_at = excluded.updated_at""",
             (doc["id"], doc.get("title"), str(e)[:500], now_iso()))
        db.commit()
        raise
    db.log_llm(f"paperless:{doc['id']}", "process", cfg.enrich.model, time.monotonic() - t0,
               res.input_tokens, res.output_tokens, res.content)
    db.x("""INSERT INTO processed(doc_id, title, content_hash, suggestion, version, model, secs, status,
                                  fail_count, error, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'suggested', 0, NULL, ?)
            ON CONFLICT(doc_id) DO UPDATE SET title = excluded.title,
                content_hash = excluded.content_hash, suggestion = excluded.suggestion,
                version = excluded.version, model = excluded.model, secs = excluded.secs,
                status = 'suggested', fail_count = 0, error = NULL, updated_at = excluded.updated_at""",
         (doc["id"], doc.get("title"), _hash(doc), json.dumps(suggestion, ensure_ascii=False),
          VERSION, cfg.enrich.model, round(time.monotonic() - t0, 2), now_iso()))
    db.commit()
    return suggestion


def record_written(db: DB, doc: dict, changes: dict, note_id: int | None, status: str = "applied") -> None:
    """Remember what was written (status applied, or rejected: only the processed tag was added)."""
    db.x("""INSERT INTO processed(doc_id, title, content_hash, status, changes, note_id, fail_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(doc_id) DO UPDATE SET status = excluded.status, changes = excluded.changes,
                note_id = excluded.note_id, error = NULL, updated_at = excluded.updated_at""",
         (doc["id"], doc.get("title"), _hash(doc), status, json.dumps(changes, ensure_ascii=False), note_id,
          now_iso()))
    db.commit()


def record_write_error(db: DB, doc_id: int, error: Exception) -> None:
    db.x("UPDATE processed SET error = ?, updated_at = ? WHERE doc_id = ?",
         (f"write failed: {error}"[:500], now_iso(), doc_id))
    db.commit()


def failed_ids(db: DB) -> set[int]:
    """Documents whose model call failed MAX_FAILS times; skipped until asked for explicitly."""
    return {r["doc_id"] for r in db.q("SELECT doc_id FROM processed WHERE status = 'failed' AND fail_count >= ?",
                                      (MAX_FAILS,))}


def run_process(db: DB, cfg: Config, dry_run: bool = False, limit: int | None = None) -> Counter:
    counts: Counter = Counter()
    with Paperless.from_config(cfg) as pl:
        objects = sync_cache(db, pl)
        registry = Registry(db, pl, objects)
        inbox = {t["id"] for t in objects["tags"] if t.get("is_inbox_tag")}
        if cfg.process.scope == "inbox" and not inbox:
            log.warning('Paperless has no inbox tag – nothing to do. Create one or set [process] scope = "all".')
            return counts
        failed = failed_ids(db)
        candidates = [d for d in select(pl, cfg, registry, inbox) if d["id"] not in failed]
        candidates = candidates[:limit] if limit else candidates
        if not candidates:
            log.info("Nothing to process.")
            return counts
        log.info("%d documents to process (scope %s)%s", len(candidates), cfg.process.scope,
                 " – dry run" if dry_run else "")
        ollama: Ollama | None = None
        enricher: Enricher | None = None
        progress = Progress(len(candidates), "Process")
        try:
            for light in candidates:
                t0 = time.monotonic()
                doc = pl.document(light["id"])
                rec = db.one("SELECT * FROM processed WHERE doc_id = ?", (doc["id"],))
                suggestion = stored_suggestion(rec, cfg, doc)
                if suggestion is None:
                    if enricher is None:
                        ollama, enricher = open_enricher(db, cfg)
                    try:
                        suggestion = suggest(db, cfg, pl, enricher, doc)
                    except OllamaUnavailable:
                        raise
                    except OllamaError as e:
                        counts["failed"] += 1
                        progress.step(f"#{doc['id']}: ERROR {e}", time.monotonic() - t0)
                        continue
                in_inbox = bool(inbox & set(doc.get("tags") or []))
                changes = plan(doc, suggestion, registry, in_inbox)
                if dry_run:
                    print(describe(doc, changes, in_inbox, cfg.paperless.processed_tag))
                    counts["suggested"] += 1
                else:
                    old_note = rec["note_id"] if rec else None
                    try:
                        note_id = apply(pl, registry, cfg, doc["id"], changes, old_note)
                    except PaperlessError as e:  # the suggestion stays stored; the next run tries again
                        log.error("#%s: writing to Paperless failed: %s", doc["id"], e)
                        record_write_error(db, doc["id"], e)
                        counts["failed"] += 1
                        continue
                    record_written(db, doc, changes, note_id)
                    counts["applied"] += 1
                progress.step(f"#{doc['id']} {suggestion.get('title')}", time.monotonic() - t0, quiet=dry_run)
        finally:
            if ollama is not None:
                ollama.close()
        progress.done()
    return counts
