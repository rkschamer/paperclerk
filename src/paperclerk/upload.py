"""Upload imported documents to Paperless (import only)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from sqlite3 import Row

from .config import Config
from .db import DB, now_iso
from .decision import REVIEW, UPLOAD_DECISIONS, decision, effective, overrides, probability
from .normalize import CorrespondentNormalizer
from .paperless import Paperless, PaperlessError, Registry, TaskInfo, cached, sync_cache
from .similarity import compare, search_terms
from .util import Progress, safe_filename

log = logging.getLogger(__name__)


@dataclass
class Plan:
    title: str
    created: str
    correspondent: str
    document_type: str
    tags: list[str] = field(default_factory=list)
    fields: dict = field(default_factory=dict)
    summary: str = ""
    filename: str = ""


@dataclass
class Match:
    doc_id: int
    title: str
    containment: float

    def describe(self) -> str:
        return f"{self.containment:.0%} of the text is in Paperless #{self.doc_id} '{self.title}'"


def build_plan(db: DB, row: Row, cfg: Config, normalizer: CorrespondentNormalizer | None = None) -> Plan:
    corr = effective(row, "correspondent") or ""
    if normalizer and corr and "correspondent" not in overrides(row):
        mails = db.mails_for(row["sha256"])
        corr = normalizer(corr, fallback=mails[0]["from_name"] if mails else "")
    tags = [cfg.paperless.processed_tag, *cfg.imports.tags]
    if decision(row, cfg) == REVIEW and cfg.imports.review_tag:
        tags.append(cfg.imports.review_tag)
    for t in effective(row, "tags") or []:
        if t.casefold() not in {x.casefold() for x in tags}:
            tags.append(t)
    filename = row["filename"] or f"document.{row['ext']}"
    if not filename.lower().endswith("." + row["ext"]):
        filename = f"{Path(filename).stem}.{row['ext']}"
    return Plan(
        title=(effective(row, "title") or Path(filename).stem)[:128],
        created=effective(row, "date") or "",
        correspondent=corr,
        document_type=effective(row, "document_type") or "",
        tags=tags,
        fields=effective(row, "fields") or {},
        summary=effective(row, "summary") or "",
        filename=safe_filename(filename),
    )


def note_text(db: DB, row: Row, cfg: Config, summary: str) -> str:
    lines = [summary, ""] if summary else []
    lines.append("Imported from e-mail by paperclerk.")
    for m in db.mails_for(row["sha256"])[:3]:
        sender = f"{m['from_name']} <{m['from_addr']}>" if m["from_name"] else m["from_addr"]
        lines.append(f"Mail of {(m['date'] or '?')[:16].replace('T', ' ')} from {sender}")
        lines.append(f"Subject: {m['subject']}")
        if m["message_id"]:
            lines.append(f"Message-ID: {m['message_id']}")
    lines.append(f"File name: {row['filename']}")
    if row["parts"]:
        lines.append(f"Merged from {len(json.loads(row['parts']))} attachments.")
    p = probability(row, cfg)
    if p is not None:
        model = row["model2"] or row["model1"]
        lines.append(f"Relevance ({model}): {p:.0%}" + (f", manual: {row['manual']}" if row["manual"] else ""))
    return "\n".join(lines)


def find_similar(pl: Paperless, row: Row, threshold: float) -> Match | None:
    """An existing Paperless document that contains (almost) all of our text."""
    text = row["text"] or ""
    terms = search_terms(text)
    if not terms or compare(text, text) is None:
        return None  # too little text for a reliable comparison
    try:
        candidates = pl.search(terms)
    except PaperlessError as e:
        log.warning("Paperless full-text search failed (%s) – content check skipped for %s", e, row["filename"])
        return None
    best: Match | None = None
    for c in candidates:
        sim = compare(text, c.get("content") or "")
        if sim and sim.is_duplicate(threshold) and (best is None or sim.containment > best.containment):
            best = Match(int(c["id"]), c.get("title") or "", sim.containment)
    return best


def _candidates(db: DB, cfg: Config, ids: list[str] | None = None) -> list[Row]:
    from .enrich import VERSION  # noqa: PLC0415

    rows = db.q("""SELECT * FROM documents WHERE skip_reason IS NULL AND part_of IS NULL AND meta IS NOT NULL
                     AND meta_version = ? ORDER BY rowid""", (VERSION,))
    rows = [r for r in rows if r["upload_status"] in (None, "failed", "similar")]  # similar: checked again
    if ids:
        prefixes = [i.lower() for i in ids]
        rows = [r for r in rows if any(r["sha256"].startswith(p) for p in prefixes)]
    return [r for r in rows if decision(r, cfg) in UPLOAD_DECISIONS]


def _field_values(registry: Registry, cfg: Config, fields: dict) -> list[dict]:
    types = {f.name: f.type for f in cfg.fields}
    result = []
    for name, value in fields.items():
        fid = registry.ensure("custom_fields", name, types.get(name, "string"))
        if fid is not None:
            result.append({"field": fid, "value": value})
    return result


class Uploader:
    def __init__(self, db: DB, cfg: Config, pl: Paperless, registry: Registry | None, content_check: bool = True):
        self.db, self.cfg, self.pl, self.registry = db, cfg, pl, registry
        self.content_check = content_check and cfg.imports.content_check
        self.pending: dict[str, str] = {
            r["task_id"]: r["sha256"] for r in db.q("SELECT task_id, sha256 FROM documents "
                                                     "WHERE upload_status = 'pending' AND task_id IS NOT NULL")
        }
        self.counts = {"done": 0, "exists": 0, "similar": 0, "duplicate": 0, "failed": 0}

    def _finish(self, sha: str, info: TaskInfo) -> None:
        if info.status == "SUCCESS" and info.document_id:
            self.db.update_doc(sha, upload_status="done", paperless_id=info.document_id, upload_error=None,
                               uploaded_at=now_iso())
            self.counts["done"] += 1
            self.complete(sha)
        elif info.duplicate_of or "duplicate" in info.message.lower():
            self.db.update_doc(sha, upload_status="duplicate", paperless_id=info.duplicate_of,
                               upload_error=info.message, uploaded_at=now_iso())
            self.counts["duplicate"] += 1
        else:
            self.db.update_doc(sha, upload_status="failed", upload_error=info.message or info.status)
            self.counts["failed"] += 1
            log.warning("Paperless could not process %s: %s", sha[:12], info.message)
        self.db.commit()

    def complete(self, sha: str) -> None:
        """After consumption: custom fields and the note (summary + mail origin)."""
        row = self.db.one("SELECT * FROM documents WHERE sha256 = ?", (sha,))
        if not row or not row["paperless_id"] or row["note_added"] or self.registry is None:
            return
        plan = build_plan(self.db, row, self.cfg)
        try:
            if plan.fields:  # keep fields a Paperless workflow may have set during consumption
                current = self.pl.document(row["paperless_id"]).get("custom_fields") or []
                merged = [{"field": cf["field"], "value": cf.get("value")} for cf in current]
                for new in _field_values(self.registry, self.cfg, plan.fields):
                    slot = next((c for c in merged if c["field"] == new["field"]), None)
                    if slot is None:
                        merged.append(new)
                    elif slot["value"] in (None, ""):
                        slot["value"] = new["value"]
                self.pl.update(row["paperless_id"], {"custom_fields": merged})
            self.pl.add_note(row["paperless_id"], note_text(self.db, row, self.cfg, plan.summary))
            self.db.update_doc(sha, note_added=1)
        except PaperlessError as e:
            log.warning("Custom fields/note for Paperless #%s not set (retried next run): %s", row["paperless_id"], e)

    def poll(self) -> None:
        for task_id, sha in list(self.pending.items()):
            info = self.pl.task(task_id)
            if info is not None and info.finished:
                del self.pending[task_id]
                self._finish(sha, info)

    def wait(self, max_pending: int, timeout: float = 900) -> None:
        deadline = time.monotonic() + timeout
        while len(self.pending) > max_pending and time.monotonic() < deadline:
            self.poll()
            if len(self.pending) > max_pending:
                time.sleep(3)

    def existing(self, row: Row) -> tuple[str, int, str] | None:
        """(status, Paperless id, reason) if Paperless already has the document."""
        doc_id = self.pl.find_by_checksum(row["sha256"], row["md5"])
        if doc_id:
            return "exists", doc_id, "same file"
        if self.content_check:
            match = find_similar(self.pl, row, self.cfg.imports.content_threshold)
            if match:
                return "similar", match.doc_id, match.describe()
        return None

    def upload(self, row: Row, normalizer: CorrespondentNormalizer) -> str:
        sha = row["sha256"]
        found = self.existing(row)
        if found:
            status, doc_id, reason = found
            self.db.update_doc(sha, upload_status=status, paperless_id=doc_id, uploaded_at=now_iso(),
                               upload_error=None if status == "exists" else reason)
            self.db.commit()
            self.counts[status] += 1
            return f"already in Paperless (#{doc_id}, {reason})"
        plan = build_plan(self.db, row, self.cfg, normalizer)
        fields: dict = {"title": plan.title}
        if plan.created:
            fields["created"] = plan.created
        if plan.correspondent and (cid := self.registry.ensure("correspondents", plan.correspondent)):
            fields["correspondent"] = str(cid)
        if plan.document_type and (did := self.registry.ensure("document_types", plan.document_type)):
            fields["document_type"] = str(did)
        fields["tags"] = [str(t) for t in (self.registry.ensure("tags", name) for name in plan.tags) if t]
        try:
            task_id = self.pl.upload(self.cfg.files_dir / row["path"], plan.filename, row["mime"], fields)
        except PaperlessError as e:
            self.db.update_doc(sha, upload_status="failed", upload_error=str(e)[:1000])
            self.db.commit()
            self.counts["failed"] += 1
            return f"ERROR {e}"
        self.db.update_doc(sha, upload_status="pending", task_id=task_id, upload_error=None)
        self.db.commit()
        self.pending[task_id] = sha
        return f"uploaded: {plan.title} | {plan.correspondent or '-'} | {plan.document_type or '-'}"

    def catch_up(self) -> None:
        """Earlier runs: finished tasks and missing notes/custom fields."""
        self.poll()
        for r in self.db.q("SELECT sha256 FROM documents WHERE upload_status = 'done' AND note_added = 0"):
            self.complete(r["sha256"])
        self.db.commit()


def preview_upload(db: DB, cfg: Config, limit: int | None = None) -> None:
    """Dry run: what an upload would do; checks Paperless for duplicates read-only if it is reachable."""
    rows = _candidates(db, cfg)
    rows = rows[:limit] if limit else rows
    if not rows:
        print("Nothing to upload.")
        return
    pl = None
    try:
        pl = Paperless.from_config(cfg)
        sync_cache(db, pl)
    except PaperlessError as e:
        log.warning("Paperless unreachable (%s) – preview without duplicate check", e)
        pl = None
    normalizer = CorrespondentNormalizer([o["name"] for o in cached(db, "correspondents")], cfg.aliases)
    checker = Uploader(db, cfg, pl, None) if pl else None
    new = 0
    try:
        for row in rows:
            p = build_plan(db, row, cfg, normalizer)
            found = checker.existing(row) if checker else None
            new += not found
            hint = f"  -> NOT uploaded: already in Paperless #{found[1]} ({found[2]})" if found else ""
            print(f"{row['sha256'][:12]}  {p.created or '????-??-??'}  {p.correspondent or '-':<25.25}  "
                  f"{p.document_type or '-':<14.14}  {p.title}  [{', '.join(p.tags)}]{hint}")
    finally:
        if pl:
            pl.close()
    print(f"\n{new} of {len(rows)} documents would be uploaded" + ("" if checker else " (duplicates not checked)"))


def run_upload(db: DB, cfg: Config, limit: int | None = None, ids: list[str] | None = None,
               content_check: bool = True) -> dict:
    rows = _candidates(db, cfg, ids)
    rows = rows[:limit] if limit else rows
    with Paperless.from_config(cfg) as pl:
        objects = sync_cache(db, pl)
        registry = Registry(db, pl, objects)
        uploader = Uploader(db, cfg, pl, registry, content_check)
        uploader.catch_up()
        if not rows:
            return uploader.counts
        normalizer = CorrespondentNormalizer([o["name"] for o in objects["correspondents"]], cfg.aliases)
        progress = Progress(len(rows), "Upload")
        for row in rows:
            t0 = time.monotonic()
            msg = uploader.upload(row, normalizer)
            progress.step(f"{row['filename']}: {msg}", time.monotonic() - t0)
            uploader.wait(cfg.imports.max_pending - 1)
        progress.done()
        if uploader.pending:
            log.info("Waiting for %d documents in Paperless' queue ...", len(uploader.pending))
        uploader.wait(0)
        if uploader.pending:
            log.warning("%d documents are still being processed by Paperless; the next run picks up their "
                        "status", len(uploader.pending))
        return uploader.counts
