from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import mail, store
from .config import Config
from .db import DB, now_iso
from .sources import GmailImap, iter_mbox

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    messages: int = 0
    with_attachments: int = 0
    attachments: int = 0
    new_docs: int = 0
    skipped: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        lines = [
            f"Mails read: {self.messages}, with attachments: {self.with_attachments}",
            f"Attachments: {self.attachments}, new unique documents: {self.new_docs}",
        ]
        if self.skipped:
            top = ", ".join(f"{reason}: {n}" for reason, n in self.skipped.most_common(8))
            lines.append(f"Filtered out: {sum(self.skipped.values())} ({top})")
        return "\n".join(lines)


def _split_labels(labels: str) -> set[str]:
    return {x.strip().strip('"').lower() for x in labels.split(",") if x.strip()}


class Ingestor:
    def __init__(self, db: DB, cfg: Config):
        self.db, self.cfg = db, cfg
        self.stats = IngestStats()
        self.skip_labels = {x.strip().lower() for x in cfg.imports.skip_labels}

    def handle(self, source_id: int, ref: str, raw: bytes, labels: str = "") -> None:
        self.stats.messages += 1
        if not mail.has_attachment_hint(raw):
            return
        if self.db.one("SELECT 1 FROM messages WHERE source_id = ? AND source_ref = ?", (source_id, ref)):
            return
        try:
            meta, parts = mail.parse(raw)
        except Exception as e:  # noqa: BLE001 – a broken mail must not stop the run
            log.warning("Mail %s unreadable: %s", ref, e)
            return
        labels = meta.labels or labels
        if self.skip_labels & _split_labels(labels) or not parts:
            return
        cur = self.db.x(
            """INSERT INTO messages(source_id, source_ref, message_id, date, from_name, from_addr, to_addr,
                                    subject, labels, body)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_id, ref, meta.message_id, meta.date, meta.from_name, meta.from_addr, meta.to_addr,
             meta.subject, labels, meta.body),
        )
        self.stats.with_attachments += 1
        for p in parts:
            self._part(cur.lastrowid, p.part, p.filename, p.content_type, p.disposition, p.content_id, p.data)

    def _part(self, msg_id: int, part: str, filename: str | None, ctype: str, disposition: str | None,
              cid: str | None, data: bytes, depth: int = 0) -> None:
        f = self.cfg.imports
        size = len(data)

        def record(sha: str | None = None, reason: str | None = None) -> None:
            self.db.x(
                """INSERT OR IGNORE INTO attachments(message_id, part, filename, content_type, size, sha256,
                                                     skip_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (msg_id, part, filename, ctype, size, sha, reason),
            )
            self.stats.attachments += 1
            if reason:
                self.stats.skipped[reason] += 1

        sniffed = store.sniff(data, filename)
        if sniffed is None:
            ext = Path(filename).suffix.lower() if filename else ""
            return record(reason=f"type {ctype or '?'}" + (f" ({ext})" if ext else ""))
        if sniffed.kind == "zip":
            if f.unpack_zip and depth == 0:
                for j, (name, inner) in enumerate(store.unpack_zip(data)):
                    self._part(msg_id, f"{part}/{j}", name, "", None, None, inner, depth + 1)
                return record(reason="ZIP archive (unpacked)")
            return record(reason="ZIP archive")
        if sniffed.kind == "image":
            if f.skip_inline_images and cid and disposition != "attachment":
                return record(reason="inline image")
            if size < f.min_image_kb * 1024:
                return record(reason="small image")
            dims = store.image_size(data)
            if dims and min(dims) < f.min_image_side:
                return record(reason="small image")

        sha, md5 = store.hashes(data)
        if not self.db.one("SELECT 1 FROM documents WHERE sha256 = ?", (sha,)):
            rel = store.store_blob(self.cfg.files_dir, data, sha, sniffed.ext)
            name = filename or f"attachment-{part.replace('/', '-')}.{sniffed.ext}"
            self.db.x(
                """INSERT INTO documents(sha256, md5, kind, ext, mime, size, path, filename, added_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sha, md5, sniffed.kind, sniffed.ext, sniffed.mime, size, rel, name, now_iso()),
            )
            self.stats.new_docs += 1
        record(sha=sha)


# Called while reading with the share read so far (None if unknown) and the counts; returning False stops
# reading (the position is saved). With a callback the reader commits at every call instead of logging.
OnProgress = Callable[[float | None, IngestStats], bool]


def ingest_mbox(db: DB, cfg: Config, path: Path, rescan: bool = False, limit: int | None = None,
                on_progress: OnProgress | None = None) -> IngestStats:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MBOX file not found: {path}")
    total = path.stat().st_size
    src = db.source("mbox", str(path))
    start = 0
    if src["cursor"] and not rescan:
        start = int(src["cursor"])
        if start > total:
            log.warning("%s is smaller than in the last run – reading from the start", path.name)
            start = 0
    if src["done"] and start >= total and not rescan:
        log.info("%s has already been read completely", path.name)
        return IngestStats()

    ing = Ingestor(db, cfg)
    log.info("Reading %s (%.1f GB) from byte %d", path.name, total / 1e9, start)
    last = time.monotonic()
    count = 0
    finished = True
    every = 0.5 if on_progress else 10
    for offset, raw in iter_mbox(path, start):
        ing.handle(src["id"], str(offset), raw)
        count += 1
        if time.monotonic() - last > every:
            db.set_cursor(src["id"], str(offset))
            db.commit()
            last = time.monotonic()
            if on_progress is None:
                log.info("  %5.1f %%  %d mails, %d new documents", 100 * offset / max(total, 1), ing.stats.messages,
                         ing.stats.new_docs)
            elif not on_progress(offset / max(total, 1), ing.stats):
                finished = False
                break
        if limit and count >= limit:
            db.set_cursor(src["id"], str(offset))
            finished = False
            break
    if finished:
        db.set_cursor(src["id"], str(total), done=True)
    db.commit()
    return ing.stats


def _imap_labels(raw: str) -> str:
    tokens = re.findall(r'"((?:[^"\\]|\\.)*)"|(\S+)', raw)
    return ",".join((a or b).replace("\\\\", "\\").lstrip("\\") for a, b in tokens)


def ingest_imap(db: DB, cfg: Config, limit: int | None = None, rescan: bool = False,
                on_progress: OnProgress | None = None) -> IngestStats:
    c = cfg.imap
    imap = GmailImap(c.host, c.user, c.password, c.folder)
    imap.connect()
    try:
        src = db.source("imap", f"{c.user}@{c.host}/{imap.folder}?{c.query}")
        after = 0
        if src["cursor"] and not rescan:
            validity, _, last_uid = src["cursor"].partition(":")
            if validity == imap.uidvalidity:
                after = int(last_uid or 0)
            else:
                log.warning("UIDVALIDITY changed – checking all mails again")
        ing = Ingestor(db, cfg)
        last = time.monotonic()
        every = 0.5 if on_progress else 10
        for n, (uid, raw, labels) in enumerate(imap.iter_messages(c.query, after), 1):
            ing.handle(src["id"], f"{imap.uidvalidity}:{uid}", raw, _imap_labels(labels))
            db.set_cursor(src["id"], f"{imap.uidvalidity}:{uid}")
            if time.monotonic() - last > every:
                db.commit()
                last = time.monotonic()
                if on_progress is None:
                    log.info("  %d mails, %d new documents", ing.stats.messages, ing.stats.new_docs)
                elif not on_progress(None, ing.stats):
                    break
            if limit and n >= limit:
                break
        db.commit()
        return ing.stats
    finally:
        imap.close()
