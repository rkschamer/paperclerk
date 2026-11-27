"""`paperclerk import`: mail attachments -> extract -> merge -> relevance -> enrich -> upload."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from sqlite3 import Row

from .config import Config
from .db import DB
from .decision import UPLOAD_DECISIONS, decision
from .enrich import VERSION, Enricher, Source, refresh_vocabulary, vocabulary
from .ingest import ingest_imap, ingest_mbox
from .merge import run_merge
from .ollama import Ollama, OllamaError, OllamaUnavailable
from .scoring import MAX_CONSECUTIVE_FAILS, MAX_FAILS, mail_lines, run_scoring, vision_images
from .textextract import run_extract
from .upload import preview_upload, run_upload
from .util import Progress

log = logging.getLogger(__name__)


def source_for(db: DB, cfg: Config, row: Row) -> Source:
    lines, mail = mail_lines(db, row)
    return Source(
        text=row["text"] or "",
        filename=row["filename"] or "",
        context=lines,
        reference_date=(mail["date"] if mail else "") or row["added_at"],
        sender_name=(mail["from_name"] if mail else "") or "",
        images=vision_images(row, cfg),
        date_fallback=True,
    )


def pending_enrich(db: DB, cfg: Config, sha: str | None = None) -> list[Row]:
    """Documents to import that have no metadata from the current pipeline and model (or only `sha`)."""
    rows = db.q(
        """SELECT * FROM documents
            WHERE skip_reason IS NULL AND part_of IS NULL AND (noul1 IS NOT NULL OR manual = 'yes')
              AND (meta IS NULL OR meta_version IS NULL OR meta_version < ? OR model3 IS NOT ?)
              AND (upload_status IS NULL OR upload_status IN ('failed', 'similar'))
              AND fail_count < ?""" + (" AND sha256 = ?" if sha else "") + " ORDER BY rowid",
        (VERSION, cfg.enrich.model, MAX_FAILS, *((sha,) if sha else ())))
    return [r for r in rows if decision(r, cfg) in UPLOAD_DECISIONS]


def has_metadata(row: Row, cfg: Config) -> bool:
    """Metadata from the current pipeline and model (older metadata is determined again)."""
    return bool(row["meta"]) and row["meta_version"] == VERSION and row["model3"] == cfg.enrich.model


def _recent_correspondents(db: DB) -> list[str]:
    counts: Counter = Counter()
    for r in db.q("SELECT meta FROM documents WHERE meta IS NOT NULL AND meta_version = ?", (VERSION,)):
        counts[json.loads(r["meta"]).get("correspondent", "")] += 1
    return [name for name, _ in counts.most_common() if name]


def open_enricher(db: DB, cfg: Config, ollama: Ollama) -> Enricher:
    """The vocabulary from Paperless (or its cached copy) and the correspondents found earlier in the import."""
    refresh_vocabulary(db, cfg)
    ollama.require(cfg.enrich.model)
    return Enricher(cfg, vocabulary(db, cfg, _recent_correspondents(db)), ollama)


def enrich_one(db: DB, cfg: Config, enricher: Enricher, row: Row) -> dict:
    """Ask the model for the metadata of one document and store it. A failed request is recorded with the
    document and re-raised."""
    sha, model = row["sha256"], cfg.enrich.model
    t0 = time.monotonic()
    try:
        suggestion, res = enricher(source_for(db, cfg, row))
    except OllamaUnavailable:
        raise
    except OllamaError as e:
        db.update_doc(sha, fail_count=row["fail_count"] + 1, last_error=str(e)[:500])
        db.log_llm(sha, "enrich", model, time.monotonic() - t0, error=str(e)[:2000])
        db.commit()
        raise
    secs = time.monotonic() - t0
    db.update_doc(sha, meta=json.dumps(suggestion, ensure_ascii=False), meta_version=VERSION, model3=model,
                  secs3=round(secs, 2), fail_count=0, last_error=None)
    db.log_llm(sha, "enrich", model, secs, res.input_tokens, res.output_tokens, res.content)
    db.commit()
    return suggestion


def run_enrich(db: DB, cfg: Config, limit: int | None = None) -> int:
    rows = pending_enrich(db, cfg)
    rows = rows[:limit] if limit else rows
    if not rows:
        return 0
    ollama = Ollama(cfg.ollama.url, cfg.ollama.keep_alive, cfg.ollama.timeout)
    try:
        enricher = open_enricher(db, cfg, ollama)
        vocab = enricher.vocab
        log.info("Metadata with %s for %d documents (%d known correspondents, %d document types, %d tags)",
                 cfg.enrich.model, len(rows), len(vocab.correspondents), len(vocab.document_types), len(vocab.tags))
        progress = Progress(len(rows), "Metadata")
        consecutive = 0
        for row in rows:
            t0 = time.monotonic()
            try:
                s = enrich_one(db, cfg, enricher, row)
            except OllamaUnavailable:
                raise
            except OllamaError as e:
                consecutive += 1
                progress.step(f"{row['filename']}: ERROR {e}", time.monotonic() - t0)
                if consecutive >= MAX_CONSECUTIVE_FAILS:
                    raise OllamaUnavailable(f"{consecutive} errors in a row – stopping. Last error: {e}")
                continue
            consecutive = 0
            progress.step(f"{row['filename']}: {s['correspondent'] or '?'} | {s['document_type'] or '-'} | "
                          f"{s['date']} | {s['title']} | {', '.join(s['tags'])}", time.monotonic() - t0)
        progress.done()
        return len(rows)
    finally:
        ollama.close()


def run_import(db: DB, cfg: Config, source: str, paths: list[Path], dry_run: bool = False,
               limit: int | None = None) -> dict:
    if source == "mbox":
        for path in paths:
            log.info("%s", ingest_mbox(db, cfg, path).summary())
    else:
        log.info("%s", ingest_imap(db, cfg).summary())
    run_extract(db, cfg, limit)
    run_merge(db, cfg)
    run_scoring(db, cfg, "score", limit)
    run_scoring(db, cfg, "escalate", limit)
    run_enrich(db, cfg, limit)
    if dry_run:
        print()
        preview_upload(db, cfg, limit)
        print("Dry run: nothing uploaded. Run the same command without --dry-run in a terminal to check, correct "
              "and upload the documents in the import view.")
        return {}
    return run_upload(db, cfg, limit)
