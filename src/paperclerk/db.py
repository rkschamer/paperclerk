from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,
    locator     TEXT NOT NULL,
    cursor      TEXT,
    done        INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT,
    UNIQUE(kind, locator)
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    source_id   INTEGER NOT NULL REFERENCES sources(id),
    source_ref  TEXT NOT NULL,
    message_id  TEXT,
    date        TEXT,
    from_name   TEXT,
    from_addr   TEXT,
    to_addr     TEXT,
    subject     TEXT,
    labels      TEXT,
    body        TEXT,
    UNIQUE(source_id, source_ref)
);

-- one row per unique imported file (SHA-256); merged documents are rows too (parts = JSON list)
CREATE TABLE IF NOT EXISTS documents (
    sha256        TEXT PRIMARY KEY,
    md5           TEXT NOT NULL,
    kind          TEXT NOT NULL,
    ext           TEXT NOT NULL,
    mime          TEXT NOT NULL,
    size          INTEGER NOT NULL,
    path          TEXT NOT NULL,
    filename      TEXT,
    added_at      TEXT NOT NULL,
    skip_reason   TEXT,
    pages         INTEGER,
    text          TEXT,
    text_method   TEXT,
    noul1         REAL,
    types1        TEXT,
    model1        TEXT,
    secs1         REAL,
    noul2         REAL,
    types2        TEXT,
    model2        TEXT,
    secs2         REAL,
    manual        TEXT,
    meta          TEXT,
    meta_version  INTEGER,
    model3        TEXT,
    secs3         REAL,
    overrides     TEXT,
    fail_count    INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    upload_status TEXT,
    task_id       TEXT,
    paperless_id  INTEGER,
    upload_error  TEXT,
    note_added    INTEGER NOT NULL DEFAULT 0,
    uploaded_at   TEXT,
    part_of       TEXT,
    parts         TEXT
);

CREATE TABLE IF NOT EXISTS attachments (
    id           INTEGER PRIMARY KEY,
    message_id   INTEGER NOT NULL REFERENCES messages(id),
    part         TEXT NOT NULL,
    filename     TEXT,
    content_type TEXT,
    size         INTEGER,
    sha256       TEXT REFERENCES documents(sha256),
    skip_reason  TEXT,
    UNIQUE(message_id, part)
);
CREATE INDEX IF NOT EXISTS ix_attachments_sha ON attachments(sha256);

-- answers to "does document b continue document a?"
CREATE TABLE IF NOT EXISTS merge_checks (
    a      TEXT NOT NULL,
    b      TEXT NOT NULL,
    p      REAL NOT NULL,
    model  TEXT,
    at     TEXT,
    PRIMARY KEY (a, b)
);

-- documents already in Paperless that `process` has looked at
CREATE TABLE IF NOT EXISTS processed (
    doc_id        INTEGER PRIMARY KEY,
    title         TEXT,
    content_hash  TEXT,
    suggestion    TEXT,
    version       INTEGER,
    model         TEXT,
    secs          REAL,
    status        TEXT,
    changes       TEXT,
    note_id       INTEGER,
    fail_count    INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS llm_log (
    id            INTEGER PRIMARY KEY,
    sha256        TEXT,
    stage         TEXT,
    model         TEXT,
    secs          REAL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    response      TEXT,
    error         TEXT,
    at            TEXT
);

-- cache of Paperless correspondents, document types, tags and custom fields
CREATE TABLE IF NOT EXISTS paperless_objects (
    kind   TEXT NOT NULL,
    id     INTEGER NOT NULL,
    name   TEXT NOT NULL,
    extra  TEXT,
    PRIMARY KEY (kind, id)
);
"""

_NEW_COLUMNS = {"part_of": "TEXT", "parts": "TEXT", "meta_version": "INTEGER"}
_SKIP_REASONS = {
    "eingebettetes Bild": "inline image", "kleines Bild": "small image", "ZIP-Archiv (entpackt)": "ZIP archive (unpacked)",
    "ZIP-Archiv": "ZIP archive", "PDF passwortgeschützt": "password-protected PDF", "PDF defekt": "broken PDF",
    "PDF ohne Seiten": "PDF without pages", "Bild defekt": "broken image",
}
_OVERRIDE_KEYS = {"korrespondent": "correspondent", "datum": "date", "titel": "title", "dokumenttyp": "document_type"}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _translate_reason(reason: str) -> str:
    if reason in _SKIP_REASONS:
        return _SKIP_REASONS[reason]
    if m := re.fullmatch(r"Typ (.*)", reason):
        return f"type {m.group(1)}"
    if m := re.fullmatch(r"Art '(.*)' deaktiviert", reason):
        return f"kind '{m.group(1)}' disabled"
    if m := re.fullmatch(r"größer als (\d+) MB", reason):
        return f"larger than {m.group(1)} MB"
    return reason


class DB:
    """One SQLite connection. The terminal UIs open one per thread; they write in short transactions only,
    never across a model or network call, because the others wait for the write lock (at most TIMEOUT s)."""

    TIMEOUT = 60

    def __init__(self, path: Path, legacy: Path | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() and legacy is not None and legacy.exists():
            log.info("Copying the maildocs database %s to %s (the old file is kept)", legacy.name, path.name)
            src = sqlite3.connect(legacy)
            dst = sqlite3.connect(path)
            src.backup(dst)
            src.close()
            dst.close()
        self.conn = sqlite3.connect(path, timeout=self.TIMEOUT)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        existing = {r[1] for r in self.conn.execute("PRAGMA table_info(documents)")}
        for col, typ in _NEW_COLUMNS.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE documents ADD COLUMN {col} {typ}")
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return
        from .config import LEGACY_CATEGORY_KEYS  # noqa: PLC0415 – avoids an import cycle

        for row in self.conn.execute("SELECT sha256, types1, types2, manual, overrides FROM documents").fetchall():
            values: dict[str, Any] = {}
            for col in ("types1", "types2"):
                if row[col]:
                    probs = json.loads(row[col])
                    if any(k in LEGACY_CATEGORY_KEYS for k in probs):
                        values[col] = json.dumps({LEGACY_CATEGORY_KEYS.get(k, k): v for k, v in probs.items()})
            if row["manual"] in ("ja", "nein"):
                values["manual"] = "yes" if row["manual"] == "ja" else "no"
            if row["overrides"]:
                over = json.loads(row["overrides"])
                values["overrides"] = json.dumps({_OVERRIDE_KEYS.get(k, k): v for k, v in over.items()},
                                                 ensure_ascii=False)
            if values:
                self.update_doc(row["sha256"], **values)
        for table in ("documents", "attachments"):
            for (reason,) in self.conn.execute(f"SELECT DISTINCT skip_reason FROM {table} "
                                               "WHERE skip_reason IS NOT NULL").fetchall():
                new = _translate_reason(reason)
                if new != reason:
                    self.conn.execute(f"UPDATE {table} SET skip_reason = ? WHERE skip_reason = ?", (new, reason))
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()

    def q(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, tuple(params)).fetchall()

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, tuple(params)).fetchone()

    def x(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, tuple(params))

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        """After a failed step: a statement that failed (e.g. "database is locked") leaves the transaction open
        on an old snapshot, and every later write on this connection would fail at once."""
        try:
            self.conn.rollback()
        except sqlite3.Error:
            pass

    def update_doc(self, sha: str, **values: Any) -> None:
        cols = ", ".join(f"{k} = ?" for k in values)
        self.conn.execute(f"UPDATE documents SET {cols} WHERE sha256 = ?", (*values.values(), sha))

    def source(self, kind: str, locator: str) -> sqlite3.Row:
        self.conn.execute("INSERT OR IGNORE INTO sources(kind, locator, updated_at) VALUES (?, ?, ?)",
                          (kind, locator, now_iso()))
        self.conn.commit()  # even when nothing was inserted the statement holds the write lock until a commit
        return self.one("SELECT * FROM sources WHERE kind = ? AND locator = ?", (kind, locator))

    def set_cursor(self, source_id: int, cursor: str, done: bool = False) -> None:
        self.conn.execute("UPDATE sources SET cursor = ?, done = ?, updated_at = ? WHERE id = ?",
                          (cursor, int(done), now_iso(), source_id))

    def find_docs(self, prefix: str) -> list[sqlite3.Row]:
        prefix = prefix.strip().lower()
        if len(prefix) < 4:
            return []
        return self.q("SELECT * FROM documents WHERE sha256 LIKE ?", (prefix + "%",))

    def mails_for(self, sha: str) -> list[sqlite3.Row]:
        """All mails the file was attached to, earliest first."""
        return self.q(
            """SELECT m.*, a.filename AS att_filename, a.part AS att_part
                 FROM attachments a JOIN messages m ON m.id = a.message_id
                WHERE a.sha256 = ?
                ORDER BY COALESCE(m.date, '9999'), m.id""",
            (sha,),
        )

    def log_llm(self, ref: str, stage: str, model: str, secs: float, input_tokens: int | None = None,
                output_tokens: int | None = None, response: str | None = None, error: str | None = None) -> None:
        self.conn.execute(
            """INSERT INTO llm_log(sha256, stage, model, secs, input_tokens, output_tokens, response, error, at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ref, stage, model, secs, input_tokens, output_tokens, response, error, now_iso()),
        )
