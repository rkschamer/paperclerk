"""`paperclerk import` in a terminal: follow the import of mail attachments and decide what is uploaded.

The view is a layer over the same steps as the plain `import`: reading the mail, text, merging split
documents, relevance, escalation, metadata, upload. A pipeline thread runs them phase by phase like the plain
run, so each model is loaded once; within a phase the document on screen goes first, and documents the user
asks for (g, or a on one without metadata) go before everything else. Nothing is uploaded until the user
accepts it. A second thread talks to Paperless (vocabulary, uploads) and stores the user's decisions, so the
UI thread never waits for the network or for a database lock; it only reads the database.
"""

from __future__ import annotations

import json
import logging
import queue
import statistics
import threading
import time
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import SystemCommand
from textual.binding import Binding
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Static

from .config import Config
from .db import DB
from .decision import (ESCALATE, IRRELEVANT, PENDING, RELEVANT, REVIEW, UPLOAD_DECISIONS, decision, evaluate,
                       overrides, probability, relevance)
from .draft import Draft, import_corrections, imported, join_tags, merge
from .enrich import vocabulary
from .importer import enrich_one, has_metadata, open_enricher, pending_enrich
from .ingest import IngestStats, ingest_imap, ingest_mbox
from .merge import candidate_groups, run_merge
from .normalize import CorrespondentNormalizer, domain_label
from .ollama import Ollama, OllamaError, OllamaUnavailable
from .paperless import Paperless, PaperlessError, Registry, cached, sync_cache
from .scoring import KIND_LABEL, MAX_CONSECUTIVE_FAILS, MAX_FAILS, Scorer, score_one, stage_model
from .scoring import pending as pending_score
from .sources import ImapError
from .textextract import TextExtractor, extract_one, pending_extract
from .ui import (LABEL_STYLE, Confirm, Crumbs, DocScreen, Help, ListScreen, ReviewApp, View, _cut, _message,
                 get_value, run_app)
from .upload import Uploader
from .util import fmt_duration

log = logging.getLogger(__name__)

INFO_WIDTH = 64  # the status block in the header (the process view has less to say)
PHASES = {"read": "Reading mail", "text": "Text", "merge": "Merging", "score": "Relevance",
          "escalate": "Escalation", "enrich": "Metadata"}


# ---------------------------------------------------------------- messages from the background threads

class Rows(Message):
    """Documents changed in the database: these, all (shas None), and/or new ones may exist."""

    def __init__(self, shas: Iterable[str] | None = (), new: bool = False):
        super().__init__()
        self.shas, self.new = (list(shas) if shas is not None else None), new


class AskDone(Message):
    def __init__(self, sha: str):
        super().__init__()
        self.sha = sha


class Vocab(Message):
    """Names for completion; error: Paperless was not reachable (the cached names are used)."""

    def __init__(self, names: dict[str, list[str]], error: str = ""):
        super().__init__()
        self.names, self.error = names, error


class PipelineFailed(Message):
    def __init__(self, error: str, fatal: bool):
        super().__init__()
        self.error, self.fatal = error, fatal


class Uploaded(Message):
    """An upload went out (first=True: quitting no longer waits for it) or Paperless finished consuming it."""

    def __init__(self, sha: str, status: str | None, message: str, first: bool):
        super().__init__()
        self.sha, self.status, self.message, self.first = sha, status, message, first


# ---------------------------------------------------------------- background work

class Control:
    """What the pipeline thread is told by the UI: pause, the document on screen, documents asked for."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self.paused = False
        self.closed = False
        self.preferred: str | None = None
        self.asked: dict[str, bool] = {}  # sha -> force: ask for metadata again even if there is some
        self.changed = False  # decisions changed: read the pending documents again
        self.reread = False  # read the mail again (F5)

    def pause(self, paused: bool) -> None:
        with self._cv:
            self.paused = paused
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self.closed = True
            self._cv.notify_all()

    def ask(self, sha: str, force: bool = False) -> None:
        with self._cv:
            self.asked[sha] = self.asked.get(sha, False) or force
            self.changed = True
            self._cv.notify_all()

    def wake(self, reread: bool = False) -> None:
        with self._cv:
            self.changed = True
            self.reread |= reread
            self._cv.notify_all()

    def take_asked(self) -> tuple[str, bool] | None:
        with self._cv:
            if not self.asked:
                return None
            sha = next(iter(self.asked))
            return sha, self.asked.pop(sha)

    def take_changed(self) -> bool:
        with self._cv:
            changed, self.changed = self.changed, False
            return changed

    def take_reread(self) -> bool:
        with self._cv:
            reread, self.reread = self.reread, False
            return reread

    def proceed(self) -> bool:
        """Blocks while paused; False once the UI has closed."""
        with self._cv:
            while self.paused and not self.closed:
                self._cv.wait()
            return not self.closed

    def idle(self) -> bool:
        """Blocks until there is something to do; False once the UI has closed."""
        with self._cv:
            while not self.closed and (self.paused or not (self.changed or self.asked or self.reread)):
                self._cv.wait()
            return not self.closed


def _median(db: DB, col: str) -> float | None:
    values = [r[0] for r in db.q(f"SELECT {col} FROM documents WHERE {col} IS NOT NULL ORDER BY rowid DESC LIMIT 50")]
    return statistics.median(values) if values else None


def names_for(db: DB, cfg: Config) -> dict[str, list[str]]:
    """Completion lists for the form, from the cached Paperless objects and the configuration."""
    vocab = vocabulary(db, cfg)
    corr = sorted(cached(db, "correspondents"), key=lambda o: -(o.get("document_count") or 0))
    return {"correspondent": [o["name"] for o in corr if not domain_label(o["name"])],
            "document_type": vocab.document_types, "tags": vocab.tags}


class ImportEngine:
    """The pipeline thread and the Paperless thread. `post` delivers messages to the UI (thread-safe)."""

    def __init__(self, cfg: Config, post: Callable[[Message], object], source: str, paths: list[Path],
                 limit: int | None = None):
        self.cfg, self.post, self.source, self.paths, self.limit = cfg, post, source, paths, limit
        self.jobs = Control()
        self.tasks: queue.Queue = queue.Queue()
        # read by the UI, written by the pipeline thread
        self.phase: str | None = "read"
        self.phase_done = self.phase_total = 0
        self.running: str | None = None
        self.running_phase: str | None = None
        self.started_at = 0.0
        self.model = ""
        self.reading = ""
        self.durations: dict[str, list[float]] = defaultdict(list)
        self.typical: dict[str, float | None] = {}
        self.waiting = 0  # uploads in Paperless' queue
        self._counts: Counter = Counter()
        self._clients: dict = {}
        self._consecutive = 0

    def start(self) -> None:
        DB(self.cfg.db_path).close()  # create or migrate the file once; two threads doing it at once collide
        for target, name in ((self._paperless_loop, "Paperless"), (self._pipeline_loop, "pipeline")):
            threading.Thread(target=self._guard, args=(target, name), name=f"paperclerk-{name}", daemon=True).start()
        self.load()

    def _guard(self, target: Callable[[], None], name: str) -> None:
        """A thread that dies must say so, or the UI would wait for it forever."""
        try:
            target()
        except Exception as e:  # noqa: BLE001
            log.exception("%s thread stopped", name)
            self.post(PipelineFailed(f"internal error in the {name} thread, see the log: {_message(e)}", fatal=True))

    def stop(self) -> None:
        self.jobs.close()
        self.tasks.put(None)

    def load(self) -> None:
        self.tasks.put(("load",))

    def reload(self) -> None:
        self.load()
        self.jobs.wake(reread=True)

    def decide(self, sha: str, manual: str, ask: bool = False) -> None:
        self.tasks.put(("decide", sha, manual, ask))

    def upload(self, sha: str, corrections: dict) -> None:
        self.tasks.put(("upload", sha, corrections))

    def ask(self, sha: str, force: bool = False) -> None:
        self.jobs.ask(sha, force)

    # -------------------------------------------------------- pipeline thread

    def _pipeline_loop(self) -> None:
        db = DB(self.cfg.db_path)
        self.typical = {"score": _median(db, "secs1"), "escalate": _median(db, "secs2"), "enrich": _median(db, "secs3")}
        try:
            reread = True
            while not self.jobs.closed:
                if reread and not self._read(db):
                    break
                if not self._phases(db):
                    break
                self._set_phase(None)
                if not self.jobs.idle():
                    break
                reread = self.jobs.take_reread()
        finally:
            self._close_clients()
            db.close()

    def _set_phase(self, phase: str | None, done: int = 0, total: int = 0) -> None:
        self.phase, self.phase_done, self.phase_total = phase, done, total  # the header shows it every second

    def _read(self, db: DB) -> bool:
        """Read new mail (the position is kept, so a finished MBOX is skipped at once)."""
        self._set_phase("read")
        self.reading = "starting…"

        def progress(share: float | None, stats: IngestStats) -> bool:
            self.reading = (f"{share:.0%} · " if share is not None else "") + _found(stats)
            self.post(Rows(new=True))
            return not self.jobs.closed

        try:
            total = IngestStats()
            for path in (self.paths if self.source == "mbox" else [None]):
                stats = (ingest_mbox(db, self.cfg, path, on_progress=progress) if path is not None
                         else ingest_imap(db, self.cfg, on_progress=progress))
                total.messages += stats.messages
                total.new_docs += stats.new_docs
            done = all(s["done"] for s in db.q("SELECT done FROM sources WHERE kind = ?", (self.source,)))
            self.reading = ("read" if done or self.source == "imap" else "stopped") + (
                f" · {_found(total)}" if total.messages else "")
        except (ImapError, OSError) as e:
            db.commit()  # what was read so far, with its position
            self.reading = f"error: {e}"
            self.post(PipelineFailed(f"Reading mail failed: {e}", fatal=False))
        except Exception as e:  # noqa: BLE001 – e.g. the database is locked: the other steps go on
            db.rollback()
            log.exception("reading mail failed")
            self.reading = f"error: {_message(e)}"
            self.post(PipelineFailed(f"Reading mail failed: {_message(e)}", fatal=False))
        self.post(Rows(new=True))
        return not self.jobs.closed

    def _phases(self, db: DB) -> bool:
        for phase in ("text", "merge", "score", "escalate", "enrich"):
            if not (self._merge(db) if phase == "merge" else self._phase(db, phase)):
                return False
        return True

    def _pending(self, db: DB, phase: str, sha: str | None = None) -> list:
        if phase == "text":
            return pending_extract(db, sha)
        if phase in ("score", "escalate"):
            return pending_score(db, self.cfg, phase, sha)
        return pending_enrich(db, self.cfg, sha)

    def _phase(self, db: DB, phase: str) -> bool:
        """One step for every document that needs it, the one on screen first. False: the UI has closed."""
        self.jobs.take_changed()
        shas = [r["sha256"] for r in self._pending(db, phase)]
        tried: set[str] = set()
        while True:
            if not self.jobs.proceed():
                return False
            if (asked := self.jobs.take_asked()) is not None:
                self._asked(db, *asked)
                continue
            if self.jobs.take_changed():
                shas = [r["sha256"] for r in self._pending(db, phase) if r["sha256"] not in tried]
            if not shas or (self.limit and self._counts[phase] >= self.limit):
                return True
            self._set_phase(phase, len(tried), len(tried) + len(shas))
            sha = self.jobs.preferred if self.jobs.preferred in shas else shas[0]
            shas.remove(sha)
            tried.add(sha)
            rows = self._pending(db, phase, sha)  # still to do? the user may have decided meanwhile
            if not rows:
                continue
            if self._step(db, phase, rows[0]):
                self._counts[phase] += 1
            else:  # the model is unavailable: this document goes first once it is back
                tried.discard(sha)
                shas.insert(0, sha)

    def _asked(self, db: DB, sha: str, force: bool) -> None:
        """A document the user asked for: every step it still needs, now."""
        try:
            for phase in ("text", "score", "escalate", "enrich"):
                row = db.one("SELECT * FROM documents WHERE sha256 = ?", (sha,))
                if row is None or row["skip_reason"] or row["part_of"]:
                    return
                needed = {
                    "text": row["text_method"] is None,
                    "score": row["text_method"] is not None and row["noul1"] is None and row["manual"] is None,
                    "escalate": decision(row, self.cfg) == ESCALATE,
                    "enrich": decision(row, self.cfg) in UPLOAD_DECISIONS
                              and (force or not has_metadata(row, self.cfg)),
                }[phase]
                if needed and not self._step(db, phase, row):
                    self.jobs.ask(sha, force)  # the model is unavailable: again once it is back
                    return
        finally:
            self.post(AskDone(sha))

    def _merge(self, db: DB) -> bool:
        if not self.cfg.merge.enabled or not candidate_groups(db):
            return True
        self._set_phase("merge")
        self.model = stage_model(self.cfg, "score")
        while True:
            try:
                if run_merge(db, self.cfg):
                    self.post(Rows(None, new=True))
                return True
            except OllamaUnavailable as e:
                self._model_down(str(e))
                if not self.jobs.proceed():
                    return False
            except Exception as e:  # noqa: BLE001 – shown; the other steps go on
                db.rollback()
                log.exception("merging failed")
                self.post(PipelineFailed(f"Merging failed: {_message(e)}", fatal=False))
                return True

    def _step(self, db: DB, phase: str, row) -> bool:
        """Run one step for one document. False: the model is unavailable (the pipeline is paused)."""
        sha, cfg = row["sha256"], self.cfg
        self.running, self.running_phase, self.started_at = sha, phase, time.monotonic()
        self.model = {"text": "none (text, OCR)", "enrich": cfg.enrich.model}.get(phase) or stage_model(cfg, phase)
        self.post(Rows([sha]))
        try:
            if phase == "text":
                if "text" not in self._clients:
                    self._clients["text"] = TextExtractor(cfg)
                extract_one(db, cfg, self._clients["text"], row)
            elif phase in ("score", "escalate"):
                score_one(db, cfg, self._scorer(db, phase), row)
            else:
                enrich_one(db, cfg, self._enricher(db), row)
            if phase != "text":
                self.durations[phase].append(time.monotonic() - self.started_at)
                self._consecutive = 0
            return True
        except OllamaUnavailable as e:
            self._model_down(str(e))
            return False
        except OllamaError as e:  # recorded with the document; the next one goes on
            self._consecutive += 1
            if self._consecutive >= MAX_CONSECUTIVE_FAILS:
                self._consecutive = 0
                self._model_down(f"{MAX_CONSECUTIVE_FAILS} errors in a row. Last error: {e}")
            return True
        except Exception as e:  # noqa: BLE001 – shown; the next document goes on
            db.rollback()
            log.exception("%s of %s failed", PHASES[phase], sha[:12])
            self.post(PipelineFailed(f"{PHASES[phase]} of {row['filename']}: {_message(e)}", fatal=False))
            return True
        finally:
            self.running = self.running_phase = None
            self.post(Rows([sha]))

    def _model_down(self, error: str) -> None:
        self.jobs.pause(True)
        self._close_clients()
        self.post(PipelineFailed(error, fatal=True))

    def _ollama(self) -> Ollama:
        if "ollama" not in self._clients:
            o = self.cfg.ollama
            self._clients["ollama"] = Ollama(o.url, o.keep_alive, o.timeout)
        return self._clients["ollama"]

    def _scorer(self, db: DB, stage: str) -> Scorer:
        key = f"scorer:{stage}"
        if key not in self._clients:
            model = stage_model(self.cfg, stage)
            self._ollama().require(model)
            self._clients[key] = Scorer(db, self.cfg, self._ollama(), model, stage)
        return self._clients[key]

    def _enricher(self, db: DB):
        if "enricher" not in self._clients:
            self._clients["enricher"] = open_enricher(db, self.cfg, self._ollama())
        return self._clients["enricher"]

    def _close_clients(self) -> None:
        if "ollama" in self._clients:
            self._clients["ollama"].close()
        self._clients = {k: v for k, v in self._clients.items() if k == "text"}

    # -------------------------------------------------------- Paperless thread

    def _paperless_loop(self) -> None:
        db = DB(self.cfg.db_path)
        self.pl: Paperless | None = None
        self.uploader: Uploader | None = None
        try:
            while True:
                try:
                    task = self.tasks.get(timeout=3 if self.uploader and self.uploader.pending else None)
                except queue.Empty:
                    task = ("poll",)
                if task is None:
                    break
                try:
                    self._paperless_task(db, task)
                except Exception as e:  # noqa: BLE001 – shown in the UI; the thread keeps running
                    db.rollback()
                    if not isinstance(e, PaperlessError):
                        log.exception("%s failed", task[0])
                    if task[0] == "upload":
                        self.post(Uploaded(task[1], None, f"Not uploaded: {_message(e)}", first=True))
                    elif task[0] == "load":
                        self.post(Vocab(names_for(db, self.cfg), _message(e)))
                    elif task[0] == "decide":
                        self.post(PipelineFailed(f"Decision not saved: {_message(e)}", fatal=False))
                    else:
                        log.warning("Paperless: %s", _message(e))
        finally:
            if self.pl is not None:
                self.pl.close()
            db.close()

    def _connect(self, db: DB) -> None:
        if self.pl is not None:
            self.pl.close()
        self.pl = self.uploader = None
        pl = Paperless.from_config(self.cfg)
        try:
            objects = sync_cache(db, pl)
        except PaperlessError:
            pl.close()
            raise
        self.pl = pl
        self.normalizer = CorrespondentNormalizer([o["name"] for o in objects["correspondents"]], self.cfg.aliases)
        self.uploader = Uploader(db, self.cfg, pl, Registry(db, pl, objects))
        changed = self.uploader.catch_up()
        if changed:
            self.post(Rows(changed))

    def _paperless_task(self, db: DB, task: tuple) -> None:
        kind = task[0]
        if kind == "decide":
            _, sha, manual, ask = task
            db.update_doc(sha, manual=manual)
            db.commit()
            self.post(Rows([sha]))
            if ask:
                self.jobs.ask(sha)
            else:
                self.jobs.wake()
            return
        if kind == "load":
            self._connect(db)
            self.post(Vocab(names_for(db, self.cfg)))
            return
        if kind == "upload":
            _, sha, corrections = task
            db.update_doc(sha, manual="yes",
                          overrides=json.dumps(corrections, ensure_ascii=False) if corrections else None)
            db.commit()
            self.post(Rows([sha]))
            if self.uploader is None:
                self._connect(db)
            row = db.one("SELECT * FROM documents WHERE sha256 = ?", (sha,))
            message = self.uploader.upload(row, self.normalizer)
            status = db.one("SELECT upload_status FROM documents WHERE sha256 = ?", (sha,))["upload_status"]
            self.post(Uploaded(sha, status, message, first=True))
            if len(self.uploader.pending) >= self.cfg.imports.max_pending:
                self._finished(db, self.uploader.wait(self.cfg.imports.max_pending - 1))
        elif kind == "poll" and self.uploader is not None:
            self._finished(db, self.uploader.poll())
        self.waiting = len(self.uploader.pending) if self.uploader else 0

    def _finished(self, db: DB, shas: list[str]) -> None:
        for sha in shas:
            row = db.one("SELECT upload_status, paperless_id, upload_error FROM documents WHERE sha256 = ?", (sha,))
            self.post(Uploaded(sha, row["upload_status"], row["upload_error"] or "", first=False))


def _found(stats: IngestStats) -> str:
    return f"{stats.messages:,} mails · {stats.new_docs:,} new"


# ---------------------------------------------------------------- state of one document in the UI

STYLES = {"new": "", "queued": "dim", "running": "bold yellow", "ready": "bold green", "uncertain": "bold yellow",
          "edited": "bold cyan", "irrelevant": "dim", "rejected": "dim", "failed": "bold red", "saving": "yellow",
          "done": "green", "exists": "green", "similar": "yellow", "skipped": "dim", "part": "dim"}
OPEN = ("new", "queued", "running", "ready", "uncertain", "failed")  # still to be decided
FINAL = ("saving", "done", "exists", "skipped", "part")  # nothing left to decide
DECISION_WORDS = {RELEVANT: "relevant", REVIEW: "uncertain", ESCALATE: "uncertain", IRRELEVANT: "irrelevant"}
COLUMNS = (("state", "STATUS"), ("rel", "REL"), ("filename", "FILE"), ("title", "TITLE"),
           ("correspondent", "CORRESPONDENT"), ("document_type", "TYPE"), ("date", "DATE"), ("tags", "TAGS"))


def import_state(row: dict, cfg: Config, running: bool, saving: bool, asked: bool) -> str:
    status = row["upload_status"]
    if saving or status == "pending":
        return "saving"
    if status == "done":
        return "done"
    if status in ("exists", "duplicate"):
        return "exists"
    if row["part_of"]:
        return "part"
    if row["skip_reason"]:
        return "skipped"
    if running:
        return "running"
    if asked:
        return "queued"
    if row["text_method"] is None:
        return "new"
    d = decision(row, cfg)
    if d == IRRELEVANT:
        return "rejected" if row["manual"] == "no" else "irrelevant"
    if status in ("similar", "failed"):
        return status
    if d in (PENDING, ESCALATE) or not has_metadata(row, cfg):
        return "failed" if row["fail_count"] >= MAX_FAILS else "queued"
    return "uncertain" if d == REVIEW else "ready"


@dataclass
class ImportItem:
    row: dict  # the documents row without its text
    mail: dict | None  # the earliest mail it came with
    base: Draft  # the model's metadata with stored corrections
    draft: Draft  # what the form shows: base plus the user's edits
    state: str = "new"
    has_meta: bool = False
    saving: bool = False  # an upload is on its way
    asked: bool = False  # the user asked for the model; the pipeline has not got to it yet
    error: str = ""

    @property
    def id(self) -> str:
        return self.row["sha256"]

    @property
    def open(self) -> bool:
        return self.state in OPEN

    @property
    def editable(self) -> bool:
        return self.state not in FINAL

    @property
    def edited(self) -> bool:
        return self.editable and self.draft != self.base

    @property
    def label(self) -> str:
        return "edited" if self.edited and self.state not in ("running", "rejected") else self.state

    @property
    def suggestion(self) -> dict | None:
        """Something to check: metadata the model has determined."""
        return json.loads(self.row["meta"]) if self.has_meta else None

    @property
    def filename(self) -> str:
        return self.row["filename"] or self.id[:12]


def sender(mail: dict | None) -> str:
    if not mail:
        return ""
    return f"{mail['from_name']} <{mail['from_addr']}>" if mail["from_name"] else (mail["from_addr"] or "")


def first_mails(db: DB, shas: list[str] | None = None) -> dict[str, dict]:
    """The earliest mail of each document (all documents if shas is None)."""
    sql = ("SELECT a.sha256, m.id, m.date, m.from_name, m.from_addr, m.subject FROM attachments a "
           "JOIN messages m ON m.id = a.message_id WHERE a.sha256 IS NOT NULL{} "
           "ORDER BY COALESCE(m.date, '9999'), m.id")
    rows: list = []
    if shas is None:
        rows = db.q(sql.format(""))
    for i in range(0, len(shas or []), 500):
        chunk = shas[i:i + 500]
        rows += db.q(sql.format(f" AND a.sha256 IN ({','.join('?' * len(chunk))})"), chunk)
    result: dict[str, dict] = {}
    for r in rows:
        result.setdefault(r["sha256"], dict(r))
    return result


# ---------------------------------------------------------------- the processing steps of one document

MARKS = {"ok": ("✓", "green"), "wait": ("·", "dim"), "run": ("▶", "bold yellow"), "ask": ("?", "bold yellow"),
         "no": ("✗", "red"), "off": ("–", "dim")}


def _pct(p: float | None) -> str:
    return f"{p:.0%}" if p is not None else "–"


def _categories(types: str | None, n: int = 2) -> str:
    """The most likely categories (those of at least 5 %)."""
    probs = json.loads(types) if types else {}
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:n]
    return ", ".join(f"{k} {v:.0%}" for i, (k, v) in enumerate(top) if i == 0 or v >= 0.05)


Step = tuple[str, str, list]  # mark, name, lines (str or Text)


def render_steps(rows: list[Step]) -> Table:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1, no_wrap=True)
    grid.add_column(width=10, no_wrap=True, style="bold")
    grid.add_column(ratio=1)
    for mark, name, lines in rows:
        symbol, style = MARKS[mark]
        grid.add_row(Text(symbol, style), name,
                     Text("\n").join(line if isinstance(line, Text) else Text(line) for line in lines if line))
    return grid


def plain_steps(rows: list[Step]) -> str:
    return "\n".join(f"{name}: " + " | ".join(str(line) for line in lines if line) for _, name, lines in rows)


def steps(db: DB, cfg: Config, item: ImportItem, text: str, running_phase: str | None) -> list[Step]:
    """Every step of the import for one document: where it came from and what each step found."""
    row, sha = item.row, item.id
    out: list[Step] = []

    def add(name: str, mark: str, *lines: str | Text) -> None:
        out.append((mark, name, list(lines)))

    def waiting(phase: str, what: str) -> tuple[str, str]:
        if running_phase == phase:
            return "run", f"{what} is at work…"
        if row["fail_count"] >= MAX_FAILS:
            return "no", f"failed {row['fail_count']}× – g tries again: {row['last_error'] or ''}"
        return "wait", f"waiting for {what}"

    mails = db.mails_for(sha)
    if mails:
        m = mails[0]
        add("Mail", "ok", f"{(m['date'] or '?')[:16].replace('T', ' ')} · {sender(m)}",
            f"Subject: {m['subject'] or ''}",
            f"Attachment: {m['att_filename'] or ''}",
            f"the same file came with {len(mails) - 1} more mail{'s' if len(mails) > 2 else ''}"
            if len(mails) > 1 else "")
    pages = f" · {row['pages']} page{'s' if row['pages'] != 1 else ''}" if row["pages"] else ""
    add("File", "ok", f"{KIND_LABEL.get(row['kind'], row['kind'])}{pages} · {row['size'] / 1024:,.0f} KB · {sha[:12]}")

    method = row["text_method"]
    if row["skip_reason"]:
        add("Text", "no", f"not imported: {row['skip_reason']}")
    elif method is None:
        add("Text", "run" if running_phase == "text" else "wait",
            "extracting…" if running_phase == "text" else "waiting")
    else:
        how = {"pdf": "from the PDF", "ocr": "by OCR", "office": "from the office file"}.get(method)
        add("Text", "ok", f"{how}, {len(text):,} characters (first {cfg.text.max_pages} pages)" if how else
            "no readable text – the models see the first page as an image")

    if row["parts"]:
        parts = json.loads(row["parts"])
        names = {r["sha256"]: r["filename"] for r in db.q(
            f"SELECT sha256, filename FROM documents WHERE sha256 IN ({','.join('?' * len(parts))})", parts)}
        checks = [db.one("SELECT p FROM merge_checks WHERE a = ? AND b = ?", (a, b)) for a, b in zip(parts, parts[1:])]
        ps = [c["p"] for c in checks if c]
        listed = _cut(", ".join(names.get(s, s[:8]) for s in parts), 90)
        add("Merge", "ok", f"merged from {len(parts)} attachments: {listed}",
            f"each part continues the one before ({_pct(min(ps))}–{_pct(max(ps))})" if ps else "")
    elif row["part_of"]:
        whole = db.one("SELECT filename FROM documents WHERE sha256 = ?", (row["part_of"],))
        add("Merge", "ok", f"part of {whole['filename'] if whole else '?'} ({row['part_of'][:12]}), which is "
                           "imported instead")
    else:
        checks = db.q("SELECT a, b, p FROM merge_checks WHERE a = ? OR b = ?", (sha, sha))
        if checks:
            other = {c["b"] if c["a"] == sha else c["a"]: c["p"] for c in checks}
            names = {r["sha256"]: r["filename"] for r in db.q(
                f"SELECT sha256, filename FROM documents WHERE sha256 IN ({','.join('?' * len(other))})", list(other))}
            add("Merge", "off", "kept separate: " + ", ".join(f"{names.get(s, s[:8])} ({_pct(p)})"
                                                               for s, p in other.items()))
    if row["skip_reason"] or row["part_of"]:
        return out

    d = decision(row, cfg)
    score_model = stage_model(cfg, "score")
    if row["noul1"] is not None:
        add("Relevance", "ok", f"{_pct(relevance(row['noul1'], row['types1'], cfg))} · worth keeping "
                               f"{_pct(row['noul1'])} · {_categories(row['types1'])}",
            Text(f"{row['model1']}, {row['secs1'] or 0:.1f} s", "dim"))
    elif row["manual"] is None and method is not None:
        add("Relevance", *waiting("score", score_model))
    if row["noul2"] is not None:
        add("Escalation", "ok", f"{_pct(relevance(row['noul2'], row['types2'], cfg))} · worth keeping "
                                f"{_pct(row['noul2'])} · {_categories(row['types2'])}",
            Text(f"{row['model2']}, {row['secs2'] or 0:.1f} s", "dim"))
    elif d == ESCALATE:
        add("Escalation", *waiting("escalate", cfg.scoring.escalation_model))

    s = cfg.scoring
    if row["manual"] == "yes":
        add("Decision", "ok", "import – your decision")
    elif row["manual"] == "no":
        add("Decision", "no", "not imported – your decision")
    elif d == RELEVANT:
        add("Decision", "ok", f"import (relevance ≥ {s.accept:.0%})")
    elif d == REVIEW:
        add("Decision", "ask", f"uncertain (between {s.reject:.0%} and {s.accept:.0%}) – you decide")
    elif d == IRRELEVANT:
        add("Decision", "no", f"not imported (relevance ≤ {s.reject:.0%})")

    if d in UPLOAD_DECISIONS:
        if item.has_meta:
            m = json.loads(row["meta"])
            notes = []
            if m.get("correspondent_raw") and m["correspondent_raw"] != m.get("correspondent"):
                notes.append(f"correspondent as written: {m['correspondent_raw']}")
            if m.get("date_source") == "reference":
                notes.append("no plausible date in the document – the mail date is used")
            corrected = [k for k in overrides(row) if k in ("title", "correspondent", "document_type", "date", "tags",
                                                            "fields", "summary")]
            if corrected:
                notes.append("your corrections: " + ", ".join(corrected))
            add("Metadata", "ok", Text(f"{row['model3']}, {row['secs3'] or 0:.0f} s", "dim"), *notes)
        elif overrides(row):
            add("Metadata", "ok", "your values")
        else:
            add("Metadata", *waiting("enrich", cfg.enrich.model))

    status = row["upload_status"]
    if item.saving:
        add("Upload", "run", "uploading…")
    elif status == "pending":
        add("Upload", "run", "Paperless is consuming it")
    elif status == "done":
        add("Upload", "ok", f"Paperless #{row['paperless_id']} · {(row['uploaded_at'] or '')[:16].replace('T', ' ')}",
            "summary note and custom fields set" if row["note_added"] else "summary note and custom fields follow")
    elif status in ("exists", "duplicate"):
        add("Upload", "ok", f"already in Paperless as #{row['paperless_id']}"
            + (" (same file)" if status == "exists" else f": {row['upload_error'] or 'duplicate'}"))
    elif status == "similar":
        add("Upload", "no", f"not uploaded: {row['upload_error']}", "a checks again (e.g. after deleting the match)")
    elif status == "failed":
        add("Upload", "no", f"failed: {row['upload_error']}", "a tries again")
    elif d in UPLOAD_DECISIONS or row["manual"] == "yes":
        add("Upload", "wait", "when you accept")
    return out


# ---------------------------------------------------------------- views

class ImportView(View):
    """Actions both import screens share; called by ListScreen/DocScreen.check_action after their own checks."""

    @property
    def ia(self) -> "ImportApp":
        return self.app  # type: ignore[return-value]

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        item = self.selected()
        if action in ("accept", "reject", "suggest", "undo", "edit", "browser", "paperless", "open") and item is None:
            return None
        if action in ("accept", "edit") and not item.editable:
            return None
        if action == "accept" and item.state == "running":  # its metadata may be about to change
            return None
        if action == "reject" and (not item.editable or item.state == "rejected"):
            return None
        if action == "suggest" and (not item.editable
                                    or item.state in ("running", "irrelevant", "rejected", "similar")):
            return None
        if action == "undo" and not item.edited:
            return None
        if action == "paperless" and not item.row["paperless_id"]:
            return None
        if action == "accept_all" and not self.ia.acceptable():
            return None
        return True

    def action_paperless(self) -> None:
        item = self.selected()
        if item is not None and item.row["paperless_id"]:
            webbrowser.open(f"{self.ia.cfg.paperless.url}/documents/{item.row['paperless_id']}/details")


HINTS = [("a", "Accept", "accept"), ("x", "Reject", "reject"), ("g", "Ask model", "suggest"),
         ("u", "Undo edits", "undo"), ("o", "Open file", "browser"), ("b", "Open in Paperless", "paperless")]


class ImportListScreen(ListScreen, ImportView):
    BINDINGS = [Binding("b", "paperless", "Open in Paperless")]
    HINTS = [("enter", "Open", "open"), *HINTS, ("/", "Filter", None), ("A", "Accept all", "accept_all"),
             ("p", "Pause model", "pause"), ("f5", "Reload", None), (":", "Commands", None), ("?", "Help", None),
             ("q", "Quit", None)]
    COLUMNS = COLUMNS
    PLACEHOLDER = "filter: words, a status such as uncertain or ready – Enter keeps it, Esc clears"

    def key(self, row_key: str) -> str:
        return row_key

    def doc_screen(self, item_id, focus: str | None = None) -> "ImportDocScreen":
        return ImportDocScreen(item_id, focus=focus)

    def _widths(self) -> dict[str, int]:
        fixed = {"state": 10, "rel": 4, "document_type": 12, "date": 10}
        free = max(60, self.size.width - sum(fixed.values()) - 2 * len(COLUMNS) - 4)
        file, title, corr = max(14, free * 20 // 100), max(20, free * 34 // 100), max(14, free * 22 // 100)
        return {**fixed, "filename": file, "title": title, "correspondent": corr,
                "tags": max(10, free - file - title - corr)}

    def _matches(self, item: ImportItem) -> bool:
        if not self.filter:
            return True
        d, m = item.draft, item.mail or {}
        hay = " ".join([item.label, DECISION_WORDS.get(decision(item.row, self.ia.cfg), ""), item.id[:12],
                        item.filename, d.title, d.correspondent, d.document_type, d.date, join_tags(d.tags),
                        m.get("subject") or "", sender(m)]).casefold()
        return all(word in hay for word in self.filter.casefold().split())

    def _cells(self, item: ImportItem) -> list[Text]:
        cfg, d, m = self.ia.cfg, item.draft, item.mail or {}
        quiet = not item.editable or item.state in ("irrelevant", "rejected")
        corrected = self.ia.corrected(item)

        def cell(key: str, value: str, fallback: str = "") -> Text:
            style = "dim" if quiet or (not value and fallback) else "cyan" if key in corrected else ""
            return Text(_cut(value or fallback, self.widths[key]), style)

        p = probability(item.row, cfg)
        model = decision(item.row, cfg, use_manual=False)
        rel_style = {RELEVANT: "green", REVIEW: "yellow", ESCALATE: "yellow", IRRELEVANT: "red"}.get(model, "dim")
        return [Text(item.label, STYLES.get(item.label, "")), Text(_pct(p) if p is not None else "", rel_style),
                cell("filename", item.filename), cell("title", d.title, m.get("subject") or ""),
                cell("correspondent", d.correspondent, sender(m)), cell("document_type", d.document_type),
                cell("date", d.date, (m.get("date") or "")[:10]), cell("tags", join_tags(d.tags))]

    def _title(self) -> None:
        ia = self.ia
        count = f"{len(self.rows_shown)}/{len(ia.items)}" if self.filter else str(len(ia.items))
        self.query_one("#box").border_title = (f" Imports({ia.engine.source})[{count}]"
                                               + (f" </{self.filter}>" if self.filter else "") + " ")
        self.query_one(Crumbs).show(["imports"], "" if ia.items else "no attachments found yet")

    def update_view(self, item=None, rebuild: bool = False) -> None:
        if rebuild:
            self.rebuild()
        elif item is not None:
            self.update_row(item)
        self._title()
        self.refresh_top()

    def refresh_items(self, changed: list[ImportItem], added: list[ImportItem]) -> None:
        """Rows changed or were added at the end (documents are listed in the order they were found)."""
        for item in changed:
            self.update_row(item)
        for item in added:
            if self._matches(item):
                self.table.add_row(*self._cells(item), key=item.id)
                self.rows_shown.append(item)
        if self.cursor_id is None and self.rows_shown:
            self.cursor_id = self.rows_shown[0].id
        self._title()
        self.refresh_top()


class ImportDocScreen(DocScreen, ImportView):
    BINDINGS = [Binding("b", "paperless", "Open in Paperless")]
    HINTS = [("e", "Edit", "edit"), *HINTS, ("n", "Next", "step(1)"), ("N", "Previous", "step(-1)"),
             ("t", "Steps and text only", None), ("p", "Pause model", "pause"), ("?", "Help", None),
             ("esc", "Back", None)]
    TEXT_TITLE = " Steps and text "

    def __init__(self, doc_id: str, focus: str | None = None):
        super().__init__(doc_id, focus)
        self.text = ""  # of the document shown
        self.trail: list[Step] = []  # its processing steps

    def update_view(self, item=None, rebuild: bool = False) -> None:
        """Show the document again if it changed (rebuild: it did); otherwise only the header."""
        if rebuild or (item is not None and item.id == self.doc_id):
            self.show(self.ia.by_id[self.doc_id])
        else:
            self.refresh_top()

    def form_title(self, item: ImportItem) -> str:
        return f" {_cut(item.filename, 60)} "

    def crumbs(self, item: ImportItem) -> list[str]:
        return ["imports", _cut(item.filename, 40)]

    def show_text(self, item: ImportItem, same: bool) -> None:
        if not same:
            row = self.ia.db.one("SELECT text FROM documents WHERE sha256 = ?", (item.id,))
            self.text = (row["text"] if row else "") or ""
        engine = self.ia.engine
        self.trail = steps(self.ia.db, self.ia.cfg, item, self.text,
                           engine.running_phase if engine.running == item.id else None)
        self.query_one("#text", Static).update(Group(
            render_steps(self.trail), Text("\n" + "─" * 20 + " Text " + "─" * 20, "dim"),
            Text(self.text.strip() or "(no readable text)")))

    def _heading(self, item: ImportItem) -> Text:
        row, m = item.row, item.mail
        kind = KIND_LABEL.get(row["kind"], row["kind"])
        facts = [f"mail {(m['date'] or '')[:10]} from {sender(m)}" if m else "no mail",
                 f"{kind}, {row['pages']} p." if row["pages"] else kind]
        t = Text(" · ".join(facts), "dim")
        paperless = f"#{row['paperless_id']}"
        lines = {
            "new": ("Waiting for its text.", "dim"),
            "queued": ("Waiting for the model – g moves it to the front.", "dim"),
            "running": ("The model is working on this document…", "yellow"),
            "ready": ("To be imported. Check the values, then a uploads them or e edits.", "green"),
            "uncertain": ("The model is not sure this is worth keeping: a uploads it, x rejects it.", "yellow"),
            "edited": ("Edited – a or Ctrl+S uploads your values, u undoes your edits.", "cyan"),
            "irrelevant": ("Not imported: the model finds it not worth keeping. a imports it anyway.", "dim"),
            "rejected": ("Not imported (your decision). a imports it after all.", "dim"),
            "failed": (("Upload failed – a tries again." if row["upload_status"] == "failed"
                        else "The model failed – g tries again."), "red"),
            "saving": ("Uploading to Paperless…", "yellow"),
            "done": (f"In Paperless as {paperless}.", "green"),
            "exists": (f"Already in Paperless as {paperless}.", "green"),
            "similar": (f"Not uploaded: {row['upload_error']}. a checks again.", "yellow"),
            "skipped": (f"Not imported: {row['skip_reason']}.", "dim"),
            "part": ("Part of a merged document, which is imported instead.", "dim"),
        }
        message, style = lines.get(item.label, ("", ""))
        t.append("\n" + message, style)
        if item.error:
            t.append(f"\n{item.error}", "red")
        return t

    def _decorate(self, item: ImportItem) -> None:
        """Highlight values that differ from the model's; show the model's value (or the problem) on the right."""
        _, errors = import_corrections(item.row, item.draft, self.ia.cfg, with_meta=item.has_meta)
        model = self.ia.model_draft(item)
        for key, row in self.row_of.items():
            differs = model is not None and get_value(item.draft, key) != get_value(model, key)
            row.set_class(differs, "changed")
            row.set_class(key in errors, "invalid")
            was = Text()
            if key in errors:
                was.append(errors[key], "red")
            elif differs and key == "tags":
                mine = {t.casefold() for t in item.draft.tags}
                theirs = {t.casefold() for t in model.tags}
                for tag in item.draft.tags:
                    if tag.casefold() not in theirs:
                        was.append(f"+{tag} ", "green")
                for tag in model.tags:
                    if tag.casefold() not in mine:
                        was.append(f"−{tag} ", "red")
            elif differs:
                value = get_value(model, key)
                was.append(f"model: {_cut(value, 36)}" if value else "model: empty")
            row.query_one(".was", Static).update(was)


IMPORT_HELP = """\
paperclerk import – follow the import of mail attachments and decide what is uploaded

The list holds every file found in the mail: identical files once, and a document merged from
several attachments as a row of its own. In the background paperclerk reads the mail, extracts
the text, merges split documents and asks the models whether each document is worth keeping and
for its metadata – the document you are looking at first. Nothing is uploaded until you accept
it. Enter shows a document's form next to its processing steps and its text.

List                                      Document
  Enter       open the document             e, Tab      edit the form
  j k ↑ ↓     move                          ↑ ↓ PgUp    scroll steps and text
  /           filter (words, a status       n  N        next / previous document
              such as uncertain)            t           steps and text only
  Esc         clear the filter              Esc, q      back to the list
  q           quit

Both                                      While editing
  a       accept: upload the values         Ctrl+S      accept
          shown. No metadata yet: import    Tab Enter   next field (Shift+Tab: previous)
          it, the model is asked first      →           take the grey completion
  x       reject: do not import it          Esc         stop editing
  g       ask the model (again) now         Tags are separated by commas. Dates may be
  u       undo your edits                   typed as 28.04.2023, amounts as 44,15 EUR.
  o       open the file
  b       open it in Paperless
  A       accept all ready documents (uncertain ones are left for you)
  p       pause / resume the model
  F5      read new mail, reload
  :       commands (filters, how the relevance thresholds do against your decisions)
  ?       this help

Status
  new         waiting for its text             irrelevant  not worth keeping, says the model
  queued      waiting for the model            rejected    not to be imported, say you
  running     the model works on it            failed      g (model) or a (upload) tries again
  ready       to check and accept              done        uploaded to Paperless
  uncertain   the model is not sure: decide    exists      already in Paperless (same file)
  edited      changed by you, not uploaded     similar     Paperless has the same content
  skipped     unreadable or too large          part        merged into another document

Your decisions are stored at once: quit at any time, the same command continues where it stopped."""


# ---------------------------------------------------------------- the app

class ImportApp(ReviewApp):
    """`paperclerk import` in a terminal."""

    CSS = f"Top #info {{ width: {INFO_WIDTH}; }}"

    def __init__(self, cfg: Config, source: str, paths: list[Path], limit: int | None = None):
        super().__init__(cfg)
        self.engine = ImportEngine(cfg, self.post_message, source, paths, limit)
        self.source_label = ", ".join(p.name for p in paths) if source == "mbox" else f"IMAP {cfg.imap.user}"
        self.db: DB | None = None
        self.columns = ""
        self.max_rid = 0
        self.dirty: set[str] = set()
        self.dirty_new = self.dirty_all = False

    def make_list_screen(self) -> ListScreen:
        return ImportListScreen()

    def prepare(self) -> None:
        self.db = DB(self.cfg.db_path)  # this thread only reads; the threads write
        self.columns = ", ".join(r[1] for r in self.db.q("PRAGMA table_info(documents)") if r[1] != "text")
        self.names = names_for(self.db, self.cfg)
        self.tag_suggester.tags = list(self.names["tags"])
        self._upsert(self._fetch())
        self.loading = False
        self.set_interval(0.25, self._flush)

    def cleanup(self) -> None:
        if self.db is not None:
            self.db.close()

    def help_text(self) -> str:
        return IMPORT_HELP

    def unsaved(self) -> list[ImportItem]:
        return [i for i in self.items if i.edited and i.state != "rejected"]

    def browse(self, item: ImportItem) -> None:
        webbrowser.open((self.cfg.files_dir / item.row["path"]).as_uri())

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        for status in ("ready", "uncertain", "edited", "irrelevant", "rejected", "failed", "done"):
            yield SystemCommand(f"Show {status} documents", f"Filter the list by the status {status}",
                                lambda s=status: self._show_only(s))
        yield SystemCommand("Show all documents", "Clear the filter", lambda: self._show_only(""))
        yield SystemCommand("Accept all", "Upload every ready document (uncertain ones are left)", self.accept_all)
        yield SystemCommand("Pause or resume the model", "Stop asking the model after the current document",
                            self.toggle_pause)
        yield SystemCommand("Reload", "Read new mail, the vocabulary from Paperless and the database", self.reload)
        yield SystemCommand("Check the relevance thresholds", "How the thresholds do against your decisions",
                            self.show_thresholds)

    # -------------------------------------------------------- documents from the database

    def _fetch(self, where: str = "", params: Iterable = ()) -> list[tuple[dict, dict | None]]:
        rows = [dict(r) for r in self.db.q(f"SELECT rowid AS rid, {self.columns} FROM documents {where} "
                                           "ORDER BY rowid", tuple(params))]
        mails = first_mails(self.db, [r["sha256"] for r in rows] if where else None)
        return [(r, mails.get(r["sha256"])) for r in rows]

    def _upsert(self, pairs: list[tuple[dict, dict | None]]) -> tuple[list[ImportItem], list[ImportItem]]:
        changed, added = [], []
        for row, mail in pairs:
            item = self.by_id.get(row["sha256"])
            base = imported(row, self.cfg, with_meta=has_metadata(row, self.cfg))
            if item is None:
                item = ImportItem(row, mail, base, base.copy())
                self.items.append(item)
                self.by_id[item.id] = item
                added.append(item)
            else:
                old = item.base
                item.row, item.mail, item.base = row, mail or item.mail, base
                item.draft = merge(old, base, item.draft)  # keep what the user has typed
                changed.append(item)
            self.max_rid = max(self.max_rid, row["rid"])
            self._refresh(item)
        return changed, added

    def _refresh(self, item: ImportItem) -> None:
        item.has_meta = has_metadata(item.row, self.cfg)
        item.state = import_state(item.row, self.cfg, self.engine.running == item.id, item.saving, item.asked)

    def _flush(self) -> None:
        """Apply the database changes the threads reported (batched: reading can add hundreds per second)."""
        if not (self.dirty or self.dirty_new or self.dirty_all):
            return
        if self.dirty_all:
            pairs = self._fetch()
        else:
            shas = sorted(self.dirty)
            pairs = [p for i in range(0, len(shas), 500)
                     for p in self._fetch(f"WHERE sha256 IN ({','.join('?' * len(shas[i:i + 500]))})", shas[i:i + 500])]
            if self.dirty_new:
                pairs += self._fetch("WHERE rowid > ?", (self.max_rid,))
        rebuild = self.dirty_all
        self.dirty.clear()
        self.dirty_new = self.dirty_all = False
        changed, added = self._upsert(pairs)
        for screen in self.screen_stack:
            if isinstance(screen, ImportListScreen) and screen.is_mounted:
                if rebuild:
                    screen.update_view(rebuild=True)
                else:
                    screen.refresh_items(changed, added)
            elif isinstance(screen, ImportDocScreen) and screen.is_mounted:
                screen.update_view(rebuild=rebuild or screen.doc_id in {i.id for i in changed})

    def corrected(self, item: ImportItem) -> set[str]:
        """Form keys where the form differs from the model's metadata."""
        model = self.model_draft(item)
        if model is None:
            return set()
        return {key for key, _ in self.rows if get_value(item.draft, key) != get_value(model, key)}

    def model_draft(self, item: ImportItem) -> Draft | None:
        return imported(item.row, self.cfg, with_corrections=False) if item.has_meta else None

    # -------------------------------------------------------- header

    def info(self) -> Text:
        e, jobs = self.engine, self.engine.jobs
        lines: list[tuple[str, Text]] = [("Mailbox:", Text(f"{_cut(self.source_label, 24)} · {e.reading or '…'}"))]
        counts = Counter(i.label for i in self.items)
        docs = Text(f"{len(self.items):,}")
        for label in ("ready", "uncertain", "edited", "failed", "done", "irrelevant", "rejected"):
            if counts[label]:
                docs.append(f" · {counts[label]:,} {label}", STYLES[label])
        lines.append(("Documents:", docs))
        if e.phase is None:
            step = Text("all done – waiting for your decisions" if not jobs.paused else "paused", "dim")
        elif e.phase in ("read", "merge"):
            step = Text(PHASES[e.phase] + ("…" if e.phase == "merge" else ""))
        else:
            step = Text(f"{PHASES[e.phase]} {e.phase_done:,}/{e.phase_total:,}")
            typical = statistics.median(e.durations[e.phase]) if e.durations[e.phase] else e.typical.get(e.phase)
            left = e.phase_total - e.phase_done
            if typical and left:
                step.append(f" · ~{fmt_duration(typical)}/doc · ~{fmt_duration(typical * left)} left", "dim")
        if jobs.paused and not self.model_error:
            step.append(" · paused, p resumes", "yellow")
        lines.append(("Step:", step))
        model = Text(e.model or "–")
        running = self.by_id.get(e.running) if e.running else None
        if self.model_error:
            model.append(" · " + self.model_error, "red")
        elif running is not None:
            elapsed = time.monotonic() - e.started_at
            model.append(f" · {_cut(running.filename, 24)} for {fmt_duration(elapsed)}", "yellow")
            if e.running_phase != "text" and not e.durations[e.running_phase or ""] and elapsed > 15:
                model.append(" (loading)", "dim")  # the first call loads the model: 1–2 minutes for 27B
        elif jobs.paused:
            model.append(" · paused", "yellow")
        else:
            model.append(" · idle", "dim")
        lines.append(("Model:", model))
        upload = Text(f"{self.pending} on the way" if self.pending else "–", "yellow" if self.pending else "dim")
        if e.waiting:
            upload.append(f" · {e.waiting} in Paperless' queue", "dim")
        if self.load_error:
            upload.append(f" · Paperless: {self.load_error}", "red")
        lines.append(("Upload:", upload))
        out = Text(no_wrap=True, overflow="ellipsis")
        for n, (label, value) in enumerate(lines):
            out.append(f"{label:<11}", LABEL_STYLE)
            value.truncate(INFO_WIDTH - 11, overflow="ellipsis")  # one line each, or the last ones drop out
            out.append_text(value)
            if n < len(lines) - 1:
                out.append("\n")
        return out

    # -------------------------------------------------------- operations

    def acceptable(self) -> list[ImportItem]:
        """Accept all: every document the model finds worth keeping (or you said to import) with metadata."""
        return [i for i in self.items if i.state == "ready"]

    def accept(self, item: ImportItem) -> str | None:
        """Upload the form, or mark the document for import and ask the model first. Returns None (uploading),
        "asked", "nothing" or the key of an invalid field."""
        if not item.editable:
            return "finished"
        if item.state == "running":
            self.notify("The model is working on it – check it when it is ready.")
            return "nothing"
        if not item.has_meta and not item.edited:
            item.row["manual"], item.asked, item.error = "yes", True, ""
            self._refresh(item)
            self.engine.decide(item.id, "yes", ask=True)
            self.notify(f"{item.filename} will be imported. The model determines its metadata first – check it, "
                        "then a uploads it." + (" The model is paused, p resumes." if self.engine.jobs.paused else ""),
                        markup=False)
            self.changed(item)
            return "asked"
        corrections, errors = import_corrections(item.row, item.draft, self.cfg, with_meta=item.has_meta)
        if errors:
            key = next(k for k, _ in self.rows if k in errors)
            self.notify(f"{item.filename} – {self.labels[key]}: {errors[key]}", title="Not accepted",
                        severity="warning", markup=False)
            self.changed(item)
            return key
        self._upload(item, corrections)
        return None

    def _upload(self, item: ImportItem, corrections: dict) -> None:
        item.saving, item.error = True, ""
        self._refresh(item)
        self.pending += 1
        self.engine.upload(item.id, corrections)
        self.changed(item)

    def reject(self, item: ImportItem) -> None:
        if not item.editable or item.state == "rejected":
            return
        item.row["manual"], item.error = "no", ""
        self._refresh(item)
        self.engine.decide(item.id, "no")
        self.notify(f"{item.filename} will not be imported – a imports it after all.", markup=False)
        self.changed(item)

    def suggest(self, item: ImportItem) -> None:
        if not item.editable or item.state == "running":
            return
        item.asked, item.error = True, ""
        self._refresh(item)
        self.engine.ask(item.id, force=item.has_meta)
        self.notify("Asked the model – this document is next."
                    + (" The model is paused, p resumes." if self.engine.jobs.paused else ""))
        self.changed(item)

    def accept_all(self) -> None:
        ready = self.acceptable()
        uncertain = sum(1 for i in self.items if i.state == "uncertain")
        left = (f" {uncertain} uncertain ones are left for you to check (filter: uncertain)." if uncertain else "")
        if not ready:
            self.notify("No document is ready to upload." + left)
            return
        edited = sum(1 for i in ready if i.edited)
        question = (f"Upload {len(ready)} documents to Paperless" + (f", {edited} of them with your edits" if edited
                                                                       else "") + "?" + left)
        self.push_screen(Confirm(question), lambda yes: self._accept_many(ready) if yes else None)

    def _accept_many(self, items: list[ImportItem]) -> None:
        written = skipped = 0
        for item in items:
            if item.state != "ready":
                continue
            corrections, errors = import_corrections(item.row, item.draft, self.cfg, with_meta=True)
            if errors:
                skipped += 1
                continue
            self._upload(item, corrections)
            written += 1
        self.notify(f"Uploading {written} documents." + (f" {skipped} skipped because of invalid values."
                                                         if skipped else ""), markup=False)

    def reload(self) -> None:
        self.engine.reload()
        self.dirty_all = True
        self.notify("Reading new mail and reloading…")

    def show_thresholds(self) -> None:
        report = evaluate(self.db, self.cfg)
        self.push_screen(Help(report or "No decisions of yours yet: accept (a) or reject (x) documents first, "
                                        "then this shows how the relevance thresholds do against them."))

    # -------------------------------------------------------- messages from the background threads

    def on_rows(self, msg: Rows) -> None:
        if msg.shas is None:
            self.dirty_all = True
        else:
            self.dirty.update(msg.shas)
        self.dirty_new |= msg.new

    def on_ask_done(self, msg: AskDone) -> None:
        item = self.by_id.get(msg.sha)
        if item is not None:
            item.asked = False
        self.dirty.add(msg.sha)

    def on_vocab(self, msg: Vocab) -> None:
        self.names, self.load_error = msg.names, msg.error
        known = {t.casefold() for t in self.tag_suggester.tags}
        self.tag_suggester.tags += [t for t in msg.names["tags"] if t.casefold() not in known]
        if msg.error:
            self.notify(f"{msg.error}\nUploads try again; F5 reloads.", title="Paperless", severity="error",
                        markup=False, timeout=10)
        self.changed()

    def on_pipeline_failed(self, msg: PipelineFailed) -> None:
        if msg.fatal:
            self.model_error = msg.error
            self.notify(f"{msg.error}\nThe model is paused – p tries again.", title="Model unavailable",
                        severity="error", markup=False, timeout=15)
        else:
            self.notify(msg.error, severity="error", markup=False, timeout=10)
        self.changed()

    def on_uploaded(self, msg: Uploaded) -> None:
        item = self.by_id.get(msg.sha)
        if msg.first:
            self.pending -= 1
            if item is not None:
                item.saving = False
        if item is not None:
            failed = msg.status in (None, "failed")
            item.error = msg.message if failed else ""
            if failed:
                self.notify(msg.message, title=f"{item.filename} not uploaded", severity="error", markup=False,
                            timeout=15)
            elif msg.status == "similar":
                self.notify(msg.message, title=f"{item.filename} not uploaded", severity="warning", markup=False)
        self.dirty.add(msg.sha)
        if self.quitting and msg.first and msg.status in (None, "failed"):  # stay, so the user sees it
            self.quitting = False
            self.notify("Quitting was cancelled because a document could not be uploaded.", severity="warning")
        self._finish_quit()


def run_import_ui(cfg: Config, source: str, paths: list[Path], limit: int | None = None) -> int:
    return run_app(ImportApp(cfg, source, paths, limit))
