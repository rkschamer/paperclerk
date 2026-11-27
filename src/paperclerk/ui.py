"""`paperclerk ui`: review, edit and accept the suggestions of `paperclerk process` in the terminal.

The UI is a layer over `process`: the same selection, model call, stored suggestions and write step. It
follows k9s: a header with status and the keys of the current view, a full-width list with a filter,
Enter opens a document, Esc goes back.

Two background threads keep it responsive. One asks the model, document by document, the one you are
looking at first. The other talks to Paperless (loading, writing). Each opens its own database
connection and HTTP clients, and results reach the UI as messages.
"""

from __future__ import annotations

import ast
import logging
import queue
import statistics
import threading
import time
import webbrowser
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.suggester import SuggestFromList, Suggester
from textual.widgets import DataTable, Input, Label, Static, TextArea

from . import __version__, process
from .config import Config
from .db import DB
from .draft import Draft, changed_keys, changes_for, current, join_tags, merge, proposal, split_tags
from .enrich import vocabulary
from .normalize import domain_label
from .ollama import OllamaError, OllamaUnavailable
from .paperless import Paperless, PaperlessError, Registry, sync_cache
from .util import fmt_duration

log = logging.getLogger(__name__)

# Textual 8.2.8 crashes when the mouse is pressed on a toast that was just removed (dismissed by a first click,
# or timed out): it starts a text selection in a widget that no longer has a parent. Nobody needs to select
# the text of a toast, so toasts are not selectable; everything else still is.
try:
    from textual.widgets._toast import Toast

    Toast.ALLOW_SELECT = False
except ImportError:  # pragma: no cover – a later Textual moved it (and has hopefully fixed the crash)
    pass


# ---------------------------------------------------------------- messages from the background threads

class Loaded(Message):
    def __init__(self, docs: list[dict], records: dict[int, dict], suggestions: dict[int, dict], inbox: set[int],
                 names: dict[str, list[str]], typical: float | None):
        super().__init__()
        self.docs, self.records, self.suggestions, self.inbox = docs, records, suggestions, inbox
        self.names, self.typical = names, typical


class LoadFailed(Message):
    def __init__(self, error: str):
        super().__init__()
        self.error = error


class ModelStarted(Message):
    def __init__(self, doc_id: int):
        super().__init__()
        self.doc_id = doc_id


class Suggested(Message):
    def __init__(self, doc_id: int, suggestion: dict, secs: float):
        super().__init__()
        self.doc_id, self.suggestion, self.secs = doc_id, suggestion, secs


class SuggestFailed(Message):
    def __init__(self, doc_id: int, error: str, fatal: bool):
        super().__init__()
        self.doc_id, self.error, self.fatal = doc_id, error, fatal


class Written(Message):
    def __init__(self, doc_id: int, status: str, note_id: int | None):
        super().__init__()
        self.doc_id, self.status, self.note_id = doc_id, status, note_id


class WriteFailed(Message):
    def __init__(self, doc_id: int, error: str):
        super().__init__()
        self.doc_id, self.error = doc_id, error


class LogLine(Message):
    def __init__(self, level: int, text: str):
        super().__init__()
        self.level, self.text = level, text


# ---------------------------------------------------------------- background work

class Jobs:
    """Documents waiting for the model, in order; the document on screen (preferred) goes first."""

    def __init__(self) -> None:
        self._ids: list[int] = []
        self._cv = threading.Condition()
        self.paused = False
        self.closed = False
        self.preferred: int | None = None

    def put(self, doc_id: int, front: bool = False) -> None:
        with self._cv:
            if doc_id in self._ids:
                if not front:
                    return
                self._ids.remove(doc_id)
            self._ids.insert(0 if front else len(self._ids), doc_id)
            self._cv.notify_all()

    def discard(self, doc_id: int) -> None:
        with self._cv:
            if doc_id in self._ids:
                self._ids.remove(doc_id)

    def pause(self, paused: bool) -> None:
        with self._cv:
            self.paused = paused
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self.closed = True
            self._cv.notify_all()

    def get(self) -> int | None:
        """The next document; blocks while the queue is empty or paused. None: shut down."""
        with self._cv:
            while not self.closed and (self.paused or not self._ids):
                self._cv.wait()
            if self.closed:
                return None
            doc_id = self.preferred if self.preferred in self._ids else self._ids[0]
            self._ids.remove(doc_id)
            return doc_id

    def __len__(self) -> int:
        with self._cv:
            return len(self._ids)


def _message(e: Exception) -> str:
    return str(e) if isinstance(e, (OllamaError, PaperlessError)) else f"{type(e).__name__}: {e}"


class Engine:
    """The two background threads. `post` delivers results to the UI (App.post_message is thread-safe)."""

    def __init__(self, cfg: Config, post: Callable[[Message], object]):
        self.cfg, self.post = cfg, post
        self.jobs = Jobs()
        self.tasks: queue.Queue = queue.Queue()
        self.registry: Registry | None = None  # owned by the Paperless thread; the UI only reads names
        self.docs: dict[int, dict] = {}  # set by the UI: the documents the model thread may be asked for
        self.running: int | None = None
        self.started_at = 0.0
        self._written: dict[int, tuple] = {}

    def start(self) -> None:
        DB(self.cfg.db_path).close()  # create or migrate the file once; two threads doing it at once collide
        for target, name in ((self._paperless_loop, "paperless"), (self._model_loop, "model")):
            threading.Thread(target=self._guard, args=(target, name), name=f"paperclerk-{name}", daemon=True).start()
        self.load()

    def _guard(self, target: Callable[[], None], name: str) -> None:
        """A thread that dies must say so, or the UI would wait for it forever."""
        try:
            target()
        except Exception as e:  # noqa: BLE001
            log.exception("%s thread stopped", name)
            self.post(LoadFailed(f"internal error, see the log: {_message(e)}") if name == "paperless"
                      else SuggestFailed(self.running or 0, f"internal error, see the log: {_message(e)}", fatal=True))

    def stop(self) -> None:
        self.jobs.close()
        self.tasks.put(None)

    def load(self) -> None:
        self.tasks.put(("load",))

    def write(self, doc: dict, changes: dict, note_id: int | None, status: str) -> None:
        self.tasks.put(("write", doc, changes, note_id, status))

    def rerecord(self, doc_id: int) -> None:
        """The model finished a document that was written meanwhile: its stored status must stay 'written'."""
        self.tasks.put(("rerecord", doc_id))

    # -------------------------------------------------------- Paperless thread

    def _paperless_loop(self) -> None:
        db = DB(self.cfg.db_path)
        pl: Paperless | None = None
        try:
            while (task := self.tasks.get()) is not None:
                kind = task[0]
                try:
                    if pl is None:
                        pl = Paperless.from_config(self.cfg)
                    if kind == "load":
                        self._load(db, pl)
                    elif kind == "write":
                        self._write(db, pl, *task[1:])
                    elif kind == "rerecord" and task[1] in self._written:
                        process.record_written(db, *self._written[task[1]])
                except Exception as e:  # noqa: BLE001 – shown in the UI; the thread keeps running
                    db.rollback()  # a failed statement would leave the connection unable to write
                    if not isinstance(e, PaperlessError):
                        log.exception("%s failed", kind)
                    if kind == "load":
                        self.post(LoadFailed(_message(e)))
                    elif kind == "write":
                        self.post(WriteFailed(task[1]["id"], _message(e)))
        finally:
            if pl is not None:
                pl.close()
            db.close()

    def _load(self, db: DB, pl: Paperless) -> None:
        cfg = self.cfg
        objects = sync_cache(db, pl)
        registry = Registry(db, pl, objects)
        inbox = {t["id"] for t in objects["tags"] if t.get("is_inbox_tag")}
        if cfg.process.scope == "inbox" and not inbox:
            raise PaperlessError('Paperless has no inbox tag, so there is nothing to process. Create one or set '
                                 '[process] scope = "all".')
        docs = process.select(pl, cfg, registry, inbox, full=True)
        records, suggestions = {}, {}
        for doc in docs:
            rec = db.one("SELECT * FROM processed WHERE doc_id = ?", (doc["id"],))
            if rec is not None:
                records[doc["id"]] = dict(rec)
                if (s := process.stored_suggestion(rec, cfg, doc)) is not None:
                    suggestions[doc["id"]] = s
        vocab = vocabulary(db, cfg)
        corr = sorted(objects["correspondents"], key=lambda o: -(o.get("document_count") or 0))
        names = {"correspondent": [o["name"] for o in corr if not domain_label(o["name"])],
                 "document_type": vocab.document_types,
                 "tags": [*vocab.tags, *(o["name"] for o in objects["tags"] if o["id"] in inbox)]}
        secs = [r["secs"] for r in db.q("SELECT secs FROM processed WHERE secs IS NOT NULL AND model = ? "
                                        "ORDER BY updated_at DESC LIMIT 20", (cfg.enrich.model,))]
        self.registry = registry
        self.post(Loaded(docs, records, suggestions, inbox, names, statistics.median(secs) if secs else None))

    def _write(self, db: DB, pl: Paperless, doc: dict, changes: dict, note_id: int | None, status: str) -> None:
        try:
            new_note = process.apply(pl, self.registry, self.cfg, doc["id"], changes, note_id, overwrite=True)
        except PaperlessError as e:
            process.record_write_error(db, doc["id"], e)
            raise
        process.record_written(db, doc, changes, new_note, status)
        self._written[doc["id"]] = (doc, changes, new_note, status)
        self.post(Written(doc["id"], status, new_note))

    # -------------------------------------------------------- model thread

    def _model_loop(self) -> None:
        db = DB(self.cfg.db_path)
        pl: Paperless | None = None
        ollama = enricher = None
        try:
            while (doc_id := self.jobs.get()) is not None:
                doc = self.docs.get(doc_id)
                if doc is None:
                    continue
                self.running, self.started_at = doc_id, time.monotonic()
                self.post(ModelStarted(doc_id))
                try:
                    if pl is None:
                        pl = Paperless.from_config(self.cfg)
                    if enricher is None:
                        ollama, enricher = process.open_enricher(db, self.cfg)
                    suggestion = process.suggest(db, self.cfg, pl, enricher, doc)
                    self.post(Suggested(doc_id, suggestion, time.monotonic() - self.started_at))
                except OllamaUnavailable as e:  # pause until the user resumes; this document goes first then
                    self.jobs.pause(True)
                    self.jobs.put(doc_id, front=True)
                    if ollama is not None:
                        ollama.close()
                    ollama = enricher = None
                    self.post(SuggestFailed(doc_id, str(e), fatal=True))
                except Exception as e:  # noqa: BLE001 – recorded for this document, the next one goes on
                    db.rollback()
                    if not isinstance(e, (OllamaError, PaperlessError)):
                        log.exception("model call for #%s failed", doc_id)
                    self.post(SuggestFailed(doc_id, _message(e), fatal=False))
                finally:
                    self.running = None
        finally:
            for client in (ollama, pl):
                if client is not None:
                    client.close()
            db.close()


# ---------------------------------------------------------------- state of one document in the UI

FINISHED = ("saving", "done", "rejected")
STYLES = {"new": "", "queued": "dim", "running": "bold yellow", "ready": "bold green", "edited": "bold cyan",
          "failed": "bold red", "saving": "yellow", "done": "green", "rejected": "dim"}


@dataclass
class Item:
    doc: dict
    in_inbox: bool
    current: Draft  # as in Paperless
    base: Draft  # with the suggestion applied
    draft: Draft  # what the form shows: base plus the user's edits
    suggestion: dict | None
    state: str  # new | queued | running | ready | failed | saving | done | rejected
    note_id: int | None = None
    fail_count: int = 0
    error: str = ""
    secs: float | None = None
    tagged: bool = False  # written or rejected in this session: Paperless has the processed tag

    @property
    def id(self) -> int:
        return self.doc["id"]

    @property
    def open(self) -> bool:
        """Still to be checked: not written, not rejected."""
        return self.state not in FINISHED

    @property
    def editable(self) -> bool:
        """The form can still be changed and accepted – also after rejecting, which only added the processed tag."""
        return self.state not in ("saving", "done")

    @property
    def edited(self) -> bool:
        return self.editable and self.draft != self.base

    @property
    def label(self) -> str:
        return "edited" if self.edited and self.state not in ("running", "rejected") else self.state


def form_rows(cfg: Config) -> list[tuple[str, str]]:
    return [("title", "Title"), ("correspondent", "Correspondent"), ("document_type", "Document type"),
            ("date", "Date"), ("tags", "Tags"), *((f"field:{f.name}", f.name) for f in cfg.fields),
            ("note", "Summary")]


def get_value(d: Draft, key: str) -> str:
    if key == "tags":
        return join_tags(d.tags)
    if key.startswith("field:"):
        return d.fields.get(key[6:], "")
    return getattr(d, key)


def set_value(d: Draft, key: str, text: str) -> None:
    if key == "tags":
        d.tags = split_tags(text)
    elif key.startswith("field:"):
        d.fields[key[6:]] = text.strip()
    else:
        setattr(d, key, text.strip())


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


class TagSuggester(Suggester):
    """Completes the tag being typed after the last comma."""

    def __init__(self) -> None:
        super().__init__(use_cache=False, case_sensitive=True)
        self.tags: list[str] = []

    async def get_suggestion(self, value: str) -> str | None:
        head, _, token = value.rpartition(",")
        token = token.lstrip()
        if not token:
            return None
        used = {t.casefold() for t in split_tags(head)}
        for tag in self.tags:
            if len(tag) > len(token) and tag.casefold().startswith(token.casefold()) and tag.casefold() not in used:
                return value + tag[len(token):]
        return None


class UiLog(logging.Handler):
    """Shows paperclerk's own log messages (objects created in Paperless, warnings) as notifications."""

    def __init__(self, app: App):
        super().__init__(logging.INFO)
        self.app = app

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "shown", False):  # logged by ReviewApp.notify: it is on the screen already
            return
        if record.name.startswith("paperclerk") and record.name != "paperclerk.progress":  # the header shows that
            self.app.post_message(LogLine(record.levelno, record.getMessage()))


# ---------------------------------------------------------------- shared widgets

LOGO = """\
┌─┐┌─┐┌─┐┌─┐┬─┐┌─┐┬  ┌─┐┬─┐┬┌─
├─┘├─┤├─┘├┤ ├┬┘│  │  ├┤ ├┬┘├┴┐
┴  ┴ ┴┴  └─┘┴└─└─┘┴─┘└─┘┴└─┴ ┴"""
KEY_STYLE = "bold dodger_blue1"
LABEL_STYLE = "bold dark_orange"


class Top(Horizontal):
    """k9s-style header: status on the left, the keys of the current view in the middle, logo on the right."""

    DEFAULT_CSS = """
    Top { height: 5; padding: 0 1; }
    Top #info { width: 46; margin-right: 2; }
    Top #hints { width: 1fr; }
    Top #logo { width: 32; color: $warning; text-style: bold; }
    Top.narrow #logo { display: none; }
    """

    def compose(self) -> ComposeResult:
        yield Static(id="info")
        yield Static(id="hints")
        yield Static(Text(LOGO + f"\n{'v' + __version__:>30}"), id="logo")

    def show(self, info: Text, hints: list[tuple[str, str, bool | None]]) -> None:
        self.query_one("#info", Static).update(info)
        visible = [h for h in hints if h[2] is not False]
        rows = 5
        grid = Table.grid(padding=(0, 3))
        columns = max(1, -(-len(visible) // rows))
        for _ in range(columns):
            grid.add_column(no_wrap=True)
        for r in range(rows):
            cells = []
            for c in range(columns):
                if c * rows + r < len(visible):
                    key, label, enabled = visible[c * rows + r]
                    cells.append(Text.assemble((f"<{key}>", KEY_STYLE if enabled else "dim"), " ",
                                               (label, "" if enabled else "dim")))
                else:
                    cells.append(Text(""))
            grid.add_row(*cells)
        self.query_one("#hints", Static).update(grid)


class Crumbs(Static):
    """Where you are, like k9s' breadcrumbs: <documents> <#123>."""

    DEFAULT_CSS = "Crumbs { height: 1; padding: 0 1; }"

    def show(self, names: list[str], note: str = "") -> None:
        t = Text()
        for i, name in enumerate(names):
            t.append(f" <{name}> ", "bold black on dark_orange" if i == len(names) - 1 else "black on steel_blue1")
            t.append(" ")
        if note:
            t.append(note, "dim")
        self.update(t)


class Confirm(ModalScreen[bool]):
    DEFAULT_CSS = """
    Confirm { align: center middle; }
    Confirm > Vertical { width: 72; height: auto; padding: 1 2; border: round $warning; background: $surface; }
    Confirm .keys { margin-top: 1; }
    """
    BINDINGS = [Binding("y,enter", "answer(True)", "Yes"), Binding("n,escape", "answer(False)", "No")]

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.question, markup=False)
            yield Static(Text.assemble(("<y>", KEY_STYLE), " Yes     ", ("<n>", KEY_STYLE), " No"), classes="keys")

    def action_answer(self, yes: bool) -> None:
        self.dismiss(yes)


HELP = """\
paperclerk ui – check suggestions before they are written to Paperless

The list holds the documents `paperclerk process` would handle. In the background the model works
through the queued ones, the document you are looking at first. Values a suggestion changes are
highlighted in the list; Enter shows the whole form next to the document text. Your edits always win
over the suggestion, even where `process` would leave a value alone.

List                                      Document
  Enter       open the document             e, Tab      edit the form
  j k ↑ ↓     move                          ↑ ↓ PgUp    scroll the text
  /           filter (words, a status       n  N        next / previous document
              such as ready, a number)      t           text only
  Esc         clear the filter              Esc, q      back to the list
  q           quit

Both                                      While editing
  a       accept: write the values to       Ctrl+S      accept
          Paperless, add the processed tag  Tab Enter   next field (Shift+Tab: previous)
          {accept:<32}  →           take the grey completion
  x       reject: change nothing, only add  Esc         stop editing
          the processed tag                 Tags are separated by commas; removing one removes
  g       ask the model (again) now         it from the document. {tags1}
  u       undo your edits                   {tags2} Dates may be typed
  o       open in Paperless (browser)       as 28.04.2023, amounts as 44,15 EUR. An empty field
  A       accept all with a suggestion      clears the value.
  p       pause / resume the model
  F5      reload from Paperless           Status
  :       commands                          queued   waiting for the model   ready   to check
  ?       this help                         running  the model works on it   edited  changed by you
                                            failed   g tries again           new     no suggestion
                                            saving / done: written           rejected: tag only

A rejected document keeps its form: a or Ctrl+S still writes it, g asks the model again. To process
a written document again, remove its processed tag in Paperless and press F5."""


def help_text(cfg: Config) -> str:
    """The help, saying what accepting does with the inbox tag ([process] remove_inbox_tags)."""
    if cfg.process.remove_inbox_tags:
        return HELP.format(accept="and remove the inbox tag", tags1="Type the inbox tag back in to",
                           tags2="keep the document in the inbox.")
    return HELP.format(accept="and keep the inbox tag", tags1="Remove the inbox tag to take",
                       tags2="the document out of the inbox.")


class Help(ModalScreen[None]):
    DEFAULT_CSS = """
    Help { align: center middle; }
    Help > VerticalScroll { width: 106; max-width: 100%; height: auto; max-height: 90%; padding: 1 2;
                            border: round $accent; background: $surface; }
    """
    BINDINGS = [Binding("escape,q,question_mark,f1", "close", "Close")]

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(self.text, markup=False)

    def action_close(self) -> None:
        self.dismiss()


# ---------------------------------------------------------------- views

class View(Screen):
    """A screen with the header and breadcrumbs; the app calls update_view when its state changes."""

    HINTS: list[tuple[str, str, str | None]] = []

    @property
    def pc(self) -> "ReviewApp":
        return self.app  # type: ignore[return-value]

    def hints(self) -> list[tuple[str, str, bool | None]]:
        result = []
        for key, label, action in self.HINTS:
            if action == "pause" and self.pc.engine.jobs.paused:
                label = "Resume model"
            name, _, args = (action or "").partition("(")
            params = tuple(ast.literal_eval(f"({args.rstrip(')')},)")) if args.strip("()") else ()
            result.append((key, label, True if action is None else self.check_action(name, params)))
        return result

    def refresh_top(self) -> None:
        if self.is_mounted:
            self.query_one(Top).show(self.pc.info(), self.hints())

    def update_view(self, item: Item | None = None, rebuild: bool = False) -> None:
        self.refresh_top()

    def on_resize(self, event: events.Resize) -> None:
        self.query_one(Top).set_class(event.size.width < 165, "narrow")  # the logo only when there is room

    def on_descendant_focus(self) -> None:
        self.refresh_top()

    def on_screen_resume(self) -> None:
        self.refresh_top()

    def typing(self) -> bool:
        return isinstance(self.focused, (Input, TextArea))

    # actions shared by both views; the item is the selected or shown document
    def selected(self) -> Item | None:
        raise NotImplementedError

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        item = self.selected()
        if action in ("accept", "reject", "suggest", "undo", "edit"):
            if item is None or not item.editable:
                return None
            if action == "reject" and not item.open:
                return None
            if action == "undo" and not item.edited:
                return None
            if action == "suggest" and item.state == "running":
                return None
        if action in ("browser", "open") and item is None:  # written documents can still be looked at
            return None
        if action == "accept_all" and not self.pc.acceptable():
            return None
        return True

    def action_suggest(self) -> None:
        if item := self.selected():
            self.pc.suggest(item)

    def action_undo(self) -> None:
        if item := self.selected():
            self.pc.undo(item)

    def action_browser(self) -> None:
        if item := self.selected():
            self.pc.browse(item)

    def action_pause(self) -> None:
        self.pc.toggle_pause()

    def action_accept_all(self) -> None:
        self.pc.accept_all()

    def action_reload(self) -> None:
        self.pc.reload()

    def action_help(self) -> None:
        self.app.push_screen(Help(self.pc.help_text()))

    def action_palette(self) -> None:
        self.app.action_command_palette()


LIST_COLUMNS = (("state", "STATUS"), ("id", "#"), ("title", "TITLE"), ("correspondent", "CORRESPONDENT"),
                ("document_type", "TYPE"), ("date", "DATE"), ("tags", "TAGS"))


class ListScreen(View):
    DEFAULT_CSS = """
    ListScreen #box { height: 1fr; border: round $primary; border-title-align: center;
                      border-title-color: $accent; border-title-style: bold; }
    ListScreen #filter { display: none; margin: 0 1; }
    ListScreen #filter.active { display: block; }
    ListScreen DataTable { height: 1fr; }
    ListScreen DataTable > .datatable--header { text-style: bold; color: $text; }
    """
    BINDINGS = [
        Binding("a", "accept", "Accept"),
        Binding("x", "reject", "Reject"),
        Binding("g", "suggest", "Ask model"),
        Binding("u", "undo", "Undo edits"),
        Binding("o", "browser", "Open in Paperless"),
        Binding("slash", "filter", "Filter"),
        Binding("A", "accept_all", "Accept all"),
        Binding("p", "pause", "Pause model"),
        Binding("f5", "reload", "Reload"),
        Binding("colon", "palette", "Commands"),
        Binding("question_mark,f1", "help", "Help"),
        Binding("q", "quit", "Quit"),
        Binding("escape", "clear_filter", "Clear filter"),
        Binding("j", "move(1)", "Down"),
        Binding("k", "move(-1)", "Up"),
    ]
    HINTS = [("enter", "Open", "open"), ("a", "Accept", "accept"), ("x", "Reject", "reject"), ("g", "Ask model", "suggest"), ("u", "Undo edits", "undo"),
             ("o", "Open in Paperless", "browser"), ("/", "Filter", None), ("A", "Accept all", "accept_all"),
             ("p", "Pause model", "pause"), ("f5", "Reload", None), (":", "Commands", None), ("?", "Help", None),
             ("q", "Quit", None)]
    COLUMNS = LIST_COLUMNS
    PLACEHOLDER = "filter: words, a status such as ready, or a number – Enter keeps it, Esc clears"

    def __init__(self) -> None:
        super().__init__()
        self.filter = ""
        self.rows_shown: list[Item] = []
        self.cursor_id: int | None = None
        self.widths: dict[str, int] = {}

    def compose(self) -> ComposeResult:
        yield Top()
        with Vertical(id="box"):
            yield Input(placeholder=self.PLACEHOLDER, id="filter", compact=True)
            yield DataTable(id="docs", cursor_type="row")
        yield Crumbs()

    def key(self, row_key: str) -> int:
        """The item id behind a table row key."""
        return int(row_key)

    def doc_screen(self, item_id, focus: str | None = None) -> "DocScreen":
        return DocScreen(item_id, focus=focus)

    def on_mount(self) -> None:
        self.rebuild()
        self.table.focus()

    @property
    def table(self) -> DataTable:
        return self.query_one("#docs", DataTable)

    def selected(self) -> Item | None:
        return self.pc.by_id.get(self.cursor_id) if self.cursor_id is not None else None

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "clear_filter":
            return bool(self.filter) or self.query_one("#filter").has_class("active")
        if self.typing() and action != "clear_filter":
            return False
        return super().check_action(action, parameters)

    # -------------------------------------------------------- table

    def _widths(self) -> dict[str, int]:
        free = max(60, self.size.width - 8 - 5 - 14 - 10 - 2 * len(LIST_COLUMNS) - 4)
        title = max(24, free * 45 // 100)
        corr = max(14, free * 25 // 100)
        return {"state": 8, "id": 5, "title": title, "correspondent": corr, "document_type": 14, "date": 10,
                "tags": max(12, free - title - corr)}

    def _matches(self, item: Item) -> bool:
        if not self.filter:
            return True
        d = item.draft
        hay = " ".join([item.label, str(item.id), d.title, d.correspondent, d.document_type, d.date,
                        join_tags(d.tags), item.doc.get("title") or ""]).casefold()
        return all(word in hay for word in self.filter.casefold().split())

    def _cells(self, item: Item) -> list[Text]:
        changes, _ = changes_for(item.current, item.draft, self.pc.cfg)
        changed = changed_keys(changes)
        cells = [Text(item.label, STYLES.get(item.label, "")), Text(str(item.id), "dim" if not item.open else "")]
        for key, _ in LIST_COLUMNS[2:]:
            style = "dim" if not item.open else "cyan" if key in changed else ""
            cells.append(Text(_cut(get_value(item.draft, key), self.widths[key]), style))
        return cells

    def rebuild(self) -> None:
        table, keep = self.table, self.cursor_id
        widths = self._widths()
        if widths != self.widths:
            self.widths = widths
            table.clear(columns=True)
            for key, label in self.COLUMNS:
                table.add_column(label, width=widths[key], key=key)
        else:
            table.clear()
        self.rows_shown = [i for i in self.pc.items if self._matches(i)]
        for item in self.rows_shown:
            table.add_row(*self._cells(item), key=str(item.id))
        if self.rows_shown:
            index = next((n for n, i in enumerate(self.rows_shown) if i.id == keep), 0)
            table.move_cursor(row=index)
            self.cursor_id = self.rows_shown[index].id
        else:
            self.cursor_id = None
        self._title()

    def _title(self) -> None:
        pc = self.pc
        count = f"{len(self.rows_shown)}/{len(pc.items)}" if self.filter else str(len(pc.items))
        self.query_one("#box").border_title = (f" Documents({pc.cfg.process.scope})[{count}]"
                                               + (f" </{self.filter}>" if self.filter else "") + " ")
        note = ("loading…" if pc.loading else "nothing to process" if not pc.items and not pc.load_error else "")
        self.query_one(Crumbs).show(["documents"], note)

    def update_view(self, item: Item | None = None, rebuild: bool = False) -> None:
        if rebuild or item is None:
            self.rebuild()
        else:
            self.update_row(item)
        self._title()
        self.refresh_top()

    def update_row(self, item) -> None:
        if str(item.id) in self.table.rows:
            for (key, _), cell in zip(self.COLUMNS, self._cells(item), strict=True):
                self.table.update_cell(str(item.id), key, cell)

    def on_resize(self, event: events.Resize) -> None:
        super().on_resize(event)
        if self.widths and self._widths() != self.widths:
            self.rebuild()

    def focus_item(self, doc_id: int) -> None:
        index = next((n for n, i in enumerate(self.rows_shown) if i.id == doc_id), None)
        if index is not None:
            self.table.move_cursor(row=index)
            self.cursor_id = doc_id

    def _move_on(self) -> None:
        target = self.pc.next_open(self.selected(), self.rows_shown)
        if target is not None:
            self.focus_item(target.id)

    # -------------------------------------------------------- events and actions

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            self.cursor_id = self.key(event.row_key.value)
            self.pc.engine.jobs.preferred = self.cursor_id
            self.refresh_top()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            self.app.push_screen(self.doc_screen(self.key(event.row_key.value)))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter":
            self.filter = event.value.strip().lstrip("/")
            self.rebuild()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.table.focus()

    def action_move(self, step: int) -> None:
        if step > 0:
            self.table.action_cursor_down()
        else:
            self.table.action_cursor_up()

    def action_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.add_class("active")
        box.focus()

    def action_clear_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.remove_class("active")
        if self.filter or box.value:
            with box.prevent(Input.Changed):
                box.value = ""
            self.filter = ""
            self.rebuild()
        self.table.focus()

    def action_accept(self) -> None:
        item = self.selected()
        if item is None:
            return
        problem = self.pc.accept(item)
        if problem is None:
            self._move_on()
        elif problem in self.pc.labels:  # an invalid value: show it in the form
            self.app.push_screen(self.doc_screen(item.id, focus=problem))

    def action_reject(self) -> None:
        if item := self.selected():
            self.pc.reject(item)
            self._move_on()

    async def action_quit(self) -> None:
        await self.app.action_quit()


class DocScreen(View):
    DEFAULT_CSS = """
    DocScreen #body { height: 1fr; }
    DocScreen.wide #body { layout: horizontal; }
    DocScreen #form-box { height: auto; border: round $primary; border-title-align: center;
                          border-title-color: $accent; border-title-style: bold; padding: 0 1; }
    DocScreen.wide #form-box { width: 1fr; height: 1fr; }
    DocScreen #text-box { height: 1fr; border: round $panel-lighten-2; border-title-align: center; padding: 0 1; }
    DocScreen.wide #text-box { width: 1fr; }
    DocScreen #text-box:focus { border: round $accent; }
    DocScreen.text-only #form-box { display: none; }
    DocScreen #heading { height: auto; margin-bottom: 1; }
    DocScreen .row { height: 1; }
    DocScreen .row.note { height: 4; }
    DocScreen .row .name { width: 16; color: $text-muted; }
    DocScreen .row Input { width: 3fr; }
    DocScreen .row TextArea { width: 3fr; height: 4; }
    DocScreen .row .was { width: 2fr; max-width: 40; padding-left: 1; color: $text-muted; text-wrap: nowrap;
                          text-overflow: ellipsis; }
    DocScreen .row.changed .name { color: $accent; text-style: bold; }
    DocScreen .row.invalid .name { color: $error; text-style: bold; }
    DocScreen .row.current .name { text-style: bold reverse; }
    """
    BINDINGS = [
        Binding("e", "edit", "Edit"),
        Binding("a", "accept", "Accept"),
        Binding("x", "reject", "Reject"),
        Binding("g", "suggest", "Ask model"),
        Binding("u", "undo", "Undo edits"),
        Binding("o", "browser", "Open in Paperless"),
        Binding("n", "step(1)", "Next"),
        Binding("N", "step(-1)", "Previous"),
        Binding("t", "text", "Text only"),
        Binding("p", "pause", "Pause model"),
        Binding("A", "accept_all", "Accept all"),
        Binding("colon", "palette", "Commands"),
        Binding("question_mark,f1", "help", "Help"),
        Binding("escape,q", "back", "Back"),
        Binding("ctrl+s", "save", "Accept", priority=True),
    ]
    HINTS = [("e", "Edit", "edit"), ("a", "Accept", "accept"), ("x", "Reject", "reject"), ("g", "Ask model", "suggest"), ("u", "Undo edits", "undo"),
             ("o", "Open in Paperless", "browser"), ("n", "Next", "step(1)"), ("N", "Previous", "step(-1)"),
             ("t", "Text only", None), ("p", "Pause model", "pause"), ("?", "Help", None), ("esc", "Back", None)]
    TYPING_HINTS = [("ctrl-s", "Accept", True), ("tab", "Next field", True), ("shift-tab", "Previous field", True),
                    ("→", "Take completion", True), ("esc", "Stop editing", True)]
    TEXT_TITLE = " Document text "

    def __init__(self, doc_id: int, focus: str | None = None):
        super().__init__()
        self.doc_id = doc_id
        self.initial_focus = focus
        self.shown: int | None = None

    def compose(self) -> ComposeResult:
        names = self.pc.names
        yield Top()
        with Container(id="body"):
            with Vertical(id="form-box"):
                yield Static(id="heading")
                for i, (key, label) in enumerate(self.pc.rows):
                    with Horizontal(classes="row note" if key == "note" else "row", id=f"row-{i}"):
                        yield Label(label, classes="name")
                        if key == "note":
                            yield TextArea(id=f"in-{i}", compact=True, soft_wrap=True)
                        else:
                            suggester = (self.pc.tag_suggester if key == "tags" else
                                         SuggestFromList(names[key], case_sensitive=False) if names.get(key) else None)
                            yield Input(id=f"in-{i}", compact=True, suggester=suggester)
                        yield Static(classes="was")
            with VerticalScroll(id="text-box"):
                yield Static(id="text", markup=False)
        yield Crumbs()

    def on_mount(self) -> None:
        self.widgets: dict[str, Input | TextArea] = {}
        self.key_of: dict[str, str] = {}
        self.row_of: dict[str, Horizontal] = {}
        for i, (key, _) in enumerate(self.pc.rows):
            self.widgets[key] = self.query_one(f"#in-{i}")
            self.key_of[f"in-{i}"] = key
            self.row_of[key] = self.query_one(f"#row-{i}", Horizontal)
        self.query_one("#text-box").border_title = self.TEXT_TITLE
        self.show(self.pc.by_id[self.doc_id])
        if self.initial_focus in self.widgets:
            self.widgets[self.initial_focus].focus()
        else:
            self.query_one("#text-box").focus()

    def selected(self) -> Item | None:
        return self.pc.by_id.get(self.doc_id)

    def hints(self) -> list[tuple[str, str, bool | None]]:
        return self.TYPING_HINTS if self.typing() else super().hints()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in ("save", "back"):
            return True
        if self.typing():
            return False
        if action == "step":
            items = self.pc.list_items()
            ids = [i.id for i in items]
            if self.doc_id not in ids:
                return None
            index = ids.index(self.doc_id) + int(parameters[0] if parameters else 1)
            return True if 0 <= index < len(ids) else None
        return super().check_action(action, parameters)

    def on_resize(self, event: events.Resize) -> None:
        super().on_resize(event)
        self.set_class(event.size.width >= 160, "wide")

    # -------------------------------------------------------- form

    def _put(self, key: str, draft: Draft) -> None:
        """Set a form field, unless it already shows the same value (keeps what the user is typing)."""
        widget = self.widgets[key]
        raw = widget.text if isinstance(widget, TextArea) else widget.value
        probe = draft.copy()
        set_value(probe, key, raw)
        text = get_value(draft, key)
        if get_value(probe, key) == text:
            return
        if isinstance(widget, TextArea):
            with widget.prevent(TextArea.Changed):
                widget.text = text
        else:
            with widget.prevent(Input.Changed):
                widget.value = text

    def show(self, item: Item) -> None:
        same = item.id == self.shown
        self.doc_id = self.shown = item.id
        self.pc.engine.jobs.preferred = item.id
        for key, widget in self.widgets.items():
            self._put(key, item.draft)
            widget.disabled = not item.editable
        if not item.editable and self.typing():
            self.query_one("#text-box").focus()
        self.query_one("#heading", Static).update(self._heading(item))
        self.query_one("#form-box").border_title = self.form_title(item)
        self.show_text(item, same)
        if not same:
            self.query_one("#text-box", VerticalScroll).scroll_home(animate=False)
        self._decorate(item)
        self.query_one(Crumbs).show(self.crumbs(item))
        self.refresh_top()

    def form_title(self, item: Item) -> str:
        return f" #{item.id} {_cut(item.doc.get('title') or '', 60)} "

    def show_text(self, item: Item, same: bool) -> None:
        """The panel next to the form; `same`: the document shown has not changed."""
        if not same:
            content = (item.doc.get("content") or "").strip()
            self.query_one("#text", Static).update(
                content or "(No text in Paperless – the model gets the first page as an image.)")

    def crumbs(self, item: Item) -> list[str]:
        return ["documents", f"#{item.id}"]

    def _heading(self, item: Item) -> Text:
        doc = item.doc
        facts = ["in the inbox" if item.in_inbox else "not in the inbox"]
        if doc.get("added"):
            facts.append(f"added {doc['added'][:10]}")
        if doc.get("original_file_name"):
            facts.append(doc["original_file_name"])
        t = Text(" · ".join(facts), "dim")
        lines = {
            "new": ("No suggestion yet – g asks the model, or fill in the form yourself (e).", "dim"),
            "queued": ("Waiting for the model – g moves it to the front.", "dim"),
            "running": ("The model is working on this document…", "yellow"),
            "ready": (f"Suggestion by {self.pc.cfg.enrich.model}" + (f" ({item.secs:.0f} s)" if item.secs else "")
                      + ". Check it, then a accepts or e edits.", "green"),
            "edited": ("Edited – a or Ctrl+S writes your values, u undoes your edits.", "cyan"),
            "failed": ("The model failed – g tries again.", "red"),
            "saving": ("Writing to Paperless…", "yellow"),
            "done": ("Written to Paperless.", "green"),
            "rejected": ("Rejected – only the processed tag was added. a or Ctrl+S still writes the form, "
                         "g asks the model again.", "dim"),
        }
        message, style = lines.get(item.label, ("", ""))
        t.append("\n" + message, style)
        if item.error:
            t.append(f"\n{item.error}", "red")
        return t

    def _decorate(self, item: Item) -> None:
        """Highlight changed and invalid values; show the old value (or the problem) on the right."""
        changes, errors = changes_for(item.current, item.draft, self.pc.cfg)
        changed = changed_keys(changes)
        for key, row in self.row_of.items():
            row.set_class(key in changed, "changed")
            row.set_class(key in errors, "invalid")
            was = Text()
            if key in errors:
                was.append(errors[key], "red")
            elif key == "tags" and key in changed:
                for tag in changes.get("tags", []):
                    was.append(f"+{tag} ", "green")
                for tag in changes.get("untag", []):
                    was.append(f"−{tag} ", "red")
            elif key == "note" and key in changed:
                was.append("replaces the earlier note" if item.note_id else "added as a note")
            elif key in changed:
                old = get_value(item.current, key)
                was.append(f"was {old}" if old else "was empty")
            row.query_one(".was", Static).update(was)

    def update_view(self, item: Item | None = None, rebuild: bool = False) -> None:
        current_item = self.pc.by_id.get(self.doc_id)
        if current_item is None:  # gone after a reload
            self.app.pop_screen()
            return
        if rebuild or item is None or item.id == self.doc_id:
            self.show(current_item)
        else:
            self.refresh_top()

    # -------------------------------------------------------- events and actions

    def on_descendant_focus(self) -> None:
        # a class instead of :focus-within, which does not restyle the label when the focus moves on
        for key, row in getattr(self, "row_of", {}).items():
            row.set_class(self.focused is self.widgets[key], "current")
        self.refresh_top()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._edited(event.input, event.value)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._edited(event.text_area, event.text_area.text)

    def _edited(self, widget: Input | TextArea, value: str) -> None:
        item, key = self.selected(), self.key_of.get(widget.id or "")
        if item is None or key is None or not item.editable:
            return
        set_value(item.draft, key, value)
        self._decorate(item)
        self.query_one("#heading", Static).update(self._heading(item))
        self.pc.changed(item, source=self)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.focus_next()

    def action_edit(self) -> None:
        item = self.selected()
        if item is not None and item.editable:
            self.remove_class("text-only")
            self.widgets["title"].focus()

    def action_accept(self) -> None:
        item = self.selected()
        if item is None:
            return
        problem = self.pc.accept(item)
        if problem is None:
            self._move_on(item)
        elif problem in self.widgets:
            self.remove_class("text-only")
            self.widgets[problem].focus()

    def action_save(self) -> None:
        self.action_accept()

    def action_reject(self) -> None:
        if item := self.selected():
            self.pc.reject(item)
            self._move_on(item)

    def _move_on(self, item: Item) -> None:
        """After accepting or rejecting: the next document to check, or back to the list."""
        target = self.pc.next_open(item, self.pc.list_items(), with_suggestion=True)
        if target is None:
            self.notify("No more documents with a suggestion to check.")
            self.action_back()
        else:
            self.show(target)
            self.query_one("#text-box").focus()

    def action_step(self, step: int) -> None:
        ids = [i.id for i in self.pc.list_items()]
        if self.doc_id in ids and 0 <= ids.index(self.doc_id) + step < len(ids):
            self.show(self.pc.by_id[ids[ids.index(self.doc_id) + step]])

    def action_text(self) -> None:
        self.toggle_class("text-only")
        self.query_one("#text-box").focus()

    def action_back(self) -> None:
        if self.typing():
            self.query_one("#text-box").focus()
            return
        if self.has_class("text-only"):
            self.remove_class("text-only")
            return
        doc_id = self.doc_id
        self.app.pop_screen()
        self.pc.list_screen.focus_item(doc_id)


# ---------------------------------------------------------------- the app

class ReviewApp(App[None]):
    """What both terminal UIs share: a list and a document view over items, an engine with background threads
    (`engine.jobs` pauses the model and knows the item on screen), writes that quitting waits for."""

    TITLE = "paperclerk"

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.rows = form_rows(cfg)
        self.labels = dict(self.rows)
        self.items: list = []
        self.by_id: dict = {}
        self.names: dict[str, list[str]] = {}
        self.engine = None  # set by the subclass
        self.loading = True
        self.load_error = ""
        self.model_error = ""
        self.pending = 0  # writes on their way
        self.quitting = False
        self.tag_suggester = TagSuggester()
        self.log_handler = UiLog(self)
        self.list_screen = self.make_list_screen()

    def make_list_screen(self) -> ListScreen:
        return ListScreen()

    # Textual calls on_mount/on_unmount of every class in the MRO, so subclasses use these hooks instead
    def prepare(self) -> None:
        """Before the list is shown and the engine starts."""

    def cleanup(self) -> None:
        """After the engine has been told to stop."""

    def on_mount(self) -> None:
        logging.getLogger().addHandler(self.log_handler)
        self.prepare()
        self.push_screen(self.list_screen)
        self.engine.start()
        self.set_interval(1, self._tick)

    def on_unmount(self) -> None:
        logging.getLogger().removeHandler(self.log_handler)
        self.engine.stop()
        self.cleanup()

    def _show_only(self, status: str) -> None:
        while len(self.screen_stack) > 2:
            self.pop_screen()
        box = self.list_screen.query_one("#filter", Input)
        box.set_class(bool(status), "active")
        box.value = status  # the filter follows through Input.Changed

    def changed(self, item=None, rebuild: bool = False, source: Screen | None = None) -> None:
        """State changed: refresh every view (the source view has already updated itself)."""
        for screen in self.screen_stack:
            if isinstance(screen, View) and screen.is_mounted:
                if screen is source:
                    screen.refresh_top()
                else:
                    screen.update_view(item, rebuild)

    def _tick(self) -> None:
        if isinstance(self.screen, View):
            self.screen.refresh_top()

    def list_items(self) -> list:
        """The items in the order of the list, with its filter."""
        return self.list_screen.rows_shown or self.items

    def next_open(self, after, items: list, with_suggestion: bool = False):
        """The next item worth looking at: one with a suggestion, else (unless required) any open one."""
        start = items.index(after) if after in items else -1
        order = items[start + 1:] + items[:max(start, 0)]
        found = next((i for i in order if i.open and (i.suggestion is not None or i.edited)), None)
        if found is None and not with_suggestion:
            found = next((i for i in order if i.open), None)
        return found

    def undo(self, item) -> None:
        if item.edited:
            item.draft = item.base.copy()
            self.changed(item)

    def toggle_pause(self) -> None:
        jobs = self.engine.jobs
        jobs.pause(not jobs.paused)
        if not jobs.paused:
            self.model_error = ""
        self.notify("Model paused – the document in progress is finished first." if jobs.paused else "Model resumed.")
        self.changed()

    def unsaved(self) -> list:
        """Items with edits that quitting would lose; a rejected document is handled already."""
        return [i for i in self.items if i.edited and i.open]

    async def action_quit(self) -> None:
        if isinstance(self.screen, ModalScreen):
            return
        edited = self.unsaved()
        if edited:
            self.push_screen(Confirm(f"{len(edited)} documents have edits that are not saved. "
                                     "Quit and lose them?"), lambda yes: self._quit() if yes else None)
        else:
            self._quit()

    def _quit(self) -> None:
        self.quitting = True
        if self.pending:
            self.notify(f"Finishing {self.pending} writes to Paperless…")
        self._finish_quit()

    def _finish_quit(self) -> None:
        if self.quitting and not self.pending:
            self.engine.stop()
            self.exit()

    def notify(self, message: str, *, title: str = "", severity: str = "information", timeout: float | None = None,
               markup: bool = True) -> None:
        """Show a notification; warnings and errors also go to the log file, so they can be read after
        the toast is gone (the background threads report most problems only to the UI)."""
        if severity in ("warning", "error"):
            log.log(logging.ERROR if severity == "error" else logging.WARNING, "%s%s", f"{title}: " if title else "",
                    message, extra={"shown": True})
        super().notify(message, title=title, severity=severity, timeout=timeout, markup=markup)

    def on_log_line(self, msg: LogLine) -> None:
        severity = ("error" if msg.level >= logging.ERROR else "warning" if msg.level >= logging.WARNING
                    else "information")
        super().notify(msg.text, severity=severity, markup=False)  # it is in the log file already

    # the parts that differ between the two UIs
    def help_text(self) -> str:
        raise NotImplementedError

    def browse(self, item) -> None:
        raise NotImplementedError


class ProcessApp(ReviewApp):
    """`paperclerk ui`: documents in Paperless that `process` would handle."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.engine = Engine(cfg, self.post_message)
        self.inbox: set[int] = set()
        self.typical: float | None = None
        self.durations: list[float] = []

    def help_text(self) -> str:
        return help_text(self.cfg)

    def browse(self, item: Item) -> None:
        webbrowser.open(f"{self.cfg.paperless.url}/documents/{item.id}/details")

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        for status in ("ready", "edited", "queued", "failed", "done"):
            yield SystemCommand(f"Show {status} documents", f"Filter the list by the status {status}",
                                lambda s=status: self._show_only(s))
        yield SystemCommand("Show all documents", "Clear the filter", lambda: self._show_only(""))
        yield SystemCommand("Accept all", "Write every document that has a suggestion", self.accept_all)
        yield SystemCommand("Pause or resume the model", "Stop asking the model after the current document",
                            self.toggle_pause)
        yield SystemCommand("Reload", "Read the documents from Paperless again", self.reload)

    # -------------------------------------------------------- state shared by the views

    def info(self) -> Text:
        jobs, engine = self.engine.jobs, self.engine
        lines: list[tuple[str, Text]] = []
        host = self.cfg.paperless.url.split("://", 1)[-1]
        lines.append(("Paperless:", Text(f"{host} · scope {self.cfg.process.scope}")))
        if self.loading:
            docs = Text("loading…", "yellow")
        elif self.load_error:
            docs = Text(_cut(self.load_error, 46), "red")
        else:
            counts = Counter(i.label for i in self.items)
            docs = Text(f"{len(self.items)}")
            for label in ("ready", "edited", "failed", "done", "rejected"):
                if counts[label]:
                    docs.append(f" · {counts[label]} {label}", STYLES[label])
        lines.append(("Documents:", docs))
        model = Text(self.cfg.enrich.model)
        running = engine.running
        if self.model_error:
            model.append(" · " + _cut(self.model_error, 36), "red")
        elif running is not None:
            elapsed = time.monotonic() - engine.started_at
            model.append(f" · #{running} for {fmt_duration(elapsed)}", "yellow")
            if not self.durations and elapsed > 15:
                model.append(" (loading)", "dim")
        elif jobs.paused:
            model.append(" · paused", "yellow")
        else:
            model.append(" · idle", "dim")
        lines.append(("Model:", model))
        queued = len(jobs)
        queue_text = Text(f"{queued} queued" if queued else "empty", "" if queued else "dim")
        typical = statistics.median(self.durations) if self.durations else self.typical
        left = queued + (running is not None)
        if typical and left:
            queue_text.append(f" · ~{fmt_duration(typical)}/doc · ~{fmt_duration(typical * left)} left", "dim")
        if jobs.paused and not self.model_error:
            queue_text.append(" · p resumes", "dim")
        lines.append(("Queue:", queue_text))
        lines.append(("Writing:", Text(f"{self.pending}…", "yellow") if self.pending else Text("–", "dim")))
        out = Text(no_wrap=True, overflow="ellipsis")
        for n, (label, value) in enumerate(lines):
            out.append(f"{label:<11}", LABEL_STYLE)
            out.append_text(value)
            if n < len(lines) - 1:
                out.append("\n")
        return out

    def acceptable(self) -> list[Item]:
        return [i for i in self.items if i.open and i.state != "running" and (i.suggestion is not None or i.edited)]

    # -------------------------------------------------------- operations

    def accept(self, item: Item) -> str | None:
        """Start writing the document. Returns None, or why not: "nothing" or the key of an invalid field."""
        if not item.editable:
            return "finished"
        if item.suggestion is None and not item.edited:
            self.notify("No suggestion yet. Press g to ask the model now" + (", edit the values yourself, or x to "
                        "reject." if item.open else " or edit the values yourself."), markup=False)
            return "nothing"
        changes, errors = changes_for(item.current, item.draft, self.cfg)
        if errors:
            key = next(k for k, _ in self.rows if k in errors)
            self.notify(f"#{item.id} {self.labels[key]}: {errors[key]}", title="Not accepted", severity="warning",
                        markup=False)
            self.changed(item)
            return key
        self._write(item, changes, "applied")
        return None

    def reject(self, item: Item) -> None:
        """Write nothing but the processed tag. The form is kept, so the document can still be accepted."""
        if item.open:
            self._write(item, {}, "rejected")
            self.notify(f"#{item.id} rejected – a or Ctrl+S on it still writes it.", markup=False)

    def _write(self, item: Item, changes: dict, status: str) -> None:
        item.state, item.error = "saving", ""
        self.engine.jobs.discard(item.id)
        self.pending += 1
        self.engine.write(item.doc, changes, item.note_id, status)
        self.changed(item)

    def suggest(self, item: Item) -> None:
        if not item.editable or item.state == "running":
            return
        item.state, item.error = "queued", ""
        self.engine.jobs.put(item.id, front=True)
        if self.engine.jobs.paused:
            self.notify("Queued first – the model is paused, p resumes.")
        self.changed(item)

    def accept_all(self) -> None:
        ready = self.acceptable()
        if not ready:
            self.notify("No document has a suggestion yet.")
            return
        edited = sum(1 for i in ready if i.edited)
        question = (f"Write {len(ready)} documents to Paperless"
                    + (f", {edited} of them with your edits" if edited else "")
                    + "? Documents without a suggestion are left alone.")
        self.push_screen(Confirm(question), lambda yes: self._accept_many(ready) if yes else None)

    def _accept_many(self, items: list[Item]) -> None:
        written = skipped = 0
        for item in items:
            if not item.open:
                continue
            changes, errors = changes_for(item.current, item.draft, self.cfg)
            if errors:
                skipped += 1
                continue
            self._write(item, changes, "applied")
            written += 1
        self.notify(f"Writing {written} documents." + (f" {skipped} skipped because of invalid values."
                                                       if skipped else ""), markup=False)

    def reload(self) -> None:
        if not self.loading:
            self.loading = True
            self.engine.load()
            self.changed()

    # -------------------------------------------------------- messages from the background threads

    def on_loaded(self, msg: Loaded) -> None:
        self.loading, self.load_error = False, ""
        self.inbox, self.names = msg.inbox, msg.names
        self.typical = msg.typical or self.typical
        registry, old = self.engine.registry, self.by_id
        items: list[Item] = []
        for doc in msg.docs:
            prev = old.get(doc["id"])
            if prev is not None and prev.state == "saving":  # the write is still on its way
                items.append(prev)
                continue
            if prev is not None and prev.tagged:  # written or rejected, but its processed tag was removed in
                prev = None  # Paperless since: process it again
            rec = msg.records.get(doc["id"], {})
            suggestion = msg.suggestions.get(doc["id"])
            in_inbox = bool(self.inbox & set(doc.get("tags") or []))
            base = proposal(doc, suggestion, registry, self.cfg, self.inbox)
            item = Item(doc, in_inbox, current(doc, registry, self.cfg), base, base.copy(), suggestion, "queued",
                        note_id=rec.get("note_id"), fail_count=rec.get("fail_count") or 0,
                        secs=rec.get("secs") if suggestion else None)
            if prev is not None:  # reload: keep the user's edits
                item.draft = merge(prev.base, base, prev.draft)
            if prev is not None and prev.state == "running":
                item.state = "running"
            elif suggestion is not None:
                item.state = "ready"
            elif rec.get("status") == "failed" and item.fail_count >= process.MAX_FAILS:
                item.state, item.error = "failed", f"Model: {rec.get('error') or ''}"
            items.append(item)
        seen = {i.id for i in items}
        kept = [i for i in old.values() if i.id not in seen and (i.tagged or not i.open)]
        items += kept  # written and rejected ones stay in view; a rejected one can still be accepted
        seen |= {i.id for i in kept}
        for gone in (i for i in old.values() if i.id not in seen):
            self.engine.jobs.discard(gone.id)
        items.sort(key=lambda i: i.id)
        self.items, self.by_id = items, {i.id: i for i in items}
        self.engine.docs = {i.id: i.doc for i in items}
        for item in items:
            if item.state == "queued":
                self.engine.jobs.put(item.id)
        self.tag_suggester.tags = list(msg.names["tags"])
        self.changed(rebuild=True)

    def on_load_failed(self, msg: LoadFailed) -> None:
        self.loading, self.load_error = False, msg.error
        self.notify(msg.error, title="Paperless", severity="error", markup=False, timeout=10)
        self.changed(rebuild=True)

    def on_model_started(self, msg: ModelStarted) -> None:
        item = self.by_id.get(msg.doc_id)
        if item is not None and item.open:
            item.state = "running"
        self.changed(item)

    def on_suggested(self, msg: Suggested) -> None:
        self.durations.append(msg.secs)
        self.model_error = ""
        item = self.by_id.get(msg.doc_id)
        if item is None:
            return
        if not item.open:  # written while the model was still working
            self.engine.rerecord(item.id)
            self.changed()
            return
        old = item.base
        item.suggestion = msg.suggestion
        item.base = proposal(item.doc, msg.suggestion, self.engine.registry, self.cfg, self.inbox)
        item.draft = merge(old, item.base, item.draft)
        item.state, item.error, item.secs, item.fail_count = "ready", "", msg.secs, 0
        known = {t.casefold() for t in self.tag_suggester.tags}
        self.tag_suggester.tags += [t for t in msg.suggestion.get("tags") or [] if t.casefold() not in known]
        self.changed(item)

    def on_suggest_failed(self, msg: SuggestFailed) -> None:
        item = self.by_id.get(msg.doc_id)
        if msg.fatal:
            self.model_error = msg.error
            self.notify(f"{msg.error}\nThe model is paused – p tries again.", title="Model unavailable",
                        severity="error", markup=False, timeout=15)
            if item is not None and item.open:
                item.state = "queued"
        elif item is not None and item.open:
            item.state = "ready" if item.suggestion is not None else "failed"
            item.error = f"Model: {msg.error}"
            item.fail_count += 1
        self.changed(item)

    def on_written(self, msg: Written) -> None:
        self.pending -= 1
        item = self.by_id.get(msg.doc_id)
        if item is not None:  # stored status applied / rejected -> shown as done / rejected
            item.state = "done" if msg.status == "applied" else msg.status
            item.note_id, item.error, item.tagged = msg.note_id, "", True
        self.changed(item)
        self._finish_quit()

    def on_write_failed(self, msg: WriteFailed) -> None:
        self.pending -= 1
        item = self.by_id.get(msg.doc_id)
        if item is not None:
            item.state = "ready" if item.suggestion is not None else "new"
            item.error = f"Not written: {msg.error}"
        self.notify(msg.error, title=f"#{msg.doc_id} not written", severity="error", markup=False, timeout=15)
        if self.quitting:  # stay, so the user sees what was not written
            self.quitting = False
            self.notify("Quitting was cancelled because a document could not be written.", severity="warning")
        self.changed(item)


def run_app(app: App) -> int:
    root = logging.getLogger()
    console = [h for h in root.handlers if type(h) is logging.StreamHandler]  # would write over the screen
    for handler in console:
        root.removeHandler(handler)
    try:
        app.run()
    finally:
        for handler in console:
            root.addHandler(handler)
    return 0


def run_ui(cfg: Config) -> int:
    return run_app(ProcessApp(cfg))
