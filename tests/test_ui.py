"""`paperclerk ui`: the form logic and the app against fake Ollama and Paperless servers."""

from __future__ import annotations

import asyncio
import time

from textual import events
from textual.widgets._toast import Toast

from paperclerk.draft import Draft, changes_for, current, merge, proposal
from paperclerk.paperless import Registry
from paperclerk.ui import Confirm, DocScreen, Help, Jobs, ProcessApp, help_text
from samples import INVOICE
from test_merge import CONTRACT

SUGGESTION = {"correspondent": "Telekom Deutschland", "document_type": "Rechnung", "date": "2023-04-17",
              "title": "Rechnung Mobilfunk April 2023", "tags": ["Telekommunikation"],
              "fields": {"Betrag": "EUR44.15", "Rechnungsnummer": "4400"}, "summary": "Mobilfunkrechnung."}


def _setup(fp):
    fp.add_document(10, "scan_2023_04", INVOICE, tags=[1], correspondent=2, created="2023-04-30",
                    custom_fields=[{"field": 1, "value": "EUR10.00"}])
    fp.add_document(11, "Mein Vertrag", "\n".join(CONTRACT), tags=[1, 2], document_type=1)
    fp.add_document(12, "Archiv", INVOICE, tags=[2])  # not in the inbox: not selected


# ---------------------------------------------------------------- form logic

def test_proposal_follows_the_write_policy(db, cfg, fake_paperless):
    _setup(fake_paperless)
    registry = Registry(db, None, fake_paperless.objects)
    doc = fake_paperless.docs[10]
    cur = current(doc, registry, cfg)
    assert (cur.correspondent, cur.tags, cur.fields["Betrag"]) == ("info@o2.com", ["new"], "EUR10.00")
    d = proposal(doc, SUGGESTION, registry, cfg, inbox={1})
    assert d.title == "Rechnung Mobilfunk April 2023" and d.correspondent == "Telekom Deutschland"
    assert d.tags == ["Telekommunikation"] and d.date == "2023-04-17"  # accepting leaves the inbox
    assert d.fields["Betrag"] == "EUR10.00"  # filled values are kept
    assert d.fields["Rechnungsnummer"] == "4400" and d.note == "Mobilfunkrechnung."
    assert proposal(doc, None, registry, cfg, inbox={1}).tags == []  # also before the model has answered
    outside = proposal(doc, SUGGESTION, registry, cfg, inbox=set())
    assert outside.title == "Rechnung Mobilfunk April 2023"  # the title was only the file name
    assert outside.correspondent == "info@o2.com" and outside.date == "2023-04-30"
    assert outside.tags == ["new", "Telekommunikation"]
    cfg.process.remove_inbox_tags = False
    assert proposal(doc, SUGGESTION, registry, cfg, inbox={1}).tags == ["new", "Telekommunikation"]


def test_help_follows_the_inbox_setting(cfg):
    on = help_text(cfg)
    cfg.process.remove_inbox_tags = False
    off = help_text(cfg)
    assert "and remove the inbox tag" in on and "and keep the inbox tag" in off
    for text in (on, off):
        line = next(x for x in text.splitlines() if "take the grey completion" in x)
        assert line.index("→") == 44  # the columns stay aligned


def test_changes_from_edits(cfg):
    cur = Draft("scan", "info@o2.com", "", "2023-04-30", ["new", "Bank"], {"Betrag": "EUR10.00", "Fällig am": ""})
    draft = Draft("Handyrechnung", "Telekom", "rechnung", "17.04.2023", ["Bank", "Mobilfunk"],
                  {"Betrag": "44,15 €", "Fällig am": ""}, "Kurz.")
    changes, errors = changes_for(cur, draft, cfg)
    assert not errors
    assert changes == {"title": ["scan", "Handyrechnung"], "correspondent": ["info@o2.com", "Telekom"],
                       "document_type": [None, "rechnung"], "created": ["2023-04-30", "2023-04-17"],
                       "tags": ["Mobilfunk"], "untag": ["new"], "fields": {"Betrag": "EUR44.15"}, "note": "Kurz."}
    cleared = Draft("scan", "", "", "2023-04-30", ["new", "Bank"], {"Betrag": "", "Fällig am": ""})
    assert changes_for(cur, cleared, cfg)[0] == {"correspondent": ["info@o2.com", ""], "fields": {"Betrag": None}}
    bad = Draft("", "", "", "31.02.2023", [], {"Betrag": "viel", "Fällig am": "bald"})
    errors = changes_for(cur, bad, cfg)[1]
    assert set(errors) == {"title", "date", "field:Betrag", "field:Fällig am"}
    assert changes_for(cur, cur.copy(), cfg)[0] == {}


def test_merge_keeps_what_the_user_changed():
    old = Draft("scan", "info@o2.com", "", "2023-04-30", ["new"], {"Betrag": ""})
    new = Draft("Rechnung April", "Telekom", "Rechnung", "2023-04-17", ["new", "Bank"], {"Betrag": "EUR44.15"}, "S.")
    mine = old.copy()
    mine.title, mine.fields["Betrag"] = "Meine Rechnung", "EUR40.00"
    result = merge(old, new, mine)
    assert result.title == "Meine Rechnung" and result.fields["Betrag"] == "EUR40.00"
    assert (result.correspondent, result.tags, result.note) == ("Telekom", ["new", "Bank"], "S.")


def test_jobs_prefer_the_document_on_screen():
    jobs = Jobs()
    for i in (1, 2, 3):
        jobs.put(i)
    jobs.put(1)  # already queued: stays in place
    jobs.preferred = 3
    assert jobs.get() == 3
    jobs.put(2, front=True)
    assert [jobs.get(), jobs.get()] == [2, 1]
    jobs.close()
    assert jobs.get() is None


# ---------------------------------------------------------------- the app

def run(app: ProcessApp, steps, size=(160, 48), notifications=False):
    async def main():
        async with app.run_test(size=size, notifications=notifications) as pilot:
            await steps(pilot)
    asyncio.run(main())


async def until(pilot, condition, timeout: float = 15):
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            raise AssertionError("timed out")
        await pilot.pause(0.05)


def _tag_names(fp, doc_id):
    names = {o["id"]: o["name"] for o in fp.objects["tags"]}
    return [names[t] for t in fp.docs[doc_id]["tags"]]


def _ready(app, n=2):
    return lambda: len(app.items) == n and all(i.state == "ready" for i in app.items)


def test_review_flow_in_the_document_view(db, cfg, fake_ollama, fake_paperless, monkeypatch):
    fp = fake_paperless
    _setup(fp)
    app = ProcessApp(cfg)
    opened = []
    monkeypatch.setattr("webbrowser.open", opened.append)

    async def steps(pilot):
        await until(pilot, _ready(app))
        screen = app.list_screen
        assert [i.id for i in screen.rows_shown] == [10, 11] and screen.cursor_id == 10
        assert "Documents(inbox)[2]" in screen.query_one("#box").border_title
        await pilot.press("o")
        assert opened == ["http://paperless.test/documents/10/details"]

        await pilot.press("enter")
        await pilot.pause()
        doc = app.screen
        assert isinstance(doc, DocScreen) and doc.doc_id == 10
        assert doc.widgets["title"].value == "Rechnung Mobilfunk April 2023"
        assert doc.widgets["correspondent"].value == "Telekom Deutschland"
        assert doc.row_of["correspondent"].has_class("changed")
        assert doc.widgets["tags"].value == "Telekommunikation, Mobilfunk"  # without the inbox tag
        assert "−new" in str(doc.row_of["tags"].query_one(".was").render())

        await pilot.press("e")  # the title is selected: typing replaces it; letters do not trigger shortcuts
        await pilot.press(*"Handyrechnung April", "tab", "tab", "tab", "tab")
        assert app.focused is doc.widgets["tags"] and app.by_id[10].label == "edited"
        await pilot.press("end", *", Vers", "right")  # completion of an existing tag
        assert app.by_id[10].draft.tags[-1] == "Versicherung"
        await pilot.press("ctrl+s")  # accept, then the next document with a suggestion
        assert doc.doc_id == 11 and app.screen is doc
        await pilot.press("e", "tab", "tab", "tab", "tab", "end", *", new")  # stays in the inbox
        await pilot.press("escape", "a")  # stop editing, accept; nothing left: back to the list
        await pilot.pause()
        assert app.screen is screen
        await until(pilot, lambda: all(i.state == "done" for i in app.items))

        assert fp.docs[10]["title"] == "Handyrechnung April"
        assert _tag_names(fp, 10) == ["Telekommunikation", "Mobilfunk", "Versicherung", "paperclerk"]
        assert _tag_names(fp, 11) == ["new", "Versicherung", "Immobilie", "paperclerk"]
        assert {r["doc_id"]: r["status"] for r in db.q("SELECT * FROM processed")} == {10: "applied", 11: "applied"}

    run(app, steps)
    assert len([c for c in fake_ollama.calls if c["path"] == "/api/chat"]) == 2


def test_invalid_values_open_the_form(db, cfg, fake_ollama, fake_paperless):
    fp = fake_paperless
    _setup(fp)
    app = ProcessApp(cfg)

    async def steps(pilot):
        await until(pilot, _ready(app))
        app.by_id[10].draft.date = "31.02.2023"
        app.by_id[10].draft.fields["Betrag"] = "50,00 €"
        await pilot.press("a")  # in the list: the form opens at the invalid value
        await pilot.pause()
        doc = app.screen
        assert isinstance(doc, DocScreen) and app.focused is doc.widgets["date"]
        assert doc.row_of["date"].has_class("invalid") and app.by_id[10].open and not fp.patches
        await pilot.press(*"17.04.2023", "ctrl+s")
        await until(pilot, lambda: app.by_id[10].state == "done")
        assert doc.doc_id == 11
        assert fp.docs[10]["created"] == "2023-04-17"
        assert {f["field"]: f["value"] for f in fp.docs[10]["custom_fields"]}[1] == "EUR50.00"  # was EUR10.00

    run(app, steps)


def test_filter_and_commands(db, cfg, fake_ollama, fake_paperless):
    _setup(fake_paperless)
    app = ProcessApp(cfg)

    async def steps(pilot):
        await until(pilot, _ready(app))
        screen = app.list_screen
        await pilot.press("slash", *"notar", "enter")
        assert [i.id for i in screen.rows_shown] == [11] and app.focused is screen.table
        assert "[1/2] </notar>" in screen.query_one("#box").border_title
        await pilot.press("escape")
        assert [i.id for i in screen.rows_shown] == [10, 11] and screen.cursor_id == 11  # stays on the match
        await pilot.press("x")  # reject #11
        await until(pilot, lambda: app.by_id[11].state == "rejected")
        app._show_only("ready")
        await pilot.pause()
        assert [i.id for i in screen.rows_shown] == [10]
        await pilot.press("colon")  # k9s' command prompt: Textual's command palette with paperclerk's commands
        await pilot.pause()
        assert type(app.screen).__name__ == "CommandPalette"
        await pilot.press("escape")

    run(app, steps)


def test_reject_and_unavailable_model(db, cfg, fake_ollama, fake_paperless):
    fp = fake_paperless
    _setup(fp)
    cfg.enrich.model = "not-installed"
    app = ProcessApp(cfg)

    async def steps(pilot):
        await until(pilot, lambda: app.model_error)
        assert app.engine.jobs.paused and all(i.state == "queued" for i in app.items)
        assert "ollama pull not-installed" in app.model_error
        await pilot.press("a")  # no suggestion and no edits: nothing is written
        await pilot.pause()
        assert app.by_id[10].open
        await pilot.press("x")
        await until(pilot, lambda: app.by_id[10].state == "rejected")
        assert fp.docs[10]["title"] == "scan_2023_04" and _tag_names(fp, 10) == ["new", "paperclerk"]
        assert db.one("SELECT status FROM processed WHERE doc_id = 10")["status"] == "rejected"

    run(app, steps)


def test_a_rejected_document_can_still_be_accepted(db, cfg, fake_ollama, fake_paperless):
    fp = fake_paperless
    _setup(fp)
    app = ProcessApp(cfg)

    async def steps(pilot):
        await until(pilot, _ready(app))
        await pilot.press("enter")
        await pilot.pause()
        doc = app.screen
        title = doc.widgets["title"].value
        await pilot.press("x")  # by mistake: rejected, and the next document opens
        await until(pilot, lambda: app.by_id[10].state == "rejected")
        assert doc.doc_id == 11 and _tag_names(fp, 10) == ["new", "paperclerk"]
        assert app.acceptable() == [app.by_id[11]]  # handled: not part of Accept all
        await pilot.press("N")  # back to it: the form still holds the suggestion and can be edited
        await pilot.pause()
        assert doc.doc_id == 10 and doc.widgets["title"].value == title and not doc.widgets["title"].disabled
        assert doc.check_action("reject", ()) is None and doc.check_action("accept", ()) is True
        await pilot.press("ctrl+s")
        await until(pilot, lambda: app.by_id[10].state == "done")
        assert fp.docs[10]["title"] == title
        assert "new" not in _tag_names(fp, 10) and "paperclerk" in _tag_names(fp, 10)
        assert db.one("SELECT status FROM processed WHERE doc_id = 10")["status"] == "applied"

    run(app, steps)


def test_removing_the_processed_tag_brings_a_document_back(db, cfg, fake_ollama, fake_paperless):
    fp = fake_paperless
    _setup(fp)
    app = ProcessApp(cfg)

    async def steps(pilot):
        await until(pilot, _ready(app))
        await pilot.press("x")
        await until(pilot, lambda: app.by_id[10].state == "rejected")
        await pilot.press("f5")  # still tagged: stays rejected
        await until(pilot, lambda: not app.loading)
        assert app.by_id[10].state == "rejected"
        processed = next(t["id"] for t in fp.objects["tags"] if t["name"] == "paperclerk")
        fp.docs[10]["tags"].remove(processed)  # the user removes the tag in Paperless
        await pilot.press("f5")
        await until(pilot, lambda: app.by_id[10].state == "ready")
        assert app.by_id[10].draft.title == "Rechnung Mobilfunk April 2023"

    run(app, steps)


def test_quit_asks_when_edits_are_unsaved(db, cfg, fake_ollama, fake_paperless):
    _setup(fake_paperless)
    app = ProcessApp(cfg)

    async def steps(pilot):
        await until(pilot, _ready(app))
        await pilot.press("enter")
        await pilot.pause()
        app.screen.widgets["title"].value = "Anders"
        await pilot.pause()
        await pilot.press("escape", "q")  # back to the list, quit
        await pilot.pause()
        assert isinstance(app.screen, Confirm)
        await pilot.press("n")
        await pilot.pause()
        assert app.screen is app.list_screen and app.by_id[10].edited
        await pilot.press("u")  # undo: nothing unsaved any more, q quits at once
        assert not app.by_id[10].edited
        await pilot.press("q")
        await pilot.pause()

    run(app, steps)
    assert app.return_code == 0


def test_small_terminal_help_and_text_only(db, cfg, fake_ollama, fake_paperless):
    _setup(fake_paperless)
    app = ProcessApp(cfg)

    async def steps(pilot):
        await until(pilot, _ready(app))
        await pilot.press("question_mark")
        assert isinstance(app.screen, Help)
        await pilot.press("a")  # the list's keys do not work behind the help
        await pilot.press("escape")
        assert app.screen is app.list_screen and app.by_id[10].open
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("t")
        assert app.screen.query_one("#form-box").display is False
        await pilot.press("escape")
        assert app.screen.query_one("#form-box").display is True
        await pilot.press("escape")
        assert app.screen is app.list_screen

    run(app, steps, size=(100, 30))


def test_a_toast_that_is_gone_does_not_crash_the_ui(db, cfg, fake_ollama, fake_paperless, caplog):
    """A second click on a dismissed toast (or a toast timing out under the mouse) used to crash Textual."""
    _setup(fake_paperless)
    app = ProcessApp(cfg)

    async def steps(pilot):
        await until(pilot, _ready(app))
        app.notify("Paperless unreachable", title="Paperless", severity="error", timeout=30)
        await until(pilot, lambda: app.screen.query(Toast) and app.screen.query(Toast).first().region.width)
        toast = app.screen.query(Toast).first()
        x, y = toast.region.x + 2, toast.region.y + 1
        await toast.parent.remove()  # as the first click does; the last frame still shows it
        await app.on_event(events.MouseDown(None, x, y, 0, 0, 1, False, False, False, screen_x=x, screen_y=y))
        await pilot.pause()
        assert app.is_running

    run(app, steps, notifications=True)
    logged = [r for r in caplog.records if "Paperless unreachable" in r.getMessage()]
    assert len(logged) == 1 and logged[0].levelname == "ERROR"  # in the log file too, once
    assert logged[0].getMessage() == "Paperless: Paperless unreachable"
