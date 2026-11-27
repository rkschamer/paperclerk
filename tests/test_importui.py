"""`paperclerk import` in a terminal: the import view against fake Ollama and Paperless servers."""

from __future__ import annotations

import json

from paperclerk.cli import main
from paperclerk.decision import evaluate
from paperclerk.draft import import_corrections, imported
from paperclerk.importui import Control, ImportApp, ImportDocScreen, plain_steps
from paperclerk.ui import Confirm, Help
from test_ui import run, until

INVOICE = "Rechnung_2023_04.pdf"


def settled(app: ImportApp):
    """Every step has run: nothing waits for the pipeline any more."""
    return lambda: (app.engine.phase is None and app.items and not app.dirty
                    and all(i.state not in ("new", "queued", "running") for i in app.items))


def by_name(app: ImportApp) -> dict:
    return {i.filename: i for i in app.items}


def uploads(fp) -> dict:
    return {d["original_file_name"]: d for d in fp.docs.values()}


def test_corrections_are_what_differs_from_the_model(cfg):
    model = {"title": "Rechnung April", "correspondent": "Telekom", "document_type": "Rechnung", "date": "2023-04-17",
             "tags": ["Mobilfunk"], "fields": {"Betrag": "EUR44.15"}, "summary": "Kurz."}
    row = {"meta": json.dumps(model), "overrides": None}
    draft = imported(row, cfg)
    assert draft.title == "Rechnung April" and draft.fields["Betrag"] == "EUR44.15" and draft.note == "Kurz."
    assert import_corrections(row, draft, cfg) == ({}, {})
    draft.title, draft.date, draft.fields["Betrag"] = "Handyrechnung", "18.04.2023", "45,00 €"
    corrections, errors = import_corrections(row, draft, cfg)
    assert not errors
    assert corrections == {"title": "Handyrechnung", "date": "2023-04-18", "fields": {"Betrag": "EUR45.00"}}
    stored = {"meta": row["meta"], "overrides": json.dumps(corrections)}
    assert imported(stored, cfg).title == "Handyrechnung" and imported(stored, cfg, with_corrections=False).title \
        == "Rechnung April"
    draft.title, draft.date = "", "31.02.2023"
    assert set(import_corrections(row, draft, cfg)[1]) == {"title", "date"}
    # without (current) metadata every value is the user's, so older metadata cannot fill the gaps
    mine = imported(row, cfg, with_meta=False)
    mine.title = "Selbst"
    assert import_corrections(row, mine, cfg, with_meta=False)[0]["correspondent"] == ""


def test_control_puts_asked_documents_first():
    c = Control()
    assert not c.take_changed() and c.take_asked() is None
    c.ask("a")
    c.ask("b", force=True)
    c.ask("a", force=True)
    assert c.take_changed() and c.take_asked() == ("a", True) and c.take_asked() == ("b", True)
    c.pause(True)
    c.close()
    assert c.proceed() is False and c.idle() is False


def test_follow_the_import_and_accept_with_an_edit(db, cfg, mbox, fake_ollama, fake_paperless, monkeypatch):
    fp = fake_paperless
    app = ImportApp(cfg, "mbox", [mbox])
    opened = []
    monkeypatch.setattr("webbrowser.open", opened.append)

    async def steps(pilot):
        await until(pilot, settled(app))
        docs = by_name(app)
        assert docs[INVOICE].state == "ready" and docs["Versicherungsschein.pdf"].state == "ready"
        assert docs["AGB_Muster_Shop.pdf"].state == "irrelevant" and docs["Kontoauszug.pdf"].state == "skipped"
        assert docs["Produktinformationsblatt.pdf"].state == "irrelevant"  # after escalation
        assert not fp.docs  # nothing is uploaded before it is accepted
        screen = app.list_screen
        assert "Imports(mbox)[12]" in screen.query_one("#box").border_title

        screen.focus_item(docs[INVOICE].id)
        await pilot.press("o")  # the attachment itself
        assert opened and opened[0].startswith("file:") and opened[0].endswith(".pdf")
        await pilot.press("enter")
        await pilot.pause()
        doc = app.screen
        assert isinstance(doc, ImportDocScreen) and doc.doc_id == docs[INVOICE].id
        assert doc.widgets["title"].value == "Rechnung Mobilfunk April 2023"
        assert doc.widgets["field:Betrag"].value == "EUR44.15"
        trail = plain_steps(doc.trail)
        for step in ("Mail: ", "Telekom", "File: PDF", "Text: from the PDF", "Relevance: 97%", "invoice 90%",
                     "Decision: import (relevance ≥ 75%)", "Metadata: qwen3.8",
                     "correspondent as written: Telekom Deutschland GmbH", "Upload: when you accept"):
            assert step in trail, step
        assert "the same file came with 1 more mail" in trail  # the reminder mail
        assert "Rechnungsbetrag" in doc.text

        await pilot.press("e", *"Handyrechnung April")  # the title is selected: typing replaces it
        assert app.by_id[doc.doc_id].label == "edited"
        assert doc.row_of["title"].has_class("changed")
        assert "model: Rechnung Mobilfunk April 2023" in str(doc.row_of["title"].query_one(".was").render())
        await pilot.press("ctrl+s")  # accept: upload, then the next document to check
        assert doc.doc_id != docs[INVOICE].id
        await until(pilot, lambda: docs[INVOICE].state == "done")
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen
        screen.focus_item(docs[INVOICE].id)
        await pilot.press("enter")  # the trail now ends with the upload; the form is final
        await pilot.pause()
        doc = app.screen
        assert doc.doc_id == docs[INVOICE].id and doc.widgets["title"].disabled
        assert "Upload: Paperless #" in plain_steps(doc.trail) and "your corrections: title" in plain_steps(doc.trail)
        await pilot.press("escape")

    run(app, steps)
    up = uploads(fp)[INVOICE]
    assert up["title"] == "Handyrechnung April" and up["upload"]["created"] == ["2023-04-17"]
    tag_ids = {o["name"]: o["id"] for o in fp.objects["tags"]}
    assert {tag_ids["paperclerk"], tag_ids["mail-import"], tag_ids["Mobilfunk"]} <= set(up["tags"])
    assert "paperclerk-review" not in tag_ids  # accepting is the review
    assert up["notes"][0]["note"].startswith("Mobilfunkrechnung")
    row = db.one("SELECT * FROM documents WHERE filename = ?", (INVOICE,))
    assert row["manual"] == "yes" and json.loads(row["overrides"]) == {"title": "Handyrechnung April"}
    assert row["upload_status"] == "done" and row["note_added"] == 1
    assert len(fp.docs) == 1


def test_reject_import_a_dropped_document_and_accept_all(db, cfg, mbox, fake_ollama, fake_paperless):
    fp = fake_paperless
    app = ImportApp(cfg, "mbox", [mbox])

    async def steps(pilot):
        await until(pilot, settled(app))
        docs = by_name(app)
        screen = app.list_screen
        screen.focus_item(docs["Bestellbestaetigung.pdf"].id)
        await pilot.press("x")  # relevant, but not wanted
        assert docs["Bestellbestaetigung.pdf"].state == "rejected"

        privacy = docs["Datenschutz.pdf"]  # a standard text: the model drops it
        assert privacy.state == "irrelevant"
        screen.focus_item(privacy.id)
        await pilot.press("a")  # import it anyway: the model is asked for its metadata first
        assert privacy.row["manual"] == "yes" and privacy.state in ("queued", "running", "ready")  # the fake is fast
        await until(pilot, lambda: privacy.state == "ready")
        assert privacy.draft.title == "Rechnung Mobilfunk April 2023"  # the fake model's answer

        ready = {i.filename for i in app.acceptable()}
        assert {INVOICE, "Versicherungsschein.pdf", "Datenschutz.pdf", "Abrechnung_2024_03.pdf"} <= ready  # ZIP
        await pilot.press("A")
        await pilot.pause()
        assert isinstance(app.screen, Confirm) and f"Upload {len(ready)} documents" in app.screen.question
        await pilot.press("y")
        await until(pilot, lambda: all(i.state in ("done", "exists") for i in app.items if i.filename in ready))
        assert app.acceptable() == []
        app.show_thresholds()
        await pilot.pause()
        assert isinstance(app.screen, Help)
        await pilot.press("escape")

    run(app, steps)
    names = set(uploads(fp))
    assert {"Datenschutz.pdf", "Versicherungsschein.pdf", INVOICE} <= names
    assert "Bestellbestaetigung.pdf" not in names and "AGB_Muster_Shop.pdf" not in names
    rows = {r["filename"]: r for r in db.q("SELECT * FROM documents")}
    assert rows["Bestellbestaetigung.pdf"]["manual"] == "no" and rows["Datenschutz.pdf"]["manual"] == "yes"
    report = evaluate(db, cfg)  # your decisions are the labels for checking the thresholds
    assert ", rejected 1 ==" in report and "Wrongly skipped: " in report and "(Datenschutz.pdf)" in report


def test_uncertain_documents_are_left_for_a_look(db, cfg, mbox, fake_ollama, fake_paperless):
    cfg.scoring.escalation_model = ""  # no second opinion: the product information sheet stays uncertain
    app = ImportApp(cfg, "mbox", [mbox])

    async def steps(pilot):
        await until(pilot, settled(app))
        sheet = by_name(app)["Produktinformationsblatt.pdf"]
        assert sheet.state == "uncertain" and sheet not in app.acceptable()
        await pilot.press("slash", *"uncertain", "enter")
        assert [i.id for i in app.list_screen.rows_shown] == [sheet.id]
        await pilot.press("A")
        await pilot.pause()
        assert "1 uncertain ones are left" in app.screen.question
        await pilot.press("n")

    run(app, steps)


def test_an_unavailable_model_pauses_the_pipeline(db, cfg, mbox, fake_ollama, fake_paperless):
    cfg.scoring.model = "not-installed"
    app = ImportApp(cfg, "mbox", [mbox])

    async def steps(pilot):
        await until(pilot, lambda: app.model_error)
        assert app.engine.jobs.paused and "ollama pull not-installed" in app.model_error
        await until(pilot, lambda: by_name(app) and by_name(app)[INVOICE].state == "queued")  # text, no relevance
        await pilot.press("q")
        await pilot.pause()

    run(app, steps)
    assert app.return_code == 0


def test_import_opens_the_view_only_in_a_terminal(cfg, mbox, monkeypatch):
    started = []
    monkeypatch.setattr("paperclerk.importui.run_import_ui", lambda *args: started.append(args) or 0)
    monkeypatch.setattr("paperclerk.cli.run_import", lambda *args: {})
    monkeypatch.setattr("paperclerk.cli.interactive", lambda: True)
    assert main(["-c", str(cfg.path), "import", "mbox", str(mbox)]) == 0
    assert started and started[0][1:] == ("mbox", [mbox], None)
    assert main(["-c", str(cfg.path), "import", "mbox", str(mbox), "--dry-run"]) == 0  # prints instead
    monkeypatch.setattr("paperclerk.cli.interactive", lambda: False)  # e.g. a scheduled task
    assert main(["-c", str(cfg.path), "import", "mbox", str(mbox)]) == 0
    assert len(started) == 1
    assert main(["-c", str(cfg.path), "import", "mbox", "missing.mbox"]) == 1
