"""The whole import with fake Ollama and fake Paperless (API v10)."""

from __future__ import annotations

import json

import openpyxl
import pytest

from conftest import TESSDATA
from paperclerk.cli import main
from paperclerk.importer import run_import
from paperclerk.ingest import ingest_mbox
from paperclerk.review import apply_review, export_review, state
from paperclerk.textextract import run_extract


def _by_name(db):
    return {r["filename"]: r for r in db.q("SELECT * FROM documents")}


def test_text_extraction(db, cfg, mbox):
    ingest_mbox(db, cfg, mbox)
    run_extract(db, cfg)
    docs = _by_name(db)
    assert docs["Rechnung_2023_04.pdf"]["text_method"] == "pdf"
    assert "Rechnungsbetrag" in docs["Rechnung_2023_04.pdf"]["text"]
    assert docs["Kontoauszug.pdf"]["skip_reason"] == "password-protected PDF"
    if TESSDATA:
        assert docs["scan0001.pdf"]["text_method"] == "ocr"
        assert "Malerarbeiten" in docs["scan0001.pdf"]["text"]
        assert docs["IMG_2041.jpg"]["text_method"] == "ocr"
        assert "Kaffeevollautomat" in docs["IMG_2041.jpg"]["text"]
    else:
        assert docs["scan0001.pdf"]["text_method"] == "none"  # -> image to the model


@pytest.mark.skipif(not TESSDATA, reason="PAPERCLERK_TEST_TESSDATA not set")
def test_full_import(db, cfg, mbox, fake_ollama, fake_paperless, tmp_path, capsys):
    assert run_import(db, cfg, "mbox", [mbox], dry_run=True) == {}
    assert not fake_paperless.created and "would be uploaded" in capsys.readouterr().out
    docs = _by_name(db)
    states = {name: state(r, cfg) for name, r in docs.items()}
    assert states["Rechnung_2023_04.pdf"] == "relevant"
    assert states["scan0001.pdf"] == "relevant"
    assert states["AGB_Muster_Shop.pdf"] == "irrelevant"
    assert states["Widerrufsbelehrung.pdf"] == "irrelevant"
    assert states["Prospekt_Fruehling.pdf"] == "irrelevant"
    assert states["Produktinformationsblatt.pdf"] == "irrelevant"  # only after escalation
    assert docs["Produktinformationsblatt.pdf"]["model2"] == "pplx-decider"
    assert states["Kontoauszug.pdf"] == "skipped"

    assert len([c for c in fake_ollama.calls if c["path"] == "/v1/systemone"]) == 11 + 1
    chats = [c for c in fake_ollama.calls if c["path"] == "/api/chat"]
    assert len(chats) == 6  # metadata only for relevant documents
    system = chats[0]["messages"][0]["content"]
    assert "Telekom" in system and "info@o2.com" not in system  # address-like correspondents are not shown
    assert "Rechnung" in system and "Telekommunikation" in system  # document types and tags from config
    assert "Betrag" in chats[0]["format"]["properties"]["fields"]["properties"]

    meta = json.loads(docs["Rechnung_2023_04.pdf"]["meta"])
    assert meta["correspondent"] == "Telekom Deutschland" and meta["document_type"] == "Rechnung"
    assert meta["date"] == "2023-04-17" and meta["date_source"] == "document"
    assert meta["fields"] == {"Betrag": "EUR44.15", "Rechnungsnummer": "4400 1234 5678", "Fällig am": "2023-04-28"}
    assert meta["tags"] == ["Telekommunikation", "Mobilfunk"] and meta["summary"]

    # review: one correction and one manual decision
    xlsx = tmp_path / "review.xlsx"
    assert export_review(db, cfg, xlsx) == 12
    wb = openpyxl.load_workbook(xlsx)
    ws = wb["Documents"]
    header = [c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2):
        name = row[header.index("File name")].value
        if name == "Rechnung_2023_04.pdf":
            row[header.index("Correspondent")].value = "Telekom"
        if name == "Datenschutz.pdf":
            row[header.index("Manual")].value = "yes"
    wb.save(xlsx)
    stats = apply_review(db, cfg, xlsx)
    assert stats["corrected fields"] == 1 and stats["manual: yes"] == 1

    counts = run_import(db, cfg, "mbox", [mbox])  # nothing new to read; enrich Datenschutz.pdf, upload
    assert counts["done"] == 7 and counts["failed"] == 0
    docs = _by_name(db)
    assert docs["AGB_Muster_Shop.pdf"]["upload_status"] is None
    fp = fake_paperless
    tag_ids = {o["name"]: o["id"] for o in fp.objects["tags"]}
    inv = next(d for d in fp.docs.values() if d["original_file_name"] == "Rechnung_2023_04.pdf")
    up = inv["upload"]
    assert up["title"] == ["Rechnung Mobilfunk April 2023"] and up["created"] == ["2023-04-17"]
    assert up["correspondent"] == ["1"] and up["document_type"] == ["1"]  # existing Telekom / Rechnung
    assert {tag_ids["paperclerk"], tag_ids["mail-import"], tag_ids["Mobilfunk"]} <= set(inv["tags"])
    created = {(kind, d["name"]): d for kind, d in fp.created}
    assert created[("tags", "paperclerk")]["matching_algorithm"] == 0
    assert created[("custom_fields", "Fällig am")]["data_type"] == "date"
    fields = {f["field"]: f["value"] for f in inv["custom_fields"]}
    assert fields[1] == "EUR44.15"  # existing custom field "Betrag"
    assert inv["notes"][0]["note"].startswith("Mobilfunkrechnung") and "Message-ID" in inv["notes"][0]["note"]

    # second run: nothing to do; duplicate detection by checksum
    assert run_import(db, cfg, "mbox", [mbox])["done"] == 0
    db.update_doc(docs["Rechnung_2023_04.pdf"]["sha256"], upload_status=None)
    db.commit()
    assert run_import(db, cfg, "mbox", [mbox])["exists"] == 1


def test_cli_status_and_show(db, cfg, mbox, capsys):
    ingest_mbox(db, cfg, mbox)
    run_extract(db, cfg)
    cfg_path = str(cfg.path)
    assert main(["-c", cfg_path, "status"]) == 0
    assert "12 unique documents" in capsys.readouterr().out
    sha = db.q("SELECT sha256 FROM documents WHERE filename = 'Rechnung_2023_04.pdf'")[0]["sha256"]
    assert main(["-c", cfg_path, "show", sha[:8]]) == 0
    assert "Rechnungsbetrag" in capsys.readouterr().out
    assert main(["-c", cfg_path, "show", "999"]) == 1  # not processed Paperless document
