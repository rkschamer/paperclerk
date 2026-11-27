"""Merging split documents and detecting content duplicates (import)."""

from __future__ import annotations

import json

import pymupdf
import pytest

from paperclerk.decision import state
from paperclerk.importer import run_import
from paperclerk.ingest import ingest_mbox
from paperclerk.merge import run_merge, split_name
from paperclerk.scoring import run_scoring
from paperclerk.textextract import run_extract
from paperclerk.upload import run_upload
from samples import mail, text_pdf, write_mbox

CONTRACT = [
    "UR-Nr. 1234/2026\nÜberlassungsvertrag\nVerhandelt zu Musterstadt am 04.11.2026. Vor mir, dem Notar Andreas "
    "Kaiser mit dem Amtssitz in Musterstadt, erschienen heute Herr Reiner Fleischhauer, geboren am 12.03.1950, und "
    "Herr Max Mustermann, geboren am 01.02.1980, beide wohnhaft in Musterstadt. Die Erschienenen erklärten zur "
    "Beurkundung den folgenden Überlassungsvertrag. Seite 1 von 3",
    "Überlassungsvertrag, Seite 2 von 3\n§ 1 Grundbesitz: Im Grundbuch von Musterstadt Blatt 4711 ist das "
    "Grundstück Flurstück 815/3 mit einer Größe von 1.250 Quadratmetern eingetragen. Der Veräußerer überlässt "
    "dieses Grundstück mit allen Rechten und Pflichten dem Erwerber. § 2 Gegenleistung: Der Erwerber räumt dem "
    "Veräußerer ein lebenslanges Wohnungsrecht an der Wohnung im Erdgeschoss ein.",
    "Überlassungsvertrag, Seite 3 von 3\n§ 3 Waldflächen: Mitüberlassen werden die Waldflächen Flurstück 902 und "
    "903 mit zusammen 34.500 Quadratmetern sowie der Anteil an der Waldgenossenschaft. § 4 Kosten: Die Kosten "
    "dieses Vertrags und seines Vollzugs trägt der Erwerber. Vorgelesen, genehmigt und eigenhändig unterschrieben.",
]

BOILERPLATE = ("Bitte überweisen Sie den Rechnungsbetrag innerhalb von vierzehn Tagen auf das unten genannte "
               "Konto. Bei Fragen zu Ihrer Abrechnung erreichen Sie unseren Kundenservice montags bis freitags "
               "zwischen acht und achtzehn Uhr. Informationen zu Ihrem Tarif und zum Datenschutz finden Sie in "
               "unserem Kundenportal. Vielen Dank, dass Sie sich für die Stadtwerke Musterstadt entschieden haben.")


def invoice(number: str, date: str, kwh: int, amount: str) -> str:
    return (f"Stadtwerke Musterstadt GmbH\nRechnung Nr. {number} vom {date}\nKundennummer 556677\n"
            f"Verbrauch {kwh} kWh, Rechnungsbetrag {amount} EUR\n{BOILERPLATE}")


INVOICE_OCT = invoice("2026-0815", "01.10.2026", 245, "89,45")
INVOICE_NOV = invoice("2026-0915", "01.11.2026", 311, "112,80")


@pytest.fixture
def split_mbox(tmp_path):
    atts = [(text_pdf(p), "application", "pdf", f"Vertrag vom 04.11.2026 {i}.pdf") for i, p in enumerate(CONTRACT, 1)]
    atts += [(text_pdf(INVOICE_OCT), "application", "pdf", "Rechnung 1.pdf"),
             (text_pdf(INVOICE_NOV), "application", "pdf", "Rechnung 2.pdf")]
    msgs = [mail("Notariat <notar@example.org>", "Überlassungsvertrag von Notar", "2026-11-08T18:38:00",
                 "Anbei der Vertragsentwurf und die Rechnungen.", atts)]
    return write_mbox(tmp_path / "split.mbox", msgs)


def _docs(db):
    return {r["filename"]: r for r in db.q("SELECT * FROM documents")}


def test_split_name():
    assert split_name("Vertrag vom 04.11.2026 12.pdf") == ("vertrag vom 04.11.2026", 12)
    assert split_name("Vertrag vom 04.11.2026.pdf") == ("vertrag vom 04.11.2026", None)
    assert split_name("scan0003.pdf") == ("scan", 3)
    assert split_name("Seite (2).pdf") == ("seite", 2)
    assert split_name("Rechnung_2026.pdf") == ("rechnung_2026", None)  # a year, not a page number


def test_merge_detects_split_contract_but_keeps_invoices_apart(db, cfg, split_mbox, fake_ollama):
    ingest_mbox(db, cfg, split_mbox)
    run_extract(db, cfg)
    assert run_merge(db, cfg) == 1
    docs = _docs(db)
    merged = docs["Vertrag vom 04.11.2026.pdf"]
    assert merged["pages"] == 3 and len(json.loads(merged["parts"])) == 3
    assert merged["text"].index("Seite 1") < merged["text"].index("Seite 3")
    with pymupdf.open(cfg.files_dir / merged["path"]) as pdf:
        assert pdf.page_count == 3 and "Seite 1 von 3" in pdf[0].get_text()
    assert all(state(docs[f"Vertrag vom 04.11.2026 {i}.pdf"], cfg) == "part" for i in (1, 2, 3))
    assert docs["Rechnung 1.pdf"]["part_of"] is None
    assert db.one("SELECT COUNT(*) n FROM merge_checks")["n"] == 3
    assert db.mails_for(merged["sha256"])[0]["subject"] == "Überlassungsvertrag von Notar"

    calls = len(fake_ollama.calls)
    assert run_merge(db, cfg) == 0 and len(fake_ollama.calls) == calls  # stored answers are reused

    run_scoring(db, cfg)
    scored = [c for c in fake_ollama.calls[calls:] if c["path"] == "/v1/systemone"]
    assert len(scored) == 3  # merged contract + two invoices, no parts
    contract = next(c for c in scored if "File name: Vertrag vom 04.11.2026.pdf" in c["state"])
    assert "PDF, 3 pages" in contract["state"] and "merged from 3 separate attachments" in contract["state"]


def test_content_duplicates_are_not_uploaded(db, cfg, split_mbox, fake_ollama, fake_paperless):
    # Paperless already has the contract (merged by hand, a different file) and the October invoice
    fake_paperless.add_document(324, "Überlassungsvertrag_Fleischhauer", "\n\n".join(CONTRACT).replace("\n", " "))
    fake_paperless.add_document(325, "Rechnung Oktober", INVOICE_OCT)
    counts = run_import(db, cfg, "mbox", [split_mbox])
    docs = _docs(db)
    assert counts["similar"] == 2 and counts["done"] == 1
    contract = docs["Vertrag vom 04.11.2026.pdf"]
    assert contract["upload_status"] == "similar" and contract["paperless_id"] == 324
    assert "#324" in contract["upload_error"]
    assert docs["Rechnung 1.pdf"]["paperless_id"] == 325
    # same template, but different numbers and amounts -> not a duplicate
    assert docs["Rechnung 2.pdf"]["upload_status"] == "done"
    assert any("Überlassungsvertrag" in q for q in fake_paperless.searches)

    forced = run_upload(db, cfg, ids=[docs["Rechnung 1.pdf"]["sha256"][:10]], content_check=False)
    assert forced["done"] == 1
