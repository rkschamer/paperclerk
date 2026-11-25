"""`paperclerk process` against a fake Paperless."""

from __future__ import annotations

from paperclerk.process import run_process
from samples import INVOICE, scanned_pdf
from test_merge import CONTRACT


def _setup(fp):
    # inbox document with Paperless' own guesses: address as correspondent, file name as title
    fp.add_document(10, "scan_2023_04", INVOICE, tags=[1], correspondent=2, created="2023-04-30")
    # reviewed document outside the inbox
    fp.add_document(11, "Mein Vertrag", "\n".join(CONTRACT), tags=[2], document_type=1,
                    original_file_name="vertrag_scan.pdf")


def test_dry_run_then_apply(db, cfg, fake_ollama, fake_paperless, capsys):
    fp = fake_paperless
    _setup(fp)
    counts = run_process(db, cfg, dry_run=True)
    assert counts == {"suggested": 1}  # scope inbox: only document 10
    out = capsys.readouterr().out
    assert "info@o2.com -> Telekom Deutschland" in out and "2023-04-30 -> 2023-04-17" in out
    assert not fp.patches and not fp.created
    chats = [c for c in fake_ollama.calls if c["path"] == "/api/chat"]
    assert len(chats) == 1 and "Added to Paperless: 2026-11-20" in chats[0]["messages"][1]["content"]

    assert run_process(db, cfg) == {"applied": 1}
    assert len([c for c in fake_ollama.calls if c["path"] == "/api/chat"]) == 1  # stored suggestion reused
    doc = fp.docs[10]
    tags = {o["id"]: o["name"] for o in fp.objects["tags"]}
    corr = {o["id"]: o["name"] for o in fp.objects["correspondents"]}
    assert doc["title"] == "Rechnung Mobilfunk April 2023" and doc["created"] == "2023-04-17"
    assert corr[doc["correspondent"]] == "Telekom Deutschland" and doc["document_type"] == 1
    assert [tags[t] for t in doc["tags"]] == ["new", "Telekommunikation", "Mobilfunk", "paperclerk"]
    fields = {f["field"]: f["value"] for f in doc["custom_fields"]}
    assert fields[1] == "EUR44.15" and len(fields) == 3
    assert len(doc["notes"]) == 1 and doc["notes"][0]["note"].startswith("Mobilfunkrechnung")
    assert all(kind != "document_types" for kind, _ in fp.created)  # "Rechnung" existed

    assert run_process(db, cfg) == {}  # processed tag set: nothing left

    # retrigger by removing the processed tag: new suggestion, the old note is replaced
    doc["tags"] = [t for t in doc["tags"] if tags[t] != "paperclerk"]
    assert run_process(db, cfg) == {"applied": 1}
    assert len([c for c in fake_ollama.calls if c["path"] == "/api/chat"]) == 2
    assert len(doc["notes"]) == 1 and doc["notes"][0]["id"] == 2


def test_outside_inbox_only_gaps_are_filled(db, cfg, fake_ollama, fake_paperless):
    fp = fake_paperless
    _setup(fp)
    cfg.process.scope = "all"
    assert run_process(db, cfg) == {"applied": 2}
    doc = fp.docs[11]
    corr = {o["id"]: o["name"] for o in fp.objects["correspondents"]}
    assert doc["title"] == "Mein Vertrag"  # set by the user, kept
    assert doc["document_type"] == 1 and doc["created"] == "2026-01-01"  # kept
    assert corr[doc["correspondent"]] == "Notar Andreas Kaiser"  # was empty: filled
    fields = {o["id"]: o["name"] for o in fp.objects["custom_fields"]}
    assert {fields[f["field"]]: f["value"] for f in doc["custom_fields"]} == {"Vertragsnummer": "1234/2026"}


def test_model_change_and_current_title(db, cfg, fake_ollama, fake_paperless):
    fake_paperless.add_document(10, "Rechnung Ernährungsberatung", INVOICE, tags=[1], original_file_name="scan.pdf")

    def chats():
        return [c for c in fake_ollama.calls if c["path"] == "/api/chat"]

    run_process(db, cfg, dry_run=True)
    assert "Current title in Paperless (often the mail subject or an earlier guess): Rechnung Ernährungsberatung" \
        in chats()[0]["messages"][1]["content"]
    run_process(db, cfg, dry_run=True)
    assert len(chats()) == 1  # same model: the stored suggestion is reused
    cfg.enrich.model = "qwen3.6:35b"
    run_process(db, cfg, dry_run=True)
    assert len(chats()) == 2 and chats()[1]["model"] == "qwen3.6:35b"


def test_long_text_shows_beginning_and_end(cfg):
    from paperclerk.enrich import Enricher, Source, Vocabulary

    enricher = Enricher(cfg, Vocabulary([], [], ["Rechnung"], [], []), ollama=None)
    text = "A" * 10000 + "RECHNUNGSENDE"
    content = enricher.message(Source(text=text, filename="x.pdf"))["content"]
    assert "beginning and end" in content and content.endswith("RECHNUNGSENDE")
    assert len(content) < cfg.enrich.max_chars + 200


def test_document_without_text_goes_to_the_model_as_image(db, cfg, fake_ollama, fake_paperless):
    fp = fake_paperless
    fp.add_document(12, "scan", "", tags=[1])
    fp.files[12] = scanned_pdf("RECHNUNG Nr. 1\nGesamtbetrag 10,00 EUR")
    assert run_process(db, cfg) == {"applied": 1}
    chat = next(c for c in fake_ollama.calls if c["path"] == "/api/chat")
    assert chat["messages"][1]["images"] and "attached as an image" in chat["messages"][1]["content"]
