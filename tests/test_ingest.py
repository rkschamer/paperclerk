from __future__ import annotations

from collections import Counter

from paperclerk import mail, store
from paperclerk.ingest import ingest_imap, ingest_mbox
from paperclerk.sources import GmailImap, iter_mbox
from samples import INVOICE, build_messages, text_pdf, write_mbox, mail as make_mail


def _skips(db):
    return Counter(r["skip_reason"] for r in db.q("SELECT skip_reason FROM attachments WHERE skip_reason IS NOT NULL"))


def test_ingest_mbox_dedup_and_prefilter(db, cfg, mbox):
    stats = ingest_mbox(db, cfg, mbox)
    # 13 mails: spam and the mail without attachment are not stored
    assert stats.messages == 13
    assert db.one("SELECT COUNT(*) n FROM messages")["n"] == 11
    docs = {r["filename"]: r for r in db.q("SELECT * FROM documents")}
    assert set(docs) == {
        "Rechnung_2023_04.pdf", "AGB_Muster_Shop.pdf", "Widerrufsbelehrung.pdf", "Bestellbestaetigung.pdf",
        "scan0001.pdf", "Versicherungsschein.pdf", "Produktinformationsblatt.pdf", "Prospekt_Fruehling.pdf",
        "Datenschutz.pdf", "IMG_2041.jpg", "Abrechnung_2024_03.pdf", "Kontoauszug.pdf",
    }
    assert docs["scan0001.pdf"]["kind"] == "pdf"  # declared as octet-stream, recognised by content
    assert docs["IMG_2041.jpg"]["kind"] == "image"
    assert (cfg.files_dir / docs["Rechnung_2023_04.pdf"]["path"]).is_file()
    assert len(db.mails_for(docs["Rechnung_2023_04.pdf"]["sha256"])) == 2  # the same invoice in two mails
    assert len(db.mails_for(docs["AGB_Muster_Shop.pdf"]["sha256"])) == 2

    skips = _skips(db)
    assert skips["inline image"] == 1
    assert skips["ZIP archive (unpacked)"] == 1
    assert any(k.startswith("type application/pkcs7-signature") for k in skips)


def test_ingest_is_resumable_and_idempotent(db, cfg, mbox):
    first = ingest_mbox(db, cfg, mbox, limit=5)
    assert first.messages == 5
    assert not db.one("SELECT * FROM sources")["done"]
    ingest_mbox(db, cfg, mbox)
    assert db.one("SELECT done FROM sources")["done"] == 1
    n_docs = db.one("SELECT COUNT(*) n FROM documents")["n"]
    n_att = db.one("SELECT COUNT(*) n FROM attachments")["n"]
    assert ingest_mbox(db, cfg, mbox).messages == 0  # already complete
    ingest_mbox(db, cfg, mbox, rescan=True)
    assert db.one("SELECT COUNT(*) n FROM documents")["n"] == n_docs
    assert db.one("SELECT COUNT(*) n FROM attachments")["n"] == n_att


def test_iter_mbox_ignores_from_in_body(tmp_path):
    msg = make_mail("a@b.de", "Test", "2024-01-01T00:00:00", "Text\n\nFrom the desk of Max\nweiter",
                    [(text_pdf(INVOICE), "application", "pdf", "r.pdf")])
    path = write_mbox(tmp_path / "x.mbox", [msg, msg])
    found = list(iter_mbox(path))
    assert len(found) == 2
    meta, parts = mail.parse(found[0][1])
    assert meta.subject == "Test" and "From the desk" in meta.body
    assert [p.filename for p in parts] == ["r.pdf"]


def test_encoded_filename_and_body():
    raw = (
        "From: =?utf-8?q?M=C3=BCller_GmbH?= <info@mueller.de>\r\n"
        "Subject: =?utf-8?q?Ihre_Rechnung_f=C3=BCr_M=C3=A4rz?=\r\n"
        "Date: Tue, 05 Mar 2024 10:00:00 +0100\r\n"
        "MIME-Version: 1.0\r\n"
        'Content-Type: multipart/mixed; boundary="b"\r\n\r\n'
        "--b\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<p>Hallo&nbsp;<b>Welt</b> https://x.de/track</p>\r\n"
        "--b\r\nContent-Type: application/octet-stream\r\n"
        "Content-Disposition: attachment; filename*=utf-8''Rechnung%20M%C3%A4rz.pdf\r\n"
        "Content-Transfer-Encoding: base64\r\n\r\nJVBERi0xLjQK\r\n--b--\r\n"
    ).encode()
    meta, parts = mail.parse(raw)
    assert meta.from_name == "Müller GmbH" and meta.from_addr == "info@mueller.de"
    assert meta.subject == "Ihre Rechnung für März"
    assert meta.date.startswith("2024-03-05T10:00:00")
    assert meta.body == "Hallo Welt [Link]"
    assert parts[0].filename == "Rechnung März.pdf"
    assert store.sniff(parts[0].data).kind == "pdf"


class FakeImapConn:
    """Answers UID SEARCH / UID FETCH like Gmail."""

    def __init__(self, messages: dict[int, bytes]):
        self.messages = messages
        self.commands: list[tuple] = []

    def uid(self, command, *args):
        self.commands.append((command, *args))
        if command == "SEARCH":
            return "OK", [b" ".join(str(u).encode() for u in self.messages)]
        uid = int(args[0])
        raw = self.messages[uid]
        header = f'1 (X-GM-LABELS ("\\\\Inbox" Rechnungen) UID {uid} BODY[] {{{len(raw)}}}'.encode()
        return "OK", [(header, raw), b")"]

    def logout(self):
        pass


def test_ingest_imap_with_fake_gmail(db, cfg, monkeypatch):
    msgs = build_messages()
    for m in msgs:
        del m["X-Gmail-Labels"]  # only present in Takeout exports
    conn = FakeImapConn({10 + i: m.as_bytes() for i, m in enumerate(msgs[:3])})

    def connect(self):
        self.conn, self.folder, self.uidvalidity = conn, "[Gmail]/Alle Nachrichten", "7"

    monkeypatch.setattr(GmailImap, "connect", connect)
    cfg.imap.user, cfg.imap.password = "max@gmail.com", "app-password"
    stats = ingest_imap(db, cfg)
    assert stats.messages == 3 and stats.new_docs == 4  # invoice, terms, withdrawal policy, order confirmation
    assert ("SEARCH", "X-GM-RAW", '"has:attachment"') in conn.commands
    assert db.one("SELECT cursor FROM sources")["cursor"] == "7:12"
    assert db.one("SELECT labels FROM messages WHERE source_ref = '7:10'")["labels"] == "Inbox,Rechnungen"

    conn.messages[13] = msgs[3].as_bytes()  # a new mail since the last run
    again = ingest_imap(db, cfg)
    assert again.messages == 1 and again.new_docs == 1
