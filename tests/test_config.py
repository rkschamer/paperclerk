"""Configuration loading and migration from maildocs."""

from __future__ import annotations

import json
import sqlite3

import pytest

from paperclerk.cli import main
from paperclerk.config import DEFAULT_CATEGORIES, LEGACY_CATEGORY_KEYS, ConfigError, convert_legacy, load
from paperclerk.db import DB

RELEVANT_LEGACY = list(LEGACY_CATEGORY_KEYS)[:14]


def legacy_config(extra_doctype: bool = False) -> str:
    doctypes = "".join(
        f'\n[[doctype]]\nkey = "{k}"\npaperless = ""\nrelevant = {"true" if k in RELEVANT_LEGACY else "false"}\n'
        f'description = "Beschreibung {k}"\n' for k in LEGACY_CATEGORY_KEYS)
    if extra_doctype:
        doctypes += '\n[[doctype]]\nkey = "steuer"\npaperless = "Steuer"\nrelevant = true\ndescription = "Steuer"\n'
    return f"""
[paths]
data_dir = "data"
[mbox]
files = ["D:/Takeout/all.mbox"]
skip_labels = ["Spam", "Trash", "Papierkorb"]
[text]
ocr = false
[decider]
model = "pplx-decider"
escalation_model = ""
accept = 0.8
[extractor]
model = "qwen3.8"
tags = ["Steuer", "Hund"]
[paperless]
url = "http://localhost:8000"
token = "0123456789abcdef"
import_tags = ["gmail-import"]
review_tag = "needs-review"
auto_matching = true
[aliases]
"Telekom Deutschland GmbH" = "Telekom"
{doctypes}"""


def test_convert_legacy_config(tmp_path):
    text, notes = convert_legacy(legacy_config())
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    cfg = load(path)
    assert cfg.paperless.url == "http://localhost:8000" and cfg.paperless.token == "0123456789abcdef"
    assert cfg.imports.tags == ["gmail-import"] and cfg.imports.review_tag == "needs-review"
    assert cfg.scoring.model == "pplx-decider" and cfg.scoring.accept == 0.8 and cfg.scoring.escalation_model == ""
    assert cfg.text.ocr is False
    assert "Hund" in cfg.enrich.tags and "Energie" in cfg.enrich.tags  # own tags added to the defaults
    assert cfg.aliases == {"Telekom Deutschland GmbH": "Telekom"}
    assert [c.key for c in cfg.categories] == [c.key for c in DEFAULT_CATEGORIES]  # default list -> new defaults
    assert any("import mbox" in n for n in notes)


def test_customised_categories_are_kept(tmp_path):
    text, notes = convert_legacy(legacy_config(extra_doctype=True))
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    cfg = load(path)
    assert cfg.category("invoice").description == "Beschreibung rechnung"
    assert cfg.category("steuer").relevant
    assert any("kept" in n for n in notes)


def test_init_converts_in_place(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(legacy_config(), encoding="utf-8")
    with pytest.raises(ConfigError, match="paperclerk init"):
        load(path)
    assert main(["-c", str(path), "init"]) == 0
    assert (tmp_path / "config.maildocs.toml").exists()
    assert load(path).paperless.token == "0123456789abcdef"


def test_db_migration_from_maildocs(tmp_path):
    old = tmp_path / "maildocs.sqlite"
    con = sqlite3.connect(old)
    con.executescript("""
        CREATE TABLE documents (sha256 TEXT PRIMARY KEY, md5 TEXT, types1 TEXT, types2 TEXT, manual TEXT,
                                overrides TEXT, skip_reason TEXT, meta TEXT);
        CREATE TABLE attachments (id INTEGER PRIMARY KEY, message_id INTEGER, part TEXT, sha256 TEXT,
                                  skip_reason TEXT);
    """)
    con.execute("INSERT INTO documents VALUES ('a', 'm', ?, NULL, 'ja', ?, 'Art ''office'' deaktiviert', NULL)",
                (json.dumps({"rechnung": 0.9, "agb": 0.1}), json.dumps({"korrespondent": "X", "titel": "T"})))
    con.execute("INSERT INTO attachments VALUES (1, 1, '0', NULL, 'kleines Bild')")
    con.commit()
    con.close()

    db = DB(tmp_path / "paperclerk.sqlite", legacy=old)
    row = db.one("SELECT * FROM documents")
    assert json.loads(row["types1"]) == {"invoice": 0.9, "terms": 0.1}
    assert row["manual"] == "yes" and json.loads(row["overrides"]) == {"correspondent": "X", "title": "T"}
    assert row["skip_reason"] == "kind 'office' disabled" and row["meta_version"] is None
    assert db.one("SELECT skip_reason FROM attachments")["skip_reason"] == "small image"
    assert db.q("SELECT * FROM processed") == []
    db.close()
    untouched = sqlite3.connect(old).execute("SELECT manual FROM documents").fetchone()[0]
    assert untouched == "ja"
