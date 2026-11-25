from __future__ import annotations

import json
from datetime import date

from paperclerk.decision import decision, effective, probability, relevance
from paperclerk.normalize import (CorrespondentNormalizer, TagNormalizer, parse_date, parse_field, pick_date,
                                  strip_legal)
from paperclerk.paperless import parse_task


def _row(**kw):
    base = {"noul1": None, "types1": None, "noul2": None, "types2": None, "manual": None, "overrides": None,
            "meta": None}
    base.update(kw)
    return base


def test_relevance_combines_yes_no_and_category_mass(cfg):
    types = json.dumps({"invoice": 0.8, "terms": 0.2})
    assert relevance(0.9, types, cfg) == 0.5 * 0.9 + 0.5 * 0.8
    cfg.scoring.type_weight = 0
    assert relevance(0.9, types, cfg) == 0.9


def test_decision_thresholds_escalation_and_manual(cfg):
    sure, junk = json.dumps({"invoice": 1.0}), json.dumps({"terms": 1.0})
    assert decision(_row(noul1=0.99, types1=sure), cfg) == "relevant"
    assert decision(_row(noul1=0.01, types1=junk), cfg) == "irrelevant"
    unsure = _row(noul1=0.9, types1=junk)  # yes/no answer and category disagree -> middle
    assert decision(unsure, cfg) == "escalate"
    assert decision({**unsure, "noul2": 0.05, "types2": junk}, cfg) == "irrelevant"
    assert decision({**unsure, "noul2": 0.6, "types2": junk}, cfg) == "review"
    cfg.scoring.escalation_model = ""
    assert decision(unsure, cfg) == "review"
    assert decision({**unsure, "manual": "yes"}, cfg) == "relevant"
    assert decision({**unsure, "manual": "yes"}, cfg, use_manual=False) == "review"
    assert probability(_row(), cfg) is None
    assert decision(_row(), cfg) == "pending"


def test_effective_prefers_corrections():
    row = _row(meta=json.dumps({"title": "Model title", "tags": ["A"]}), overrides=json.dumps({"title": "Mine"}))
    assert effective(row, "title") == "Mine"
    assert effective(row, "tags") == ["A"]
    assert effective(row, "fields") == {} and effective(row, "correspondent") == ""


def test_correspondent_normalizer():
    assert strip_legal("Telekom Deutschland GmbH") == "Telekom Deutschland"
    assert strip_legal("Muster GmbH & Co. KG") == "Muster"
    assert strip_legal("Allianz Versicherungs-AG") == "Allianz Versicherungs-AG"
    n = CorrespondentNormalizer(["Telekom", "Deutsche Bank", "Stadtwerke München", "info@o2.com"],
                                {"Telekom Deutschland GmbH": "Telekom"})
    assert n("Telekom Deutschland GmbH") == "Telekom"  # alias
    assert n("TELEKOM") == "Telekom"
    assert n("Deutsche Bank AG") == "Deutsche Bank"
    assert n("Deutsche Bahn") == "Deutsche Bahn"  # similar, but not the same
    assert n("Stadtwerke Münster") == "Stadtwerke Münster"
    assert n("Stadtwerke Muenchen") == "Stadtwerke München"  # tolerates spelling variants
    assert n("Malerbetrieb Schulze GmbH") == "Malerbetrieb Schulze"
    assert n("Malerbetrieb Schulze") == "Malerbetrieb Schulze"  # learnt, same spelling
    assert n("info@o2.com") == "O2"  # address-like Paperless names are not reused
    assert n("") == ""


def test_correspondent_from_email_address():
    known = CorrespondentNormalizer(["EnBW", "Amazon"])
    assert known("noreply-mobility@enbw.com") == "EnBW"
    assert known("amazon.de") == "Amazon"
    assert known("rechnung@mail.amazon.co.uk") == "Amazon"
    fresh = CorrespondentNormalizer([])
    assert fresh("noreply-mobility@enbw.com", fallback="EnBW mobility+") == "EnBW mobility+"
    assert fresh("info@stadtwerke-musterstadt.de", fallback="noreply") == "Stadtwerke-musterstadt"
    assert fresh("Telekom", fallback="egal") == "Telekom"


def test_tag_normalizer():
    tags = TagNormalizer(["gesundheit", "Versicherung"], allow_new=True, blocked={"paperclerk", "new"})
    assert tags(["Gesundheit", "versicherungen", "Strom", "paperclerk", "#Auto", "Steuer"]) == \
        ["gesundheit", "Versicherung", "Strom"]  # existing spelling, fuzzy match, new tag, max 3
    strict = TagNormalizer(["gesundheit"], allow_new=False, blocked=set())
    assert strict(["Gesundheit", "Strom"]) == ["gesundheit"]


def test_parse_field():
    assert parse_field("monetary", "EUR 1234.56") == "EUR1234.56"
    assert parse_field("monetary", "1.250,00 €") == "EUR1250.00"
    assert parse_field("monetary", "CHF 12") == "CHF12.00"
    assert parse_field("date", "28.04.2023") == "2023-04-28"
    assert parse_field("integer", "1.234") == 1  # dot = decimal separator as instructed
    assert parse_field("float", "89,45") == 89.45
    assert parse_field("string", "  4400 1234  ") == "4400 1234"
    assert parse_field("boolean", "ja") is True
    assert parse_field("url", "keine") is None
    assert parse_field("string", "") is None and parse_field("monetary", "unbekannt") is None


def test_dates():
    assert parse_date("17.04.2023") == date(2023, 4, 17)
    assert parse_date("2023-4-7") == date(2023, 4, 7)
    assert parse_date("31.02.2023") is None
    ref = "2023-04-18T08:00:00+00:00"
    assert pick_date("2023-04-17", ref) == ("2023-04-17", "document")
    assert pick_date("2023-09-30", ref) == ("2023-04-18", "reference")  # long after the mail: due date
    assert pick_date("", ref) == ("2023-04-18", "reference")
    assert pick_date("2023-09-30", ref, fallback_to_reference=False) == ("", "")
    assert pick_date("1899-01-01", "") == ("", "")


def test_parse_task_v9_v10_and_legacy():
    v10_ok = parse_task({"task_id": "x", "status": "success", "result_data": {"document_id": 42},
                         "related_document_ids": [42]})
    assert (v10_ok.status, v10_ok.document_id, v10_ok.duplicate_of) == ("SUCCESS", 42, None)
    v10_dup = parse_task({"status": "failure", "result_data": {"duplicate_of": 7}, "related_document_ids": [7]})
    assert (v10_dup.status, v10_dup.document_id, v10_dup.duplicate_of) == ("FAILURE", None, 7)
    v9_ok = parse_task({"status": "SUCCESS", "result": "Success. New document id 12 created", "related_document": "12"})
    assert v9_ok.document_id == 12 and v9_ok.finished
    v9_dup = parse_task({"status": "FAILURE", "result": "Not consuming: It is a duplicate of document #12",
                         "related_document": 12})
    assert v9_dup.duplicate_of == 12 and v9_dup.document_id is None
    legacy = parse_task({"status": "FAILURE", "result": "a.pdf: Not consuming a.pdf: It is a duplicate of Rechnung (#31)"})
    assert legacy.duplicate_of == 31
    assert not parse_task({"status": "STARTED"}).finished
