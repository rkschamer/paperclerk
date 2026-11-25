"""Import review: export decisions and metadata as XLSX, read corrections back, evaluate thresholds."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from sqlite3 import Row

from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from .config import Config
from .db import DB
from .decision import IRRELEVANT, RELEVANT, REVIEW, decision, effective, overrides, probability, relevance
from .normalize import parse_date

log = logging.getLogger(__name__)

SHEET = "Documents"
SKIPPED, NEW, PART = "skipped", "new", "part"
COLUMNS = [
    ("ID", 14), ("State", 11), ("Manual", 9), ("Relevance %", 11), ("Document type", 16), ("Correspondent", 24),
    ("Date", 11), ("Title", 40), ("Tags", 20), ("File", 7), ("File name", 30), ("Mail date", 11), ("From", 28),
    ("Subject", 40), ("Pages", 7), ("Text", 7), ("Upload", 10), ("Paperless ID", 12), ("Note", 26), ("Excerpt", 60),
]
EDITABLE = {"Document type": "document_type", "Correspondent": "correspondent", "Date": "date", "Title": "title",
            "Tags": "tags"}
FILLS = {RELEVANT: "C6EFCE", IRRELEVANT: "FFC7CE", REVIEW: "FFEB9C"}


def state(row: Row, cfg: Config) -> str:
    if row["part_of"]:
        return PART
    if row["skip_reason"]:
        return SKIPPED
    if row["text_method"] is None:
        return NEW
    return decision(row, cfg)


def _xl(value: object) -> object:
    return ILLEGAL_CHARACTERS_RE.sub("", value) if isinstance(value, str) else value


def _note(row: Row) -> str:
    if row["part_of"]:
        return f"part of {row['part_of'][:12]}"
    if row["parts"]:
        return f"merged from {len(json.loads(row['parts']))} attachments"
    return row["skip_reason"] or row["last_error"] or row["upload_error"] or ""


def _values(db: DB, row: Row, cfg: Config) -> dict:
    mails = db.mails_for(row["sha256"])
    m = mails[0] if mails else None
    p = probability(row, cfg)
    sender = ""
    if m:
        sender = f"{m['from_name']} <{m['from_addr'] or ''}>" if m["from_name"] else (m["from_addr"] or "")
    return {
        "ID": row["sha256"][:12],
        "State": state(row, cfg),
        "Manual": row["manual"] or "",
        "Relevance %": round(p * 100) if p is not None else None,
        "Document type": effective(row, "document_type"),
        "Correspondent": effective(row, "correspondent"),
        "Date": effective(row, "date"),
        "Title": effective(row, "title"),
        "Tags": ", ".join(effective(row, "tags")),
        "File": "open",
        "File name": row["filename"],
        "Mail date": (m["date"] or "")[:10] if m else "",
        "From": sender,
        "Subject": m["subject"] if m else "",
        "Pages": row["pages"],
        "Text": row["text_method"] or "",
        "Upload": row["upload_status"] or "",
        "Paperless ID": row["paperless_id"],
        "Note": _note(row),
        "Excerpt": " / ".join((row["text"] or "")[:400].split("\n")),
    }


def export_review(db: DB, cfg: Config, out: Path) -> int:
    rows = db.q("SELECT * FROM documents ORDER BY rowid")
    wb = Workbook()
    ws = wb.active
    ws.title = SHEET
    headers = [c for c, _ in COLUMNS] + [f"orig:{h}" for h in EDITABLE]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    link_col = headers.index("File") + 1
    for i, row in enumerate(rows, start=2):
        values = _values(db, row, cfg)
        ws.append([_xl(values[h]) for h, _ in COLUMNS] + [_xl(values[h]) for h in EDITABLE])
        link = ws.cell(row=i, column=link_col)
        link.hyperlink = (cfg.files_dir / row["path"]).as_uri()
        link.style = "Hyperlink"
        if fill := FILLS.get(values["State"]):
            ws.cell(row=i, column=2).fill = PatternFill("solid", start_color=fill)
    for idx, (_, width) in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width
    for idx in range(len(COLUMNS) + 1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(idx)].hidden = True
    dv = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"C2:C{max(2, len(rows) + 1)}")
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{len(rows) + 1}"
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return len(rows)


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


_YES = {"yes", "y", "x", "1", "true", "ja", "j", "relevant"}
_NO = {"no", "n", "0", "false", "nein", "irrelevant"}


def _manual(value: object) -> str | None | bool:
    v = _cell(value).casefold()
    if not v:
        return None
    if v in _YES:
        return "yes"
    if v in _NO:
        return "no"
    return False


def apply_review(db: DB, cfg: Config, path: Path) -> Counter:
    wb = load_workbook(path, data_only=True)
    ws = wb[SHEET] if SHEET in wb.sheetnames else wb.active
    rows = ws.iter_rows(values_only=True)
    header = [_cell(h) for h in next(rows)]
    col = {h: i for i, h in enumerate(header)}
    if "ID" not in col or "Manual" not in col:
        raise ValueError(f"{path.name}: columns 'ID' and 'Manual' not found")
    stats: Counter = Counter()
    for values in rows:
        ident = _cell(values[col["ID"]])
        if not ident:
            continue
        found = db.find_docs(ident)
        if len(found) != 1:
            stats["unknown ID"] += 1
            continue
        doc = found[0]
        manual = _manual(values[col["Manual"]])
        if manual is False:
            log.warning("%s: 'Manual' value %r not understood (use yes/no)", ident, values[col["Manual"]])
            manual = doc["manual"]
        over = overrides(doc)
        for head, key in EDITABLE.items():
            if head not in col or f"orig:{head}" not in col:
                continue
            new, old = _cell(values[col[head]]), _cell(values[col[f"orig:{head}"]])
            if new == old:
                continue
            if key == "tags":
                over[key] = [t.strip() for t in new.split(",") if t.strip()]
            elif key == "date":
                d = parse_date(new)
                if new and not d:
                    log.warning("%s: date %r not understood (YYYY-MM-DD or DD.MM.YYYY)", ident, new)
                    continue
                over[key] = d.isoformat() if d else ""
            else:
                over[key] = new
            stats["corrected fields"] += 1
        if manual != doc["manual"]:
            stats[f"manual: {manual or 'removed'}"] += 1
        db.update_doc(doc["sha256"], manual=manual, overrides=json.dumps(over, ensure_ascii=False) if over else None)
        stats["rows"] += 1
    db.commit()
    return stats


def evaluate(db: DB, cfg: Config) -> str:
    labeled = db.q("SELECT * FROM documents WHERE manual IN ('yes', 'no') AND noul1 IS NOT NULL")
    if not labeled:
        return ""
    every = db.q("SELECT * FROM documents WHERE noul1 IS NOT NULL AND skip_reason IS NULL AND part_of IS NULL")
    out: list[str] = []
    variants = [(f"Relevance ({labeled[0]['model1']})", lambda r: relevance(r["noul1"], r["types1"], cfg))]
    if any(r["noul2"] is not None for r in labeled):
        variants.append(("Relevance + escalation", lambda r: probability(r, cfg)))
    for name, fn in variants:
        pos = [fn(r) for r in labeled if r["manual"] == "yes"]
        neg = [fn(r) for r in labeled if r["manual"] == "no"]
        ps = [fn(r) for r in every]
        out += ["", f"== {name}: {len(pos)} marked relevant, {len(neg)} marked irrelevant =="]
        out.append("reject <= t:  share of all documents skipped | relevant ones wrongly skipped")
        for t in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            mark = "  <- current" if abs(t - cfg.scoring.reject) < 1e-9 else ""
            out.append(f"  {t:4.2f}:  {sum(p <= t for p in ps) / len(ps):6.1%}  |  {sum(p <= t for p in pos)}/"
                       f"{len(pos)}{mark}")
        out.append("accept >= t:  share of all documents imported | irrelevant ones wrongly imported")
        for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95):
            mark = "  <- current" if abs(t - cfg.scoring.accept) < 1e-9 else ""
            out.append(f"  {t:4.2f}:  {sum(p >= t for p in ps) / len(ps):6.1%}  |  {sum(p >= t for p in neg)}/"
                       f"{len(neg)}{mark}")
    matrix = Counter((r["manual"], decision(r, cfg, use_manual=False)) for r in labeled)
    out += ["", "== Current decision vs. manual label =="]
    for manual in ("yes", "no"):
        parts = ", ".join(f"{d}: {matrix[(manual, d)]}" for d in (RELEVANT, REVIEW, "escalate", IRRELEVANT))
        out.append(f"  manual {manual:<3} -> {parts}")
    wrong = [r for r in labeled if r["manual"] == "yes" and decision(r, cfg, use_manual=False) == IRRELEVANT]
    if wrong:
        out.append("Wrongly skipped: " + ", ".join(f"{r['sha256'][:12]} ({r['filename']})" for r in wrong[:20]))
    return "\n".join(out)
