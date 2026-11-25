"""Import only: detect documents split across several attachments and merge them into one PDF.

Candidates are attachments of the same mail whose file names differ only in a running number
("Vertrag 1.pdf", "Vertrag 2.pdf", ...). Because that could also be two different invoices, the decision
model is asked at every boundary whether part B continues part A. Answers are stored in merge_checks and
reused in later runs.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import PurePosixPath
from sqlite3 import Row

import pymupdf

from . import store
from .config import Config
from .db import DB, now_iso
from .ollama import Ollama, OllamaError, OllamaUnavailable
from .textextract import MAX_STORED_CHARS, TextExtractor, compact, load_frames, render_page, usable
from .util import Progress

log = logging.getLogger(__name__)

# running number at the end of the name: "X 12", "X_003", "X-2", "X (2)"; four-digit years do not count
_NUMBERED = re.compile(r"^(?P<stem>.*?)[\s_\-.]*\(?(?<!\d)(?P<num>\d{1,4})\)?$")
_TAIL = " _-.("

QUESTION = {
    "continuation": {
        "type": "noul",
        "instructions": "Does part B belong to the same document as part A, i.e. does part B continue the document "
                        "from part A?",
        "criteria": {
            "true": "Part B continues the same document: the next page, another section or an enclosure of the same "
                    "contract, letter, notice or invoice.",
            "false": "Part B is a separate, different document, e.g. another invoice, a different letter or a "
                     "standard text such as terms and conditions.",
        },
    }
}


def _parse(filename: str) -> tuple[str, int | None]:
    stem = PurePosixPath(filename or "").stem
    m = _NUMBERED.match(stem)
    if m and m.group("stem").strip(_TAIL):
        num = int(m.group("num"))
        if not (len(m.group("num")) == 4 and 1900 <= num <= 2099):
            return m.group("stem").rstrip(_TAIL), num
    return stem.rstrip(_TAIL), None


def split_name(filename: str) -> tuple[str, int | None]:
    """'Vertrag vom 04.11.2026 12.pdf' -> ('vertrag vom 04.11.2026', 12)"""
    stem, num = _parse(filename)
    return stem.casefold(), num


def merged_name(filename: str) -> str:
    return (_parse(filename)[0] or "document") + ".pdf"


def candidate_groups(db: DB) -> list[list[Row]]:
    rows = db.q(
        """SELECT a.message_id AS msg, a.id AS att_id, a.filename AS att_name, d.*
             FROM attachments a JOIN documents d ON d.sha256 = a.sha256
            WHERE a.part NOT LIKE 'merge:%' AND d.skip_reason IS NULL AND d.text_method IS NOT NULL
              AND d.part_of IS NULL AND d.parts IS NULL AND d.kind IN ('pdf', 'image')
              AND d.upload_status IS NULL
            ORDER BY a.message_id, a.id"""
    )
    buckets: dict[tuple, list[tuple[int | None, Row]]] = defaultdict(list)
    for r in rows:
        stem, num = split_name(r["att_name"] or r["filename"])
        buckets[(r["msg"], stem)].append((num, r))
    groups: list[list[Row]] = []
    used: set[str] = set()
    for items in buckets.values():
        if len(items) < 2 or all(num is None for num, _ in items):
            continue
        items.sort(key=lambda x: (-1 if x[0] is None else x[0], x[1]["att_id"]))
        docs, seen = [], set()
        for _, r in items:
            if r["sha256"] not in seen and r["sha256"] not in used:
                seen.add(r["sha256"])
                docs.append(r)
        if len(docs) >= 2:
            groups.append(docs)
            used |= seen
    return groups


def _stored(db: DB, a: str, b: str) -> float | None:
    r = db.one("SELECT p FROM merge_checks WHERE a = ? AND b = ?", (a, b))
    return r["p"] if r else None


def _remember(db: DB, a: str, b: str, p: float, model: str) -> None:
    db.x("INSERT OR REPLACE INTO merge_checks(a, b, p, model, at) VALUES (?, ?, ?, ?, ?)", (a, b, p, model, now_iso()))


class ContinuationCheck:
    def __init__(self, db: DB, cfg: Config, ollama: Ollama):
        self.db, self.cfg, self.ollama = db, cfg, ollama
        self.model = cfg.scoring.model or cfg.enrich.model
        self.text = TextExtractor(cfg)

    def _end_text(self, row: Row) -> str:
        if (row["pages"] or 1) <= self.cfg.text.max_pages:
            return row["text"] or ""
        return self.text.page_text(self.cfg.files_dir / row["path"], row["kind"], -1)

    def __call__(self, a: Row, b: Row, position: tuple[int, int] | None = None) -> float:
        """position = (number of file A, number of files in the series)"""
        n = self.cfg.merge.max_chars
        end_a, start_b = self._end_text(a)[-n:].strip(), (b["text"] or "")[:n].strip()
        images: list[str] = []
        side = self.cfg.text.vision_max_side
        if not end_a and (img := render_page(self.cfg.files_dir / a["path"], a["kind"], -1, side)):
            images.append(img)
            end_a = f"(no readable text – attached as image {len(images)})"
        if not start_b and (img := render_page(self.cfg.files_dir / b["path"], b["kind"], 0, side)):
            images.append(img)
            start_b = f"(no readable text – attached as image {len(images)})"

        def pages(r: Row) -> str:
            return f"{r['pages']} page{'s' if (r['pages'] or 1) != 1 else ''}" if r["pages"] else r["kind"]

        context = ""
        if position:
            context = (f"Context: both files were attached to the same e-mail and belong to a consecutively "
                       f"numbered series of {position[1]} files with the same name (boundary between file "
                       f"{position[0]} and {position[0] + 1}).\n\n")
        state = (f'{context}Part A: file "{a["att_name"] or a["filename"]}" ({pages(a)}), end of the text:\n'
                 f'"""\n{end_a}\n"""\n\n'
                 f'Part B: file "{b["att_name"] or b["filename"]}" ({pages(b)}), start of the text:\n'
                 f'"""\n{start_b}\n"""')
        t0 = time.monotonic()
        if self.cfg.scoring.model:
            res = self.ollama.systemone(self.model, state, QUESTION, images or None)
            try:
                p = float(res["answers"]["continuation"]["noul"])
            except (KeyError, TypeError, ValueError) as e:
                raise OllamaError(f"unexpected answer: {str(res)[:200]} ({e})") from None
            tokens = (res.get("usage") or {}).get("input_tokens")
        else:
            schema = {"type": "object", "properties": {"continuation": {"type": "boolean"}},
                      "required": ["continuation"]}
            user: dict = {"role": "user", "content": state}
            if images:
                user["images"] = images
            q = QUESTION["continuation"]
            system = (f"{q['instructions']} Reply with JSON only. continuation = true: {q['criteria']['true']} "
                      f"continuation = false: {q['criteria']['false']}")
            res = self.ollama.chat(self.model, [{"role": "system", "content": system}, user], schema=schema,
                                   think=self.cfg.enrich.think,
                                   options={"temperature": 0, "num_ctx": self.cfg.enrich.num_ctx})
            try:
                p = 1.0 if json.loads(res.content)["continuation"] else 0.0
            except (ValueError, KeyError, TypeError) as e:
                raise OllamaError(f"invalid answer: {res.content[:200]!r} ({e})") from None
            tokens = res.input_tokens
        secs = time.monotonic() - t0
        _remember(self.db, a["sha256"], b["sha256"], p, self.model)
        self.db.log_llm(a["sha256"], "merge", self.model, secs, tokens, 0, json.dumps({"b": b["sha256"], "p": p}))
        self.db.commit()
        return p


def _inherit_manual(rows: list[Row]) -> str | None:
    labels = [r["manual"] for r in rows]
    if "yes" in labels:
        return "yes"
    if labels and all(x == "no" for x in labels):
        return "no"
    return None


def merge_docs(db: DB, cfg: Config, rows: list[Row], name: str) -> str:
    """Merge the parts in this order into one PDF; returns the id of the new document."""
    out = pymupdf.open()
    for r in rows:
        path = cfg.files_dir / r["path"]
        if r["kind"] == "pdf":
            with pymupdf.open(path) as src:
                if src.needs_pass:
                    src.authenticate("")
                out.insert_pdf(src)
        else:
            for frame in load_frames(path, 1000):
                buf = io.BytesIO()
                frame.save(buf, format="JPEG", quality=90)
                width = 595.0  # A4 width; the image keeps its resolution
                page = out.new_page(width=width, height=width * frame.height / frame.width)
                page.insert_image(page.rect, stream=buf.getvalue())
    page_count = out.page_count
    data = out.tobytes(garbage=3, deflate=True)
    out.close()

    sha, md5 = store.hashes(data)
    text = compact("\n\n".join(r["text"] or "" for r in rows))[:MAX_STORED_CHARS]
    method = "ocr" if any(r["text_method"] == "ocr" for r in rows) else "pdf"
    if not usable(text, cfg.text.min_chars_per_page):
        method = "none"
    if not db.one("SELECT 1 FROM documents WHERE sha256 = ?", (sha,)):
        rel = store.store_blob(cfg.files_dir, data, sha, "pdf")
        db.x(
            """INSERT INTO documents(sha256, md5, kind, ext, mime, size, path, filename, added_at, pages, text,
                                     text_method, parts, manual)
               VALUES (?, ?, 'pdf', 'pdf', 'application/pdf', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sha, md5, len(data), rel, name, now_iso(), page_count, text, method,
             json.dumps([r["sha256"] for r in rows]), _inherit_manual(rows)),
        )
    for r in rows:
        db.update_doc(r["sha256"], part_of=sha)
    msg_ids: set[int] = set()
    for r in rows:
        msg_ids |= {a["message_id"] for a in db.q(
            "SELECT message_id FROM attachments WHERE sha256 = ? AND part NOT LIKE 'merge:%'", (r["sha256"],))}
    for mid in sorted(msg_ids):
        db.x("""INSERT OR IGNORE INTO attachments(message_id, part, filename, content_type, size, sha256)
                VALUES (?, ?, ?, 'application/pdf', ?, ?)""", (mid, f"merge:{sha[:16]}", name, len(data), sha))
    db.commit()
    return sha


def run_merge(db: DB, cfg: Config, limit: int | None = None) -> int:
    """Find candidates, check the boundaries, merge parts that belong together."""
    if not cfg.merge.enabled:
        return 0
    groups = candidate_groups(db)
    groups = groups[:limit] if limit else groups
    if not groups:
        return 0
    todo = sum(1 for g in groups for a, b in zip(g, g[1:]) if _stored(db, a["sha256"], b["sha256"]) is None)
    ollama = Ollama(cfg.ollama.url, cfg.ollama.keep_alive, cfg.ollama.timeout)
    merged = 0
    try:
        check = ContinuationCheck(db, cfg, ollama)
        if todo:
            ollama.require(check.model)
            log.info("Merging: %d groups, %d boundaries to check with %s", len(groups), todo, check.model)
        progress = Progress(todo, "Boundary")
        for docs in groups:
            runs, current = [], [docs[0]]
            for i, (a, b) in enumerate(zip(docs, docs[1:]), 1):
                p = _stored(db, a["sha256"], b["sha256"])
                if p is None:
                    t0 = time.monotonic()
                    try:
                        p = check(a, b, (i, len(docs)))
                    except OllamaUnavailable:
                        raise
                    except OllamaError as e:
                        log.warning("Boundary %s | %s not checked: %s", a["att_name"], b["att_name"], e)
                        p = 0.0
                    progress.step(f"{a['att_name']} | {b['att_name']}: continuation {p:.0%}", time.monotonic() - t0)
                if p >= cfg.merge.threshold:
                    current.append(b)
                else:
                    runs.append(current)
                    current = [b]
            runs.append(current)
            for run in runs:
                if len(run) < 2:
                    continue
                name = merged_name(run[0]["att_name"] or run[0]["filename"])
                try:
                    sha = merge_docs(db, cfg, run, name)
                except Exception as e:  # noqa: BLE001 – one broken PDF must not stop the run
                    log.error("Merging %s failed: %s", run[0]["att_name"], e)
                    continue
                merged += 1
                log.info("Merged %d parts (%s ... %s) -> %s [%s]", len(run), run[0]["att_name"], run[-1]["att_name"],
                         name, sha[:12])
        if todo:
            progress.done()
    finally:
        ollama.close()
    return merged
