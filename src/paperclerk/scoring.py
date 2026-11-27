"""Import only: is a mail attachment worth keeping? (pplx-decider via Ollama's /v1/systemone)"""

from __future__ import annotations

import json
import logging
import time
from sqlite3 import Row

from .config import Config
from .db import DB
from .decision import ESCALATE, category_key, decision, probability
from .ollama import Ollama, OllamaError, OllamaUnavailable
from .textextract import render_for_vision
from .util import Progress

log = logging.getLogger(__name__)

KIND_LABEL = {"pdf": "PDF", "image": "image", "office": "office document"}
MAX_FAILS = 3
MAX_CONSECUTIVE_FAILS = 5


def _pages(n: int | None) -> str:
    return f", {n} page{'s' if n != 1 else ''}" if n else ""


def mail_lines(db: DB, row: Row) -> tuple[list[str], Row | None]:
    """Where an imported file comes from, as context lines for the models; plus its earliest mail."""
    mails = db.mails_for(row["sha256"])
    lines = [f"File type: {KIND_LABEL.get(row['kind'], row['kind'])}{_pages(row['pages'])}"]
    m = mails[0] if mails else None
    if m:
        sender = f"{m['from_name']} <{m['from_addr']}>" if m["from_name"] else (m["from_addr"] or "?")
        lines += [f"E-mail from: {sender}", f"E-mail subject: {m['subject']}",
                  f"E-mail date: {(m['date'] or '')[:10]}"]
        if m["body"]:
            lines.append(f"E-mail text (excerpt): {m['body'][:500]}")
        if len(mails) > 1:
            lines.append(f"Note: the same file was attached to {len(mails)} different e-mails.")
    if row["parts"]:
        lines.append(f"Note: merged from {len(json.loads(row['parts']))} separate attachments.")
    return lines, m


def describe(db: DB, row: Row, max_chars: int, has_images: bool = False) -> str:
    lines, _ = mail_lines(db, row)
    lines.insert(0, f"File name: {row['filename']}")
    text = (row["text"] or "").strip()
    if text:
        lines += ["", "Document text (excerpt):", text[:max_chars]]
    elif has_images:
        lines += ["", "No readable text – the first page is attached as an image."]
    else:
        lines += ["", "No readable text."]
    return "\n".join(lines)


def vision_images(row: Row, cfg: Config) -> list[str] | None:
    if row["text_method"] != "none" or row["kind"] not in ("pdf", "image"):
        return None
    return render_for_vision(cfg.files_dir / row["path"], row["kind"], cfg.text.vision_max_side) or None


def questions(cfg: Config) -> dict:
    return {
        "relevant": {
            "type": "noul",
            "instructions": "Is this e-mail attachment an individual document that should be kept permanently in a "
                            "personal document archive?",
            "criteria": {"true": cfg.scoring.relevant_true, "false": cfg.scoring.relevant_false},
        },
        "category": {
            "type": "choice",
            "instructions": "What kind of document is this e-mail attachment?",
            "criteria": {c.key: c.description for c in cfg.categories},
        },
    }


def _fallback_prompt(cfg: Config) -> tuple[str, dict]:
    cats = "\n".join(f"- {c.key}: {c.description}" for c in cfg.categories)
    system = ("You classify e-mail attachments for a personal document archive. Reply with JSON only.\n"
              f"relevant = true: {cfg.scoring.relevant_true}\nrelevant = false: {cfg.scoring.relevant_false}\n"
              f"category: exactly one of these keys:\n{cats}")
    schema = {"type": "object", "properties": {"category": {"type": "string", "enum": [c.key for c in cfg.categories]},
                                               "relevant": {"type": "boolean"}},
              "required": ["category", "relevant"]}
    return system, schema


class Scorer:
    def __init__(self, db: DB, cfg: Config, ollama: Ollama, model: str, stage: str):
        self.db, self.cfg, self.ollama, self.model, self.stage = db, cfg, ollama, model, stage
        self.questions = questions(cfg)
        self.fallback = _fallback_prompt(cfg) if stage == "score" and not cfg.scoring.model else None

    def __call__(self, row: Row) -> tuple[float, dict[str, float], int | None]:
        images = vision_images(row, self.cfg)
        state = describe(self.db, row, self.cfg.scoring.max_chars, bool(images))
        if self.fallback:
            system, schema = self.fallback
            user = {"role": "user", "content": state}
            if images:
                user["images"] = images
            res = self.ollama.chat(self.model, [{"role": "system", "content": system}, user], schema=schema,
                                   think=self.cfg.enrich.think,
                                   options={"temperature": 0, "num_ctx": self.cfg.enrich.num_ctx})
            try:
                data = json.loads(res.content)
                return (1.0 if data["relevant"] else 0.0), {str(data["category"]): 1.0}, res.input_tokens
            except (ValueError, KeyError, TypeError) as e:
                raise OllamaError(f"invalid answer: {res.content[:200]!r} ({e})") from None
        res = self.ollama.systemone(self.model, state, self.questions, images)
        try:
            answers = res["answers"]
            noul = float(answers["relevant"]["noul"])
            probs = {str(k): float(v) for k, v in answers["category"]["probabilities"].items()}
        except (KeyError, TypeError, ValueError) as e:
            raise OllamaError(f"unexpected answer from /v1/systemone: {str(res)[:300]} ({e})") from None
        return noul, probs, (res.get("usage") or {}).get("input_tokens")


def pending(db: DB, cfg: Config, stage: str, sha: str | None = None) -> list[Row]:
    """Documents waiting for relevance (stage "score") or for a second opinion ("escalate")."""
    one, params = (" AND sha256 = ?", (MAX_FAILS, sha)) if sha else ("", (MAX_FAILS,))
    if stage == "score":
        return db.q(
            f"""SELECT * FROM documents
                WHERE text_method IS NOT NULL AND skip_reason IS NULL AND part_of IS NULL AND noul1 IS NULL
                  AND fail_count < ?{one}
                ORDER BY rowid""", params)
    if not cfg.scoring.escalation_model:
        return []
    rows = db.q(
        f"""SELECT * FROM documents
            WHERE noul1 IS NOT NULL AND noul2 IS NULL AND skip_reason IS NULL AND part_of IS NULL
              AND manual IS NULL AND fail_count < ?{one}
            ORDER BY rowid""", params)
    return [r for r in rows if decision(r, cfg) == ESCALATE]


def stage_model(cfg: Config, stage: str) -> str:
    return cfg.scoring.escalation_model if stage == "escalate" else cfg.scoring.model or cfg.enrich.model


def score_one(db: DB, cfg: Config, scorer: Scorer, row: Row) -> Row:
    """Ask the decider about one document and store the answer; returns the updated row. A failed request is
    recorded with the document and re-raised."""
    sha, stage, model = row["sha256"], scorer.stage, scorer.model
    t0 = time.monotonic()
    try:
        noul, probs, tokens = scorer(row)
    except OllamaUnavailable:
        raise
    except OllamaError as e:
        db.update_doc(sha, fail_count=row["fail_count"] + 1, last_error=str(e)[:500])
        db.log_llm(sha, stage, model, time.monotonic() - t0, error=str(e)[:2000])
        db.commit()
        raise
    secs = time.monotonic() - t0
    if stage == "escalate":
        db.update_doc(sha, noul2=noul, types2=json.dumps(probs), model2=model, secs2=round(secs, 2), fail_count=0,
                      last_error=None)
    else:
        db.update_doc(sha, noul1=noul, types1=json.dumps(probs), model1=model, secs1=round(secs, 2), fail_count=0,
                      last_error=None)
    db.log_llm(sha, stage, model, secs, tokens, 0, json.dumps({"noul": noul, "category": probs}))
    db.commit()
    return db.one("SELECT * FROM documents WHERE sha256 = ?", (sha,))


def run_scoring(db: DB, cfg: Config, stage: str = "score", limit: int | None = None) -> int:
    """Relevance of new attachments (stage="score") or second opinion on uncertain ones ("escalate")."""
    model = stage_model(cfg, stage)
    rows = pending(db, cfg, stage)
    rows = rows[:limit] if limit else rows
    label = "Escalation" if stage == "escalate" else "Relevance"
    if not rows:
        return 0
    ollama = Ollama(cfg.ollama.url, cfg.ollama.keep_alive, cfg.ollama.timeout)
    try:
        ollama.require(model)
        scorer = Scorer(db, cfg, ollama, model, stage)
        log.info("%s with %s for %d documents", label, model, len(rows))
        progress = Progress(len(rows), label)
        consecutive = 0
        for row in rows:
            t0 = time.monotonic()
            try:
                updated = score_one(db, cfg, scorer, row)
            except OllamaUnavailable:
                raise
            except OllamaError as e:
                consecutive += 1
                progress.step(f"{row['filename']}: ERROR {e}", time.monotonic() - t0)
                if consecutive >= MAX_CONSECUTIVE_FAILS:
                    raise OllamaUnavailable(f"{consecutive} errors in a row – stopping. Last error: {e}")
                continue
            consecutive = 0
            p = probability(updated, cfg)
            progress.step(f"{row['filename']}: {decision(updated, cfg)} ({p:.0%}, {category_key(updated)})",
                          time.monotonic() - t0)
        progress.done()
        return len(rows)
    finally:
        ollama.close()
