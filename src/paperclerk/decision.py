"""Derive the import decision from the stored model answers.

Probabilities are stored raw; thresholds and categories only apply here, so configuration changes take
effect immediately without asking the models again.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Mapping

from .config import Config

RELEVANT = "relevant"
IRRELEVANT = "irrelevant"
REVIEW = "review"
ESCALATE = "escalate"
PENDING = "pending"
UPLOAD_DECISIONS = (RELEVANT, REVIEW)

Row = Mapping[str, Any]


def _json(value: str | None) -> Any:
    return json.loads(value) if value else None


def relevance(noul: float | None, types_json: str | None, cfg: Config) -> float | None:
    if noul is None:
        return None
    weight = cfg.scoring.type_weight
    probs = _json(types_json) or {}
    known = {c.key: c.relevant for c in cfg.categories}
    total = sum(v for k, v in probs.items() if k in known)
    if weight == 0 or total <= 0:
        return float(noul)
    mass = sum(v for k, v in probs.items() if known.get(k)) / total
    return (1 - weight) * float(noul) + weight * mass


def probability(row: Row, cfg: Config) -> float | None:
    if row["noul2"] is not None:
        return relevance(row["noul2"], row["types2"], cfg)
    return relevance(row["noul1"], row["types1"], cfg)


def decision(row: Row, cfg: Config, use_manual: bool = True) -> str:
    if use_manual and row["manual"] == "yes":
        return RELEVANT
    if use_manual and row["manual"] == "no":
        return IRRELEVANT
    p = probability(row, cfg)
    if p is None:
        return PENDING
    if p >= cfg.scoring.accept:
        return RELEVANT
    if p <= cfg.scoring.reject:
        return IRRELEVANT
    if row["noul2"] is None and cfg.scoring.escalation_model:
        return ESCALATE
    return REVIEW


def type_probs(row: Row) -> dict[str, float]:
    return _json(row["types2"]) or _json(row["types1"]) or {}


def category_key(row: Row) -> str | None:
    probs = type_probs(row)
    return max(probs, key=probs.get) if probs else None


def overrides(row: Row) -> dict:
    return _json(row["overrides"]) or {}


def meta(row: Row) -> dict:
    return _json(row["meta"]) or {}


_EMPTY: dict[str, type] = {"tags": list, "fields": dict}


def effective(row: Row, name: str) -> Any:
    """A metadata value: manual correction first, then the model's suggestion."""
    over = overrides(row)
    if name in over:
        return over[name]
    value = meta(row).get(name)
    return value if value is not None else _EMPTY.get(name, str)()


# ---------------------------------------------------------------- where an imported document stands

SKIPPED, NEW, PART = "skipped", "new", "part"


def state(row: Row, cfg: Config) -> str:
    if row["part_of"]:
        return PART
    if row["skip_reason"]:
        return SKIPPED
    if row["text_method"] is None:
        return NEW
    return decision(row, cfg)


def evaluate(db, cfg: Config) -> str:
    """How the thresholds perform against the user's decisions (manual yes/no), as a text report."""
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
        out += ["", f"== {name}: you imported {len(pos)}, rejected {len(neg)} =="]
        out.append("reject <= t:  share of all documents skipped | imported ones that would be skipped")
        for t in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            mark = "  <- current" if abs(t - cfg.scoring.reject) < 1e-9 else ""
            out.append(f"  {t:4.2f}:  {sum(p <= t for p in ps) / len(ps):6.1%}  |  {sum(p <= t for p in pos)}/"
                       f"{len(pos)}{mark}")
        out.append("accept >= t:  share of all documents imported | rejected ones that would be imported")
        for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95):
            mark = "  <- current" if abs(t - cfg.scoring.accept) < 1e-9 else ""
            out.append(f"  {t:4.2f}:  {sum(p >= t for p in ps) / len(ps):6.1%}  |  {sum(p >= t for p in neg)}/"
                       f"{len(neg)}{mark}")
    matrix = Counter((r["manual"], decision(r, cfg, use_manual=False)) for r in labeled)
    out += ["", "== The model's decision vs. yours =="]
    for manual, label in (("yes", "imported"), ("no", "rejected")):
        parts = ", ".join(f"{d}: {matrix[(manual, d)]}" for d in (RELEVANT, REVIEW, ESCALATE, IRRELEVANT))
        out.append(f"  you {label:<8} -> {parts}")
    wrong = [r for r in labeled if r["manual"] == "yes" and decision(r, cfg, use_manual=False) == IRRELEVANT]
    if wrong:
        out.append("Wrongly skipped: " + ", ".join(f"{r['sha256'][:12]} ({r['filename']})" for r in wrong[:20]))
    return "\n".join(out).strip("\n")
