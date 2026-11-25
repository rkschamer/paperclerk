"""Derive the import decision from the stored model answers.

Probabilities are stored raw; thresholds and categories only apply here, so configuration changes take
effect immediately without asking the models again.
"""

from __future__ import annotations

import json
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
