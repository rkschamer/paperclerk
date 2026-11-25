from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx

from .config import Config
from .db import DB

log = logging.getLogger(__name__)

KINDS = ("correspondents", "document_types", "tags", "custom_fields")
LABEL = {"correspondents": "correspondent", "document_types": "document type", "tags": "tag",
         "custom_fields": "custom field"}
MATCH_NONE = 0


class PaperlessError(RuntimeError):
    pass


@dataclass
class TaskInfo:
    status: str  # PENDING | STARTED | SUCCESS | FAILURE | REVOKED | ...
    document_id: int | None
    duplicate_of: int | None
    message: str

    @property
    def finished(self) -> bool:
        return self.status in ("SUCCESS", "FAILURE", "REVOKED")


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_task(obj: dict) -> TaskInfo:
    """Task object of API v9 (upper-case status, related_document) or v10 (result_data)."""
    status = str(obj.get("status") or "UNKNOWN").upper()
    data = obj.get("result_data") if isinstance(obj.get("result_data"), dict) else {}
    message = obj.get("result")
    if not message and data:
        message = data.get("error") or data.get("message") or json.dumps(data, ensure_ascii=False)
    message = str(message or "")
    duplicate = _int(data.get("duplicate_of"))
    if duplicate is None:
        m = re.search(r"duplicate of .*?#(\d+)", message, re.I)
        duplicate = _int(m.group(1)) if m else None
    doc_id = _int(data.get("document_id"))
    ids = obj.get("related_document_ids")
    if doc_id is None and isinstance(ids, list) and ids:
        doc_id = _int(ids[0])
    if doc_id is None:
        doc_id = _int(obj.get("related_document"))
    if doc_id is None:
        m = re.search(r"New document id (\d+)", message)
        doc_id = _int(m.group(1)) if m else None
    if doc_id is not None and doc_id == duplicate and status != "SUCCESS":
        doc_id = None
    return TaskInfo(status, doc_id, duplicate, message[:1000])


class Paperless:
    def __init__(self, url: str, token: str, timeout: float = 120, verify: bool = True,
                 transport: httpx.BaseTransport | None = None):
        if not url or not token:
            raise PaperlessError("Paperless: set [paperless] url and token (or PAPERLESS_URL / PAPERLESS_TOKEN)")
        self.url = url.rstrip("/")
        self.api_version: int | None = None
        self.http = httpx.Client(base_url=self.url, timeout=timeout, verify=verify, follow_redirects=True,
                                 transport=transport,
                                 headers={"Authorization": f"Token {token}", "Accept": "application/json"})

    @classmethod
    def from_config(cls, cfg: Config) -> "Paperless":
        p = cfg.paperless
        return cls(p.url, p.token, p.timeout, p.verify_tls)

    def __enter__(self) -> "Paperless":
        return self

    def __exit__(self, *exc: object) -> None:
        self.http.close()

    def close(self) -> None:
        self.http.close()

    def _req(self, method: str, path: str, retries: int = 3, **kw: Any) -> httpx.Response:
        for attempt in range(retries + 1):
            try:
                r = self.http.request(method, path, **kw)
            except httpx.TransportError as e:
                if attempt == retries:
                    raise PaperlessError(f"Paperless unreachable ({self.url}): {e}") from None
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code in (502, 503, 504) and attempt < retries:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                hint = ("the response did not come from Paperless – a login proxy (SSO) in front of it blocks the "
                        "request; use an address that bypasses it" if "x-api-version" not in r.headers
                        else "check the token and its permissions")
                raise PaperlessError(f"Paperless: access denied (HTTP {r.status_code}) – {hint}")
            if self.api_version is None and r.headers.get("x-api-version", "").isdigit():
                self.api_version = int(r.headers["x-api-version"])
            return r
        raise PaperlessError("unreachable")

    def _json(self, r: httpx.Response) -> Any:
        if r.status_code >= 400:
            raise PaperlessError(f"Paperless {r.request.method} {r.request.url.path}: HTTP {r.status_code}: "
                                 f"{r.text[:300]}")
        return r.json() if r.content else None

    # ---------------------------------------------------------------- objects

    def server_info(self) -> dict:
        r = self._req("GET", "/api/documents/", params={"page_size": 1, "fields": "id"})
        data = self._json(r)
        return {"version": r.headers.get("X-Version", "?"), "api_version": r.headers.get("X-Api-Version", "?"),
                "documents": data.get("count")}

    def list_all(self, kind: str) -> list[dict]:
        return list(self._pages(f"/api/{kind}/", {}))

    def _pages(self, path: str, params: dict, page_size: int = 250) -> Iterator[dict]:
        page = 1
        while True:
            data = self._json(self._req("GET", path, params={**params, "page": page, "page_size": page_size}))
            yield from data.get("results", [])
            if not data.get("next"):
                return
            page += 1

    def find_by_name(self, kind: str, name: str) -> dict | None:
        data = self._json(self._req("GET", f"/api/{kind}/", params={"name__iexact": name}))
        results = data.get("results", [])
        return results[0] if results else None

    def create(self, kind: str, name: str, data_type: str | None = None) -> dict:
        payload: dict[str, Any] = {"name": name}
        if kind == "custom_fields":
            payload["data_type"] = data_type or "string"
        else:
            payload["matching_algorithm"] = MATCH_NONE
        r = self._req("POST", f"/api/{kind}/", json=payload)
        if r.status_code == 400:
            existing = self.find_by_name(kind, name)
            if existing:
                return existing
        return self._json(r)

    # ---------------------------------------------------------------- documents

    def documents(self, params: dict) -> Iterator[dict]:
        yield from self._pages("/api/documents/", params, page_size=100)

    def document(self, doc_id: int) -> dict:
        return self._json(self._req("GET", f"/api/documents/{doc_id}/"))

    def update(self, doc_id: int, payload: dict) -> dict:
        if "created" in payload and self.api_version is not None and self.api_version < 9:
            payload = {**payload, "created_date": payload.pop("created")}
        return self._json(self._req("PATCH", f"/api/documents/{doc_id}/", json=payload))

    def download(self, doc_id: int) -> bytes:
        r = self._req("GET", f"/api/documents/{doc_id}/download/")
        if r.status_code >= 400:
            raise PaperlessError(f"download of #{doc_id} failed: HTTP {r.status_code}")
        return r.content

    def find_by_checksum(self, *checksums: str) -> int | None:
        for checksum in checksums:
            data = self._json(self._req("GET", "/api/documents/",
                                        params={"checksum__iexact": checksum, "page_size": 1, "fields": "id"}))
            if data.get("results"):
                return int(data["results"][0]["id"])
        return None

    def search(self, terms: list[str], limit: int = 5) -> list[dict]:
        """Full-text search (any term matches), hits with content, best first."""
        if not terms:
            return []
        data = self._json(self._req("GET", "/api/documents/", params={
            "query": " OR ".join(terms), "page_size": limit, "fields": "id,title,content"}))
        return data.get("results", [])

    def upload(self, path: Path, filename: str, mime: str, fields: dict) -> str:
        with path.open("rb") as fh:
            r = self._req("POST", "/api/documents/post_document/", retries=0, data=fields,
                          files={"document": (filename, fh, mime)})
        data = self._json(r)
        if isinstance(data, dict):
            data = data.get("task_id") or data.get("id")
        if not data:
            raise PaperlessError(f"Paperless returned no task id: {r.text[:200]}")
        return str(data)

    def task(self, task_id: str) -> TaskInfo | None:
        data = self._json(self._req("GET", "/api/tasks/", params={"task_id": task_id}))
        items = data.get("results", []) if isinstance(data, dict) else data
        for obj in items or []:
            if obj.get("task_id") == task_id:
                return parse_task(obj)
        return None

    def add_note(self, doc_id: int, text: str) -> int | None:
        """Adds a note and returns its id."""
        notes = self._json(self._req("POST", f"/api/documents/{doc_id}/notes/", json={"note": text})) or []
        ours = [n for n in notes if isinstance(n, dict) and n.get("note") == text]
        return max((int(n["id"]) for n in ours), default=None)

    def delete_note(self, doc_id: int, note_id: int) -> None:
        r = self._req("DELETE", f"/api/documents/{doc_id}/notes/", params={"id": note_id})
        if r.status_code >= 400 and r.status_code != 404:
            raise PaperlessError(f"deleting note {note_id} of #{doc_id} failed: HTTP {r.status_code}")


def sync_cache(db: DB, pl: Paperless) -> dict[str, list[dict]]:
    """Read correspondents, document types, tags and custom fields from Paperless into the local cache."""
    result = {}
    for kind in KINDS:
        objs = pl.list_all(kind)
        db.x("DELETE FROM paperless_objects WHERE kind = ?", (kind,))
        for o in objs:
            extra = {"document_count": o.get("document_count"), "is_inbox_tag": o.get("is_inbox_tag", False),
                     "data_type": o.get("data_type")}
            db.x("INSERT OR REPLACE INTO paperless_objects(kind, id, name, extra) VALUES (?, ?, ?, ?)",
                 (kind, o["id"], o["name"], json.dumps(extra)))
        result[kind] = objs
    db.commit()
    return result


def cached(db: DB, kind: str) -> list[dict]:
    """Objects from the local cache: [{id, name, document_count, is_inbox_tag, data_type}]"""
    return [{"id": r["id"], "name": r["name"], **json.loads(r["extra"] or "{}")}
            for r in db.q("SELECT id, name, extra FROM paperless_objects WHERE kind = ? ORDER BY name", (kind,))]


class Registry:
    """Names -> Paperless ids; creates missing objects (always with matching 'none')."""

    def __init__(self, db: DB, pl: Paperless, objects: dict[str, list[dict]] | None = None):
        self.db, self.pl = db, pl
        objects = objects if objects is not None else sync_cache(db, pl)
        self.ids = {kind: {o["name"].casefold(): o["id"] for o in objects[kind]} for kind in KINDS}
        self.names = {kind: {o["id"]: o["name"] for o in objects[kind]} for kind in KINDS}

    def get(self, kind: str, name: str) -> int | None:
        return self.ids[kind].get(name.strip().casefold())

    def ensure(self, kind: str, name: str, data_type: str | None = None) -> int | None:
        name = name.strip()[:128]
        if not name:
            return None
        if (found := self.get(kind, name)) is not None:
            return found
        obj = self.pl.create(kind, name, data_type)
        self.ids[kind][name.casefold()] = obj["id"]
        self.names[kind][obj["id"]] = obj["name"]
        extra = {"document_count": 0, "data_type": obj.get("data_type")}
        self.db.x("INSERT OR REPLACE INTO paperless_objects(kind, id, name, extra) VALUES (?, ?, ?, ?)",
                  (kind, obj["id"], obj["name"], json.dumps(extra)))
        log.info("Paperless: created %s '%s' (id %s)", LABEL[kind], obj["name"], obj["id"])
        return obj["id"]
