from __future__ import annotations

import hashlib
import json
import os
import uuid
from email import policy
from email.parser import BytesParser
from pathlib import Path

import httpx
import pymupdf
import pytest

import paperclerk.importer
import paperclerk.merge
import paperclerk.process
import paperclerk.scoring
import paperclerk.upload
from paperclerk.config import load
from paperclerk.db import DB
from paperclerk.ollama import Ollama
from paperclerk.paperless import Paperless
from samples import write_mbox

# Path to deu/eng.traineddata so the OCR tests run (otherwise they are skipped)
TESSDATA = os.environ.get("PAPERCLERK_TEST_TESSDATA", "")


@pytest.fixture
def cfg(tmp_path: Path):
    ocr = "true" if TESSDATA else "false"
    (tmp_path / "config.toml").write_text(
        f"""
data_dir = "data"
[paperless]
url = "http://paperless.test"
token = "secret"
[text]
ocr = {ocr}
tessdata_dir = '{TESSDATA}'
[import.scoring]
model = "pplx-decider-flash"
escalation_model = "pplx-decider"
""",
        encoding="utf-8",
    )
    return load(tmp_path / "config.toml")


@pytest.fixture
def db(cfg):
    d = DB(cfg.db_path)
    yield d
    d.close()


@pytest.fixture
def mbox(tmp_path: Path) -> Path:
    return write_mbox(tmp_path / "test.mbox")


# ------------------------------------------------------------------ fake Ollama

RELEVANT_WORDS = {
    "Rechnungsbetrag": "invoice", "RECHNUNG": "invoice", "Versicherungsschein": "insurance",
    "Entgeltabrechnung": "payslip", "Bestellbestätigung": "order", "Kaffeevollautomat": "receipt",
    "Überlassungsvertrag": "contract",
}
IRRELEVANT_WORDS = {"Allgemeine Geschäftsbedingungen": "terms", "Widerrufsbelehrung": "withdrawal",
                    "Datenschutzhinweise": "privacy", "Frühlingsangebote": "advertising"}

INVOICE_ANSWER = {
    "correspondent": "Telekom Deutschland GmbH", "document_type": "Rechnung", "date": "17.04.2023",
    "title": "Rechnung Mobilfunk April 2023", "tags": ["Telekommunikation", "Mobilfunk"],
    "fields": {"Betrag": "EUR 44.15", "Rechnungsnummer": "4400 1234 5678", "Kundennummer": "",
               "Vertragsnummer": "", "Fällig am": "2023-04-28"},
    "summary": "Mobilfunkrechnung für April 2023 über 44,15 EUR, Abbuchung am 28.04.2023.",
}
CONTRACT_ANSWER = {
    "correspondent": "Notar Andreas Kaiser", "document_type": "Vertrag", "date": "2026-11-04",
    "title": "Entwurf Überlassungsvertrag", "tags": ["Immobilie"],
    "fields": {"Betrag": "", "Rechnungsnummer": "", "Kundennummer": "", "Vertragsnummer": "1234/2026",
               "Fällig am": ""},
    "summary": "Entwurf eines notariellen Überlassungsvertrags.",
}


class FakeOllama:
    def __init__(self):
        self.calls: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.calls.append({"path": request.url.path, **body})
        path = request.url.path
        if path == "/api/tags":
            names = ["pplx-decider-flash:latest", "pplx-decider:latest", "qwen3.8:latest", "qwen3.6:35b"]
            return httpx.Response(200, json={"models": [{"name": n} for n in names]})
        if path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "vision", "thinking"]})
        if path == "/v1/systemone":
            return httpx.Response(200, json=self._decide(body))
        if path == "/api/chat":
            assert body["think"] is False and body["options"]["temperature"] == 0
            text = body["messages"][-1]["content"]
            answer = CONTRACT_ANSWER if "Überlassungsvertrag" in text else INVOICE_ANSWER
            return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(answer)},
                                             "prompt_eval_count": 900, "eval_count": 80})
        return httpx.Response(404, json={"error": "not found"})

    def _decide(self, body: dict) -> dict:
        if "continuation" in body["questions"]:  # merging: same contract => continuation
            part_a, _, part_b = body["state"].partition("Part B:")
            same = "Überlassungsvertrag" in part_a and "Überlassungsvertrag" in part_b
            return {"answers": {"continuation": {"type": "noul", "noul": 0.95 if same else 0.05}},
                    "usage": {"input_tokens": 300}}
        state = body["state"].split("Document text (excerpt):", 1)[-1]
        key, p = "general", 0.1
        for word, k in RELEVANT_WORDS.items():
            if word in state:
                key, p = k, 0.97
        for word, k in IRRELEVANT_WORDS.items():
            if word in state:
                key, p = k, 0.03
        if "Produktinformationsblatt" in state:  # deliberately uncertain -> escalation
            key, p = ("product_info", 0.1) if body["model"] == "pplx-decider" else ("insurance", 0.5)
        keys = list(body["questions"]["category"]["criteria"])
        probs = {k: (0.9 if k == key else 0.1 / (len(keys) - 1)) for k in keys}
        return {"model": body["model"], "usage": {"input_tokens": 500, "output_tokens": 0},
                "answers": {"relevant": {"type": "noul", "noul": p},
                            "category": {"type": "choice", "choice": key, "probabilities": probs,
                                         "confidence": 0.9}}}


@pytest.fixture
def fake_ollama(monkeypatch):
    fake = FakeOllama()

    def factory(*args, **kwargs):
        return Ollama(*args, transport=httpx.MockTransport(fake.handler), **kwargs)

    for module in (paperclerk.scoring, paperclerk.importer, paperclerk.merge, paperclerk.process):
        monkeypatch.setattr(module, "Ollama", factory)
    return fake


# ------------------------------------------------------------------ fake Paperless (API v10)

def _pdf_text(data: bytes) -> str:
    if not data.startswith(b"%PDF"):
        return ""
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        return "\n".join(page.get_text() for page in pdf)


class FakePaperless:
    def __init__(self):
        self.objects = {
            "correspondents": [{"id": 1, "name": "Telekom", "document_count": 5},
                               {"id": 2, "name": "info@o2.com", "document_count": 3}],
            "document_types": [{"id": 1, "name": "Rechnung"}],
            "tags": [{"id": 1, "name": "new", "is_inbox_tag": True}, {"id": 2, "name": "Versicherung"}],
            "custom_fields": [{"id": 1, "name": "Betrag", "data_type": "monetary"}],
        }
        self.docs: dict[int, dict] = {}
        self.files: dict[int, bytes] = {}
        self.tasks: dict[str, dict] = {}
        self.created: list[tuple[str, dict]] = []
        self.patches: list[tuple[int, dict]] = []
        self.searches: list[str] = []
        self.next_note = 1

    def add_document(self, doc_id: int, title: str, content: str, tags: list[int] | None = None, **extra) -> dict:
        doc = {"id": doc_id, "title": title, "content": content, "tags": tags or [], "correspondent": None,
               "document_type": None, "created": "2026-01-01", "added": "2026-11-20T10:00:00+01:00",
               "original_file_name": f"{title}.pdf", "custom_fields": [], "notes": [], "checksums": set(), **extra}
        self.docs[doc_id] = doc
        return doc

    def _public(self, doc: dict) -> dict:
        return {k: v for k, v in doc.items() if k not in ("checksums", "notes")}

    def handler(self, request: httpx.Request) -> httpx.Response:
        headers = {"X-Version": "3.5.1", "X-Api-Version": "10"}
        path, method, params = request.url.path, request.method, request.url.params
        parts = path.strip("/").split("/")
        kind = parts[1] if len(parts) == 2 else ""
        if kind in self.objects:
            if method == "GET":
                name = params.get("name__iexact")
                items = [o for o in self.objects[kind] if name is None or o["name"].lower() == name.lower()]
                return httpx.Response(200, json={"count": len(items), "next": None, "results": items}, headers=headers)
            data = json.loads(request.content)
            obj = {"id": 100 + len(self.created), **data}
            self.objects[kind].append(obj)
            self.created.append((kind, data))
            return httpx.Response(201, json=obj, headers=headers)
        if path == "/api/documents/" and method == "GET":
            docs = list(self.docs.values())
            if "query" in params:
                self.searches.append(params["query"])
                terms = [t.strip().casefold() for t in params["query"].split(" OR ")]
                docs = [d for d in docs if any(t in d["content"].casefold() for t in terms)]
                docs.sort(key=lambda d: -sum(t in d["content"].casefold() for t in terms))
            if checksum := params.get("checksum__iexact"):
                docs = [d for d in docs if checksum.lower() in d["checksums"]]
            if none := params.get("tags__id__none"):
                docs = [d for d in docs if int(none) not in d["tags"]]
            if any_of := params.get("tags__id__in"):
                wanted = {int(x) for x in any_of.split(",") if x}
                docs = [d for d in docs if wanted & set(d["tags"])]
            docs.sort(key=lambda d: d["id"]) if "query" not in params else None
            size = int(params.get("page_size", 25))
            return httpx.Response(200, json={"count": len(docs), "next": None,
                                             "results": [self._public(d) for d in docs[:size]]}, headers=headers)
        if len(parts) == 3 and parts[1] == "documents" and parts[2].isdigit():
            doc = self.docs[int(parts[2])]
            if method == "PATCH":
                data = json.loads(request.content)
                self.patches.append((doc["id"], data))
                doc.update(data)
            return httpx.Response(200, json=self._public(doc), headers=headers)
        if len(parts) == 4 and parts[3] == "download":
            return httpx.Response(200, content=self.files.get(int(parts[2]), b""), headers=headers)
        if len(parts) == 4 and parts[3] == "notes":
            doc = self.docs[int(parts[2])]
            if method == "POST":
                doc["notes"].append({"id": self.next_note, "note": json.loads(request.content)["note"]})
                self.next_note += 1
            elif method == "DELETE":
                doc["notes"] = [n for n in doc["notes"] if n["id"] != int(params["id"])]
            return httpx.Response(200, json=doc["notes"], headers=headers)
        if path == "/api/documents/post_document/":
            ctype = request.headers["content-type"].encode()
            msg = BytesParser(policy=policy.default).parsebytes(b"Content-Type: " + ctype + b"\r\n\r\n"
                                                                 + request.content)
            fields: dict[str, list] = {}
            content = b""
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                value = part.get_payload(decode=True)
                if name == "document":
                    content = value
                    fields["filename"] = [part.get_filename()]
                else:
                    fields.setdefault(name, []).append(value.decode())
            doc_id = 500 + len(self.docs)
            doc = self.add_document(doc_id, fields["title"][0], _pdf_text(content),
                                    [int(t) for t in fields.get("tags", [])],
                                    original_file_name=fields["filename"][0], upload=fields)
            doc["checksums"] = {hashlib.sha256(content).hexdigest(), hashlib.md5(content).hexdigest()}
            self.files[doc_id] = content
            task_id = str(uuid.uuid4())
            self.tasks[task_id] = {"task_id": task_id, "status": "success", "result_data": {"document_id": doc_id},
                                   "related_document_ids": [doc_id]}
            return httpx.Response(200, json=task_id, headers=headers)
        if path == "/api/tasks/":
            task = self.tasks.get(params.get("task_id"))
            return httpx.Response(200, json={"count": int(bool(task)), "next": None, "results": [task] if task else []},
                                  headers=headers)
        return httpx.Response(404, json={"detail": "Not found."})


@pytest.fixture
def fake_paperless(monkeypatch):
    fake = FakePaperless()

    def from_config(cfg):
        p = cfg.paperless
        return Paperless(p.url, p.token, transport=httpx.MockTransport(fake.handler))

    monkeypatch.setattr(Paperless, "from_config", staticmethod(from_config))
    return fake
