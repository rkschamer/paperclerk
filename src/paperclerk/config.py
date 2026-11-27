from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    pass


# ---------------------------------------------------------------- vocabulary defaults

@dataclass
class Category:
    """Relevance category for mail attachments (import only)."""
    key: str
    relevant: bool
    description: str


DEFAULT_CATEGORIES = [
    Category("invoice", True, "Invoice, credit note, statement of charges, dunning letter or payment request "
                              "addressed to the recipient"),
    Category("receipt", True, "Till receipt, receipt, payment or donation confirmation"),
    Category("order", True, "Order, booking or job confirmation listing specific items"),
    Category("contract", True, "Contract, contract confirmation or amendment, termination or confirmation of "
                               "termination"),
    Category("insurance", True, "Individual insurance policy, endorsement, premium notice or claim settlement"),
    Category("bank", True, "Bank, securities or credit card statement, loan or other financial document"),
    Category("authority", True, "Notice or letter from a public authority or the tax office, tax certificate"),
    Category("payslip", True, "Payslip, salary statement or employer certificate"),
    Category("health", True, "Doctor's letter, medical findings, prescription, treatment plan, letter from the "
                             "health insurer"),
    Category("ticket", True, "Train, flight or event ticket, boarding pass, travel or hotel booking"),
    Category("certificate", True, "School or job reference, certificate, deed, attendance confirmation or other "
                                  "proof"),
    Category("warranty", True, "Warranty or guarantee document"),
    Category("letter", True, "Individual letter addressed personally to the recipient"),
    Category("other_personal", True, "Other document with a personal connection to the recipient"),
    Category("terms", False, "General terms and conditions, terms of use, delivery or insurance conditions"),
    Category("withdrawal", False, "Withdrawal or cancellation policy, model withdrawal form"),
    Category("privacy", False, "Privacy notice, consent form or GDPR information sheet"),
    Category("product_info", False, "Product information sheet (also for insurance or loans), data sheet, manual, "
                                    "price list or service description"),
    Category("advertising", False, "Advertising, newsletter, brochure, catalogue, invitation or circular"),
    Category("image", False, "Photo, graphic, logo, banner or screenshot that is not a document"),
    Category("general", False, "Other general text without a personal connection to the recipient"),
]

# maildocs used German category keys; stored probabilities are migrated with this map
LEGACY_CATEGORY_KEYS = {
    "rechnung": "invoice", "quittung": "receipt", "bestellung": "order", "vertrag": "contract",
    "versicherung": "insurance", "bank": "bank", "behoerde": "authority", "lohn": "payslip", "gesundheit": "health",
    "ticket": "ticket", "nachweis": "certificate", "garantie": "warranty", "brief": "letter",
    "sonstiges": "other_personal", "agb": "terms", "widerruf": "withdrawal", "datenschutz": "privacy",
    "produktinfo": "product_info", "werbung": "advertising", "bild": "image", "allgemein": "general",
}

RELEVANT_TRUE = ("An individual document issued specifically for the recipient (e.g. invoice, receipt, contract, "
                 "insurance policy, official notice, bank statement, payslip, ticket, certificate) that may later "
                 "be needed as a record or proof.")
RELEVANT_FALSE = ("Not worth keeping: a standard text sent identically to many customers (e.g. terms and "
                  "conditions, withdrawal policy, privacy notice, product information, price list), advertising or "
                  "a newsletter, or an image that is not a document (logo, photo, banner).")

DEFAULT_DOCUMENT_TYPES = ["Rechnung", "Quittung", "Bestellung", "Vertrag", "Versicherung", "Kontoauszug",
                          "Bescheid", "Gehaltsabrechnung", "Gesundheit", "Ticket", "Urkunde", "Garantie", "Brief"]
DEFAULT_TAGS = ["Steuer", "Versicherung", "Gesundheit", "Auto", "Wohnung", "Energie", "Telekommunikation", "Bank",
                "Arbeit", "Einkauf", "Reise", "Familie"]

FIELD_TYPES = ("string", "monetary", "date", "integer", "float", "boolean", "url")


@dataclass
class FieldDef:
    """A Paperless custom field the model fills in."""
    name: str
    type: str
    description: str


DEFAULT_FIELDS = [
    FieldDef("Betrag", "monetary", "Total amount of an invoice, receipt or payment, including VAT"),
    FieldDef("Rechnungsnummer", "string", "Invoice or receipt number as printed (not an order or customer number)"),
    FieldDef("Kundennummer", "string", "Customer or account number the correspondent uses for the recipient (not a personnel number)"),
    FieldDef("Vertragsnummer", "string", "Number of a contract, insurance policy or membership (not an order, invoice, customer or personnel number)"),
    FieldDef("Fällig am", "date", "Payment due date or deadline stated in the document"),
]


# ---------------------------------------------------------------- sections

@dataclass
class PaperlessConfig:
    url: str = "http://localhost:8000"
    token: str = ""
    processed_tag: str = "paperclerk"
    timeout: int = 120
    verify_tls: bool = True


@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434"
    keep_alive: str = "30m"
    timeout: int = 1800


@dataclass
class EnrichConfig:
    model: str = "qwen3.8"
    num_ctx: int = 8192
    think: bool | str = False
    max_chars: int = 7000
    language: str = "German"
    summary: bool = True
    document_types: list[str] = field(default_factory=lambda: list(DEFAULT_DOCUMENT_TYPES))
    tags: list[str] = field(default_factory=lambda: list(DEFAULT_TAGS))
    new_tags: bool = True
    exclude_tags: list[str] = field(default_factory=list)
    known_correspondents: int = 300


@dataclass
class ProcessConfig:
    scope: str = "inbox"
    remove_inbox_tags: bool = True  # when a document is accepted in `paperclerk ui`


@dataclass
class ImportConfig:
    tags: list[str] = field(default_factory=lambda: ["mail-import"])
    review_tag: str = "paperclerk-review"
    kinds: list[str] = field(default_factory=lambda: ["pdf", "image"])
    min_image_kb: int = 15
    min_image_side: int = 400
    max_file_mb: int = 100
    skip_inline_images: bool = True
    unpack_zip: bool = True
    skip_labels: list[str] = field(default_factory=lambda: ["Spam", "Trash", "Papierkorb"])
    content_check: bool = True
    content_threshold: float = 0.75
    max_pending: int = 8


@dataclass
class ScoringConfig:
    model: str = "pplx-decider"
    escalation_model: str = ""
    accept: float = 0.75
    reject: float = 0.15
    type_weight: float = 0.5
    max_chars: int = 6000
    relevant_true: str = RELEVANT_TRUE
    relevant_false: str = RELEVANT_FALSE


@dataclass
class MergeConfig:
    enabled: bool = True
    threshold: float = 0.35
    max_chars: int = 1500


@dataclass
class ImapConfig:
    host: str = "imap.gmail.com"
    user: str = ""
    password: str = ""
    folder: str = ""
    query: str = "has:attachment"


@dataclass
class TextConfig:
    max_pages: int = 3
    ocr: bool = True
    tessdata_dir: str = ""
    ocr_languages: str = "deu+eng"
    min_chars_per_page: int = 50
    vision_max_side: int = 1280


@dataclass
class Config:
    path: Path
    data_dir: Path
    paperless: PaperlessConfig
    ollama: OllamaConfig
    enrich: EnrichConfig
    fields: list[FieldDef]
    aliases: dict[str, str]
    process: ProcessConfig
    imports: ImportConfig
    scoring: ScoringConfig
    categories: list[Category]
    merge: MergeConfig
    imap: ImapConfig
    text: TextConfig

    @property
    def base_dir(self) -> Path:
        return self.path.resolve().parent

    @property
    def db_path(self) -> Path:
        return self.data_dir / "paperclerk.sqlite"

    @property
    def legacy_db_path(self) -> Path:
        return self.data_dir / "maildocs.sqlite"

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def tessdata(self) -> Path:
        if self.text.tessdata_dir:
            return self.resolve(self.text.tessdata_dir)
        return self.data_dir / "tessdata"

    def resolve(self, p: str | Path) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def category(self, key: str | None) -> Category | None:
        return next((c for c in self.categories if c.key == key), None)

    def system_tags(self) -> set[str]:
        """Tags paperclerk manages itself (casefolded); the model never assigns them."""
        names = [self.paperless.processed_tag, self.imports.review_tag, *self.imports.tags, *self.enrich.exclude_tags]
        return {n.casefold() for n in names if n}


# ---------------------------------------------------------------- loading

TOP_LEVEL = {"data_dir", "paperless", "ollama", "enrich", "aliases", "process", "import", "imap", "text"}
LEGACY_SECTIONS = {"paths", "decider", "extractor", "filter", "mbox", "doctype", "merge"}


def _section(cls, raw: dict, name: str, extra: tuple[str, ...] = ()):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known - set(extra)
    if unknown:
        raise ConfigError(f"Unknown keys in [{name}]: {', '.join(sorted(unknown))}")
    obj = cls()
    for key, value in raw.items():
        if key in extra:
            continue
        default = getattr(obj, key)
        if isinstance(default, bool) and not isinstance(value, (bool, str)):
            raise ConfigError(f"[{name}] {key} must be true or false")
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(f"[{name}] {key} must be a number")
            value = type(default)(value)
        if isinstance(default, list) and not isinstance(value, list):
            raise ConfigError(f"[{name}] {key} must be a list")
        setattr(obj, key, value)
    return obj


def _categories(raw: list) -> list[Category]:
    if not raw:
        return [Category(c.key, c.relevant, c.description) for c in DEFAULT_CATEGORIES]
    result, seen = [], set()
    for i, entry in enumerate(raw, 1):
        try:
            c = Category(str(entry["key"]).strip(), bool(entry["relevant"]), str(entry["description"]).strip())
        except KeyError as e:
            raise ConfigError(f"[[import.category]] no. {i}: missing {e}") from None
        if not c.key or c.key in seen:
            raise ConfigError(f"[[import.category]] no. {i}: key empty or duplicate ({c.key!r})")
        seen.add(c.key)
        result.append(c)
    if not any(c.relevant for c in result) or all(c.relevant for c in result):
        raise ConfigError("[[import.category]]: there must be relevant and irrelevant categories")
    return result


def _fields(raw: list | None) -> list[FieldDef]:
    if raw is None:
        return [FieldDef(f.name, f.type, f.description) for f in DEFAULT_FIELDS]
    result = []
    for i, entry in enumerate(raw, 1):
        try:
            f = FieldDef(str(entry["name"]).strip(), str(entry["type"]).strip(), str(entry["description"]).strip())
        except KeyError as e:
            raise ConfigError(f"[[enrich.field]] no. {i}: missing {e}") from None
        if f.type not in FIELD_TYPES:
            raise ConfigError(f"[[enrich.field]] {f.name}: type must be one of {', '.join(FIELD_TYPES)}")
        result.append(f)
    return result


def load(path: Path) -> Config:
    if not path.exists():
        raise ConfigError(f"{path} not found. Create it with `paperclerk init`.")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
    if LEGACY_SECTIONS & set(raw):
        raise ConfigError(f"{path} is a maildocs configuration. Convert it with `paperclerk init` "
                          "(the old file is kept as a backup).")
    unknown = set(raw) - TOP_LEVEL
    if unknown:
        raise ConfigError(f"Unknown sections: {', '.join(sorted(unknown))}")

    base = path.resolve().parent
    data_dir = Path(raw.get("data_dir", "data")).expanduser()
    data_dir = data_dir if data_dir.is_absolute() else (base / data_dir).resolve()
    enrich_raw = raw.get("enrich", {})
    import_raw = raw.get("import", {})
    cfg = Config(
        path=path,
        data_dir=data_dir,
        paperless=_section(PaperlessConfig, raw.get("paperless", {}), "paperless"),
        ollama=_section(OllamaConfig, raw.get("ollama", {}), "ollama"),
        enrich=_section(EnrichConfig, enrich_raw, "enrich", ("field",)),
        fields=_fields(enrich_raw.get("field")),
        aliases={str(k): str(v) for k, v in raw.get("aliases", {}).items()},
        process=_section(ProcessConfig, raw.get("process", {}), "process"),
        imports=_section(ImportConfig, import_raw, "import", ("scoring", "category", "merge")),
        scoring=_section(ScoringConfig, import_raw.get("scoring", {}), "import.scoring"),
        categories=_categories(import_raw.get("category", [])),
        merge=_section(MergeConfig, import_raw.get("merge", {}), "import.merge"),
        imap=_section(ImapConfig, raw.get("imap", {}), "imap"),
        text=_section(TextConfig, raw.get("text", {}), "text"),
    )
    env = os.environ
    cfg.paperless.url = env.get("PAPERLESS_URL", cfg.paperless.url).rstrip("/")
    cfg.paperless.token = env.get("PAPERLESS_TOKEN", cfg.paperless.token)
    cfg.imap.user = env.get("GMAIL_USER", cfg.imap.user)
    cfg.imap.password = env.get("GMAIL_APP_PASSWORD", cfg.imap.password)

    s = cfg.scoring
    if not 0 <= s.reject <= s.accept <= 1:
        raise ConfigError("[import.scoring] requires 0 <= reject <= accept <= 1")
    for name, value in (("[import.scoring] type_weight", s.type_weight), ("[import.merge] threshold",
                        cfg.merge.threshold), ("[import] content_threshold", cfg.imports.content_threshold)):
        if not 0 <= value <= 1:
            raise ConfigError(f"{name} must be between 0 and 1")
    if cfg.process.scope not in ("inbox", "all"):
        raise ConfigError('[process] scope must be "inbox" or "all"')
    bad = set(cfg.imports.kinds) - {"pdf", "image", "office"}
    if bad:
        raise ConfigError(f"[import] kinds: unknown {sorted(bad)} (allowed: pdf, image, office)")
    if not cfg.paperless.processed_tag:
        raise ConfigError("[paperless] processed_tag must not be empty")
    return cfg


def template_text() -> str:
    return resources.files("paperclerk").joinpath("config.example.toml").read_text(encoding="utf-8")


# ---------------------------------------------------------------- maildocs conversion

def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


# (legacy section, legacy key) -> (new section, new key); "" = top level
LEGACY_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("paths", "data_dir"): ("", "data_dir"),
    **{("paperless", k): ("paperless", k) for k in ("url", "token", "timeout", "verify_tls")},
    ("paperless", "import_tags"): ("import", "tags"),
    ("paperless", "review_tag"): ("import", "review_tag"),
    ("paperless", "max_pending"): ("import", "max_pending"),
    ("paperless", "content_check"): ("import", "content_check"),
    ("paperless", "content_threshold"): ("import", "content_threshold"),
    **{("ollama", k): ("ollama", k) for k in ("url", "keep_alive", "timeout")},
    **{("decider", k): ("import.scoring", k) for k in ("model", "escalation_model", "accept", "reject",
                                                       "type_weight", "max_chars")},
    **{("extractor", k): ("enrich", k) for k in ("model", "num_ctx", "think", "max_chars",
                                                 "known_correspondents", "exclude_tags")},
    **{("filter", k): ("import", k) for k in ("kinds", "min_image_kb", "min_image_side", "max_file_mb",
                                              "skip_inline_images", "unpack_zip")},
    ("mbox", "skip_labels"): ("import", "skip_labels"),
    **{("text", k): ("text", k) for k in ("max_pages", "ocr", "tessdata_dir", "ocr_languages",
                                          "min_chars_per_page", "vision_max_side")},
    **{("imap", k): ("imap", k) for k in ("host", "user", "password", "folder", "query")},
    **{("merge", k): ("import.merge", k) for k in ("enabled", "threshold", "max_chars")},
}
_SECTION_DEFAULTS = {"": {"data_dir": "data"}, "paperless": PaperlessConfig(), "ollama": OllamaConfig(),
                     "enrich": EnrichConfig(), "import": ImportConfig(), "import.scoring": ScoringConfig(),
                     "import.merge": MergeConfig(), "imap": ImapConfig(), "text": TextConfig()}


def _default(section: str, key: str) -> Any:
    d = _SECTION_DEFAULTS[section]
    return d.get(key) if isinstance(d, dict) else getattr(d, key, None)


def convert_legacy(text: str) -> tuple[str, list[str]]:
    """maildocs config.toml -> paperclerk config.toml. Returns (new text, notes for the user)."""
    raw = tomllib.loads(text)
    values: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    for (section, key), (new_section, new_key) in LEGACY_MAP.items():
        if key in raw.get(section, {}):
            value = raw[section][key]
            if value != _default(new_section, new_key):
                values.setdefault(new_section, {})[new_key] = value
    tags = raw.get("extractor", {}).get("tags") or []
    if tags:
        values.setdefault("enrich", {})["tags"] = sorted({*DEFAULT_TAGS, *tags})
    if raw.get("mbox", {}).get("files"):
        notes.append("[mbox] files is gone – pass the file instead: paperclerk import mbox <file>")
    if raw.get("paperless", {}).get("auto_matching") is not None:
        notes.append("Objects paperclerk creates in Paperless now always get matching 'none' (see docs/design.md)")

    out, section = [], ""
    for line in template_text().splitlines():
        header = re.match(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*$", line)
        if header:
            section = header.group(1)
            out.append(line)
            if section == "aliases":
                out += [f"{toml_value(k)} = {toml_value(v)}" for k, v in raw.get("aliases", {}).items()]
            continue
        assign = re.match(r"^(#\s*)?([A-Za-z_]+)\s*=", line)
        if assign and assign.group(2) in values.get(section, {}):
            key = assign.group(2)
            out.append(f"{key} = {toml_value(values[section].pop(key))}")
            continue
        out.append(line)
    leftovers = {s: v for s, v in values.items() if v}
    if leftovers:
        raise ConfigError(f"conversion incomplete, template lacks: {leftovers}")

    legacy = raw.get("doctype", [])
    if legacy:
        keys = {(d.get("key"), bool(d.get("relevant"))) for d in legacy}
        mapped = {(LEGACY_CATEGORY_KEYS.get(k), r) for k, r in keys}
        if mapped != {(c.key, c.relevant) for c in DEFAULT_CATEGORIES}:
            notes.append("Your [[doctype]] list differed from the defaults: it was kept as [[import.category]]")
            cut = next(i for i, line in enumerate(out) if line.strip() == "[[import.category]]")
            out = out[:cut]
            for d in legacy:
                out += ["[[import.category]]", f"key = {toml_value(LEGACY_CATEGORY_KEYS.get(d['key'], d['key']))}",
                        f"relevant = {toml_value(bool(d['relevant']))}",
                        f"description = {toml_value(d['description'])}", ""]
        notes.append("Document types are now chosen from Paperless plus [enrich] document_types")
    return "\n".join(out).rstrip() + "\n", notes
