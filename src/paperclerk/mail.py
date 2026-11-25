from __future__ import annotations

import email
import html
import re
from dataclasses import dataclass
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime

# quick filter on raw bytes: mails without any hint of an attachment are not parsed at all
QUICK_ATTACHMENT_RE = re.compile(
    rb"(?im)^content-disposition:\s*attachment|filename\*?=|\bname\*?=|^content-type:\s*(?:application|image)/"
)

_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"[^\S\n]+")
_NL_RE = re.compile(r"\n\s*\n+")


@dataclass
class MailMeta:
    message_id: str
    date: str
    from_name: str
    from_addr: str
    to_addr: str
    subject: str
    labels: str
    body: str


@dataclass
class Part:
    part: str
    filename: str | None
    content_type: str
    disposition: str | None
    content_id: str | None
    data: bytes


def has_attachment_hint(raw: bytes) -> bool:
    return QUICK_ATTACHMENT_RE.search(raw) is not None


def parse(raw: bytes) -> tuple[MailMeta, list[Part]]:
    msg = email.message_from_bytes(raw, policy=policy.default)
    from_name, from_addr = _address(msg, "From")
    meta = MailMeta(
        message_id=_header(msg, "Message-ID"),
        date=_date(msg),
        from_name=from_name,
        from_addr=from_addr,
        to_addr=", ".join(a for _, a in _addresses(msg, "To") if a)[:500],
        subject=_header(msg, "Subject"),
        labels=_header(msg, "X-Gmail-Labels"),
        body=_body(msg),
    )
    return meta, list(_parts(msg))


def _decode_raw(value: str) -> str:
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _header(msg: Message, name: str) -> str:
    try:
        value = msg.get(name)
        if value is None:
            return ""
        return _clean_line(str(value))
    except Exception:
        for key, value in msg.raw_items():
            if key.lower() == name.lower():
                return _clean_line(_decode_raw(str(value)))
        return ""


def _addresses(msg: Message, name: str) -> list[tuple[str, str]]:
    value = _header(msg, name)
    if not value:
        return []
    try:
        return [(_clean_line(n), a.strip().lower()) for n, a in getaddresses([value])]
    except Exception:
        return [("", value)]


def _address(msg: Message, name: str) -> tuple[str, str]:
    found = _addresses(msg, name)
    return found[0] if found else ("", "")


def _date(msg: Message) -> str:
    raw = None
    for key, value in msg.raw_items():
        if key.lower() == "date":
            raw = str(value)
            break
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw.strip()).isoformat(timespec="seconds")
    except Exception:
        return ""


def _clean_line(s: str) -> str:
    return _WS_RE.sub(" ", s.replace("\n", " ").replace("\x00", "")).strip()


def html_to_text(s: str) -> str:
    s = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return html.unescape(s)


def compact_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\x00", "")
    s = "\n".join(_WS_RE.sub(" ", line).strip() for line in s.split("\n"))
    return _NL_RE.sub("\n\n", s).strip()


def _body(msg: Message, limit: int = 2000) -> str:
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is None:
            return ""
        content = part.get_content()
        if not isinstance(content, str):
            return ""
        if part.get_content_type() == "text/html":
            content = html_to_text(content)
        content = _URL_RE.sub("[Link]", content)
        return compact_text(content)[:limit]
    except Exception:
        return ""


def _filename(part: Message) -> str | None:
    try:
        name = part.get_filename()
    except Exception:
        name = None
    if not name:
        return None
    name = str(name)
    if "=?" in name:
        name = _decode_raw(name)
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f]", "", name).strip().strip('"').strip()
    return name[:200] or None


def _parts(msg: Message):
    for index, part in enumerate(msg.walk()):
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        filename = _filename(part)
        if not filename and not ctype.startswith(("application/", "image/")):
            continue
        try:
            disposition = part.get_content_disposition()
        except Exception:
            disposition = None
        try:
            data = part.get_payload(decode=True)
        except Exception:
            data = None
        if not data:
            continue
        cid = part.get("Content-ID")
        yield Part(str(index), filename, ctype, disposition, str(cid) if cid else None, data)
