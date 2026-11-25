from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image

OFFICE_MIME = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "odt": "application/vnd.oasis.opendocument.text",
    "ods": "application/vnd.oasis.opendocument.spreadsheet",
    "odp": "application/vnd.oasis.opendocument.presentation",
    "doc": "application/msword",
    "xls": "application/vnd.ms-excel",
    "ppt": "application/vnd.ms-powerpoint",
    "rtf": "application/rtf",
}
IMAGE_MIME = {"jpg": "image/jpeg", "png": "image/png", "gif": "image/gif", "tiff": "image/tiff",
              "webp": "image/webp"}


@dataclass(frozen=True)
class Sniffed:
    kind: str  # pdf | image | office | zip
    ext: str
    mime: str


def _ext(filename: str | None) -> str:
    return PurePosixPath(filename).suffix.lower().lstrip(".") if filename else ""


def sniff(data: bytes, filename: str | None = None) -> Sniffed | None:
    """Determine the file type from the content (the declared content type is often wrong)."""
    head = data[:16]
    if head.startswith(b"%PDF"):
        return Sniffed("pdf", "pdf", "application/pdf")
    if head.startswith(b"\xff\xd8\xff"):
        return Sniffed("image", "jpg", IMAGE_MIME["jpg"])
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return Sniffed("image", "png", IMAGE_MIME["png"])
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return Sniffed("image", "gif", IMAGE_MIME["gif"])
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return Sniffed("image", "tiff", IMAGE_MIME["tiff"])
    if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return Sniffed("image", "webp", IMAGE_MIME["webp"])
    if head.startswith(b"PK\x03\x04"):
        return _sniff_zip(data)
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        ext = _ext(filename)
        if ext in ("doc", "xls", "ppt"):
            return Sniffed("office", ext, OFFICE_MIME[ext])
        return None
    if head.startswith(b"{\\rtf"):
        return Sniffed("office", "rtf", OFFICE_MIME["rtf"])
    if b"%PDF-" in data[:1024]:  # PDF with leading garbage – only after all other signatures
        return Sniffed("pdf", "pdf", "application/pdf")
    return None


def _sniff_zip(data: bytes) -> Sniffed | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
            if "word/document.xml" in names:
                return Sniffed("office", "docx", OFFICE_MIME["docx"])
            if "xl/workbook.xml" in names:
                return Sniffed("office", "xlsx", OFFICE_MIME["xlsx"])
            if "ppt/presentation.xml" in names:
                return Sniffed("office", "pptx", OFFICE_MIME["pptx"])
            if "mimetype" in names:
                mt = z.read("mimetype")[:100].decode("ascii", "ignore").strip()
                for ext, mime in OFFICE_MIME.items():
                    if mt == mime:
                        return Sniffed("office", ext, mime)
    except (zipfile.BadZipFile, OSError, KeyError, RuntimeError, ValueError):
        return None
    return Sniffed("zip", "zip", "application/zip")


def unpack_zip(data: bytes, max_files: int = 50, max_total: int = 300 * 1024 * 1024) -> list[tuple[str, bytes]]:
    """Read unencrypted files from a ZIP (with protection against ZIP bombs)."""
    out: list[tuple[str, bytes]] = []
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                if info.is_dir() or info.flag_bits & 0x1:
                    continue
                name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
                if not name or name.startswith(".") or "__MACOSX" in info.filename:
                    continue
                total += info.file_size
                if total > max_total or len(out) >= max_files:
                    break
                out.append((name, z.read(info)))
    except (zipfile.BadZipFile, OSError, RuntimeError, ValueError, NotImplementedError, EOFError):
        pass
    return out


def image_size(data: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:
        return None


def hashes(data: bytes) -> tuple[str, str]:
    return hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest()


def store_blob(files_dir: Path, data: bytes, sha: str, ext: str) -> str:
    rel = f"{sha[:2]}/{sha}.{ext}"
    path = files_dir / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return rel
