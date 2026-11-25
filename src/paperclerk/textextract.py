from __future__ import annotations

import base64
import html
import io
import logging
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps

from .config import Config
from .db import DB
from .util import Progress

log = logging.getLogger(__name__)

for _fn in ("mupdf_display_errors", "mupdf_display_warnings"):
    if hasattr(pymupdf.TOOLS, _fn):
        getattr(pymupdf.TOOLS, _fn)(False)

MAX_STORED_CHARS = 20000


@dataclass
class TextResult:
    text: str = ""
    pages: int | None = None
    method: str = "none"  # pdf | ocr | office | none (=> Vision)
    skip_reason: str | None = None


def compact(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n")
    lines = (re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def usable(text: str, min_chars: int) -> bool:
    s = text.strip()
    if len(s) < min_chars:
        return False
    dense = re.sub(r"\s", "", s)
    letters = sum(c.isalpha() for c in dense)
    return letters / max(1, len(dense)) >= 0.4 and s.count("\ufffd") < 0.05 * len(s)


def load_frames(path: Path, max_frames: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        for i in range(min(n, max_frames)):
            im.seek(i)
            frame = ImageOps.exif_transpose(im) if i == 0 else im.copy()
            frames.append(frame.convert("RGB"))
    return frames


def _jpeg_b64(im: Image.Image, max_side: int) -> str:
    im = im.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_for_vision(path: Path, kind: str, max_side: int, max_pages: int = 1) -> list[str]:
    """First page(s) as JPEG/base64 for models with image input."""
    images: list[Image.Image] = []
    try:
        if kind == "pdf":
            with pymupdf.open(path) as doc:
                if doc.needs_pass and not doc.authenticate(""):
                    return []
                for page in list(doc)[:max_pages]:
                    zoom = max_side / max(page.rect.width, page.rect.height, 1)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                    images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        elif kind == "image":
            images = load_frames(path, max_pages)
    except Exception as e:  # noqa: BLE001
        log.debug("Rendering failed for %s: %s", path.name, e)
        return []
    return [_jpeg_b64(im, max_side) for im in images]


def render_page(path: Path, kind: str, index: int, max_side: int) -> str | None:
    """A single page (index -1 = last) as JPEG/base64."""
    try:
        if kind == "pdf":
            with pymupdf.open(path) as doc:
                if doc.needs_pass and not doc.authenticate(""):
                    return None
                page = doc[index]
                zoom = max_side / max(page.rect.width, page.rect.height, 1)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        elif kind == "image":
            im = load_frames(path, 1000)[index]
        else:
            return None
    except Exception as e:  # noqa: BLE001
        log.debug("Rendering page %d failed for %s: %s", index, path.name, e)
        return None
    return _jpeg_b64(im, max_side)


class TextExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tessdata: str | None = None
        if cfg.text.ocr:
            td = cfg.tessdata
            langs = [x for x in cfg.text.ocr_languages.split("+") if x]
            if all((td / f"{x}.traineddata").is_file() for x in langs):
                self.tessdata = str(td)
            else:
                log.warning("OCR language data missing in %s – scans go to the model as images. "
                            "Fix: paperclerk init", td)

    def extract(self, path: Path, kind: str, ext: str) -> TextResult:
        if kind == "pdf":
            return self._pdf(path)
        if kind == "image":
            return self._image(path)
        if kind == "office":
            return self._office(path, ext)
        return TextResult(skip_reason=f"unbekannte Art {kind}")

    def _finish(self, text: str, pages: int | None, method: str) -> TextResult:
        text = compact(text)
        if not usable(text, self.cfg.text.min_chars_per_page):
            method = "none"
        return TextResult(text=text[:MAX_STORED_CHARS], pages=pages, method=method)

    def page_text(self, path: Path, kind: str, index: int) -> str:
        """Text of a single page (index -1 = last), with OCR if needed."""
        try:
            if kind == "pdf":
                with pymupdf.open(path) as doc:
                    if doc.needs_pass and not doc.authenticate(""):
                        return ""
                    page = doc[index]
                    text = page.get_text("text")
                    if self.tessdata and not usable(text, self.cfg.text.min_chars_per_page):
                        text = self._ocr_page(page)
                    return compact(text)
            if kind == "image" and self.tessdata:
                return compact(self._ocr_image(load_frames(path, 1000)[index]))
        except Exception as e:  # noqa: BLE001
            log.debug("Page %d of %s: %s", index, path.name, e)
        return ""

    def _ocr_page(self, page: pymupdf.Page, dpi: int = 300) -> str:
        longest = max(page.rect.width, page.rect.height, 1)
        dpi = int(min(dpi, 4000 * 72 / longest))
        tp = page.get_textpage_ocr(language=self.cfg.text.ocr_languages, dpi=dpi, full=True,
                                   tessdata=self.tessdata)
        return page.get_text("text", textpage=tp)

    def _pdf(self, path: Path) -> TextResult:
        try:
            doc = pymupdf.open(path)
        except Exception:  # noqa: BLE001
            return TextResult(skip_reason="broken PDF")
        with doc:
            if doc.needs_pass and not doc.authenticate(""):
                return TextResult(skip_reason="password-protected PDF")
            if doc.page_count == 0:
                return TextResult(pages=0, skip_reason="PDF without pages")
            min_chars = self.cfg.text.min_chars_per_page
            texts, used_ocr = [], False
            for i in range(min(doc.page_count, self.cfg.text.max_pages)):
                try:
                    page = doc[i]
                    text = page.get_text("text")
                    if self.tessdata and not usable(text, min_chars):
                        ocr = self._ocr_page(page)
                        if len(ocr.strip()) > len(text.strip()):
                            text, used_ocr = ocr, True
                except Exception as e:  # noqa: BLE001
                    log.debug("Page %d of %s: %s", i + 1, path.name, e)
                    text = ""
                texts.append(text)
            return self._finish("\n\n".join(texts), doc.page_count, "ocr" if used_ocr else "pdf")

    def _image(self, path: Path) -> TextResult:
        try:
            frames = load_frames(path, self.cfg.text.max_pages)
        except Exception:  # noqa: BLE001
            return TextResult(skip_reason="broken image")
        if not self.tessdata:
            return TextResult(pages=len(frames), method="none")
        texts = []
        for im in frames:
            try:
                texts.append(self._ocr_image(im))
            except Exception as e:  # noqa: BLE001
                log.debug("OCR %s: %s", path.name, e)
        return self._finish("\n\n".join(texts), len(frames), "ocr")

    def _ocr_image(self, im: Image.Image) -> str:
        im = im.copy()
        im.thumbnail((3000, 3000))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        with pymupdf.open() as doc:
            page = doc.new_page(width=im.width, height=im.height)
            page.insert_image(page.rect, stream=buf.getvalue())
            scale = min(2.5, max(1.0, 2000 / max(im.width, im.height)))
            return self._ocr_page(page, dpi=int(72 * scale))

    def _office(self, path: Path, ext: str) -> TextResult:
        text = ""
        try:
            if ext in ("docx", "xlsx", "pptx", "odt", "ods", "odp"):
                with zipfile.ZipFile(path) as z:
                    names = z.namelist()
                    if ext == "docx":
                        parts = ["word/document.xml"]
                    elif ext == "xlsx":
                        parts = [n for n in names if n == "xl/sharedStrings.xml"]
                    elif ext == "pptx":
                        parts = sorted(n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n))
                    else:
                        parts = ["content.xml"]
                    xml = "\n".join(z.read(n).decode("utf-8", "ignore") for n in parts if n in names)
                xml = re.sub(r"</(w:p|a:p|text:p|text:h|si)>", "\n", xml)
                xml = re.sub(r"<w:tab/>", "\t", xml)
                text = html.unescape(re.sub(r"<[^>]+>", "", xml))
            elif ext == "rtf":
                raw = path.read_bytes().decode("latin-1", "ignore")
                raw = re.sub(r"\\par[d]?\b", "\n", raw)
                raw = re.sub(r"\{\\\*[^{}]*\}|\\[a-z]+-?\d* ?|\\'[0-9a-f]{2}|[{}]", "", raw)
                text = raw
        except Exception as e:  # noqa: BLE001
            log.debug("Office text %s: %s", path.name, e)
        return self._finish(text, None, "office")


def run_extract(db: DB, cfg: Config, limit: int | None = None) -> int:
    rows = db.q(
        """SELECT sha256, kind, ext, path, size, filename FROM documents
            WHERE text_method IS NULL AND skip_reason IS NULL ORDER BY rowid"""
        + (f" LIMIT {int(limit)}" if limit else "")
    )
    if not rows:
        return 0
    ex = TextExtractor(cfg)
    progress = Progress(len(rows), "Text")
    max_bytes = cfg.imports.max_file_mb * 1024 * 1024
    for row in rows:
        t0 = time.monotonic()
        if row["kind"] not in cfg.imports.kinds:
            res = TextResult(skip_reason=f"kind '{row['kind']}' disabled")
        elif row["size"] > max_bytes:
            res = TextResult(skip_reason=f"larger than {cfg.imports.max_file_mb} MB")
        else:
            res = ex.extract(cfg.files_dir / row["path"], row["kind"], row["ext"])
        if res.skip_reason:
            db.update_doc(row["sha256"], skip_reason=res.skip_reason)
        else:
            db.update_doc(row["sha256"], text=res.text, pages=res.pages, text_method=res.method)
        db.commit()
        info = res.skip_reason or f"{res.method}, {len(res.text)} chars"
        progress.step(f"{row['filename']}: {info}", time.monotonic() - t0, quiet=True)
    progress.done()
    return len(rows)
