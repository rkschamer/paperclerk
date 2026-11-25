from __future__ import annotations

import imaplib
import logging
import re
import time
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

# envelope line in mbox format, e.g. "From 1781234567890@xxx Wed Mar 15 10:22:33 +0000 2023"
_FROM_LINE = re.compile(rb"^From \S+ +\w{3} \w{3} +\d{1,2} \d\d:\d\d:\d\d")


def iter_mbox(path: Path, start: int = 0) -> Iterator[tuple[int, bytes]]:
    """Yields (byte offset of the From line, raw message), streaming, also for very large files."""
    with path.open("rb") as f:
        f.seek(start)
        first = f.readline()
        # unknown envelope format: then every "From " line after an empty line separates messages
        lenient = first.startswith(b"From ") and not _FROM_LINE.match(first)
        f.seek(start)
        offset = start
        msg_start: int | None = None
        buf: list[bytes] = []
        prev_blank = True
        for line in f:
            if line.startswith(b"From ") and (_FROM_LINE.match(line) if not lenient else prev_blank):
                if msg_start is not None and buf:
                    yield msg_start, b"".join(buf)
                msg_start, buf = offset, []
            elif msg_start is not None:
                buf.append(line)
            prev_blank = not line.strip()
            offset += len(line)
        if msg_start is not None and buf:
            yield msg_start, b"".join(buf)


class ImapError(Exception):
    pass


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class GmailImap:
    """Reads mails from Gmail via IMAP (\\All folder, Gmail search via X-GM-RAW)."""

    def __init__(self, host: str, user: str, password: str, folder: str = ""):
        if not user or not password:
            raise ImapError("IMAP: set [imap] user and password (app password)")
        self.host, self.user, self.password, self.folder = host, user, password, folder
        self.conn: imaplib.IMAP4_SSL | None = None
        self.uidvalidity = ""

    def connect(self) -> None:
        self.conn = imaplib.IMAP4_SSL(self.host)
        try:
            self.conn.login(self.user, self.password)
        except imaplib.IMAP4.error as e:
            raise ImapError(f"IMAP login failed: {e}") from None
        if not self.folder:
            self.folder = self._find_all_mail()
        typ, _ = self.conn.select(_quote(self.folder), readonly=True)
        if typ != "OK":
            raise ImapError(f"Folder {self.folder!r} not found")
        resp = self.conn.response("UIDVALIDITY")[1]
        self.uidvalidity = (resp[0] or b"").decode() if resp and resp[0] else ""

    def _find_all_mail(self) -> str:
        typ, lines = self.conn.list()
        for line in lines or []:
            if isinstance(line, bytes) and b"\\All" in line:
                # the name stays in modified UTF-7, as the server expects it for SELECT
                m = re.search(rb'"([^"]+)"\s*$', line) or re.search(rb"(\S+)\s*$", line)
                if m:
                    return m.group(1).decode("ascii", "replace")
        return "[Gmail]/All Mail"

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.logout()
            except Exception:
                pass
            self.conn = None

    def search(self, query: str) -> list[int]:
        typ, data = self.conn.uid("SEARCH", "X-GM-RAW", _quote(query))
        if typ != "OK":
            raise ImapError(f"IMAP search failed: {data}")
        return sorted(int(x) for x in (data[0] or b"").split())

    def fetch(self, uid: int) -> tuple[bytes, str] | None:
        typ, data = self.conn.uid("FETCH", str(uid), "(X-GM-LABELS BODY.PEEK[])")
        if typ != "OK" or not data:
            return None
        for item in data:
            if isinstance(item, tuple) and len(item) == 2:
                header, raw = item
                m = re.search(rb"X-GM-LABELS \((.*?)\)", header or b"")
                labels = m.group(1).decode("utf-8", "replace") if m else ""
                return raw, labels
        return None

    def iter_messages(self, query: str, after_uid: int = 0, retries: int = 5) -> Iterator[tuple[int, bytes, str]]:
        uids = [u for u in self.search(query) if u > after_uid]
        log.info("IMAP: %d mails matching %r (folder %s)", len(uids), query, self.folder)
        i = 0
        failures = 0
        while i < len(uids):
            uid = uids[i]
            try:
                got = self.fetch(uid)
                failures = 0
            except (imaplib.IMAP4.abort, OSError) as e:
                failures += 1
                if failures > retries:
                    raise ImapError(f"IMAP connection dropped repeatedly: {e}") from None
                log.warning("IMAP connection dropped (%s) – retrying in %ds", e, 10 * failures)
                time.sleep(10 * failures)
                self.close()
                self.connect()
                continue
            i += 1
            if got:
                yield uid, got[0], got[1]
