from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

log = logging.getLogger("paperclerk")


def fmt_duration(secs: float) -> str:
    secs = int(max(0, secs))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s"
    if secs < 86400:
        return f"{secs // 3600}h{secs % 3600 // 60:02d}m"
    return f"{secs // 86400}d{secs % 86400 // 3600:02d}h"


def safe_filename(name: str, maxlen: int = 120) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if len(name) > maxlen:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 5:
            name = stem[: maxlen - len(ext) - 1] + "." + ext
        else:
            name = name[:maxlen]
    return name or "file"


class Progress:
    """Progress output with an estimate of the remaining time."""

    def __init__(self, total: int, label: str):
        self.total, self.label = total, label
        self.i = 0
        self.t0 = time.monotonic()
        self.last = 0.0

    def step(self, msg: str, secs: float, quiet: bool = False) -> None:
        self.i += 1
        now = time.monotonic()
        avg = (now - self.t0) / self.i
        eta = avg * (self.total - self.i)
        if quiet and now - self.last < 10 and self.i != self.total:
            return
        self.last = now
        log.info("[%d/%d] %s: %s (%.1fs, ~%s left)", self.i, self.total, self.label, msg, secs, fmt_duration(eta))

    def done(self) -> None:
        if self.i:
            elapsed = time.monotonic() - self.t0
            log.info("%s: %d documents in %s (avg %.1fs)", self.label, self.i, fmt_duration(elapsed), elapsed / self.i)


def setup_logging(log_file: Path | None, verbose: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
    for noisy in ("httpx", "httpcore", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
