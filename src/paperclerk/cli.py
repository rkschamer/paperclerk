from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import tomllib
from collections import Counter
from pathlib import Path

import httpx

from . import __version__
from .config import LEGACY_SECTIONS, Config, ConfigError, convert_legacy, load, template_text
from .db import DB
from .decision import PART, UPLOAD_DECISIONS, category_key, meta, overrides, probability, state, type_probs
from .importer import run_import
from .ollama import Ollama, OllamaError, OllamaUnavailable
from .paperless import Paperless, PaperlessError, sync_cache
from .process import run_process
from .sources import GmailImap, ImapError
from .upload import build_plan
from .util import fmt_duration, setup_logging

log = logging.getLogger("paperclerk")


# ---------------------------------------------------------------- init / check

def _download_ocr(cfg: Config) -> None:
    target = cfg.tessdata
    for lang in [x for x in cfg.text.ocr_languages.split("+") if x]:
        dest = target / f"{lang}.traineddata"
        if dest.is_file():
            continue
        url = f"https://github.com/tesseract-ocr/tessdata_fast/raw/main/{lang}.traineddata"
        try:
            r = httpx.get(url, follow_redirects=True, timeout=120)
            r.raise_for_status()
        except httpx.HTTPError as e:
            print(f"OCR data {lang} not downloaded ({e}); run `paperclerk init` again later")
            continue
        target.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        print(f"OCR data {lang}.traineddata downloaded ({len(r.content) / 1e6:.1f} MB)")


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.config)
    if not path.exists():
        path.write_text(template_text(), encoding="utf-8")
        print(f"Created {path}.")
    else:
        text = path.read_text(encoding="utf-8")
        if LEGACY_SECTIONS & set(tomllib.loads(text)):
            backup = path.with_name(f"{path.stem}.maildocs{path.suffix}")
            if backup.exists():
                print(f"{backup} already exists – move it away first.")
                return 1
            new_text, notes = convert_legacy(text)
            path.rename(backup)
            path.write_text(new_text, encoding="utf-8")
            print(f"Converted the maildocs configuration to {path} (old file: {backup.name}).")
            for note in notes:
                print(f"  - {note}")
        else:
            print(f"{path} already exists.")
    try:
        cfg = load(path)
    except ConfigError as e:
        print(f"Error: {e}")
        return 2
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    if cfg.text.ocr:
        _download_ocr(cfg)
    print("Next: set [paperless] url and token, then run `paperclerk check`.")
    return 0


def cmd_check(args: argparse.Namespace, db: DB, cfg: Config) -> int:
    ok = True
    print(f"Configuration: {cfg.path.resolve()}\nData:          {cfg.data_dir}")
    langs = [x for x in cfg.text.ocr_languages.split("+") if x]
    missing = [x for x in langs if not (cfg.tessdata / f"{x}.traineddata").is_file()]
    if not cfg.text.ocr:
        print("OCR:           off (scans go to the models as images)")
    elif missing:
        print(f"OCR:           language data missing ({', '.join(missing)}) -> paperclerk init")
    else:
        print(f"OCR:           ok ({cfg.text.ocr_languages})")

    ollama = Ollama(cfg.ollama.url, cfg.ollama.keep_alive, cfg.ollama.timeout)
    try:
        print(f"Ollama:        {cfg.ollama.url} (version {ollama.version()})")
        for role, model in (("metadata", cfg.enrich.model), ("relevance", cfg.scoring.model),
                            ("escalation", cfg.scoring.escalation_model)):
            if not model:
                continue
            present = ollama.has_model(model)
            ok &= present
            print(f"  {role:<11}  {model}: {'ok' if present else f'MISSING -> ollama pull {model}'}")
    except OllamaError as e:
        ok = False
        print(f"Ollama:        ERROR {e}")
    finally:
        ollama.close()

    try:
        with Paperless.from_config(cfg) as pl:
            info = pl.server_info()
            objs = sync_cache(db, pl)
        print(f"Paperless:     {cfg.paperless.url} (version {info['version']}, API {info['api_version']}, "
              f"{info['documents']} documents)")
        names = {kind: {o["name"].casefold() for o in objs[kind]} for kind in objs}
        inbox = [o["name"] for o in objs["tags"] if o.get("is_inbox_tag")]
        tag = cfg.paperless.processed_tag
        print(f"  processed tag  '{tag}': {'exists' if tag.casefold() in names['tags'] else 'created on first use'}")
        if cfg.process.scope == "inbox":
            print(f"  inbox tags     {', '.join(inbox) if inbox else 'NONE – `process` would find nothing'}")
            ok &= bool(inbox)
        new_types = [t for t in cfg.enrich.document_types if t.casefold() not in names["document_types"]]
        new_fields = [f.name for f in cfg.fields if f.name.casefold() not in names["custom_fields"]]
        if new_types:
            print(f"  created on first use: document types {', '.join(new_types)}")
        if new_fields:
            print(f"  created on first use: custom fields {', '.join(new_fields)}")
    except PaperlessError as e:
        ok = False
        print(f"Paperless:     ERROR {e}")

    if cfg.imap.user:
        imap = GmailImap(cfg.imap.host, cfg.imap.user, cfg.imap.password, cfg.imap.folder)
        try:
            imap.connect()
            print(f"IMAP:          {cfg.imap.user}: ok (folder {imap.folder})")
        except (ImapError, OSError) as e:
            ok = False
            print(f"IMAP:          ERROR {e}")
        finally:
            imap.close()
    return 0 if ok else 1


# ---------------------------------------------------------------- import / process

def interactive() -> bool:
    """A person is watching: the terminal UI can run (not with redirected output, e.g. a scheduled task)."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def cmd_import(args: argparse.Namespace, db: DB, cfg: Config) -> int:
    paths = [Path(p) for p in getattr(args, "files", [])]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"MBOX file not found: {path.resolve()}")
    if interactive() and not args.dry_run:
        from .importui import run_import_ui  # noqa: PLC0415 – Textual is only loaded for the UI

        return run_import_ui(cfg, args.source, paths, args.limit)
    counts = run_import(db, cfg, args.source, paths, args.dry_run, args.limit)
    if counts:
        print("Upload: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 0


def cmd_process(args: argparse.Namespace, db: DB, cfg: Config) -> int:
    counts = run_process(db, cfg, args.dry_run, args.limit)
    if counts:
        print("Process: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        if args.dry_run:
            print("Dry run: nothing written. Run without --dry-run to apply these suggestions.")
    return 0


def cmd_ui(args: argparse.Namespace, db: DB, cfg: Config) -> int:
    from .ui import run_ui  # noqa: PLC0415 – Textual is only loaded for the UI

    return run_ui(cfg)


# ---------------------------------------------------------------- status / show

def _median(db: DB, col: str) -> float | None:
    """Typical time per document (median, so the one-off model load does not distort it)."""
    values = [r[0] for r in db.q(f"SELECT {col} FROM documents WHERE {col} IS NOT NULL")]
    return statistics.median(values) if values else None


def cmd_status(args: argparse.Namespace, db: DB, cfg: Config) -> int:
    rows = db.q("SELECT * FROM documents")
    if rows:
        for s in db.q("SELECT * FROM sources ORDER BY id"):
            print(f"Source {s['kind']}: {s['locator']} – {'complete' if s['done'] else 'at ' + str(s['cursor'])}")
        n_msgs = db.one("SELECT COUNT(*) n FROM messages")["n"]
        att = db.q("SELECT skip_reason, COUNT(*) n FROM attachments WHERE part NOT LIKE 'merge:%' "
                   "GROUP BY skip_reason ORDER BY n DESC")
        print(f"Import: {n_msgs} mails with attachments, {sum(r['n'] for r in att)} attachments, "
              f"{len(rows)} unique documents")
        skipped = [(r["skip_reason"], r["n"]) for r in att if r["skip_reason"]]
        if skipped:
            print("  filtered out: " + ", ".join(f"{reason} {n}" for reason, n in skipped[:8]))
        states = Counter(state(r, cfg) for r in rows)
        print("  " + " | ".join(f"{k}: {v}" for k, v in states.most_common()))
        merged = sum(1 for r in rows if r["parts"])
        if merged:
            print(f"  merged: {merged} documents from {states[PART]} parts")
        wanted = [r for r in rows if r["text_method"] and state(r, cfg) in UPLOAD_DECISIONS]
        from .enrich import VERSION  # noqa: PLC0415

        print(f"  metadata: {sum(1 for r in wanted if r['meta_version'] == VERSION)}/{len(wanted)}")
        uploads = Counter(r["upload_status"] for r in rows if r["upload_status"])
        if uploads:
            print("  upload: " + ", ".join(f"{k} {v}" for k, v in uploads.most_common()))
        failed = sum(1 for r in rows if r["fail_count"])
        if failed:
            print(f"  model errors for {failed} documents (details: paperclerk show <id>)")
        todo = [("relevance", sum(1 for r in rows if r["text_method"] and not r["skip_reason"] and not r["part_of"]
                                  and r["noul1"] is None), _median(db, "secs1")),
                ("metadata", sum(1 for r in wanted if r["meta_version"] != VERSION), _median(db, "secs3"))]
        parts = [f"{label} ~{t:.1f}s/doc" + (f", {n} left (~{fmt_duration(n * t)})" if n else "")
                 for label, n, t in todo if t is not None]
        if parts:
            print("  timing: " + " | ".join(parts))
    processed = Counter(r["status"] for r in db.q("SELECT status FROM processed"))
    if processed:
        last = db.one("SELECT MAX(updated_at) t FROM processed")["t"]
        print("Process: " + ", ".join(f"{k} {v}" for k, v in processed.most_common()) + f" (last {last[:16]})")
    if not rows and not processed:
        print("Nothing yet. Start with `paperclerk import mbox <file>` or `paperclerk process`.")
    return 0


def _show_processed(db: DB, doc_id: int) -> int:
    rec = db.one("SELECT * FROM processed WHERE doc_id = ?", (doc_id,))
    if rec is None:
        print(f"Paperless document #{doc_id} has not been processed.")
        return 1
    print(f"Paperless #{doc_id} {rec['title']!r}: {rec['status']} ({rec['updated_at']}, {rec['model']}, {rec['secs']}s)")
    if rec["suggestion"]:
        print(f"Suggestion:  {rec['suggestion']}")
    if rec["changes"]:
        print(f"Applied:     {rec['changes']}")
    if rec["error"]:
        print(f"Error ({rec['fail_count']}x): {rec['error']}")
    return 0


def cmd_show(args: argparse.Namespace, db: DB, cfg: Config) -> int:
    ident = args.id.lstrip("#")
    if ident.isdigit() and len(ident) <= 7:
        return _show_processed(db, int(ident))
    docs = db.find_docs(ident)
    if len(docs) != 1:
        print(f"{'No' if not docs else 'More than one'} imported document matches '{ident}' (use at least 4 "
              "characters of the ID, or a Paperless document number)")
        return 1
    row = docs[0]
    p = probability(row, cfg)
    print(f"ID:          {row['sha256']}\nFile:        {cfg.files_dir / row['path']}\nName:        {row['filename']}")
    print(f"Kind:        {row['kind']} ({row['mime']}, {row['size'] / 1024:.0f} KB, {row['pages']} pages), "
          f"text: {row['text_method']}")
    print(f"State:       {state(row, cfg)}" + (f" (relevance {p:.0%})" if p is not None else "")
          + (f", manual: {row['manual']}" if row["manual"] else "")
          + (f", skipped: {row['skip_reason']}" if row["skip_reason"] else ""))
    top = sorted(type_probs(row).items(), key=lambda kv: -kv[1])[:5]
    if top:
        print("Category:    " + ", ".join(f"{k} {v:.0%}" for k, v in top) + f" -> {category_key(row)}")
    if row["meta"]:
        print(f"Metadata:    {json.dumps(meta(row), ensure_ascii=False)} ({row['model3']}, {row['secs3']}s)")
    if row["overrides"]:
        print(f"Corrections: {json.dumps(overrides(row), ensure_ascii=False)}")
    if row["meta"] or row["overrides"]:
        plan = build_plan(db, row, cfg)
        print(f"Upload plan: {plan.created} | {plan.correspondent or '-'} | {plan.document_type or '-'} | "
              f"{plan.title} | {', '.join(plan.tags)}")
    if row["parts"]:
        print(f"Parts:       {', '.join(s[:12] for s in json.loads(row['parts']))}")
    if row["part_of"]:
        print(f"Part of:     {row['part_of'][:12]}")
    if row["upload_status"]:
        print(f"Upload:      {row['upload_status']}, Paperless #{row['paperless_id']} {row['upload_error'] or ''}")
    if row["last_error"]:
        print(f"Last error ({row['fail_count']}x): {row['last_error']}")
    for m in db.mails_for(row["sha256"]):
        print(f"Mail:        {(m['date'] or '')[:16]} {m['from_addr']} | {m['subject']} | {m['att_filename']}")
    if row["text"]:
        print("-" * 60 + "\n" + row["text"][:1200])
    return 0


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paperclerk",
                                description="Enrich Paperless-ngx documents with local LLMs, import mail attachments")
    p.add_argument("-c", "--config", default=os.environ.get("PAPERCLERK_CONFIG", "config.toml"),
                   help="configuration file (default: ./config.toml)")
    p.add_argument("-v", "--verbose", action="store_true", help="more output")
    p.add_argument("--version", action="version", version=f"paperclerk {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    def add(name: str, func, help_: str, parent=sub) -> argparse.ArgumentParser:
        sp = parent.add_parser(name, help=help_, description=help_)
        sp.set_defaults(func=func)
        return sp

    def run_options(sp: argparse.ArgumentParser, dry_run: str = "do everything except writing to Paperless") -> None:
        sp.add_argument("--dry-run", action="store_true", help=dry_run)
        sp.add_argument("-n", "--limit", type=int, help="at most N documents (for test runs)")

    add("init", None, "create config.toml (or convert a maildocs one) and download OCR data")
    add("check", cmd_check, "check configuration, Ollama models, Paperless and IMAP")
    imp = add("import", cmd_import, "import mail attachments into Paperless; in a terminal this opens the import "
              "view, where you follow every step and accept what gets uploaded")
    sources = imp.add_subparsers(dest="source", required=True, metavar="source")
    mbox = add("mbox", cmd_import, "Google Takeout MBOX file(s)", sources)
    mbox.add_argument("files", nargs="+", help="MBOX file(s)")
    no_view = "do everything except uploading, without the import view; print what would be uploaded"
    run_options(mbox, no_view)
    run_options(add("imap", cmd_import, "Gmail via IMAP ([imap] in the configuration)", sources), no_view)
    run_options(add("process", cmd_process, "enrich Paperless documents paperclerk has not processed yet"))
    add("ui", cmd_ui, "check, edit and accept the suggestions of `process` in a terminal UI")
    add("status", cmd_status, "progress of imports and processing")
    show = add("show", cmd_show, "details of an imported document (ID) or a processed Paperless document (number)")
    show.add_argument("id")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.cmd == "init":
        return cmd_init(args)
    try:
        cfg = load(Path(args.config))
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    setup_logging(cfg.data_dir / "paperclerk.log", args.verbose)
    db = DB(cfg.db_path, legacy=cfg.legacy_db_path)
    try:
        return args.func(args, db, cfg) or 0
    except KeyboardInterrupt:
        log.warning("Interrupted – progress is saved, just run the command again.")
        return 130
    except (OllamaUnavailable, PaperlessError, ImapError, FileNotFoundError) as e:
        log.error("%s", e)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
