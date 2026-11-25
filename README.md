# paperclerk

paperclerk fills in the metadata of [Paperless-ngx](https://docs.paperless-ngx.com/) documents with local
open-weight models through [Ollama](https://ollama.com): correspondent, document type, date, title, tags,
custom fields and a short summary. It also imports the attachments of a Gmail mailbox and only uploads
those worth keeping. No document leaves your machine and your Paperless server.

- `paperclerk process` enriches documents in Paperless that it has not processed yet (by default those in
  the inbox) and marks them with the tag `paperclerk`.
- `paperclerk import mbox|imap` reads mail attachments, drops logos, signatures and standard texts such as
  terms and conditions, merges documents that were split across several files, skips what Paperless already
  has and uploads the rest with the same metadata.

[docs/design.md](docs/design.md) explains how it works and why.

## Requirements

- Python ≥ 3.11 with [uv](https://docs.astral.sh/uv/)
- Ollama with `qwen3.8` (metadata) and, for imports, `pplx-decider` (relevance):
  `ollama pull qwen3.8` and `ollama pull pplx-decider`
- A Paperless-ngx API token (*My Profile → API Auth Token*). If a login proxy such as Authelia sits in front
  of Paperless, use an address that bypasses it, e.g. `kubectl port-forward svc/paperless 8000:8000`.

## Setup

```powershell
uv sync
uv run paperclerk init      # writes config.toml (or converts a maildocs one) and downloads OCR data
# set [paperless] url and token in config.toml
uv run paperclerk check     # Ollama models, Paperless, inbox tags, what will be created
```

## Enrich documents in Paperless

```powershell
uv run paperclerk process --dry-run -n 5   # show the suggested changes for five documents
uv run paperclerk process                   # apply (the stored suggestions are reused)
```

Inbox documents get Paperless' automatic guesses replaced; documents outside the inbox only get missing
values (with `[process] scope = "all"`). Tags are only ever added. To process a document again, remove its
`paperclerk` tag.

## Import mail attachments

```powershell
uv run paperclerk import mbox "Takeout\Mail\All mail Including Spam and Trash.mbox" --dry-run -n 50
uv run paperclerk status
uv run paperclerk review export             # spreadsheet: mark wrong decisions, correct fields
uv run paperclerk review apply
uv run paperclerk import mbox "Takeout\Mail\All mail Including Spam and Trash.mbox"   # upload
```

Every step can be interrupted and resumed; running the same command again continues where it stopped. For
new mail use `uv run paperclerk import imap` (Gmail app password in `[imap]`).

## Commands

| Command | Purpose |
| --- | --- |
| `init` | create or convert `config.toml`, download OCR language data |
| `check` | check configuration, models, Paperless and IMAP |
| `import mbox FILE...` / `import imap` | import mail attachments (`--dry-run`, `-n N`) |
| `process` | enrich unprocessed Paperless documents (`--dry-run`, `-n N`) |
| `review export` / `review apply [FILE]` | check and correct import results in a spreadsheet |
| `status` | progress of imports and processing |
| `show ID` | an imported document (hash prefix) or a processed Paperless document (number) |

## Development

```powershell
$env:PAPERCLERK_TEST_TESSDATA = "D:\git\process-mail-documents\data\tessdata"   # otherwise OCR tests are skipped
uv run pytest
```

The tests build an mbox with German sample documents (`tests/samples.py`) and run the import and `process`
against fake Ollama and Paperless servers (API v10).
