# paperclerk

paperclerk fills in the metadata of [Paperless-ngx](https://docs.paperless-ngx.com/) documents with local
open-weight models through [Ollama](https://ollama.com): correspondent, document type, date, title, tags,
custom fields and a short summary. It also imports the attachments of a Gmail mailbox and only uploads
those worth keeping. No document leaves your machine and your Paperless server.

- `paperclerk process` enriches documents in Paperless that it has not processed yet (by default those in
  the inbox) and marks them with the tag `paperclerk`.
- `paperclerk import mbox|imap` reads mail attachments, drops logos, signatures and standard texts such as
  terms and conditions, merges documents that were split across several files, skips what Paperless already
  has and uploads the rest with the same metadata. In a terminal it opens an import view where you follow
  every step and decide what is uploaded.

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
values (with `[process] scope = "all"`). Tags are only ever added, and the inbox tag stays until you have
reviewed the document. To process a document again, remove its `paperclerk` tag.

To check the suggestions before they are written, use the terminal UI (works in Windows Terminal):

```powershell
uv run paperclerk ui
```

It looks like k9s: a list of the documents `process` would handle, with the values a suggestion changes
highlighted, while the model works through the rest in the background. `a` accepts: the values are
written and the inbox tag (e.g. `new`) is removed, since you have now reviewed the document
(`[process] remove_inbox_tags = false` keeps it). `x` rejects:
only the processed tag is added and the document stays in the inbox; changed your mind, `a` on it still
writes it. Enter opens a document with an
editable form next to its text, `/` filters, `?` shows all keys. What you type in the form is what gets
written; type the inbox tag back into the tags to keep a document in the inbox.

## Import mail attachments

```powershell
uv run paperclerk import mbox "Takeout\Mail\All mail Including Spam and Trash.mbox"
```

This opens the import view, in the same style as `paperclerk ui`. It lists every file found in the mail
while paperclerk works through the steps in the background: reading the mail, extracting the text, merging
documents split across several attachments, asking the relevance model whether each one is worth keeping,
and the metadata model for title, correspondent and so on. The header shows the current step, how far it
has got and how long the rest will take; the document you are looking at is processed first.

Nothing is uploaded until you accept it. `a` uploads a document with the values in its form, `x` rejects
it, and `A` uploads every *ready* document but leaves the *uncertain* ones (the relevance model was not
sure) for you to check (`/uncertain` filters the list). A document the model dropped can still be imported:
`a` on it asks the model for its metadata first. Enter opens a document with its editable form next to its
processing steps (which mail it came with, how its text was read, the relevance and category, the decision,
the metadata, the upload) and its text; `o` opens the file itself. `?` shows all keys.

Your decisions are stored at once. Quit at any time; the same command continues where it stopped. For new
mail use `uv run paperclerk import imap` (Gmail app password in `[imap]`). When the output is not a
terminal (e.g. a scheduled task), or with `--dry-run`, `import` runs without the view: `--dry-run` prints
what would be uploaded, a plain run uploads every relevant document and gives uncertain ones the review tag
(`paperclerk-review`).

## Commands

| Command | Purpose |
| --- | --- |
| `init` | create or convert `config.toml`, download OCR language data |
| `check` | check configuration, models, Paperless and IMAP |
| `import mbox FILE...` / `import imap` | import mail attachments; in a terminal in the import view (`--dry-run`, `-n N`) |
| `process` | enrich unprocessed Paperless documents (`--dry-run`, `-n N`) |
| `ui` | check, edit and accept the suggestions of `process` in a terminal UI |
| `status` | progress of imports and processing |
| `show ID` | an imported document (hash prefix) or a processed Paperless document (number) |

## Development

```powershell
$env:PAPERCLERK_TEST_TESSDATA = "D:\git\process-mail-documents\data\tessdata"   # otherwise OCR tests are skipped
uv run pytest
```

The tests build an mbox with German sample documents (`tests/samples.py`) and run the import, `process` and
both terminal UIs against fake Ollama and Paperless servers (API v10).
