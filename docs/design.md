# paperclerk – design

paperclerk fills in the metadata of Paperless-ngx documents with local open-weight models: correspondent,
document type, date, title, tags, custom fields and a summary. It also imports mail attachments from Gmail
and decides which of them are worth keeping. Document content never leaves the local machine and the
homelab.

This document records the design decisions and the reasons behind them. Measurements were taken on the
workstation the tool runs on: RTX 3070 Ti (8 GB), 32 GB RAM, Windows 11, Ollama 0.40.

## 1. Decisions at a glance

| Topic | Decision | Section |
| --- | --- | --- |
| Where the models run | On the GPU workstation, not in the cluster; the tool talks to Paperless over its API | 4 |
| Commands | `init`, `check`, `import mbox/imap`, `process`, `ui`, `review`, `status`, `show` – nothing else until needed | 3 |
| Interactive review | `paperclerk ui`: a k9s-style terminal UI over `process`; the form is what gets written | 6.5 |
| One pipeline | `import` and `process` share one enrichment step; only `import` scores relevance first | 5 |
| One model call per document | A single generative call returns all metadata as JSON, constrained by a schema | 5 |
| Vocabulary | Paperless' existing objects plus configured document types and tags; new tags allowed (≤ 3) | 5.2 |
| Processed marker | A tag (`paperclerk`) in Paperless, not only the local database | 6.1 |
| Write policy | Inbox documents: the model replaces Paperless' automatic guesses. Reviewed documents: only gaps are filled. Tags are only added | 6.2 |
| Inbox tag | `process` leaves it; accepting in `ui` removes it, because accepting is the review (`[process] remove_inbox_tags`, default true) | 6.2, 6.5 |
| Dry runs | Suggestions are stored; the next real run applies them without asking the model again | 6.3 |
| Relevance (import) | pplx-decider via `/v1/systemone`: calibrated yes/no plus category probabilities | 7.5 |
| Split documents (import) | Numbered attachments of one mail are merged if the model confirms each boundary | 7.4 |
| Duplicates (import) | Checksum first, then full-text search plus text comparison | 7.6 |
| Objects created in Paperless | Always with matching algorithm "none" | 8.2 |
| Local state | One SQLite file; every step resumable | 9 |
| Language | Code, CLI, docs and prompts in English; titles, tags and summaries in the configured language (German) | 5.3 |

## 2. Why the tool exists

Paperless' built-in AI runs in the cluster on a CPU-only Ollama with a 4B model (`qwen3:4b-instruct-2507`).
It takes about 90 s per document and the suggestions were not good enough. The workstation's GPU runs 27B
models with partial CPU offload at a few seconds per document. paperclerk brings those models to the
documents, both for the one-off import of old mail attachments and for documents that arrive in Paperless
later (scanner, Paperless' own mail fetcher, web upload).

## 3. Command line

The command line is deliberately small. Options and commands are added when a real need shows up, not in
advance.

```text
paperclerk init                      create config.toml (or convert a maildocs one), download OCR data
paperclerk check                     configuration, models, Paperless, IMAP
paperclerk import mbox FILE...       import mail attachments from a Google Takeout export
paperclerk import imap               the same directly from Gmail
paperclerk process                   enrich Paperless documents that have not been processed yet
paperclerk ui                        the same interactively: check, edit and accept suggestions
paperclerk review export|apply [F]   check and correct import results in a spreadsheet
paperclerk status                    progress
paperclerk show ID                   one imported document (hash prefix) or Paperless document (number)
```

`import` and `process` take `--dry-run` (everything except writing to Paperless) and `-n N` (test runs).

- **Name.** *paperclerk*: a clerk does the paperwork for Paperless. Alternatives considered: *filer*,
  *archivist*.
- **`import mbox FILE` instead of `import -t mbox`.** The source is a positional sub-command: it reads
  naturally, each source has its own arguments (files for mbox, nothing for IMAP), and further sources
  (e.g. a folder of PDFs) fit the same pattern.
- **One `import` command instead of separate steps.** The former maildocs commands (`ingest`, `extract`,
  `decide`, `escalate`, `meta`, `upload`, `sync`) run in order inside `import`. Every step only handles what
  is still open, so running the same command again simply continues. `--dry-run` stops before the upload;
  running the command again without it uploads.
- **Removed:** `setup-ocr` (part of `init`), `evaluate` (printed by `review apply`), `files`, `reset`,
  manual `merge`/`unmerge`, `upload --retry-failed` (failed uploads are retried automatically).

## 4. Architecture

```text
            Gmail (Takeout MBOX or IMAP)                          Paperless-ngx
                       │                                        │            ▲
   import ─────────────┼────────────────────┐         process   │ API        │ API
                       ▼                    │                   ▼            │
         ingest → extract text → merge split documents        select (no processed tag,
                       │                                        in the inbox)
                       ▼                                         │
              relevance (pplx-decider)                           │
                       │ relevant / review                       │
                       ▼                                         ▼
                ┌───────────────── shared enrichment (qwen3.8) ─────────────┐
                │ correspondent · document type · date · title · tags ·     │
                │ custom fields · summary                                   │
                └───────────────────────────────────────────────────────────┘
                       │                                         │
                       ▼                                         ▼
         upload (duplicate check, objects, note)          write back (policy, note)
                       │                                         │
                       └────────────── processed tag ────────────┘
```

All state lives in `<data_dir>/paperclerk.sqlite` (section 9). Paperless is the source of truth for the
vocabulary and for whether a document has been processed.

## 4.1 Models and hardware

| Use | Model | Measured | Notes |
| --- | --- | --- | --- |
| Relevance, continuation checks | `pplx-decider` (27B, Q4_K_M) | ~3.5 s per document, ~3 s per boundary | no output tokens; ~75 s to load |
| Enrichment | `qwen3.8` (27B, thinking off) | ~22–28 s per document | 800–930 input, 180–230 output tokens at ~9 tokens/s; ~100 s to load |

- **pplx-decider-flash does not run here.** All tags of the 4B model are MLX builds (Apple Silicon); Ollama
  on Windows refuses them. The 27B decider is fast anyway because it generates nothing, so it does stage 1
  directly and there is no escalation model. On a Mac: flash for stage 1, the 27B model for escalation.
- **Thinking off.** Thinking multiplies the output tokens, which are the expensive part with CPU offload.
- **Output length is the cost driver.** Fields and summary roughly quadruple the output compared with
  title/correspondent/date only (~50 tokens, ~7 s). `summary = false` saves about a fifth.
- **Phases, not interleaving.** Each step runs for all pending documents before the next one starts, so
  each model is loaded once per run instead of swapping per document.
- **Faster alternative.** `qwen3.6:35b` (MoE, ~3B active parameters) needed 4–5 s per document for the
  same output (~5× faster) with nearly identical results on the sample set; it was a little weaker on
  details (a painter's invoice tagged "Arbeit" instead of "Wohnung", fewer due dates found). It needs
  ~23 GB of RAM and does not start while Docker Desktop holds its memory. Switch with
  `[enrich] model = "qwen3.6:35b"`.

## 5. Shared enrichment

### 5.1 One call, all fields

Each document gets one chat call with a JSON schema (Ollama `format`), temperature 0, `num_ctx` 8192. The
answer contains: `correspondent`, `document_type`, `date`, `title`, `tags`, `fields` (one key per custom
field) and `summary`. One call is cheaper than several specialised ones because the document text is read
once, and the fields help each other (the document type informs the title).

Input: file name, context lines and up to 7,000 characters of text. Context for imports: mail sender,
subject, date and an excerpt of the mail text. Context for `process`: the date the document was added and
its current title, which is often the mail subject (Paperless' mail fetcher) or an earlier guess. Longer
texts are cut to their beginning (¾) and end (¼): files that combine documents often put the invoice on the
last page. Without readable text the first page goes along as an image (qwen3.8 has vision).

Title and document type must describe the same document. For a file that combines several documents (an
invoice with an enclosed consent form or terms), the prompt asks for the one that matters most for the
records, usually the invoice or contract. This came from a real case: a file titled "Rechnung
Ernährungsberatung" was typed "Vertrag" with a title about the enclosed consent form.

### 5.2 Vocabulary

The model chooses from lists so that Paperless does not fill up with near-duplicates.

- **Correspondents:** Paperless' correspondents ordered by document count, plus aliases and names
  recognised earlier in the run, at most 300. Names that are e-mail addresses (Paperless' mail fetcher
  creates such correspondents) are not shown, so the model is not encouraged to copy them.
- **Document types:** Paperless' document types plus `[enrich] document_types` (default: Rechnung,
  Quittung, Bestellung, Vertrag, Versicherung, Kontoauszug, Bescheid, Gehaltsabrechnung, Gesundheit, Ticket,
  Urkunde, Garantie, Brief). The schema makes this an enum; "" is allowed. Configured types are created when
  first assigned. Only existing types would not be enough: the Paperless instance had a single one.
- **Tags:** Paperless' tags without inbox tags and without the tool's own tags, plus `[enrich] tags`. The
  model may propose a new broad topic if nothing fits (`new_tags = true`), at most three tags per document.
  New tags join the list immediately, so later documents reuse them instead of inventing synonyms. With
  `new_tags = false` the schema allows listed tags only.
- **Custom fields:** `[[enrich.field]]` (default: Betrag, Rechnungsnummer, Kundennummer, Vertragsnummer,
  Fällig am). Each field has a type and a description for the model; the description says what the field is
  *not* (an order number is not a contract number), which fixed the most common confusions.

### 5.3 Prompt and language

Prompts are in English, the model's strongest instruction language. Titles, new tags and summaries are
written in `[enrich] language` (German). The fixed instructions and the correspondent list come first and
the tag list last, so Ollama can reuse its prompt cache when new tags are added.

### 5.4 Normalisation

The model's answer is checked before it is used:

| Field | Rule |
| --- | --- |
| Correspondent | Aliases, then exact match ignoring legal form, case and umlaut spelling, then fuzzy match (ratio ≥ 94); otherwise the new name without legal form. An e-mail address or domain becomes a known correspondent with that name, else the mail sender's display name, else the domain name. |
| Document type | Must be in the vocabulary (case-insensitive), else none |
| Date | Valid and no later than 31 days after the reference date (mail date or date added to Paperless); otherwise the mail date (import) or no change (process) – this catches due dates |
| Title | At most 128 characters; fallback: file name |
| Tags | Mapped onto existing tags (case, umlauts, fuzzy ratio ≥ 92); unknown tags only if allowed; tool tags never |
| Custom fields | Parsed per type: monetary `EUR1234.56` (currency code or symbol, German or English decimal separators), dates `YYYY-MM-DD`, numbers, booleans, URLs; unparseable values are dropped |
| Summary | At most 600 characters |

### 5.5 Versioning

Stored suggestions carry a version number (`enrich.VERSION`) and the model that produced them. When the
prompt or schema change in a way that matters, the version is raised; when `[enrich] model` changes, the
model no longer matches. In both cases documents that are not yet written to Paperless are enriched again
automatically, so trying another model is a matter of editing one line.

## 6. `process`: documents already in Paperless

### 6.1 Selection

A document is processed when it does **not** carry the processed tag (`[paperless] processed_tag`, default
`paperclerk`) and, with the default `[process] scope = "inbox"`, carries an inbox tag.

- The tag is the marker, not the local database: it is visible in Paperless, survives a lost database and
  can be removed by hand to have a document processed again.
- Paperless puts new documents into the inbox automatically, so `scope = "inbox"` picks up exactly the
  unreviewed documents. A curated archive is not touched on the first run. `scope = "all"` processes
  everything, with the stricter write policy below.
- Imported documents get the processed tag on upload, so `process` does not process them a second time.

### 6.2 Write policy

| Field | Document in the inbox | Document outside the inbox |
| --- | --- | --- |
| Title | replaced | only if it is still the file name |
| Correspondent, document type | replaced (Paperless' guesses are often wrong; its mail fetcher uses the sender address) | only if empty |
| Date | replaced if plausible | never |
| Tags | added, never removed | added, never removed |
| Custom fields | only empty ones | only empty ones |
| Summary | note added (an earlier paperclerk note is replaced) | same |
| Processed tag | added | added |

`process` leaves the inbox tag on: the inbox stands for "the user should look at this", and nobody has
looked yet. Accepting in `paperclerk ui` is that look, so there the inbox tag is removed (6.5). Tags and
custom fields are read again right before writing, so changes made in the meantime are not lost. In
`paperclerk ui` the policy only shapes the first proposal; what the user types wins (section 6.5).

### 6.3 Dry run and cache

`process --dry-run` asks the model, prints the planned changes per document and stores the suggestion in
the local database. The next real run applies stored suggestions without asking the model again, as long
as the document's text has not changed (content hash) and pipeline version and model are the same. The changes
are re-planned against the current state of the document, so edits made in between are respected.

Removing the processed tag runs the model again on the next run (the stored suggestion is reused only if
it was never applied).

### 6.4 Documents without text

If Paperless has no OCR text, the tool downloads the document and sends the first page as an image.

### 6.5 `ui`: checking suggestions interactively

`paperclerk ui` puts a terminal UI on top of `process`. It uses the same selection, model call, stored
suggestions and write step, so the two can be mixed: suggestions made by `process --dry-run` show up in
the UI, and documents accepted or rejected in the UI are not touched by `process` again.

- **Layout after k9s.** A header with the state (Paperless, model, queue, writes) and the keys of the
  current view; a full-width list of documents in which the values a suggestion changes are highlighted,
  so most documents can be checked and accepted from the list; `/` filters it (words, a status such as
  `ready`, a number); Enter opens a document with the form next to its text; Esc goes back one level;
  breadcrumbs at the bottom; `:` opens a command palette. Single keys work everywhere except while
  typing, where Ctrl+S accepts and Esc stops editing. After accepting in the document view the next
  document with a suggestion opens, so checking a batch is one key per document.
- **The form is what gets written.** It starts as the document's current values with the suggestion
  applied by the write policy (6.2); every value the user changes wins, also where `process` would
  leave it alone: a filled custom field is overwritten, an emptied one is cleared, a tag removed from
  the list is removed from the document. The changes are computed against the document as it was loaded
  and applied to a fresh copy, as in 6.2. Values are checked while typing (dates like 28.04.2023,
  amounts like 44,15 EUR); an invalid value blocks accepting and the form opens at it.
- **Two outcomes per document.** *Accept* (`a`, also *Accept all*) writes the form, adds the processed
  tag and removes the inbox tags (unless configured otherwise, see below): the user has now reviewed
  the document, so it needs no second pass in Paperless. *Reject* (`x`) writes nothing but the processed tag, so the document does not come back;
  the local status is `rejected`. Its inbox tag stays, because the document still needs a look in
  Paperless. Removing the processed tag brings either of them back, in a running UI with the next
  reload (`F5`).
- **A reject can be taken back in the session.** A wrong `x` is easy to hit and the UI moves on to the
  next document, so rejecting keeps the form (suggestion and edits) and leaves it editable: `a` / Ctrl+S
  on the rejected document still writes it, `g` asks the model again. Rejected documents stay out of
  *Accept all*, the "next document" step and the unsaved-edits check on quit – they are handled. After
  a restart a rejected document is no longer listed (it has the processed tag); removing the tag in
  Paperless brings it back. Written documents are final in the UI: fixing them is an edit in Paperless.
- **Leaving the inbox is part of the form.** The form starts without the inbox tags, so the tag row
  shows their removal like any other change (`−new`) and accepting writes exactly what is shown. To keep
  a document in the inbox (a bill still to pay), the user types the inbox tag back in. An earlier
  version kept the inbox tag on accept and had a second key (`d`) to remove it; in practice accepting
  always meant "reviewed", and one key per document is enough.
- **`[process] remove_inbox_tags`** (default `true`) turns this off for a workflow where documents are
  reviewed again in Paperless itself. The form then starts with the inbox tags; deleting one by hand
  still removes it. A setting rather than a second key, because it describes the workflow, not a
  decision per document. Which tags count as inbox tags is not configured in paperclerk: it is
  Paperless' "Inbox tag" flag (`is_inbox_tag`, read with the other tags on every load).
- **The model runs in the background.** On start every document without a usable stored suggestion is
  queued; the model works through them one by one while the user checks the ready ones. The document
  the user is looking at goes first, `g` asks for one right away, `p` pauses. If Ollama is unreachable
  or the model is missing, the queue pauses and says so instead of failing document by document.
- **Two worker threads.** One for the model, one for Paperless (loading, writing), each with its own
  SQLite connection and HTTP clients; the UI thread never waits for the network. Writes are queued, so
  accepting is instant; quitting waits for queued writes. The database is created before the threads
  start (two threads creating it at once collided).
- **Edits live in memory** until accepted. Quitting with unsaved edits asks first. Storing drafts would
  raise the question of what `process` should do with them; nobody needed that yet.
- **Textual** as the UI library: Python's `curses` does not exist on Windows, and Textual runs in Windows
  Terminal, brings inputs with completion, tables and dialogs, and can be driven headless in tests.

## 7. `import`: mail attachments

### 7.1 Sources

- **Google Takeout MBOX** for the existing mailbox: one file, no API limits. Read as a stream with a byte
  cursor, so a 20 GB file can be interrupted and resumed. Envelope lines are recognised by their format, so
  "From " inside a mail body does not split messages.
- **Gmail IMAP** with an app password for ongoing use: the `\All` folder (also when localised, e.g. "Alle
  Nachrichten"), Gmail search via `X-GM-RAW` (default `has:attachment`), resumes after the last UID. Gmail
  limits IMAP downloads to about 2.5 GB per day; the next run continues.
- Mails labelled Spam/Trash (Takeout header `X-Gmail-Labels`) are skipped.

### 7.2 Filtering and deduplication

- Mails without any sign of an attachment are not parsed at all (byte-level check), which keeps reading
  fast.
- File types come from the content (magic bytes), not the declared content type. Kept: PDF and images;
  Office files only with `kinds` including "office" (Paperless needs Tika/Gotenberg for them).
- Not even stored: inline images with a Content-ID, images under 15 KB or with a side under 400 px (logos,
  signatures, banners), S/MIME signatures, calendar files and other unsupported types. In the real mailbox
  that removed 1,499 of 3,193 attachments before any model call.
- ZIP archives are unpacked (unencrypted, at most 50 files / 300 MB).
- Identical files are stored and processed once (SHA-256); the document keeps the link to every mail it was
  attached to. Standard texts such as terms and conditions arrive many times with the same bytes. In the
  real mailbox the remaining 1,694 attachments were 1,388 distinct files.
- Every stored file stays in `<data_dir>/files`, including rejected ones; nothing is deleted.

### 7.3 Text

PyMuPDF reads the first three pages. Pages with too little text are OCRed with MuPDF's built-in Tesseract,
which only needs the `deu`/`eng` language files (downloaded by `init`), no Tesseract installation. If OCR
yields nothing, the models get the first page as an image. Encrypted PDFs are skipped and listed.

### 7.4 Merging split documents

Some senders split one document into one file per page ("Vertrag vom 04.11.2026 1.pdf" … "18.pdf").

- **Candidates:** attachments of the same mail whose names differ only in a running number (`X 1`, `X_003`,
  `X (2)`; four-digit years do not count). Sorted numerically (1, 2 … 10).
- **Confirmation:** names alone are not enough – "Rechnung 1.pdf" and "Rechnung 2.pdf" can be two invoices.
  The decider is asked at every boundary whether part B continues part A, with the end of A, the start of B
  and the position in the series. Answers are stored and reused.
- **Threshold 0.35.** Measured on the real case: pages of one contract 76–100 %, the notary's cover letter
  followed by the enclosed draft 49 %, two different invoices 8 %. With 0.5 the 18 pages became 3 + 15 and
  the second part got a wrong date.
- The merged PDF is a new document; only it is scored, enriched and uploaded. The parts keep their files and
  are marked as parts.

### 7.5 Relevance

The decider answers two questions in one forward pass (`/v1/systemone`): a yes/no question ("an individual
document worth keeping?") and a choice among 21 categories (14 relevant such as invoice, contract, payslip;
7 irrelevant such as terms, withdrawal policy, advertising).

relevance = ½ · P(yes) + ½ · P(relevant categories) (`type_weight`). When the two signals disagree the
document lands in the middle, which is the case worth a second look.

| Relevance | Result |
| --- | --- |
| ≥ 0.75 (`accept`) | imported |
| ≤ 0.15 (`reject`) | not imported (file and mail link kept) |
| between | escalation model if configured, otherwise imported with the review tag |
| manual yes/no (review sheet) | always wins |

Asymmetric thresholds: wrongly dropping a document is worse than importing one too many. The probabilities
are stored raw, so changed thresholds or category flags take effect without asking the model again. Sample
set (11 German documents): all correct with English prompts, including an insurance product information
sheet (12 %).

### 7.6 Duplicates

1. **Same file:** before uploading, Paperless is asked for the checksum (SHA-256 from Paperless 3, MD5 in
   older versions). Status `exists`.
2. **Same content, different file** (scanned by hand, merged, regenerated PDF): full-text search in Paperless
   with the document's longest words, then comparison of the texts. A duplicate needs ≥ 75 % of our word
   trigrams in the other document **and** ≥ 90 % of our numbers (dates, amounts, invoice numbers); the second
   condition keeps two monthly invoices with the same boilerplate apart. Status `similar`, with the Paperless
   document named. Real case: the merged contract was found 93 % inside the document already in Paperless.
   Very short texts (till receipts) are only checked by checksum.
3. Newer Paperless versions accept duplicates by default, so these checks are the only protection.

`similar` documents are checked again on every run: once the matching document is removed from Paperless,
the next run uploads ours. A command option to upload despite a match is left out until it is needed.

### 7.7 Upload

`post_document` with title, date, correspondent, document type and tags (processed tag, import tags, review
tag if uncertain, content tags). Paperless consumes asynchronously; at most 8 uploads wait at a time. When a
task succeeds, custom fields are set with a PATCH and the note is added (summary, mails with Message-ID,
relevance). Setting them afterwards works with every API version. Unfinished tasks and missing notes are
picked up by the next run.

### 7.8 Review

`review export` writes all imported documents into a spreadsheet: state, relevance, metadata, link to the
file, mail, text excerpt. The user fills in "Manual" (yes/no) and corrects fields; `review apply` reads the
changes back (detected against hidden copies of the original values) and prints how the thresholds perform
against the manual labels.

## 8. Paperless integration

### 8.1 Tags

| Tag | Set by | Purpose |
| --- | --- | --- |
| `paperclerk` (processed tag) | import and process | marks enriched documents; remove to re-process |
| `mail-import` (import tags) | import | origin |
| `paperclerk-review` (review tag) | import | relevance uncertain |
| content tags | import and process | topic |
| inbox tags (here `new`, `paperless-gpt`) | Paperless | not reviewed yet: `process` picks these documents; accepting in `ui` removes the tags |

### 8.2 Creating objects

Correspondents, configured document types, tags and custom fields are created when first used. They get
matching algorithm "none": paperclerk does the classification, and Paperless' automatic matching would add
competing guesses (and would learn to hand out tool tags such as the review tag). maildocs used "auto" for
some objects; this was changed.

### 8.3 API

- Works with API v10 (Paperless 3.x, tested with 3.5.1) and v9: task status upper or lower case,
  `related_document` or `related_document_ids`/`result_data`; for API < 9 the date is sent as
  `created_date`.
- Custom field values are sent in Paperless' formats (`EUR44.15`, `2023-04-28`); a PATCH sends the
  complete list of a document's custom fields, merged with the existing values.
- Notes are created with `POST …/notes/`; the id is kept so the note can be replaced later.

### 8.4 Access

The tool authenticates with an API token. A login proxy in front of Paperless (here Authelia) rejects token
requests with 401 before they reach Paperless; the tool recognises this (no `X-Api-Version` header) and says
so. Options: use an address that bypasses the proxy (currently `kubectl port-forward` to
`http://localhost:8000`) or exempt `/api` from the proxy for the LAN. Paperless is only needed for
`check`, `process`, the start of the enrichment step (vocabulary; the cached copy is used if Paperless is
unreachable) and the upload.

## 9. Local state

One SQLite file, `<data_dir>/paperclerk.sqlite` (WAL mode):

| Table | Content |
| --- | --- |
| `sources`, `messages`, `attachments` | mail sources with cursors, mails with attachments, every attachment incl. why it was dropped |
| `documents` | one row per unique imported file or merged document: text, model answers, suggestion, corrections, upload state |
| `merge_checks` | answers to "does B continue A?" |
| `processed` | Paperless documents handled by `process` or `ui`: suggestion, status (suggested, applied, rejected, failed), written changes, note id |
| `paperless_objects` | cache of correspondents, document types, tags, custom fields |
| `llm_log` | every model call with timing, tokens and raw answer |

Every step commits per document. Interrupting with Ctrl+C loses at most the document in progress; the next
run continues. Documents whose model call failed three times are skipped (shown by `status` and `show`).

**Migration from maildocs.** On first start the old `maildocs.sqlite` is copied to `paperclerk.sqlite` (the
old file stays). German category keys, manual labels (ja/nein), correction keys and skip reasons are
translated. Stored metadata from maildocs is recomputed with the new pipeline, because the schema gained
document type, tags, fields and summary. `paperclerk init` converts the old `config.toml`; the original is
kept as `config.maildocs.toml`.

## 10. Configuration

`config.toml`, documented in place (`paperclerk init` writes the template). Sections: `[paperless]`,
`[ollama]`, `[enrich]` with `[[enrich.field]]`, `[aliases]`, `[process]`, `[import]` with
`[import.scoring]`, `[import.merge]` and `[[import.category]]`, `[imap]`, `[text]`. Unknown keys are errors,
so typos do not silently fall back to defaults. Secrets can come from `PAPERLESS_URL`, `PAPERLESS_TOKEN`,
`GMAIL_USER`, `GMAIL_APP_PASSWORD`.

## 11. Privacy and security

- Documents are read and classified on the local machine; Ollama runs locally. The only outbound requests
  are to Paperless, Gmail (IMAP) and, once, GitHub for the OCR language files.
- `config.toml` holds the Paperless token and the app password and is excluded from version control, as is
  the data directory (it contains every attachment).
- The API token has the rights of its Paperless user; a dedicated user with change rights on documents is
  enough.

## 12. Alternatives considered

- **Paperless' built-in AI in the cluster:** too slow and too weak on a CPU-only 4B model (section 2).
- **paperless-gpt / paperless-ai pointed at the workstation's Ollama:** covers part of `process`, but needs
  the cluster to reach the workstation, would not share the import pipeline, the local state, the dry-run
  cache or the write policy.
- **Uploading everything and letting Paperless sort it out:** thousands of terms-and-conditions PDFs would
  land in Paperless first.
- **Several specialised model calls per document:** reads the text several times; one schema-constrained
  call is cheaper and consistent.
- **Only existing document types and tags:** too restrictive for an instance with one document type.
- **A spreadsheet for `process` as for imports:** the round trip (export, edit, apply) is slow for a few
  new documents a week; the terminal UI shows text and suggestion together and writes at once.
- **`curses`, urwid or prompt_toolkit for the UI:** `curses` is missing on Windows; the other two would
  need the table, completion and dialogs built by hand. Textual has them and can be tested headless.

## 13. Open points

- Running `process` regularly (e.g. Windows Task Scheduler) needs a stable path to the Paperless API;
  the port-forward is a manual step.
- Paperless contains correspondents that are e-mail addresses (from its mail fetcher). paperclerk no longer
  assigns them but does not merge them either; a cleanup command could reassign their documents.
- A faster enrichment model (MoE) could cut the time per document; see 4.1.
