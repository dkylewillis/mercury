# Mercury

A Python library for PDF ingestion, semantic search, and visual grounding. Convert PDFs to structured documents, chunk them intelligently, embed with sentence-transformers, and retrieve relevant passages with bounding-box metadata for highlighting source text.

## Features

- PDF → `DoclingDocument` conversion (text-native; no in-memory page image generation)
- On-demand page rendering from stored PDFs via `pypdfium2` for visual grounding
- Hybrid chunking (structure-aware + token-limit-aware)
- ChromaDB vector store with cosine similarity search
- Context window: fetch N chunks before/after each hit
- Multi-document search with per-document filtering
- Visual grounding: normalized bounding boxes on every chunk for PDF highlighting
- Ingestion status tracking (`pending` → `complete` / `failed`) with repair support
- Per-stage progress signals during ingest (converting → storing → chunking → embedding)
- Machine-typed error codes on all failures for reliable agent branching
- Document manifest for fast listing without loading full documents
- JSON-first CLI designed for AI agent consumption

## Data layout

Each collection is fully self-contained:

```
mercury_data/
  mercury/              ← default collection
    doc_store/
      manifest.json     ← lightweight metadata + ingestion status for every document
      <hash>.pdf        ← original PDF copy (used for on-demand page rendering)
    chroma/             ← ChromaDB embeddings
  legal/                ← python cli.py --collection legal ingest ...
    doc_store/
    chroma/
```

## Modules

### `models.py`

| Class | Purpose |
|---|---|
| `BoundingBox` | Normalized (0–1) page region, top-left origin |
| `Chunk` | Text passage with headings, bboxes, and document provenance |
| `DocumentRecord` | Manifest entry: metadata + `status` (`pending`/`complete`/`failed`) + `pdf_path` |
| `QueryResult` | Semantic search hit + surrounding context window |

### `converter.py`

```python
Converter()
  .convert(source: str) -> DoclingDocument
  .convert_page_range(source: str, start_page: int, end_page: int) -> DoclingDocument
  .convert_in_page_chunks(source: str, chunk_size: int) -> Generator[(start, end, DoclingDocument)]
  .page_count(source: str) -> int                        # static; fast pypdfium2 read, no ML
```

Use `convert_page_range` or `convert_in_page_chunks` for large PDFs with memory-spike pages.
Each page range is converted independently so peak memory is bounded to `chunk_size` pages.

### `rendering.py`

```python
render_page(pdf_path: str, page_number: int, scale: float = 2.0) -> PIL.Image
```

`render_page` opens a single page from a stored PDF on demand — no memory overhead at ingest time.

### `chunker.py`

```python
Chunker(dl_doc: DoclingDocument, tokenizer="sentence-transformers/all-MiniLM-L6-v2")
  .chunk() -> List[Chunk]
```

### `document_store.py`

```python
DocumentStore(base_path="./doc_store")
  .create(dl_doc, source_pdf_path, page_count=None) -> str  # copies PDF, writes status=pending
  .set_status(file_hash, status)                          # "pending" | "complete" | "failed"
  .get_pdf_path(file_hash)          -> Path               # path to stored PDF copy
  .exists(file_hash)                -> bool               # True only if status=complete
  .delete(file_hash)                                      # removes PDF + manifest entry
  .list()                           -> List[DocumentRecord]
  .list_incomplete()                -> List[DocumentRecord]  # status != complete
```

### `vector_store.py`

```python
VectorStore(collection_name="mercury", persist_directory="./chroma_data", embedding_model=...)
  .create(chunks)
  .query(query_text, top_k=5, file_hash=None, window=0) -> List[QueryResult]
  .list_documents()                 -> List[dict]         # unique docs with chunk counts
  .delete(file_hash)                                      # remove all chunks for a document
  .update(chunks)                                         # delete then re-add
```

`file_hash` in `query()` accepts `None` (all docs), a single hash string, or a list of hash strings.

## Python API Usage

```python
from converter import Converter
from rendering import render_page
from chunker import Chunker
from document_store import DocumentStore
from vector_store import VectorStore

# Collection paths — keep doc_store and chroma co-located under the same root
COLLECTION = "mercury"
DOC_STORE  = f"./mercury_data/{COLLECTION}/doc_store"
CHROMA     = f"./mercury_data/{COLLECTION}/chroma"

# --- Ingest (whole document) ---
converter = Converter()
dl_doc = converter.convert("report.pdf")

doc_store = DocumentStore(DOC_STORE)
file_hash = doc_store.create(dl_doc, source_pdf_path="report.pdf")  # status=pending

chunks = Chunker(dl_doc).chunk()

vs = VectorStore(collection_name=COLLECTION, persist_directory=CHROMA)
vs.create(chunks)
doc_store.set_status(file_hash, "complete")

# --- Ingest (chunked, for large/heavy PDFs) ---
converter = Converter()
total_pages = Converter.page_count("report.pdf")

file_hash = doc_store.create(
    next(doc for _, _, doc in converter.convert_in_page_chunks("report.pdf", chunk_size=50)),
    source_pdf_path="report.pdf",
    page_count=total_pages,
)
chunk_offset = 0
for start, end, dl_doc in converter.convert_in_page_chunks("report.pdf", chunk_size=50):
    range_chunks = Chunker(dl_doc).chunk()
    for chunk in range_chunks:
        chunk.index = chunk_offset
        chunk.id = f"{file_hash}_{chunk_offset}"
        chunk_offset += 1
    vs.create(range_chunks)
doc_store.set_status(file_hash, "complete")

# --- Query (all documents in collection) ---
results = vs.query("transformer architecture", top_k=5, window=1)

# --- Query (specific documents) ---
results = vs.query("transformer architecture", file_hash="11465328351749295394")
results = vs.query("transformer architecture", file_hash=["hash_a", "hash_b"])

# --- Inspect results ---
for r in results:
    print(r.chunk.headings, r.chunk.page_number)
    print([c.index for c in r.context])   # surrounding chunks

# --- Visual grounding ---
top = results[0].chunk
pdf_path = doc_store.get_pdf_path(top.file_hash)
for bbox in top.bboxes:
    img = render_page(str(pdf_path), bbox.page_no)
    # bbox.l / .r / .t / .b are normalized 0-1; multiply by img.width / img.height
```

## CLI

All commands print a single line of JSON to stdout.
Errors go to stderr as `{"error": "..."}` with exit code 1.

### Global flags

These flags apply to every command and select which collection to operate on.

| Flag | Default | Env override | Description |
|---|---|---|---|
| `--data-dir` | `./mercury_data` | `MERCURY_DATA_DIR` | Root directory for all collections |
| `--collection` | `mercury` | — | Collection name |

### `status`

Single-call collection health check — designed for an agent to verify the system before querying.

```
python cli.py [--collection NAME] status
```

```
$ python cli.py status
{
  "collection": "mercury",
  "document_count": 3,
  "chunk_count": 4821,
  "incomplete_count": 0,
  "missing_in_chroma_count": 0,
  "healthy": true
}
```

`healthy: false` means `repair` should be run.

### `ingest`

Convert, store, chunk, and embed a PDF. The original PDF is copied into the doc store.
Re-ingesting the same file (by content hash) returns `already_ingested` immediately.

By default the PDF is converted in 50-page chunks to bound peak memory (useful for large
documents with memory-spike pages like dense tables). Pass `--chunk-size 0` to convert the
whole document at once.

Emits one progress line per stage (and per page-range chunk) before the final result:

```
# Default chunked mode (--chunk-size 50)
{"status": "converting", "file": "data/report.pdf", "page_range": "1-50",   "range_index": 1, "total_ranges": 4, "total_pages": 195}
{"status": "storing",    "file_hash": "...", "page_count": 195}
{"status": "converting", "file": "data/report.pdf", "page_range": "51-100",  "range_index": 2, "total_ranges": 4}
{"status": "chunking",   "page_range": "51-100", "range_index": 2, "total_ranges": 4}
{"status": "embedding",  "page_range": "51-100", "range_index": 2, "total_ranges": 4, "chunk_count": 38}
...
{"status": "ingested",   "collection": "mercury", "file_hash": "...", "page_count": 195, "chunk_count": 312}

# Whole-document mode (--chunk-size 0)
{"status": "converting",  "file": "data/report.pdf"}
{"status": "storing",     "file_hash": "...", "page_count": 9}
{"status": "chunking"}
{"status": "embedding",   "chunk_count": 50}
{"status": "ingested",    "collection": "mercury", "file_hash": "...", ...}
```

```
python cli.py [--data-dir DIR] [--collection NAME] ingest [--chunk-size N] <file>
```

```
$ python cli.py ingest data/report.pdf
# … progress lines as above …
{"status": "ingested", "collection": "mercury", "file_hash": "11465328351749295394", "page_count": 195, "chunk_count": 312}

$ python cli.py ingest data/report.pdf --chunk-size 100   # 100 pages per conversion pass
$ python cli.py ingest data/report.pdf --chunk-size 0     # whole document at once

$ python cli.py ingest data/report.pdf
{"status": "already_ingested", "collection": "mercury", "file_hash": "11465328351749295394", ...}

$ python cli.py --collection legal ingest data/contract.pdf
{"status": "ingested", "collection": "legal", "file_hash": "99887766...", ...}
```

### `list`

List ingested documents.

```
python cli.py [--collection NAME] list [--name SUBSTR] [--source manifest|vector]
```

| `--source` | Reads from | Use when |
|---|---|---|
| `manifest` (default) | `manifest.json` on disk | Fast everyday listing; includes `page_count`, `ingested_at`, `status` |
| `vector` | ChromaDB directly | Verifying what is actually searchable; includes `chunk_count` per document |

The two sources can diverge if an ingest fails partway through (document is in the manifest but has no chunks in ChromaDB, or vice versa). Use `repair` to detect and fix these discrepancies.

```
$ python cli.py list
{"count": 1, "source": "manifest", "documents": [
  {"file_hash": "11465328351749295394", "document_name": "report",
   "filename": "report.pdf", "file_extension": ".pdf",
   "page_count": 9, "ingested_at": "2026-03-15T18:36:50",
   "status": "complete", "pdf_path": "..."}
]}

$ python cli.py list --source vector
{"count": 1, "source": "vector", "documents": [
  {"file_hash": "11465328351749295394", "document_name": "report",
   "file_extension": ".pdf", "chunk_count": 50}
]}

$ python cli.py list --name report
```

### `query`

Semantic search across one or more ingested documents in a collection.

```
python cli.py [--collection NAME] query <text> [--top-k N] [--name SUBSTR] [--file-hash HASH]... [--window N]
```

| Flag | Default | Description |
|---|---|---|
| `--top-k` | 5 | Max number of results |
| `--name` | — | Restrict to documents whose name contains this substring. Resolved to `file_hash` automatically — no hash lookup needed. |
| `--file-hash` | all docs | Restrict by explicit hash (repeat for multiple) |
| `--window` | 0 | Adjacent chunks to include around each hit |

```
$ python cli.py query "What are the main AI models?" --top-k 3 --window 1
{
  "query": "What are the main AI models?",
  "top_k": 3, "window": 1, "result_count": 3,
  "results": [
    {
      "chunk": {
        "id": "11465328351749295394_10", "index": 10,
        "page_number": 3, "headings": ["3.2 AI models"],
        "text": "3.2 AI models\nAs part of Docling...",
        "bboxes": [{"l": 0.176, "r": 0.823, "t": 0.488, "b": 0.582, "page_no": 3}],
        "file_hash": "11465328351749295394", "document_name": "report",
        "file_extension": ".pdf"
      },
      "context": [
        {"index": 9, "headings": ["3.1 PDF backends"], ...},
        {"index": 10, "headings": ["3.2 AI models"],   ...},
        {"index": 11, "headings": ["Layout Analysis Model"], ...}
      ]
    }
  ]
}

# Restrict to one document by name (no hash lookup required)
$ python cli.py query "front setbacks" --name "coweta"

# Restrict by explicit hash
$ python cli.py query "tables" --file-hash 11465328351749295394

# Search across two specific documents
$ python cli.py query "tables" --file-hash <hash1> --file-hash <hash2>
```

### `repair`

Scan for and optionally fix incomplete ingestions. An ingestion is considered incomplete if its manifest status is `pending`/`failed`, or if it is `complete` in the manifest but has no chunks in ChromaDB.

```
python cli.py [--collection NAME] repair [--fix] [--chunk-size N]
```

```
$ python cli.py repair
{"issue_count": 1,
 "incomplete": [{"file_hash": "...", "status": "failed", ...}],
 "missing_in_chroma": []}

$ python cli.py repair --fix
{"fixed": [{"file_hash": "...", "document_name": "report"}],
 "failed": [], ...}
```

`--fix` re-ingests each broken document from its stored PDF copy. No source file required.

### `delete`

Remove a document from both the document store and vector store.

```
python cli.py [--collection NAME] delete <file_hash>
```

```
$ python cli.py delete 11465328351749295394
{"status": "deleted", "file_hash": "11465328351749295394"}

$ python cli.py --collection legal delete 99887766
{"status": "deleted", "file_hash": "99887766"}
```

## Error handling

All errors go to stderr as a single JSON line with exit code 1:

```json
{"error": "Conversion failed: ...", "error_code": "conversion_failed"}
```

| `error_code` | Cause |
|---|---|
| `conversion_failed` | PDF could not be parsed by docling |
| `store_failed` | Could not write PDF or manifest to disk |
| `chunking_failed` | Chunking step raised an exception |
| `embedding_failed` | ChromaDB write failed |
| `query_failed` | Vector search failed |
| `not_found` | `--name` filter matched no documents |
| `list_failed` | Could not read the store |
| `delete_failed` | Could not delete the document |
| `status_failed` | Could not read the store for status check |

## Multi-collection workflow

```bash
# Ingest into separate collections
python cli.py --collection research ingest data/paper.pdf
python cli.py --collection legal    ingest data/contract.pdf

# Query within a specific collection only
python cli.py --collection research query "neural architecture"
python cli.py --collection legal    query "termination clause"

# Use a shared network or backup root
python cli.py --data-dir /mnt/shared --collection research ingest data/paper.pdf

# Or set the root once via environment variable
$env:MERCURY_DATA_DIR = "/mnt/shared"
python cli.py --collection research query "neural architecture"
```

## Inspiration

- https://docling-project.github.io/docling/examples/visual_grounding/
