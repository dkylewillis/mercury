"""
Mercury CLI — JSON-first interface designed for AI agent consumption.

Commands
--------
ingest   Convert a PDF, store it, chunk it, and embed it.
query    Semantic search across one or more documents.
list     List ingested documents (with optional name filter).
status   Show collection health (doc count, chunk count, issues).
repair   Find and optionally fix incomplete ingestions.
delete   Remove a document from the document store and vector store.

All output is newline-terminated JSON printed to stdout.
Long-running commands (ingest, repair --fix) emit intermediate progress
lines before the final result line.
Errors are printed to stderr as {"error": "...", "error_code": "..."} with exit code 1.

Error codes
-----------
    conversion_failed   PDF could not be parsed
    store_failed        Could not write to doc store
    chunking_failed     Chunking step failed
    embedding_failed    ChromaDB write failed
    query_failed        Vector search failed
    not_found           --name filter matched no documents
    list_failed         Could not read store
    delete_failed       Could not delete document
    status_failed       Could not read store for status check

Usage examples
--------------
    python cli.py ingest data/report.pdf
    python cli.py query "What are the main AI models?" --top-k 3 --window 1
    python cli.py query "tables" --name "coweta"
    python cli.py query "tables" --file-hash 11465328351749295394
    python cli.py list
    python cli.py list --name report
    python cli.py status
    python cli.py repair
    python cli.py repair --fix
    python cli.py delete 11465328351749295394
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from agents import expand_query, answer_question
from chunker import Chunker
from converter import Converter
from document_store import DocumentStore
from vector_store import VectorStore

DEFAULT_DATA_DIR = os.environ.get("MERCURY_DATA_DIR", "./mercury_data")
DEFAULT_COLLECTION = "mercury"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _ok(payload: Any) -> None:
    print(json.dumps(payload, default=str), flush=True)


def _progress(payload: Any) -> None:
    """Emit a progress line to stdout during long-running operations."""
    print(json.dumps(payload, default=str), flush=True)


def _err(message: str, error_code: str = "error") -> None:
    print(json.dumps({"error": message, "error_code": error_code}), file=sys.stderr)
    sys.exit(1)


def _stores(args: argparse.Namespace):
    """Return (DocumentStore, VectorStore) rooted under <data_dir>/<collection>/."""
    root = Path(args.data_dir) / args.collection
    doc_store = DocumentStore(str(root / "doc_store"))
    vs = VectorStore(
        collection_name=args.collection,
        persist_directory=str(root / "chroma"),
    )
    return doc_store, vs


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    if not Path(args.file).exists():
        _err(f"File not found: {args.file}", "not_found")

    converter = Converter()
    doc_store, vs = _stores(args)

    if args.chunk_size > 0:
        _cmd_ingest_chunked(args, converter, doc_store, vs)
        return

    _progress({"status": "converting", "file": args.file})
    try:
        dl_doc = converter.convert(args.file)
    except Exception as e:
        _err(f"Conversion failed: {e}", "conversion_failed")

    file_hash = str(dl_doc.origin.binary_hash)
    if doc_store.exists(file_hash):
        _ok({
            "status": "already_ingested",
            "collection": args.collection,
            "file_hash": file_hash,
            "document_name": dl_doc.name,
            "filename": dl_doc.origin.filename,
        })
        return

    _progress({"status": "storing", "file_hash": file_hash, "page_count": len(dl_doc.pages)})
    try:
        file_hash = doc_store.create(dl_doc, source_pdf_path=args.file)
    except Exception as e:
        _err(f"Document store failed: {e}", "store_failed")

    _progress({"status": "chunking"})
    try:
        chunker = Chunker(dl_doc)
        chunks = chunker.chunk()
    except Exception as e:
        doc_store.set_status(file_hash, "failed")
        _err(f"Chunking failed: {e}", "chunking_failed")

    _progress({"status": "embedding", "chunk_count": len(chunks)})
    try:
        vs.create(chunks)
        doc_store.set_status(file_hash, "complete")
    except Exception as e:
        doc_store.set_status(file_hash, "failed")
        _err(f"Embedding failed: {e}", "embedding_failed")

    _ok({
        "status": "ingested",
        "collection": args.collection,
        "file_hash": file_hash,
        "document_name": dl_doc.name,
        "filename": dl_doc.origin.filename,
        "page_count": len(dl_doc.pages),
        "chunk_count": len(chunks),
    })


def _cmd_ingest_chunked(
    args: argparse.Namespace,
    converter: Converter,
    doc_store: DocumentStore,
    vs: VectorStore,
) -> None:
    """Ingest a PDF by converting it in sequential page-range chunks.

    Each chunk is converted, chunked, and embedded independently so that peak
    memory never exceeds what a single page range requires.  Chunk indices are
    renumbered globally so that window-based context retrieval works correctly
    across range boundaries.
    """
    total_pages = converter.page_count(args.file)
    chunk_size = args.chunk_size
    ranges = [
        (start, min(start + chunk_size - 1, total_pages))
        for start in range(1, total_pages + 1, chunk_size)
    ]
    total_ranges = len(ranges)

    # Convert the first range to obtain document identity before storing.
    start, end = ranges[0]
    _progress({
        "status": "converting",
        "file": args.file,
        "page_range": f"{start}-{end}",
        "range_index": 1,
        "total_ranges": total_ranges,
        "total_pages": total_pages,
    })
    try:
        first_dl_doc = converter.convert_page_range(args.file, start, end)
    except Exception as e:
        _err(f"Conversion failed on pages {start}-{end}: {e}", "conversion_failed")

    file_hash = str(first_dl_doc.origin.binary_hash)
    if doc_store.exists(file_hash):
        _ok({
            "status": "already_ingested",
            "collection": args.collection,
            "file_hash": file_hash,
            "document_name": first_dl_doc.name,
            "filename": first_dl_doc.origin.filename,
        })
        return

    _progress({"status": "storing", "file_hash": file_hash, "page_count": total_pages})
    try:
        file_hash = doc_store.create(
            first_dl_doc, source_pdf_path=args.file, page_count=total_pages
        )
    except Exception as e:
        _err(f"Document store failed: {e}", "store_failed")

    chunk_offset = 0
    total_chunks = 0
    for i, (start, end) in enumerate(ranges):
        dl_doc = first_dl_doc if i == 0 else None
        if dl_doc is None:
            _progress({
                "status": "converting",
                "file": args.file,
                "page_range": f"{start}-{end}",
                "range_index": i + 1,
                "total_ranges": total_ranges,
            })
            try:
                dl_doc = converter.convert_page_range(args.file, start, end)
            except Exception as e:
                doc_store.set_status(file_hash, "failed")
                _err(f"Conversion failed on pages {start}-{end}: {e}", "conversion_failed")

        _progress({
            "status": "chunking",
            "page_range": f"{start}-{end}",
            "range_index": i + 1,
            "total_ranges": total_ranges,
        })
        try:
            range_chunks = Chunker(dl_doc).chunk()
        except Exception as e:
            doc_store.set_status(file_hash, "failed")
            _err(f"Chunking failed on pages {start}-{end}: {e}", "chunking_failed")

        # Renumber indices to be globally sequential across all ranges so that
        # window-based context retrieval works correctly for cross-range queries.
        for chunk in range_chunks:
            new_index = chunk_offset + chunk.index
            chunk.index = new_index
            chunk.id = f"{file_hash}_{new_index}"

        _progress({
            "status": "embedding",
            "page_range": f"{start}-{end}",
            "range_index": i + 1,
            "total_ranges": total_ranges,
            "chunk_count": len(range_chunks),
        })
        try:
            vs.create(range_chunks)
        except Exception as e:
            doc_store.set_status(file_hash, "failed")
            _err(f"Embedding failed on pages {start}-{end}: {e}", "embedding_failed")

        chunk_offset += len(range_chunks)
        total_chunks += len(range_chunks)

    doc_store.set_status(file_hash, "complete")
    _ok({
        "status": "ingested",
        "collection": args.collection,
        "file_hash": file_hash,
        "document_name": first_dl_doc.name,
        "filename": first_dl_doc.origin.filename,
        "page_count": total_pages,
        "chunk_count": total_chunks,
    })


def cmd_query(args: argparse.Namespace) -> None:
    # --file-hash may be specified multiple times; normalise to None / str / list
    hashes = args.file_hash or []
    if len(hashes) == 0:
        file_hash = None
    elif len(hashes) == 1:
        file_hash = hashes[0]
    else:
        file_hash = hashes

    doc_store, vs = _stores(args)

    # Resolve --name to file_hash(es) when no explicit hash given
    if args.name and not hashes:
        needle = args.name.lower()
        matched = [
            r.file_hash for r in doc_store.list()
            if needle in r.document_name.lower() or needle in r.filename.lower()
        ]
        if not matched:
            _err(f"No documents found matching name: {args.name}", "not_found")
        file_hash = matched[0] if len(matched) == 1 else matched

    try:
        results = vs.query(
            query_text=args.query,
            top_k=args.top_k,
            file_hash=file_hash,
            window=args.window,
        )
    except Exception as e:
        _err(f"Query failed: {e}", "query_failed")

    _ok({
        "query": args.query,
        "top_k": args.top_k,
        "window": args.window,
        "result_count": len(results),
        "results": [
            {
                "chunk": r.chunk.model_dump(),
                "context": [c.model_dump() for c in r.context],
            }
            for r in results
        ],
    })


def cmd_list(args: argparse.Namespace) -> None:
    doc_store, vs = _stores(args)

    if args.source == "vector":
        try:
            docs = vs.list_documents()
        except Exception as e:
            _err(f"List failed: {e}", "list_failed")
        if args.name:
            needle = args.name.lower()
            docs = [d for d in docs if needle in d["document_name"].lower()]
        _ok({"count": len(docs), "source": "vector", "documents": docs})
        return

    try:
        records = doc_store.list()
    except Exception as e:
        _err(f"List failed: {e}", "list_failed")

    if args.name:
        needle = args.name.lower()
        records = [r for r in records if needle in r.document_name.lower() or needle in r.filename.lower()]

    _ok({
        "count": len(records),
        "source": "manifest",
        "documents": [r.model_dump() for r in records],
    })


def cmd_status(args: argparse.Namespace) -> None:
    doc_store, vs = _stores(args)

    try:
        records = doc_store.list()
        chroma_docs = vs.list_documents()
    except Exception as e:
        _err(f"Status check failed: {e}", "status_failed")

    chroma_hashes = {d["file_hash"] for d in chroma_docs}
    complete = [r for r in records if r.status == "complete"]
    incomplete = [r for r in records if r.status != "complete"]
    missing_in_chroma = [r for r in complete if r.file_hash not in chroma_hashes]
    total_chunks = sum(d["chunk_count"] for d in chroma_docs)

    _ok({
        "collection": args.collection,
        "document_count": len(complete),
        "chunk_count": total_chunks,
        "incomplete_count": len(incomplete),
        "missing_in_chroma_count": len(missing_in_chroma),
        "healthy": len(incomplete) == 0 and len(missing_in_chroma) == 0,
    })


def cmd_repair(args: argparse.Namespace) -> None:
    doc_store, vs = _stores(args)

    incomplete = doc_store.list_incomplete()
    chroma_hashes = {d["file_hash"] for d in vs.list_documents()}
    missing_in_chroma = [
        r for r in doc_store.list()
        if r.status == "complete" and r.file_hash not in chroma_hashes
    ]

    total_issues = len(incomplete) + len(missing_in_chroma)
    issues = {
        "incomplete": [r.model_dump() for r in incomplete],
        "missing_in_chroma": [r.model_dump() for r in missing_in_chroma],
    }

    if not args.fix:
        _ok({"issue_count": total_issues, **issues})
        return

    to_fix = {r.file_hash: r for r in incomplete + missing_in_chroma}
    fixed, failed = [], []
    for record in to_fix.values():
        try:
            pdf_path = doc_store.get_pdf_path(record.file_hash)
        except FileNotFoundError:
            failed.append({"file_hash": record.file_hash, "reason": "PDF not found in store"})
            continue
        try:
            converter = Converter()
            pdf_str = str(pdf_path)
            chunk_size = getattr(args, "chunk_size", 50)
            all_chunks: list = []
            chunk_offset = 0
            for _s, _e, doc in converter.convert_in_page_chunks(pdf_str, chunk_size):
                for chunk in Chunker(doc).chunk():
                    chunk.index = chunk_offset
                    chunk.id = f"{record.file_hash}_{chunk_offset}"
                    all_chunks.append(chunk)
                    chunk_offset += 1
            vs.update(all_chunks)
            doc_store.set_status(record.file_hash, "complete")
            fixed.append({"file_hash": record.file_hash, "document_name": record.document_name})
        except Exception as e:
            doc_store.set_status(record.file_hash, "failed")
            failed.append({"file_hash": record.file_hash, "reason": str(e)})

    _ok({"fixed": fixed, "failed": failed, **issues})


def _resolve_name_filter(args, doc_store):
    """Resolve --name to a file_hash filter (None, str, or list)."""
    if not args.name:
        return None
    needle = args.name.lower()
    matched = [
        r.file_hash for r in doc_store.list()
        if needle in r.document_name.lower() or needle in r.filename.lower()
    ]
    if not matched:
        _err(f"No documents found matching name: {args.name}", "not_found")
    return matched[0] if len(matched) == 1 else matched


def cmd_ask(args: argparse.Namespace) -> None:
    doc_store, vs = _stores(args)
    file_hash = _resolve_name_filter(args, doc_store)

    _progress({"status": "expanding_query", "question": args.question})
    try:
        keyphrases = expand_query(args.question)
    except Exception as e:
        _err(f"Query expansion failed: {e}", "query_expansion_failed")

    _progress({"status": "searching", "expanded_query": keyphrases})
    try:
        results = vs.query(
            query_text=keyphrases,
            top_k=args.top_k,
            file_hash=file_hash,
            window=args.window,
        )
    except Exception as e:
        _err(f"Query failed: {e}", "query_failed")

    if not results:
        _ok({
            "answer": "No relevant documents were found for your question.",
            "expanded_query": keyphrases,
            "sources": [],
        })
        return

    _progress({"status": "answering", "result_count": len(results)})
    try:
        answer = answer_question(args.question, results)
    except Exception as e:
        _err(f"Answer generation failed: {e}", "answer_failed")

    sources = []
    for r in results:
        sources.append({
            "document": r.chunk.document_name,
            "page": r.chunk.page_number,
            "file_hash": r.chunk.file_hash,
        })

    _ok({
        "answer": answer,
        "expanded_query": keyphrases,
        "sources": sources,
    })


def cmd_delete(args: argparse.Namespace) -> None:
    doc_store, vs = _stores(args)

    try:
        doc_store.delete(args.file_hash)
        vs.delete(args.file_hash)
    except Exception as e:
        _err(f"Delete failed: {e}", "delete_failed")

    _ok({"status": "deleted", "file_hash": args.file_hash})


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mercury",
        description="Mercury document ingestion and semantic search CLI.",
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR, metavar="DIR",
        help=f"Root directory for all collections (default: {DEFAULT_DATA_DIR}, env: MERCURY_DATA_DIR).",
    )
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION, metavar="NAME",
        help=f"Collection to operate on (default: {DEFAULT_COLLECTION}). "
             "Data is stored under <data-dir>/<collection>/.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Convert, store, chunk, and embed a PDF.")
    p_ingest.add_argument("file", help="Path to the PDF file.")
    p_ingest.add_argument(
        "--chunk-size", type=int, default=50, dest="chunk_size", metavar="N",
        help="Pages per conversion chunk (default: 50). "
             "Set to 0 to convert the whole document at once.",
    )

    # query
    p_query = sub.add_parser("query", help="Semantic search over ingested documents.")
    p_query.add_argument("query", help="Natural-language query string.")
    p_query.add_argument("--top-k", type=int, default=5, metavar="N",
                         help="Maximum number of results (default: 5).")
    p_query.add_argument("--file-hash", action="append", metavar="HASH",
                         help="Restrict to one document. Repeat to search multiple documents.")
    p_query.add_argument("--name", metavar="SUBSTR",
                         help="Restrict to documents whose name contains this substring (case-insensitive). Resolved to file_hash automatically.")
    p_query.add_argument("--window", type=int, default=0, metavar="N",
                         help="Number of adjacent chunks to include around each hit (default: 0).")

    # list
    p_list = sub.add_parser("list", help="List ingested documents.")
    p_list.add_argument("--name", metavar="SUBSTR",
                        help="Filter by document name or filename substring (case-insensitive).")
    p_list.add_argument("--source", choices=["manifest", "vector"], default="manifest",
                        help="Source to list from: 'manifest' (default) or 'vector' (ChromaDB).")

    # status
    sub.add_parser("status", help="Show collection health: doc count, chunk count, and any issues.")

    # repair
    p_repair = sub.add_parser("repair", help="Find and optionally fix incomplete ingestions.")
    p_repair.add_argument("--fix", action="store_true",
                          help="Re-ingest each incomplete or missing document from its stored PDF.")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question: expand query, search, and answer.")
    p_ask.add_argument("question", help="Natural-language question.")
    p_ask.add_argument("--top-k", type=int, default=5, metavar="N",
                       help="Maximum number of search results (default: 5).")
    p_ask.add_argument("--window", type=int, default=0, metavar="N",
                       help="Adjacent chunks to include around each hit (default: 0).")
    p_ask.add_argument("--name", metavar="SUBSTR",
                       help="Restrict to documents whose name contains this substring.")

    # delete
    p_delete = sub.add_parser("delete", help="Remove a document from all stores.")
    p_delete.add_argument("file_hash", help="The binary_hash of the document to remove.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "ask": cmd_ask,
        "list": cmd_list,
        "status": cmd_status,
        "repair": cmd_repair,
        "delete": cmd_delete,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
