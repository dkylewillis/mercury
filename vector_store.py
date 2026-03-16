import json
from pathlib import Path
from typing import List, Optional, Union

import chromadb
from sentence_transformers import SentenceTransformer

from models import BoundingBox, Chunk, QueryResult


class VectorStore:
    """ChromaDB-backed vector store for :class:`~models.Chunk` objects.

    Chunks are embedded with a sentence-transformers model and stored in a
    persistent local ChromaDB collection.  The ``file_hash`` field on every
    chunk is stored as metadata so that results can later be filtered to a
    single document, and the corresponding DoclingDocument can be retrieved
    from a :class:`~document_store.DocumentStore` for visual grounding.
    """

    def __init__(
        self,
        collection_name: str = "mercury",
        persist_directory: str = "./chroma_data",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.embeddings = SentenceTransformer(embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _to_meta(self, chunk: Chunk) -> dict:
        """Flatten a Chunk to a ChromaDB-compatible metadata dict (str/int/float values)."""
        return {
            "index": chunk.index,
            # Use -1 as a sentinel for None; ChromaDB does not support null values.
            "page_number": chunk.page_number if chunk.page_number is not None else -1,
            "headings": json.dumps(chunk.headings),
            "doc_items": json.dumps(chunk.doc_items),
            "bboxes": json.dumps([b.model_dump() for b in chunk.bboxes]),
            "document_name": chunk.document_name,
            "file_hash": chunk.file_hash,
            "file_extension": chunk.file_extension,
        }

    def _from_meta(self, id: str, text: str, meta: dict) -> Chunk:
        """Reconstruct a Chunk from a ChromaDB result row."""
        page_no = meta["page_number"]
        return Chunk(
            id=id,
            index=meta["index"],
            text=text,
            page_number=page_no if page_no != -1 else None,
            headings=json.loads(meta["headings"]),
            doc_items=json.loads(meta["doc_items"]),
            bboxes=[BoundingBox(**b) for b in json.loads(meta["bboxes"])],
            document_name=meta["document_name"],
            file_hash=meta["file_hash"],
            file_extension=meta["file_extension"],
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, chunks: List[Chunk]) -> None:
        """Embed and store a list of chunks in the collection.

        Args:
            chunks: Chunks to add.  Their ``id`` values must be unique within
                    the collection (the Chunker generates ``"{file_hash}_{index}"``
                    ids which satisfy this across documents).
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embeddings.encode(texts).tolist()
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[self._to_meta(c) for c in chunks],
        )

    def _fetch_window(self, hit: Chunk, window: int) -> List[Chunk]:
        """Fetch up to *window* chunks before and after *hit* from the same document."""
        if window == 0:
            return [hit]

        start = max(0, hit.index - window)
        end = hit.index + window

        raw = self.collection.get(
            where={
                "$and": [
                    {"file_hash": {"$eq": hit.file_hash}},
                    {"index": {"$gte": start}},
                    {"index": {"$lte": end}},
                ]
            },
            include=["documents", "metadatas"],
        )

        chunks = [
            self._from_meta(
                id=raw["ids"][i],
                text=raw["documents"][i],
                meta=raw["metadatas"][i],
            )
            for i in range(len(raw["ids"]))
        ]
        chunks.sort(key=lambda c: c.index)
        return chunks

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        file_hash: Optional[Union[str, List[str]]] = None,
        window: int = 0,
    ) -> List[QueryResult]:
        """Semantic search over stored chunks.

        Args:
            query_text: Natural-language query.
            top_k: Maximum number of results to return.
            file_hash: Restrict results to one or more documents.
                       Pass a single hash string, a list of hash strings,
                       or ``None`` to search across all documents.
            window: Number of adjacent chunks to include before and after each
                    hit.  ``0`` returns only the matching chunk itself.

        Returns:
            List of :class:`~models.QueryResult` objects ranked by cosine
            similarity.  Each result exposes the matching ``chunk`` and a
            ``context`` list of surrounding chunks sorted by index.
        """
        total = self.collection.count()
        if total == 0:
            return []

        n_results = min(top_k, total)
        query_embedding = self.embeddings.encode(query_text).tolist()

        if file_hash is None:
            where = None
        elif isinstance(file_hash, list):
            where = {"file_hash": {"$in": file_hash}}
        else:
            where = {"file_hash": {"$eq": file_hash}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        query_results = []
        for i in range(len(results["documents"][0])):
            hit = self._from_meta(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                meta=results["metadatas"][0][i],
            )
            context = self._fetch_window(hit, window)
            query_results.append(QueryResult(chunk=hit, context=context))

        return query_results

    def list_documents(self) -> List[dict]:
        """Return one summary entry per unique document stored in the collection.

        Returns:
            List of dicts with ``file_hash``, ``document_name``, ``file_extension``,
            and ``chunk_count`` keys, sorted by document_name.
        """
        raw = self.collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in raw["metadatas"]:
            fh = meta["file_hash"]
            if fh not in seen:
                seen[fh] = {
                    "file_hash": fh,
                    "document_name": meta["document_name"],
                    "file_extension": meta["file_extension"],
                    "chunk_count": 0,
                }
            seen[fh]["chunk_count"] += 1
        return sorted(seen.values(), key=lambda d: d["document_name"])

    def delete(self, file_hash: str) -> None:
        """Remove all chunks belonging to a document.

        Args:
            file_hash: The ``binary_hash`` of the document whose chunks should
                       be removed.
        """
        results = self.collection.get(where={"file_hash": file_hash})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def update(self, chunks: List[Chunk]) -> None:
        """Replace all chunks for a document (delete existing, then re-add).

        Args:
            chunks: New chunks for the document.  All chunks must share the
                    same ``file_hash``.
        """
        if not chunks:
            return
        self.delete(chunks[0].file_hash)
        self.create(chunks)
