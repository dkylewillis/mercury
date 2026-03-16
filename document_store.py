import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

from docling_core.types.doc.document import DoclingDocument

from models import DocumentRecord


class DocumentStore:
    """Stores PDF copies and manifest metadata, keyed by binary_hash.

    The binary_hash is the file name (``<hash>.pdf``) and is stored on every
    Chunk as ``file_hash``, enabling round-trip PDF retrieval for visual
    grounding via ``get_pdf_path``.
    """

    def __init__(self, base_path: str = "./doc_store"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.base_path / "manifest.json"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pdf_path(self, file_hash: str) -> Path:
        return self.base_path / f"{file_hash}.pdf"

    def _read_manifest(self) -> dict:
        if not self._manifest_path.exists():
            return {}
        with self._manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self, data: dict) -> None:
        tmp = self._manifest_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        tmp.replace(self._manifest_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, dl_doc: DoclingDocument, source_pdf_path: str) -> str:
        """Copy the source PDF into the store and write a pending manifest entry.

        Args:
            dl_doc: Converted DoclingDocument (must have a valid ``origin``).
            source_pdf_path: Path to the original PDF to copy into the store.

        Returns:
            The ``binary_hash`` used as the storage key.
        """
        file_hash = str(dl_doc.origin.binary_hash)
        dest = self._pdf_path(file_hash)
        if not dest.exists():
            shutil.copy2(source_pdf_path, dest)

        filename = dl_doc.origin.filename or ""
        record = DocumentRecord(
            file_hash=file_hash,
            document_name=dl_doc.name,
            filename=filename,
            file_extension=Path(filename).suffix if filename else "",
            page_count=len(dl_doc.pages),
            ingested_at=datetime.now(timezone.utc),
            status="pending",
            pdf_path=str(dest),
        )
        manifest = self._read_manifest()
        manifest[file_hash] = json.loads(record.model_dump_json())
        self._write_manifest(manifest)

        return file_hash

    def set_status(self, file_hash: str, status: Literal["pending", "complete", "failed"]) -> None:
        """Update the ingestion status of a document in the manifest."""
        manifest = self._read_manifest()
        if file_hash in manifest:
            manifest[file_hash]["status"] = status
            self._write_manifest(manifest)

    def get_pdf_path(self, file_hash: str) -> Path:
        """Return the path to the stored PDF.

        Raises:
            FileNotFoundError: If no PDF is stored for this hash.
        """
        path = self._pdf_path(file_hash)
        if not path.exists():
            raise FileNotFoundError(f"No PDF found for hash: {file_hash}")
        return path

    def exists(self, file_hash: str) -> bool:
        """Return True only if the document has been fully ingested (status=complete).

        Old manifest entries without a status field are treated as complete for
        backward compatibility.
        """
        entry = self._read_manifest().get(file_hash)
        return entry is not None and entry.get("status", "complete") == "complete"

    def delete(self, file_hash: str) -> None:
        """Delete the stored PDF and manifest entry. No-op if it does not exist."""
        pdf = self._pdf_path(file_hash)
        if pdf.exists():
            pdf.unlink()
        manifest = self._read_manifest()
        manifest.pop(file_hash, None)
        self._write_manifest(manifest)

    def list(self) -> List[DocumentRecord]:
        """Return a DocumentRecord for every stored document."""
        manifest = self._read_manifest()
        return [DocumentRecord(**v) for v in manifest.values()]

    def list_incomplete(self) -> List[DocumentRecord]:
        """Return all manifest entries that are not status=complete."""
        manifest = self._read_manifest()
        return [
            DocumentRecord(**v)
            for v in manifest.values()
            if v.get("status", "complete") != "complete"
        ]
