from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class BoundingBox(BaseModel):
    """Normalized bounding box (0-1 coordinates, top-left origin) for a page region."""

    l: float
    r: float
    t: float
    b: float
    page_no: int


class Chunk(BaseModel):
    """A piece of text extracted from a document with full provenance metadata."""

    id: str
    index: int
    text: str
    page_number: Optional[int] = None
    headings: List[str] = Field(default_factory=list)
    doc_items: List[str] = Field(default_factory=list)
    bboxes: List[BoundingBox] = Field(default_factory=list)
    document_name: str = ""
    file_hash: str = ""
    file_extension: str = ""


class DocumentRecord(BaseModel):
    """Lightweight metadata record stored in the manifest for GUI listing/filtering."""

    file_hash: str
    document_name: str
    filename: str
    file_extension: str
    page_count: int
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # "complete" is the default so that old manifest entries without this field
    # are treated as successfully ingested.
    status: Literal["pending", "complete", "failed"] = "complete"
    pdf_path: str = ""


class QueryResult(BaseModel):
    """A semantic search hit together with its surrounding context window."""

    chunk: Chunk
    """The chunk that matched the query."""
    context: List[Chunk] = Field(default_factory=list)
    """Chunks in the window around the hit, sorted by index (includes the hit)."""
