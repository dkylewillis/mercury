import os
from typing import List

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownParams
from docling_core.types.doc.document import DoclingDocument

from models import BoundingBox, Chunk


class _ImgPlaceholderSerializerProvider(ChunkingSerializerProvider):
    """Replaces images with a placeholder comment to keep chunk text clean."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            params=MarkdownParams(image_placeholder="<!-- image -->"),
        )


class Chunker:
    """Chunks a DoclingDocument into :class:`~models.Chunk` objects.

    Each chunk carries the original text (contextualized with surrounding
    headings), normalized bounding boxes for every covered page region, and
    the document provenance needed for visual grounding.
    """

    def __init__(
        self,
        dl_doc: DoclingDocument,
        tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Args:
            dl_doc: The converted DoclingDocument to chunk.
            tokenizer: HuggingFace model ID used by HybridChunker for
                       token-limit-aware splitting.  Must match the embedding
                       model used in VectorStore so chunk sizes are correct.
        """
        self.dl_doc = dl_doc
        self.file_hash = str(dl_doc.origin.binary_hash) if dl_doc.origin else ""
        self.filename = dl_doc.origin.filename if dl_doc.origin else ""
        self.file_extension = os.path.splitext(self.filename)[1] if self.filename else ".pdf"

        # HybridChunker accepts a HuggingFace model ID string directly
        self._chunker = HybridChunker(
            tokenizer=tokenizer,
            serializer_provider=_ImgPlaceholderSerializerProvider(),
        )

    def chunk(self) -> List[Chunk]:
        """Split the document into chunks with text, headings, and bounding boxes.

        Returns:
            Ordered list of :class:`~models.Chunk` objects.  Each chunk ID is
            ``"{file_hash}_{index}"`` to stay unique across documents in the
            same vector store collection.
        """
        raw_chunks = list(self._chunker.chunk(self.dl_doc))
        if not raw_chunks:
            return []

        chunks: List[Chunk] = []
        for raw in raw_chunks:
            index = len(chunks)
            doc_item_refs = [item.self_ref for item in raw.meta.doc_items]
            contextualized_text = self._chunker.contextualize(chunk=raw)

            bboxes: List[BoundingBox] = []
            for doc_item in raw.meta.doc_items:
                if doc_item.prov:
                    for prov in doc_item.prov:
                        try:
                            page = self.dl_doc.pages[prov.page_no]
                            bbox = prov.bbox.to_top_left_origin(page_height=page.size.height)
                            bbox = bbox.normalized(page.size)
                            bboxes.append(
                                BoundingBox(
                                    l=bbox.l,
                                    r=bbox.r,
                                    t=bbox.t,
                                    b=bbox.b,
                                    page_no=prov.page_no,
                                )
                            )
                        except (IndexError, AttributeError):
                            pass

            chunks.append(
                Chunk(
                    id=f"{self.file_hash}_{index}",
                    index=index,
                    text=contextualized_text,
                    page_number=bboxes[0].page_no if bboxes else None,
                    headings=getattr(raw.meta, "headings", None) or [],
                    doc_items=doc_item_refs,
                    bboxes=bboxes,
                    document_name=self.dl_doc.name,
                    file_hash=self.file_hash,
                    file_extension=self.file_extension,
                )
            )

        return chunks
