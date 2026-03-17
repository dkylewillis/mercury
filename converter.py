from typing import Generator, Tuple

import pypdfium2 as pdfium

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter as _DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument


class Converter:
    """Converts documents to DoclingDocument. Page images are rendered on demand for visual grounding."""

    def __init__(self):
        self._converter = _DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=PdfPipelineOptions(
                        generate_page_images=False,
                        generate_picture_images=False,
                        generate_table_images=False,
                        do_ocr=False,
                        images_scale=1.0,
                    ),
                )
            }
        )

    @staticmethod
    def page_count(source: str) -> int:
        """Return the total page count of a PDF without running the ML pipeline.

        Uses pypdfium2 for a fast read of the PDF cross-reference table.

        Args:
            source: Local path to a PDF file.

        Returns:
            Total number of pages.
        """
        doc = pdfium.PdfDocument(source)
        try:
            return len(doc)
        finally:
            doc.close()

    def convert(self, source: str) -> DoclingDocument:
        """Convert a PDF to a DoclingDocument.

        Args:
            source: Local file path or URL to a PDF.

        Returns:
            DoclingDocument containing layout, text, and tables.
        """
        return self._converter.convert(source=source).document

    def convert_page_range(self, source: str, start_page: int, end_page: int) -> DoclingDocument:
        """Convert a contiguous page range of a PDF to a DoclingDocument.

        This avoids loading the entire document into memory at once, which is
        useful for very large PDFs that cause memory spikes on specific pages.

        Args:
            source: Local file path to a PDF.
            start_page: First page to convert (1-indexed, inclusive).
            end_page: Last page to convert (1-indexed, inclusive).

        Returns:
            DoclingDocument for the requested page range.
        """
        return self._converter.convert(
            source=source,
            page_range=(start_page, end_page),
        ).document

    def convert_in_page_chunks(
        self, source: str, chunk_size: int
    ) -> Generator[Tuple[int, int, DoclingDocument], None, None]:
        """Convert a PDF in sequential page-range chunks to limit peak memory use.

        Converts ``chunk_size`` pages at a time (e.g. 1–50, 51–100, …) so that
        the ML pipeline never has to hold the full document in memory at once.
        Use this for documents with memory-spike pages.

        Args:
            source: Local file path to a PDF.
            chunk_size: Number of pages per chunk (e.g. 50).

        Yields:
            ``(start_page, end_page, DoclingDocument)`` tuples where
            *start_page* and *end_page* are 1-indexed and inclusive.
        """
        total_pages = self.page_count(source)
        for start in range(1, total_pages + 1, chunk_size):
            end = min(start + chunk_size - 1, total_pages)
            yield start, end, self.convert_page_range(source, start, end)
