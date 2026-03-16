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
                        do_ocr=False,
                    ),
                )
            }
        )

    def convert(self, source: str) -> DoclingDocument:
        """Convert a PDF to a DoclingDocument.

        Args:
            source: Local file path or URL to a PDF.

        Returns:
            DoclingDocument containing layout, text, and tables.
        """
        return self._converter.convert(source=source).document
