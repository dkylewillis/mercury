import pypdfium2


def render_page(pdf_path: str, page_number: int, scale: float = 2.0):
    """Render a single PDF page as a PIL Image for visual grounding.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page number.
        scale: Resolution scale (2.0 = ~144 DPI).

    Returns:
        PIL.Image.Image of the rendered page.
    """
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        page = pdf[page_number - 1]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        page.close()
    finally:
        pdf.close()
    return image
