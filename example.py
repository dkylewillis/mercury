"""
Example: ingest a PDF and run a semantic search with visual grounding.

Usage:
    python example.py
"""

import json
from pathlib import Path

from converter import Converter
from chunker import Chunker
from document_store import DocumentStore
from vector_store import VectorStore

PDF_PATH = "data/docling_tech_report.pdf"
QUERY = "Which are the main AI models in Docling?"
TOP_K = 3


def main():
    # ------------------------------------------------------------------
    # 1. Convert
    # ------------------------------------------------------------------
    print("Converting PDF...")
    converter = Converter()
    dl_doc = converter.convert(PDF_PATH)
    print(f"  document name : {dl_doc.name}")
    print(f"  binary hash   : {dl_doc.origin.binary_hash}")

    # ------------------------------------------------------------------
    # 2. Store PDF copy (needed for visual grounding later)
    # ------------------------------------------------------------------
    doc_store = DocumentStore("./doc_store")
    file_hash = doc_store.create(dl_doc, source_pdf_path=PDF_PATH)
    print(f"  saved to doc_store as {file_hash}.pdf")

    # ------------------------------------------------------------------
    # 3. Chunk
    # ------------------------------------------------------------------
    print("\nChunking...")
    chunker = Chunker(dl_doc)
    chunks = chunker.chunk()
    print(f"  {len(chunks)} chunks created")

    # ------------------------------------------------------------------
    # 4. Embed + store in ChromaDB
    # ------------------------------------------------------------------
    print("\nEmbedding and storing chunks...")
    vs = VectorStore(persist_directory="./chroma_data")
    vs.create(chunks)
    print("  done")

    # ------------------------------------------------------------------
    # 5. Semantic search
    # ------------------------------------------------------------------
    print(f'\nQuerying: "{QUERY}"')
    results = vs.query(QUERY, top_k=TOP_K, window=1)

    for i, result in enumerate(results, 1):
        chunk = result.chunk
        print(f"\n--- Result {i} (index {chunk.index}) ---")
        print(f"  page       : {chunk.page_number}")
        print(f"  headings   : {chunk.headings}")
        print(f"  text       : {chunk.text[:300]}{'...' if len(chunk.text) > 300 else ''}")
        print(f"  bboxes     : {len(chunk.bboxes)} bounding box(es)")
        print(f"  context window ({len(result.context)} chunk(s)):")
        for ctx in result.context:
            marker = " ◀ HIT" if ctx.index == chunk.index else ""
            print(f"    [{ctx.index}] {ctx.text[:120].replace(chr(10), ' ')}...{marker}")

    # ------------------------------------------------------------------
    # 6. Visual grounding (page image + bounding boxes)
    # ------------------------------------------------------------------
    if results and results[0].chunk.bboxes:
        print("\nVisual grounding for top result...")
        top_result = results[0]
        hit = top_result.chunk

        # Render pages on demand from the stored PDF
        from rendering import render_page
        pdf_path = doc_store.get_pdf_path(hit.file_hash)

        try:
            from PIL import ImageDraw

            # Collect all chunks to draw, keyed by page so we open each image once
            # hit chunk -> red, context chunks -> cornflowerblue
            pages: dict = {}  # page_no -> PIL image (mutable copy)
            for chunk in top_result.context:
                is_hit = chunk.index == hit.index
                color = "red" if is_hit else "cornflowerblue"
                for bbox in chunk.bboxes:
                    if bbox.page_no not in pages:
                        pages[bbox.page_no] = render_page(str(pdf_path), bbox.page_no).copy()
                    img = pages[bbox.page_no]
                    padding = 4
                    l = round(bbox.l * img.width)  - padding
                    r = round(bbox.r * img.width)  + padding
                    t = round(bbox.t * img.height) - padding
                    b = round(bbox.b * img.height) + padding
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(xy=[(l, t), (r, b)], outline=color, width=2)

            for page_no, img in pages.items():
                out_path = Path(f"grounded_page_{page_no}.png")
                img.save(out_path)
                print(f"  saved {out_path}  (red=hit, blue=context)")

        except ImportError:
            print("  (Pillow not installed — skipping image export)")
    else:
        print("\nNo bboxes available for visual grounding.")


if __name__ == "__main__":
    main()
