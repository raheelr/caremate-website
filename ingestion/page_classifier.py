"""
Page Classifier
---------------
Scans ALL 747 pages of the STG PDF with pdfplumber to classify each page.
Determines which pages need Claude Vision (image-heavy, low text) vs
which are well-handled by text extraction alone.
"""

import pdfplumber
from dataclasses import dataclass


@dataclass
class PageClassification:
    page_num: int            # 1-indexed
    char_count: int          # pdfplumber extracted text chars
    has_images: bool         # pdfplumber.Page.images detected
    table_count: int         # pdfplumber.Page.find_tables()
    needs_vision: bool       # needs Claude Vision
    chapter_num: int         # 0 for front matter


# Chapter 12 page range (STIs with flowcharts)
CHAPTER_12_START_PAGE = 355  # Approximate — refined at runtime
CHAPTER_12_END_PAGE = 420


class PageClassifier:
    """Classifies every page of the STG PDF for extraction routing."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.classifications: dict[int, PageClassification] = {}

    def classify_all(self) -> dict[int, PageClassification]:
        """Scan all pages and classify each one. Returns {page_num: classification}."""
        print("  Classifying all pages...")

        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"  Total PDF pages: {total_pages}")

            chapter_12_pages = set()

            for i, page in enumerate(pdf.pages):
                page_num = i + 1  # 1-indexed
                text = page.extract_text() or ""
                char_count = len(text)
                has_images = bool(page.images)
                tables = page.find_tables() or []
                table_count = len(tables)

                # Detect chapter number from page text
                chapter_num = self._detect_chapter(text)

                if chapter_num == 12:
                    chapter_12_pages.add(page_num)

                # Determine if vision is needed
                needs_vision = self._needs_vision(
                    char_count=char_count,
                    has_images=has_images,
                    chapter_num=chapter_num,
                )

                self.classifications[page_num] = PageClassification(
                    page_num=page_num,
                    char_count=char_count,
                    has_images=has_images,
                    table_count=table_count,
                    needs_vision=needs_vision,
                    chapter_num=chapter_num,
                )

                if page_num % 100 == 0:
                    print(f"    Classified {page_num}/{total_pages} pages...")

        vision_count = sum(1 for c in self.classifications.values() if c.needs_vision)
        image_count = sum(1 for c in self.classifications.values() if c.has_images)
        table_count = sum(c.table_count for c in self.classifications.values())
        print(f"  Classification complete: {total_pages} pages, "
              f"{image_count} with images, {table_count} tables detected, "
              f"{vision_count} need Vision")

        return self.classifications

    def refine_with_docling(self, docling_results: dict) -> None:
        """
        After Docling runs, refine vision needs.
        If Docling also produced < 300 chars for a page with images, it definitely needs Vision.
        """
        added = 0
        for page_num, classification in self.classifications.items():
            if classification.needs_vision:
                continue  # Already flagged
            if not classification.has_images:
                continue  # No images to process

            docling_chars = 0
            if page_num in docling_results:
                docling_chars = docling_results[page_num].char_count

            # Post-Docling check: both pdfplumber AND Docling failed to get text
            if classification.char_count < 500 and docling_chars < 300:
                classification.needs_vision = True
                added += 1

        if added:
            print(f"  Refined: {added} additional pages flagged for Vision after Docling check")

    def get_vision_pages(self) -> list[int]:
        """Return sorted list of page numbers that need Claude Vision."""
        return sorted(
            p for p, c in self.classifications.items() if c.needs_vision
        )

    def _needs_vision(self, char_count: int, has_images: bool, chapter_num: int) -> bool:
        """Determine if a page needs Claude Vision extraction."""
        # Rule 1: Low text + images
        if char_count < 300 and has_images:
            return True

        # Rule 2: Chapter 12 (STI flowcharts) — all pages with images
        if chapter_num == 12 and has_images:
            return True

        # Rule 3: Has images but relatively low text
        if has_images and char_count < 500:
            return True

        return False

    def _detect_chapter(self, text: str) -> int:
        """Detect chapter number from page text."""
        import re
        # Look for "CHAPTER X" at the top of the page
        match = re.search(r'CHAPTER\s+(\d+)', text[:200], re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Look for condition codes like "12.1" at start of lines
        for line in text.split('\n')[:5]:
            code_match = re.match(r'^(\d+)\.\d+', line.strip())
            if code_match:
                return int(code_match.group(1))

        return 0
