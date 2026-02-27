"""
Multi-Source Merger
-------------------
Merges the best output from three extraction tools (pdfplumber, Docling,
Claude Vision) for each condition segment. Produces a unified
MergedConditionInput that the extractor can work with.

Merge strategy:
- Use Docling as primary if it has >= 80% of pdfplumber's content
- Use pdfplumber as primary otherwise
- Always append Vision content when available (additive)
- Append Docling tables as a dedicated section
"""

from dataclasses import dataclass, field
from typing import Optional

from segmenter import ConditionSegment


@dataclass
class MergedConditionInput:
    """Unified input for the Claude extractor, combining all extraction sources."""
    stg_code: str
    name: str
    icd10_codes: list[str]
    start_page: int
    end_page: int
    chapter_number: int
    chapter_name: str

    # Merged content
    merged_text: str           # Best combined text
    tables_markdown: str = ""  # Docling tables in markdown
    tables_structured: list = field(default_factory=list)  # Docling tables as {headers, rows}
    vision_content: str = ""   # Flowchart/diagram descriptions

    # Source metrics
    pdfplumber_chars: int = 0
    docling_chars: int = 0
    vision_chars: int = 0
    primary_source: str = "pdfplumber"  # 'pdfplumber' | 'docling' | 'vision'

    # Preserve pdfplumber sections
    sections: dict = field(default_factory=dict)

    # Computed properties
    @property
    def raw_text(self) -> str:
        """Full combined text for extraction prompt. Backward compat with ConditionSegment."""
        parts = [self.merged_text]
        if self.tables_markdown:
            parts.append(f"\n\n--- STRUCTURED TABLES (from Docling) ---\n{self.tables_markdown}")
        if self.vision_content:
            parts.append(f"\n\n--- VISUAL CONTENT (from Claude Vision) ---\n{self.vision_content}")
        return '\n'.join(parts)

    @property
    def display_name(self) -> str:
        return f"{self.stg_code} {self.name}"

    @property
    def has_tables(self) -> bool:
        return bool(self.tables_markdown)

    @property
    def has_vision(self) -> bool:
        return bool(self.vision_content)

    @property
    def source_pages(self) -> list[int]:
        return list(range(self.start_page, self.end_page + 1))


class MultiSourceMerger:
    """Merges pdfplumber, Docling, and Vision outputs per condition."""

    def __init__(
        self,
        segments: list[ConditionSegment],
        docling_pages: dict = None,   # {page_num: DoclingPageResult}
        vision_pages: dict = None,    # {page_num: VisionPageResult}
    ):
        self.segments = segments
        self.docling_pages = docling_pages or {}
        self.vision_pages = vision_pages or {}

    def merge_all(self) -> list[MergedConditionInput]:
        """Merge all sources for every condition segment."""
        print(f"  Merging {len(self.segments)} conditions from "
              f"{len(self.docling_pages)} Docling pages and "
              f"{len(self.vision_pages)} Vision pages...")

        results = []
        docling_primary_count = 0
        vision_added_count = 0
        table_count = 0

        for segment in self.segments:
            merged = self._merge_single(segment)
            results.append(merged)

            if merged.primary_source == 'docling':
                docling_primary_count += 1
            if merged.has_vision:
                vision_added_count += 1
            if merged.has_tables:
                table_count += 1

        print(f"  Merge complete: {len(results)} conditions")
        print(f"    Primary source — pdfplumber: {len(results) - docling_primary_count}, "
              f"docling: {docling_primary_count}")
        print(f"    Vision content added: {vision_added_count}")
        print(f"    Tables found: {table_count}")

        return results

    def _merge_single(self, segment: ConditionSegment) -> MergedConditionInput:
        """Merge all sources for a single condition."""
        pdfplumber_text = segment.raw_text
        pdfplumber_chars = len(pdfplumber_text)

        # Gather Docling content for this condition's page range
        docling_text, docling_tables_md, docling_tables_struct = self._gather_docling(
            segment.start_page, segment.end_page
        )
        docling_chars = len(docling_text)

        # Gather Vision content for this condition's page range
        vision_text = self._gather_vision(segment.start_page, segment.end_page)
        vision_chars = len(vision_text)

        # Choose primary source
        primary_source = 'pdfplumber'
        merged_text = pdfplumber_text

        if docling_chars > 0 and docling_chars >= pdfplumber_chars * 0.8:
            # Docling has sufficient content — use it (better table handling)
            primary_source = 'docling'
            merged_text = docling_text
        elif docling_chars > pdfplumber_chars:
            # Docling has MORE content even if below 80% threshold
            primary_source = 'docling'
            merged_text = docling_text

        # Special case: if pdfplumber has almost nothing, use whatever has most
        if pdfplumber_chars < 100:
            if vision_chars > docling_chars and vision_chars > pdfplumber_chars:
                primary_source = 'vision'
                merged_text = vision_text
            elif docling_chars > pdfplumber_chars:
                primary_source = 'docling'
                merged_text = docling_text

        return MergedConditionInput(
            stg_code=segment.stg_code,
            name=segment.name,
            icd10_codes=segment.icd10_codes,
            start_page=segment.start_page,
            end_page=segment.end_page,
            chapter_number=segment.chapter_number,
            chapter_name=segment.chapter_name,
            merged_text=merged_text,
            tables_markdown=docling_tables_md,
            tables_structured=docling_tables_struct,
            vision_content=vision_text,
            pdfplumber_chars=pdfplumber_chars,
            docling_chars=docling_chars,
            vision_chars=vision_chars,
            primary_source=primary_source,
            sections=segment.sections,
        )

    def _gather_docling(self, start_page: int, end_page: int) -> tuple[str, str, list]:
        """Gather Docling content for a page range."""
        texts = []
        table_mds = []
        table_structs = []

        for page_num in range(start_page, end_page + 1):
            if page_num not in self.docling_pages:
                continue
            page_result = self.docling_pages[page_num]
            if page_result.markdown:
                texts.append(page_result.markdown)
            for table in page_result.tables:
                if table.get('raw_markdown'):
                    table_mds.append(table['raw_markdown'])
                table_structs.append(table)

        return (
            '\n\n'.join(texts),
            '\n\n'.join(table_mds),
            table_structs,
        )

    def _gather_vision(self, start_page: int, end_page: int) -> str:
        """Gather Vision content for a page range."""
        texts = []
        for page_num in range(start_page, end_page + 1):
            if page_num not in self.vision_pages:
                continue
            vision_result = self.vision_pages[page_num]
            if vision_result.description:
                texts.append(f"[Page {page_num}]\n{vision_result.description}")
        return '\n\n'.join(texts)
