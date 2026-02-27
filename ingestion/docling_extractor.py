"""
Docling Extractor
-----------------
Processes the ENTIRE STG PDF through Docling for AI-powered extraction.
Docling uses TableFormer for high-accuracy table detection (97.9%)
and produces structured markdown with proper table formatting.

Caches results to avoid re-running the 4-10 min processing.
"""

import json
import os
import warnings
from dataclasses import dataclass, field


@dataclass
class DoclingPageResult:
    page_num: int
    markdown: str              # Full page as structured markdown
    tables: list[dict] = field(default_factory=list)  # [{headers, rows, raw_markdown}]
    char_count: int = 0


class DoclingExtractor:
    """Extracts all pages from a PDF using Docling's AI-powered parser."""

    def __init__(self, pdf_path: str, cache_path: str = "docling_cache.json"):
        self.pdf_path = pdf_path
        self.cache_path = cache_path

    def extract_all_pages(self) -> dict[int, DoclingPageResult]:
        """
        Process the entire PDF through Docling.
        Returns {page_num: DoclingPageResult} for all pages.
        Uses cache if available.
        """
        # Check cache first
        if os.path.exists(self.cache_path):
            print(f"  Loading Docling cache from {self.cache_path}...")
            return self._load_cache()

        print("  Running Docling on full PDF (this takes 4-10 minutes)...")

        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            print("  ERROR: docling not installed. Run: pip install docling>=2.5.0")
            print("  Returning empty results — pipeline will use pdfplumber only.")
            return {}

        # Configure Docling with table structure detection
        try:
            from docling.document_converter import PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = True

            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
        except (ImportError, TypeError, Exception) as e:
            # Fallback for different Docling API versions
            print(f"  Using default Docling configuration (config error: {e})...")
            converter = DocumentConverter()

        # Process the full PDF
        result = converter.convert(self.pdf_path)
        doc = result.document

        pages: dict[int, DoclingPageResult] = {}

        # Extract full markdown from the document
        full_md = doc.export_to_markdown()

        # Try to split by page using Docling's element provenance
        page_content: dict[int, list[str]] = {}
        page_tables: dict[int, list[dict]] = {}

        try:
            for element in doc.iterate_items():
                # iterate_items returns (NodeItem, level) tuples
                if isinstance(element, tuple):
                    item = element[0]
                else:
                    item = element

                # Get page number from provenance
                page_num = 1
                if hasattr(item, 'prov') and item.prov:
                    prov_list = item.prov if isinstance(item.prov, list) else [item.prov]
                    for prov in prov_list:
                        if hasattr(prov, 'page_no') and prov.page_no:
                            page_num = prov.page_no
                            break

                if page_num not in page_content:
                    page_content[page_num] = []
                    page_tables[page_num] = []

                # Export element to markdown
                md = ""
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    if hasattr(item, 'export_to_markdown'):
                        try:
                            md = item.export_to_markdown(doc)
                        except TypeError:
                            try:
                                md = item.export_to_markdown()
                            except Exception:
                                md = getattr(item, 'text', str(item))
                    elif hasattr(item, 'text'):
                        md = item.text
                    else:
                        md = str(item)

                if md and md.strip():
                    page_content[page_num].append(md)

                # Check if this is a table element
                item_type = type(item).__name__
                is_table = (
                    item_type in ('TableItem', 'Table', 'TableData')
                    or (hasattr(item, 'data') and hasattr(item.data, 'table_cells'))
                )
                if is_table:
                    table_dict = self._parse_table_element(item)
                    if table_dict:
                        page_tables[page_num].append(table_dict)

        except Exception as e:
            print(f"  Warning: element iteration failed ({e}), using full markdown")
            page_content = {}

        # Build results from per-page content
        for page_num in sorted(page_content.keys()):
            md_text = '\n\n'.join(page_content[page_num])
            pages[page_num] = DoclingPageResult(
                page_num=page_num,
                markdown=md_text,
                tables=page_tables.get(page_num, []),
                char_count=len(md_text),
            )

        # If per-page splitting failed, fall back to full markdown as page 1
        if not pages and full_md:
            pages[1] = DoclingPageResult(
                page_num=1,
                markdown=full_md,
                tables=[],
                char_count=len(full_md),
            )

        print(f"  Docling extracted {len(pages)} pages with content")

        # Cache results
        self._save_cache(pages)

        return pages

    def _parse_table_element(self, element) -> dict | None:
        """Parse a Docling table element into {headers, rows, raw_markdown}."""
        try:
            if hasattr(element, 'export_to_markdown'):
                raw_md = element.export_to_markdown()
            else:
                return None

            lines = [l.strip() for l in raw_md.strip().split('\n') if l.strip()]
            if len(lines) < 2:
                return None

            # Parse markdown table
            headers = [c.strip() for c in lines[0].split('|') if c.strip()]
            rows = []
            for line in lines[2:]:  # Skip header separator
                if '|' in line:
                    row = [c.strip() for c in line.split('|') if c.strip()]
                    if row:
                        rows.append(row)

            return {
                'headers': headers,
                'rows': rows,
                'raw_markdown': raw_md,
            }
        except Exception:
            return None

    def _save_cache(self, pages: dict[int, DoclingPageResult]):
        """Save Docling results to JSON cache."""
        cache_data = {}
        for page_num, result in pages.items():
            cache_data[str(page_num)] = {
                'page_num': result.page_num,
                'markdown': result.markdown,
                'tables': result.tables,
                'char_count': result.char_count,
            }

        with open(self.cache_path, 'w') as f:
            json.dump(cache_data, f)

        print(f"  Cached Docling results to {self.cache_path}")

    def _load_cache(self) -> dict[int, DoclingPageResult]:
        """Load Docling results from JSON cache."""
        with open(self.cache_path) as f:
            cache_data = json.load(f)

        pages = {}
        for page_str, data in cache_data.items():
            page_num = int(page_str)
            pages[page_num] = DoclingPageResult(
                page_num=data['page_num'],
                markdown=data['markdown'],
                tables=data.get('tables', []),
                char_count=data['char_count'],
            )

        print(f"  Loaded {len(pages)} pages from cache")
        return pages
