"""
Vision Extractor
----------------
Uses Claude Vision (Haiku) to extract clinical content from image-heavy
PDF pages — flowcharts, diagrams, and pages where text extraction fails.

Converts PDF pages to PNG at 200 DPI via PyMuPDF (fitz), then sends
to Claude with a clinical extraction prompt.
"""

import base64
import io
import time
import anthropic
from dataclasses import dataclass, field


VISION_PROMPT = """You are extracting clinical content from a page of the South African
Standard Treatment Guidelines (STG). This page contains images, flowcharts, or diagrams
that cannot be read by text extraction tools.

Extract ALL visible clinical content from this image:

1. **Flowcharts/Algorithms**: Describe every decision path step by step.
   Format: "Step 1: [condition check] → YES: [action] / NO: [next step]"

2. **Tables**: Reproduce in markdown table format with all rows and columns.

3. **All visible text**: Include every piece of text you can read, preserving headings.

4. **Condition headings**: Note any condition names with their STG codes (e.g. "12.1 VDS").

5. **Medicine names and doses**: Extract every medicine mentioned with its dose, route,
   frequency, and duration.

6. **Referral criteria**: Any "Refer" or "Refer urgently" text.

Be thorough — extract EVERYTHING visible on this page. Do not summarize or skip content.
If text is partially obscured, note what you can read and flag uncertainty."""


@dataclass
class VisionPageResult:
    page_num: int
    description: str           # Full clinical content from the image
    flowchart_steps: list[str] = field(default_factory=list)
    token_cost: int = 0


class VisionExtractor:
    """Extracts clinical content from PDF pages using Claude Vision."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = "claude-haiku-4-5-20251001"
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def extract_pages(
        self,
        pdf_path: str,
        page_nums: list[int],
    ) -> dict[int, VisionPageResult]:
        """
        Extract clinical content from specified pages using Claude Vision.

        Args:
            pdf_path: Path to the STG PDF
            page_nums: List of 1-indexed page numbers to process

        Returns:
            {page_num: VisionPageResult}
        """
        if not page_nums:
            return {}

        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("  ERROR: PyMuPDF not installed. Run: pip install PyMuPDF>=1.24.0")
            print("  Skipping Vision extraction.")
            return {}

        print(f"  Processing {len(page_nums)} pages with Claude Vision...")
        results = {}

        doc = fitz.open(pdf_path)

        for i, page_num in enumerate(page_nums):
            page_idx = page_num - 1  # fitz uses 0-indexed
            if page_idx < 0 or page_idx >= len(doc):
                print(f"    Page {page_num}: out of range, skipping")
                continue

            print(f"    [{i+1}/{len(page_nums)}] Page {page_num}...")

            # Render page to PNG at 200 DPI
            page = doc[page_idx]
            mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')

            # Send to Claude Vision
            result = self._extract_single_page(page_num, img_b64)
            if result:
                results[page_num] = result

            # Rate limiting: small pause between requests
            if i < len(page_nums) - 1:
                time.sleep(0.5)

        doc.close()

        total_tokens = self.total_input_tokens + self.total_output_tokens
        print(f"  Vision extraction complete: {len(results)}/{len(page_nums)} pages, "
              f"{total_tokens:,} tokens used")

        return results

    def _extract_single_page(self, page_num: int, img_b64: str) -> VisionPageResult | None:
        """Extract content from a single page image."""
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_b64,
                                }
                            },
                            {
                                "type": "text",
                                "text": VISION_PROMPT,
                            }
                        ]
                    }]
                )

                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens

                description = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        description += block.text

                # Parse flowchart steps from the description
                flowchart_steps = []
                for line in description.split('\n'):
                    stripped = line.strip()
                    if stripped.startswith('Step ') or '→' in stripped:
                        flowchart_steps.append(stripped)

                return VisionPageResult(
                    page_num=page_num,
                    description=description,
                    flowchart_steps=flowchart_steps,
                    token_cost=response.usage.input_tokens + response.usage.output_tokens,
                )

            except Exception as e:
                if "overloaded" in str(e).lower() and attempt < 2:
                    wait = (attempt + 1) * 10
                    print(f"      API overloaded — waiting {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"      Vision failed for page {page_num}: {e}")
                return None

        return None

    def print_cost_summary(self):
        """Print token usage for Vision extraction."""
        print(f"\n  Vision Token Usage:")
        print(f"    Input tokens:  {self.total_input_tokens:,}")
        print(f"    Output tokens: {self.total_output_tokens:,}")
        cost = (self.total_input_tokens * 0.0000008) + (self.total_output_tokens * 0.000004)
        print(f"    Estimated cost: ~${cost:.2f}")
