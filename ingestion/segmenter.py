"""
STG PDF Segmenter
-----------------
Splits the 747-page STG into individual condition segments.
Each condition gets its own text block, ready for Claude extraction.

Key insight: The STG has predictable numbered headings like:
  "1.2 CANDIDIASIS, ORAL (THRUSH)"
  "2.9.1 DIARRHOEA, ACUTE IN CHILDREN"

We use these as natural condition boundaries.
"""

import re
import pdfplumber
from dataclasses import dataclass, field
from typing import Optional


# ── Patterns ────────────────────────────────────────────────────────────────

# Matches condition headings: "1.2 CANDIDIASIS, ORAL (THRUSH)"
CONDITION_HEADING = re.compile(
    r'^(\d+\.\d+(?:\.\d+)*)\s+([A-Z][A-Z\s\(\)/,\-\'\d]+)$'
)

# ICD-10 code pattern (appears on line after condition heading)
ICD10_CODE = re.compile(r'^[A-Z]\d{2}(?:\.\d+)?(?:[/-][A-Z0-9\.]+)*$')

# Chapter header (appears at top of each page)
CHAPTER_HEADER = re.compile(r'^CHAPTER\s+\d+\s+', re.IGNORECASE)

# Section markers within a condition
SECTION_MARKERS = {
    'DESCRIPTION':          'description',
    'DANGER SIGNS':         'danger_signs',
    'GENERAL MEASURES':     'general_measures',
    'MEDICINE TREATMENT':   'medicine_treatment',
    'REFERRAL':             'referral',
    'PROPHYLAXIS':          'general_measures',
    'INVESTIGATIONS':       'description',
    'DIAGNOSIS':            'description',
}

# Pages before clinical content (front matter)
CLINICAL_START_PAGE = 48  # 0-indexed (page 49 in the PDF)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ConditionSegment:
    """A single condition extracted from the STG."""
    stg_code: str           # "1.2"
    name: str               # "CANDIDIASIS, ORAL (THRUSH)"
    icd10_codes: list[str]  # ["B37.0"]
    start_page: int         # 1-indexed PDF page number
    end_page: int
    
    # Raw text by section
    raw_text: str = ""
    sections: dict = field(default_factory=dict)
    
    # Chapter context
    chapter_number: int = 0
    chapter_name: str = ""
    
    @property
    def display_name(self):
        return f"{self.stg_code} {self.name}"
    
    @property
    def has_danger_signs(self):
        return bool(self.sections.get('danger_signs', '').strip())
    
    @property 
    def has_medicine_treatment(self):
        return bool(self.sections.get('medicine_treatment', '').strip())


# ── Main segmenter ───────────────────────────────────────────────────────────

class STGSegmenter:
    """
    Segments the STG PDF into individual condition blocks.
    
    Strategy:
    1. Extract text page by page
    2. Identify condition headings by their numbered format
    3. Everything between two headings belongs to the first condition
    4. Parse sections within each condition block
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self._chapters = self._build_chapter_map()
    
    def _build_chapter_map(self) -> dict[int, str]:
        """Map chapter numbers to their names."""
        return {
            1: "Dental and Oral Conditions",
            2: "Gastro-Intestinal Conditions",
            3: "Nutrition and Anaemia",
            4: "Cardiovascular Conditions",
            5: "Dermatological Conditions",
            6: "Endocrine Conditions",
            7: "Eye Conditions",  # Note: reordered vs TOC
            8: "Urinary Conditions",
            9: "Obstetrics and Gynaecology",
            10: "Infectious Conditions",
            11: "HIV and AIDS",
            12: "Sexually Transmitted Infections",
            13: "Immunisation",
            14: "Musculoskeletal Conditions",
            15: "Central Nervous System Conditions",
            16: "Mental Health Conditions",
            17: "Respiratory Conditions",
            18: "Eye Conditions",
            19: "Ear, Nose and Throat Conditions",
            20: "Pain",
            21: "Emergencies and Injuries",
            22: "Palliative Care",
            23: "Paediatric Dosing Tables",
        }
    
    def segment(self) -> list[ConditionSegment]:
        """
        Main entry point. Returns list of all conditions found in the STG.
        """
        print(f"Opening PDF: {self.pdf_path}")
        
        # Step 1: Extract all pages as text
        pages = self._extract_pages()
        print(f"Extracted {len(pages)} clinical pages")
        
        # Step 2: Find all condition boundaries
        boundaries = self._find_condition_boundaries(pages)
        print(f"Found {len(boundaries)} condition headings")
        
        # Step 3: Slice text into condition segments
        segments = self._slice_into_segments(pages, boundaries)
        print(f"Created {len(segments)} condition segments")
        
        # Step 4: Parse sections within each segment
        for seg in segments:
            seg.sections = self._parse_sections(seg.raw_text)
        
        # Step 5: Filter out heading-only parent sections
        # These are headings like "4.7 HYPERTENSION" that have no clinical content —
        # their actual content lives in sub-sections like "4.7.1 HYPERTENSION IN ADULTS"
        filtered = []
        for seg in segments:
            # Count meaningful content lines (exclude the heading itself and ICD codes)
            lines = [l for l in seg.raw_text.split('\n') if l.strip()]
            content_lines = [
                l for l in lines
                if not CONDITION_HEADING.match(l.strip())
                and not ICD10_CODE.match(l.strip())
            ]
            content_text = ' '.join(content_lines)
            
            # Check for redirect-only sections ("See section X.Y" with no real content)
            redirect_keywords = ['see section', 'refer to section', 'refer to chapter', 'see chapter']
            is_redirect_only = (
                len(content_text) < 400 and
                any(kw in content_text.lower() for kw in redirect_keywords) and
                not any(kw in content_text.upper() for kw in ['DESCRIPTION', 'MEDICINE', 'TREATMENT', 'DANGER'])
            )
            
            if is_redirect_only:
                # Always skip redirect-only sections — no useful content to extract
                pass
            elif len(content_text) >= 150:
                filtered.append(seg)
            else:
                # Only skip if there are sub-sections to handle it
                # (i.e. another segment starting with this code + '.')
                has_children = any(
                    s.stg_code.startswith(seg.stg_code + '.')
                    for s in segments
                )
                if not has_children:
                    filtered.append(seg)  # Keep it — no children will cover it
        
        skipped = len(segments) - len(filtered)
        if skipped:
            print(f"  Skipped {skipped} heading-only parent sections (content in sub-sections)")
        
        return filtered
    
    def _extract_pages(self) -> list[dict]:
        """Extract text from all clinical pages."""
        pages = []
        with pdfplumber.open(self.pdf_path) as pdf:
            total = len(pdf.pages)
            for i in range(CLINICAL_START_PAGE, total):
                page = pdf.pages[i]
                text = page.extract_text() or ""
                # Strip the chapter header that appears on every page
                text = self._strip_page_header(text)
                pages.append({
                    'page_num': i + 1,  # 1-indexed
                    'text': text,
                    'lines': [l.strip() for l in text.split('\n') if l.strip()]
                })
        return pages
    
    def _strip_page_header(self, text: str) -> str:
        """Remove repeating chapter headers from page tops."""
        lines = text.split('\n')
        if lines and CHAPTER_HEADER.match(lines[0].strip()):
            lines = lines[1:]
        return '\n'.join(lines)
    
    def _find_condition_boundaries(self, pages: list[dict]) -> list[dict]:
        """
        Find every condition heading and its location.
        Returns list of {stg_code, name, page_num, line_idx}
        """
        boundaries = []
        
        for page_data in pages:
            lines = page_data['lines']
            page_num = page_data['page_num']
            
            for i, line in enumerate(lines):
                match = CONDITION_HEADING.match(line)
                if not match:
                    continue
                
                stg_code = match.group(1)
                name = match.group(2).strip()
                
                # Skip very short "names" that are likely false positives
                if len(name) < 3:
                    continue
                
                # Look ahead for ICD-10 code (usually next non-empty line)
                icd_codes = []
                for j in range(i + 1, min(i + 4, len(lines))):
                    if ICD10_CODE.match(lines[j]):
                        # Could be multiple codes on same line separated by /
                        codes = re.split(r'[/\-]', lines[j])
                        icd_codes = [c.strip() for c in codes if ICD10_CODE.match(c.strip())]
                        break
                
                chapter_num = int(stg_code.split('.')[0])
                
                boundaries.append({
                    'stg_code': stg_code,
                    'name': name,
                    'icd10_codes': icd_codes,
                    'page_num': page_num,
                    'chapter_num': chapter_num,
                    'chapter_name': self._chapters.get(chapter_num, f"Chapter {chapter_num}"),
                })
        
        return boundaries
    
    def _slice_into_segments(
        self, 
        pages: list[dict], 
        boundaries: list[dict]
    ) -> list[ConditionSegment]:
        """
        Given condition boundaries, extract the text that belongs to each.
        Everything between boundary[n] and boundary[n+1] is condition n's content.
        """
        if not boundaries:
            return []
        
        # Build a flat list of (page_num, line_idx, text)
        all_lines = []
        for page_data in pages:
            for line in page_data['lines']:
                all_lines.append((page_data['page_num'], line))
        
        # Find the position of each boundary in all_lines
        def find_line_position(boundary: dict) -> int:
            target_code = boundary['stg_code']
            for idx, (page, line) in enumerate(all_lines):
                if page == boundary['page_num']:
                    m = CONDITION_HEADING.match(line)
                    if m and m.group(1) == target_code:
                        return idx
            return -1
        
        segments = []
        for i, boundary in enumerate(boundaries):
            start_pos = find_line_position(boundary)
            if start_pos == -1:
                continue
            
            # End is either the next boundary or end of document
            if i + 1 < len(boundaries):
                end_pos = find_line_position(boundaries[i + 1])
                if end_pos == -1:
                    end_pos = len(all_lines)
            else:
                end_pos = len(all_lines)
            
            # Extract the text block
            segment_lines = all_lines[start_pos:end_pos]
            raw_text = '\n'.join(line for _, line in segment_lines)
            
            # Determine page range
            pages_in_segment = list({page for page, _ in segment_lines})
            start_page = min(pages_in_segment) if pages_in_segment else boundary['page_num']
            end_page = max(pages_in_segment) if pages_in_segment else boundary['page_num']
            
            segments.append(ConditionSegment(
                stg_code=boundary['stg_code'],
                name=boundary['name'],
                icd10_codes=boundary['icd10_codes'],
                start_page=start_page,
                end_page=end_page,
                raw_text=raw_text,
                chapter_number=boundary['chapter_num'],
                chapter_name=boundary['chapter_name'],
            ))
        
        return segments
    
    def _parse_sections(self, raw_text: str) -> dict[str, str]:
        """
        Split a condition's raw text into its clinical sections.
        DESCRIPTION, DANGER SIGNS, GENERAL MEASURES, 
        MEDICINE TREATMENT, REFERRAL
        """
        sections = {}
        current_section = 'preamble'
        current_lines = []
        
        for line in raw_text.split('\n'):
            stripped = line.strip().upper()
            
            # Check if this line is a section marker
            matched_section = None
            for marker, section_key in SECTION_MARKERS.items():
                if stripped == marker or stripped.startswith(marker + ':'):
                    matched_section = section_key
                    break
            
            if matched_section:
                # Save what we had
                if current_lines:
                    existing = sections.get(current_section, '')
                    sections[current_section] = (existing + '\n' + '\n'.join(current_lines)).strip()
                current_section = matched_section
                current_lines = []
            else:
                current_lines.append(line)
        
        # Save the last section
        if current_lines:
            existing = sections.get(current_section, '')
            sections[current_section] = (existing + '\n' + '\n'.join(current_lines)).strip()
        
        return sections


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    pdf_path = "/mnt/user-data/uploads/Primary-Healthcare-Standard-Treatment-Guidelines-and-Essential-Medicines-List-8th-Edition-2024-Updated-December-2025__1_.pdf"
    
    segmenter = STGSegmenter(pdf_path)
    segments = segmenter.segment()
    
    print(f"\n{'='*60}")
    print(f"SEGMENTATION COMPLETE")
    print(f"Total conditions: {len(segments)}")
    print(f"{'='*60}")
    
    # Show first 5
    for seg in segments[:5]:
        print(f"\n{seg.display_name}")
        print(f"  ICD-10: {seg.icd10_codes}")
        print(f"  Pages: {seg.start_page}-{seg.end_page}")
        print(f"  Chapter: {seg.chapter_name}")
        print(f"  Sections: {list(seg.sections.keys())}")
        print(f"  Has danger signs: {seg.has_danger_signs}")
        print(f"  Has treatment: {seg.has_medicine_treatment}")
        if seg.sections.get('medicine_treatment'):
            preview = seg.sections['medicine_treatment'][:200]
            print(f"  Treatment preview: {preview}...")
