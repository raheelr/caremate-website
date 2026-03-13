"""
Markdown Knowledge Base Search
------------------------------
File-based search across .claude-plugin/knowledge-base/ markdown files.
No DB, no embeddings — simple keyword matching over structured markdown sections.

Used by the Clinical Assistant's `search_knowledge_base` tool to find
hospital-level, paediatric, maternal, and other guidelines beyond the
primary care STG database.
"""

import os
import re
from dataclasses import dataclass

# ── Constants ────────────────────────────────────────────────────────────────

KB_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".claude-plugin", "knowledge-base",
)

# Map source filter values → subdirectory names
SOURCE_DIRS = {
    "hospital-eml": "hospital-eml",
    "paediatric-eml": "paediatric-eml",
    "maternal-perinatal": "maternal-perinatal",
    "obstetrics-gynae": "obstetrics-gynae",
    "sats-triage": "sats-triage",
    "stg-primary": "stg-primary",
    "road-to-health": "road-to-health",
}

# Source directory → care level label
SOURCE_LABELS = {
    "hospital-eml": "Hospital Level (Adult EML 2019)",
    "paediatric-eml": "Hospital Level (Paediatric STG 2017)",
    "maternal-perinatal": "Maternal & Perinatal Care (2024)",
    "obstetrics-gynae": "O&G Guidelines (SASOG/BetterObs)",
    "sats-triage": "SA Triage Scale (SATS/TEWS)",
    "stg-primary": "Primary Care (STG 2020)",
    "road-to-health": "Road to Health (Under-5 Care)",
}


@dataclass
class KBSection:
    """A matched section from the knowledge base."""
    heading: str
    content: str
    source_file: str
    source_label: str
    parent_heading: str  # The H1/chapter heading
    score: float


def _tokenize_query(query: str) -> list[str]:
    """Split query into lowercase searchable tokens, removing stop words."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
        "for", "of", "with", "and", "or", "not", "this", "that", "it", "by",
        "from", "as", "be", "has", "have", "had", "do", "does", "did", "will",
        "would", "can", "could", "should", "may", "might", "what", "how",
        "when", "where", "which", "who", "why", "i", "my", "me", "we",
    }
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def _parse_sections(text: str) -> list[tuple[str, str]]:
    """Parse markdown into (heading, content) sections, split on ## headers.

    Returns list of (heading, body_text) tuples.
    The first section (before any ##) uses the H1 heading or "Overview".
    """
    lines = text.split("\n")
    sections = []
    current_heading = "Overview"
    current_lines = []

    # Extract H1 as the document title
    h1 = ""
    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            h1 = line.lstrip("# ").strip()
            break

    for line in lines:
        if line.startswith("## "):
            # Save previous section
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_heading, body))
            current_heading = line.lstrip("# ").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_heading, body))

    return sections, h1


def _score_section(tokens: list[str], heading: str, body: str) -> float:
    """Score a section by how many query tokens appear in it.

    Heading matches count double. Returns 0-1 normalized score.
    """
    if not tokens:
        return 0.0

    heading_lower = heading.lower()
    body_lower = body.lower()
    total = 0.0

    for token in tokens:
        in_heading = token in heading_lower
        in_body = token in body_lower
        if in_heading:
            total += 2.0  # Heading match worth double
        elif in_body:
            total += 1.0

    # Normalize by max possible score (all tokens in heading)
    max_score = len(tokens) * 2.0
    return total / max_score if max_score > 0 else 0.0


def search_markdown_kb(
    query: str,
    source: str = "all",
    max_results: int = 5,
) -> list[dict]:
    """Search the markdown knowledge base for sections matching a query.

    Args:
        query: Search terms (condition name, symptom, drug, clinical question)
        source: Filter to specific KB source ("all" or one of SOURCE_DIRS keys)
        max_results: Maximum number of sections to return

    Returns:
        List of dicts with: heading, content, source_file, source_label,
        parent_heading, score
    """
    tokens = _tokenize_query(query)
    if not tokens:
        return []

    # Determine which directories to search
    if source == "all" or source not in SOURCE_DIRS:
        dirs_to_search = list(SOURCE_DIRS.items())
    else:
        dirs_to_search = [(source, SOURCE_DIRS[source])]

    results: list[KBSection] = []

    for source_key, subdir in dirs_to_search:
        dir_path = os.path.join(KB_ROOT, subdir)
        if not os.path.isdir(dir_path):
            continue

        label = SOURCE_LABELS.get(source_key, subdir)

        for filename in os.listdir(dir_path):
            if not filename.endswith(".md") or filename.startswith("_"):
                continue

            filepath = os.path.join(dir_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            except (IOError, UnicodeDecodeError):
                continue

            sections, h1 = _parse_sections(text)
            parent = h1 or filename.replace(".md", "").replace("-", " ").title()

            for heading, body in sections:
                score = _score_section(tokens, heading, body)
                if score < 0.2:  # Minimum relevance threshold
                    continue

                results.append(KBSection(
                    heading=heading,
                    content=body[:2000],  # Cap section length
                    source_file=f"{subdir}/{filename}",
                    source_label=label,
                    parent_heading=parent,
                    score=score,
                ))

    # Sort by score descending, take top N
    results.sort(key=lambda r: r.score, reverse=True)
    top = results[:max_results]

    return [
        {
            "heading": r.heading,
            "content": r.content,
            "source_file": r.source_file,
            "source_label": r.source_label,
            "parent_heading": r.parent_heading,
            "relevance_score": round(r.score, 3),
        }
        for r in top
    ]
