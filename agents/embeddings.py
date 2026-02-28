"""
Voyage AI embedding wrapper.
Uses voyage-3-lite (1024 dims) — Anthropic's recommended embedding partner.

Usage:
    from agents.embeddings import get_embedding, get_embeddings_batch

    # Single query
    vec = await get_embedding("sore throat and fever")

    # Batch (up to 128 texts per call)
    vecs = await get_embeddings_batch(["text1", "text2", ...])

Graceful degradation: returns None if VOYAGE_API_KEY not set.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_client = None
MODEL = "voyage-3-lite"
DIMENSIONS = 512


def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        return None
    import voyageai
    _client = voyageai.Client(api_key=api_key)
    return _client


async def get_embedding(text: str) -> Optional[list[float]]:
    """Embed a single text. Returns None if Voyage AI unavailable.
    Runs the blocking API call in a thread executor so it doesn't block
    the event loop and can overlap with concurrent DB queries."""
    import asyncio
    client = _get_client()
    if not client:
        return None
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: client.embed([text], model=MODEL, input_type="query")
        )
        return result.embeddings[0]
    except Exception as e:
        logger.warning(f"Voyage AI embedding failed: {e}")
        return None


async def get_embeddings_batch(texts: list[str], input_type: str = "document") -> Optional[list[list[float]]]:
    """Embed a batch of texts. Returns None if Voyage AI unavailable.
    Retries on rate limit errors with exponential backoff."""
    import time
    client = _get_client()
    if not client:
        return None
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = client.embed(texts, model=MODEL, input_type=input_type)
            return result.embeddings
        except Exception as e:
            err_str = str(e)
            if 'rate limit' in err_str.lower() or 'reduced rate' in err_str.lower() or '429' in err_str:
                wait = 25 * (2 ** attempt)  # 25, 50, 100, 200, 400s
                logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.warning(f"Voyage AI batch embedding failed: {e}")
                return None
    logger.error(f"Voyage AI rate limit exceeded after {max_retries} retries")
    return None
