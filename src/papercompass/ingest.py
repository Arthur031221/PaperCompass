"""Ties the arXiv client, embedder and database together.

Two entry points:
  ingest_categories: pull recent papers for a set of arXiv categories into
    the corpus, embedding each one.
  add_seeds: resolve a list of arXiv ids into full paper records (fetching
    and embedding any not already cached) and mark them as seed papers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from papercompass.arxiv_client import ArxivClient, ArxivEntry
from papercompass.db import Database, Paper
from papercompass.embeddings import Embedder

logger = logging.getLogger("papercompass.ingest")

DEFAULT_CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "cs.CV", "stat.ML"]
PAGE_SIZE = 100


def _entry_to_paper(entry: ArxivEntry) -> Paper:
    return Paper(
        arxiv_id=entry.arxiv_id,
        title=entry.title,
        abstract=entry.abstract,
        authors=entry.authors,
        categories=entry.categories,
        published=entry.published,
        updated=entry.updated,
        url=entry.url,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def ingest_categories(
    db: Database,
    client: ArxivClient,
    embedder: Embedder,
    categories: list[str] | None = None,
    days: int = 90,
    max_per_category: int = 300,
) -> int:
    """Fetch recent papers per category, embed new ones, store all of them.

    Returns the number of papers newly embedded (existing papers are
    skipped, since arXiv listings overlap heavily day to day).
    """
    categories = categories or DEFAULT_CATEGORIES
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    new_count = 0

    for category in categories:
        start = 0
        collected: list[ArxivEntry] = []
        while start < max_per_category and len(collected) < max_per_category:
            page_size = min(PAGE_SIZE, max_per_category - start)
            page = client.query_category(category, max_results=page_size, start=start)
            if not page:
                break
            stop = False
            for entry in page:
                published = _parse_date(entry.published)
                if published and published < cutoff:
                    stop = True
                    break
                collected.append(entry)
                if len(collected) >= max_per_category:
                    break
            if stop or len(page) < page_size:
                break
            start += page_size

        to_embed: list[ArxivEntry] = []
        for entry in collected:
            if db.get_paper(entry.arxiv_id) is None:
                to_embed.append(entry)

        if to_embed:
            texts = [f"{e.title}\n\n{e.abstract}" for e in to_embed]
            vectors = embedder.encode(texts)
            for entry, vector in zip(to_embed, vectors):
                paper = _entry_to_paper(entry)
                paper.embedding = vector
                db.upsert_paper(paper)
                new_count += 1
        logger.info(
            "ingested category=%s fetched=%d new=%d", category, len(collected), len(to_embed)
        )

    return new_count


def add_seeds(
    db: Database,
    client: ArxivClient,
    embedder: Embedder,
    arxiv_ids: list[str],
    source: str,
) -> list[str]:
    """Ensure each id is in the corpus with an embedding, mark it a seed.

    Returns the list of ids that could not be resolved.
    """
    missing_ids = [aid for aid in arxiv_ids if db.get_paper(aid) is None]
    if missing_ids:
        entries = client.fetch_by_ids(missing_ids)
        found_ids = {e.arxiv_id for e in entries}
        texts = [f"{e.title}\n\n{e.abstract}" for e in entries]
        vectors = embedder.encode(texts) if texts else []
        for entry, vector in zip(entries, vectors):
            paper = _entry_to_paper(entry)
            paper.embedding = vector
            db.upsert_paper(paper)
        unresolved = [aid for aid in missing_ids if aid not in found_ids]
    else:
        unresolved = []

    added_at = datetime.now(timezone.utc).isoformat()
    for aid in arxiv_ids:
        if aid in unresolved:
            continue
        existing = db.get_paper(aid)
        if existing and existing.embedding is None:
            vector = embedder.encode([f"{existing.title}\n\n{existing.abstract}"])[0]
            existing.embedding = vector
            db.upsert_paper(existing)
        db.add_seed(aid, source, added_at)

    return unresolved


def _parse_date(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
