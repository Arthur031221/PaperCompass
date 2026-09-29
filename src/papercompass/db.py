"""SQLite storage for papers, embeddings and seed library entries."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT NOT NULL,
    categories TEXT NOT NULL,
    published TEXT NOT NULL,
    updated TEXT NOT NULL,
    url TEXT NOT NULL,
    embedding BLOB,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS seeds (
    arxiv_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    added_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);
"""


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    published: str = ""
    updated: str = ""
    url: str = ""
    embedding: np.ndarray | None = None
    fetched_at: str = ""

    def to_dict(self, include_embedding: bool = False) -> dict:
        d = {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published,
            "updated": self.updated,
            "url": self.url,
        }
        if include_embedding and self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d


def _row_to_paper(row: sqlite3.Row) -> Paper:
    embedding = None
    if row["embedding"] is not None:
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
    return Paper(
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        abstract=row["abstract"],
        authors=json.loads(row["authors"]),
        categories=json.loads(row["categories"]),
        published=row["published"],
        updated=row["updated"],
        url=row["url"],
        embedding=embedding,
        fetched_at=row["fetched_at"],
    )


class Database:
    """Thin wrapper around a SQLite file storing papers and seed entries."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # papercompass is a single-user local tool: FastAPI's sync endpoints run
        # in a worker thread pool, so the connection must be usable across
        # threads. There is no concurrent write pattern that needs locking
        # beyond what SQLite already serializes internally.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def cursor(self):
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        finally:
            cur.close()

    def upsert_paper(self, paper: Paper) -> None:
        embedding_bytes = None
        if paper.embedding is not None:
            embedding_bytes = np.asarray(paper.embedding, dtype=np.float32).tobytes()
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO papers
                    (arxiv_id, title, abstract, authors, categories, published,
                     updated, url, embedding, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    title=excluded.title,
                    abstract=excluded.abstract,
                    authors=excluded.authors,
                    categories=excluded.categories,
                    published=excluded.published,
                    updated=excluded.updated,
                    url=excluded.url,
                    embedding=COALESCE(excluded.embedding, papers.embedding),
                    fetched_at=excluded.fetched_at
                """,
                (
                    paper.arxiv_id,
                    paper.title,
                    paper.abstract,
                    json.dumps(paper.authors),
                    json.dumps(paper.categories),
                    paper.published,
                    paper.updated,
                    paper.url,
                    embedding_bytes,
                    paper.fetched_at,
                ),
            )

    def get_paper(self, arxiv_id: str) -> Paper | None:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,))
            row = cur.fetchone()
        return _row_to_paper(row) if row else None

    def all_papers(self, only_with_embedding: bool = True) -> list[Paper]:
        with self.cursor() as cur:
            if only_with_embedding:
                cur.execute("SELECT * FROM papers WHERE embedding IS NOT NULL")
            else:
                cur.execute("SELECT * FROM papers")
            rows = cur.fetchall()
        return [_row_to_paper(r) for r in rows]

    def count_papers(self) -> int:
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM papers")
            return cur.fetchone()["c"]

    def add_seed(self, arxiv_id: str, source: str, added_at: str) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO seeds (arxiv_id, source, added_at)
                VALUES (?, ?, ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET source=excluded.source
                """,
                (arxiv_id, source, added_at),
            )

    def remove_seed(self, arxiv_id: str) -> None:
        with self.cursor() as cur:
            cur.execute("DELETE FROM seeds WHERE arxiv_id = ?", (arxiv_id,))

    def seed_ids(self) -> list[str]:
        with self.cursor() as cur:
            cur.execute("SELECT arxiv_id FROM seeds")
            return [r["arxiv_id"] for r in cur.fetchall()]

    def seed_papers(self) -> list[Paper]:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT papers.* FROM papers
                JOIN seeds ON seeds.arxiv_id = papers.arxiv_id
                """
            )
            rows = cur.fetchall()
        return [_row_to_paper(r) for r in rows]

    def search_text(self, query: str, limit: int = 50) -> list[Paper]:
        like = f"%{query}%"
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM papers
                WHERE title LIKE ? OR abstract LIKE ?
                ORDER BY published DESC
                LIMIT ?
                """,
                (like, like, limit),
            )
            rows = cur.fetchall()
        return [_row_to_paper(r) for r in rows]
