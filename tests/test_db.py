from __future__ import annotations

import numpy as np

from papercompass.db import Database, Paper


def make_paper(arxiv_id, embedding=None):
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Title {arxiv_id}",
        abstract="An abstract.",
        authors=["A. Author"],
        categories=["cs.LG"],
        published="2024-01-01T00:00:00Z",
        updated="2024-01-01T00:00:00Z",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        embedding=embedding,
        fetched_at="2024-01-02T00:00:00Z",
    )


def test_upsert_and_get_paper(tmp_path):
    db = Database(tmp_path / "test.db")
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    db.upsert_paper(make_paper("1234.5678", embedding=vec))
    fetched = db.get_paper("1234.5678")
    assert fetched is not None
    assert fetched.title == "Title 1234.5678"
    assert np.allclose(fetched.embedding, vec)


def test_upsert_preserves_embedding_when_not_provided(tmp_path):
    db = Database(tmp_path / "test.db")
    vec = np.array([1.0, 2.0], dtype=np.float32)
    db.upsert_paper(make_paper("1111.2222", embedding=vec))
    # Re-upsert without an embedding (e.g. a metadata refresh) should keep it.
    db.upsert_paper(make_paper("1111.2222", embedding=None))
    fetched = db.get_paper("1111.2222")
    assert np.allclose(fetched.embedding, vec)


def test_get_missing_paper_returns_none(tmp_path):
    db = Database(tmp_path / "test.db")
    assert db.get_paper("0000.0000") is None


def test_all_papers_only_with_embedding_by_default(tmp_path):
    db = Database(tmp_path / "test.db")
    db.upsert_paper(make_paper("1111.1111", embedding=np.array([1.0], dtype=np.float32)))
    db.upsert_paper(make_paper("2222.2222", embedding=None))
    assert len(db.all_papers()) == 1
    assert len(db.all_papers(only_with_embedding=False)) == 2


def test_seed_lifecycle(tmp_path):
    db = Database(tmp_path / "test.db")
    db.upsert_paper(make_paper("1111.1111", embedding=np.array([1.0], dtype=np.float32)))
    db.add_seed("1111.1111", source="list", added_at="2024-01-01T00:00:00Z")
    assert db.seed_ids() == ["1111.1111"]
    assert len(db.seed_papers()) == 1
    db.remove_seed("1111.1111")
    assert db.seed_ids() == []


def test_search_text_matches_title_and_abstract(tmp_path):
    db = Database(tmp_path / "test.db")
    p = make_paper("1111.1111")
    p.title = "Contrastive Learning for Retrieval"
    db.upsert_paper(p)
    results = db.search_text("contrastive")
    assert len(results) == 1
    assert db.search_text("nonexistent-term") == []


def test_count_papers(tmp_path):
    db = Database(tmp_path / "test.db")
    assert db.count_papers() == 0
    db.upsert_paper(make_paper("1111.1111"))
    assert db.count_papers() == 1
