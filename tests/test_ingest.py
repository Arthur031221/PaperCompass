from __future__ import annotations

from datetime import datetime, timedelta, timezone

from papercompass.arxiv_client import ArxivEntry
from papercompass.db import Database
from papercompass.ingest import add_seeds, ingest_categories


class FakeClient:
    def __init__(self, entries_by_category=None, entries_by_id=None):
        self.entries_by_category = entries_by_category or {}
        self.entries_by_id = entries_by_id or {}
        self.category_calls = []
        self.id_calls = []

    def query_category(self, category, max_results=100, start=0):
        self.category_calls.append((category, start))
        if start > 0:
            return []
        return self.entries_by_category.get(category, [])

    def fetch_by_ids(self, arxiv_ids):
        self.id_calls.append(list(arxiv_ids))
        return [self.entries_by_id[i] for i in arxiv_ids if i in self.entries_by_id]


def make_entry(arxiv_id, days_ago=1, title=None):
    published = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return ArxivEntry(
        arxiv_id=arxiv_id,
        title=title or f"Paper {arxiv_id}",
        abstract="An abstract about machine learning.",
        authors=["A. Author"],
        categories=["cs.LG"],
        published=published,
        updated=published,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def test_ingest_categories_embeds_new_papers(tmp_path, fake_embedder):
    db = Database(tmp_path / "test.db")
    entries = [make_entry("1111.1111"), make_entry("2222.2222")]
    client = FakeClient(entries_by_category={"cs.LG": entries})
    new_count = ingest_categories(db, client, fake_embedder, categories=["cs.LG"], days=90)
    assert new_count == 2
    assert db.count_papers() == 2
    assert fake_embedder.calls == 1  # batched into a single encode call


def test_ingest_categories_skips_existing_papers(tmp_path, fake_embedder):
    db = Database(tmp_path / "test.db")
    entries = [make_entry("1111.1111")]
    client = FakeClient(entries_by_category={"cs.LG": entries})
    ingest_categories(db, client, fake_embedder, categories=["cs.LG"], days=90)
    new_count = ingest_categories(db, client, fake_embedder, categories=["cs.LG"], days=90)
    assert new_count == 0


def test_ingest_categories_respects_cutoff(tmp_path, fake_embedder):
    db = Database(tmp_path / "test.db")
    entries = [make_entry("1111.1111", days_ago=1), make_entry("9999.9999", days_ago=200)]
    client = FakeClient(entries_by_category={"cs.LG": entries})
    new_count = ingest_categories(db, client, fake_embedder, categories=["cs.LG"], days=90)
    assert new_count == 1
    assert db.get_paper("1111.1111") is not None
    assert db.get_paper("9999.9999") is None


def test_add_seeds_fetches_missing_and_marks_seed(tmp_path, fake_embedder):
    db = Database(tmp_path / "test.db")
    entry = make_entry("1111.1111")
    client = FakeClient(entries_by_id={"1111.1111": entry})
    unresolved = add_seeds(db, client, fake_embedder, ["1111.1111"], source="list")
    assert unresolved == []
    assert db.seed_ids() == ["1111.1111"]
    assert db.get_paper("1111.1111").embedding is not None


def test_add_seeds_reports_unresolved_ids(tmp_path, fake_embedder):
    db = Database(tmp_path / "test.db")
    client = FakeClient(entries_by_id={})
    unresolved = add_seeds(db, client, fake_embedder, ["0000.0000"], source="list")
    assert unresolved == ["0000.0000"]
    assert db.seed_ids() == []


def test_add_seeds_reuses_existing_paper_without_refetch(tmp_path, fake_embedder):
    db = Database(tmp_path / "test.db")
    entries = [make_entry("1111.1111")]
    client = FakeClient(entries_by_category={"cs.LG": entries})
    ingest_categories(db, client, fake_embedder, categories=["cs.LG"], days=90)
    calls_before = len(client.id_calls)
    unresolved = add_seeds(db, client, fake_embedder, ["1111.1111"], source="list")
    assert unresolved == []
    assert len(client.id_calls) == calls_before  # no id fetch needed, already cached
    assert db.seed_ids() == ["1111.1111"]
