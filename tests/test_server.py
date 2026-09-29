from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from papercompass.arxiv_client import ArxivEntry
from papercompass.server import create_app


class FakeClient:
    def __init__(self, entries_by_category=None, entries_by_id=None):
        self.entries_by_category = entries_by_category or {}
        self.entries_by_id = entries_by_id or {}

    def query_category(self, category, max_results=100, start=0):
        if start > 0:
            return []
        return self.entries_by_category.get(category, [])

    def fetch_by_ids(self, arxiv_ids):
        return [self.entries_by_id[i] for i in arxiv_ids if i in self.entries_by_id]


def make_entry(arxiv_id, title=None, days_ago=1):
    published = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return ArxivEntry(
        arxiv_id=arxiv_id,
        title=title or f"Paper about {arxiv_id}",
        abstract="An abstract about machine learning and neural networks.",
        authors=["A. Author"],
        categories=["cs.LG"],
        published=published,
        updated=published,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


@pytest.fixture()
def client_app(tmp_data_dir, fake_embedder):
    fake_arxiv = FakeClient(
        entries_by_id={
            "1706.03762": make_entry("1706.03762", "Attention Is All You Need"),
            "2005.14165": make_entry("2005.14165", "Language Models are Few-Shot Learners"),
        },
        entries_by_category={
            "cs.LG": [
                make_entry("1111.1111", "A Survey of Deep Learning"),
                make_entry("2222.2222", "Transformers for Vision"),
            ]
        },
    )
    app = create_app(tmp_data_dir, embedder=fake_embedder, client=fake_arxiv)
    return TestClient(app)


def test_stats_empty(client_app):
    resp = client_app.get("/api/stats")
    assert resp.status_code == 200
    assert resp.json() == {"papers": 0, "seeds": 0}


def test_import_text_list(client_app):
    resp = client_app.post(
        "/api/seeds/import-text",
        json={"text": "1706.03762\n2005.14165", "format": "list"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["added"]) == {"1706.03762", "2005.14165"}
    assert body["unresolved"] == []

    seeds = client_app.get("/api/seeds").json()
    assert len(seeds) == 2

    stats = client_app.get("/api/stats").json()
    assert stats["seeds"] == 2


def test_import_text_no_ids_is_400(client_app):
    resp = client_app.post(
        "/api/seeds/import-text", json={"text": "nothing here", "format": "list"}
    )
    assert resp.status_code == 400


def test_import_text_unresolved_ids(client_app):
    resp = client_app.post(
        "/api/seeds/import-text", json={"text": "9999.99999", "format": "list"}
    )
    assert resp.status_code == 200
    assert resp.json()["unresolved"] == ["9999.99999"]


def test_import_bibtex_file(client_app):
    bib = b"@article{x, title={T}, eprint={1706.03762}, archivePrefix={arXiv}}"
    resp = client_app.post(
        "/api/seeds/import-file",
        files={"file": ("library.bib", bib, "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.json()["added"] == ["1706.03762"]


def test_delete_seed(client_app):
    client_app.post("/api/seeds/import-text", json={"text": "1706.03762", "format": "list"})
    resp = client_app.delete("/api/seeds/1706.03762")
    assert resp.status_code == 200
    assert client_app.get("/api/seeds").json() == []


def test_ingest_endpoint(client_app):
    resp = client_app.post("/api/ingest", json={"categories": ["cs.LG"], "days": 90})
    assert resp.status_code == 200
    body = resp.json()
    assert body["new_papers"] == 2
    assert body["total_papers"] == 2


def test_feed_empty_without_seeds(client_app):
    client_app.post("/api/ingest", json={"categories": ["cs.LG"], "days": 90})
    resp = client_app.get("/api/feed")
    assert resp.status_code == 200
    assert resp.json()["recommendations"] == []


def test_feed_ranks_against_seeds(client_app):
    client_app.post("/api/ingest", json={"categories": ["cs.LG"], "days": 90})
    client_app.post("/api/seeds/import-text", json={"text": "1706.03762", "format": "list"})
    resp = client_app.get("/api/feed")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["recommendations"]) == 2
    rec = body["recommendations"][0]
    assert "score" in rec
    assert len(rec["why"]) == 1
    assert rec["why"][0]["arxiv_id"] == "1706.03762"
    # seeds themselves must not appear in the feed
    ids = [r["arxiv_id"] for r in body["recommendations"]]
    assert "1706.03762" not in ids


def test_search_returns_scored_results(client_app):
    client_app.post("/api/ingest", json={"categories": ["cs.LG"], "days": 90})
    resp = client_app.get("/api/search", params={"q": "deep learning"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert "score" in results[0]


def test_search_empty_query(client_app):
    resp = client_app.get("/api/search", params={"q": ""})
    assert resp.json()["results"] == []


def test_index_page_served(client_app):
    resp = client_app.get("/")
    assert resp.status_code == 200
    assert "papercompass" in resp.text
