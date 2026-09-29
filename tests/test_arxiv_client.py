from __future__ import annotations

import time

from papercompass.arxiv_client import ArxivClient, _parse_feed, normalize_arxiv_id


def test_normalize_arxiv_id_plain():
    assert normalize_arxiv_id("1706.03762") == "1706.03762"


def test_normalize_arxiv_id_with_version():
    assert normalize_arxiv_id("1706.03762v5") == "1706.03762"


def test_normalize_arxiv_id_from_url():
    assert normalize_arxiv_id("https://arxiv.org/abs/2005.14165v4") == "2005.14165"


def test_normalize_arxiv_id_old_style():
    assert normalize_arxiv_id("math/0211159") == "math/0211159"


def test_normalize_arxiv_id_no_match():
    assert normalize_arxiv_id("not an id") is None


def test_parse_feed_extracts_entries(sample_feed_xml):
    entries = _parse_feed(sample_feed_xml)
    assert len(entries) == 2
    assert entries[0].arxiv_id == "1706.03762"
    assert entries[0].title == "Attention Is All You Need"
    assert "Ashish Vaswani" in entries[0].authors
    assert "cs.CL" in entries[0].categories
    assert entries[1].arxiv_id == "2005.14165"


class FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return FakeResponse(self.text)


def test_client_caches_and_rate_limits(tmp_path, sample_feed_xml):
    session = FakeSession(sample_feed_xml)
    client = ArxivClient(tmp_path / "cache", min_interval=0.05, session=session)
    entries1 = client.query_category("cs.LG", max_results=2)
    entries2 = client.query_category("cs.LG", max_results=2)
    assert len(entries1) == 2
    assert entries1[0].arxiv_id == entries2[0].arxiv_id
    # Second call should be served from cache, not hit the session again.
    assert session.calls == 1


def test_client_throttles_between_uncached_requests(tmp_path, sample_feed_xml):
    session = FakeSession(sample_feed_xml)
    client = ArxivClient(tmp_path / "cache", min_interval=0.2, session=session)
    client.query_category("cs.LG", max_results=1, start=0)
    t0 = time.monotonic()
    client.query_category("cs.CL", max_results=1, start=0)  # different params, not cached
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.15  # allow a little scheduling slack


def test_fetch_by_ids(tmp_path, sample_feed_xml):
    session = FakeSession(sample_feed_xml)
    client = ArxivClient(tmp_path / "cache", min_interval=0.01, session=session)
    entries = client.fetch_by_ids(["1706.03762", "2005.14165"])
    assert {e.arxiv_id for e in entries} == {"1706.03762", "2005.14165"}
