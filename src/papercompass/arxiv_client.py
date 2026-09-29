"""Client for the arXiv API: polite rate limiting, on-disk caching, Atom parsing."""

from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# arXiv asks that automated clients wait at least 3 seconds between requests.
MIN_REQUEST_INTERVAL = 3.0

# New-style (2007+, YYMM.NNNNN[N]) and old-style (archive/YYMMNNN) arXiv ids.
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")


@dataclass
class ArxivEntry:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    url: str


def normalize_arxiv_id(raw: str) -> str | None:
    """Extract a bare arXiv id (no version suffix) from an id, URL, or DOI-like string."""
    raw = raw.strip()
    match = ARXIV_ID_RE.search(raw)
    if not match:
        return None
    ident = match.group(1)
    ident = re.sub(r"v\d+$", "", ident)
    return ident


class ArxivClient:
    """Fetches and caches arXiv metadata, respecting the API's rate limit guidance."""

    def __init__(
        self,
        cache_dir: str | Path,
        min_interval: float = MIN_REQUEST_INTERVAL,
        cache_ttl: float = 6 * 3600,
        session: requests.Session | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.cache_ttl = cache_ttl
        self.session = session or requests.Session()
        self._last_request = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _cache_path(self, params: dict) -> Path:
        key = hashlib.sha256(urlencode(sorted(params.items())).encode()).hexdigest()
        return self.cache_dir / f"{key}.xml"

    def _fetch_raw(self, params: dict) -> str:
        cache_path = self._cache_path(params)
        if cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self.cache_ttl:
                return cache_path.read_text()
        self._throttle()
        resp = self.session.get(ARXIV_API, params=params, timeout=30)
        resp.raise_for_status()
        cache_path.write_text(resp.text)
        return resp.text

    def query_category(
        self, category: str, max_results: int = 100, start: int = 0
    ) -> list[ArxivEntry]:
        """Fetch recent papers in a category, newest first."""
        params = {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        raw = self._fetch_raw(params)
        return _parse_feed(raw)

    def fetch_by_ids(self, arxiv_ids: list[str]) -> list[ArxivEntry]:
        """Fetch metadata for a specific list of arXiv ids (batched, cached)."""
        entries: list[ArxivEntry] = []
        batch_size = 50
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i : i + batch_size]
            params = {"id_list": ",".join(batch), "max_results": len(batch)}
            raw = self._fetch_raw(params)
            entries.extend(_parse_feed(raw))
        return entries


def _parse_feed(raw_xml: str) -> list[ArxivEntry]:
    root = ET.fromstring(raw_xml)
    entries = []
    for entry_el in root.findall(f"{ATOM_NS}entry"):
        id_url = _text(entry_el, f"{ATOM_NS}id")
        arxiv_id = normalize_arxiv_id(id_url or "")
        if not arxiv_id:
            continue
        title = " ".join((_text(entry_el, f"{ATOM_NS}title") or "").split())
        abstract = " ".join((_text(entry_el, f"{ATOM_NS}summary") or "").split())
        authors = [
            (_text(a, f"{ATOM_NS}name") or "").strip()
            for a in entry_el.findall(f"{ATOM_NS}author")
        ]
        categories = [
            c.get("term", "") for c in entry_el.findall(f"{ATOM_NS}category") if c.get("term")
        ]
        published = _text(entry_el, f"{ATOM_NS}published") or ""
        updated = _text(entry_el, f"{ATOM_NS}updated") or ""
        entries.append(
            ArxivEntry(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=[a for a in authors if a],
                categories=categories,
                published=published,
                updated=updated,
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        )
    return entries


def _text(el: ET.Element, tag: str) -> str | None:
    found = el.find(tag)
    return found.text if found is not None else None
