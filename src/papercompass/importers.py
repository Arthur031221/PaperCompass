"""Parsers that turn a BibTeX file, a Zotero CSV export, or a plain list of
arXiv ids into a deduplicated, ordered list of arXiv ids to seed the library.
"""

from __future__ import annotations

import csv
import io

from papercompass.arxiv_client import ARXIV_ID_RE, normalize_arxiv_id


def _dedupe(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


def _find_ids_in_text(text: str) -> list[str]:
    return [normalize_arxiv_id(m) for m in ARXIV_ID_RE.findall(text)]


def parse_bibtex(text: str) -> list[str]:
    """Extract arXiv ids from a .bib file.

    Looks at every field of every entry (eprint, url, note, journal, ...)
    since different reference managers export the arXiv id differently.
    """
    ids: list[str] = []
    # Split on entry starts so ids from one entry's fields are still found
    # even if the file has no blank lines between entries.
    for entry in text.split("@")[1:]:
        found = _find_ids_in_text(entry)
        ids.extend(f for f in found if f)
    return _dedupe(ids)


def parse_zotero_csv(text: str) -> list[str]:
    """Extract arXiv ids from a Zotero CSV export.

    Zotero puts the arXiv id in different columns depending on how the item
    was saved: Url, DOI, Extra ("arXiv:1706.03762"), or occasionally the
    title. Scan every column of every row.
    """
    ids: list[str] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        for value in row.values():
            if not value:
                continue
            found = _find_ids_in_text(value)
            ids.extend(f for f in found if f)
    return _dedupe(ids)


def parse_id_list(text: str) -> list[str]:
    """Extract arXiv ids from a plain list, one id (or URL) per line, or
    comma/whitespace separated.
    """
    ids = _find_ids_in_text(text)
    return _dedupe([f for f in ids if f])


def detect_and_parse(filename: str, text: str) -> list[str]:
    """Pick a parser based on file extension, falling back to plain-list."""
    lower = filename.lower()
    if lower.endswith(".bib") or lower.endswith(".bibtex"):
        return parse_bibtex(text)
    if lower.endswith(".csv"):
        return parse_zotero_csv(text)
    return parse_id_list(text)
