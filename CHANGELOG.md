# Changelog

All notable changes to this project are documented in this file.

## 0.1.0, 2026-09-30

First release.

- arXiv API ingest for a configurable set of categories over the last N days, with
  polite rate limiting (3 seconds between requests) and on-disk response caching.
- Seed library import from a plain list of arXiv ids, a BibTeX file, or a Zotero
  CSV export.
- Ranking of the corpus by cosine similarity to the seed library using
  sentence-transformers all-MiniLM-L6-v2 on CPU, with a "why recommended"
  explanation naming the 3 nearest seed papers for every recommendation.
- Semantic search over the corpus.
- FastAPI backend, single static HTML page frontend, SQLite storage.
- CLI: `papercompass serve`, `papercompass ingest`, `papercompass import`.
- Held-out recall benchmark in `eval/` with real numbers measured on Apple
  Silicon CPU.
- Static precomputed demo published to GitHub Pages from `docs/`.
