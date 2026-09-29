# papercompass

Recommendations over your own paper library, not another daily digest bot.

On a held-out test built from a curated 50-paper machine learning reading list, papercompass recovers 90% of the withheld papers (9 of 10) in its top 50 recommendations, ranked against a pool of 673 real arXiv papers.[^1]

[![CI](https://github.com/Arthur031221/papercompass/actions/workflows/ci.yml/badge.svg)](https://github.com/Arthur031221/papercompass/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![version](https://img.shields.io/badge/version-0.1.0-informational)

![papercompass CLI demo](demo/demo.gif)

## Why

karpathy/arxiv-sanity-lite hit 1,697 stars because it solved a real problem: a personal arXiv feed ranked by what you actually read, not by what is trending today. It has had no commits since 2023-06-19. Everything that replaced it is a digest bot: it emails you a daily list filtered by keyword or category, the same list for everyone who picks the same keywords. None of them rank the corpus against a library you curated yourself. Reading a paper and getting nothing pointed at it in return is the original arxiv-sanity complaint, and it is still unsolved five years later.

## Install

```
uvx papercompass serve
```

Before the PyPI release is up, run it straight from the repository:

```
uvx --from git+https://github.com/Arthur031221/papercompass papercompass serve
```

Or clone and run with uv:

```
git clone https://github.com/Arthur031221/papercompass
cd papercompass
uv run papercompass serve
```

Needs Python 3.12. The first run downloads sentence-transformers all-MiniLM-L6-v2 (about 87 MB) to `~/.cache/huggingface`. Everything after that runs offline except the arXiv API calls you trigger with `ingest`.

## Quick start

```
papercompass serve
```

Open `http://127.0.0.1:8000`. In the UI:

1. Click "Ingest recent papers" once, to pull the last 90 days of `cs.LG`, `cs.CL`, `cs.AI`, `cs.CV`, and `stat.ML` into the corpus. Takes a minute or two the first time, arXiv's API is rate limited to one request per 3 seconds and this is a few hundred papers.
2. Paste 10 to 20 arXiv ids you have already read into "Plain list of arXiv ids" and import them, or upload a BibTeX file or a Zotero CSV export.
3. Click "Refresh feed". Every recommendation shows the 3 seed papers it is closest to, so you can tell at a glance why it showed up.

Or from the command line, no browser needed:

```
papercompass ingest --categories cs.LG,cs.CL --days 90
papercompass import my_library.bib
```

## How it works

- arXiv's API (`export.arxiv.org/api/query`) is the only data source, ingest fetches Atom feeds per category, paginated, capped at 90 days by default, and caches every response to disk so re-running ingest does not refetch what it already has.
- Every paper's title and abstract are embedded with `sentence-transformers/all-MiniLM-L6-v2` on CPU, 384-dimensional, normalized. No GPU, no API key, no per-paper network call beyond the arXiv fetch itself.
- A recommendation score is the mean cosine similarity between a candidate paper and its 3 nearest seed papers. Those 3 papers are also the "why recommended" explanation, the score is never a black box number.
- Seed import parses arXiv ids out of a BibTeX file (every field of every entry, since exporters disagree on where the id goes), a Zotero CSV export (every column, Zotero usually puts it in Url, DOI, or Extra), or a plain list, one id or URL per line.
- Storage is one SQLite file. Embeddings are stored as raw float32 blobs, there is no vector database, because a personal library, even ingesting 90 days of 5 categories, is a few thousand rows, not a few million.
- Search embeds your query with the same model and ranks the corpus by cosine similarity to it, this is semantic search, not keyword matching.

## Comparison

| | papercompass | arxiv-sanity-lite | arXiv digest bots | Semantic Scholar recommendations |
|---|---|---|---|---|
| Ranks against your own library | Yes, seed papers you choose | Yes, but unmaintained | No, keyword or category filter only | Yes, but needs a Semantic Scholar account and your library lives on their server |
| Explains each recommendation | Yes, 3 nearest seed papers with scores | Similarity search only, no explanation | No | Partial, "based on your library" with no specific papers named |
| Runs offline after setup | Yes, embedding is local, only ingest needs network | Yes, but the project is dead | No, delivery is email or Slack, most require a hosted keyword config | No, cloud only |
| Import from BibTeX or Zotero | Yes | No, manual library only | No | Zotero sync exists, BibTeX does not |
| Last commit | 2026-09-30 | 2023-06-19, 1,697 stars, unmaintained | Typically under 20 stars, most abandoned within months | Actively maintained, closed source |
| Self-hosted | Yes, SQLite plus a local process | Yes | Varies | No |

papercompass does not do anything arxiv-sanity-lite could not do when it was maintained. The difference is it is maintained, it adds an explanation for every recommendation instead of a bare similarity score, and it imports an existing Zotero or BibTeX library instead of requiring you to rebuild one paper at a time.

## Command reference

```
papercompass serve [--host HOST] [--port PORT] [--data-dir DIR]
papercompass ingest [--categories cs.LG,cs.CL,...] [--days N] [--max-per-category N] [--data-dir DIR] [--json]
papercompass import FILE [--data-dir DIR] [--json]
```

`--data-dir` defaults to `~/.papercompass` and holds the SQLite database and the arXiv response cache. `import` accepts a `.bib` file, a `.csv` file (Zotero export), or a plain text file with one arXiv id per line.

API routes, if you want to script against a running server: `GET /api/stats`, `GET /api/seeds`, `POST /api/seeds/import-text`, `POST /api/seeds/import-file`, `DELETE /api/seeds/{arxiv_id}`, `POST /api/ingest`, `GET /api/feed`, `GET /api/search?q=`.

## Limits and FAQ

**Does it read PDFs?** No. It uses the title and abstract from the arXiv API. Full text would improve ranking but arXiv does not serve it through the API and downloading PDFs for a whole corpus is a different, heavier project.

**Does it rank by citation count or venue?** No. Ranking is pure text similarity to your seed library. A landmark paper with an abstract that reads differently from your seeds will rank low, see the Adam optimizer paper in `eval/RESULTS.md` for a real example of this failing.

**Multi-user or cloud hosted?** No, it is one SQLite file and one local process, built for one person's reading list.

**Does ingest run forever?** No, `--days` bounds it to a window (default 90) and `--max-per-category` caps it further. It stops as soon as it hits a paper older than the window.

**What if arXiv is unreachable?** Ingest and seed resolution fail with an error for the ids that could not be fetched, everything already cached keeps working, `/api/feed` and `/api/search` never touch the network.

## Benchmark

Full method and raw numbers in [`eval/RESULTS.md`](eval/RESULTS.md), reproduce with `uv run python eval/benchmark.py`.

## Demo

A static, precomputed version of the ranked feed is published on GitHub Pages: [arthur031221.github.io/papercompass](https://arthur031221.github.io/papercompass/). It shows papercompass's recommendations when the seed library is the full 50-paper curated reading list in `eval/reading_list.json`, ranked against a real corpus of recent arXiv papers. Regenerate it with `uv run python docs/generate_demo.py`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and feature requests use the issue templates.

## License

MIT, see [LICENSE](LICENSE).

[^1]: Recall@50 on a stratified 80/20 split (40 seed papers, 10 held out) of the 50-paper reading list in `eval/reading_list.json`, ranked against a candidate pool of 673 papers (the 10 held-out papers plus a distractor corpus ingested the same way `papercompass ingest` does: categories cs.LG, cs.CL, cs.CV, stat.ML, cs.AI, last 90 days). Model: sentence-transformers/all-MiniLM-L6-v2, CPU. Measured on a MacBook Air, Apple Silicon, 2026-09-30. Consistent at 90% across 4 different random splits. Full method in `eval/RESULTS.md`.
