# Contributing

## Setup

```
uv sync --group dev
```

Python 3.12, managed by uv. The first test run or `papercompass serve` call
downloads the sentence-transformers all-MiniLM-L6-v2 model (about 87 MB) to
the standard Hugging Face cache, `~/.cache/huggingface`.

## Running tests

```
uv run pytest -q
```

Tests mock the embedding model (see `tests/conftest.py`), so the suite runs
in under a second and does not need network access or the real model.

## Lint

```
uv run ruff check .
```

## Project layout

- `src/papercompass/db.py`: SQLite storage.
- `src/papercompass/arxiv_client.py`: arXiv API client, rate limiting, caching.
- `src/papercompass/embeddings.py`: the embedding backend, real and mockable.
- `src/papercompass/importers.py`: BibTeX, Zotero CSV, and plain id list parsing.
- `src/papercompass/ranking.py`: similarity ranking and "why recommended".
- `src/papercompass/ingest.py`: ties the above together for ingest and seed import.
- `src/papercompass/server.py`: FastAPI app and API routes.
- `src/papercompass/static/index.html`: the single-page frontend.
- `src/papercompass/cli.py`: `papercompass serve`, `ingest`, `import`.
- `eval/`: the held-out recall benchmark.
- `docs/`: the static GitHub Pages demo.

## Making a change

1. Open an issue first for anything beyond a small fix, so we agree on the
   approach before you write code.
2. Keep pull requests focused. Add a test for behavior changes.
3. Run tests and lint before opening a pull request.
4. Follow the existing code style: short functions, docstrings on public
   functions and classes, no unnecessary abstraction.

## Reporting bugs

Use the bug report issue template. Include your Python version, OS, and the
command that failed with its full output.
