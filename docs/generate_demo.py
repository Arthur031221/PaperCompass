#!/usr/bin/env python3
"""Generate docs/data.json: precomputed recommendations for GitHub Pages.

Uses the full 50-paper curated reading list (eval/reading_list.json) as the
seed library, ranks it against a real ingested arXiv corpus, and writes the
top results plus their "why recommended" explanations to a static JSON file
that docs/index.html fetches. No server involved, this is what makes the
GitHub Pages demo work without a backend.

Run with: uv run python docs/generate_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from papercompass.arxiv_client import ArxivClient  # noqa: E402
from papercompass.db import Database  # noqa: E402
from papercompass.embeddings import SentenceTransformerEmbedder  # noqa: E402
from papercompass.ingest import DEFAULT_CATEGORIES, add_seeds, ingest_categories  # noqa: E402
from papercompass.ranking import rank_by_seeds  # noqa: E402

TOP_N = 30


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    reading_list = json.loads((root / "eval" / "reading_list.json").read_text())
    seed_ids = [p["arxiv_id"] for p in reading_list["papers"]]

    data_dir = root / "eval" / ".bench-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(data_dir / "papercompass.db")
    client = ArxivClient(data_dir / "arxiv_cache")
    embedder = SentenceTransformerEmbedder()

    t0 = time.monotonic()
    ingest_categories(
        db, client, embedder, categories=DEFAULT_CATEGORIES, days=90, max_per_category=150
    )
    unresolved = add_seeds(db, client, embedder, seed_ids, source="demo")
    if unresolved:
        print(f"warning: could not resolve {unresolved}", file=sys.stderr)

    seeds = db.seed_papers()
    candidates = db.all_papers()
    ranked = rank_by_seeds(candidates, seeds, why_k=3)[:TOP_N]
    elapsed = time.monotonic() - t0

    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "seed_count": len(seeds),
        "candidate_pool_size": len(candidates),
        "generation_seconds": round(elapsed, 1),
        "recommendations": [
            {
                **r.paper.to_dict(),
                "score": round(r.score, 4),
                "why": [
                    {"arxiv_id": w.arxiv_id, "title": w.title, "similarity": round(w.similarity, 4)}
                    for w in r.why
                ],
            }
            for r in ranked
        ],
    }

    out_path = root / "docs" / "data.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out['recommendations'])} recommendations to {out_path}")
    print(f"seed count {len(seeds)}, candidate pool {len(candidates)}, {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
