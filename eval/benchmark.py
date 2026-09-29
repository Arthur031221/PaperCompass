#!/usr/bin/env python3
"""Held-out recall benchmark for papercompass.

Method
------
1. Take the curated 50-paper reading list in eval/reading_list.json, which
   spans 8 topic clusters (transformers/LLMs, CNN architectures, detection
   and segmentation, generative models, reinforcement learning,
   self-supervised and multimodal, foundational techniques, other).
2. Stratified 80/20 split by topic cluster: 40 papers become the seed
   library, 10 are held out.
3. Ingest a distractor corpus of recent arXiv papers (same categories
   papercompass ingests by default) so the held-out papers have to be
   found among hundreds of unrelated candidates, not just among
   themselves.
4. Rank the full candidate pool (distractors plus the 10 held-out papers,
   with the 40 seeds excluded) by similarity to the 40 seed papers using
   the same papercompass.ranking.rank_by_seeds function the server uses.
5. Recall@K = fraction of the 10 held-out papers that land in the top K
   ranked candidates.

This uses the real sentence-transformers model and real arXiv API calls
(cached to disk), so a run takes a few minutes and needs network access.
Run it with: uv run python eval/benchmark.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from papercompass.arxiv_client import ArxivClient  # noqa: E402
from papercompass.db import Database  # noqa: E402
from papercompass.embeddings import SentenceTransformerEmbedder  # noqa: E402
from papercompass.ingest import add_seeds, ingest_categories  # noqa: E402
from papercompass.ranking import rank_by_seeds  # noqa: E402

DEFAULT_CATEGORIES = ["cs.LG", "cs.CL", "cs.CV", "stat.ML", "cs.AI"]


def stratified_split(papers: list[dict], holdout_fraction: float, seed: int):
    rng = random.Random(seed)
    by_topic: dict[str, list[str]] = {}
    for p in papers:
        by_topic.setdefault(p["topic"], []).append(p["arxiv_id"])

    held_out: list[str] = []
    seeds: list[str] = []
    for topic, ids in sorted(by_topic.items()):
        ids = sorted(ids)  # deterministic order before shuffling
        rng.shuffle(ids)
        n_hold = max(1, round(len(ids) * holdout_fraction))
        held_out.extend(ids[:n_hold])
        seeds.extend(ids[n_hold:])
    return seeds, held_out


def run(args: argparse.Namespace) -> dict:
    reading_list = json.loads((Path(__file__).parent / "reading_list.json").read_text())
    papers = reading_list["papers"]
    seed_ids, holdout_ids = stratified_split(papers, args.holdout_fraction, args.split_seed)
    print(f"seed library: {len(seed_ids)} papers, held out: {len(holdout_ids)} papers")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(data_dir / "papercompass.db")
    client = ArxivClient(data_dir / "arxiv_cache", min_interval=args.min_interval)

    print("loading sentence-transformers/all-MiniLM-L6-v2 on CPU...")
    t_load0 = time.monotonic()
    embedder = SentenceTransformerEmbedder()
    embedder.encode(["warm up"])  # trigger the lazy model load, outside timed sections
    model_load_s = time.monotonic() - t_load0

    print(f"ingesting distractor corpus: {args.categories}, last {args.days} days, "
          f"up to {args.max_per_category} per category...")
    t_ingest0 = time.monotonic()
    ingest_categories(
        db, client, embedder,
        categories=args.categories, days=args.days, max_per_category=args.max_per_category,
    )
    ingest_s = time.monotonic() - t_ingest0
    corpus_size_before_reading_list = db.count_papers()
    print(f"distractor corpus: {corpus_size_before_reading_list} papers ({ingest_s:.1f}s)")

    print("resolving the 50 reading-list papers and marking 40 as seeds...")
    t_seed0 = time.monotonic()
    all_reading_ids = seed_ids + holdout_ids
    unresolved = add_seeds(db, client, embedder, all_reading_ids, source="eval-all")
    if unresolved:
        print(f"WARNING: could not resolve {unresolved}", file=sys.stderr)
    # add_seeds marks every id passed to it as a seed. Un-mark the held-out ones.
    for hid in holdout_ids:
        db.remove_seed(hid)
    seed_resolve_s = time.monotonic() - t_seed0

    seeds = db.seed_papers()
    assert {s.arxiv_id for s in seeds} == set(seed_ids) - set(unresolved), "seed set mismatch"

    print(f"ranking {db.count_papers()} candidates against {len(seeds)} seed papers...")
    t_rank0 = time.monotonic()
    candidates = db.all_papers()
    ranked = rank_by_seeds(candidates, seeds, why_k=3)
    rank_s = time.monotonic() - t_rank0

    top_k_ids = {r.paper.arxiv_id for r in ranked[: args.top_k]}
    resolved_holdout = [h for h in holdout_ids if h not in unresolved]
    hits = [h for h in resolved_holdout if h in top_k_ids]
    recall = len(hits) / len(resolved_holdout) if resolved_holdout else 0.0

    ranks_of_holdout = {}
    id_to_rank = {r.paper.arxiv_id: i + 1 for i, r in enumerate(ranked)}
    for h in resolved_holdout:
        ranks_of_holdout[h] = id_to_rank.get(h)

    result = {
        "method": {
            "reading_list_size": len(papers),
            "seed_count": len(seeds),
            "holdout_count": len(resolved_holdout),
            "holdout_fraction": args.holdout_fraction,
            "split_seed": args.split_seed,
            "distractor_categories": args.categories,
            "distractor_days": args.days,
            "distractor_max_per_category": args.max_per_category,
            "candidate_pool_size": len(candidates),
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "top_k": args.top_k,
        },
        "result": {
            "recall_at_k": round(recall, 4),
            "hits": sorted(hits),
            "misses": sorted(set(resolved_holdout) - set(hits)),
            "ranks_of_holdout_papers": ranks_of_holdout,
            "unresolved_reading_list_ids": unresolved,
        },
        "timing_seconds": {
            "model_load": round(model_load_s, 2),
            "distractor_ingest": round(ingest_s, 2),
            "reading_list_resolve_and_embed": round(seed_resolve_s, 2),
            "ranking": round(rank_s, 4),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default="eval/.bench-data")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--max-per-category", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--min-interval", type=float, default=3.0, help="seconds between arXiv API calls"
    )
    parser.add_argument("--output", default="eval/results.json")
    args = parser.parse_args()

    t0 = time.monotonic()
    result = run(args)
    result["timing_seconds"]["wall_clock_total"] = round(time.monotonic() - t0, 2)

    Path(args.output).write_text(json.dumps(result, indent=2))
    print()
    print(json.dumps(result, indent=2))
    print()
    print(f"recall@{args.top_k}: {result['result']['recall_at_k']:.0%} "
          f"({len(result['result']['hits'])}/{result['method']['holdout_count']})")
    print(f"total wall clock: {result['timing_seconds']['wall_clock_total']}s")
    print(f"results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
