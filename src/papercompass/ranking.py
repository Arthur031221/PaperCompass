"""Rank a corpus of papers by similarity to a seed library.

Score for a candidate paper is the mean cosine similarity to its 3 nearest
seed papers (or fewer, if the seed library is smaller). Those same nearest
seeds are surfaced as the "why recommended" explanation, so the score a
paper gets is always traceable to specific papers already in the library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from papercompass.db import Paper
from papercompass.embeddings import cosine_similarity

DEFAULT_WHY_K = 3


@dataclass
class WhyMatch:
    arxiv_id: str
    title: str
    similarity: float


@dataclass
class Recommendation:
    paper: Paper
    score: float
    why: list[WhyMatch]


def rank_by_seeds(
    candidates: list[Paper],
    seeds: list[Paper],
    why_k: int = DEFAULT_WHY_K,
    exclude_seed_ids: bool = True,
) -> list[Recommendation]:
    """Return candidates ranked descending by similarity to the seed set."""
    seeds_with_embeddings = [s for s in seeds if s.embedding is not None]
    if not seeds_with_embeddings:
        return []

    seed_ids = {s.arxiv_id for s in seeds_with_embeddings}
    seed_matrix = np.stack([s.embedding for s in seeds_with_embeddings])

    pool = candidates
    if exclude_seed_ids:
        pool = [c for c in candidates if c.arxiv_id not in seed_ids]
    pool = [c for c in pool if c.embedding is not None]
    if not pool:
        return []

    candidate_matrix = np.stack([c.embedding for c in pool])
    sims = cosine_similarity(candidate_matrix, seed_matrix)  # (n_candidates, n_seeds)

    k = min(why_k, len(seeds_with_embeddings))
    results = []
    for row_idx, paper in enumerate(pool):
        row = sims[row_idx]
        top_idx = np.argsort(-row)[:k]
        score = float(np.mean(row[top_idx]))
        why = [
            WhyMatch(
                arxiv_id=seeds_with_embeddings[i].arxiv_id,
                title=seeds_with_embeddings[i].title,
                similarity=float(row[i]),
            )
            for i in top_idx
        ]
        results.append(Recommendation(paper=paper, score=score, why=why))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
