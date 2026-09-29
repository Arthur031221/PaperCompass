from __future__ import annotations

import numpy as np

from papercompass.db import Paper
from papercompass.ranking import rank_by_seeds


def paper(arxiv_id, vec):
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="",
        embedding=np.array(vec, dtype=np.float32),
    )


def test_rank_by_seeds_orders_by_similarity():
    seeds = [paper("seed1", [1.0, 0.0])]
    candidates = [
        paper("close", [0.95, 0.05]),
        paper("far", [0.0, 1.0]),
        paper("mid", [0.6, 0.4]),
    ]
    ranked = rank_by_seeds(candidates, seeds)
    assert [r.paper.arxiv_id for r in ranked] == ["close", "mid", "far"]
    assert ranked[0].score > ranked[1].score > ranked[2].score


def test_rank_by_seeds_why_explains_nearest_seeds():
    seeds = [paper("seedA", [1.0, 0.0]), paper("seedB", [0.0, 1.0])]
    candidates = [paper("cand", [0.9, 0.1])]
    ranked = rank_by_seeds(candidates, seeds, why_k=2)
    assert ranked[0].why[0].arxiv_id == "seedA"
    assert len(ranked[0].why) == 2


def test_rank_by_seeds_excludes_seed_ids_by_default():
    seeds = [paper("seed1", [1.0, 0.0])]
    candidates = [paper("seed1", [1.0, 0.0]), paper("other", [0.9, 0.1])]
    ranked = rank_by_seeds(candidates, seeds)
    assert [r.paper.arxiv_id for r in ranked] == ["other"]


def test_rank_by_seeds_no_seeds_returns_empty():
    assert rank_by_seeds([paper("a", [1.0, 0.0])], []) == []


def test_rank_by_seeds_no_embedded_candidates_returns_empty():
    seeds = [paper("seed1", [1.0, 0.0])]
    unembedded = Paper(arxiv_id="x", title="x", abstract="", embedding=None)
    assert rank_by_seeds([unembedded], seeds) == []
