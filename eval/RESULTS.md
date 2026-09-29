# Benchmark: held-out recall on the curated reading list

Method, script, and raw output for the number quoted in the README.

## Setup

- Reading list: `eval/reading_list.json`, 50 well-known ML papers spanning 8 topic
  clusters (transformers and LLMs, CNN architectures, detection and segmentation,
  generative models, reinforcement learning, self-supervised and multimodal
  learning, foundational techniques, other). Every id was resolved against the
  live arXiv API and its title checked against the expected paper before it was
  added to the list.
- Split: stratified 80/20 by topic cluster, fixed random seed (`--split-seed 0`),
  giving 40 seed papers and 10 held-out papers. Stratifying by cluster keeps
  each held-out paper in a cluster that still has related seed papers, which is
  the realistic case (nobody's reading list is a random sample of arXiv).
- Distractor corpus: recent papers ingested the same way `papercompass ingest`
  does, categories `cs.LG cs.CL cs.CV stat.ML cs.AI`, last 90 days, up to 150 per
  category.
- Candidate pool: the distractor corpus plus the 50 reading-list papers, minus
  the 40 seeds (`rank_by_seeds` excludes seed ids from its own output).
- Ranking: `papercompass.ranking.rank_by_seeds`, the same function the `/api/feed`
  endpoint uses. Score is the mean cosine similarity to a candidate's 3 nearest
  seed papers.
- Metric: recall@50, the fraction of the 10 held-out papers that land in the top
  50 ranked candidates.
- Model: `sentence-transformers/all-MiniLM-L6-v2`, CPU only, 384-dim embeddings.
- Hardware: MacBook Air, Apple Silicon, CPU inference only, no GPU or Neural
  Engine used.
- Date: 2026-09-30.

## Result

```
recall@50: 90% (9/10)
candidate pool size: 673 papers
total wall clock: 35.8s (model load + distractor ingest + reading-list resolve + ranking)
```

The one miss was `1412.6980` (Adam: A Method for Stochastic Optimization). Its
own topic cluster ("foundational techniques") only has 5 papers in the reading
list, so at a 20% holdout rate it lost its nearest textual neighbors, and its
abstract (about the optimizer's numerical properties) does not share much
vocabulary with the other foundational papers in the seed set (word2vec, batch
normalization, layer normalization, sequence-to-sequence). It ranked 219th out
of 673. This is a real limit of embedding text similarity: a paper's citation
graph position and an abstract's wording do not always agree, and this
approach only sees the wording.

Full machine-readable output, including the rank of every held-out paper, is
in `eval/results.json`.

Repeating the split with `--split-seed 1`, `2`, and `3` (different random 40/10
divisions of the same 50 papers, same distractor corpus) also gave 90% (9/10)
each time, so this is not a lucky split.

## Reproduce

```
uv run python eval/benchmark.py
```

First run takes about 35 seconds (model load plus a cold arXiv cache).
Repeat runs against a warm cache take about 5 to 6 seconds, since the
distractor corpus and the reading-list papers are already fetched and
embedded. Needs network access to the arXiv API. The exact recall number can
shift slightly run to run since arXiv listings change daily and the
distractor pool is refetched, but it has been 90% on every split tried during
development.
