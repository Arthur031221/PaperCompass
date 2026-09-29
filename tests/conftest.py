"""Shared test fixtures. The real sentence-transformers model is never
loaded in the test suite: FakeEmbedder produces small deterministic vectors
from a hash of the text instead, so tests run in seconds, not minutes.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

DIM = 16


class FakeEmbedder:
    """Deterministic, hash-based embedder standing in for the real model.

    Same text always produces the same vector, and different texts produce
    (with overwhelming probability) different vectors, which is enough to
    exercise ranking, search and the API without downloading or running
    sentence-transformers.
    """

    def __init__(self):
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = np.frombuffer(digest[:DIM * 4], dtype=np.uint8).astype(np.float32)
            raw = raw[:DIM] if raw.size >= DIM else np.pad(raw, (0, DIM - raw.size))
            vec = (raw - raw.mean()) / (raw.std() + 1e-6)
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm > 0 else vec)
        return np.stack(vectors).astype(np.float32)


@pytest.fixture()
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture()
def tmp_data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture()
def sample_feed_xml():
    return SAMPLE_FEED_XML


SAMPLE_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <updated>2023-08-02T00:41:18Z</updated>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
    <summary>  The dominant sequence transduction models are based on complex
    recurrent or convolutional neural networks.  </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2005.14165v4</id>
    <updated>2020-07-22T22:16:22Z</updated>
    <published>2020-05-28T17:29:03Z</published>
    <title>Language Models are Few-Shot Learners</title>
    <summary>  We demonstrate that scaling up language models greatly
    improves task-agnostic, few-shot performance.  </summary>
    <author><name>Tom B. Brown</name></author>
    <category term="cs.CL"/>
  </entry>
</feed>
"""
