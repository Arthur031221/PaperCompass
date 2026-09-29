"""FastAPI app: seed library import, ranked feed, search, single static page."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from papercompass.arxiv_client import ArxivClient
from papercompass.db import Database
from papercompass.embeddings import Embedder, SentenceTransformerEmbedder, cosine_similarity
from papercompass.importers import detect_and_parse
from papercompass.ingest import DEFAULT_CATEGORIES, add_seeds, ingest_categories
from papercompass.ranking import rank_by_seeds

logger = logging.getLogger("papercompass.server")

STATIC_DIR = Path(__file__).parent / "static"


class ImportTextRequest(BaseModel):
    text: str
    format: str = "list"  # list, bibtex, zotero


class IngestRequest(BaseModel):
    categories: list[str] | None = None
    days: int = 90
    max_per_category: int = 100


def create_app(
    data_dir: str | Path,
    embedder: Embedder | None = None,
    client: ArxivClient | None = None,
) -> FastAPI:
    """Build the FastAPI app. Tests pass a fake embedder and client so no
    network call or model load happens.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(data_dir / "papercompass.db")
    embedder = embedder or SentenceTransformerEmbedder()
    client = client or ArxivClient(data_dir / "arxiv_cache")

    app = FastAPI(title="papercompass", version="0.1.0")
    app.state.db = db
    app.state.embedder = embedder
    app.state.client = client

    @app.get("/api/stats")
    def stats():
        return {"papers": db.count_papers(), "seeds": len(db.seed_ids())}

    @app.get("/api/seeds")
    def list_seeds():
        return [p.to_dict() for p in db.seed_papers()]

    @app.delete("/api/seeds/{arxiv_id}")
    def delete_seed(arxiv_id: str):
        db.remove_seed(arxiv_id)
        return {"removed": arxiv_id}

    @app.post("/api/seeds/import-text")
    def import_text(req: ImportTextRequest):
        if req.format == "bibtex":
            from papercompass.importers import parse_bibtex

            ids = parse_bibtex(req.text)
        elif req.format == "zotero":
            from papercompass.importers import parse_zotero_csv

            ids = parse_zotero_csv(req.text)
        else:
            from papercompass.importers import parse_id_list

            ids = parse_id_list(req.text)
        if not ids:
            raise HTTPException(400, "no arXiv ids found in the supplied text")
        unresolved = add_seeds(db, client, embedder, ids, source=req.format)
        added = [i for i in ids if i not in unresolved]
        return {"added": added, "unresolved": unresolved}

    @app.post("/api/seeds/import-file")
    async def import_file(file: UploadFile):
        raw = (await file.read()).decode("utf-8", errors="replace")
        ids = detect_and_parse(file.filename or "", raw)
        if not ids:
            raise HTTPException(400, "no arXiv ids found in the uploaded file")
        source = "bibtex" if (file.filename or "").lower().endswith((".bib", ".bibtex")) else (
            "zotero" if (file.filename or "").lower().endswith(".csv") else "list"
        )
        unresolved = add_seeds(db, client, embedder, ids, source=source)
        added = [i for i in ids if i not in unresolved]
        return {"added": added, "unresolved": unresolved}

    @app.post("/api/ingest")
    def ingest(req: IngestRequest):
        categories = req.categories or DEFAULT_CATEGORIES
        new_count = ingest_categories(
            db,
            client,
            embedder,
            categories=categories,
            days=req.days,
            max_per_category=req.max_per_category,
        )
        return {"new_papers": new_count, "total_papers": db.count_papers()}

    @app.get("/api/feed")
    def feed(limit: int = 50):
        seeds = db.seed_papers()
        if not seeds:
            return {"recommendations": [], "message": "no seed papers yet, import a library first"}
        candidates = db.all_papers()
        ranked = rank_by_seeds(candidates, seeds)[:limit]
        return {
            "recommendations": [
                {
                    **r.paper.to_dict(),
                    "score": round(r.score, 4),
                    "why": [
                        {
                            "arxiv_id": w.arxiv_id,
                            "title": w.title,
                            "similarity": round(w.similarity, 4),
                        }
                        for w in r.why
                    ],
                }
                for r in ranked
            ]
        }

    @app.get("/api/search")
    def search(q: str, limit: int = 20):
        q = q.strip()
        if not q:
            return {"results": []}
        papers = db.all_papers()
        if not papers:
            return {"results": []}
        query_vec = embedder.encode([q])[0]
        import numpy as np

        matrix = np.stack([p.embedding for p in papers])
        sims = cosine_similarity(query_vec.reshape(1, -1), matrix)[0]
        order = sims.argsort()[::-1][:limit]
        return {
            "results": [
                {**papers[i].to_dict(), "score": round(float(sims[i]), 4)} for i in order
            ]
        }

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def index():
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app
