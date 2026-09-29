"""Command line entry point: `papercompass serve`, `papercompass ingest`, `papercompass import`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / ".papercompass"


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from papercompass.server import create_app

    app = create_app(args.data_dir)
    print(f"papercompass serving on http://{args.host}:{args.port}")
    print(f"data directory: {args.data_dir}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from papercompass.arxiv_client import ArxivClient
    from papercompass.db import Database
    from papercompass.embeddings import SentenceTransformerEmbedder
    from papercompass.ingest import ingest_categories

    data_dir = Path(args.data_dir)
    db = Database(data_dir / "papercompass.db")
    client = ArxivClient(data_dir / "arxiv_cache")
    embedder = SentenceTransformerEmbedder()
    categories = args.categories.split(",") if args.categories else None
    new_count = ingest_categories(
        db, client, embedder, categories=categories, days=args.days,
        max_per_category=args.max_per_category,
    )
    result = {"new_papers": new_count, "total_papers": db.count_papers()}
    if args.json:
        print(json.dumps(result))
    else:
        print(f"Ingested {new_count} new papers. Corpus now has {result['total_papers']} papers.")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    from papercompass.arxiv_client import ArxivClient
    from papercompass.db import Database
    from papercompass.embeddings import SentenceTransformerEmbedder
    from papercompass.importers import detect_and_parse
    from papercompass.ingest import add_seeds

    path = Path(args.file)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1
    text = path.read_text(errors="replace")
    ids = detect_and_parse(path.name, text)
    if not ids:
        print("error: no arXiv ids found in the supplied file", file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir)
    db = Database(data_dir / "papercompass.db")
    client = ArxivClient(data_dir / "arxiv_cache")
    embedder = SentenceTransformerEmbedder()
    unresolved = add_seeds(db, client, embedder, ids, source="cli-import")
    added = [i for i in ids if i not in unresolved]
    result = {"added": added, "unresolved": unresolved}
    if args.json:
        print(json.dumps(result))
    else:
        print(f"Added {len(added)} seed papers.")
        if unresolved:
            print(f"Could not resolve: {', '.join(unresolved)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="papercompass",
        description="Recommendations over your own arXiv reading library, offline after setup.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="start the local web UI and API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p_serve.set_defaults(func=_cmd_serve)

    p_ingest = sub.add_parser("ingest", help="fetch recent arXiv papers into the corpus")
    p_ingest.add_argument("--categories", default=None, help="comma separated, e.g. cs.LG,cs.CL")
    p_ingest.add_argument("--days", type=int, default=90)
    p_ingest.add_argument("--max-per-category", type=int, default=100)
    p_ingest.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p_ingest.add_argument("--json", action="store_true")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_import = sub.add_parser("import", help="import a seed library from a file")
    p_import.add_argument("file", help="BibTeX (.bib), Zotero CSV (.csv), or plain text id list")
    p_import.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p_import.add_argument("--json", action="store_true")
    p_import.set_defaults(func=_cmd_import)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
