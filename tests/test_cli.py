from __future__ import annotations

import pytest

from papercompass.cli import build_parser, main


def test_parser_requires_command():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_serve_defaults():
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.command == "serve"


def test_parser_ingest_defaults():
    parser = build_parser()
    args = parser.parse_args(["ingest"])
    assert args.days == 90
    assert args.max_per_category == 100


def test_import_missing_file_returns_error(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.txt"
    code = main(["import", str(missing), "--data-dir", str(tmp_path / "data")])
    assert code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_import_no_ids_returns_error(tmp_path, capsys):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("no ids in here")
    code = main(["import", str(empty_file), "--data-dir", str(tmp_path / "data")])
    assert code == 1
    captured = capsys.readouterr()
    assert "no arXiv ids" in captured.err
