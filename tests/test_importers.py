from __future__ import annotations

from papercompass.importers import (
    detect_and_parse,
    parse_bibtex,
    parse_id_list,
    parse_zotero_csv,
)

BIBTEX_SAMPLE = """
@article{vaswani2017attention,
  title = {Attention is All You Need},
  author = {Vaswani, Ashish and Shazeer, Noam},
  eprint = {1706.03762},
  archivePrefix = {arXiv},
  year = {2017}
}

@article{brown2020language,
  title = {Language Models are Few-Shot Learners},
  author = {Brown, Tom B.},
  url = {https://arxiv.org/abs/2005.14165},
  year = {2020}
}

@book{ignoreme,
  title = {Not an arXiv paper},
  year = {1999}
}
"""

ZOTERO_CSV_SAMPLE = """Key,Item Type,Title,Author,Url,DOI,Extra
AB12CD34,journalArticle,Attention is All You Need,Vaswani et al.,https://arxiv.org/abs/1706.03762,,
EF56GH78,journalArticle,Language Models are Few-Shot Learners,Brown et al.,,,arXiv:2005.14165
IJ90KL12,journalArticle,Some Unrelated Book,Someone,https://example.com/book,,
"""

ID_LIST_SAMPLE = """
1706.03762
https://arxiv.org/abs/2005.14165v4
2010.11929, 1810.04805
"""


def test_parse_bibtex_finds_ids_and_ignores_non_arxiv():
    ids = parse_bibtex(BIBTEX_SAMPLE)
    assert ids == ["1706.03762", "2005.14165"]


def test_parse_zotero_csv_scans_all_columns():
    ids = parse_zotero_csv(ZOTERO_CSV_SAMPLE)
    assert ids == ["1706.03762", "2005.14165"]


def test_parse_id_list_dedupes_and_normalizes():
    ids = parse_id_list(ID_LIST_SAMPLE)
    assert ids == ["1706.03762", "2005.14165", "2010.11929", "1810.04805"]


def test_parse_id_list_empty():
    assert parse_id_list("") == []


def test_detect_and_parse_by_extension():
    assert detect_and_parse("library.bib", BIBTEX_SAMPLE) == ["1706.03762", "2005.14165"]
    assert detect_and_parse("export.csv", ZOTERO_CSV_SAMPLE) == ["1706.03762", "2005.14165"]
    assert detect_and_parse("ids.txt", ID_LIST_SAMPLE)[0] == "1706.03762"
