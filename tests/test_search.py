import math

from analyst.embeddings import HashingEmbedder
from analyst.search import word_query


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_stable_and_normalised():
    embedder = HashingEmbedder()
    first, second = embedder.query("sales revenue 2025"), embedder.query("sales revenue 2025")
    assert first == second and len(first) == embedder.dimension == 384
    assert math.isclose(sum(value * value for value in first), 1.0, rel_tol=1e-9)


def test_texts_that_share_words_are_closer():
    embedder = HashingEmbedder()
    query = embedder.query("dividend per share")
    close, far = embedder.passages(["a dividend of PLN 2.50 per share", "the Group operated 412 bakeries"])
    assert cosine(query, close) > cosine(query, far)


def test_word_query_is_an_or_of_distinct_words():
    assert word_query("How many stores, and how many NEW stores?") == "how | many | stores | and | new"
    assert word_query("?!") == ""


def test_the_page_with_the_words_comes_first(search):
    hits = search.search("dividend per share recommended by the Management Board")
    assert hits[0].page_no == 3
    assert hits[0].citation == "Baltic Mills 2025, p. 3"
    assert "PLN 2.50 per share" in hits[0].text


def test_a_figure_can_be_found_by_its_line_item(search):
    hits = search.search("net profit")
    assert hits[0].page_no == 2 and "85 300" in hits[0].text


def test_filters_restrict_to_one_report(search):
    assert search.search("bakeries", company="baltic mills", fiscal_year=2025)[0].page_no == 1
    assert search.search("bakeries", company="Baltic Mills", fiscal_year=2024) == []
    assert search.search("bakeries", company="Another Company") == []


def test_limit_is_respected(search):
    assert len(search.search("the Group", limit=2)) == 2


def test_reports_and_pages_can_be_read(search):
    assert search.reports() == [{"report_id": "baltic-2025", "company": "Baltic Mills", "fiscal_year": 2025,
                                 "title": "Baltic Mills annual report 2025", "pages": 3}]
    assert "Financial highlights" in search.page("baltic-2025", 2)
    assert search.page("baltic-2025", 9) is None
    assert search.page("unknown", 1) is None
