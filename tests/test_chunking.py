from analyst.chunking import split_page


def numbered_lines(count: int, words_per_line: int = 10) -> str:
    return "\n".join(" ".join(f"w{line}_{word}" for word in range(words_per_line)) for line in range(count))


def test_a_short_page_is_one_passage():
    assert split_page("Sales revenue 866 989\nNet profit 520 869") == ["Sales revenue 866 989\nNet profit 520 869"]


def test_an_empty_page_gives_nothing():
    assert split_page("") == []
    assert split_page("\n\n  \n") == []


def test_passages_respect_the_word_limit_and_keep_lines_whole():
    passages = split_page(numbered_lines(50), max_words=60, overlap_words=10)
    assert len(passages) > 1
    for passage in passages:
        assert len(passage.split()) <= 60
        assert all(len(line.split()) == 10 for line in passage.splitlines())


def test_neighbouring_passages_share_their_edge():
    passages = split_page(numbered_lines(20), max_words=60, overlap_words=10)
    for earlier, later in zip(passages, passages[1:], strict=False):
        assert earlier.splitlines()[-1] == later.splitlines()[0]


def test_every_line_ends_up_somewhere():
    text = numbered_lines(37)
    passages = split_page(text, max_words=60, overlap_words=10)
    assert set(text.splitlines()) == {line for passage in passages for line in passage.splitlines()}


def test_no_passage_is_only_a_repeat_of_the_one_before():
    # eleven lines of ten words: six fill the first passage, the second starts with the sixth again
    passages = split_page(numbered_lines(11), max_words=60, overlap_words=10)
    assert len(passages) == 2
    assert len(passages[1].splitlines()) == 6
    assert passages[1].splitlines()[-1] == numbered_lines(11).splitlines()[-1]


def test_a_line_longer_than_a_passage_is_cut():
    passages = split_page(" ".join(f"w{n}" for n in range(130)), max_words=50, overlap_words=0)
    assert [len(p.split()) for p in passages] == [50, 50, 30]
