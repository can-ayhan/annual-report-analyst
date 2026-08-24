"""Split a page into passages short enough to embed and to quote.

A passage never leaves its page, which is what makes page citations exact.
Lines are kept whole where possible because in these reports a line is often
a table row.
"""

from __future__ import annotations

MAX_WORDS = 180
OVERLAP_WORDS = 30


def split_page(text: str, max_words: int = MAX_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[str]:
    passages: list[str] = []
    current: list[str] = []      # lines of the passage being filled
    words = 0
    new_lines = 0                # lines in it that no earlier passage has
    for line in _lines_within(text, max_words):
        size = len(line.split())
        if new_lines and words + size > max_words:
            passages.append("\n".join(current))
            current = _tail(current, overlap_words)
            words = sum(len(kept.split()) for kept in current)
            new_lines = 0
        current.append(line)
        words += size
        new_lines += 1
    if new_lines:
        passages.append("\n".join(current))
    return passages


def _lines_within(text: str, max_words: int) -> list[str]:
    """Non-empty lines; a line longer than a passage is cut into pieces that fit."""
    lines: list[str] = []
    for line in text.splitlines():
        words = line.split()
        for start in range(0, len(words), max_words):
            lines.append(" ".join(words[start:start + max_words]))
    return lines


def _tail(lines: list[str], overlap_words: int) -> list[str]:
    """The last lines of a passage, up to overlap_words, repeated at the start of the next one."""
    kept: list[str] = []
    words = 0
    for line in reversed(lines):
        size = len(line.split())
        if words + size > overlap_words:
            break
        kept.insert(0, line)
        words += size
    return kept
