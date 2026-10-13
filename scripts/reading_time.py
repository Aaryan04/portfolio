"""Estimate the reading time of a Markdown article.

Usage: python3 blogs/reading_time.py path/to/article.md [more.md ...]

Counts words (prose and tables) at 220 words per minute, with code blocks excluded and inline maths counted
as one word. Adds 30 seconds per display equation ($$...$$) and 2 seconds per table row for scanning. Exits with
code 1 if any article goes over the 30-minute limit in research-papers/WRITING_GUIDE.md.
"""
import re
import sys

WPM, LIMIT = 220, 30.0


def estimate(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    display = len(re.findall(r"\$\$.*?\$\$", text, flags=re.S))
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\$[^$\n]+\$", " X ", text)
    lines = text.splitlines()
    table = [l for l in lines if l.lstrip().startswith("|") and not re.fullmatch(r"[|:\-\s]+", l.strip())]
    prose = "\n".join(l for l in lines if not l.lstrip().startswith("|"))
    count = lambda t: len(re.findall(r"[A-Za-z0-9][\w'’.-]*", t))
    words, table_words = count(prose), count("\n".join(table))
    # table text is read like prose, plus ~2 s per row for scanning across columns
    minutes = (words + table_words) / WPM + display * 0.5 + len(table) * 2 / 60
    return words + table_words, display, len(table), minutes


def main(paths):
    over = False
    for p in paths:
        with open(p, encoding="utf-8") as f:
            words, display, rows, minutes = estimate(f.read())
        flag = "OK" if minutes <= LIMIT else "OVER LIMIT"
        over |= minutes > LIMIT
        print(f"{p}: {words} words, {display} display equations, {rows} table rows -> ~{minutes:.0f} min [{flag}]")
    return 1 if over else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
