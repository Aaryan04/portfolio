# Reading section content

Source for the portfolio's Reading page (`/reading/`), opened from the "Reading" item in the site nav. Pages are generated, so edit files here and rebuild.

## Add a paper explainer or blog post

1. Write `content/papers/<slug>.md` (or `content/blogs/<slug>.md`) with front matter. Copy the header of `papers/attention-is-all-you-need.md`, and follow [`WRITING_GUIDE.md`](WRITING_GUIDE.md).
2. Rebuild:
   ```bash
   pip install markdown          # once
   python3 scripts/build_reading.py
   ```
3. Preview with `python3 -m http.server` from the repo root, then open http://localhost:8000/reading/.
4. Commit the Markdown and the regenerated files (`reading/`, `sitemap.xml`). GitHub Pages serves them as they are.

## Add an interactive guide or external link

Add an entry to `reading.json` with `id`, `type` (`guide`, `side`, `blog` or `paper`), `title`, `summary`, `url`, `tags` and `date`. Then rebuild.

## What the build generates

- `reading/index.html`: the reading list, newest first, with type filters and search. The home page only links to it from the nav; the build does not change `index.html`.
- `reading/papers/<slug>/index.html`: article pages with a table of contents, KaTeX maths and interactive figures.
- `reading/feed.xml` (RSS) and `sitemap.xml`.
