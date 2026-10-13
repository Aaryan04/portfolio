# Writing guide for paper explainers and blog posts

Every article on the portfolio's Reading page (`content/papers/` and `content/blogs/`) follows these rules so the collection reads consistently.

## The three core rules

1. **30 minutes maximum.** A reader must be able to finish an article in 30 minutes or less.
   - Budget about **6,000 words of prose at most**, at roughly 220 words per minute.
   - Count about **30 seconds for every display equation** and leave room for tables.
   - Check with `python3 scripts/reading_time.py content/papers/<slug>.md`. The build script also warns if an article is over the limit.
   - The reading time shown on the site is computed automatically; you do not need to write it in the article.

2. **Cover everything in the paper.** A reader who finishes the article should know what the paper says, not just its headline.
   - Walk through every section of the paper: introduction, background, method, analysis, training setup, every results table, ablations, extra experiments, conclusion and the appendix.
   - Include the key numbers: datasets and sizes, model sizes and hyperparameters, headline results with the baselines they beat, and what the ablations showed.
   - Include the authors' own limitations and future work.
   - When a table is large, reproduce the rows that carry the argument and say they are "selected rows".

3. **Written for a first-time reader.** Assume the reader has never seen the paper and knows only basic ideas about machine learning.
   - Explain every prerequisite in place, the first time it is needed (for example RNNs, BLEU, beam search, label smoothing). Do not just link elsewhere.
   - Pair every important equation with a plain-language reading, and ideally a tiny worked example with real numbers.
   - Use an analogy where it genuinely helps, and label it as an analogy.

## Accuracy rules

- **Point to the source.** Write section and table references in brackets, like (§3.2) or (Table 3), so readers can check against the paper.
- **Quote sparingly and exactly.** Short direct quotes in quotation marks are fine for the authors' key claims.
- **Keep the paper separate from hindsight.** Anything from after the paper (later models, what aged well, what changed) goes in a clearly labelled section such as "Reading it today". Never present it as the paper's claim.
- **Note version differences.** If arXiv versions or the conference version report different numbers, say which you use and mention the other.
- **Do not invent numbers.** If a figure cannot be confirmed, leave it out or flag it.

## Standard structure for a paper explainer

The page header (title, subtitle, authors, affiliation, venue, paper link, prerequisites, reading time) is generated from the front matter, so the article body starts with the TL;DR. Keep this order unless the paper's own structure demands otherwise:

1. **TL;DR:** 4 to 6 bullets a reader could repeat to a friend.
2. **How this article is organised:** a short map that mirrors the paper's sections.
3. **The problem and prior work:** what was hard before this paper, and why.
4. **The method:** piece by piece, with equations, intuition and a worked example.
5. **Analysis and motivation:** the paper's own argument for why the method works.
6. **Experimental setup:** data, hardware, optimisation and regularisation.
7. **Results:** main tables, ablations and secondary experiments, each with a sentence on what it shows.
8. **Appendix highlights:** visualisations, proofs, extra results.
9. **The authors' conclusion and future work.**
10. **Reading it today** (optional, clearly labelled as post-paper context).
11. **Glossary** of every term a first-time reader might not know.
12. **Cheat sheet** of the numbers and equations worth remembering.

## Style

- Plain, direct sentences. Define a term before using it.
- Use headings, short paragraphs, tables and bullet lists; avoid walls of text.
- Use GitHub-flavoured Markdown. Write maths with `$...$` (inline) and `$$...$$` (display), and diagrams with Mermaid code blocks, so everything renders on GitHub.
- Use British or American spelling consistently within an article.
- No hype words. Let the numbers make the case.

## Interactive figures

Add a live figure anywhere in an article with a fenced block whose language is `demo`, followed by the demo name and a caption:

````markdown
```demo attention
Caption shown under the figure.
```
````

Demos are implemented in `assets/site/paper-demos.js`; add a new entry to `DEMOS` for a new figure. Mermaid diagrams (```` ```mermaid ````) also work and load their library only on pages that use them. Label figures that use made-up values as illustrative.

## Files and naming

```
content/
├── reading.json              # external entries: interactive guides, side reads
├── WRITING_GUIDE.md          # this file
├── papers/<slug>.md          # paper explainers (front matter + Markdown)
└── blogs/<slug>.md           # blog posts, same format (optional)
scripts/
├── build_reading.py          # builds the Reading pages, feed and sitemap
└── reading_time.py           # reading-time checker
```

- `<slug>` is the title in lowercase kebab-case, for example `attention-is-all-you-need`. It becomes the URL: `/reading/papers/<slug>/`.
- Start each article with front matter like `content/papers/attention-is-all-you-need.md`: `title`, `type`, `subtitle`, `authors`, `affiliation`, `venue`, `year`, `paper_url`, `paper_label`, `date` (publish date, YYYY-MM-DD), `summary`, `tags` (comma-separated), `prerequisites`, `featured`.
- Do not copy figures from papers unless their licence allows it. Redraw them as demos, tables or your own images.
