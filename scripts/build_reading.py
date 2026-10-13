"""Build the Reading section of the portfolio.

Sources
  content/reading.json      external entries (interactive guides, side reads, external posts)
  content/papers/*.md       paper explainers (front matter + Markdown body)
  content/blogs/*.md        blog posts (same format; optional)

Outputs (all committed, so GitHub Pages serves them as plain static files)
  index.html                the "Reading" section between the <!-- reading:start/end --> markers
  reading/index.html        the full reading list
  reading/papers/<slug>/    one page per paper   (reading/blogs/<slug>/ for blogs)
  reading/feed.xml          RSS feed of papers and blogs
  sitemap.xml               every page on the site

Usage:  pip install markdown   then   python3 scripts/build_reading.py
"""
import datetime as dt
import html
import json
import os
import re
import sys

import markdown

sys.path.insert(0, os.path.dirname(__file__))
from reading_time import estimate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = "https://aaryan04.github.io/portfolio/"
RESUME = "https://drive.google.com/file/d/1o-hSDuU1eyi58XIbdlitKkuIcR2H6NIx/view?usp=sharing"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
NAV = [("about", "About Me"), ("education", "Education"), ("experience", "Experience"), ("projects", "Projects"),
       ("publications", "Publications"), ("reading", "Reading"), ("contact", "Contact")]

esc = lambda s: html.escape(str(s), quote=True)


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


def write(path, text):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(text)


def nice_date(d):
    y, m, day = (int(x) for x in d.split("-"))
    return f"{MONTHS[m - 1]} {day}, {y}"


# ---------------------------------------------------------------- content loading
def parse_front_matter(text):
    m = re.match(r"---\n(.*?)\n---\n", text, flags=re.S)
    if not m:
        raise ValueError("missing front matter")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    meta["tags"] = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
    meta["featured"] = meta.get("featured", "").lower() == "true"
    return meta, text[m.end():]


def load_items():
    cat = json.loads(read("content/reading.json"))
    items = [dict(it, external=True) for it in cat["items"]]
    for kind, folder in (("paper", "papers"), ("blog", "blogs")):
        d = os.path.join(ROOT, "content", folder)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md"):
                continue
            meta, body = parse_front_matter(read(f"content/{folder}/{fn}"))
            slug = fn[:-3]
            words, displays, rows, minutes = estimate(body)
            if minutes > 30:
                print(f"WARNING: {fn} is ~{minutes:.0f} min, over the 30-minute limit in content/WRITING_GUIDE.md")
            items.append(dict(meta, id=slug, type=meta.get("type", kind), body=body, external=False,
                              path=f"reading/{folder}/{slug}/", minutes=max(1, round(minutes))))
    items.sort(key=lambda it: it["date"], reverse=True)
    return cat["types"], items


# ---------------------------------------------------------------- markdown rendering
def render_markdown(body):
    """Markdown -> HTML, keeping maths, demo blocks and mermaid diagrams intact."""
    stash = []

    def keep(s):
        stash.append(s)
        return f"\u0000{len(stash) - 1}\u0000"

    def demo(m):
        name, caption = m.group(1).strip(), m.group(2).strip()
        cap = markdown.markdown(caption).removeprefix("<p>").removesuffix("</p>")
        return "\n\n" + keep(f'<figure class="paper-demo" data-demo="{esc(name)}"><p class="demo-kicker">Try it</p>'
                             f'<div class="demo-body"></div><figcaption>{cap}</figcaption></figure>') + "\n\n"

    body = re.sub(r"```demo[ \t]+([\w-]+)\n(.*?)```", demo, body, flags=re.S)
    body = re.sub(r"```mermaid\n(.*?)```", lambda m: "\n\n" + keep(f'<pre class="mermaid">{esc(m.group(1))}</pre>') + "\n\n", body, flags=re.S)
    body = re.sub(r"\$\$(.+?)\$\$", lambda m: keep(f'<div class="math-display">$${esc(m.group(1))}$$</div>'), body, flags=re.S)
    body = re.sub(r"(?<![\\$])\$([^$\n]+?)\$", lambda m: keep(f'<span class="math-inline">${esc(m.group(1))}$</span>'), body)

    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc"],
                           extension_configs={"toc": {"toc_depth": "2-2", "permalink": False}})
    out = md.convert(body)
    # stashed blocks that ended up alone in a paragraph should not stay wrapped in <p>
    out = re.sub(r"<p>\u0000(\d+)\u0000</p>", lambda m: stash[int(m.group(1))], out)
    out = re.sub(r"\u0000(\d+)\u0000", lambda m: stash[int(m.group(1))], out)
    out = re.sub(r"<table>", '<div class="table-wrap"><table>', out).replace("</table>", "</table></div>")
    out = re.sub(r'<h2 id="([^"]+)">(.*?)</h2>',
                 r'<h2 id="\1"><a class="anchor" href="#\1" aria-label="Link to this section">#</a>\2</h2>', out)
    toc = [(t["id"], re.sub(r"<[^>]+>", "", t["name"])) for t in md.toc_tokens]
    return out, toc


# ---------------------------------------------------------------- shared chrome
def sprite():
    m = re.search(r'(<svg width="0" height="0".*?</svg>)', read("index.html"), flags=re.S)
    return m.group(1)


def head(title, desc, rel, url, extra=""):
    return f'''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(title)}</title>
    <meta name="description" content="{esc(desc)}" />
    <meta name="author" content="Aaryan Shah" />
    <meta name="theme-color" content="#F5F0E8" media="(prefers-color-scheme: light)" />
    <meta name="theme-color" content="#0E0C0A" media="(prefers-color-scheme: dark)" />
    <link rel="canonical" href="{SITE}{url}" />
    <meta property="og:type" content="article" />
    <meta property="og:title" content="{esc(title)}" />
    <meta property="og:description" content="{esc(desc)}" />
    <meta property="og:url" content="{SITE}{url}" />
    <meta property="og:image" content="{SITE}assets/site/img/og-image.jpg" />
    <meta name="twitter:card" content="summary_large_image" />
    <link rel="alternate" type="application/rss+xml" title="Aaryan Shah, reading list" href="{rel}reading/feed.xml" />
    <link rel="icon" href="{rel}assets/site/favicon.svg" type="image/svg+xml" />
    <link rel="preload" href="{rel}assets/site/fonts/fraunces-latin-opsz-normal.woff2" as="font" type="font/woff2" crossorigin />
    <link rel="preload" href="{rel}assets/site/fonts/inter-latin-wght-normal.woff2" as="font" type="font/woff2" crossorigin />
    <link rel="stylesheet" href="{rel}assets/site/site.css" />
    <link rel="stylesheet" href="{rel}assets/site/reading.css" />{extra}
    <script>
      (function () {{
        var d = document.documentElement;
        d.classList.add("js");
        try {{
          var t = localStorage.getItem("theme");
          if (t === "light" || t === "dark") d.setAttribute("data-theme", t);
        }} catch (e) {{}}
      }})();
    </script>
  </head>
'''


def chrome_top(rel, active="reading"):
    current = ' class="is-active" aria-current="page"'
    links = "\n".join(
        f'          <a href="{rel}reading/"{current if active == k else ""}>{v}</a>' if k == "reading"
        else f'          <a href="{rel}#{k}">{v}</a>' for k, v in NAV)
    mobile = "\n".join(
        f'        <a href="{rel}reading/"><span>{i:02d}</span>{v}</a>' if k == "reading" else f'        <a href="{rel}#{k}"><span>{i:02d}</span>{v}</a>'
        for i, (k, v) in enumerate(NAV, 1))
    return f'''  <body class="reading-page">
    {sprite()}

    <a class="skip-link" href="#main">Skip to content</a>
    <div class="progress" aria-hidden="true"><span></span></div>

    <header class="nav" id="nav">
      <div class="nav-inner">
        <a class="brand" href="{rel}" aria-label="Aaryan Shah, home">
          <span class="brand-mark">AS</span>
          <span class="brand-name">Aaryan Shah</span>
        </a>
        <nav class="nav-links" aria-label="Primary">
{links}
        </nav>
        <div class="nav-actions">
          <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle dark mode">
            <svg class="i-sun" aria-hidden="true"><use href="#i-sun" /></svg>
            <svg class="i-moon" aria-hidden="true"><use href="#i-moon" /></svg>
          </button>
          <a class="btn btn-sm btn-outline nav-resume" target="_blank" rel="noopener" href="{RESUME}">
            Résumé <svg aria-hidden="true"><use href="#i-arrow-ur" /></svg>
          </a>
          <button class="menu-toggle" id="menu-toggle" type="button" aria-label="Open menu" aria-expanded="false" aria-controls="mobile-menu">
            <span></span><span></span>
          </button>
        </div>
      </div>
    </header>

    <div class="mobile-menu" id="mobile-menu" hidden>
      <nav aria-label="Mobile">
{mobile}
      </nav>
      <a class="btn btn-solid" target="_blank" rel="noopener" href="{RESUME}">Résumé <svg aria-hidden="true"><use href="#i-arrow-ur" /></svg></a>
    </div>
'''


def chrome_bottom(rel, scripts=""):
    return f'''
    <footer class="footer">
      <div class="container footer-inner">
        <p class="footer-sig">Aaryan <em>Shah</em></p>
        <p class="mono muted">© <span id="year">{dt.date.today().year}</span> · <a href="{rel}reading/">Reading list</a> · <a href="{rel}reading/feed.xml">RSS</a></p>
        <a class="to-top" href="#top" aria-label="Back to top"><svg aria-hidden="true"><use href="#i-arrow-up" /></svg></a>
      </div>
    </footer>

    <script src="{rel}assets/site/site.js" defer></script>
    <script src="{rel}assets/site/reading.js" defer></script>{scripts}
  </body>
</html>
'''


# ---------------------------------------------------------------- cards
def meta_line(it, types):
    bits = [] if it["external"] else [f'{it["minutes"]} min read']
    bits.append(nice_date(it["date"]))
    return " · ".join(bits)


def card(it, types, rel, cls="rcard", heading="h3"):
    label = types[it["type"]]["label"]
    href = it["url"] if it["external"] else rel + it["path"]
    target = ' target="_blank" rel="noopener"' if it["external"] else ""
    cta = "Open the guide" if it["type"] == "guide" else ("Open it" if it["external"] else "Read the explainer" if it["type"] == "paper" else "Read the post")
    icon = "i-arrow-ur" if it["external"] else "i-arrow-r"
    tags = "".join(f"<li>{esc(t)}</li>" for t in it["tags"])
    search = " ".join([it["title"], it.get("subtitle", ""), it["summary"], label, " ".join(it["tags"])]).lower()
    sub = f'<p class="rcard-sub">{esc(it["subtitle"])}</p>' if it.get("subtitle") else ""
    return f'''<article class="{cls} reveal" data-type="{it["type"]}" data-search="{esc(search)}">
  <div class="rcard-top"><span class="rcard-type t-{it["type"]}">{label}</span><span class="rcard-meta">{meta_line(it, types)}</span></div>
  <{heading} class="rcard-title"><a href="{href}"{target}>{esc(it["title"])}</a></{heading}>
  {sub}<p class="rcard-summary">{esc(it["summary"])}</p>
  <ul class="tags">{tags}</ul>
  <span class="link-arrow" aria-hidden="true">{cta} <svg><use href="#{icon}" /></svg></span>
</article>'''


# ---------------------------------------------------------------- pages
def home_section(types, items):
    internal = [it for it in items if not it["external"]][:2]
    rest = [it for it in items if it not in internal]
    picks = internal + [it for it in rest if it.get("featured")] + [it for it in rest if not it.get("featured")]
    # one wide card plus an even number of regular cards keeps the two-column grid tidy
    picks = picks[:5] if picks and not picks[0]["external"] else picks[:6]
    counts = {t: sum(1 for it in items if it["type"] == t) for t in types}
    summary = " · ".join(f'{n} {types[t]["plural"].lower() if n != 1 else types[t]["label"].lower()}' for t, n in counts.items() if n)
    cards = "\n".join(card(it, types, "", cls="rcard" + (" rcard-wide" if i == 0 and not it["external"] else "")) for i, it in enumerate(picks))
    return f'''<!-- reading:start (generated by scripts/build_reading.py; edit content/ and rebuild) -->
      <section class="section" id="reading">
        <div class="container">
          <header class="sec-head reveal">
            <span class="sec-num">06</span>
            <h2 class="sec-title">Reading <em>List</em></h2>
          </header>
          <div class="reading-intro reveal">
            <p>Things I build to learn in public: interactive guides that explain a field end to end, and paper explainers that take a landmark paper apart for a first-time reader in under 30 minutes.</p>
            <p class="mono muted">{summary}</p>
          </div>
          <div class="rgrid">
{cards}
          </div>
          <div class="reading-more reveal"><a class="btn btn-solid" href="reading/">Browse the full reading list <svg aria-hidden="true"><use href="#i-arrow-r" /></svg></a></div>
        </div>
      </section>
      <!-- reading:end -->'''


def list_page(types, items):
    rel = "../"
    counts = {t: sum(1 for it in items if it["type"] == t) for t in types}
    chips = '<button class="filter is-on" type="button" data-filter="all" aria-pressed="true">All <span>{}</span></button>'.format(len(items))
    chips += "".join(f'<button class="filter" type="button" data-filter="{t}" aria-pressed="false">{types[t]["plural"]} <span>{n}</span></button>'
                     for t, n in counts.items() if n)
    cards = "\n".join(card(it, types, rel) for it in items)
    desc = "Aaryan Shah's reading list: interactive guides on LLMs, GPUs, diffusion and recommender systems, plus paper explainers written for first-time readers."
    return (head("Reading List · Aaryan Shah", desc, rel, "reading/") + chrome_top(rel) + f'''
    <main id="main">
      <section class="section reading-hero" id="top">
        <div class="container">
          <a class="crumb" href="{rel}"><svg aria-hidden="true"><use href="#i-arrow-r" /></svg> Back to portfolio</a>
          <header class="sec-head reveal">
            <span class="sec-num">{len(items):02d}</span>
            <h1 class="sec-title">Reading <em>List</em></h1>
          </header>
          <div class="reading-intro reveal">
            <p>Everything I have written or built to understand a topic properly. <b>Interactive guides</b> open in their own site with animations and simulations; <b>paper explainers</b> live here and cover a whole paper for a first-time reader in 30 minutes or less.</p>
          </div>
          <div class="reading-tools reveal">
            <div class="filters" role="group" aria-label="Filter by type">{chips}</div>
            <label class="search"><span class="visually-hidden">Search the reading list</span><input id="reading-search" type="search" placeholder="Search titles, topics, tags…" autocomplete="off" /></label>
          </div>
          <p class="reading-count mono muted" id="reading-count" aria-live="polite">{len(items)} items</p>
          <div class="rgrid rgrid-list" id="reading-grid">
{cards}
          </div>
          <p class="reading-empty" id="reading-empty" hidden>Nothing matches that search yet.</p>
        </div>
      </section>
    </main>
''' + chrome_bottom(rel))


def paper_page(it, types, items):
    rel = "../../../"
    body_html, toc = render_markdown(it["body"])
    label = types[it["type"]]["label"]
    toc_html = "".join(f'<li><a href="#{tid}">{esc(name)}</a></li>' for tid, name in toc)
    others = [o for o in items if o is not it][:3]
    more = "\n".join(card(o, types, rel) for o in others)
    paper_link = (f'<a class="chip-link" href="{esc(it["paper_url"])}" target="_blank" rel="noopener"><svg aria-hidden="true"><use href="#i-paper" /></svg>'
                  f'{esc(it.get("paper_label", "Read the paper"))}</a>') if it.get("paper_url") else ""
    rows = [("Authors", it.get("authors")), ("Affiliation", it.get("affiliation")), ("Published", " · ".join(x for x in (it.get("venue"),) if x)),
            ("You need", it.get("prerequisites"))]
    facts = "".join(f"<div><dt>{k}</dt><dd>{esc(v)}</dd></div>" for k, v in rows if v)
    katex = (f'\n    <link rel="stylesheet" href="{rel}assets/site/vendor/katex/katex.min.css" />')
    scripts = (f'\n    <script src="{rel}assets/site/vendor/katex/katex.min.js" defer></script>'
               f'\n    <script src="{rel}assets/site/vendor/katex/auto-render.min.js" defer></script>')
    if 'class="paper-demo"' in body_html:
        scripts += f'\n    <script src="{rel}assets/site/paper-demos.js" defer></script>'
    title = f'{it["title"]}, explained · Aaryan Shah' if it["type"] == "paper" else f'{it["title"]} · Aaryan Shah'
    return (head(title, it["summary"], rel, it["path"], katex) + chrome_top(rel) + f'''
    <main id="main" class="article-main">
      <div class="container">
        <a class="crumb" href="{rel}reading/"><svg aria-hidden="true"><use href="#i-arrow-r" /></svg> Reading list</a>
        <header class="article-head" id="top">
          <p class="article-kicker"><span class="rcard-type t-{it["type"]}">{label}</span><span class="mono muted">{it["minutes"]} min read · {nice_date(it["date"])}</span></p>
          <h1 class="article-title">{esc(it["title"])}</h1>
          <p class="article-sub">{esc(it.get("subtitle", ""))}</p>
          <dl class="article-facts">{facts}</dl>
          <div class="pub-links">{paper_link}<a class="chip-link" href="{rel}reading/"><svg aria-hidden="true"><use href="#i-arrow-r" /></svg>All readings</a></div>
        </header>
        <div class="article-grid">
          <aside class="toc" aria-label="On this page">
            <details class="toc-box" open><summary>On this page</summary><ol>{toc_html}</ol></details>
          </aside>
          <article class="prose">
{body_html}
          </article>
        </div>
        <section class="keep-reading">
          <h2 class="kr-title">Keep <em>reading</em></h2>
          <div class="rgrid">
{more}
          </div>
        </section>
      </div>
    </main>
''' + chrome_bottom(rel, scripts))


def feed(items):
    entries = [it for it in items if not it["external"]]
    out = "".join(f'''
    <item>
      <title>{esc(it["title"])}</title>
      <link>{SITE}{it["path"]}</link>
      <guid>{SITE}{it["path"]}</guid>
      <pubDate>{dt.datetime.strptime(it["date"], "%Y-%m-%d").strftime("%a, %d %b %Y 00:00:00 +0000")}</pubDate>
      <description>{esc(it["summary"])}</description>
    </item>''' for it in entries)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Aaryan Shah, reading list</title>
    <link>{SITE}reading/</link>
    <description>Paper explainers and posts by Aaryan Shah.</description>{out}
  </channel>
</rss>
'''


def sitemap(items):
    urls = ["", "reading/"] + [it["path"] for it in items if not it["external"]]
    body = "".join(f"\n  <url><loc>{SITE}{u}</loc><lastmod>{dt.date.today().isoformat()}</lastmod></url>" for u in urls)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{body}\n</urlset>\n'


def main():
    types, items = load_items()
    home = read("index.html")
    new_section = home_section(types, items)
    if "<!-- reading:start" in home:
        home = re.sub(r"<!-- reading:start.*?<!-- reading:end -->", lambda m: new_section, home, flags=re.S)
    else:
        home = home.replace("      <!-- Contact -->", "      " + new_section + "\n\n      <!-- Contact -->", 1)
    write("index.html", home)
    write("reading/index.html", list_page(types, items))
    for it in items:
        if not it["external"]:
            write(it["path"] + "index.html", paper_page(it, types, items))
    write("reading/feed.xml", feed(items))
    write("sitemap.xml", sitemap(items))
    print(f"Built reading list: {len(items)} items, {sum(not it['external'] for it in items)} article page(s).")


if __name__ == "__main__":
    main()
