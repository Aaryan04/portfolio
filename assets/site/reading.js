(function () {
  "use strict";

  /* ---------- Card spotlight (same effect as project cards) ---------- */
  if (window.matchMedia("(hover: hover)").matches) {
    document.querySelectorAll(".rcard").forEach(function (card) {
      card.addEventListener("pointermove", function (e) {
        var r = card.getBoundingClientRect();
        card.style.setProperty("--mx", e.clientX - r.left + "px");
        card.style.setProperty("--my", e.clientY - r.top + "px");
      });
    });
  }

  /* ---------- Reading list: filter + search ---------- */
  var grid = document.getElementById("reading-grid");
  if (grid) {
    var cards = Array.prototype.slice.call(grid.querySelectorAll(".rcard"));
    var buttons = Array.prototype.slice.call(document.querySelectorAll(".filter"));
    var search = document.getElementById("reading-search");
    var count = document.getElementById("reading-count");
    var empty = document.getElementById("reading-empty");
    var state = { type: "all", q: "" };
    try {
      var p = new URLSearchParams(location.search);
      if (p.get("type")) state.type = p.get("type");
      if (p.get("q")) state.q = p.get("q");
    } catch (e) {}

    function apply() {
      var words = state.q.toLowerCase().split(/\s+/).filter(Boolean);
      var shown = 0;
      cards.forEach(function (c) {
        var ok = (state.type === "all" || c.getAttribute("data-type") === state.type) &&
          words.every(function (w) { return c.getAttribute("data-search").indexOf(w) !== -1; });
        c.hidden = !ok;
        if (ok) {
          shown++;
          c.classList.add("is-in");
        }
      });
      buttons.forEach(function (b) {
        var on = b.getAttribute("data-filter") === state.type;
        b.classList.toggle("is-on", on);
        b.setAttribute("aria-pressed", String(on));
      });
      count.textContent = shown + (shown === 1 ? " item" : " items");
      empty.hidden = shown !== 0;
      try {
        var u = new URL(location.href);
        if (state.type === "all") u.searchParams.delete("type"); else u.searchParams.set("type", state.type);
        if (state.q) u.searchParams.set("q", state.q); else u.searchParams.delete("q");
        history.replaceState(null, "", u);
      } catch (e) {}
    }
    buttons.forEach(function (b) {
      b.addEventListener("click", function () {
        state.type = b.getAttribute("data-filter");
        apply();
      });
    });
    if (search) {
      search.value = state.q;
      search.addEventListener("input", function () {
        state.q = search.value.trim();
        apply();
      });
    }
    apply();
  }

  /* ---------- Article: maths ---------- */
  function renderMath() {
    var prose = document.querySelector(".prose");
    if (!prose || !window.renderMathInElement) return;
    window.renderMathInElement(prose, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    });
  }
  if (document.querySelector(".prose")) {
    if (window.renderMathInElement) renderMath();
    else window.addEventListener("load", renderMath);
  }

  /* ---------- Article: Mermaid diagrams, loaded only when a page has one ---------- */
  var diagrams = document.querySelectorAll("pre.mermaid");
  if (diagrams.length) {
    diagrams.forEach(function (d) {
      d.setAttribute("data-src", d.textContent);
    });
    var s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js";
    s.onload = function () {
      var draw = function () {
        var dark = document.documentElement.getAttribute("data-theme") === "dark" ||
          (!document.documentElement.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
        diagrams.forEach(function (d) {
          d.removeAttribute("data-processed");
          d.textContent = d.getAttribute("data-src");
        });
        window.mermaid.initialize({ startOnLoad: false, theme: dark ? "dark" : "neutral", fontFamily: "Inter, sans-serif" });
        window.mermaid.run({ nodes: diagrams });
      };
      draw();
      new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    };
    document.head.appendChild(s);
  }

  /* ---------- Article: table of contents ---------- */
  var toc = document.querySelector(".toc");
  if (toc) {
    var box = toc.querySelector("details");
    if (window.innerWidth < 1080 && box) box.open = false;
    var links = Array.prototype.slice.call(toc.querySelectorAll("a"));
    var byId = {};
    links.forEach(function (a) {
      byId[a.getAttribute("href").slice(1)] = a;
      a.addEventListener("click", function () {
        if (window.innerWidth < 1080 && box) box.open = false;
      });
    });
    if ("IntersectionObserver" in window) {
      var spy = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (en) {
            if (!en.isIntersecting) return;
            links.forEach(function (a) { a.classList.remove("is-active"); });
            var a = byId[en.target.id];
            if (a) {
              a.classList.add("is-active");
              if (window.innerWidth >= 1080) {
                var top = a.offsetTop - toc.clientHeight / 2;
                toc.scrollTo({ top: top, behavior: "smooth" });
              }
            }
          });
        },
        { rootMargin: "-20% 0px -70% 0px" }
      );
      document.querySelectorAll(".prose h2[id]").forEach(function (h) { spy.observe(h); });
    }
  }
})();
