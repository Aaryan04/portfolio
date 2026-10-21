/* Interactive figures for paper explainers.
   Markdown blocks written as ```demo <name> … ``` become <figure class="paper-demo" data-demo="<name>">,
   and this file mounts the matching demo. Colours come from the site's CSS tokens, so demos follow the theme. */
(function () {
  "use strict";
  var root = document.documentElement;
  var redraws = [];
  function css(name) {
    return getComputedStyle(root).getPropertyValue(name).trim();
  }
  function C() {
    return { ink: css("--ink"), ink2: css("--ink-2"), muted: css("--muted"), line: css("--line"), lineS: css("--line-strong"),
      accent: css("--accent"), gold: css("--gold"), bg: css("--bg"), bg2: css("--bg-2"), surface: css("--surface"),
      cool: root.getAttribute("data-theme") === "dark" || (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches) ? "#7fa8d9" : "#3567a3" };
  }
  var MONO = '12px "JetBrains Mono", ui-monospace, monospace';
  var SANS = '600 13px Inter, system-ui, sans-serif';
  function rerender() { redraws.forEach(function (f) { f(); }); }
  new MutationObserver(rerender).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(rerender);

  function el(tag, attrs, kids) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === "text") e.textContent = attrs[k];
      else if (k === "html") e.innerHTML = attrs[k];
      else e.setAttribute(k, attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c) e.appendChild(c); });
    return e;
  }
  function canvas(host, height, draw) {
    var c = el("canvas");
    host.appendChild(c);
    var ctx = c.getContext("2d"), s = { c: c, w: 0, h: 0 };
    s.redraw = function () {
      if (!s.w) return;
      ctx.save();
      ctx.clearRect(0, 0, s.w, s.h);
      draw(ctx, s.w, s.h, C());
      ctx.restore();
    };
    function resize() {
      var w = Math.floor(c.parentElement.clientWidth);
      if (!w) return;
      var h = Math.round(typeof height === "function" ? height(w) : height);
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (w === s.w && h === s.h) return;
      s.w = w; s.h = h;
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      s.redraw();
    }
    new ResizeObserver(resize).observe(host);
    resize();
    redraws.push(s.redraw);
    return s;
  }
  function slider(host, o) {
    var id = "d" + Math.random().toString(36).slice(2, 8);
    var inp = el("input", { type: "range", id: id, min: o.min, max: o.max, step: o.step || 1, value: o.value });
    var out = el("output", { for: id });
    function upd() { out.textContent = o.fmt ? o.fmt(+inp.value) : inp.value; }
    inp.addEventListener("input", function () { upd(); o.on(+inp.value); });
    upd();
    host.appendChild(el("label", { for: id }, [document.createTextNode(o.label), inp, out]));
    return inp;
  }
  function button(host, label, fn, pressed) {
    var b = el("button", { class: "dbtn", type: "button", text: label });
    if (pressed != null) b.setAttribute("aria-pressed", String(pressed));
    b.addEventListener("click", fn);
    host.appendChild(b);
    return b;
  }
  function softmax(a) {
    var m = Math.max.apply(null, a), e = a.map(function (x) { return Math.exp(x - m); }), s = e.reduce(function (p, q) { return p + q; }, 0);
    return e.map(function (x) { return x / s; });
  }
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }

  var DEMOS = {};

  /* ---------- 1. Architecture (Figure 1) ---------- */
  DEMOS.architecture = function (host) {
    var B = {
      ein: ["Input embedding + positional encoding", "io", "Each source token becomes a 512-number vector (§3.4), and a sinusoidal position vector is added so the model knows word order (§3.5)."],
      esa: ["Multi-head self-attention", "attn", "Every input position attends to every input position, in 8 parallel heads of 64 dimensions each (§3.2.2). This is where words gather context from the whole sentence."],
      en1: ["Add & LayerNorm", "norm", "Residual connection plus layer normalisation: LayerNorm(x + Sublayer(x)). Keeps the deep stack trainable (§3.1)."],
      eff: ["Feed-forward network", "", "The same two-layer network (512 → 2048 → 512, ReLU) applied to each position separately (§3.3)."],
      en2: ["Add & LayerNorm", "norm", "Another residual connection and normalisation around the feed-forward layer (§3.1)."],
      din: ["Output embedding (shifted right) + positional encoding", "io", "The target sentence so far, shifted one position right so position i only sees earlier tokens. Same embeddings as the input, with shared weights (§3.4)."],
      dsa: ["Masked multi-head self-attention", "attn", "Like encoder self-attention, but scores for future positions are set to −∞ before the softmax, so the decoder cannot peek ahead (§3.2.3)."],
      dn1: ["Add & LayerNorm", "norm", "Residual connection and normalisation (§3.1)."],
      dxa: ["Encoder–decoder attention", "attn", "Queries come from the decoder; keys and values come from the encoder's final output. This is how each output position reads the source sentence (§3.2.3)."],
      dn2: ["Add & LayerNorm", "norm", "Residual connection and normalisation (§3.1)."],
      dff: ["Feed-forward network", "", "Position-wise feed-forward layer, as in the encoder (§3.3)."],
      dn3: ["Add & LayerNorm", "norm", "Residual connection and normalisation (§3.1)."],
      out: ["Linear + softmax → next-token probabilities", "io", "The decoder's final vectors are projected to vocabulary size and turned into probabilities for the next token. The projection shares its weights with the embeddings (§3.4)."],
    };
    var note = el("div", { class: "arch-note", "aria-live": "polite" });
    var buttons = [];
    function col(label, ids, extra) {
      var c = el("div", { class: "arch-col" });
      ids.forEach(function (id) {
        var b = el("button", { class: "arch-block" + (B[id][1] ? " k-" + B[id][1] : ""), type: "button", text: B[id][0], "aria-pressed": "false" });
        b.addEventListener("click", function () { pick(id); });
        buttons.push([id, b]);
        c.appendChild(b);
      });
      if (extra) c.appendChild(extra);
      c.appendChild(el("p", { class: "arch-lab", text: label }));
      return c;
    }
    function pick(id) {
      buttons.forEach(function (p) { p[1].setAttribute("aria-pressed", String(p[0] === id)); });
      note.innerHTML = "<b>" + B[id][0] + ".</b> " + B[id][2];
    }
    var enc = col("Encoder", ["ein", "esa", "en1", "eff", "en2"], el("p", { class: "arch-x", text: "× 6 layers" }));
    var dec = col("Decoder", ["din", "dsa", "dn1", "dxa", "dn2", "dff", "dn3", "out"], el("p", { class: "arch-x", text: "× 6 layers (except the top block)" }));
    host.appendChild(el("div", { class: "arch" }, [enc, dec]));
    host.appendChild(note);
    pick("esa");
  };

  /* ---------- 2. Attention worked example ---------- */
  DEMOS.attention = function (host) {
    var W = ["the", "animal", "street"], K = [[0, 1], [2, 0], [1, 1]], V = [[1, 0], [0, 1], [1, 1]];
    var ang = 0, len = 1, scale = true, res;
    function compute() {
      var q = [len * Math.cos(ang * Math.PI / 180), len * Math.sin(ang * Math.PI / 180)];
      var s = K.map(function (k) { return q[0] * k[0] + q[1] * k[1]; });
      var sc = s.map(function (x) { return scale ? x / Math.SQRT2 : x; });
      var w = softmax(sc);
      var o = [0, 1].map(function (j) { return w.reduce(function (a, wi, i) { return a + wi * V[i][j]; }, 0); });
      res = { q: q, s: s, sc: sc, w: w, o: o };
    }
    compute();
    var cv = canvas(host, function (w) { return w < 560 ? 430 : 270; }, function (ctx, w, h, c) {
      var narrow = w < 560, S = narrow ? Math.min(w, 250) : Math.min(h, w * 0.42), ox = narrow ? (w - S) / 2 : 0;
      var cx = ox + S / 2, cy = S / 2, u = S / 6.2;
      ctx.strokeStyle = c.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ox + 6, cy); ctx.lineTo(ox + S - 6, cy); ctx.moveTo(cx, 6); ctx.lineTo(cx, S - 6); ctx.stroke();
      function arrow(x, y, col, lw, label, flip) {
        var X = cx + x * u, Y = cy - y * u, a = Math.atan2(Y - cy, X - cx);
        ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(X, Y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X - 9 * Math.cos(a - 0.4), Y - 9 * Math.sin(a - 0.4)); ctx.lineTo(X - 9 * Math.cos(a + 0.4), Y - 9 * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill();
        if (!label) return;
        // label just beyond the tip, pushed sideways so it does not sit on another arrow
        var side = flip ? 1 : -1, lx = X + 12 * Math.cos(a) + side * 12 * Math.sin(a), ly = Y + 12 * Math.sin(a) - side * 12 * Math.cos(a);
        ctx.font = SANS; ctx.textBaseline = "middle"; ctx.textAlign = Math.cos(a) > 0.3 ? "left" : Math.cos(a) < -0.3 ? "right" : "center";
        ctx.fillText(label, lx, ly); ctx.textBaseline = "alphabetic";
      }
      K.forEach(function (k, i) { arrow(k[0], k[1], c.gold, 2, W[i], false); });
      arrow(res.q[0], res.q[1], c.accent, 3, "it (query)", true);
      ctx.fillStyle = c.ink; ctx.beginPath(); ctx.arc(cx + res.o[0] * u, cy - res.o[1] * u, 5, 0, 7); ctx.fill();
      ctx.font = MONO; ctx.textAlign = "left"; ctx.fillStyle = c.muted; ctx.fillText("gold: keys · red: query · ● output", ox + 4, S - 6);
      // bars
      var bx = narrow ? 8 : S + 32, by = narrow ? S + 22 : 18, bw = w - bx - 8, rowH = narrow ? 52 : (h - 36) / 3;
      ctx.font = MONO; ctx.fillStyle = c.muted; ctx.fillText("attention weight from \"it\"", bx, by - 4);
      W.forEach(function (word, i) {
        var y = by + 8 + i * rowH;
        ctx.fillStyle = c.ink; ctx.font = SANS; ctx.textAlign = "left"; ctx.fillText(word, bx, y + 14);
        ctx.fillStyle = c.bg2; ctx.fillRect(bx + 70, y + 2, bw - 130, 16);
        ctx.fillStyle = c.accent; ctx.fillRect(bx + 70, y + 2, (bw - 130) * res.w[i], 16);
        ctx.font = MONO; ctx.fillStyle = c.ink; ctx.textAlign = "right"; ctx.fillText((res.w[i] * 100).toFixed(0) + "%", bx + bw, y + 15);
        ctx.fillStyle = c.muted; ctx.textAlign = "left";
        ctx.fillText("score " + res.s[i].toFixed(2) + (scale ? " → ÷√2 = " + res.sc[i].toFixed(2) : ""), bx + 70, y + 34);
      });
    });
    var ctrl = el("div", { class: "demo-ctrl" });
    host.appendChild(ctrl);
    var out = el("p", { class: "demo-out" });
    slider(ctrl, { label: "Query direction", min: -180, max: 180, value: 0, fmt: function (v) { return v + "°"; }, on: function (v) { ang = v; upd(); } });
    slider(ctrl, { label: "Query length", min: 0.25, max: 3, step: 0.05, value: 1, fmt: function (v) { return v.toFixed(2); }, on: function (v) { len = v; upd(); } });
    var sb = button(ctrl, "Scale by √d_k", function () { scale = !scale; sb.setAttribute("aria-pressed", String(scale)); upd(); }, true);
    host.appendChild(out);
    function upd() {
      compute(); cv.redraw();
      var top = res.w.indexOf(Math.max.apply(null, res.w));
      out.innerHTML = "\"it\" attends mostly to <b class=\"hl\">" + W[top] + "</b> (" + (res.w[top] * 100).toFixed(0) + "%). Output vector = Σ weight × value = <b>(" + res.o[0].toFixed(2) + ", " + res.o[1].toFixed(2) + ")</b>. A longer query makes the scores larger and the weights sharper.";
    }
    upd();
  };

  /* ---------- 3. Why scale by √d_k ---------- */
  DEMOS.scaling = function (host) {
    var DK = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], di = 6, seed = 3, raw, wRaw, wSc;
    function compute() {
      var r = rng(seed * 7919 + DK[di]), d = DK[di];
      var q = []; for (var j = 0; j < d; j++) q.push(gauss(r));
      raw = [];
      for (var i = 0; i < 8; i++) { var s = 0; for (j = 0; j < d; j++) s += q[j] * gauss(r); raw.push(s); }
      wRaw = softmax(raw);
      wSc = softmax(raw.map(function (x) { return x / Math.sqrt(d); }));
    }
    compute();
    var cv = canvas(host, function (w) { return w < 560 ? 250 : 220; }, function (ctx, w, h, c) {
      var rows = [["Without scaling: softmax(q·k)", wRaw, c.muted], ["With scaling: softmax(q·k / √d_k)", wSc, c.accent]];
      var top = 16, rh = (h - top - 6) / 2, gap = 6, bw = (w - gap * 7) / 8;
      rows.forEach(function (row, ri) {
        var y0 = top + ri * rh;
        ctx.font = MONO; ctx.fillStyle = c.ink2; ctx.textAlign = "left"; ctx.fillText(row[0], 0, y0 + 2);
        var base = y0 + rh - 22, maxH = rh - 44;
        for (var i = 0; i < 8; i++) {
          var x = i * (bw + gap), v = row[1][i];
          ctx.fillStyle = c.bg2; ctx.fillRect(x, base - maxH, bw, maxH);
          ctx.fillStyle = row[2]; ctx.fillRect(x, base - maxH * v, bw, maxH * v);
          ctx.fillStyle = c.muted; ctx.textAlign = "center"; ctx.fillText((v * 100).toFixed(0) + "%", x + bw / 2, base + 14);
        }
      });
    });
    var ctrl = el("div", { class: "demo-ctrl" });
    host.appendChild(ctrl);
    var out = el("p", { class: "demo-out" });
    slider(ctrl, { label: "d_k", min: 0, max: DK.length - 1, value: di, fmt: function (v) { return String(DK[v]); }, on: function (v) { di = v; upd(); } });
    button(ctrl, "New random query and keys", function () { seed++; upd(); });
    host.appendChild(out);
    function upd() {
      compute(); cv.redraw();
      var sd = Math.sqrt(raw.reduce(function (a, x) { return a + x * x; }, 0) / raw.length);
      out.innerHTML = "d_k = <b>" + DK[di] + "</b>: raw scores spread with std ≈ <b>" + sd.toFixed(1) + "</b> (theory: √d_k = " + Math.sqrt(DK[di]).toFixed(1) + "). Largest weight unscaled <b>" + (Math.max.apply(null, wRaw) * 100).toFixed(0) + "%</b>, scaled <b class=\"hl\">" + (Math.max.apply(null, wSc) * 100).toFixed(0) + "%</b>. A near-one-hot softmax has almost no gradient, so training stalls.";
    }
    upd();
  };

  /* ---------- 4. Positional encoding ---------- */
  DEMOS.positional = function (host) {
    var P = 64, D = 128, row = 10;
    function pe(pos, i) { var k = Math.floor(i / 2), a = pos / Math.pow(10000, (2 * k) / D); return i % 2 ? Math.cos(a) : Math.sin(a); }
    function mix(a, b, t) {
      var pa = parse(a), pb = parse(b);
      return "rgb(" + [0, 1, 2].map(function (j) { return Math.round(pa[j] + (pb[j] - pa[j]) * t); }).join(",") + ")";
    }
    function parse(col) {
      if (col[0] === "#") { var n = parseInt(col.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
      var m = col.match(/\d+(\.\d+)?/g); return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
    }
    var geom;
    var cv = canvas(host, function (w) { return w < 560 ? 330 : 300; }, function (ctx, w, h, c) {
      var top = 18, hh = h - 120, cw = (w - 40) / D, ch = hh / P;
      geom = { top: top, ch: ch, left: 40 };
      for (var p = 0; p < P; p++) for (var i = 0; i < D; i++) {
        var v = pe(p, i);
        ctx.fillStyle = v >= 0 ? mix(c.surface, c.accent, v) : mix(c.surface, c.cool, -v);
        ctx.fillRect(40 + i * cw, top + p * ch, Math.ceil(cw), Math.ceil(ch));
      }
      ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5; ctx.strokeRect(40, top + row * ch, w - 40, ch);
      ctx.font = MONO; ctx.fillStyle = c.muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      [0, 16, 32, 48, 63].forEach(function (p) { ctx.fillText(p, 34, top + p * ch + ch / 2); });
      ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      ctx.fillText("position ↓   dimension →", 40, 12);
      // waveform of the selected row
      var y0 = top + hh + 26, gh = 70, mid = y0 + gh / 2;
      ctx.strokeStyle = c.line; ctx.beginPath(); ctx.moveTo(40, mid); ctx.lineTo(w, mid); ctx.stroke();
      [[0, c.accent], [1, c.cool]].forEach(function (pair) {
        ctx.strokeStyle = pair[1]; ctx.lineWidth = 1.8; ctx.beginPath();
        for (var j = pair[0]; j < D; j += 2) { var x = 40 + (j + 0.5) * cw, y = mid - pe(row, j) * (gh / 2 - 4); j < 2 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
        ctx.stroke();
      });
      ctx.fillStyle = c.muted; ctx.fillText("position " + row + ": sine dimensions (red) and cosine dimensions (blue), fast to slow", 40, y0 - 6);
    });
    var out = el("p", { class: "demo-out" });
    host.appendChild(out);
    function pick(e) {
      var r = cv.c.getBoundingClientRect(), y = e.clientY - r.top;
      var p = Math.floor((y - geom.top) / geom.ch);
      if (p >= 0 && p < P && p !== row) { row = p; cv.redraw(); upd(); }
    }
    cv.c.addEventListener("pointermove", pick);
    cv.c.addEventListener("pointerdown", pick);
    function upd() {
      var fast = 2 * Math.PI, slow = 2 * Math.PI * Math.pow(10000, 126 / D);
      out.innerHTML = "Position <b>" + row + "</b>: dims 0–1 = (" + pe(row, 0).toFixed(2) + ", " + pe(row, 1).toFixed(2) + "), dims 126–127 = (" + pe(row, 126).toFixed(2) + ", " + pe(row, 127).toFixed(2) + "). The first pair repeats every " + fast.toFixed(1) + " positions; the last pair every " + Math.round(slow).toLocaleString() + ". Every position gets a unique pattern.";
    }
    upd();
  };

  /* ---------- 5. Learning-rate schedule ---------- */
  DEMOS.schedule = function (host) {
    var warm = 4000, dm = 512, STEPS = 100000;
    function lr(s, w, d) { s = Math.max(s, 1); return Math.pow(d, -0.5) * Math.min(Math.pow(s, -0.5), s * Math.pow(w, -1.5)); }
    var cv = canvas(host, function (w) { return w < 560 ? 230 : 240; }, function (ctx, w, h, c) {
      var ox = 58, oy = 10, pw = w - ox - 10, ph = h - oy - 34;
      var ymax = Math.max(lr(4000, 4000, 512), lr(warm, warm, dm)) * 1.12;
      var X = function (s) { return ox + (s / STEPS) * pw; }, Y = function (v) { return oy + ph - (v / ymax) * ph; };
      ctx.strokeStyle = c.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox, oy + ph); ctx.lineTo(ox + pw, oy + ph); ctx.stroke();
      ctx.font = MONO; ctx.fillStyle = c.muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      [0, 0.5, 1].forEach(function (f) { ctx.fillText((ymax * f / 1.12 * 1e4).toFixed(1) + "e-4", ox - 6, Y(ymax * f / 1.12)); });
      ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
      [0, 25000, 50000, 75000, 100000].forEach(function (s) { ctx.fillText(s / 1000 + "k", X(s), oy + ph + 16); });
      ctx.fillText("training step", ox + pw / 2, h - 2);
      function curve(w2, d2, col, lw) {
        ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
        for (var i = 0; i <= 400; i++) { var s = (i / 400) * STEPS, y = Y(lr(s, w2, d2)); i ? ctx.lineTo(X(s), y) : ctx.moveTo(X(s), y); }
        ctx.stroke();
      }
      curve(4000, 512, c.lineS, 1.5);
      curve(warm, dm, c.accent, 2.5);
      ctx.fillStyle = c.accent; ctx.beginPath(); ctx.arc(X(warm), Y(lr(warm, warm, dm)), 4.5, 0, 7); ctx.fill();
      ctx.textAlign = "left"; ctx.fillStyle = c.muted; ctx.fillText("grey: the paper's setting (warm-up 4,000, d_model 512)", ox + 10, oy + 12);
    });
    var ctrl = el("div", { class: "demo-ctrl" });
    host.appendChild(ctrl);
    var out = el("p", { class: "demo-out" });
    slider(ctrl, { label: "Warm-up steps", min: 500, max: 16000, step: 500, value: warm, fmt: function (v) { return v.toLocaleString(); }, on: function (v) { warm = v; upd(); } });
    slider(ctrl, { label: "d_model", min: 0, max: 2, value: 1, fmt: function (v) { return String([256, 512, 1024][v]); }, on: function (v) { dm = [256, 512, 1024][v]; upd(); } });
    host.appendChild(out);
    function upd() {
      cv.redraw();
      out.innerHTML = "Peak learning rate <b class=\"hl\">" + lr(warm, warm, dm).toExponential(2) + "</b> at step " + warm.toLocaleString() + ", falling to <b>" + lr(STEPS, warm, dm).toExponential(2) + "</b> by step 100,000. After warm-up, the rate decays like 1/√step whatever the warm-up length.";
    }
    upd();
  };

  /* =================== GenRec (Netflix) =================== */

  /* ---------- 6. Verbalization and context engineering ---------- */
  DEMOS.verbalizer = function (host) {
    // Illustrative history, oldest first. "old" = before the recent window; "low" = low-signal event.
    var EV = [
      { d: "01-10", day: "January 10", dev: "TV", t: "Mindhunter", ep: [1, 1], min: 58, thumb: true, old: true },
      { d: "01-12", day: "January 12", dev: "TV", t: "The Crown", ep: [1, 1], min: 4, old: true, low: true },
      { d: "01-20", day: "January 20", dev: "iPad", t: "Ozark", ep: [1, 1], min: 60, old: true },
      { d: "02-03", day: "February 3", dev: "TV", t: "Narcos", ep: [1, 1], min: 49, binge: 1 },
      { d: "02-03", day: "February 3", dev: "TV", t: "Narcos", ep: [1, 2], min: 52, binge: 1 },
      { d: "02-03", day: "February 3", dev: "TV", t: "Narcos", ep: [1, 3], min: 50, binge: 1 },
      { d: "02-04", day: "February 4", dev: "TV", t: "Narcos", ep: [1, 4], min: 47, binge: 1 },
      { d: "02-04", day: "February 4", dev: "TV", t: "Narcos", ep: [1, 5], min: 51, binge: 1 },
      { d: "02-04", day: "February 4", dev: "TV", t: "Narcos", ep: [1, 6], min: 53, binge: 1 },
      { d: "02-10", day: "February 10", dev: "phone", t: "Bridgerton", kind: "trailer", min: 1, low: true },
      { d: "02-18", day: "February 18", dev: "TV", t: "Dark", ep: [1, 1], min: 55, thumb: true },
      { d: "03-01", day: "March 1", dev: "TV", t: "Glass Onion", movie: true, min: 139 },
      { d: "03-02", day: "March 2", dev: "phone", t: "Squid Game", kind: "click", min: 0, low: true },
      { d: "03-05", day: "March 5", dev: "TV", t: "Wednesday", kind: "list" },
      { d: "03-06", day: "March 6", dev: "TV", t: "Stranger Things", ep: [4, 1], min: 78 },
      { d: "03-07", day: "March 7", dev: "TV", t: "Stranger Things", ep: [4, 2], min: 75, thumb: true },
    ];
    var R = [
      ["omit", "Omit low-signal events"], ["binge", "Compress binges"], ["old", "Summarize old history"],
      ["compact", "Compact wording"], ["shots", "Drop few-shot examples"], ["elab", "Elaborate a new release"],
    ];
    var on = {};
    function line(e, c) {
      if (c) {
        var what = e.kind === "trailer" ? "trailer" : e.kind === "click" ? "click" : e.kind === "list" ? "add-to-list" : "play " + e.min + "m";
        return e.d + " " + e.dev + " " + e.t + (e.ep ? " S" + e.ep[0] + "E" + e.ep[1] : "") + " " + what + (e.thumb ? " +thumb" : "");
      }
      var head = "On " + e.day + ", on their " + e.dev + ", the member ";
      if (e.kind === "trailer") return head + "watched the trailer for " + e.t + " for about a minute.";
      if (e.kind === "click") return head + "clicked on " + e.t + " while browsing but did not play it.";
      if (e.kind === "list") return head + "added " + e.t + " to My List.";
      return head + "played " + (e.movie ? "the movie " : "the series ") + e.t + (e.ep ? " (season " + e.ep[0] + ", episode " + e.ep[1] + ")" : "") +
        " and watched for " + e.min + " minutes" + (e.thumb ? ", then gave it a thumbs up." : ".");
    }
    function build(o) {
      var c = o.compact, L = [];
      L.push(c ? "ctx: Fri 03-08 21:00 TV US member-since=2019" : "Context: It is Friday, March 8, at 9 PM. The member is watching on a TV in the United States and has been a member since 2019.");
      if (!o.shots) {
        L.push(c ? "examples:" : "Here are two examples of the task:");
        L.push(c ? "  Ozark S1 +thumb, Narcos S1 -> next: Breaking Bad" : "  Example 1: A member who finished Ozark season 1 and gave it a thumbs up, then watched Narcos, next played Breaking Bad.");
        L.push(c ? "  Dark S1, Black Mirror -> next: Russian Doll" : "  Example 2: A member who watched Dark season 1 and two episodes of Black Mirror next played Russian Doll.");
      }
      L.push(c ? "history:" : "Here is this member's viewing history, oldest first:");
      var ev = EV.filter(function (e) { return !(o.omit && e.low) && !(o.old && e.old); });
      if (o.old) L.push(c ? "  earlier: crime dramas (Mindhunter +thumb, Ozark)" : "  Earlier history, summarized: mostly crime dramas, including Mindhunter (thumbs up) and Ozark.");
      var done = false;
      ev.forEach(function (e) {
        if (o.binge && e.binge) {
          if (done) return;
          done = true;
          L.push(c ? "  02-03..04 TV Narcos S1E1-6 binge ~5h" : "  On February 3 and 4, on their TV, the member binge-watched Narcos season 1, episodes 1 to 6, about 5 hours in total.");
          return;
        }
        L.push("  " + line(e, c));
      });
      if (o.elab) L.push(c ? "new: 3 Body Problem | series | sci-fi drama | aliens, physicists, a VR game | like: Dark" :
        "New on Netflix: 3 Body Problem, a sci-fi drama series. A group of physicists uncover an alien threat hidden in a mysterious virtual-reality game. Members who liked Dark often enjoy it.");
      L.push(c ? "task: rank next plays" : "Task: Rank the titles in the catalog by how likely this member is to play them next.");
      return L.join("\n");
    }
    var tok = function (s) { return Math.ceil(s.length / 4); };
    var BASE = tok(build({}));
    var ctrl = el("div", { class: "demo-ctrl" });
    host.appendChild(ctrl);
    var btns = R.map(function (r) {
      var b = button(ctrl, r[1], function () { on[r[0]] = !on[r[0]]; upd(); }, false);
      return [r[0], b];
    });
    var meter = el("div", { class: "vz-meter" }, [el("span")]);
    var pre = el("pre", { class: "vz-pre", tabindex: "0", "aria-label": "Verbalized prompt" });
    var out = el("p", { class: "demo-out", "aria-live": "polite" });
    host.insertBefore(pre, ctrl);
    host.insertBefore(meter, ctrl);
    host.appendChild(out);
    var all = el("div", { class: "demo-ctrl" });
    host.appendChild(all);
    button(all, "Apply the paper's recipe", function () { on = { omit: 1, binge: 1, old: 1, compact: 1, shots: 1 }; upd(); });
    button(all, "Reset", function () { on = {}; upd(); });
    function upd() {
      btns.forEach(function (p) { p[1].setAttribute("aria-pressed", String(!!on[p[0]])); });
      var txt = build(on), t = tok(txt), f = t / BASE;
      pre.textContent = txt;
      meter.firstChild.style.width = Math.min(100, f * 100) + "%";
      out.innerHTML = "≈ <b class=\"hl\">" + t + " tokens</b>, <b>" + Math.round(f * 100) + "%</b> of the original " + BASE + " (rough count: 4 characters per token). " +
        (f < 0.45 ? "This toy history shrinks further than real ones do; the paper's compaction went from about 5,000 to 1,700 tokens, a third of the cost, with negligible quality loss." : on.elab ? "Elaborating spends tokens on purpose, on an item the model knows little about." : "Cost is roughly proportional to this number.");
    }
    upd();
  };

  /* ---------- 7. Catalog-aware scoring head ---------- */
  DEMOS["catalog-head"] = function (host) {
    var T = [["Stranger Things", 32, 1, "Stranger"], ["Dark", 52, 0.95], ["3 Body Problem", 68, 0.9, "3 Body"], ["Umbrella Academy", 18, 0.85, "Umbrella"], ["Wednesday", 4, 0.8],
      ["Squid Game", 112, 0.75], ["Narcos", 158, 0.9], ["Ozark", 176, 0.95], ["Mindhunter", 196, 0.85], ["Glass Onion", 248, 0.7],
      ["Bridgerton", 298, 0.8], ["The Crown", 322, 0.85]];
    var ang = 40, len = 3, pos = 1, wgt = 1, res;
    function compute() {
      var s = T.map(function (t) { return len * t[2] * Math.cos((ang - t[1]) * Math.PI / 180); });
      var p = softmax(s), order = T.map(function (_, i) { return i; }).sort(function (a, b) { return s[b] - s[a]; });
      res = { s: s, p: p, order: order, rank: order.indexOf(pos) + 1 };
    }
    compute();
    var cv = canvas(host, function (w) { return w < 560 ? 520 : 300; }, function (ctx, w, h, c) {
      // titles sit outside the dots, so keep ~80px on each side of the circle for labels
      var narrow = w < 560, PW = narrow ? w : w * 0.5, u = Math.min(PW / 2 - 82, narrow ? 100 : h / 2 - 26), S = narrow ? 2 * u + 56 : h;
      var ox = 0, cx = PW / 2, cy = S / 2 - (narrow ? 4 : 8);
      ctx.strokeStyle = c.line; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, u, 0, 7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - u - 12, cy); ctx.lineTo(cx + u + 12, cy); ctx.moveTo(cx, cy - u - 12); ctx.lineTo(cx, cy + u + 12); ctx.stroke();
      ctx.font = MONO;
      T.forEach(function (t, i) {
        var a = t[1] * Math.PI / 180, x = cx + Math.cos(a) * t[2] * u, y = cy - Math.sin(a) * t[2] * u;
        ctx.fillStyle = i === pos ? c.accent : c.gold;
        ctx.beginPath(); ctx.arc(x, y, i === pos ? 6 : 4.5, 0, 7); ctx.fill();
        ctx.fillStyle = i === pos ? c.accent : c.ink2;
        ctx.textAlign = Math.cos(a) > 0.2 ? "left" : Math.cos(a) < -0.2 ? "right" : "center";
        ctx.textBaseline = "middle";
        ctx.fillText(narrow && t[3] ? t[3] : t[0], x + Math.cos(a) * 10, y - Math.sin(a) * 10 + (Math.abs(Math.cos(a)) <= 0.2 ? -Math.sign(Math.sin(a)) * 4 : 0));
      });
      var a = ang * Math.PI / 180, L = Math.min(len / 3, 1.7) * u * 0.9, X = cx + Math.cos(a) * L, Y = cy - Math.sin(a) * L;
      ctx.strokeStyle = c.cool; ctx.fillStyle = c.cool; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(X, Y); ctx.stroke();
      var t2 = Math.atan2(Y - cy, X - cx);
      ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X - 10 * Math.cos(t2 - 0.4), Y - 10 * Math.sin(t2 - 0.4)); ctx.lineTo(X - 10 * Math.cos(t2 + 0.4), Y - 10 * Math.sin(t2 + 0.4)); ctx.closePath(); ctx.fill();
      ctx.textBaseline = "alphabetic"; ctx.textAlign = "left"; ctx.fillStyle = c.muted;
      ctx.textAlign = "center";
      ctx.fillText(narrow ? "blue: h · dots: item embeddings e_i" : "blue: member vector h · dots: item embeddings e_i", cx, S - 6);
      ctx.textAlign = "left";
      // ranked list
      var bx = narrow ? 4 : PW + 12, by = narrow ? S + 22 : 16, bw = w - bx - 4, n = 8, rh = ((narrow ? h - by : h - by) - 6) / n;
      ctx.fillStyle = c.muted; ctx.fillText("rank · title · p = softmax(h · e_i)", bx, by - 2);
      res.order.slice(0, n).forEach(function (i, k) {
        var y = by + 8 + k * rh, lab = (k + 1) + ". " + T[i][0], bar0 = bx + Math.min(150, bw * 0.45), barW = bw - (bar0 - bx) - 46;
        ctx.font = SANS; ctx.fillStyle = i === pos ? c.accent : c.ink; ctx.fillText(lab, bx, y + 13);
        ctx.fillStyle = c.bg2; ctx.fillRect(bar0, y + 2, barW, 14);
        ctx.fillStyle = i === pos ? c.accent : c.lineS; ctx.fillRect(bar0, y + 2, barW * res.p[i], 14);
        ctx.font = MONO; ctx.fillStyle = c.ink; ctx.textAlign = "right"; ctx.fillText(res.p[i].toFixed(2), bx + bw, y + 14); ctx.textAlign = "left";
      });
    });
    var ctrl = el("div", { class: "demo-ctrl" });
    host.appendChild(ctrl);
    slider(ctrl, { label: "Direction of h", min: 0, max: 359, value: ang, fmt: function (v) { return v + "°"; }, on: function (v) { ang = v; upd(); } });
    slider(ctrl, { label: "Length of h", min: 0.5, max: 6, step: 0.1, value: len, fmt: function (v) { return v.toFixed(1); }, on: function (v) { len = v; upd(); } });
    var id = "d" + Math.random().toString(36).slice(2, 8), sel = el("select", { id: id });
    T.forEach(function (t, i) { var o = el("option", { value: i, text: t[0] }); if (i === pos) o.selected = true; sel.appendChild(o); });
    sel.addEventListener("change", function () { pos = +sel.value; upd(); });
    ctrl.appendChild(el("label", { for: id }, [document.createTextNode("Member actually played"), sel]));
    slider(ctrl, { label: "Reward weight w", min: 0, max: 2, step: 0.1, value: wgt, fmt: function (v) { return v.toFixed(1); }, on: function (v) { wgt = v; upd(); } });
    var out = el("p", { class: "demo-out", "aria-live": "polite" });
    host.appendChild(out);
    function upd() {
      compute(); cv.redraw();
      var ce = -Math.log(res.p[pos]);
      out.innerHTML = "<b>" + T[pos][0] + "</b> is ranked <b class=\"hl\">#" + res.rank + "</b> with p = <b>" + res.p[pos].toFixed(3) + "</b>. Cross-entropy −ln p = <b>" + ce.toFixed(2) +
        "</b>; × reward weight " + wgt.toFixed(1) + " = <b class=\"hl\">" + (wgt * ce).toFixed(2) + "</b>. Reciprocal rank 1/" + res.rank + " = " + (1 / res.rank).toFixed(2) +
        ". A longer h makes the softmax sharper; training moves h and the e_i so real plays score high.";
    }
    upd();
  };

  /* ---------- 8. The GenRec system (Figures 1 and 2) ---------- */
  DEMOS["genrec-pipeline"] = function (host) {
    var B = {
      logs: ["Raw logs: history, item metadata, context", "io", "Plays, durations, thumbs, list adds, the catalog's metadata (title, genre, synopsis, release date) and the request context (time, device, surface, locale). This is the same information the old ranker turned into thousands of features (§3, Figure 1)."],
      verb: ["Verbalizer V: write it as text", "", "x = V(H, {M_i}, τ). Context engineering decides what goes in: keep high-signal events in full, omit noisy ones, compress binges, summarize old history, elaborate on new releases (§4.3)."],
      tok: ["Tokenizer", "norm", "The text becomes about 1,700 tokens after compaction, down from about 5,000 (§5.4)."],
      llm: ["Decoder-only LLM (Netflix foundation model)", "attn", "The Phase-1 foundation LLM, post-trained in Phase 2. Served on vLLM in prefill-only mode: the prompt is read once, in parallel, with no token-by-token decoding (§4.5, §4.7)."],
      pool: ["Pooled hidden state h", "norm", "The hidden state at a pooling position: a d-dimensional summary of the member's preferences and context (§4.5)."],
      head: ["Catalog-aware scoring head φ(h, e_i)", "attn", "Combines h with a learned embedding e_i for every catalog item. Only catalog items have embeddings, so the model cannot recommend a title Netflix doesn't have (§4.5)."],
      rank: ["Softmax → scores → ranking π", "io", "Scores for the whole catalog (or a candidate set) from one forward pass; sorting them gives the ranking. Figure 1's illustrative output: Umbrella Academy 0.54, Dark 0.32, Breaking Bad 0.10."],
      p1: ["Phase 1: open-source LLM + Netflix data", "io", "Infrequent, broad training that builds a Netflix-aware foundation model with world knowledge, personalization and content understanding. Worth +10–20% MRR over an off-the-shelf base (§4.1, §5.3)."],
      conv: ["Histories as conversations", "", "Each example is a user message (context, profile, history, item details, task) and an assistant message: what the member actually did next (§4.2)."],
      rl: ["Ranking loss: cross-entropy over the catalog", "attn", "Positives are high-value engagements (long plays, strong explicit feedback), with thresholds per content type. Sampled softmax can stand in for huge catalogs (§4.4, §4.5)."],
      lm: ["Language-modeling loss", "", "Next-token prediction on the verbalized text keeps language skills for reading histories and for prompt-based steering. Not used at inference (§4.4)."],
      rw: ["Reward weights w", "attn", "Separate reward models score each engagement for links to long-term satisfaction and for content-type balance; each example's ranking loss is scaled by its weight. Chosen over GRPO for simplicity and cost (§4.6)."],
      tot: ["L = αL_rank + βL_lang + γL_misc", "norm", "Weights sum to one and are tuned offline. Phase 2 is re-run often: it adds +35–50% over a fresh Phase-1 model, about +80% two weeks later (§4.4, §5.3)."],
    };
    var note = el("div", { class: "arch-note", "aria-live": "polite" }), buttons = [];
    function col(label, ids) {
      var c = el("div", { class: "arch-col arch-down" }, [el("p", { class: "arch-lab", text: label })]);
      ids.forEach(function (id, k) {
        var b = el("button", { class: "arch-block" + (B[id][1] ? " k-" + B[id][1] : ""), type: "button", text: B[id][0], "aria-pressed": "false" });
        b.addEventListener("click", function () { pick(id); });
        buttons.push([id, b]);
        c.appendChild(b);
        if (k < ids.length - 1) c.appendChild(el("span", { class: "arch-arrow", "aria-hidden": "true", text: "↓" }));
      });
      return c;
    }
    function pick(id) {
      buttons.forEach(function (p) { p[1].setAttribute("aria-pressed", String(p[0] === id)); });
      note.innerHTML = "<b>" + B[id][0] + ".</b> " + B[id][2];
    }
    host.appendChild(el("div", { class: "arch" }, [
      col("Serving: every request", ["logs", "verb", "tok", "llm", "pool", "head", "rank"]),
      col("Training: Phase 2 (after Phase 1)", ["p1", "conv", "rl", "lm", "rw", "tot"]),
    ]));
    host.appendChild(note);
    pick("head");
  };

  /* ---------- 9. Serving cost: prefill-only vs generation ---------- */
  DEMOS["serving-cost"] = function (host) {
    var SIZES = [1, 3, 10], CANDS = [10, 50, 100, 500, 1000, 5000], TPT = 6;
    var si = 2, T = 5000, ci = 4, REF = 2 * 10e9 * 5000;
    function calc() {
      var N = SIZES[si] * 1e9, M = CANDS[ci];
      return { pre: 2 * N * T / REF, gen: 2 * N * (T + M * TPT) / REF, steps: 1 + M * TPT, M: M };
    }
    var cv = canvas(host, function (w) { return w < 560 ? 200 : 170; }, function (ctx, w, h, c) {
      var r = calc(), lab = w < 560 ? 0 : 170, x0 = lab, bw = w - x0 - 64, max = Math.max(r.gen, 1.2);
      var rows = [["Prefill-only (GenRec)", r.pre, c.accent, "1 forward pass"],
        ["Generate titles", r.gen, c.lineS, r.steps.toLocaleString() + " sequential passes"]];
      ctx.font = MONO; ctx.fillStyle = c.muted; ctx.textAlign = "left";
      ctx.fillText(w < 560 ? "compute per request (dashed = 1.00×)" : "compute per request; dashed line = 10B params × 5,000-token prefill (1.00×)", 0, 12);
      rows.forEach(function (row, k) {
        var y = (w < 560 ? 34 : 34) + k * (w < 560 ? 76 : 60);
        ctx.font = SANS; ctx.fillStyle = c.ink; ctx.textAlign = "left";
        if (lab) ctx.fillText(row[0], 0, y + 15); else ctx.fillText(row[0], 0, y);
        var yb = lab ? y : y + 8;
        ctx.fillStyle = c.bg2; ctx.fillRect(x0, yb + 2, bw, 18);
        ctx.fillStyle = row[2]; ctx.fillRect(x0, yb + 2, bw * Math.min(row[1] / max, 1), 18);
        ctx.font = MONO; ctx.fillStyle = c.ink; ctx.textAlign = "right"; ctx.fillText(row[1].toFixed(2) + "×", w, yb + 16);
        ctx.textAlign = "left"; ctx.fillStyle = c.muted; ctx.fillText(row[3], x0, yb + 36);
      });
      var xr = x0 + bw * (1 / max);
      ctx.strokeStyle = c.ink2; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(xr, 22); ctx.lineTo(xr, h - 4); ctx.stroke(); ctx.setLineDash([]);
    });
    var ctrl = el("div", { class: "demo-ctrl" });
    host.appendChild(ctrl);
    slider(ctrl, { label: "Model size", min: 0, max: 2, value: si, fmt: function (v) { return "~" + SIZES[v] + "B"; }, on: function (v) { si = v; upd(); } });
    var ctxIn = slider(ctrl, { label: "Prompt tokens", min: 500, max: 6000, step: 100, value: T, fmt: function (v) { return v.toLocaleString(); }, on: function (v) { T = v; upd(); } });
    slider(ctrl, { label: "Titles to rank", min: 0, max: CANDS.length - 1, value: ci, fmt: function (v) { return CANDS[v].toLocaleString(); }, on: function (v) { ci = v; upd(); } });
    var b2 = el("div", { class: "demo-ctrl" });
    host.appendChild(b2);
    button(b2, "Paper's compaction: 1,700 tokens", function () { T = 1700; ctxIn.value = T; ctxIn.dispatchEvent(new Event("input")); });
    var out = el("p", { class: "demo-out", "aria-live": "polite" });
    host.appendChild(out);
    function upd() {
      cv.redraw();
      var r = calc();
      out.innerHTML = "Prefill-only: <b class=\"hl\">" + r.pre.toFixed(2) + "×</b> the reference compute in one parallel pass. Generating " + r.M.toLocaleString() + " titles at " + TPT +
        " tokens each: <b>" + r.gen.toFixed(2) + "×</b> compute and <b>" + r.steps.toLocaleString() + "</b> passes that must run one after another, so latency grows with every token. Cost ≈ model size × tokens (§4.7).";
    }
    upd();
  };

  /* ---------- 10. Results (Figures 3–5, Table 1) ---------- */
  DEMOS["genrec-results"] = function (host) {
    var tab = 0, hover = -1;
    var DATA = [[1, 1.0], [2, 1.05], [5, 1.108], [10, 1.137], [20, 1.16]];
    var HIST = [["N", 0.921, "−7.9%"], ["2N", 1.0, "chosen"], ["3N", 1.017, "+1.7%"]];
    var TABS = ["Data scaling", "History length", "Phase 1 vs 2", "A/B test"];
    var pts = [];
    var cv = canvas(host, function (w) { return w < 560 ? 280 : 260; }, function (ctx, w, h, c) {
      pts = [];
      var ox = 52, oy = 14, pw = w - ox - 16, ph = h - oy - 40;
      function axes(ylab) {
        ctx.strokeStyle = c.line; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox, oy + ph); ctx.lineTo(ox + pw, oy + ph); ctx.stroke();
        ctx.save(); ctx.translate(12, oy + ph / 2); ctx.rotate(-Math.PI / 2); ctx.font = MONO; ctx.fillStyle = c.muted; ctx.textAlign = "center"; ctx.fillText(ylab, 0, 0); ctx.restore();
      }
      function dot(x, y, i, col) {
        ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, hover === i ? 7 : 5, 0, 7); ctx.fill();
        pts.push([x, y, i]);
      }
      ctx.font = MONO; ctx.textBaseline = "alphabetic";
      if (tab === 0) {
        axes("normalized MRR");
        var X = function (v) { return ox + 20 + (Math.log(v) / Math.log(20)) * (pw - 40); }, Y = function (v) { return oy + ph - ((v - 0.99) / 0.18) * ph; };
        ctx.fillStyle = c.muted; ctx.textAlign = "right";
        [1, 1.05, 1.1, 1.15].forEach(function (v) { ctx.fillText(v.toFixed(2), ox - 6, Y(v) + 4); });
        ctx.textAlign = "center";
        DATA.forEach(function (d) { ctx.fillText(d[0] + "×", X(d[0]), oy + ph + 16); });
        ctx.fillText(w < 560 ? "Phase-2 data (log), ~10B model" : "Phase-2 training data (log scale), ~10B model", ox + pw / 2, h - 4);
        ctx.strokeStyle = c.accent; ctx.lineWidth = 2.5; ctx.beginPath();
        DATA.forEach(function (d, i) { i ? ctx.lineTo(X(d[0]), Y(d[1])) : ctx.moveTo(X(d[0]), Y(d[1])); }); ctx.stroke();
        DATA.forEach(function (d, i) { dot(X(d[0]), Y(d[1]), i, c.accent); });
      } else if (tab === 1) {
        axes("normalized MRR");
        var X1 = function (i) { return ox + 40 + i * (pw - 80) / 2; }, Y1 = function (v) { return oy + ph - ((v - 0.9) / 0.13) * ph; };
        ctx.fillStyle = c.muted; ctx.textAlign = "right";
        [0.9, 0.95, 1.0].forEach(function (v) { ctx.fillText(v.toFixed(2), ox - 6, Y1(v) + 4); });
        ctx.strokeStyle = c.lineS; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(ox, Y1(1)); ctx.lineTo(ox + pw, Y1(1)); ctx.stroke(); ctx.setLineDash([]);
        ctx.textAlign = "center";
        HIST.forEach(function (d, i) { ctx.fillText(d[0], X1(i), oy + ph + 16); });
        ctx.fillText(w < 560 ? "events in the prompt (cost →)" : "engagement events in the prompt (more tokens and cost →)", ox + pw / 2, h - 4);
        ctx.strokeStyle = c.cool; ctx.lineWidth = 2.5; ctx.beginPath();
        HIST.forEach(function (d, i) { i ? ctx.lineTo(X1(i), Y1(d[1])) : ctx.moveTo(X1(i), Y1(d[1])); }); ctx.stroke();
        HIST.forEach(function (d, i) {
          dot(X1(i), Y1(d[1]), i, i === 1 ? c.accent : c.cool);
          ctx.fillStyle = c.ink; ctx.font = SANS; ctx.textAlign = "center"; ctx.fillText(d[2], X1(i), Y1(d[1]) + (i === 0 ? 22 : -14)); ctx.font = MONO;
        });
      } else if (tab === 2) {
        var rows = [["Phase 1 vs off-the-shelf LLM base", 10, 20], ["Phase 2 vs fresh Phase 1 (day 0)", 35, 50], ["Phase 2 vs Phase 1, two weeks later", 78, 82]];
        var lab = w < 560 ? 0 : 250, x0 = lab, bw = w - x0 - 20, Xp = function (v) { return x0 + v / 90 * bw; }, rh = (h - 40) / 3;
        ctx.strokeStyle = c.line; ctx.beginPath(); ctx.moveTo(x0, 8); ctx.lineTo(x0, h - 28); ctx.stroke();
        ctx.fillStyle = c.muted; ctx.textAlign = "center";
        [0, 20, 40, 60, 80].forEach(function (v) { ctx.fillText("+" + v + "%", Xp(v), h - 12); });
        rows.forEach(function (r, i) {
          var y = 10 + i * rh;
          ctx.font = SANS; ctx.fillStyle = c.ink; ctx.textAlign = "left";
          ctx.fillText(r[0], 0, lab ? y + rh / 2 + 4 : y + 12);
          var yb = lab ? y + rh / 2 - 10 : y + 20;
          ctx.fillStyle = i === 0 ? c.gold : c.accent; ctx.globalAlpha = 0.35; ctx.fillRect(x0, yb, Xp(r[1]) - x0, 20); ctx.globalAlpha = 1;
          ctx.fillRect(Xp(r[1]), yb, Math.max(Xp(r[2]) - Xp(r[1]), 3), 20);
          ctx.font = MONO; ctx.fillStyle = c.ink; ctx.textAlign = "left";
          ctx.fillText(i === 2 ? "≈ +80%" : "+" + r[1] + "–" + r[2] + "%", Xp(r[2]) + 6, yb + 15);
          pts.push([Xp(r[2]), yb + 10, i]);
        });
        ctx.font = MONO;
      } else {
        var P = [["Short-term homepage engagement", 0.115, 0.2, "p = 3.1 × 10⁻¹⁰"], ["Long-term core metric", 0.006, 0.015, "p = 0.025"]];
        var half = w / 2;
        P.forEach(function (m, i) {
          var x0 = i * half + 44, bw = half - 70, base = oy + ph, top = oy + 20, Yv = function (v) { return base - (v / m[2]) * (base - top); };
          ctx.strokeStyle = c.line; ctx.beginPath(); ctx.moveTo(x0, top - 6); ctx.lineTo(x0, base); ctx.lineTo(x0 + bw, base); ctx.stroke();
          ctx.fillStyle = c.muted; ctx.textAlign = "right"; ctx.fillText("0", x0 - 4, base + 4); ctx.fillText(m[2] + "%", x0 - 4, top + 4);
          var bx = x0 + bw * 0.3, bwid = bw * 0.4;
          ctx.fillStyle = c.accent; ctx.fillRect(bx, Yv(m[1]), bwid, base - Yv(m[1]));
          pts.push([bx + bwid / 2, Yv(m[1]), i]);
          ctx.fillStyle = c.ink; ctx.font = SANS; ctx.textAlign = "center"; ctx.fillText("+" + m[1] + "%", bx + bwid / 2, Yv(m[1]) - 8);
          ctx.font = MONO; ctx.fillStyle = c.muted; ctx.fillText(m[3], bx + bwid / 2, base + 16);
          ctx.fillStyle = c.ink2; ctx.fillText(m[0], x0 + bw / 2, h - 4);
        });
      }
    });
    function near(e) {
      var r = cv.c.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, best = -1, bd = 900;
      pts.forEach(function (p) { var d = (p[0] - x) * (p[0] - x) + (p[1] - y) * (p[1] - y); if (d < bd) { bd = d; best = p[2]; } });
      if (best !== hover) { hover = best; cv.redraw(); upd(); }
    }
    cv.c.addEventListener("pointermove", near);
    cv.c.addEventListener("pointerdown", near);
    var ctrl = el("div", { class: "demo-ctrl" });
    host.appendChild(ctrl);
    var tb = TABS.map(function (t, i) { return button(ctrl, t, function () { tab = i; hover = -1; tb.forEach(function (b, k) { b.setAttribute("aria-pressed", String(k === i)); }); cv.redraw(); upd(); }, i === 0); });
    var out = el("p", { class: "demo-out", "aria-live": "polite" });
    host.appendChild(out);
    function upd() {
      var msg;
      if (tab === 0) msg = hover >= 0 ? DATA[hover][0] + "× data → normalized MRR ≈ <b class=\"hl\">" + DATA[hover][1].toFixed(3) + "</b>." :
        "MRR rises monotonically from 1× to 20× Phase-2 data (≈ +16%). The ~1B model scales the same way from a lower level (§5.2, Figure 4).";
      else if (tab === 1) msg = hover >= 0 ? HIST[hover][0] + " events: normalized MRR <b class=\"hl\">" + HIST[hover][1].toFixed(3) + "</b> (" + HIST[hover][2] + ")." :
        "The elbow: halving the chosen history (2N → N) costs 7.9%; adding 50% more (3N) gains only 1.7% for 50% more tokens (§5.4, Figure 5).";
      else if (tab === 2) msg = "Phase 1 adds <b>10–20%</b> over an open-source base. Phase 2 adds <b class=\"hl\">35–50%</b> on top of a fresh Phase-1 model, growing to <b>≈ 80%</b> after two weeks as Phase 1 goes stale (§5.3, Table 1).";
      else msg = "4 weeks, ~10% of traffic, a low-data, low-signal GenRec vs the production ranker. Both lifts are statistically significant; note the different y-axis scales (§5.1, Figure 3).";
      out.innerHTML = msg;
    }
    upd();
  };

  /* ---------- mount ---------- */
  document.querySelectorAll(".paper-demo[data-demo]").forEach(function (fig) {
    var fn = DEMOS[fig.getAttribute("data-demo")], body = fig.querySelector(".demo-body");
    if (!fn) { body.textContent = "Missing demo: " + fig.getAttribute("data-demo"); return; }
    try { fn(body); } catch (err) { console.error(err); body.textContent = "This figure failed to load."; }
  });
})();
