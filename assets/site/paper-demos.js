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

  /* ---------- mount ---------- */
  document.querySelectorAll(".paper-demo[data-demo]").forEach(function (fig) {
    var fn = DEMOS[fig.getAttribute("data-demo")], body = fig.querySelector(".demo-body");
    if (!fn) { body.textContent = "Missing demo: " + fig.getAttribute("data-demo"); return; }
    try { fn(body); } catch (err) { console.error(err); body.textContent = "This figure failed to load."; }
  });
})();
