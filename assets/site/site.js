(function () {
  "use strict";

  var root = document.documentElement;
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

  /* ---------- Theme ---------- */
  var themeBtn = document.getElementById("theme-toggle");
  var systemDark = window.matchMedia("(prefers-color-scheme: dark)");
  function currentTheme() {
    return root.getAttribute("data-theme") || (systemDark.matches ? "dark" : "light");
  }
  function syncThemeLabel() {
    themeBtn.setAttribute("aria-label", currentTheme() === "dark" ? "Switch to light mode" : "Switch to dark mode");
  }
  themeBtn.addEventListener("click", function () {
    var next = currentTheme() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch (e) {}
    syncThemeLabel();
  });
  syncThemeLabel();

  /* ---------- Menu ---------- */
  var menuBtn = document.getElementById("menu-toggle");
  var menu = document.getElementById("menu");
  function setMenu(open) {
    menuBtn.setAttribute("aria-expanded", String(open));
    menuBtn.textContent = open ? "Close" : "Menu";
    menuBtn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    document.body.style.overflow = open ? "hidden" : "";
    if (open) {
      menu.hidden = false;
      requestAnimationFrame(function () {
        menu.classList.add("is-open");
      });
    } else {
      menu.classList.remove("is-open");
      setTimeout(function () {
        if (menuBtn.getAttribute("aria-expanded") === "false") menu.hidden = true;
      }, 600);
    }
    updateNav();
  }
  menuBtn.addEventListener("click", function () {
    setMenu(menuBtn.getAttribute("aria-expanded") !== "true");
  });
  menu.addEventListener("click", function (e) {
    if (e.target.closest("a")) setMenu(false);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && menuBtn.getAttribute("aria-expanded") === "true") setMenu(false);
  });

  /* ---------- Nav colour, progress and page "noise level" ---------- */
  var nav = document.getElementById("nav");
  var hero = document.querySelector(".hero");
  var contact = document.getElementById("contact");
  var progress = document.getElementById("progress");
  var sigmaOut = document.getElementById("sigma");
  function overlaps(el, y) {
    var r = el.getBoundingClientRect();
    return r.top <= y && r.bottom > y;
  }
  function updateNav() {
    var mid = nav.offsetHeight / 2;
    var onBlue = menuBtn.getAttribute("aria-expanded") === "true" || overlaps(hero, mid) || overlaps(contact, mid);
    nav.classList.toggle("on-blue", onBlue);
    var max = document.documentElement.scrollHeight - window.innerHeight;
    var p = max > 0 ? Math.min(window.scrollY / max, 1) : 0;
    progress.style.setProperty("--p", p);
    sigmaOut.textContent = (1 - p).toFixed(3);
  }
  var navTick = false;
  window.addEventListener(
    "scroll",
    function () {
      if (!navTick) {
        navTick = true;
        requestAnimationFrame(function () {
          updateNav();
          navTick = false;
        });
      }
    },
    { passive: true }
  );
  window.addEventListener("resize", updateNav);
  updateNav();

  /* ---------- Active section ---------- */
  var navLinks = Array.prototype.slice.call(document.querySelectorAll(".nav-links a"));
  if ("IntersectionObserver" in window) {
    var spy = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          navLinks.forEach(function (a) {
            a.classList.toggle("is-active", a.getAttribute("href") === "#" + entry.target.id);
          });
        });
      },
      { rootMargin: "-45% 0px -50% 0px" }
    );
    navLinks.forEach(function (a) {
      var s = document.querySelector(a.getAttribute("href"));
      if (s) spy.observe(s);
    });
  }

  /* ==========================================================================
     Hero: the name is sampled out of Gaussian noise, like a diffusion model.
     x_t = x_0 + sigma(t) * eps, with a cosine schedule from t = 1000 to 0.
     The pointer injects local noise; a click re-samples from scratch.
     ========================================================================== */
  var canvas = document.getElementById("field");
  var stepOut = document.getElementById("t-step");
  var stateOut = document.getElementById("t-state");
  var ctx = canvas && canvas.getContext ? canvas.getContext("2d") : null;

  if (ctx && hero) {
    root.classList.add("has-field");
    var W = 0,
      H = 0,
      dpr = 1,
      cell = 7,
      dots = [],
      floor = [],
      start = 0,
      DURATION = 2800,
      running = false,
      visible = true,
      pointer = { x: -9999, y: -9999, active: false },
      lastState = "";

    var gauss = function () {
      var u = 1 - Math.random(),
        v = Math.random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };

    var setState = function (s) {
      if (s !== lastState) {
        stateOut.textContent = s;
        lastState = s;
      }
    };

    var layout = function () {
      W = hero.clientWidth;
      H = hero.clientHeight;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      var narrow = W < 700 || H > W * 1.1;
      cell = W < 560 ? 3 : W < 1000 ? 5 : W < 1600 ? 6 : 7;
      var gutter = parseFloat(getComputedStyle(hero.querySelector(".hero-foot")).left) || 24;

      var off = document.createElement("canvas");
      off.width = W;
      off.height = H;
      var o = off.getContext("2d");
      var lines = ["AARYAN", "SHAH"];
      var setFont = function (size) {
        o.font = "800 " + size + "px Archivo, 'Helvetica Neue', Arial, sans-serif";
        try {
          o.fontStretch = narrow ? "extra-condensed" : "expanded";
        } catch (e) {}
      };
      setFont(100);
      var widest = Math.max(o.measureText(lines[0]).width, o.measureText(lines[1]).width);
      var top = narrow ? 120 : 130;
      var bottom = narrow ? 190 : 170;
      var availH = H - top - bottom;
      var size = Math.min((100 * (W - gutter * 2)) / widest, availH / (lines.length * 0.86));
      setFont(size);
      var m = o.measureText("A");
      var cap = m.actualBoundingBoxAscent || size * 0.72;
      var lead = cap * 1.14;
      var blockH = cap + lead * (lines.length - 1);
      // On narrow screens the name is width-bound, so stretch it vertically
      // (a tall dot-matrix sign) to use the height of the hero.
      var k = narrow ? Math.max(1, Math.min(2.2, (availH * 0.8) / blockH)) : 1;
      o.fillStyle = "#000";
      o.textBaseline = "alphabetic";
      o.setTransform(1, 0, 0, k, 0, top + (availH - blockH * k) / 2);
      lines.forEach(function (line, i) {
        o.fillText(line, gutter - (m.actualBoundingBoxLeft || 0), cap + i * lead);
      });
      o.setTransform(1, 0, 0, 1, 0, 0);

      var data = o.getImageData(0, 0, W, H).data;
      var spread = Math.max(W, H) * 0.42;
      dots = [];
      for (var y = Math.floor(cell / 2); y < H; y += cell) {
        for (var x = Math.floor(cell / 2); x < W; x += cell) {
          if (data[(y * W + x) * 4 + 3] > 120) {
            dots.push({ tx: x, ty: y, nx: gauss() * spread, ny: gauss() * spread * 0.7, h: 0 });
          }
        }
      }
      var nFloor = Math.round((W * H) / (cell * cell * 28));
      floor = [];
      for (var i = 0; i < nFloor; i++) floor.push({ x: Math.random() * W, y: Math.random() * H });
    };

    var resample = function () {
      dots.forEach(function (d) {
        var spread = Math.max(W, H) * 0.42;
        d.nx = gauss() * spread;
        d.ny = gauss() * spread * 0.7;
      });
      start = performance.now();
      kick();
    };

    var sigmaAt = function (now) {
      if (reduceMotion) return 0;
      var s = Math.min((now - start) / DURATION, 1);
      var c = Math.cos((s * Math.PI) / 2);
      return c * c;
    };

    var draw = function (now) {
      var sigma = sigmaAt(now);
      var t = sigma > 0.001 ? Math.max(1, Math.round(sigma * 1000)) : 0;
      stepOut.textContent = String(t).padStart(4, "0");

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#efebe3";

      // Noise floor: static that fades as the sample converges.
      if (sigma > 0.002) {
        ctx.globalAlpha = Math.min(1, sigma * 1.4) * 0.55;
        var fs = cell * 0.5;
        for (var i = 0; i < floor.length; i++) {
          var f = floor[i];
          if (Math.random() < 0.5) {
            f.x = Math.random() * W;
            f.y = Math.random() * H;
          }
          ctx.fillRect(f.x, f.y, fs, fs);
        }
        ctx.globalAlpha = 1;
      }

      var size = cell * 0.66;
      var r2 = Math.pow(Math.max(W, H) * 0.075, 2);
      var heat = 0;
      for (var j = 0; j < dots.length; j++) {
        var d = dots[j];
        d.h *= 0.93;
        if (pointer.active) {
          var dx = d.tx - pointer.x,
            dy = d.ty - pointer.y;
          var p = Math.exp(-(dx * dx + dy * dy) / r2);
          if (p > d.h) d.h = p;
        }
        if (d.h < 0.002) d.h = 0;
        heat += d.h;
        var s = sigma + d.h * 0.09;
        var jitter = s * cell * 1.6;
        var x = d.tx + d.nx * s + (s > 0.001 ? (Math.random() - 0.5) * jitter : 0);
        var y = d.ty + d.ny * s + (s > 0.001 ? (Math.random() - 0.5) * jitter : 0);
        ctx.fillRect(x - size / 2, y - size / 2, size, size);
      }

      if (sigma > 0.001) setState("sampling…");
      else if (heat > 1) setState("perturbed — denoising");
      else setState("converged ✓");

      return sigma > 0.001 || heat > 0.5 || pointer.active;
    };

    var loop = function (now) {
      if (!visible) {
        running = false;
        return;
      }
      var more = draw(now);
      if (more) requestAnimationFrame(loop);
      else running = false;
    };
    var kick = function () {
      if (!running && visible) {
        running = true;
        requestAnimationFrame(loop);
      }
    };

    var boot = function () {
      layout();
      start = performance.now();
      if (reduceMotion) {
        draw(start);
      } else {
        kick();
      }
    };

    if (!reduceMotion && finePointer) {
      hero.addEventListener("pointermove", function (e) {
        var r = hero.getBoundingClientRect();
        pointer.x = e.clientX - r.left;
        pointer.y = e.clientY - r.top;
        pointer.active = true;
        kick();
      });
      hero.addEventListener("pointerleave", function () {
        pointer.active = false;
      });
      hero.addEventListener("click", function (e) {
        if (e.target.closest("a, button")) return;
        resample();
      });
    } else if (!reduceMotion) {
      hero.addEventListener("click", function (e) {
        if (e.target.closest("a, button")) return;
        resample();
      });
    }

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        if (visible) kick();
      }).observe(hero);
    }

    var lastW = 0;
    window.addEventListener("resize", function () {
      // Ignore height-only changes from mobile browser chrome.
      if (hero.clientWidth === lastW && Math.abs(hero.clientHeight - H) < 120) return;
      lastW = hero.clientWidth;
      layout();
      draw(performance.now());
      kick();
    });

    var fontReady = document.fonts && document.fonts.load ? document.fonts.load("800 100px Archivo") : Promise.resolve();
    Promise.race([fontReady, new Promise(function (r) { setTimeout(r, 1500); })]).then(function () {
      lastW = hero.clientWidth;
      boot();
    });
  }

  /* ---------- Statement: words resolve out of blur ---------- */
  var statement = document.getElementById("statement");
  if (statement) {
    var idx = 0;
    Array.prototype.slice.call(statement.childNodes).forEach(function (node) {
      if (node.nodeType === 3) {
        var frag = document.createDocumentFragment();
        node.textContent.split(/(\s+)/).forEach(function (part) {
          if (!part) return;
          if (/^\s+$/.test(part)) {
            frag.appendChild(document.createTextNode(part));
          } else {
            var s = document.createElement("span");
            s.className = "w";
            s.style.setProperty("--i", idx++);
            s.textContent = part;
            frag.appendChild(s);
          }
        });
        statement.replaceChild(frag, node);
      } else if (node.nodeType === 1) {
        var wrap = document.createElement("span");
        wrap.className = "w";
        wrap.style.setProperty("--i", idx++);
        statement.replaceChild(wrap, node);
        wrap.appendChild(node);
      }
    });
  }

  /* ---------- Reveal + kicker decode ---------- */
  var GLYPHS = "▚▞▖▗▘▝░▒01#/<>";
  var scramble = function (el) {
    if (el.dataset.done) return;
    el.dataset.done = "1";
    var target = el.textContent;
    if (reduceMotion) return;
    var t0 = performance.now(),
      dur = 700;
    (function step(now) {
      var k = Math.min((now - t0) / dur, 1);
      var shown = Math.floor(k * target.length);
      var out = target.slice(0, shown);
      for (var i = shown; i < target.length; i++) {
        out += target[i] === " " ? " " : GLYPHS[(Math.random() * GLYPHS.length) | 0];
      }
      el.textContent = out;
      if (k < 1) requestAnimationFrame(step);
      else el.textContent = target;
    })(t0);
  };

  var revealTargets = document.querySelectorAll(".reveal, .statement, .scramble");
  if ("IntersectionObserver" in window && !reduceMotion) {
    var io = new IntersectionObserver(
      function (entries) {
        var i = 0;
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          var el = entry.target;
          if (el.classList.contains("scramble")) scramble(el);
          else {
            el.style.setProperty("--d", Math.min(i++ * 0.07, 0.35) + "s");
            el.classList.add("is-in");
          }
          io.unobserve(el);
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -6% 0px" }
    );
    revealTargets.forEach(function (el) {
      io.observe(el);
    });
  } else {
    revealTargets.forEach(function (el) {
      el.classList.add("is-in");
    });
  }

  /* ---------- Count-up metrics ---------- */
  var counters = document.querySelectorAll(".count");
  if ("IntersectionObserver" in window && !reduceMotion) {
    var co = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          var el = entry.target;
          var to = parseFloat(el.getAttribute("data-to"));
          var suffix = el.getAttribute("data-suffix") || "";
          var t0 = performance.now();
          (function step(now) {
            var k = Math.min((now - t0) / 1500, 1);
            el.textContent = Math.round(to * (1 - Math.pow(1 - k, 4))) + suffix;
            if (k < 1) requestAnimationFrame(step);
          })(t0);
          co.unobserve(el);
        });
      },
      { threshold: 0.6 }
    );
    counters.forEach(function (c) {
      c.textContent = "0" + (c.getAttribute("data-suffix") || "");
      co.observe(c);
    });
  }

  /* ---------- Projects: accordion + cursor preview ---------- */
  var items = Array.prototype.slice.call(document.querySelectorAll(".exp"));
  items.forEach(function (item) {
    var btn = item.querySelector(".exp-head");
    btn.addEventListener("click", function () {
      var open = !item.classList.contains("is-open");
      item.classList.toggle("is-open", open);
      btn.setAttribute("aria-expanded", String(open));
      if (open) hidePreview();
    });
    var img = item.querySelector(".exp-media img");
    if (img && finePointer) {
      var duo = img.getAttribute("src");
      var color = img.getAttribute("data-color");
      img.parentNode.addEventListener("mouseenter", function () {
        img.src = color;
      });
      img.parentNode.addEventListener("mouseleave", function () {
        img.src = duo;
      });
    }
  });

  var preview = document.getElementById("preview");
  var previewImg = preview.querySelector("img");
  function hidePreview() {
    preview.classList.remove("is-on");
  }
  if (finePointer && !reduceMotion) {
    var px = 0,
      py = 0,
      cx = 0,
      cy = 0,
      raf = 0;
    var follow = function () {
      cx += (px - cx) * 0.18;
      cy += (py - cy) * 0.18;
      preview.style.setProperty("--x", cx + "px");
      preview.style.setProperty("--y", cy + "px");
      raf = Math.abs(px - cx) + Math.abs(py - cy) > 0.5 ? requestAnimationFrame(follow) : 0;
    };
    items.forEach(function (item) {
      var head = item.querySelector(".exp-head");
      head.addEventListener("mouseenter", function () {
        if (item.classList.contains("is-open")) return;
        var src = item.getAttribute("data-preview");
        if (previewImg.getAttribute("src") !== src) previewImg.src = src;
        preview.classList.add("is-on");
      });
      head.addEventListener("mouseleave", hidePreview);
      head.addEventListener("mousemove", function (e) {
        px = e.clientX + 24;
        py = e.clientY - preview.offsetHeight / 2;
        if (!preview.classList.contains("is-on")) {
          cx = px;
          cy = py;
        }
        if (!raf) raf = requestAnimationFrame(follow);
      });
    });
    window.addEventListener("scroll", hidePreview, { passive: true });
  }

  /* ---------- Copy email ---------- */
  var copyBtn = document.getElementById("copy-email");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var label = copyBtn.querySelector("span");
      var email = copyBtn.getAttribute("data-email");
      var done = function () {
        label.textContent = "Copied";
        setTimeout(function () {
          label.textContent = "Copy";
        }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(email).then(done, function () {
          window.location.href = "mailto:" + email;
        });
      } else {
        window.location.href = "mailto:" + email;
      }
    });
  }

  /* ---------- Contact form (Formspree, stays on page) ---------- */
  var form = document.getElementById("contactForm");
  var status = document.getElementById("form-status");
  if (form && window.fetch) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var btn = form.querySelector("button[type=submit]");
      btn.disabled = true;
      status.textContent = "202 Accepted — sending…";
      fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        headers: { Accept: "application/json" },
      })
        .then(function (res) {
          if (!res.ok) throw new Error("Request failed");
          form.reset();
          status.textContent = "200 OK — thanks, your message has been sent.";
        })
        .catch(function () {
          status.textContent = "Error — please email shah.aar27@gmail.com directly.";
        })
        .finally(function () {
          btn.disabled = false;
        });
    });
  }

  /* ---------- Footer wordmark: fit to the full width ---------- */
  var mark = document.querySelector(".footer-mark");
  var fitMark = function () {
    if (!mark) return;
    mark.style.fontSize = "";
    var avail = mark.clientWidth;
    var natural = mark.firstElementChild.getBoundingClientRect().width;
    if (natural > 0) mark.style.fontSize = (parseFloat(getComputedStyle(mark).fontSize) * avail * 0.995) / natural + "px";
  };
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitMark);
  else fitMark();
  window.addEventListener("resize", fitMark);

  /* ---------- Footer ---------- */
  var year = document.getElementById("year");
  if (year) year.textContent = new Date().getFullYear();
})();
