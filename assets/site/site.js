(function () {
  "use strict";

  var root = document.documentElement;
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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

  /* ---------- Mobile menu ---------- */
  var menuBtn = document.getElementById("menu-toggle");
  var menu = document.getElementById("mobile-menu");
  function setMenu(open) {
    menuBtn.setAttribute("aria-expanded", String(open));
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
      }, 350);
    }
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

  /* ---------- Nav state + scroll progress ---------- */
  var nav = document.getElementById("nav");
  var bar = document.querySelector(".progress span");
  var ticking = false;
  function onScroll() {
    var y = window.scrollY;
    var max = document.documentElement.scrollHeight - window.innerHeight;
    nav.classList.toggle("is-scrolled", y > 24);
    bar.style.setProperty("--p", max > 0 ? Math.min(y / max, 1) : 0);
    ticking = false;
  }
  window.addEventListener(
    "scroll",
    function () {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(onScroll);
      }
    },
    { passive: true }
  );
  onScroll();

  /* ---------- Active section in nav ---------- */
  var navLinks = Array.prototype.slice.call(document.querySelectorAll(".nav-links a"));
  var sections = navLinks
    .map(function (a) {
      return document.querySelector(a.getAttribute("href"));
    })
    .filter(Boolean);
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
    sections.forEach(function (s) {
      spy.observe(s);
    });
  }

  /* ---------- Reveal on scroll (with sibling stagger) ---------- */
  var reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && !reduceMotion) {
    var io = new IntersectionObserver(
      function (entries) {
        var batch = entries.filter(function (e) {
          return e.isIntersecting;
        });
        batch.forEach(function (entry, i) {
          entry.target.style.setProperty("--d", Math.min(i * 0.08, 0.4) + "s");
          entry.target.classList.add("is-in");
          io.unobserve(entry.target);
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -6% 0px" }
    );
    reveals.forEach(function (el) {
      io.observe(el);
    });
  } else {
    reveals.forEach(function (el) {
      el.classList.add("is-in");
    });
  }

  /* ---------- Hero intro ---------- */
  requestAnimationFrame(function () {
    requestAnimationFrame(function () {
      document.body.classList.add("is-loaded");
    });
  });

  /* ---------- Rotating roles ---------- */
  var roles = document.querySelectorAll(".roles-track span");
  if (roles.length > 1 && !reduceMotion) {
    var idx = 0;
    setInterval(function () {
      var prev = roles[idx];
      idx = (idx + 1) % roles.length;
      var next = roles[idx];
      prev.classList.remove("is-active");
      prev.classList.add("is-leaving");
      next.classList.remove("is-leaving");
      next.classList.add("is-active");
      setTimeout(function () {
        prev.classList.remove("is-leaving");
      }, 800);
    }, 2600);
  }

  /* ---------- Count-up stats ---------- */
  var counters = document.querySelectorAll(".count");
  if ("IntersectionObserver" in window && !reduceMotion) {
    var co = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          var el = entry.target;
          var to = parseFloat(el.getAttribute("data-to"));
          var suffix = el.getAttribute("data-suffix") || "";
          var start = performance.now();
          var dur = 1600;
          (function step(now) {
            var t = Math.min((now - start) / dur, 1);
            var eased = 1 - Math.pow(1 - t, 4);
            el.textContent = Math.round(to * eased) + suffix;
            if (t < 1) requestAnimationFrame(step);
          })(start);
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

  /* ---------- Project card spotlight ---------- */
  if (window.matchMedia("(hover: hover)").matches) {
    document.querySelectorAll(".project").forEach(function (card) {
      card.addEventListener("pointermove", function (e) {
        var r = card.getBoundingClientRect();
        card.style.setProperty("--mx", e.clientX - r.left + "px");
        card.style.setProperty("--my", e.clientY - r.top + "px");
      });
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
      status.className = "form-status";
      status.textContent = "Sending…";
      fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        headers: { Accept: "application/json" },
      })
        .then(function (res) {
          if (!res.ok) throw new Error("Request failed");
          form.reset();
          status.className = "form-status is-ok";
          status.textContent = "Thanks — your message has been sent.";
        })
        .catch(function () {
          status.className = "form-status is-err";
          status.textContent = "Something went wrong. Please email shah.aar27@gmail.com directly.";
        })
        .finally(function () {
          btn.disabled = false;
        });
    });
  }

  /* ---------- Footer year ---------- */
  var year = document.getElementById("year");
  if (year) year.textContent = new Date().getFullYear();
})();
