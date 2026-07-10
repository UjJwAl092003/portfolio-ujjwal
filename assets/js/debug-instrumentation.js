/*
  Debug instrumentation (logging only)
  Injected to identify repeated init / refresh / re-render events on mobile.
  NOTE: This file is intended for temporary debugging.
*/

(function () {
  // Avoid double-instrumentation across hot reloads.
  if (window.__DEBUG_INSTRUMENTATION_INSTALLED__) return;
  window.__DEBUG_INSTRUMENTATION_INSTALLED__ = true;

  const counters = Object.create(null);

  function nextCounter(key) {
    counters[key] = (counters[key] || 0) + 1;
    return counters[key];
  }

  function warnIfDuplicateInit(key, maxBeforeWarn = 1) {
    if (counters[key] > maxBeforeWarn) {
      console.warn(
        `WARNING: Duplicate initialization detected. ${key} #${counters[key]}`,
      );
    }
  }

  function initLog(fileLabel, counterKey) {
    const n = nextCounter(counterKey || fileLabel);
    console.log(`[INIT] ${fileLabel} #${n}`);
    if (counterKey) warnIfDuplicateInit(fileLabel);
    return n;
  }

  // --- Global event logs ---
  const eventNames = [
    "DOMContentLoaded",
    "load",
    "pageshow",
    "pagehide",
    "visibilitychange",
    "resize",
    "orientationchange",
    "hashchange",
    "popstate",
  ];

  function logEvent(evName, extra) {
    if (evName === "visibilitychange") {
      console.log(`[EVENT] visibilitychange hidden=${document.hidden}`);
    } else {
      console.log(`[EVENT] ${evName}${extra ? " " + extra : ""}`);
    }
  }

  // Attach once.
  const installed = new Set();
  function on(name, handler, target) {
    const key = `${target ? "target" : "window"}:${name}`;
    if (installed.has(key)) return;
    installed.add(key);
    (target || window).addEventListener(name, handler, { capture: true });
  }

  on("DOMContentLoaded", () => {
    initLog("event DOMContentLoaded", "DOMContentLoaded");
    logEvent("DOMContentLoaded");
  });

  on("load", () => {
    initLog("event load", "load");
    logEvent("load");
  });

  on("pageshow", (e) => {
    initLog("event pageshow", "pageshow");
    logEvent("pageshow", e && e.persisted ? `persisted=${e.persisted}` : "");
  });

  on("pagehide", (e) => {
    initLog("event pagehide", "pagehide");
    logEvent("pagehide", e && e.persisted ? `persisted=${e.persisted}` : "");
  });

  on(
    "visibilitychange",
    () => {
      initLog("event visibilitychange", "visibilitychange");
      logEvent("visibilitychange");
    },
    document,
  );

  on("resize", () => {
    initLog("event resize", "resize");
    logEvent("resize");
  });

  on("orientationchange", () => {
    initLog("event orientationchange", "orientationchange");
    logEvent("orientationchange");
  });

  on("hashchange", (e) => {
    initLog("event hashchange", "hashchange");
    const h = e && e.newURL ? e.newURL : window.location.hash;
    logEvent("hashchange", h);
  });

  on("popstate", (e) => {
    initLog("event popstate", "popstate");
    logEvent("popstate", window.location.pathname + window.location.search);
  });

  // --- fetch logging ---
  const originalFetch = window.fetch ? window.fetch.bind(window) : null;
  if (originalFetch) {
    window.fetch = async function debugFetch(input, init) {
      try {
        const url = typeof input === "string" ? input : input && input.url;
        console.log(`[FETCH] ${url || input}`);
      } catch (_) {}
      return originalFetch(input, init);
    };
  }

  // --- navigation logging ---
  // window.location assignment
  try {
    const loc = window.location;
    const proto = Object.getPrototypeOf(loc);
    const desc = Object.getOwnPropertyDescriptor(proto, "href");
    // We can't safely redefine all of location; so log when typical assignments happen.
  } catch (_) {}

  // history push/replace logging
  if (window.history) {
    const origPush = window.history.pushState;
    const origReplace = window.history.replaceState;

    if (typeof origPush === "function") {
      window.history.pushState = function (...args) {
        console.log("[NAV] history.pushState", args[0]);
        return origPush.apply(this, args);
      };
    }

    if (typeof origReplace === "function") {
      window.history.replaceState = function (...args) {
        console.log("[NAV] history.replaceState", args[0]);
        return origReplace.apply(this, args);
      };
    }
  }

  // intercept window.location.href assignment via defineProperty is unreliable across browsers,
  // but we can at least wrap common navigation helper patterns.
  // We'll also log when clicking anchor with target=_self by listening to capture.
  window.addEventListener(
    "click",
    (e) => {
      const a = e.target && e.target.closest ? e.target.closest("a") : null;
      if (!a) return;
      const href = a.getAttribute("href");
      if (!href || href === "#") return;
      const target = a.getAttribute("target");
      if (!target || target === "_self") {
        console.log(`[NAV] window.location href click href=${href}`);
      }
    },
    true,
  );

  // Expose helper to label init points.
  window.__debugInstrumentation = {
    initLog,
  };
})();
