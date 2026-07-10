(function () {
  async function loadManifest(manifestUrl) {
    const res = await fetch(manifestUrl, { cache: "no-store" });
    if (!res.ok) throw new Error(`Manifest load failed: ${res.status}`);
    return res.json();
  }

  function toDateSortable(d) {
    // Expect YYYY-MM-DD (ISO). Fallback keeps stable ordering.
    return d ? new Date(d + "T00:00:00Z").getTime() : 0;
  }

  function shortenDescription(desc, maxChars) {
    const s = (desc || "").trim();
    if (!s) return "";
    if (s.length <= maxChars) return s;
    return s.slice(0, Math.max(0, maxChars - 1)).trimEnd() + "…";
  }

  async function init() {
    if (window.__debugInstrumentation && window.__debugInstrumentation.initLog)
      window.__debugInstrumentation.initLog("now-reading", "now-reading");
    const root = document.getElementById("now-reading");
    if (!root) return;

    const blogLink = document.getElementById("now-reading-blog-link");
    const blogDesc = document.getElementById("now-reading-blog-desc");
    if (!blogLink || !blogDesc) return;

    let manifest;
    try {
      manifest = await loadManifest("blogs/manifest.json");
    } catch (e) {
      // Keep existing static placeholders.
      return;
    }

    if (!Array.isArray(manifest) || manifest.length === 0) return;

    const sorted = manifest
      .slice()
      .sort((a, b) => toDateSortable(b.date) - toDateSortable(a.date));

    const latest = sorted[0];
    if (!latest) return;

    const url = latest.url || `blogs/${latest.slug}.html`;
    blogLink.href = url;

    const desc = latest.description || "";
    const truncated = shortenDescription(desc, 80);
    blogDesc.textContent = truncated || (desc ? desc : blogDesc.textContent);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
