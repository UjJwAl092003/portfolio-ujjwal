/*
  Blog Renderer (static markdown)
  - Loads blogs/manifest.json
  - Finds markdown file by slug
  - Parses YAML front matter
  - Renders Markdown via marked.js (GFM)
  - Generates TOC from h2/h3
  - Syntax highlights via highlight.js
*/

(function () {
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "<")
      .replace(/>/g, ">")
      .replace(/\\"/g, '"')
      .replace(/'/g, "&#039;");
  }

  function stripQuotes(s) {
    const str = String(s);
    if (
      (str.startsWith('"') && str.endsWith('"')) ||
      (str.startsWith("'") && str.endsWith("'"))
    ) {
      return str.slice(1, -1);
    }
    return str;
  }

  function parseFrontMatter(md) {
    const fmMatch = md.match(/^---\s*\n([\s\S]*?)\n---\s*\n/);
    if (!fmMatch) return { frontMatter: {}, body: md };

    const yaml = fmMatch[1];
    const body = md.slice(fmMatch[0].length);

    const lines = yaml.split(/\r?\n/);
    const fm = {};

    let currentArrayKey = null;

    for (let raw of lines) {
      const line = raw.trimEnd();
      const t = line.trim();
      if (!t) continue;

      // key:   (start array)
      const arrayKeyMatch = t.match(/^([A-Za-z0-9_\-]+):\s*$/);
      if (arrayKeyMatch) {
        const key = arrayKeyMatch[1];
        fm[key] = [];
        currentArrayKey = key;
        continue;
      }

      // array item: - value
      if (t.startsWith("- ") && currentArrayKey) {
        const item = t.slice(2).trim();
        fm[currentArrayKey].push(stripQuotes(item));
        continue;
      }

      // key: value (inline)
      const keyVal = t.match(/^([A-Za-z0-9_\-]+):\s*(.*)$/);
      if (!keyVal) continue;

      const key = keyVal[1];
      let val = keyVal[2] || "";

      // Inline array: [a,b]
      if (val.startsWith("[") && val.endsWith("]")) {
        const arr = val
          .replace(/^\[/, "")
          .replace(/\]$/, "")
          .split(",")
          .map((s) => stripQuotes(s.trim()))
          .filter(Boolean);
        fm[key] = arr;
        currentArrayKey = null;
        continue;
      }

      fm[key] = stripQuotes(val.trim());
      currentArrayKey = null;
    }

    return { frontMatter: fm, body };
  }

  function extractToc(html) {
    const container = document.createElement("div");
    container.innerHTML = html;

    const headings = container.querySelectorAll("h2, h3");
    const tocItems = [];

    headings.forEach((h) => {
      const level = h.tagName.toLowerCase();
      const text = h.textContent.trim();
      if (!text) return;

      if (!h.id) {
        const id = text
          .toLowerCase()
          .replace(/[^a-z0-9\s-]/g, "")
          .replace(/\s+/g, "-")
          .replace(/-+/g, "-")
          .trim();
        h.id = id;
      }

      tocItems.push({ level, id: h.id, text });
    });

    return { tocItems, container };
  }

  function renderMd(md, { tocMountId, contentMountId }) {
    marked.setOptions({
      gfm: true,
      breaks: false,
      headerIds: false,
    });

    const rawHtml = marked.parse(md);
    const { tocItems, container } = extractToc(rawHtml);

    const tocRoot = document.getElementById(tocMountId);
    if (tocRoot) {
      if (!tocItems.length) {
        tocRoot.style.display = "none";
      } else {
        const list = tocItems
          .map((t) => {
            const indent = t.level === "h3" ? "toc-indent" : "";
            return `<li class="${indent}"><a href="#${t.id}">${escapeHtml(
              t.text,
            )}</a></li>`;
          })
          .join("");

        tocRoot.innerHTML =
          `<nav class="blog-toc">` +
          `<header class="blog-toc-header">Table of Contents</header>` +
          `<ul>${list}</ul>` +
          `</nav>`;
      }
    }

    const contentRoot = document.getElementById(contentMountId);
    if (contentRoot) {
      contentRoot.innerHTML = container.innerHTML;
    }

    if (window.hljs) window.hljs.highlightAll();
  }

  async function loadManifest(manifestUrl) {
    const res = await fetch(manifestUrl, { cache: "no-store" });
    if (!res.ok) throw new Error(`Manifest load failed: ${res.status}`);
    return res.json();
  }

  async function loadText(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`Markdown load failed: ${res.status}`);
    return res.text();
  }

  async function renderBySlug({ slug, manifestUrl, markdownBasePath }) {
    const manifest = await loadManifest(manifestUrl);

    const item = manifest.find((x) => x.slug === slug);
    const root = document.getElementById("blog-root");
    if (!root) return;

    if (!item) {
      root.innerHTML = `<div class="box"><h2>Blog not found</h2><p>Missing slug: ${escapeHtml(
        slug,
      )}</p></div>`;
      return;
    }

    // Our markdown files live alongside the HTML inside the `blogs/` folder.
    // On blogs/<slug>.html we should fetch ./<slug>.md
    const markdownFile = item.markdownFile
      ? item.markdownFile.replace(/^blogs\//, "")
      : `./${slug}.md`;

    const mdUrl = markdownBasePath
      ? `${markdownBasePath}${markdownFile}`
      : markdownFile;

    // If we were given an HTML page under /blogs/, `markdownBasePath` is typically `../`
    // Ensure the markdown URL correctly points into the `blogs/` folder.
    // (Example: blogs/naive-bayes-explained.html -> ../blogs/<slug>.md)
    const fallbackMdUrl = `${markdownBasePath || ""}blogs/${slug}.md`;

    const mdRaw = await loadText(mdUrl).catch(() => loadText(fallbackMdUrl));
    const parsed = parseFrontMatter(mdRaw);

    const fm = {
      title: parsed.frontMatter.title || item.title,
      slug: parsed.frontMatter.slug || item.slug,
      date: parsed.frontMatter.date || item.date,
      category: parsed.frontMatter.category || item.category,
      readingTime: parsed.frontMatter.readingTime || item.readingTime,
      description: parsed.frontMatter.description || item.description,
      tags: parsed.frontMatter.tags || item.tags,
      featuredImage: parsed.frontMatter.featuredImage || item.featuredImage,
      author: parsed.frontMatter.author || item.author || "Ujjwal Singh",
    };

    root.innerHTML = `
      <article class="blog-article">
        <header class="blog-header">
          <h1>${escapeHtml(fm.title || "")}</h1>
          <div class="blog-meta">
            <span class="blog-meta-item"><strong>${escapeHtml(
              fm.category || "",
            )}</strong></span>
            ${fm.date ? `<span class="blog-meta-dot">•</span><span>${escapeHtml(fm.date)}</span>` : ""}
            ${fm.readingTime ? `<span class="blog-meta-dot">•</span><span>${escapeHtml(fm.readingTime)}</span>` : ""}
          </div>
          ${fm.featuredImage ? `<div class="blog-featured"><img src="${escapeHtml(fm.featuredImage)}" alt="" /></div>` : ""}
          <div class="blog-author">By ${escapeHtml(fm.author || "Ujjwal Singh")}</div>
        </header>

        <div class="blog-layout">
          <aside id="blog-toc" class="blog-toc-wrap"></aside>
          <main id="blog-content" class="blog-content-wrap"></main>
        </div>

        <footer class="blog-nav">
          <div class="blog-nav-left">
            <a class="button" href="#" id="blog-prev" aria-disabled="true" style="display:none">&laquo; Previous</a>
          </div>
          <div class="blog-nav-right">
            <a class="button" href="../blogs.html" id="blog-back">Back to Blogs</a>
          </div>
          <div class="blog-nav-next">
            <a class="button" href="#" id="blog-next" aria-disabled="true" style="display:none">Next &raquo;</a>
          </div>
        </footer>
      </article>
    `;

    renderMd(parsed.body, {
      tocMountId: "blog-toc",
      contentMountId: "blog-content",
    });

    if (manifest.length <= 1) return;

    const idx = manifest.findIndex((x) => x.slug === slug);
    const prev = idx > 0 ? manifest[idx - 1] : null;
    const next =
      idx >= 0 && idx < manifest.length - 1 ? manifest[idx + 1] : null;

    const prevEl = document.getElementById("blog-prev");
    if (prevEl && prev) {
      prevEl.style.display = "inline-block";
      prevEl.href = `./${prev.slug}.html`;
    }

    const nextEl = document.getElementById("blog-next");
    if (nextEl && next) {
      nextEl.style.display = "inline-block";
      nextEl.href = `./${next.slug}.html`;
    }
  }

  window.BlogRenderer = { renderBySlug };
})();
