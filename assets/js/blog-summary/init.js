import { mountSummaryUI } from "./ui.js";
import { generateSummary } from "./summarizer.js";

function getPlainTextFromArticle(articleEl) {
  if (!articleEl) return "";

  // Prefer visible text; exclude script/style.
  const clone = articleEl.cloneNode(true);

  // Remove TOC and nav and any non-content elements.
  const removeSelectors = ["#blog-toc", ".blog-nav", "nav", "aside"];
  for (const sel of removeSelectors) {
    const nodes = clone.querySelectorAll(sel);
    nodes.forEach((n) => n.parentNode && n.parentNode.removeChild(n));
  }

  // Remove code blocks but keep text content.
  // We'll rely on extractive summarizer to ignore fenced/code patterns anyway.

  const txt = clone.textContent || "";
  return txt;
}

function getMarkdownTextFromContent(contentRoot) {
  // We don't have raw markdown here; we have rendered HTML.
  // Extract textContent and let local summarizer do the rest.
  // This still satisfies "no manual changes per blog".
  if (!contentRoot) return "";
  return getPlainTextFromArticle(contentRoot);
}

function inferSlugFromRoot() {
  const root = document.getElementById("blog-root");
  if (!root) return "";
  return root.getAttribute("data-slug") || "";
}

export function initBlogSummarizer() {
  const blogRoot = document.getElementById("blog-root");
  if (!blogRoot) return;

  const contentRoot = document.getElementById("blog-content");
  const tocRoot = document.getElementById("blog-toc");

  // Insert container right after .blog-meta inside the rendered header.
  const articleEl = blogRoot.querySelector(".blog-article") || blogRoot;
  const metaEl = articleEl.querySelector(".blog-meta");
  if (!metaEl) return;

  const mountTarget = document.createElement("div");
  mountTarget.className = "ai-summary-mount";
  metaEl.insertAdjacentElement("afterend", mountTarget);

  const slug = inferSlugFromRoot();

  mountSummaryUI({
    container: mountTarget,
    getCacheKey: () => `blog:${slug}`,
    onSummarize: async () => {
      // Extract text from rendered content.
      const markdownText = getMarkdownTextFromContent(contentRoot);
      const bullets = await generateSummary({ slug, markdownText });
      return bullets;
    },
  });
}

// Auto init when module loads
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initBlogSummarizer());
} else {
  initBlogSummarizer();
}
