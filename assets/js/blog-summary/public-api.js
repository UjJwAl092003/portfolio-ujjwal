// Global namespace export for the blog summarizer.
// Allows blog-renderer.js (non-module) to trigger the summarizer after render.

import { initBlogSummarizer as init } from "./init.js";

(function () {
  window.BlogSummary = window.BlogSummary || {};
  window.BlogSummary.initBlogSummarizer = function () {
    if (window.__debugInstrumentation && window.__debugInstrumentation.initLog)
      window.__debugInstrumentation.initLog("AI Summary", "ai-summary");
    // If the API is called multiple times, init.js guards and will no-op.

    try {
      init();
    } catch (e) {
      // Fail silently to avoid breaking blog pages.
    }
  };
})();
