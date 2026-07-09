// Global namespace export for the blog summarizer.
// Allows blog-renderer.js (non-module) to trigger the summarizer after render.

import { initBlogSummarizer as init } from "./init.js";

(function () {
  window.BlogSummary = window.BlogSummary || {};
  window.BlogSummary.initBlogSummarizer = function () {
    try {
      init();
    } catch (e) {
      // Fail silently to avoid breaking blog pages.
    }
  };
})();
