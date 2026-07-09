// Non-module global initializer for blog summarizer.
// Loaded from each blog HTML page.

(function () {
  window.BlogSummary = window.BlogSummary || {};

  // In ES-module capable environments, assets/js/blog-summary/init.js will run
  // and directly mount the UI.
  // This file is a safe compatibility shim.
  window.BlogSummary.initBlogSummarizer = function () {
    // no-op: module init handles everything.
  };
})();


