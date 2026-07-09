// Browser-friendly namespace wrapper for the blog summarizer.
// Purpose: allow non-module pages to call initBlogSummarizer without refactoring existing code.

(function () {
  // Dynamic script loading is intentionally avoided here.
  // This file expects to be loaded AFTER the module files in the browser, which we ensure via blog-renderer.js.
  // If ES modules are not supported by the environment, the features won't load.
  // Since the rest of the site uses plain script tags, we keep this wrapper minimal.
})();
