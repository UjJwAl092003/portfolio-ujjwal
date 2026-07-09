import { createMemoryCache } from "./cache.js";
import { extractiveSummarize } from "./extractive-summarizer.js";

const sessionCache = createMemoryCache();

function getApiKey() {
  // Future: wire to real API keys via env/config. For now, keep blank by default.
  try {
    return localStorage.getItem("BLOG_SUMMARY_API_KEY") || "";
  } catch (_) {
    return "";
  }
}

async function tryLLMSummarize({ markdownText }) {
  // Placeholder for future LLM integration.
  // Return null to fall back to local summarizer.
  const apiKey = getApiKey();
  if (!apiKey) return null;

  // If you later connect OpenAI/Gemini/Claude/etc, implement it here.
  // Keep the extraction/UI unchanged.
  return null;
}

export async function generateSummary({ slug, markdownText }) {
  const cacheKey = `blog-summary:${slug || "unknown"}`;

  if (sessionCache.has(cacheKey)) {
    return sessionCache.get(cacheKey);
  }

  // 1) Try LLM (if configured in the future)
  const llm = await tryLLMSummarize({ markdownText });
  if (llm && Array.isArray(llm) && llm.length) {
    sessionCache.set(cacheKey, llm);
    return llm;
  }

  // 2) Local extractive summary
  const bullets = extractiveSummarize(markdownText, {
    minBullets: 5,
    maxBullets: 8,
  });
  sessionCache.set(cacheKey, bullets);
  return bullets;
}
