export function mountSummaryUI({ container, getCacheKey, onSummarize }) {
  if (!container) return;

  const button = document.createElement("button");
  button.type = "button";
  button.className = "ai-summary-button";
  button.innerHTML = "✨ AI Summary";

  const card = document.createElement("div");
  card.className = "ai-summary-card";

  const label = document.createElement("div");
  label.className = "ai-summary-label";
  label.textContent = "AI Generated Summary";

  const statusWrap = document.createElement("div");
  statusWrap.className = "ai-summary-status";

  const contentWrap = document.createElement("div");
  contentWrap.className = "ai-summary-content";

  card.appendChild(label);
  card.appendChild(statusWrap);
  card.appendChild(contentWrap);

  container.appendChild(button);
  container.appendChild(card);

  let isGenerating = false;

  function setStatus({ text, mode }) {
    // mode: 'loading' | 'error'
    statusWrap.innerHTML = "";
    statusWrap.classList.remove("is-error");

    if (mode === "error") {
      statusWrap.classList.add("is-error");
      statusWrap.textContent = text || "Unable to generate summary.";
      const retry = document.createElement("button");
      retry.type = "button";
      retry.className = "ai-summary-retry";
      retry.textContent = "Retry";
      retry.addEventListener("click", () => start());
      statusWrap.appendChild(retry);
      return;
    }

    if (mode === "loading") {
      const spinner = document.createElement("span");
      spinner.className = "ai-summary-spinner";
      statusWrap.appendChild(spinner);
    }

    const p = document.createElement("p");
    p.textContent = text || "Generating AI summary...";
    statusWrap.appendChild(p);
  }

  function showSummary(bullets) {
    contentWrap.innerHTML = "";

    if (!bullets || !bullets.length) {
      setStatus({ text: "Unable to generate summary.", mode: "error" });
      return;
    }

    // Render as compact paragraphs/sections instead of a long bullet list.
    // bullets[] are short strings produced by extractive summarizer.
    const maxHighlights = Math.min(4, bullets.length);

    for (let i = 0; i < maxHighlights; i++) {
      const section = document.createElement("div");
      section.className = "ai-summary-section";

      const p = document.createElement("p");
      p.className = "ai-summary-text";
      p.textContent = bullets[i];

      section.appendChild(p);
      contentWrap.appendChild(section);
    }

    // Animate expand/fade
    contentWrap.classList.add("is-ready");
    statusWrap.innerHTML = "";
  }

  async function start() {
    if (isGenerating) return;
    isGenerating = true;

    // If already generated in this session, render from cache.
    const cacheKey = getCacheKey();
    if (!cacheKey) {
      // still allow generation
    }

    // status
    setStatus({ mode: "loading", text: "Generating AI summary..." });
    contentWrap.classList.remove("is-ready");

    try {
      const bullets = await onSummarize({ cacheKey });
      showSummary(bullets);
    } catch (e) {
      setStatus({ mode: "error", text: "Unable to generate summary." });
    } finally {
      isGenerating = false;
    }
  }

  button.addEventListener("click", () => start());
}
