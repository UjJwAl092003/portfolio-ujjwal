function normalizeText(s) {
  return String(s)
    .replace(/\s+/g, " ")
    .replace(/\u00A0/g, " ")
    .trim();
}

function splitSentences(text) {
  // Keep it lightweight: split on punctuation followed by whitespace/newline.
  // This is intentionally conservative to avoid mangling code blocks (we strip code upstream).
  const t = normalizeText(text);
  if (!t) return [];

  const parts = t
    .split(/(?<=[.!?])\s+(?=[A-Z0-9])/g)
    .flatMap((p) => (p ? [p.trim()] : []));

  // Fallback if above fails (e.g., headings in markdown not matching caps pattern)
  if (parts.length <= 1) {
    return t
      .split(/(?<=[.!?])\s+/g)
      .map((p) => p.trim())
      .filter(Boolean);
  }

  return parts;
}

function tokenize(text) {
  return normalizeText(text)
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/g)
    .filter(Boolean);
}

function buildTermFreq(tokens) {
  const tf = new Map();
  for (const tok of tokens) {
    tf.set(tok, (tf.get(tok) || 0) + 1);
  }
  return tf;
}

function sentenceScore(sentence, tf, headingBoostTokens) {
  const tokens = tokenize(sentence);
  if (!tokens.length) return 0;

  // Term overlap score
  let score = 0;
  for (const tok of tokens) {
    const w = tf.get(tok) || 0;
    if (w) score += w;
  }

  // Penalize very long sentences slightly to keep bullets concise.
  score *= 1 / (1 + Math.max(0, tokens.length - 22) / 40);

  // Heading keyword boosts (very lightweight)
  for (const hb of headingBoostTokens) {
    if (hb && sentence.toLowerCase().includes(hb)) score += 8;
  }

  return score;
}

function pickHighlights(sentences, scores, desiredCount) {
  // Greedy selection with basic diversity (don't pick near-duplicates).
  const chosen = [];
  const chosenTexts = [];

  for (let k = 0; k < desiredCount; k++) {
    let bestIdx = -1;
    let bestScore = -Infinity;

    for (let i = 0; i < sentences.length; i++) {
      if (chosen.includes(i)) continue;
      const s = sentences[i];

      // De-dup by overlap of normalized tokens.
      const norm = tokenize(s).slice(0, 16).join(" ");
      let isDup = false;
      for (const ct of chosenTexts) {
        if (ct && norm && (ct.includes(norm) || norm.includes(ct))) {
          isDup = true;
          break;
        }
      }
      if (isDup) continue;

      const sc = scores[i] || 0;
      if (sc > bestScore) {
        bestScore = sc;
        bestIdx = i;
      }
    }

    if (bestIdx === -1) break;
    chosen.push(bestIdx);
    chosenTexts.push(tokenize(sentences[bestIdx]).slice(0, 16).join(" "));
  }

  // Keep the original order for readability.
  chosen.sort((a, b) => a - b);
  return chosen.map((idx) => sentences[idx]);
}

export function extractiveSummarize(markdownText, opts = {}) {
  const { minBullets = 5, maxBullets = 8 } = opts;

  const raw = normalizeText(markdownText);
  if (!raw) return [];

  // Remove fenced code blocks as they distort sentence scoring.
  const textNoCode = raw.replace(/```[\s\S]*?```/g, " ");
  // Remove inline code.
  const textNoInline = textNoCode.replace(/`[^`]*`/g, " ");

  const sentences = splitSentences(textNoInline);
  if (!sentences.length) return [];

  // Term frequency over the whole text.
  const tokens = tokenize(textNoInline);
  const tf = buildTermFreq(tokens);

  // Very small heading keyword set to help surface concepts.
  // (Extract some frequent capitalized/known words is overkill; just take frequent non-stopwords.)
  const headingBoostTokens = Array.from(tf.keys()).slice(0, 18);

  const scores = sentences.map((s) => sentenceScore(s, tf, headingBoostTokens));

  // Intro bias: first few sentences often contain the thesis.
  for (let i = 0; i < Math.min(4, scores.length); i++) {
    scores[i] += 18;
  }

  // Conclusion bias: last few sentences.
  for (let i = Math.max(0, scores.length - 3); i < scores.length; i++) {
    scores[i] += 12;
  }

  const desired = Math.max(
    minBullets,
    Math.min(maxBullets, Math.round((sentences.length / 18) * 6)),
  );

  const picked = pickHighlights(sentences, scores, desired);

  // Ensure 5–8 bullets when possible.
  if (picked.length < minBullets) {
    // Fill using top scored sentences not yet picked.
    const scoredIdx = sentences
      .map((_, i) => i)
      .sort((a, b) => (scores[b] || 0) - (scores[a] || 0));

    for (const idx of scoredIdx) {
      if (picked.length >= minBullets) break;
      if (!picked.includes(sentences[idx])) picked.push(sentences[idx]);
    }
  }

  return picked.slice(0, maxBullets);
}
