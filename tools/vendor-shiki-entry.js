// Source for web/vendor/shiki.js (committed build output).
// Rebuild with:
//   npx esbuild tools/vendor-shiki-entry.js --bundle --minify --format=esm \
//     --outfile=web/vendor/shiki.js
// Minimal Shiki setup for the live benchmark: JavaScript grammar plus one
// theme on the JavaScript regex engine, so no oniguruma WASM is fetched.
import { createHighlighterCore } from "@shikijs/core";
import { createJavaScriptRegexEngine } from "@shikijs/engine-javascript";
import javascript from "@shikijs/langs/javascript";
import githubDark from "@shikijs/themes/github-dark";

let highlighterPromise = null;
function getHighlighter() {
  return (highlighterPromise ??= createHighlighterCore({
    themes: [githubDark],
    langs: [javascript],
    engine: createJavaScriptRegexEngine(),
  }));
}

export async function highlight(code) {
  const highlighter = await getHighlighter();
  return highlighter.codeToHtml(code, { lang: "javascript", theme: "github-dark" });
}
