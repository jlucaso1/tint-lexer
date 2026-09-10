import { getHighlighter } from "./index.js";

let highlighterPromise = null;

self.onmessage = async (event) => {
  const { id, code, mode, options } = event.data;
  try {
    highlighterPromise ??= getHighlighter(options);
    const highlighter = await highlighterPromise;
    if (mode === "flat") {
      const flat = await highlighter.parseFlat(code);
      self.postMessage({ id, flat }, [flat.buffer]);
    } else {
      const spans = await highlighter.highlight(code);
      self.postMessage({ id, spans });
    }
  } catch (error) {
    self.postMessage({ id, error: error?.message || String(error) });
  }
};
