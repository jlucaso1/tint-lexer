const MAX_SOURCE_BYTES = 256 * 1024;
let highlighter = typeof globalThis.highlighter !== "undefined" ? globalThis.highlighter : null;
let pending = null;
let running = false;
let failed = false;

async function drain() {
  if (running || !highlighter || failed) return;
  running = true;
  try {
    while (pending) {
      const { id, source } = pending;
      pending = null;
      postMessage({ type: "inference", id });
      try {
        const start = performance.now();
        const spans = await highlighter.highlight(source);
        postMessage({
          type: "result",
          id,
          spans,
          isCpu: highlighter.isCpu,
          elapsed: performance.now() - start,
        });
      } catch (error) {
        postMessage({ type: "error", id, message: String(error) });
      }
    }
  } finally {
    running = false;
  }
}

self.onmessage = ({ data }) => {
  if (data?.type === "cancel") {
    pending = null;
    return;
  }
  if (failed || data?.type !== "highlight" || !Number.isSafeInteger(data.id)) return;
  if (
    typeof data.source !== "string" ||
    data.source.length > MAX_SOURCE_BYTES ||
    new TextEncoder().encode(data.source).length > MAX_SOURCE_BYTES
  ) {
    pending = null;
    postMessage({ type: "error", id: data.id, message: "Input exceeds the 256 KiB demo limit." });
    return;
  }
  pending = { id: data.id, source: data.source };
  void drain();
};

async function initialize() {
  if (highlighter) return;
  try {
    postMessage({ type: "loading", message: "Initializing Tint highlighter (WebGPU / WASM CPU fallback)..." });
    const { Highlighter } = await import("./tint.js");
    highlighter = await Highlighter.load();
    const mode = highlighter.isCpu ? "CPU (WASM fallback)" : "WebGPU";
    postMessage({ type: "ready", mode, isCpu: highlighter.isCpu });
    void drain();
  } catch (error) {
    failed = true;
    pending = null;
    postMessage({ type: "unavailable", message: String(error) });
  }
}

void initialize();
