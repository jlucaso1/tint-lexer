import { initLiveBenchmark } from "./bench-live.js";

const samples = {
  rust: `// Word counts are case-sensitive.
use std::collections::HashMap;

fn count_words(source: &str) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for word in source.split_whitespace() {
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}

fn main() {
    let message = "hello tint hello";
    println!("{:?}", count_words(message));
}
`,
  typescript: `// Readings equal to the limit are excluded.
type Reading = { name: string; value: number };

function aboveThreshold(readings: Reading[], limit = 12): string[] {
  return readings
    .filter(({ value }) => value > limit)
    .map(({ name }) => name);
}

const readings: Reading[] = [
  { name: "north", value: 18 },
  { name: "south", value: 9 },
];

console.log(aboveThreshold(readings));
`,
  python: `# Normalize words before counting them.
from collections import Counter

def count_words(source: str) -> dict[str, int]:
    words = source.lower().split()
    return dict(Counter(words))

message = "hello tint hello"
counts = count_words(message)

for word, count in sorted(counts.items()):
    print(f"{word}: {count}")
`,
};
const source = document.querySelector("#source");
const code = document.querySelector("#code");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const outputState = document.querySelector("#output-state");
const badge = document.querySelector(".local");
const classes = new Set(["plain", "comment", "string", "number", "keyword", "type", "function", "constant", "operator"]);
const MAX_BYTES = 256 * 1024;
const MAX_DOM_SPANS = 12000;
let generation = 0;
let ready = false;
let unavailable = false;
let isCpuMode = false;
let worker;

function showError(message) {
  error.textContent = message;
  error.hidden = false;
}

function edit() {
  generation += 1;
  code.textContent = source.value;
  outputState.textContent = "PLAIN TEXT";
  const bytes = new TextEncoder().encode(source.value).length;
  document.querySelector("#size").textContent = `${bytes.toLocaleString()} B`;
  if (bytes > MAX_BYTES) {
    worker?.postMessage({ type: "cancel" });
    showError("Input exceeds 256 KiB. Shorten it to resume inference; output remains plain text.");
    status.textContent = "Input too large. Inference paused.";
    return;
  }
  if (unavailable) return;
  error.hidden = true;
  if (ready) status.textContent = isCpuMode ? "Queued for CPU inference..." : "Queued for GPU inference...";
  worker?.postMessage({ type: "highlight", id: generation, source: source.value });
}

function render(spans) {
  if (!Array.isArray(spans)) throw new Error("The model returned invalid spans.");
  if (spans.length > MAX_DOM_SPANS) {
    outputState.textContent = "PLAIN TEXT";
    showError("The result exceeds the 12,000-span display limit. Output remains plain text; try a smaller input.");
    return;
  }
  let end = 0;
  const fragment = document.createDocumentFragment();
  for (const span of spans) {
    if (!Number.isSafeInteger(span.start) || !Number.isSafeInteger(span.end) || span.start !== end || span.end <= end || span.end > source.value.length || !classes.has(span.class)) {
      throw new Error("The model returned invalid span offsets or classes.");
    }
    const text = source.value.slice(span.start, span.end);
    if (span.class === "plain") fragment.append(document.createTextNode(text));
    else {
      const node = document.createElement("span");
      node.className = `syntax-${span.class}`;
      node.textContent = text;
      fragment.append(node);
    }
    end = span.end;
  }
  if (end !== source.value.length) throw new Error("The model result does not cover the source.");
  code.replaceChildren(fragment);
  outputState.textContent = "PREDICTED LABELS";
}

source.value = samples.rust;
source.addEventListener("input", edit);
document.querySelector("#sample").addEventListener("change", (event) => {
  if (event.target.value === "auto") {
    source.value = "";
    source.focus();
    edit();
    return;
  }
  source.value = samples[event.target.value];
  edit();
});

try {
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "loading") { status.textContent = data.message; return; }
    if (data.type === "ready") {
      ready = true;
      isCpuMode = data.isCpu === true;
      status.textContent = `${data.mode} ready.`;
      if (badge) badge.textContent = isCpuMode ? "CPU FALLBACK (WASM)" : "WEBGPU ACCELERATED";
      return;
    }
    if (data.type === "unavailable") {
      unavailable = true;
      status.textContent = "Model unavailable. Showing plain text.";
      showError(data.message);
      return;
    }
    if (data.id !== generation) return;
    if (data.type === "inference") {
      status.textContent = isCpuMode ? "Running CPU inference..." : "Running GPU inference...";
    }
    if (data.type === "error") {
      status.textContent = "Inference failed. Showing plain text.";
      showError(data.message);
    }
    if (data.type === "result") {
      try {
        render(data.spans);
        const engine = data.isCpu ? "CPU (WASM)" : "GPU (WebGPU)";
        status.textContent = `${engine} inference complete in ${data.elapsed.toFixed(1)} ms. Includes tokenization.`;
      } catch (cause) {
        status.textContent = "Invalid model result. Showing plain text.";
        showError(String(cause));
      }
    }
  };
  worker.onerror = (event) => {
    unavailable = true;
    worker.terminate();
    status.textContent = "Worker unavailable. Showing plain text.";
    code.textContent = source.value;
    outputState.textContent = "PLAIN TEXT";
    showError(event.message || "The worker stopped unexpectedly. Reload to retry.");
  };
} catch (cause) {
  unavailable = true;
  status.textContent = "Worker unavailable. Showing plain text.";
  showError(String(cause));
}
edit();
initLiveBenchmark();
