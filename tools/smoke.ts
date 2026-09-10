import assert from "node:assert/strict";
import {
  highlight,
  highlightSync,
  parseFlat,
  parseFlatSync,
  forEachSpan,
  forEachSpanSync,
  highlightToHtml,
  highlightToHtmlSync,
  highlightAnsi,
  highlightAnsiSync,
  Highlighter,
  CLASSES,
} from "../dist/index.js";

interface SampleSnippet {
  name: string;
  code: string;
}

const snippets: SampleSnippet[] = [
  {
    name: "TypeScript Component",
    code: `import { useState } from "react";
export function App({ title }: { title: string }) {
  const [active, setActive] = useState(false);
  return <button onClick={() => setActive(!active)}>{title}</button>;
}`,
  },
  {
    name: "Rust Struct & Implementation",
    code: `pub struct BufferPool {
    capacity: usize,
    storage: Vec<u8>,
}
impl BufferPool {
    pub fn new(cap: usize) -> Self {
        Self { capacity: cap, storage: Vec::with_capacity(cap) }
    }
}`,
  },
  {
    name: "Python Async Worker",
    code: `import asyncio

async def fetch_item(item_id: int) -> dict:
    await asyncio.sleep(0.01)
    return {"id": item_id, "status": "processed"}
`,
  },
  {
    name: "JSON Config",
    code: `{
  "name": "tint-lexer",
  "private": true,
  "features": ["webgpu", "cpu-fallback", "flat-spans"]
}`,
  },
];

async function runSmokeTests(): Promise<void> {
  console.log("Running smoke validation across APIs and workloads...\n");

  const cpuHighlighter = await Highlighter.load({ preferCpu: true });
  assert.equal(cpuHighlighter.isCpu, true, "preferCpu must instantiate CPU engine");

  for (const snippet of snippets) {
    const start = performance.now();

    // 1. Sync and Async Span Highlights
    const syncSpans = highlightSync(snippet.code);
    const asyncSpans = await highlight(snippet.code);

    assert.ok(syncSpans.length > 0, `${snippet.name} should yield tokens`);
    assert.equal(syncSpans.length, asyncSpans.length, `${snippet.name} sync/async token length match`);

    // Verify token bounds
    for (const span of syncSpans) {
      assert.ok(span.start >= 0 && span.end <= snippet.code.length, "Token bounds within source");
      assert.ok(span.start <= span.end, "Span start <= end");
      assert.ok(CLASSES.includes(span.type), "Valid class category");
    }

    // 2. Flat Spans API
    const flatSync = parseFlatSync(snippet.code);
    const flatAsync = await parseFlat(snippet.code);
    assert.equal(flatSync.length, syncSpans.length * 3, "Flat span packing size");
    assert.deepEqual(flatSync, flatAsync, "Flat span sync vs async match");

    // 3. Zero-alloc Iterator API
    let count = 0;
    forEachSpanSync(snippet.code, (start, end, classId, className) => {
      assert.equal(className, CLASSES[classId]);
      count++;
    });
    assert.equal(count, syncSpans.length, "forEachSpan count matches");

    // 4. HTML Formatter Direct
    const html = highlightToHtmlSync(snippet.code, { pre: true });
    assert.ok(html.startsWith('<pre class="tint"><code>'), "HTML pre tag open");
    assert.ok(html.endsWith("</code></pre>"), "HTML pre tag close");

    // 5. ANSI Formatter Direct
    const ansi = highlightAnsiSync(snippet.code);
    assert.ok(ansi.length >= snippet.code.length, "ANSI output contains source text");

    const elapsed = (performance.now() - start).toFixed(2);
    console.log(`✓ ${snippet.name.padEnd(30)}: ${syncSpans.length} spans (${elapsed} ms)`);
  }

  // Edge cases
  assert.deepEqual(highlightSync(""), [], "Empty source yields empty spans");
  assert.deepEqual(parseFlatSync(""), new Uint32Array(0), "Empty source yields empty flat array");
  let called = false;
  forEachSpanSync("", () => { called = true; });
  assert.equal(called, false, "Empty source does not invoke callback");

  cpuHighlighter.dispose();
  console.log("\nAll smoke checks passed successfully.\n");
}

runSmokeTests().catch((err: Error) => {
  console.error("Smoke check failed:", err);
  process.exit(1);
});
