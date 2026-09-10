import test from "node:test";
import assert from "node:assert/strict";
import {
  parse,
  parseSync,
  highlight,
  highlightSync,
  highlightToHtml,
  highlightToHtmlSync,
  highlightAnsi,
  highlightAnsiSync,
  tokenize,
  Highlighter,
  getHighlighter,
  parseFlat,
  parseFlatSync,
  forEachSpan,
  forEachSpanSync,
  CLASSES,
  type SyntaxSpan,
} from "../dist/index.js";

const sampleCode = `
import React, { useState } from 'react';

// Counter component
export function Counter({ initial = 0 }: { initial?: number }) {
  const [count, setCount] = useState(initial);
  return (
    <div className="counter">
      <span>{count}</span>
      <button onClick={() => setCount(count + 1)}>+1</button>
    </div>
  );
}
`;

test("parse and parseSync return consistent tokens", async () => {
  const asyncSpans: SyntaxSpan[] = await parse(sampleCode);
  const syncSpans: SyntaxSpan[] = parseSync(sampleCode);

  assert.ok(asyncSpans.length > 0, "Async spans should not be empty");
  assert.equal(asyncSpans.length, syncSpans.length, "Async and sync token count must match");

  for (let i = 0; i < asyncSpans.length; i++) {
    assert.equal(asyncSpans[i].type, syncSpans[i].type, `Token ${i} type must match`);
    assert.equal(asyncSpans[i].start, syncSpans[i].start, `Token ${i} start must match`);
    assert.equal(asyncSpans[i].end, syncSpans[i].end, `Token ${i} end must match`);
  }
});

test("highlight and highlightSync are aliases for parse and parseSync", async () => {
  const spansA = await highlight("const x = 42;");
  const spansB = highlightSync("const x = 42;");
  assert.deepEqual(spansA, spansB);
});

test("tokenize returns packed u32 words", () => {
  const tokens = tokenize("const x = 42;");
  assert.ok(tokens instanceof Uint32Array);
  assert.ok(tokens.length >= 4);
  assert.equal(tokens.length % 4, 0, "Token array length must be a multiple of 4");
});

test("Highlighter load with preferCpu forces CPU execution", async () => {
  const cpu = await Highlighter.load({ preferCpu: true });
  assert.equal(cpu.isCpu, true);

  const spans = await cpu.highlight("let value = 'test';");
  assert.ok(spans.length > 0);
  assert.ok(spans.some(s => s.type === "keyword"));
  assert.ok(spans.some(s => s.type === "string"));
  cpu.dispose();
});

test("Highlighter loadSync returns initialized CPU highlighter", () => {
  const cpu = Highlighter.loadSync();
  assert.equal(cpu.isCpu, true);

  const spans = cpu.highlightSync("function sum(a, b) { return a + b; }");
  assert.ok(spans.length > 0);
  assert.ok(spans.some(s => s.type === "function"));
  cpu.dispose();
});

test("concurrent highlight calls resolve cleanly in parallel", async () => {
  const snippets = [
    "const alpha = 1;",
    "function beta() { return 2; }",
    "class Gamma extends Delta {}",
    "// single line comment\nconst epsilon = true;",
    "export default { key: 'value' };",
  ];

  const results = await Promise.all(snippets.map(code => parse(code)));
  assert.equal(results.length, snippets.length);
  for (const spans of results) {
    assert.ok(spans.length > 0);
  }
});

test("handles edge cases: empty input and whitespace", async () => {
  assert.deepEqual(await parse(""), []);
  assert.deepEqual(parseSync(""), []);

  const wsSpans = await parse("   \n\t   \n  ");
  assert.ok(Array.isArray(wsSpans));
});

test("handles unicode, emoji, and multi-byte characters", async () => {
  const code = 'const 🚀 = "✨ unicode: áéíóú 漢字";';
  const spans = await parse(code);
  assert.ok(spans.length > 0);

  // Check that offsets do not throw and cover valid boundaries
  for (const span of spans) {
    assert.ok(span.start >= 0);
    assert.ok(span.end <= code.length);
    assert.ok(span.start <= span.end);
  }
});

test("WorkerHighlighter offloads highlighting to background worker", async () => {
  if (typeof Worker === "undefined") return;

  const worker = createWorkerHighlighter(new URL("../dist/worker.js", import.meta.url));
  const spans = await worker.highlight("const inWorker = true;");
  assert.ok(spans.length > 0);
  assert.ok(spans.some(s => s.type === "keyword"));
  worker.dispose();
});

test("highlightToHtml and highlightToHtmlSync render valid HTML spans", async () => {
  const code = "const count = 42; // comment";
  const htmlAsync = await highlightToHtml(code);
  const htmlSync = highlightToHtmlSync(code);

  assert.equal(htmlAsync, htmlSync);
  assert.ok(htmlAsync.includes('<span class="tint__token--keyword">const</span>'));
  assert.ok(htmlAsync.includes('<span class="tint__token--number">42</span>'));

  // Test custom prefix and pre wrapping
  const customHtml = highlightToHtmlSync(code, { classPrefix: "sh__token--", pre: true });
  assert.ok(customHtml.startsWith('<pre class="tint"><code>'));
  assert.ok(customHtml.endsWith("</code></pre>"));
  assert.ok(customHtml.includes('<span class="sh__token--keyword">const</span>'));
});

test("highlightAnsi and highlightAnsiSync format terminal color escape sequences", async () => {
  const code = "function test() {}";
  const ansiAsync = await highlightAnsi(code);
  const ansiSync = highlightAnsiSync(code);

  assert.equal(ansiAsync, ansiSync);
  assert.ok(ansiAsync.includes("\x1b[34mtest\x1b[0m"), "Function name should have ANSI color");
});

test("parseFlat and parseFlatSync return packed Uint32Array matching spans", async () => {
  const spans = parseSync(sampleCode);
  const flatSync = parseFlatSync(sampleCode);
  const flatAsync = await parseFlat(sampleCode);

  assert.ok(flatSync instanceof Uint32Array);
  assert.ok(flatAsync instanceof Uint32Array);
  assert.equal(flatSync.length, spans.length * 3);
  assert.deepEqual(flatSync, flatAsync);

  for (let i = 0; i < spans.length; i++) {
    const start = flatSync[i * 3 + 0];
    const end = flatSync[i * 3 + 1];
    const classId = flatSync[i * 3 + 2];
    assert.equal(start, spans[i].start);
    assert.equal(end, spans[i].end);
    assert.equal(CLASSES[classId], spans[i].type);
  }
});

test("forEachSpan and forEachSpanSync iterate zero-alloc without creating objects", async () => {
  const expectedSpans = parseSync(sampleCode);
  const visitedSync: Array<{ start: number; end: number; classId: number; className: string }> = [];

  forEachSpanSync(sampleCode, (start, end, classId, className) => {
    visitedSync.push({ start, end, classId, className });
  });

  assert.equal(visitedSync.length, expectedSpans.length);
  for (let i = 0; i < expectedSpans.length; i++) {
    assert.equal(visitedSync[i].start, expectedSpans[i].start);
    assert.equal(visitedSync[i].end, expectedSpans[i].end);
    assert.equal(visitedSync[i].className, expectedSpans[i].type);
  }

  const visitedAsync: typeof visitedSync = [];
  await forEachSpan(sampleCode, (start, end, classId, className) => {
    visitedAsync.push({ start, end, classId, className });
  });
  assert.deepEqual(visitedAsync, visitedSync);
});


