import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

let tokenize, memory;
try {
  const bytes = await readFile(new URL("./pkg/tint_tokenizer_bg.wasm", import.meta.url));
  const module = await import("./pkg/tint_tokenizer.js");
  memory = module.initSync({ module: bytes }).memory;
  tokenize = module.tokenize;
} catch (error) {
  if (error.code !== "ENOENT" && error.code !== "ERR_MODULE_NOT_FOUND") throw error;
}

test("generated WASM exports Uint32Array with exact feature packing and UTF-16 ends", { skip: !tokenize && "Build tint-tokenizer WASM first." }, () => {
  const examples = [
    ["a", [1, 5, 22, 182, 310]],
    ["\r\n", [3, 6, 37, 98, 223]],
    ["\u{1f600}", [1, 5, 53, 212, 340]],
    ["\u2003", [2, 5, 37, 212, 340]],
    ["+", [4, 5, 53, 128, 256]],
  ];
  let source = "", end = 0;
  const expected = [];
  for (const [text, first] of examples) {
    source += text;
    end += text.length;
    let hash = 2166136261;
    for (const byte of new TextEncoder().encode(text)) hash = Math.imul(hash ^ byte, 16777619) >>> 0;
    const features = [...first, 341 + hash % 1024];
    expected.push(end, features[0] | features[1] << 11, features[2] | features[3] << 11, features[4] | features[5] << 11);
  }
  const result = tokenize(source);
  assert.ok(result instanceof Uint32Array);
  assert.deepEqual(result, new Uint32Array(expected));
  assert.deepEqual(tokenize(""), new Uint32Array());
});

test("WASM enforces byte and token caps and remains usable after errors", { skip: !tokenize && "Build tint-tokenizer WASM first." }, () => {
  assert.throws(() => tokenize("+".repeat(4 * 1024 * 1024 + 1)), error => String(error).includes("4194304 tokens"));
  assert.throws(() => tokenize("\u2003".repeat(Math.floor(16 * 1024 * 1024 / 3) + 1)), error => String(error).includes("16 MiB"));
  assert.equal(tokenize("ok")[0], 2);
});

test("WASM reuses allocations across alternating input sizes without corrupting retained output", { skip: !tokenize && "Build tint-tokenizer WASM first." }, () => {
  const sources = ["a+".repeat(1024), "word+".repeat(16000), "x\u{1f600}\r\n".repeat(4096)];
  for (let round = 0; round < 3; round++) for (const source of sources) tokenize(source);
  const warmedBytes = memory.buffer.byteLength;
  const retained = tokenize("a+b");
  const expected = retained.slice();
  for (let round = 0; round < 20; round++) {
    for (const source of sources) {
      const tokens = tokenize(source);
      assert.equal(tokens.at(-4), source.length);
    }
  }
  assert.deepEqual(retained, expected);
  assert.equal(memory.buffer.byteLength, warmedBytes);
});
