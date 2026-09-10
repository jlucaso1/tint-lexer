import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { decodeModel, Highlighter } from "./runtime.js";

function fixture(config = { radius: 0, embedding_dim: 1, hidden_dim: 1 }, state = false) {
  const { radius, embedding_dim: dim, hidden_dim: hidden } = config;
  const count = 1365 * dim + ((radius * 2 + 1) * dim + (state ? 9 : 0)) * hidden + hidden + hidden * 9 + 9;
  const bytes = new Uint8Array(Math.ceil(count / 64) * 4 + Math.ceil(count / 2));
  const view = new DataView(bytes.buffer);
  const expected = new Float32Array(count);
  let offset = 0;
  for (let start = 0; start < count; start += 64) {
    const length = Math.min(64, count - start);
    const scale = Math.fround((start + 1) / 100);
    view.setFloat32(offset, scale, true);
    offset += 4;
    for (let i = 0; i < length; i++) {
      const code = (start + i) % 15 + 1;
      bytes[offset + (i >> 1)] |= code << ((i & 1) * 4);
      expected[start + i] = (code - 8) * scale;
    }
    offset += Math.ceil(length / 2);
  }
  const metadata = {
    format_version: 1, feature_version: state ? 2 : 1, architecture: state ? "window-mlp-v2" : "window-mlp-v1",
    config: state ? { ...config, context_state: true } : config,
    classes: ["plain", "comment", "string", "number", "keyword", "type", "function", "constant", "operator"],
    quantization: "q4-block64-v1", weights_sha256: checksum(bytes), source_weights_sha256: "0".repeat(64),
  };
  return { metadata, bytes, expected };
}

function checksum(bytes) { return createHash("sha256").update(bytes).digest("hex"); }

test("q4 decoder preserves block order, signed nibbles, partial blocks and f32 rounding", async () => {
  for (const config of [{ radius: 0, embedding_dim: 1, hidden_dim: 1 }, { radius: 0, embedding_dim: 1, hidden_dim: 2 }, { radius: 2, embedding_dim: 2, hidden_dim: 3 }]) {
    const { metadata, bytes, expected } = fixture(config);
    const guarded = new Uint8Array(bytes.length + 8);
    guarded.set(bytes, 4);
    for (const input of [metadata, JSON.stringify(metadata)]) {
      const result = await decodeModel(input, guarded.subarray(4, -4));
      assert.deepEqual(result.weights, expected);
      assert.deepEqual(result.config, { ...config, context_state: false });
    }
  }
});

test("decoder accepts version two metadata with wider hidden input", async () => {
  const config = { radius: 1, embedding_dim: 2, hidden_dim: 3 };
  const { metadata, bytes, expected } = fixture(config, true);
  const result = await decodeModel(metadata, bytes);
  assert.deepEqual(result.weights, expected);
  assert.deepEqual(result.config, { ...config, context_state: true });
});

test("metadata rejects unknown, missing and unsupported fields", async () => {
  const { metadata, bytes } = fixture();
  for (const bad of [null, [], {}, { ...metadata, extra: true }, { ...metadata, feature_version: 2 },
    { ...metadata, architecture: "window-mlp-v2" },
    { ...metadata, config: { ...metadata.config, context_state: true } },
    { ...metadata, quantization: "q8" }, { ...metadata, source_weights_sha256: "X".repeat(64) },
    { ...metadata, classes: [...metadata.classes].reverse() },
    ...[{ radius: -1 }, { radius: 17 }, { radius: 0.5 }, { embedding_dim: 0 }, { embedding_dim: 129 }, { hidden_dim: 257 }, { hidden_dim: NaN }, { extra: 1 }]
      .map(change => ({ ...metadata, config: { ...metadata.config, ...change } }))]) {
    await assert.rejects(decodeModel(bad, bytes), /metadata|dimensions/);
  }
  const v2 = fixture({ radius: 0, embedding_dim: 1, hidden_dim: 1 }, true);
  for (const bad of [{ ...v2.metadata, feature_version: 1 }, { ...v2.metadata, architecture: "window-mlp-v1" },
    { ...v2.metadata, config: { radius: 0, embedding_dim: 1, hidden_dim: 1 } },
    { ...v2.metadata, config: { ...v2.metadata.config, context_state: false } }]) {
    await assert.rejects(decodeModel(bad, v2.bytes), /metadata|dimensions/);
  }
  await assert.rejects(decodeModel(" ".repeat(16384), bytes), /16 KiB/);
});

test("weights reject wrong lengths, checksum, scales, codes, overflow and padding", async () => {
  const { metadata, bytes } = fixture();
  await assert.rejects(decodeModel(metadata, bytes.subarray(1)), /byte count/);
  await assert.rejects(decodeModel(metadata, new Uint8Array(bytes.length)), /checksum/);
  await assert.rejects(decodeModel(metadata, Array.from(bytes)), /byte count/);
  for (const scale of [0, -1, Infinity, NaN, 3e38]) {
    const invalid = bytes.slice();
    new DataView(invalid.buffer).setFloat32(0, scale, true);
    await assert.rejects(decodeModel({ ...metadata, weights_sha256: checksum(invalid) }, invalid), /scale|Non-finite/);
  }
  for (const mutate of [value => { value[4] &= 240; }, value => { value[value.length - 1] |= 16; }]) {
    const invalid = bytes.slice();
    mutate(invalid);
    await assert.rejects(decodeModel({ ...metadata, weights_sha256: checksum(invalid) }, invalid), /code|padding/);
  }
});

test("decoder snapshots weights and configuration before yielding", async () => {
  const { metadata, bytes, expected } = fixture();
  const promise = decodeModel(metadata, bytes);
  bytes.fill(0);
  metadata.weights_sha256 = "f".repeat(64);
  metadata.config.radius = 16;
  const result = await promise;
  assert.deepEqual(result.weights, expected);
  assert.equal(result.config.radius, 0);
});

test("load validates artifacts before reporting missing WebGPU", async () => {
  const { metadata, bytes } = fixture();
  await assert.rejects(Highlighter.load({}, bytes), /metadata/);
  await assert.rejects(Highlighter.load(metadata, bytes), /WebGPU is unavailable/);
});

test("highlight validates token capacity before allocating GPU buffers", async () => {
  const model = new Highlighter();
  Object.assign(model, { failure: null, busy: false, tokenize: () => ({ length: 4194305 * 4 }) });
  await assert.rejects(model.highlight("x"), /4194304 tokens/);
  assert.equal(model.busy, false);
  model.tokenize = () => new Uint32Array();
  assert.deepEqual(await model.highlight(""), []);
  await assert.rejects(model.highlight(null), /string/);
  await assert.rejects(model.highlight("x".repeat(16 * 1024 * 1024 + 1)), /16 MiB/);
  model.busy = true;
  await assert.rejects(model.highlight("x"), /Concurrent/);
  model.failure = new Error("WebGPU device lost");
  await assert.rejects(model.highlight(""), /device lost/);
});
