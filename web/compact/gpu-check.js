import initializeTokenizer, { decodeModel, Highlighter } from "./runtime.js";

function reference(tokens, config, weights) {
  const { radius, embedding_dim: dim, hidden_dim: hidden } = config;
  const count = tokens.length / 4;
  const input = (radius * 2 + 1) * dim;
  const hw = 1365 * dim, hb = hw + input * hidden, ow = hb + hidden, ob = ow + hidden * 9;
  const embedded = new Float32Array((count + radius * 2) * dim);
  const round = Math.fround;
  for (let t = 0; t < count + radius * 2; t++) {
    const index = t - radius;
    for (let d = 0; d < dim; d++) {
      let sum = 0;
      for (let f = 0; f < 6; f++) {
        const id = index < 0 || index >= count ? 0 : (tokens[index * 4 + 1 + (f >> 1)] >> ((f & 1) * 11)) & 2047;
        sum = round(sum + weights[id * dim + d]);
      }
      embedded[t * dim + d] = sum / 6;
    }
  }
  const activations = new Float32Array(hidden);
  const classes = ["plain", "comment", "string", "number", "keyword", "type", "function", "constant", "operator"];
  const spans = [];
  let start = 0;
  for (let t = 0; t < count; t++) {
    const kind = tokens[t * 4 + 1] & 2047;
    let best = 0, maximum = -Infinity;
    if (kind !== 2 && kind !== 3) {
      for (let d = 0; d < hidden; d++) {
        let sum = 0;
        for (let k = 0; k < input; k++) sum = round(sum + round(embedded[t * dim + k] * weights[hw + k * hidden + d]));
        activations[d] = Math.tanh(round(sum + weights[hb + d]));
      }
      for (let c = 0; c < 9; c++) {
        let sum = 0;
        for (let k = 0; k < hidden; k++) sum = round(sum + round(activations[k] * weights[ow + k * 9 + c]));
        const value = round(sum + weights[ob + c]);
        if (value > maximum) { best = c; maximum = value; }
      }
    }
    const end = tokens[t * 4], label = classes[best];
    if (spans.at(-1)?.class === label) spans.at(-1).end = end;
    else spans.push({ start, end, class: label });
    start = end;
  }
  return spans;
}

export async function checkGPU() {
  const tokenize = await initializeTokenizer();
  const reports = [];
  for (const config of [...[0, 2, 16].map(radius => ({ radius, embedding_dim: 3, hidden_dim: 5 })),
    { radius: 16, embedding_dim: 128, hidden_dim: 256 }, { radius: 16, embedding_dim: 128, hidden_dim: 127 },
    { radius: 0, embedding_dim: 1, hidden_dim: 1 }]) {
    const { radius, embedding_dim: dim, hidden_dim: hidden } = config;
    const count = 1365 * dim + (radius * 2 + 1) * dim * hidden + hidden + hidden * 9 + 9;
    const bytes = new Uint8Array(Math.ceil(count / 64) * 4 + Math.ceil(count / 2));
    const view = new DataView(bytes.buffer);
    let offset = 0, state = 123456789;
    for (let start = 0; start < count; start += 64) {
      view.setFloat32(offset, dim === 1 ? 1 : 0.08, true);
      offset += 4;
      const length = Math.min(64, count - start);
      for (let i = 0; i < length; i++) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        bytes[offset + (i >> 1)] |= (dim === 1 ? 8 : state % 15 + 1) << ((i & 1) * 4);
      }
      offset += Math.ceil(length / 2);
    }
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), value => value.toString(16).padStart(2, "0")).join("");
    const metadata = { format_version: 1, feature_version: 1, architecture: "window-mlp-v1", config,
      classes: ["plain", "comment", "string", "number", "keyword", "type", "function", "constant", "operator"],
      quantization: "q4-block64-v1", weights_sha256: hash, source_weights_sha256: "0".repeat(64) };
    const { weights } = await decodeModel(metadata, bytes);
    const model = await Highlighter.load(metadata, bytes);
    try {
      const weightsBuffer = model.buffers[0];
      if (model.buffers.length !== 1 || model.capacity !== 0) throw new Error("Document buffers were allocated eagerly.");
      const maximum = 4194304;
      const projectedBytes = weightsBuffer.size + maximum * 24 +
        (model.tile + 2 * radius) * dim * 4 + model.tile * hidden * 4 + Math.ceil(maximum / model.tile) * model.alignment;
      if (projectedBytes >= 128 * 1024 * 1024) throw new Error("Maximum-capacity GPU allocation exceeds 128 MiB.");
      const invocations = model.device.limits.maxComputeWorkgroupsPerDimension * 64;
      if ((model.tile + 2 * radius) * dim > invocations || model.tile * hidden > invocations) throw new Error("Tile exceeds dispatch limits.");
      let submissions = 0, uploads = 0, maps = 0, passes = 0;
      const queue = model.device.queue;
      const submit = queue.submit.bind(queue), write = queue.writeBuffer.bind(queue);
      queue.submit = (...args) => { submissions++; return submit(...args); };
      queue.writeBuffer = (...args) => { uploads++; return write(...args); };
      const buffer = model.buffer;
      model.buffer = (...args) => {
        const value = buffer(...args);
        if (value.usage & GPUBufferUsage.MAP_READ) {
          const map = value.mapAsync.bind(value);
          value.mapAsync = (...args) => { maps++; return map(...args); };
        }
        return value;
      };
      const encode = model.device.createCommandEncoder.bind(model.device);
      model.device.createCommandEncoder = (...args) => {
        const encoder = encode(...args), begin = encoder.beginComputePass.bind(encoder);
        encoder.beginComputePass = (...args) => { passes++; return begin(...args); };
        return encoder;
      };
      const active = model.highlight("x");
      await model.highlight("x").then(() => { throw new Error("Concurrent highlight was accepted."); }, error => {
        if (!error.message.includes("Concurrent")) throw error;
      });
      await active;
      const sources = ["", "x", "\r\n \t", "\u{1f600}a\u{301} + z\r\n"];
      if (dim === 3) {
        const long = "a+/*42*/\r\n".repeat(8200), packed = tokenize(long);
        for (const count of [32767, 32768, 32769, 65539]) sources.push(long.slice(0, packed[(count - 1) * 4]));
        sources.push("x", sources[6], "\r\n", sources[7], "");
      }
      if (dim === 1) sources.push("+".repeat(1048577), "x", "");
      for (const source of sources) {
        const packed = tokenize(source);
        const expected = dim === 1 ? source ? [{ start: 0, end: source.length, class: "plain" }] : [] : reference(packed, config, weights);
        const capacity = model.capacity, buffers = model.buffers.slice();
        submissions = uploads = maps = passes = 0;
        const actual = await model.highlight(source);
        const calls = source ? 1 : 0;
        if (submissions !== calls || maps !== calls || passes !== calls || uploads !== calls * 2) throw new Error("Inference did not use one pass, submission, readback and two uploads.");
        if (packed.length / 4 <= capacity && buffers.some((buffer, i) => buffer !== model.buffers[i])) throw new Error("Buffers changed without growth.");
        if (model.buffers[0] !== weightsBuffer || model.buffers.length !== 7) throw new Error("Weights or buffer ownership changed.");
        if (model.capacity < packed.length / 4 || (model.capacity & (model.capacity - 1))) throw new Error("Invalid document capacity.");
        const gpuBytes = model.buffers.reduce((sum, buffer) => sum + buffer.size, 0);
        if (gpuBytes >= 128 * 1024 * 1024) throw new Error("GPU allocation exceeded 128 MiB.");
        if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`GPU/CPU mismatch at radius ${radius}, ${packed.length / 4} tokens.`);
        if (packed.length && packed[packed.length - 4] !== source.length) throw new Error("Tokenizer UTF-16 end mismatch.");
        reports.push({ ...config, tokens: packed.length / 4, spans: actual.length, tile: model.tile, capacity: model.capacity, gpuBytes, projectedBytes });
      }
      model.device.destroy();
      await model.device.lost;
      await model.highlight("x").then(() => { throw new Error("Lost GPU accepted input."); }, error => {
        if (!error.message.includes("device lost")) throw error;
      });
    } finally {
      model.dispose();
    }
    await model.highlight("").then(() => { throw new Error("Disposed model accepted input."); }, error => {
      if (!error.message.includes("disposed")) throw error;
    });
  }
  return reports;
}

export async function checkArtifact(metadata, bytes, source) {
  const tokenize = await initializeTokenizer();
  const { config, weights } = await decodeModel(metadata, bytes);
  const model = await Highlighter.load(metadata, bytes);
  try {
    const expected = reference(tokenize(source), config, weights);
    const start = performance.now();
    const actual = await model.highlight(source);
    const elapsed = performance.now() - start;
    if (JSON.stringify(expected) !== JSON.stringify(actual)) throw new Error("Artifact GPU/CPU span mismatch.");
    return { spans: actual.length, elapsed, gpuBytes: model.buffers.reduce((sum, buffer) => sum + buffer.size, 0) };
  } finally {
    model.dispose();
  }
}

export async function checkNativeArtifact() {
  const [metadata, bytes, reference] = await Promise.all([
    fetch("/model.q4.json").then(response => response.json()),
    fetch("/model.q4.bin").then(response => response.arrayBuffer()),
    fetch("/bench/generated/native-reference.json").then(response => response.json()),
  ]);
  if (metadata.weights_sha256 !== reference.weights_sha256) throw new Error("Native reference belongs to another artifact.");
  const model = await Highlighter.load(metadata, new Uint8Array(bytes));
  const normalize = spans => spans.map(span => [span.start, span.end, span.class]);
  try {
    const checks = [];
    for (const document of reference.documents) {
      const actual = await model.highlight(document.source);
      if (JSON.stringify(normalize(actual)) !== JSON.stringify(normalize(document.spans))) throw new Error(`Burn/WebGPU mismatch in ${document.id}.`);
      checks.push({ id: document.id, spans: actual.length });
    }
    return checks;
  } finally {
    model.dispose();
  }
}
