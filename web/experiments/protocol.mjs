export const REQUIRED_CASES = ['rust-1024', 'rust-65536', 'rust-262144', 'three-0.97.0-10x'];
export const SCRIPT_FILES = ['web/experiments/protocol.mjs', 'web/experiments/runner.mjs', 'web/experiments/worker.mjs'];
export const RUNTIME_FILES = ['web/compact/dist/runtime.js', 'web/compact/pkg/tint_tokenizer.js', 'web/compact/pkg/tint_tokenizer_bg.wasm'];
export const check = (condition, message) => { if (!condition) throw new Error(message); };
export const decode = bytes => new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
export async function sha256(bytes) {
  if (typeof bytes === 'string') bytes = new TextEncoder().encode(bytes);
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), byte => byte.toString(16).padStart(2, '0')).join('');
}
export function generatedURL(value, origin, prefix = '/experiments/generated/') {
  check(typeof value === 'string' && value.startsWith(prefix) && !/[\\%?#\s]/u.test(value)
    && value.split('/').slice(1).every(part => part && part !== '.' && part !== '..'), 'Unsafe generated URL');
  const url = new URL(value, origin);
  check(url.origin === origin && url.pathname === value, 'Generated URL must be same-origin');
  return url.href;
}
export async function download(url, limit = 64 * 1024 * 1024) {
  const response = await fetch(url, { cache: 'no-store', redirect: 'error', signal: AbortSignal.timeout(60000) });
  check(response.ok && response.body, `Download failed ${url}`);
  const chunks = []; let length = 0;
  for await (const chunk of response.body) {
    length += chunk.length;
    check(length <= limit, `Download exceeds byte limit ${url}`);
    chunks.push(chunk);
  }
  const bytes = new Uint8Array(length); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}
export async function verifiedDownload(url, expected, limit) {
  const bytes = await download(url, limit);
  check(await sha256(bytes) === expected, `SHA-256 mismatch ${url}`);
  return bytes;
}
export function validateSpans(source, spans) {
  const classes = ['plain', 'comment', 'string', 'number', 'keyword', 'type', 'function', 'constant', 'operator'];
  check(Array.isArray(spans), 'Invalid spans');
  let end = 0;
  for (const span of spans) {
    check(span && Number.isSafeInteger(span.start) && Number.isSafeInteger(span.end)
      && span.start === end && span.end > end && span.end <= source.length && classes.includes(span.class), 'Invalid span coverage or class');
    end = span.end;
    check(!(end < source.length && source.charCodeAt(end - 1) >= 0xd800 && source.charCodeAt(end - 1) <= 0xdbff
      && source.charCodeAt(end) >= 0xdc00 && source.charCodeAt(end) <= 0xdfff), 'Span splits UTF-16 surrogate pair');
  }
  check(end === source.length, 'Incomplete source coverage');
}
export function summarize(samples) {
  check(Array.isArray(samples) && samples.length === 9 && samples.every(sample => sample && Number.isFinite(sample.elapsed_ms) && sample.elapsed_ms > 0
    && Number.isSafeInteger(sample.span_count) && sample.span_count > 0), 'Invalid timing samples');
  const times = samples.map(sample => sample.elapsed_ms).sort((a, b) => a - b);
  return { median_ms: times[4], p95_ms: times[8] };
}
export function validateManifest(manifest, path, origin) {
  generatedURL(path, origin);
  check(/^\/experiments\/generated\/[A-Za-z0-9][A-Za-z0-9_-]*\/manifest\.json$/.test(path), 'Invalid manifest path');
  const prefix = path.slice(0, -'manifest.json'.length);
  check(manifest.schema_version === 1 && manifest.iterations === 9 && manifest.warmups === 3, 'Invalid benchmark protocol');
  const expectedKeys = ['baseline_weights_sha256', 'candidate_weights_sha256', 'runtime_sha256', 'corpus_sha256'];
  check(manifest.expected && Object.keys(manifest.expected).length === expectedKeys.length
    && expectedKeys.every(key => /^[a-f0-9]{64}$/.test(manifest.expected[key])), 'Invalid experiment identity');
  for (const [entries, files, digest] of [[manifest.scripts, SCRIPT_FILES, manifest.benchmark_sha256],
    [manifest.runtime, RUNTIME_FILES, manifest.expected?.runtime_sha256]]) {
    check(Array.isArray(entries) && entries.length === files.length && /^[a-f0-9]{64}$/.test(digest), 'Invalid payload manifest');
    entries.forEach((entry, i) => check(entry.file === files[i] && /^[a-f0-9]{64}$/.test(entry.sha256), 'Invalid payload entry'));
  }
  for (const variant of ['baseline', 'candidate']) {
    const model = manifest.models?.[variant];
    check(model && model.metadata_url === `${prefix}${variant}.q4.json` && model.weights_url === `${prefix}${variant}.q4.bin`
      && /^[a-f0-9]{64}$/.test(model.metadata_sha256) && model.weights_sha256 === manifest.expected[`${variant}_weights_sha256`], 'Invalid model manifest');
    generatedURL(model.metadata_url, origin, prefix); generatedURL(model.weights_url, origin, prefix);
  }
  check(Array.isArray(manifest.cases) && manifest.cases.length === 4, 'Invalid benchmark cases');
  manifest.cases.forEach((item, i) => {
    const size = [1024, 65536, 262144, 5556500][i];
    check(item.id === REQUIRED_CASES[i] && /^[a-f0-9]{64}$/.test(item.source_sha256)
      && /^[a-f0-9]{64}$/.test(item.fixture_sha256) && item.bytes === size
      && item.chars === size && item.copies === (i === 3 ? 10 : 1), 'Invalid benchmark case');
    check(item.source_url === `${prefix}${item.id}.source`, 'Invalid source URL');
    generatedURL(item.source_url, origin, prefix);
  });
  return prefix;
}
