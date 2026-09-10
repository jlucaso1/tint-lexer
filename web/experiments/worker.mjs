import { check, decode, generatedURL, verifiedDownload, sha256, validateManifest, validateSpans } from './protocol.mjs';

const query = new URL(self.location.href).searchParams;
const manifestPath = query.get('manifest'), variant = query.get('variant');
let engine, source, manifest;
let queue = Promise.resolve();
async function run(message) {
  if (message.type === 'initialize') {
    check(!engine && ['baseline', 'candidate'].includes(variant), 'Invalid worker initialization');
    manifest = JSON.parse(decode(await verifiedDownload(generatedURL(manifestPath, self.location.origin), query.get('sha256'))));
    validateManifest(manifest, manifestPath, self.location.origin);
    for (const [entries, expected] of [[manifest.scripts, manifest.benchmark_sha256], [manifest.runtime, manifest.expected.runtime_sha256]]) {
      check(await sha256(JSON.stringify(entries)) === expected, 'Payload digest mismatch');
      for (const entry of entries) await verifiedDownload(`/${entry.file.slice(4)}`, entry.sha256);
    }
    const model = manifest.models[variant];
    const metadata = await verifiedDownload(generatedURL(model.metadata_url, self.location.origin), model.metadata_sha256, 16383);
    const weights = await verifiedDownload(generatedURL(model.weights_url, self.location.origin), model.weights_sha256, 8 * 1024 * 1024);
    const config = JSON.parse(decode(metadata)).config;
    check(config?.radius === 4 && config.embedding_dim === 8 && config.hidden_dim === 64, 'Model config mismatch');
    const runtime = await import('/compact/dist/runtime.js');
    await runtime.default();
    engine = await runtime.Highlighter.load(decode(metadata), weights);
    const adapter = await navigator.gpu.requestAdapter();
    check(adapter, 'Adapter descriptor unavailable');
    const info = adapter.info;
    return { adapter: JSON.stringify(Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(key => [key, info?.[key] || 'unknown (privacy-limited)']))),
      adapter_note: 'Descriptor from a separate requestAdapter call with the runtime default options. Not an attestation of the runtime device.',
      weights_sha256: model.weights_sha256, runtime_sha256: manifest.expected.runtime_sha256 };
  }
  check(engine, 'Worker not initialized');
  if (message.type === 'source') {
    const item = manifest.cases.find(item => item.id === message.case_id);
    check(item, 'Unknown workload');
    const bytes = await verifiedDownload(generatedURL(item.source_url, self.location.origin), item.fixture_sha256, item.bytes / item.copies);
    source = decode(bytes).repeat(item.copies);
    const encoded = new TextEncoder().encode(source);
    check(source.length === item.chars && encoded.length === item.bytes && await sha256(encoded) === item.source_sha256, 'Resident source checksum mismatch');
    return { source_sha256: item.source_sha256, bytes: encoded.length, chars: source.length };
  }
  if (message.type === 'sample') {
    check(typeof source === 'string', 'Source not loaded');
    const start = performance.now();
    const spans = await engine.highlight(source);
    const elapsed_ms = performance.now() - start;
    validateSpans(source, spans);
    check(Number.isFinite(elapsed_ms) && elapsed_ms > 0, 'Invalid worker timing');
    return { elapsed_ms, span_count: spans.length };
  }
  throw new Error('Unknown worker operation');
}
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try { self.postMessage({ id: data.id, result: await run(data) }); }
    catch (error) { self.postMessage({ id: data.id, error: String(error?.stack ?? error) }); }
  });
};
