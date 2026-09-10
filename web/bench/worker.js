import { validateSpans, teacherSpans, scoreDocument } from './scoring.js';

let engine, variant, source, tokenizer, initialization;
let queue = Promise.resolve();
const utf8 = new TextEncoder();
async function download(path, limit) {
  const response = await fetch(path, { signal: AbortSignal.timeout(60000) });
  if (!response.ok || !response.body) throw new Error(`${path}: HTTP ${response.status}`);
  const chunks = []; let length = 0;
  for await (const chunk of response.body) {
    length += chunk.length;
    if (length > limit) throw new Error(`${path} exceeds download cap`);
    chunks.push(chunk);
  }
  const bytes = new Uint8Array(length); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}
async function getTokenizer() {
  if (!tokenizer) {
    const module = await import('/compact/pkg/tint_tokenizer.js');
    await module.default();
    tokenizer = module.tokenize;
  }
  return tokenizer;
}
async function run(message) {
  if (message.type === 'source') {
    if (typeof message.source !== 'string' || message.source.length > 16 * 1024 * 1024) throw new Error('Invalid benchmark source');
    const bytes = utf8.encode(message.source);
    if (bytes.length > 16 * 1024 * 1024) throw new Error('Source exceeds 16 MiB');
    source = message.source;
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    return { bytes: bytes.length, chars: source.length, sha256: Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('') };
  }
  if (message.type === 'initialize') {
    if (initialization || !['original', 'compact', 'burn'].includes(message.variant) || source === undefined) throw new Error('Invalid initialization');
    variant = message.variant;
    initialization = true;
    const start = performance.now();
    let metadata = null;
    if (variant === 'original') {
      engine = await import('/bench/generated/gpu-lexer.js');
      if (typeof engine.highlight !== 'function') throw new Error('Original highlight export missing');
    } else {
      const compact = variant === 'compact';
      const [module, json, bytes] = await Promise.all([
        import(compact ? '/compact/dist/runtime.js' : '/pkg/tint_web.js'),
        download(compact ? '/model.q4.json' : '/model.json', 16383),
        download(compact ? '/model.q4.bin' : '/model.bin', 8 * 1024 * 1024),
      ]);
      await module.default();
      const text = new TextDecoder('utf-8', { fatal: true }).decode(json);
      metadata = JSON.parse(text);
      engine = await module.Highlighter.load(text, bytes);
    }
    const loaded = performance.now();
    const spans = await engine.highlight(source);
    const finished = performance.now();
    validateSpans(source, spans, variant === 'original' ? 'type' : 'class');
    const config = metadata?.config;
    return { init_ms: loaded - start, first_call_ms: finished - loaded, cold_start_ms: finished - start,
      cold_source_chars: source.length, config: config ?? null,
      parameter_count: config ? 1365 * config.embedding_dim + (2 * config.radius + 1) * config.embedding_dim * config.hidden_dim + config.hidden_dim + config.hidden_dim * 9 + 9 : null,
      note: variant === 'original' ? 'Import-only init_ms; lazy GPU initialization is in first_call_ms. No empty highlight used.' : 'init_ms includes imports, model downloads, validation and engine load.' };
  }
  if (!engine || source === undefined) throw new Error('Engine or source not initialized');
  if (message.type === 'run' || message.type === 'quality') {
    let tokens;
    if (variant === 'burn' && utf8.encode(source).length > 4 * 1024 * 1024) {
      return { status: 'skipped', reason: 'Burn source exceeds 4 MiB' };
    }
    if (variant === 'burn' || message.type === 'quality') {
      tokens = (await getTokenizer())(source);
      if (variant === 'burn' && tokens.length / 4 > 262144) {
        return { status: 'skipped', reason: 'Burn source exceeds 262144 tokens' };
      }
    }
    const start = performance.now();
    const spans = await engine.highlight(source);
    const elapsed = performance.now() - start;
    const field = variant === 'original' ? 'type' : 'class';
    validateSpans(source, spans, field);
    if (message.type === 'quality') return { status: 'ok', score: scoreDocument(source, tokens, teacherSpans(source, message.truth), spans, field) };
    return { status: 'ok', elapsed_ms: elapsed, span_count: spans.length };
  }
  throw new Error(`Unknown worker operation: ${message.type}`);
}
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try { self.postMessage({ id: data.id, result: await run(data) }); }
    catch (error) { self.postMessage({ id: data.id, error: String(error?.stack ?? error) }); }
  });
};
