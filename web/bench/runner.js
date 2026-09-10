import { addScore, emptyScore, summarizeScore, timingSummary } from './scoring.js';

export class EngineWorker {
  constructor(name, { WorkerClass = globalThis.Worker, timeout = 120000 } = {}) {
    this.name = name;
    this.timeout = timeout;
    this.nextId = 0;
    this.pending = new Map();
    this.worker = new WorkerClass(new URL('./worker.js', import.meta.url), { type: 'module', name: `bench-${name}` });
    this.worker.onmessage = ({ data }) => {
      const pending = this.pending.get(data.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.pending.delete(data.id);
      if (data.error) pending.reject(new Error(data.error));
      else pending.resolve(data.result);
    };
    this.worker.onerror = event => this.close(new Error(event.message || 'Worker failed'));
    this.worker.onmessageerror = () => this.close(new Error('Worker message could not be decoded'));
  }
  request(type, payload = {}) {
    if (this.failure) return Promise.reject(this.failure);
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.close(new Error(`${this.name} ${type} timed out after ${this.timeout} ms`)), this.timeout);
      this.pending.set(id, { resolve, reject, timer });
      try { this.worker.postMessage({ ...payload, type, id }); }
      catch (error) { this.close(error); }
    });
  }
  close(error = new Error('Worker closed')) {
    this.failure ??= error;
    this.worker.terminate();
    for (const pending of this.pending.values()) { clearTimeout(pending.timer); pending.reject(this.failure); }
    this.pending.clear();
  }
}

async function json(path, optional = false) {
  try {
    const response = await fetch(path, { signal: AbortSignal.timeout(60000) });
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    if (!optional) throw error;
    return { status: 'unavailable', reason: String(error) };
  }
}
export function engineOrder(names, turn) {
  return turn % 2 ? [...names].reverse() : [...names];
}
async function hardware() {
  const info = { user_agent: navigator.userAgent, platform: navigator.platform, hardware_concurrency: navigator.hardwareConcurrency,
    device_memory_gib: navigator.deviceMemory ?? null, secure_context: isSecureContext, adapter: null,
    screen: { width: screen.width, height: screen.height, device_pixel_ratio: devicePixelRatio } };
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter) {
      const details = adapter.info ?? await adapter.requestAdapterInfo?.();
      info.adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'subgroupMinSize', 'subgroupMaxSize'].map(key => [key, details?.[key] ?? null]));
      info.adapter_features = [...adapter.features];
      info.adapter_note = 'Default adapter requested separately for reporting; engine-internal adapter selection is not instrumented.';
    }
  } catch (error) { info.adapter_error = String(error); }
  return info;
}
let active = false;
export async function runBenchmarks({ includeLarge = false, iterations = 5, warmups = 2, includeBurn = false, timeout = 120000, onProgress = () => {} } = {}) {
  if (active) throw new Error('A benchmark suite is already running');
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 100 || !Number.isInteger(warmups) || warmups < 0 || warmups > 100 ||
      !Number.isFinite(timeout) || timeout < 1) throw new Error('Invalid benchmark counts or timeout');
  active = true;
  const workers = new Map();
  try {
    const manifest = await json('/bench/generated/manifest.json');
    const report = { schema_version: 1, started_at: new Date().toISOString(), options: { includeLarge, iterations, warmups, includeBurn, timeout },
      environment: await hardware(), artifacts: manifest.original, sizes: await json('/bench/generated/size.json', true),
      quality_fixture: manifest.quality, engines: {}, performance: [], quality: {}, skipped: [],
      methodology: 'Worker-local highlight time through produced spans, including CPU tokenization, GPU upload/readback and merge. Excludes source postMessage, integrity validation, scoring and DOM. Two extra warmups by default after a separate first-source cold call. p95 uses nearest rank. Cold start is not a cold network/cache guarantee.',
      caveat: 'Synthetic timing and tiny test fixtures do not establish general accuracy or a speed win. Teacher uses the custom nine-class theme, not the article Shiki theme. Do not compare article hardware timings.' };
    const cases = [...manifest.cases];
    if (!cases.length) throw new Error('No benchmark cases');
    if (includeLarge) {
      if (manifest.large.status !== 'ok') report.skipped.push({ case: 'three-0.97.0-10x', reason: manifest.large.reason });
      else {
        const response = await fetch(manifest.large.url, { signal: AbortSignal.timeout(60000) });
        if (!response.ok) throw new Error(`Large fixture: HTTP ${response.status}`);
        const source = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(await response.arrayBuffer()).repeat(10);
        cases.push({ ...manifest.large, source });
      }
    }
    const names = ['original', 'compact', ...(includeBurn ? ['burn'] : [])];
    async function setSource(worker, item) {
      const identity = await worker.request('source', { source: item.source });
      if (identity.sha256 !== item.sha256 || identity.bytes !== item.bytes || identity.chars !== item.chars) throw new Error('Worker source identity mismatch');
      return identity;
    }
    function unavailable(name, error) {
      report.engines[name] = { ...report.engines[name], status: 'unavailable', reason: String(error) };
      workers.get(name)?.close();
      workers.delete(name);
    }
    for (const name of names) {
      onProgress(`Initializing ${name}`);
      try {
        const worker = new EngineWorker(name, { timeout });
        workers.set(name, worker);
        await setSource(worker, cases[0]);
        report.engines[name] = { status: 'ok', cold_source: { id: cases[0].id, sha256: cases[0].sha256 },
          ...await worker.request('initialize', { variant: name }) };
      } catch (error) { unavailable(name, error); }
    }
    for (const [caseIndex, item] of cases.entries()) {
      onProgress(`Timing ${item.id}`);
      const rows = new Map();
      for (const name of engineOrder(names, caseIndex)) {
        const row = { engine: name, case: item.id, language: item.language, bytes: item.bytes, chars: item.chars, sha256: item.sha256, status: 'unavailable', samples: [] };
        rows.set(name, row);
        if (!workers.has(name)) { row.reason = report.engines[name].reason; continue; }
        try {
          await setSource(workers.get(name), item);
          row.status = 'ok';
        } catch (error) { row.reason = String(error); unavailable(name, error); }
      }
      for (let trial = 0; trial < warmups + iterations; trial++) {
        for (const name of engineOrder(names, caseIndex + trial)) {
          const row = rows.get(name);
          if (row.status !== 'ok') continue;
          try {
             const result = await workers.get(name).request('run');
             if (result.status !== 'ok') { row.status = result.status; row.reason = result.reason; continue; }
             row.span_count = result.span_count;
             if (trial >= warmups) row.samples.push(result.elapsed_ms);
          } catch (error) { row.status = 'unavailable'; row.reason = String(error); unavailable(name, error); }
        }
      }
      for (const row of rows.values()) {
        const { samples, ...rest } = row;
        report.performance.push({ ...rest, ...(row.status === 'ok' ? timingSummary(samples) : { partial_samples_ms: samples }) });
      }
    }
    const documents = await json(manifest.quality.url);
    for (const name of names) report.quality[name] = { status: workers.has(name) ? 'ok' : 'unavailable', languages: {}, documents: [] };
    for (const [index, doc] of documents.entries()) {
      onProgress(`Scoring ${doc.id}`);
      const bytes = new TextEncoder().encode(doc.source);
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      const identity = { bytes: bytes.length, chars: doc.source.length,
        sha256: Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('') };
      for (const name of engineOrder(names, index)) {
        const worker = workers.get(name), quality = report.quality[name];
        if (!worker) {
          quality.status = quality.documents.some(item => item.status === 'ok') ? 'partial' : 'unavailable';
          quality.reason = report.engines[name].reason;
          quality.documents.push({ id: doc.id, language: doc.language, ...identity, status: 'unavailable', reason: quality.reason });
          continue;
        }
        try {
          await setSource(worker, { source: doc.source, ...identity });
          const result = await worker.request('quality', { truth: doc.spans });
          quality.documents.push({ id: doc.id, language: doc.language, ...identity, ...result });
          if (result.status === 'ok') addScore(quality.languages[doc.language] ??= emptyScore(), result.score);
          else quality.status = 'partial';
        } catch (error) {
          quality.status = 'partial';
          quality.documents.push({ id: doc.id, language: doc.language, ...identity, status: 'unavailable', reason: String(error) });
          unavailable(name, error);
        }
      }
    }
    for (const quality of Object.values(report.quality)) {
      const total = emptyScore();
      for (const [language, score] of Object.entries(quality.languages)) {
        addScore(total, score);
        quality.languages[language] = summarizeScore(score);
      }
      quality.total = summarizeScore(total);
    }
    report.finished_at = new Date().toISOString();
    return report;
  } finally {
    for (const worker of workers.values()) worker.close();
    active = false;
  }
}
