import { check, decode, generatedURL, download, verifiedDownload, sha256, validateManifest, summarize } from './protocol.mjs';

let running = false;
function workerClient(url) {
  const worker = new Worker(url, { type: 'module' });
  let sequence = 0, failed;
  const pending = new Map();
  function fail(error) {
    failed = error;
    for (const entry of pending.values()) { clearTimeout(entry.timer); entry.reject(error); }
    pending.clear();
  }
  worker.onerror = event => fail(new Error(event.message || 'Benchmark worker failed'));
  worker.onmessageerror = () => fail(new Error('Worker message could not be decoded'));
  worker.onmessage = ({ data }) => {
    const entry = pending.get(data.id);
    if (!entry) return;
    pending.delete(data.id); clearTimeout(entry.timer);
    if (data.error) entry.reject(new Error(data.error)); else entry.resolve(data.result);
  };
  return {
    call(message) {
      if (failed) return Promise.reject(failed);
      return new Promise((resolve, reject) => {
        const id = ++sequence;
        const timer = setTimeout(() => { fail(new Error('Benchmark worker timed out')); worker.terminate(); }, 10 * 60 * 1000);
        pending.set(id, { resolve, reject, timer });
        worker.postMessage({ ...message, id });
      });
    },
    close() { fail(new Error('Worker closed')); worker.terminate(); },
  };
}
export async function runQualityBenchmark({ manifest: path = '/experiments/generated/quality-v1/manifest.json', iterations = 9, warmups = 3 } = {}) {
  check(!running, 'A quality benchmark is already running');
  check(iterations === 9 && warmups === 3, 'Matched protocol requires 3 warmups and 9 measured iterations');
  running = true;
  const workers = {};
  try {
    const manifestBytes = await download(generatedURL(path, location.origin)), manifest = JSON.parse(decode(manifestBytes));
    validateManifest(manifest, path, location.origin);
    const manifestSHA = await sha256(manifestBytes);
    for (const [entries, expected] of [[manifest.scripts, manifest.benchmark_sha256], [manifest.runtime, manifest.expected.runtime_sha256]]) {
      check(await sha256(JSON.stringify(entries)) === expected, 'Payload digest mismatch');
      for (const entry of entries) await verifiedDownload(`/${entry.file.slice(4)}`, entry.sha256);
    }
    const initialization = {};
    for (const variant of ['baseline', 'candidate']) {
      const url = new URL('./worker.mjs', import.meta.url);
      url.search = new URLSearchParams({ manifest: path, variant, sha256: manifestSHA });
      workers[variant] = workerClient(url);
      initialization[variant] = await workers[variant].call({ type: 'initialize' });
      check(initialization[variant].weights_sha256 === manifest.expected[`${variant}_weights_sha256`]
        && initialization[variant].runtime_sha256 === manifest.expected.runtime_sha256, 'Worker model identity mismatch');
    }
    check(initialization.baseline.adapter === initialization.candidate.adapter, 'Worker adapter descriptors differ');
    const ua = navigator.userAgent;
    const environment = { user_agent: ua, adapter: initialization.baseline.adapter,
      os: navigator.userAgentData?.platform || navigator.platform || 'unknown',
      architecture: /(?:x86_64|Win64|x64|amd64)/i.test(ua) ? 'x86_64 (user-agent)' : /(?:aarch64|arm64)/i.test(ua) ? 'arm64 (user-agent)' : 'unknown (not exposed)',
      browser_version: ua.match(/(?:Edg|Firefox|Chrome|Version)\/[\d.]+/)?.[0] || ua,
      power_mode: 'uncontrolled', benchmark_sha256: manifest.benchmark_sha256 };
    const report = { schema_version: 1, ...manifest.expected, environment, cases: [] };
    const details = { manifest: path, manifest_sha256: manifestSHA, iterations, warmups, initialization,
      clock: 'Worker performance.now around await Highlighter.highlight, including spans. Excludes IPC, source loading, validation, and DOM.',
      percentile: 'Nearest rank. Median is sorted sample 5 of 9; p95 is sample 9 of 9.',
      caveat: manifest.caveat, cases: [] };
    for (const [caseIndex, item] of manifest.cases.entries()) {
      for (const variant of ['baseline', 'candidate']) {
        const resident = await workers[variant].call({ type: 'source', case_id: item.id });
        check(resident.source_sha256 === item.source_sha256 && resident.bytes === item.bytes && resident.chars === item.chars, 'Worker source identity mismatch');
      }
      const detail = { id: item.id, source_sha256: item.source_sha256, baseline: [], candidate: [], warmup_samples: [], order: [] };
      for (let round = 0; round < warmups + iterations; round++) {
        const order = (round + caseIndex) % 2 === 0 ? ['baseline', 'candidate'] : ['candidate', 'baseline'];
        detail.order.push(order);
        for (const variant of order) {
          const sample = await workers[variant].call({ type: 'sample' });
          if (round < warmups) detail.warmup_samples.push({ round, variant, ...sample });
          else detail[variant].push(sample);
        }
      }
      report.cases.push({ id: item.id, source_sha256: item.source_sha256, baseline: summarize(detail.baseline), candidate: summarize(detail.candidate) });
      details.cases.push(detail);
    }
    for (const entries of [manifest.scripts, manifest.runtime]) {
      for (const entry of entries) await verifiedDownload(`/${entry.file.slice(4)}`, entry.sha256);
    }
    check(await sha256(await download(generatedURL(path, location.origin))) === manifestSHA, 'Manifest changed during benchmark');
    return { report, details };
  } finally {
    for (const worker of Object.values(workers)) worker.close();
    running = false;
  }
}
if (typeof window !== 'undefined') window.runQualityBenchmark = runQualityBenchmark;
