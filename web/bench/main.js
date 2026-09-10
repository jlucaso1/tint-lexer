import { runBenchmarks } from './runner.js';

window.runBenchmarks = runBenchmarks;
const find = id => document.getElementById(id);
let latest;
function row(target, cells) {
  const tr = document.createElement('tr');
  for (const value of cells) { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); }
  find(target).append(tr);
}
find('run').onclick = async () => {
  find('run').disabled = true;
  find('download').disabled = true;
  find('timings').replaceChildren(); find('quality').replaceChildren();
  try {
    window.benchmarkPromise = runBenchmarks({ includeLarge: find('large').checked, includeBurn: find('burn').checked,
      onProgress: text => { find('status').textContent = text; } });
    latest = await window.benchmarkPromise;
    for (const item of latest.performance) row('timings', [item.case, item.engine, item.status === 'ok' ? 'ok' : `${item.status}: ${item.reason}`, item.median_ms?.toFixed(2) ?? '-', item.p95_ms?.toFixed(2) ?? '-']);
    for (const [engine, quality] of Object.entries(latest.quality)) {
      for (const [language, score] of Object.entries(quality.languages)) row('quality', [engine, language, score.scored_tokens, score.agreement?.toFixed(4) ?? '-', score.macro_f1?.toFixed(4) ?? '-']);
    }
    find('details').textContent = JSON.stringify({ engines: latest.engines, environment: latest.environment, skipped: latest.skipped, quality_status: Object.fromEntries(Object.entries(latest.quality).map(([name, value]) => [name, { status: value.status, reason: value.reason }])) }, null, 2);
    find('status').textContent = 'Suite finished. Check unavailable or partial results before comparing engines.';
    find('download').disabled = false;
  } catch (error) { find('status').textContent = String(error); }
  finally { find('run').disabled = false; }
};
find('download').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(latest, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = 'tint-benchmark.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
