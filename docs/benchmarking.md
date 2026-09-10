# Benchmarking

The benchmark compares the published `gpu-lexer@0.0.1` with the compact runtime in separate dedicated module workers. Burn is an optional third engine. It is not the baseline, and its failure does not stop the other engines.

## Prepare and run

Run from the repository root with Node 20 or newer. The tools require the pinned development dependencies `esbuild@0.28.2` and `gpu-lexer@0.0.1`.

```sh
npm ci
npm run data -- --force
node tools/build-compact.mjs --model artifacts/compact-q4
node tools/prepare-bench.mjs
node tools/compress-web.mjs
node tools/serve-web.mjs
```

The model directory must already contain valid compact artifacts. Training and quantization are separate from this benchmark. Open `http://localhost:4173/bench/index.html` in a WebGPU-enabled browser on a secure context. The static server already serves nested benchmark paths. It does not map `/bench/` to an index file.

The page runs both default engines, shows timing and teacher-agreement tables, and downloads the complete JSON report. Keep the tab visible, close other GPU workloads, and record the browser version and machine conditions. A timeout or engine error produces an unavailable result rather than waiting indefinitely. The default per-operation timeout is 120 seconds; large workloads may need more.

The runner also exports a machine-callable function from `/bench/runner.js` and installs it as `window.runBenchmarks`.

```js
window.benchmarkPromise = window.runBenchmarks({
  includeLarge: false,
  iterations: 5,
  warmups: 2,
});
window.benchmarkPromise.then(report => { window.benchmarkReport = report; });
```

Start this promise without awaiting it in browser automation calls with short command timeouts. Poll `window.benchmarkReport` later. Attach a rejection handler in automation to capture preparation or page-level errors. Calls cannot overlap within one page.

## Sources and artifacts

Default timing cases repeat deterministic ASCII Rust, TypeScript and Python examples to exactly 1 KiB, 64 KiB and 256 KiB, producing nine cases. Truncation can leave incomplete syntax. These are synthetic timing inputs, not quality data. Each case records its UTF-8 bytes, UTF-16 code units and SHA-256. Workers verify those values before timing.

`prepare-bench.mjs` strictly loads `data/generated/test.jsonl` in Node and publishes the documents as `/bench/generated/quality.json`. It rejects unexpected fields, duplicate IDs, invalid labels, coverage gaps and invalid Unicode boundaries. The JSONL remains the source of truth. A publisher can replace it with a real held-out corpus without changing the browser API, then rerun preparation. Current fixtures are tiny toy examples. Their scores do not support general accuracy claims.

To prepare the optional article-sized workload:

```sh
node tools/prepare-bench.mjs --three
```

This downloads `https://unpkg.com/three@0.97.0/build/three.min.js` with a 2 MiB streaming limit and a 60-second timeout. It caches the exact bytes under `artifacts/bench-fixtures/three.min.js`, writes SHA metadata alongside the cache, and publishes the unmodified file at `web/bench/generated/three.min.js`. Subsequent preparations reuse the cache. The browser concatenates ten copies without inserting separators, roughly the article's 5.56 million characters. The report records actual byte and character counts, not that rounded claim. Use `includeLarge: true` after preparation. A default preparation marks the large case unprepared even if an older file remains on disk.

Preparation bundles the installed original `dist/index.js` into a browser ESM bundle with esbuild, without changing the original engine. It records the package version, original file hash, generated bundle hash and esbuild version. Generated files live under `web/bench/generated/`; do not hand-edit them.

## Timing contract

Each worker loads its engine once and processes all messages serially. The page runs one highlight at a time across engines. It reverses engine order on alternating cases and trials to reduce fixed-order warming bias. Cold engine initialization uses the same first Rust 1 KiB source. Cold initialization order remains original, compact, then optional Burn; repeat whole sessions if investigating startup order effects.

Source transfer, SHA checks and any scoring tokenizer work finish before the highlight timer starts inside the worker. The timer ends when the engine returns spans. It includes the engine's CPU tokenization, GPU uploads, inference, readback and span merge. It excludes message transport, span validation, teacher scoring and DOM work. Timed responses contain elapsed time and span count, never full spans. Every result must cover the same source with ordered, complete, half-open UTF-16 spans and valid classes. Original uses `type`; compact and Burn use `class`.

Two warmups per case precede five measured calls by default. Warmups and the initial cold call are not included in samples. Median uses the middle value, or the mean of the two middle values. p95 uses nearest rank; with five runs it is the maximum. Five samples provide very little evidence about tail latency. The report keeps all measured samples.

`init_ms` covers imports and explicit loading, including model fetches for compact and Burn. Original initializes its GPU lazily, so its `init_ms` is only import time. The benchmark does not use an empty original highlight as an initialization proxy. `first_call_ms` measures the first real highlight and includes original's lazy initialization. `cold_start_ms` is import/load plus that first highlight on the same source. Do not compare `init_ms` alone across engines. Dedicated workers reset JavaScript engine state, not browser HTTP, shader or driver caches. These are not guaranteed cold-network measurements.

The report includes user agent, platform, logical hardware concurrency, available device-memory hints, screen information and exposed GPU adapter fields. The adapter query is separate from engine-internal selection and may be privacy-redacted. Compact and Burn parameter counts derive from their metadata config. Original's parameter count is `null` because its public API does not expose model metadata.

## Teacher scoring

Both default engines highlight identical held-out documents. Scoring separately imports `/compact/pkg/tint_tokenizer.js` to obtain common Rust tokenizer units. The scoring import and tokenization are outside the timing window. Packed token ends use UTF-16 and kinds 2 and 3 identify whitespace and newlines.

Teacher span offsets are UTF-8 bytes. A single pass over source code points builds the UTF-8-to-UTF-16 boundary map; conversion rejects offsets inside a code point and verifies complete coverage. Unpaired surrogates are rejected. CRLF and supplementary Unicode characters retain their original offsets.

Whitespace units are excluded. A unit crossing different teacher labels is ambiguous and excluded. A prediction crossing different classes on a non-ambiguous truth unit counts as wrong, not excluded. Adjacent spans with the same class are not mixed. Confusion matrices have nine truth rows and ten prediction columns, with `mixed` last. Mixed predictions contribute false negatives to their true class.

Reports include agreement and macro F1 per language and across all scored tokens, plus per-class precision, recall and true support. Macro F1 averages only classes with true support. Zero-denominator precision and recall are zero; empty totals have null agreement and macro F1. Exclusion counts and mixed-prediction counts remain visible. The teacher is this project's custom nine-class Shiki theme, not the article's Shiki theme. Agreement with that teacher is not a language-independent correctness measure.

## Optional Burn

Enable `includeBurn: true` or the page checkbox only when `/pkg/tint_web.js`, its WASM, `/model.json` and `/model.bin` still exist. Each source must fit Burn's 4 MiB and 262,144-token caps. Over-cap cases are skipped. Missing artifacts, unsupported GPU pipelines and worker timeouts mark Burn unavailable without aborting the default comparison. No CPU fallback is introduced. Partial quality totals must not be compared with complete totals without matching document coverage.

## Size accounting

```sh
node tools/measure-size.mjs
```

The script writes JSON to stdout. Preparation calls the same functions and publishes `generated/size.json` for browser reports. Reprepare after deployment artifacts change. Missing files produce explicit unavailable totals rather than silently undercounting.

Compact library totals include the deployed minified runtime, tokenizer JS/WASM, q4 metadata and weights. Burn totals include its JS/WASM and float metadata/weights. The original published total reads the exact npm `dist/index.js`, which is 59,570 raw bytes in the published 0.0.1 package. A separate total measures the esbuild browser bundle so bootstrap and minifier changes do not get confused with published bytes. SHA-256 values accompany each measured file.

Each file is independently Brotli-compressed at quality 11 using Node's Brotli implementation, then sizes are summed. The report also records raw bytes and file count. Full-page totals add the same current demo HTML, CSS, main script and worker to every library. These totals are normalized accounting, not claims that the current compact demo can load the other engines unchanged. Benchmark UI, fixtures, scoring code and benchmark workers are excluded.

Use measured published bytes and explicit compression settings rather than the article's rounded 27.5 KB figure. The size script measures compressed payloads, not observed network traffic or HTTP overhead. The static server serves fresh Brotli sidecars to clients accepting `br`; otherwise it serves raw files. Rerun compression after benchmark preparation when comparing startup transfers.

## Tests and limits

```sh
node --test tools/benchmark.test.mjs
```

Tests require neither WebGPU nor generated runtime builds. They cover strict fixture parsing, Unicode and CRLF offsets, mixed labels, span failures, metrics, independent per-file Brotli totals, bounded downloads, worker timeout/error propagation, serial worker runs and optional-engine failure isolation with mocks. Real GPU results still require a browser session with valid deployment artifacts. Quantization loss belongs to model validation and is not measured here. Do not claim a baseline win from small samples or compare these timings against article results obtained on different hardware.

## Native parity

After building a compact deployment and its dequantized native artifact:

```sh
node tools/prepare-native-check.mjs --model artifacts/compact-dequant
```

On the local site, run this in the browser console:

```js
const checks = await import('/compact/gpu-check.js');
await checks.checkNativeArtifact();
await checks.checkGPU();
```

The first check compares deployed GPU spans with native Burn output for the test
fixtures. The second covers generated weights, document edges, tile seams,
buffer growth, large inputs, and lifecycle failures. These checks are not timed
benchmarks and require a working WebGPU adapter.
