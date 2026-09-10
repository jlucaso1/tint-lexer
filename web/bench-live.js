// Live in-browser benchmark runner for Tint demo page

const ENGINES = [
  {
    id: "tint-webgpu",
    name: "tint-lexer (WebGPU)",
    badge: "WebGPU",
    bundleUrl: "./tint.js",
    type: "gpu",
    async run(code, state) {
      if (!globalThis.navigator?.gpu) {
        throw new Error("WebGPU unavailable in this browser");
      }
      if (!state.tintGpu) {
        const { Highlighter } = await import("./tint.js");
        state.tintGpu = await Highlighter.load({ preferCpu: false });
        if (state.tintGpu.isCpu) {
          throw new Error("WebGPU initialization fell back to CPU");
        }
      }
      return state.tintGpu.highlight(code);
    }
  },
  {
    id: "gpu-lexer",
    name: "gpu-lexer (WebGPU)",
    badge: "WebGPU",
    bundleUrl: "./vendor/gpu-lexer.js",
    type: "gpu",
    async run(code, state) {
      if (!globalThis.navigator?.gpu) {
        throw new Error("WebGPU unavailable in this browser");
      }
      if (!state.gpuLexer) {
        state.gpuLexer = await import("./vendor/gpu-lexer.js");
      }
      const fn = state.gpuLexer.parse || state.gpuLexer.highlight;
      return fn(code);
    }
  },
  {
    id: "tint-cpu",
    name: "tint-lexer (CPU)",
    badge: "WASM SIMD",
    bundleUrl: "./tint.js",
    type: "cpu",
    async run(code, state) {
      if (!state.tintCpu) {
        state.tintCpu = await import("./tint.js");
      }
      return state.tintCpu.parseFlatSync(code);
    }
  },
  {
    id: "sugar-high",
    name: "Sugar High",
    badge: "JS regex",
    bundleUrl: "./vendor/sugar-high.js",
    type: "cpu",
    async run(code, state) {
      if (!state.sugarHigh) {
        state.sugarHigh = await import("./vendor/sugar-high.js");
      }
      return state.sugarHigh.highlight(code);
    }
  },
  {
    id: "highlight-js",
    name: "Highlight.js",
    badge: "JS parser",
    bundleUrl: "./vendor/highlight.js",
    type: "cpu",
    async run(code, state) {
      if (!state.highlightJs) {
        state.highlightJs = await import("./vendor/highlight.js");
      }
      return state.highlightJs.highlight(code);
    }
  },
  {
    id: "prism-js",
    name: "Prism.js",
    badge: "JS regex",
    bundleUrl: "./vendor/prism.js",
    type: "cpu",
    async run(code, state) {
      if (!state.prismJs) {
        state.prismJs = await import("./vendor/prism.js");
      }
      return state.prismJs.highlight(code);
    }
  }
];

let rawThreeJs = null;
const engineState = {};
const bundleSizes = {};

// Measure real bundle size on the fly (raw + gzip) via fetch + CompressionStream
async function measureRealBundleSize(url) {
  try {
    const res = await fetch(url);
    const buf = await res.arrayBuffer();
    const rawBytes = buf.byteLength;
    let gzipBytes = rawBytes;
    if (typeof CompressionStream !== "undefined") {
      try {
        const cs = new CompressionStream("gzip");
        const writer = cs.writable.getWriter();
        writer.write(buf);
        writer.close();
        const compressed = await new Response(cs.readable).arrayBuffer();
        gzipBytes = compressed.byteLength;
      } catch (err) {
        console.warn("gzip compression failed:", err);
      }
    }
    return { rawBytes, gzipBytes };
  } catch (err) {
    console.error("Failed to measure bundle size for", url, err);
    return { rawBytes: 0, gzipBytes: 0 };
  }
}

export async function initLiveBenchmark() {
  const container = document.querySelector("#bench-rows");
  const startBtn = document.querySelector("#bench-start-btn");
  const statusEl = document.querySelector("#bench-status");
  const envNote = document.querySelector(".bench-env-note");
  if (!container || !startBtn) return;

  // Detect GPU adapter if possible
  if (globalThis.navigator?.gpu && envNote) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (adapter?.info?.description) {
        envNote.textContent = `GPU: ${adapter.info.description} • Real bundle sizes measured live via HTTP stream.`;
      }
    } catch {}
  }

  // Render initial rows
  renderRows(ENGINES.map(e => ({
    ...e,
    status: "idle",
    timeMs: null,
    barPct: 0
  })));

  // Load bundle sizes dynamically in the background on page load
  (async () => {
    statusEl.textContent = "Measuring real bundle sizes live via HTTP stream...";
    for (const eng of ENGINES) {
      if (!bundleSizes[eng.bundleUrl]) {
        bundleSizes[eng.bundleUrl] = await measureRealBundleSize(eng.bundleUrl);
      }
      const sizeEl = document.querySelector(`.bundle-size[data-engine="${eng.id}"]`);
      if (sizeEl) {
        const sz = bundleSizes[eng.bundleUrl];
        const rawKiB = (sz.rawBytes / 1024).toFixed(1);
        const gzKiB = (sz.gzipBytes / 1024).toFixed(1);
        sizeEl.innerHTML = `<span class="size-pill">${rawKiB} KiB</span> <span class="size-sub">${gzKiB} gz</span>`;
      }
    }
    statusEl.textContent = "Ready. Click Start Benchmark to execute live in your browser.";
  })();

  startBtn.addEventListener("click", () => runBenchmark());
}

function renderRows(items) {
  const container = document.querySelector("#bench-rows");
  if (!container) return;

  container.innerHTML = "";
  items.forEach((item, index) => {
    const isWinner = index === 0 && item.timeMs !== null;
    const row = document.createElement("div");
    row.className = `bench-row ${isWinner ? "winner" : ""}`;
    row.dataset.engine = item.id;

    let timeText = "—";
    if (item.status === "running") {
      timeText = '<span class="spinner-inline"></span> running...';
    } else if (item.status === "error") {
      timeText = '<span class="bench-err">unsupported</span>';
    } else if (item.timeMs !== null) {
      timeText = item.timeMs >= 1000 ? `${(item.timeMs / 1000).toFixed(2)}s` : `${item.timeMs.toFixed(1)}ms`;
    }

    const sz = bundleSizes[item.bundleUrl];
    const sizeDisplay = sz
      ? `<span class="size-pill">${(sz.rawBytes / 1024).toFixed(1)} KiB</span> <span class="size-sub">${(sz.gzipBytes / 1024).toFixed(1)} gz</span>`
      : '<span class="size-measuring">measuring...</span>';

    row.innerHTML = `
      <div class="bench-label-col">
        <span class="bench-rank">${index + 1}</span>
        <span class="bench-name">${item.name}</span>
        <span class="bench-badge ${item.type}">${item.badge}</span>
      </div>
      <div class="bench-size-col bundle-size" data-engine="${item.id}">
        ${sizeDisplay}
      </div>
      <div class="bench-track-wrapper">
        <div class="bench-bar ${isWinner ? "winner" : ""}" style="width: ${Math.max(item.barPct, item.timeMs ? 1.5 : 0)}%;"></div>
      </div>
      <div class="bench-value-col ${isWinner ? "winner" : ""}">
        ${timeText}
      </div>
    `;
    container.appendChild(row);
  });
}

async function runBenchmark() {
  const startBtn = document.querySelector("#bench-start-btn");
  const statusEl = document.querySelector("#bench-status");
  const workloadSel = document.querySelector("#bench-workload");
  const multiplier = Number(workloadSel?.value || 10);

  startBtn.disabled = true;
  startBtn.textContent = "Running...";

  try {
    if (!rawThreeJs) {
      statusEl.textContent = "Fetching three.min.js fixture...";
      const resp = await fetch("./three.min.js");
      rawThreeJs = await resp.text();
    }

    const source = rawThreeJs.repeat(multiplier);
    const mib = (source.length / 1024 / 1024).toFixed(2);
    statusEl.textContent = `Workload prepared: ${multiplier}× three.min.js (${mib} MB, ${source.length.toLocaleString()} chars).`;

    const results = ENGINES.map(e => ({
      ...e,
      status: "pending",
      timeMs: null,
      barPct: 0
    }));

    renderRows(results);

    // Run each engine sequentially
    for (let i = 0; i < ENGINES.length; i++) {
      const eng = ENGINES[i];
      const resItem = results.find(r => r.id === eng.id);
      resItem.status = "running";
      updateRowStatus(results);

      statusEl.textContent = `[${i + 1}/${ENGINES.length}] Warming up and benchmarking ${eng.name}...`;

      try {
        // Warmup (on a smaller 128k slice)
        const warmupSlice = source.slice(0, Math.min(131072, source.length));
        await eng.run(warmupSlice, engineState);

        // Timed run
        const t0 = performance.now();
        await eng.run(source, engineState);
        const elapsed = performance.now() - t0;

        resItem.status = "done";
        resItem.timeMs = elapsed;
      } catch (err) {
        console.warn(`Engine ${eng.name} error:`, err);
        resItem.status = "error";
        resItem.timeMs = null;
      }

      // Re-calculate bar percentages based on current maximum time
      const validTimes = results.filter(r => r.timeMs !== null).map(r => r.timeMs);
      const maxTime = Math.max(...validTimes, 100);
      results.forEach(r => {
        if (r.timeMs !== null) {
          r.barPct = (r.timeMs / maxTime) * 100;
        }
      });

      // Sort: done with timeMs ascending, running, pending, error
      results.sort((a, b) => {
        if (a.timeMs !== null && b.timeMs !== null) return a.timeMs - b.timeMs;
        if (a.timeMs !== null) return -1;
        if (b.timeMs !== null) return 1;
        if (a.status === "running") return -1;
        if (b.status === "running") return 1;
        if (a.status === "error") return 1;
        if (b.status === "error") return -1;
        return 0;
      });

      renderRows(results);
      await new Promise(r => setTimeout(r, 60)); // Yield to paint
    }

    const fastest = results.find(r => r.timeMs !== null);
    statusEl.innerHTML = `✓ Benchmark complete! <strong>${fastest ? fastest.name : "Tint"}</strong> is fastest (${fastest ? (fastest.timeMs >= 1000 ? (fastest.timeMs/1000).toFixed(2) + "s" : fastest.timeMs.toFixed(1) + "ms") : ""}).`;
  } catch (err) {
    statusEl.textContent = `Benchmark failed: ${err.message}`;
  } finally {
    startBtn.disabled = false;
    startBtn.textContent = "↻ Restart Benchmark";
  }
}

function updateRowStatus(items) {
  items.forEach((item) => {
    const row = document.querySelector(`.bench-row[data-engine="${item.id}"]`);
    if (!row) return;
    const valCol = row.querySelector(".bench-value-col");
    if (valCol) {
      if (item.status === "running") {
        valCol.innerHTML = '<span class="spinner-inline"></span> running...';
      } else if (item.status === "pending") {
        valCol.textContent = "—";
      }
    }
  });
}
