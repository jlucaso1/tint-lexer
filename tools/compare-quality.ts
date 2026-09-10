// Fair quality comparison: our compact runtime vs gpu-lexer, same harness.
// Runs under Deno (native WebGPU + TypeScript, no build step).
// Usage: deno task compare:quality
//
// Method: for every teacher document, both engines highlight the source.
// Scores every UTF-16 code unit that is not whitespace, comparing the
// teacher span class against each engine's span class. Uncovered units
// count as mismatches. This differs slightly from the Rust gate, which
// scores structural tokens and skips tokens straddling two teacher spans;
// character scoring has no straddle case. Wall time per document is also
// recorded (device-dependent; see report.environment).

import { highlight as vercelHighlight } from "../node_modules/gpu-lexer/dist/index.js";
import { Highlighter } from "../dist/index.js";

import { CLASSES, byteToUtf16, newTallies, observe, paint, summarize } from "./compare-metrics.ts";
import type { ClassName, DocSpans, EngineSpans, Tallies } from "./compare-metrics.ts";
function checkArgs(args: string[]): [string, string] {
  if (args.length !== 2) {
    throw new Error("usage: compare-quality.ts <test.jsonl> <report.json>");
  }
  return [args[0], args[1]];
}

async function main(): Promise<void> {
  const [teacherPath, reportPath] = checkArgs(Deno.args);
  const text = await Deno.readTextFile(teacherPath);
  const docs: DocSpans[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const doc = JSON.parse(line);
    const map = byteToUtf16(doc.source);
    docs.push({
      id: doc.id,
      language: doc.language,
      source: doc.source,
      spans: doc.spans.map(
        (s: { start: number; end: number; class: ClassName }) => {
          const start = map[s.start];
          const end = map[s.end];
          if (start < 0 || end < 0 || end < start) {
            throw new Error(`teacher span off char boundary in ${doc.id}`);
          }
          return { start, end, class: s.class };
        },
      ),
    });
  }

  const engine = await Highlighter.load();
  let adapter = "unknown";
  try {
    const a = await navigator.gpu?.requestAdapter();
    adapter = a ? "present" : "none";
  } catch {
    adapter = "error";
  }

  const ours = newTallies();
  const theirs = newTallies();
  const perLanguage: Record<string, { ours: Tallies; theirs: Tallies }> = {};
  const times: Record<string, { ours: number[]; theirs: number[] }> = {};
  let teacherGaps = 0;
  let engineGaps = 0;

  for (const doc of docs) {
    const units = doc.source.length;
    const truth = paint(units, doc.spans);
    for (let i = 0; i < units; i++) {
      if (truth[i] === null) teacherGaps++;
    }
    const scored: { truth: number; index: number }[] = [];
    for (let index = 0; index < units; index++) {
      if (!/\s/.test(doc.source[index]) && truth[index] !== null) {
        scored.push({ truth: CLASSES.indexOf(truth[index]!), index });
      }
    }

    const t0 = performance.now();
    const ourRaw = (await engine.highlight(doc.source)) as {
      start: number;
      end: number;
      class: string;
    }[];
    const ourSpans: EngineSpans[] = ourRaw.map((s) => ({
      start: s.start,
      end: s.end,
      class: s.class as ClassName,
    }));
    const t1 = performance.now();
    const theirRaw = await vercelHighlight(doc.source);
    const t2 = performance.now();
    const lang = (perLanguage[doc.language] ??= {
      ours: newTallies(),
      theirs: newTallies(),
    });
    const time = (times[doc.language] ??= { ours: [], theirs: [] });
    time.ours.push(t1 - t0);
    time.theirs.push(t2 - t1);

    const ourPaint = paint(
      units,
      ourSpans.map((s) => ({ ...s, class: s.class as ClassName })),
    );
    const theirPaint = paint(
      units,
      theirRaw.map((s: { start: number; end: number; type: string }) => ({
        start: s.start,
        end: s.end,
        class: (CLASSES as readonly string[]).includes(s.type)
          ? (s.type as ClassName)
          : "plain",
      })),
    );
    for (const { truth, index } of scored) {
      if (ourPaint[index] === null || theirPaint[index] === null) {
        engineGaps++;
      }
      const toId = (c: ClassName | null): number | null =>
        c === null ? null : CLASSES.indexOf(c);
      observe(ours, truth, toId(ourPaint[index]));
      observe(theirs, truth, toId(theirPaint[index]));
      observe(lang.ours, truth, toId(ourPaint[index]));
      observe(lang.theirs, truth, toId(theirPaint[index]));
    }
  }

  const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  const report = {
    schema_version: 1,
    corpus: teacherPath,
    documents: docs.length,
    environment: {
      adapter,
      runtime: navigator.userAgent ?? "deno",
      deno: Deno.version.deno,
    },
    method: "UTF-16 code units, whitespace excluded, coverage gaps counted wrong",
    teacher_gaps: teacherGaps,
    engine_gaps: engineGaps,
    tint: summarize(ours),
    gpulexer: summarize(theirs),
    timings_ms_median: Object.fromEntries(
      Object.entries(times).map(([lang, t]) => [
        lang,
        { tint: median(t.ours), gpulexer: median(t.theirs) },
      ]),
    ),
    per_language: Object.fromEntries(
      Object.entries(perLanguage).map(([lang, t]) => [
        lang,
        { tint: summarize(t.ours), gpulexer: summarize(t.theirs) },
      ]),
    ),
  };
  await Deno.writeTextFile(reportPath, JSON.stringify(report, null, 2));
  console.log(`tint-lexer:    ${report.tint.agreement.toFixed(4)} agree / ${report.tint.macro_f1.toFixed(4)} F1 on ${report.tint.evaluated_units} units`);
  console.log(`gpu-lexer:     ${report.gpulexer.agreement.toFixed(4)} agree / ${report.gpulexer.macro_f1.toFixed(4)} F1 on ${report.gpulexer.evaluated_units} units`);
  console.log(`teacher gaps: ${teacherGaps}, engine gaps: ${engineGaps}`);
}

if (import.meta.main) await main();
