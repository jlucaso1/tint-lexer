// Pure scoring helpers shared by the harness and its unit tests.
// No engine imports here, so `deno test` runs without network or GPU.

export const CLASSES = [
  "plain",
  "comment",
  "string",
  "number",
  "keyword",
  "type",
  "function",
  "constant",
  "operator",
] as const;
export type ClassName = (typeof CLASSES)[number];

export interface DocSpans {
  id: string;
  language: string;
  source: string;
  spans: { start: number; end: number; class: ClassName }[];
}

export interface EngineSpans {
  start: number;
  end: number;
  class: ClassName;
}

export function byteToUtf16(source: string): number[] {
  const bytes = new TextEncoder().encode(source);
  const map = new Array<number>(bytes.length + 1).fill(-1);
  let byte = 0;
  let utf16 = 0;
  map[0] = 0;
  for (const char of source) {
    const len = new TextEncoder().encode(char).length;
    byte += len;
    utf16 += char.length;
    map[byte] = utf16;
  }
  return map;
}

export function paint(
  length: number,
  spans: { start: number; end: number; class: ClassName }[],
): (ClassName | null)[] {
  const out: (ClassName | null)[] = new Array(length).fill(null);
  for (const span of spans) {
    for (let i = span.start; i < span.end && i < length; i++) {
      if (out[i] === null) out[i] = span.class;
    }
  }
  return out;
}

export interface Tallies {
  confusion: number[][];
  scored: number;
}

export function newTallies(): Tallies {
  return {
    confusion: Array.from({ length: 9 }, () => new Array(9).fill(0)),
    scored: 0,
  };
}

export function observe(
  tallies: Tallies,
  truth: number,
  predicted: number | null,
): void {
  tallies.scored++;
  if (predicted !== null) tallies.confusion[truth][predicted]++;
}

export function summarize(tallies: Tallies) {
  const perClass = CLASSES.map((name, id) => {
    const support = tallies.confusion[id].reduce((a, b) => a + b, 0);
    const predicted = tallies.confusion.reduce((a, row) => a + row[id], 0);
    const tp = tallies.confusion[id][id];
    const precision = predicted === 0 ? 0 : tp / predicted;
    const recall = support === 0 ? 0 : tp / support;
    const f1 = support + predicted === 0 ? 0 : (2 * tp) / (support + predicted);
    return { class: name, support, precision, recall, f1 };
  });
  const supported = perClass.filter((c) => c.support > 0);
  const correct = tallies.confusion.reduce((a, row, i) => a + row[i], 0);
  return {
    evaluated_units: tallies.scored,
    agreement: tallies.scored === 0 ? 0 : correct / tallies.scored,
    macro_f1: supported.length === 0
      ? 0
      : supported.reduce((a, c) => a + c.f1, 0) / supported.length,
    per_class: perClass,
  };
}

