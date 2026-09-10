declare const Deno: { test: (name: string, fn: () => void) => void } | undefined;

const test = typeof Deno !== "undefined"
  ? (name: string, fn: () => void) => Deno.test(name, fn)
  : (await import("node:test")).default;

import {
  byteToUtf16,
  newTallies,
  observe,
  paint,
  summarize,
} from "./compare-metrics.ts";

function equal(actual: unknown, expected: unknown, label: string): void {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${label}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`);
  }
}

function close(actual: number, expected: number, label: string): void {
  if (Math.abs(actual - expected) > 1e-12) {
    throw new Error(`${label}: got ${actual}, want ${expected}`);
  }
}

test("byte offsets map to UTF-16 units", () => {
  equal(byteToUtf16("a\u{1F600}e"), [0, 1, -1, -1, -1, 3, 4], "utf16 map");
});

test("paint keeps first span on overlap and null on gaps", () => {
  equal(
    paint(5, [
      { start: 0, end: 3, class: "string" },
      { start: 2, end: 4, class: "comment" },
    ]),
    ["string", "string", "string", "comment", null],
    "paint",
  );
});

test("agreement and macro F1 match the Rust gate formulas", () => {
  const t = newTallies();
  observe(t, 0, 0);
  observe(t, 0, 1);
  observe(t, 1, 1);
  observe(t, 2, null);
  const s = summarize(t);
  equal(s.evaluated_units, 4, "units");
  close(s.agreement, 0.5, "agreement");
  close(s.per_class[0].f1, 2 / 3, "plain f1");
  close(s.per_class[1].f1, 2 / 3, "comment f1");
  equal(s.per_class[2].f1, 0, "string f1");
  close(s.macro_f1, (2 / 3 + 2 / 3) / 2, "macro f1");
});
