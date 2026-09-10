import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const script = await readFile(new URL("./worker.js", import.meta.url), "utf8");

test("worker serializes inference, replaces queued edits and accepts empty input", async () => {
  const messages: any[] = [];
  const calls: any[] = [];
  const completions: any[] = [];
  const self: any = { isSecureContext: true };
  const context = vm.createContext({
    self,
    postMessage: (message: any) => messages.push(message),
    TextEncoder,
    performance,
    highlighter: null,
    model: {
      isCpu: true,
      highlight: (source: string) => {
        calls.push(source);
        return new Promise((resolve) => completions.push(resolve));
      },
    },
  });

  // Execute worker script inside VM
  context.highlighter = context.model;
  vm.runInContext(script, context);

  const send = (id: number, source: string) => self.onmessage({ data: { type: "highlight", id, source } });
  send(1, "first");
  send(2, "obsolete");
  send(3, "newest");
  assert.deepEqual(calls, ["first"]);

  completions.shift()([]);
  await new Promise(setImmediate);
  assert.deepEqual(calls, ["first", "newest"]);

  send(4, "");
  completions.shift()([]);
  await new Promise(setImmediate);
  assert.deepEqual(calls, ["first", "newest", ""]);

  send(5, "cancelled");
  self.onmessage({ data: { type: "cancel" } });
  completions.shift()([]);
  await new Promise(setImmediate);
  assert.equal(calls.length, 3);
  assert.deepEqual(
    messages.filter((m) => m.type === "result").map((m) => m.id),
    [1, 3, 4]
  );

  // Input length enforcement
  send(6, "é".repeat(131073));
  assert.equal(messages.at(-1)?.type, "error");
  assert.equal(calls.length, 3);
});

test("worker reports initialization failure gracefully", async () => {
  const messages: any[] = [];
  const self: any = { isSecureContext: true };
  const context = vm.createContext({
    self,
    postMessage: (message: any) => messages.push(message),
    TextEncoder,
    performance,
  });

  vm.runInContext(script, context);
  await new Promise(setImmediate);
  assert.ok(messages.some((m) => m.type === "loading" || m.type === "unavailable"));
});
