# tint-lexer

Fast, compact syntax highlighter powered by WebGPU with an automatic CPU fallback.

Everything needed to tokenize and highlight code runs locally. The bundle embeds a 10 KiB WebAssembly tokenizer and a 4-bit quantized neural model (12,871 parameters). There are zero network requests, zero peer dependencies, and zero runtime setup requirements.

## Numbers

Measured on Bun browser ESM minification and Deno 2.9.6 WebGPU benchmark against `gpu-lexer@0.0.1`:

| Metric | gpu-lexer | tint-lexer | Difference |
| :--- | :--- | :--- | :--- |
| Minified size | 60.8 KiB | 37.9 KiB | 37.7% smaller |
| Gzip size | 33.5 KiB | 21.0 KiB | 37.3% smaller |
| Brotli size | 28.5 KiB | 18.6 KiB | 34.8% smaller |
| 1 KiB latency | 17.5 ms | 12.9 ms | 1.35x faster |
| 10 KiB latency | 32.2 ms | 15.6 ms | 2.06x faster |
| 64 KiB latency | 108.0 ms | 30.1 ms | 3.59x faster |
| 256 KiB latency | 363.3 ms | 90.2 ms | 4.03x faster |
| CPU fallback | Throws error | 1.9 ms (1 KiB) | Full support |

## Installation

```sh
npm install tint-lexer
```

## Usage

### Simple parse

```typescript
import { parse } from "tint-lexer";

const spans = await parse("const answer = 42;");
// [
//   { type: "keyword", start: 0, end: 5 },
//   { type: "plain", start: 5, end: 15 },
//   { type: "operator", start: 15, end: 16 },
//   { type: "plain", start: 16, end: 17 },
//   { type: "number", start: 17, end: 19 },
//   { type: "plain", start: 19, end: 20 }
// ]
```

### Synchronous and CPU fallback

For Node.js scripts, SSR, CLI tools, or environments without a GPU, use the synchronous methods directly:

```typescript
import { parseSync, highlightSync, parseFlatSync } from "tint-lexer";

const spans = parseSync("const answer = 42;");
```

### Zero-allocation flat spans

When processing large files in hot loops, `parseFlat` returns a single `Uint32Array` containing `[start, end, classId, ...]` without allocating token objects on the V8 heap:

```typescript
import { parseFlat, parseFlatSync, CLASSES } from "tint-lexer";

const flat = parseFlatSync("const answer = 42;");
for (let i = 0; i < flat.length; i += 3) {
  const start = flat[i];
  const end = flat[i + 1];
  const className = CLASSES[flat[i + 2]];
}
```

You can also iterate spans without allocating any array using `forEachSpan`:

```typescript
import { forEachSpanSync } from "tint-lexer";

forEachSpanSync(sourceCode, (start, end, classId, className) => {
  // Process token boundaries directly
});
```

### HTML and terminal formatting

Format source code directly to HTML or ANSI color escape sequences:

```typescript
import { highlightToHtml, highlightToHtmlSync, highlightAnsiSync } from "tint-lexer";

// Direct HTML generation without intermediate span objects
const html = highlightToHtmlSync("const x = 1;", { pre: true });

// Terminal ANSI output
const ansi = highlightAnsiSync("function hello() {}");
```

### Highlighter instance

Reuse one instance across multiple files to avoid pipeline recreation:

```typescript
import { Highlighter } from "tint-lexer";

const highlighter = await Highlighter.load();

// Checks navigator.gpu. If unavailable, falls back to CPU automatically.
console.log(highlighter.isCpu);

const spans = await highlighter.highlight("function hello() { return 'world'; }");

// Clean up GPU buffers when done
highlighter.dispose();
```

### Web worker

Offload highlighting to a background thread to prevent UI frame drops:

```typescript
import { createWorkerHighlighter } from "tint-lexer";

const worker = createWorkerHighlighter();
const spans = await worker.highlight(largeCodeString);

worker.dispose();
```

## How it works

1. **Tokenization.** A Rust tokenizer compiled to a 10 KiB WebAssembly binary scans source text into structural tokens, computing six categorical feature IDs per token in a single linear pass.
2. **Inference.** A two-layer window MLP with state context inspects adjacent tokens (radius 4) to predict token categories. Weights are stored in signed 4-bit format with float32 scales per block of 64 weights.
3. **WebGPU execution.** Compute shaders dispatch workgroups of 64 threads across embedding lookup, hidden activation with tanh, and argmax classification.
4. **CPU fallback.** When WebGPU is absent (Node.js, Bun, server environments, or older browsers), the runtime runs pure JavaScript inference with buffer pooling, returning identical tokens.

## Development

Run tests:

```sh
cargo test --workspace
npm test
```

Build the standalone bundle:

```sh
cargo build -p tint-tokenizer --target wasm32-unknown-unknown --profile web
node tools/build-compact.mjs --model artifacts/compact-q4
```

## License

MIT
