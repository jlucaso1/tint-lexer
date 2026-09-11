# tint-lexer

Fast, compact syntax highlighter powered by WebGPU with an automatic CPU fallback.

Everything needed to tokenize and highlight code runs locally. The bundle embeds a 21 KiB WebAssembly tokenizer and a 4-bit quantized neural model (12,871 parameters). There are zero network requests, zero peer dependencies, and zero runtime setup requirements.

## Numbers

Measured from the repo files with Node gzip and Brotli-11 (`dist/index.js` against `web/vendor/gpu-lexer.js`):

| Metric | gpu-lexer | tint-lexer | Difference |
| :--- | :--- | :--- | :--- |
| Minified size | 58.0 KiB | 55.5 KiB | 4.4% smaller |
| Gzip size | 32.3 KiB | 24.3 KiB | 24.8% smaller |
| Brotli size | 27.4 KiB | 21.4 KiB | 21.9% smaller |
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

1. **Tokenization.** A Rust tokenizer compiled to a 21 KiB WebAssembly binary scans source text into structural tokens, computing six categorical feature IDs per token in a single linear pass. The module uses a zero-import numeric ABI, so the JavaScript glue stays small. The legacy wasm-bindgen ABI remains available behind the `tint-tokenizer/compat` cargo feature for the compact playground.
2. **Inference.** A two-layer window MLP with state context inspects adjacent tokens (radius 4) to predict token categories. Weights are stored in signed 4-bit format with float32 scales per block of 64 weights.
3. **WebGPU execution.** Compute shaders dispatch workgroups of 64 threads across embedding lookup, hidden activation with tanh, and argmax classification.
4. **CPU fallback.** When WebGPU is absent (Node.js, Bun, server environments, or older browsers), the runtime runs the same inference in WebAssembly with SIMD, returning identical tokens.

## Development

Run tests:

```sh
cargo test --workspace
npm test
```

Build the standalone bundle (recompiles the WASM tokenizer, re-embeds it, minifies with esbuild, refreshes `dist/` and `web/tint.js`):

```sh
npm run build
```

Check the shipped sizes:

```sh
npm run size
```

Rebuild the legacy wasm-bindgen package for the compact playground (`web/compact/pkg`, gitignored):

```sh
cargo build -p tint-tokenizer --target wasm32-unknown-unknown --profile web --features tint-tokenizer/compat
wasm-bindgen --target web --out-dir web/compact/pkg target/wasm32-unknown-unknown/web/tint_tokenizer.wasm
```

Rebuild the Shiki vendor bundle used by the live demo benchmark (see the command in `tools/vendor-shiki-entry.js`):

```sh
npx esbuild tools/vendor-shiki-entry.js --bundle --minify --format=esm --outfile=web/vendor/shiki.js
```

## License

MIT
