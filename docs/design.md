# Design

## Data flow

```text
source + Shiki annotations
          |
     versioned JSONL
          |
    tokenizer + alignment
          |
  shuffled token windows -> Burn + Adam -> model.json + model.bin
                                              |
new source -> same tokenizer -> Burn inference -> merged spans
```

`tint-core` does not depend on Burn or a GPU API. `tint-model` is generic over
Burn's `Backend`. The CLI chooses a native backend. The default web deployment
uses `tint-tokenizer` and dedicated WGSL kernels; `tint-web` retains the larger
Burn/WGPU implementation as a reference. Neither model accepts language IDs.

## Token and label contracts

Source strings are UTF-8. Tokens partition the entire source into word runs,
whitespace runs, newline tokens, and individual ASCII symbols. CRLF is one
newline token. Non-ASCII non-whitespace characters join word runs, including
combining marks and emoji.

Core and native spans use half-open UTF-8 byte offsets. The web API converts
them to UTF-16 code units in one forward scan. No substring is interpreted as
HTML during rendering.

Document version 1 requires `version`, `id`, `group`, `language`, `source`, and
`spans`. Unknown fields are errors. Spans must completely cover the source,
without gaps, overlap, empty spans, or partial Unicode characters. Empty source
requires empty spans.

Each non-whitespace structural token receives a label only when its teacher
spans agree on one class. Conflicting annotations are counted and excluded.
Whitespace remains context but always predicts `plain`.

## Features and model

Feature version 1 has six categorical IDs per token. IDs are disjoint by feature;
zero is a learned boundary-padding ID.

| Feature | IDs |
| --- | --- |
| Structural kind | 1 through 4 |
| Scalar length, capped at 16 | 5 through 20 |
| Six-bit character-shape mask | 21 through 84 |
| First scalar, non-ASCII mapped to 127 | 85 through 212 |
| Last scalar, non-ASCII mapped to 127 | 213 through 340 |
| Wrapping FNV-1a hash of UTF-8 bytes, modulo 1024 | 341 through 1364 |

The shape mask records lowercase, uppercase, numeric, underscore, whitespace,
and other characters. See `tokenize` for the exact versioned calculation.

`window-mlp-v1` defaults to radius 4, embedding dimension 24, and hidden dimension
64 in the Burn reference. The compact training preset uses embedding dimension 8
and the same radius and hidden dimension. For the reference configuration, a
batch of B token positions has this forward pass:

```text
categorical IDs   [B, 9 * 6]
embedding        [B, 9, 6, 24]
feature mean     [B, 9, 24]
flatten          [B, 216]
linear + tanh    [B, 64]
linear logits    [B, 9]
```

Training applies cross-entropy to logits. Inference takes argmax. Windows are
built from the original document, so batching never truncates their context.
Batch boundaries do not alter the model's receptive field.

## Artifacts

`model.json` declares format version, feature version, architecture, dimensions,
ordered classes, and the SHA-256 of `model.bin`. Validation runs before creating
weight tensors. Dimensions, byte count, checksum, and finite weight values must
all pass.

`model.bin` stores row-major, little-endian float32 values in this order:

1. Embedding weights, `[1365, embedding_dim]`.
2. Hidden weights, `[window_size * embedding_dim, hidden_dim]`.
3. Hidden bias, `[hidden_dim]`.
4. Output weights, `[hidden_dim, 9]`.
5. Output bias, `[9]`.

The format does not depend on Burn's record serialization. `report.json` records
the training run and input fingerprints separately. Deployment needs only the
metadata and weights. These are inference artifacts, not resumable optimizer
checkpoints.

`model.q4.json` and `model.q4.bin` form a separate deployment format. Quantization
`q4-block64-v1` divides the same parameter order into blocks of at most 64 values.
Each block stores a little-endian float32 scale followed by two signed codes per
byte, low nibble first. Codes 1 through 15 represent -7 through 7; zero is invalid
except for the unused high nibble in a partial final block.

Scale is the block's maximum absolute value divided by 7, bounded below by
`f32::MIN_POSITIVE`. All-zero blocks use scale 1. Rounding uses ties away from
zero. Export reports maximum absolute error and RMSE. Dequantization validates
all blocks and assigns the reconstructed float weights a new checksum.

`source_weights_sha256` records provenance, not identity of reconstructed floats.
Dequantization does not copy the original training report. Evaluation explicitly
marks the resulting artifact's training-leakage check unavailable.

## Compact execution

The tokenizer streams core tokens into four u32 values per token. The first is
the UTF-16 end offset; the remaining three pack pairs of 11-bit feature IDs.
It does not retain a second vector of token structs. Talc manages allocations
for the single-threaded WASM target, and Binaryen optimizes the generated module.
The shipped module exports a zero-import numeric ABI: `tint_tokenize`,
`tint_tokenize_and_infer`, token and label pointer getters, and a minimal
malloc family. Errors are negative codes, so the JavaScript glue needs no
string marshaling. The wasm-bindgen exports exist only with the `compat`
cargo feature, for the compact playground.

Inference uploads the packed array once. Three kernels compute feature means,
hidden activations, and output classes. Whitespace skips hidden-layer work because
its output is always `plain`; its embeddings remain available as context for
neighboring tokens. The kernels reuse scratch buffers across tiles,
with halos taken from absolute document positions. Only actual document edges
use the learned padding embedding. Per-tile uniforms use dynamic offsets.

All tiles run in one compute pass and submission, followed by one label readback.
GPU buffers allocate lazily and grow to power-of-two capacities. Allocation is
bounded to less than 128 MiB across supported dimensions and input limits.
The output spans own their values; subsequent calls cannot mutate earlier output.
Concurrent calls on one instance are rejected. The UI worker serializes calls and
keeps only the newest queued edit.

## Training and evaluation

The CLI initializes Burn and sample shuffling with the requested seed. It uses
Adam and a token-weighted mean training loss. Validation runs without autodiff
after each epoch. Highest validation macro F1 wins; ties retain the earlier
epoch. CPU tests check repeated seeded runs, but reproducibility across devices,
backends, compiler versions, or dependency updates is not promised.

Splits must have disjoint IDs, groups, and exact source hashes. Use repository
groups rather than randomly splitting files from the same project. Near-duplicate
detection and representative corpus collection remain future work.

Confusion-matrix rows are true labels and columns are predictions. Macro F1
averages classes with nonzero true support. Reports also include accuracy over
scored tokens, per-class support, whitespace exclusions, and ambiguous-token
exclusions. Test results with `validation_overlap: true` are validation reuse,
not independent evaluation.

## Extending the prototype

- Add corpora through the manifest and keep project groups intact. Inspect ambiguous-label counts before changing the model.
- Tune local capacity through `ModelConfig`. All dimensions are persisted and validated.
- Add recurrent or tree context as a new architecture, with its own artifact discriminator and CPU reference tests. Do not reinterpret `window-mlp-v1` weights.
- Bump `FEATURE_VERSION` when feature values or tokenization change. Retrain rather than silently loading old weights.
- Evaluate quantization changes against the float model before changing the deployment format. Size reduction alone is insufficient.
- Profile kernel changes and preserve numerical parity against the Burn reference. Do not change training and inference semantics independently.
- Add incremental inference only with an explicit cache-invalidation contract. A local-window model and a global-context model require different invalidation rules.

There is no plugin system or universal model trait beyond Burn's backend API.
Introduce another abstraction when a second implementation needs it.
