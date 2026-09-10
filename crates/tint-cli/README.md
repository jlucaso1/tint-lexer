# tint CLI

Run from the workspace with `cargo run -p tint-cli -- <command>`, or build with
`cargo build -p tint-cli --release` and use `target/release/tint`.

```sh
tint train --train train.jsonl --validation validation.jsonl --output new-model
tint evaluate --data test.jsonl --model new-model
tint highlight --model new-model --input example.rs
tint highlight --model new-model --input example.rs --utf16
tint quantize --model new-model --output new-model-q4
tint dequantize --model new-model-q4 --output new-model-restored
```

Training, evaluation, and highlighting accept `--backend cpu|wgpu` and `--batch-size`, defaulting to `cpu`
and `128`. CPU uses Burn 0.21 NdArray. WGPU requires a build with
`cargo build -p tint-cli --release --features wgpu` and a working GPU driver.
The CLI does not silently fall back to CPU.

Training options default to `--epochs 20 --learning-rate 0.003 --seed 42
--radius 4 --embedding-dim 24 --hidden-dim 64`. Adam uses Burn defaults.
Burn initialization and ChaCha8 sample shuffling are seeded separately with the
same seed. Results are not guaranteed to match across backends or hardware.
Windows are generated per batch, not stored for the whole corpus.

The output directory must not exist. Its parent must exist. The CLI creates the
directory only after training and report serialization succeed. Failed writes
can leave an incomplete directory, which must not be treated as an artifact.
There is no resume or checkpoint support.

## Data and metrics

Each nonempty JSONL line must be a strict `tint-core` version 1 `Document`.
Blank lines, unknown fields, invalid offsets, and incomplete span coverage are
errors. A final newline is optional. Error messages identify the file and line.
Each dataset must contain at least one unambiguous non-whitespace label.
IDs must be unique within and across training splits. Groups and source SHA256
hashes must not cross training and validation splits.

Whitespace and tokens crossing conflicting class annotations are excluded from
training and scoring. Reports count both exclusions separately and include the
total non-whitespace token count. Agreement is accuracy over scored tokens.
Macro F1 averages only classes with nonzero true support. Zero denominators
produce zero. Every overall and language report includes all nine classes and
a 9-by-9 confusion matrix. Rows are true classes and columns are predictions,
in the order stored in `model.json`.

Validation runs without autodiff after each epoch. The highest validation macro
F1 selects the saved weights; the earliest epoch wins ties. `report.json`
contains all epoch training losses and validation metrics, metrics for the best
model, configuration, seed, backend, input file hashes, and document fingerprints.
Training loss is the token-weighted mean of batch losses before each update.

`evaluate` verifies the metadata and weight checksum. When `report.json` exists,
it must match the model configuration and weight hash. Evaluation rejects IDs,
groups, or source hashes seen in training. Validation reuse is allowed and sets
`validation_overlap` to `true`; such a result is not an independent test score.
Without a report, evaluation warns on stderr and emits `leakage_check` as
`unavailable` and `validation_overlap` as `null`.
Fingerprints detect exact source copies, not edited duplicates.
Checksums detect corruption, not malicious replacement of an entire artifact.

## Files and limits

- Dataset files are limited to 64 MiB each, JSONL lines to 16 MiB.
- Training and validation together may retain at most 500,000 tokens. Evaluation has the same token cap for its single dataset.
- Each source is limited to 4 MiB and 262,144 tokens.
- Batch size must be 1 through 4096. Epochs must be 1 through 1000.
- Radius must be 0 through 16, embedding dimension 1 through 128, and hidden dimension 1 through 256.
- Learning rate must be finite, positive, and representable as a positive finite `f32`.
- Epoch history is limited to 32 MiB of compact JSON; the complete report is limited to 64 MiB.
- Model metadata is limited to 64 KiB and weights to 8 MiB.

This prototype keeps bounded dataset bytes, token arrays, labels, and sample
indices in memory. Limits are constants, not CLI settings. Tokenization may
temporarily allocate tokens beyond the per-document cap before rejecting the
document. Large batch and model dimensions can still require substantial memory.
Inputs must be regular UTF-8 files. Stdin and HTML output are not supported.

`model.bin` contains little-endian float32 weights; `model.json` describes and
checksums them. Progress and warnings go to stderr. Successful commands emit one
JSON value on stdout. Highlight output is an envelope with `offset_encoding` of
`utf8` or `utf16` and `spans`. Offsets are half-open byte offsets by default, or
UTF-16 code units with `--utf16`. Whitespace receives the `plain` class.

Quantization commands require a new output directory and do not initialize a
tensor backend. Export writes checked `q4-block64-v1` weights and an error report.
Dequantization writes a new float artifact without a training report; its checksum
describes reconstructed weights, not the original float values.
