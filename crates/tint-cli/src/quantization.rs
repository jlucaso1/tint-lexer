use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use serde::Serialize;
use tint_model::{
    MAX_WEIGHTS_BYTES, ModelMetadata,
    quantization::{QuantizedMetadata, quantize},
};

use crate::{data::read_bounded, write_json};

#[derive(Args)]
pub struct ConvertArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Serialize)]
struct QuantizationSummary {
    original_bytes: usize,
    quantized_bytes: usize,
    parameter_count: usize,
    max_absolute_error: f64,
    rmse: f64,
    source_weights_sha256: String,
    weights_sha256: String,
}

pub fn export(args: &ConvertArgs) -> Result<()> {
    let metadata: ModelMetadata =
        serde_json::from_slice(&read_bounded(&args.model.join("model.json"), 64 * 1024)?)
            .context("parsing model.json")?;
    metadata.validate()?;
    let weights = read_bounded(&args.model.join("model.bin"), MAX_WEIGHTS_BYTES)?;
    let (quantized, bytes) = quantize(&metadata, &weights)?;
    let (_, restored) = quantized.dequantize(&bytes)?;
    let mut max_absolute_error = 0.0_f64;
    let mut squared_error = 0.0_f64;
    for (a, b) in weights
        .as_chunks::<4>()
        .0
        .iter()
        .zip(restored.as_chunks::<4>().0)
    {
        let error = f64::from(f32::from_le_bytes(*a)) - f64::from(f32::from_le_bytes(*b));
        max_absolute_error = max_absolute_error.max(error.abs());
        squared_error += error * error;
    }
    let summary = QuantizationSummary {
        original_bytes: weights.len(),
        quantized_bytes: bytes.len(),
        parameter_count: metadata.config.parameter_count(),
        max_absolute_error,
        rmse: (squared_error / metadata.config.parameter_count() as f64).sqrt(),
        source_weights_sha256: quantized.source_weights_sha256.clone(),
        weights_sha256: quantized.weights_sha256.clone(),
    };
    let metadata_json = serde_json::to_vec(&quantized)?;
    let summary_json = serde_json::to_vec_pretty(&summary)?;
    fs::create_dir(&args.output)
        .with_context(|| format!("creating new output directory {}", args.output.display()))?;
    fs::write(args.output.join("model.q4.json"), metadata_json)?;
    fs::write(args.output.join("model.q4.bin"), bytes)?;
    fs::write(args.output.join("quantization.json"), summary_json)?;
    write_json(&summary)
}

pub fn restore(args: &ConvertArgs) -> Result<()> {
    let metadata: QuantizedMetadata =
        serde_json::from_slice(&read_bounded(&args.model.join("model.q4.json"), 64 * 1024)?)
            .context("parsing model.q4.json")?;
    metadata.validate()?;
    let bytes = read_bounded(&args.model.join("model.q4.bin"), MAX_WEIGHTS_BYTES)?;
    let (restored, weights) = metadata.dequantize(&bytes)?;
    let metadata_json = serde_json::to_vec_pretty(&restored)?;
    let summary = serde_json::json!({
        "original_bytes": weights.len(),
        "quantized_bytes": bytes.len(),
        "parameter_count": restored.config.parameter_count(),
        "source_weights_sha256": metadata.source_weights_sha256,
        "quantized_weights_sha256": metadata.weights_sha256,
        "weights_sha256": restored.weights_sha256,
        "leakage_check": "unavailable",
    });
    fs::create_dir(&args.output)
        .with_context(|| format!("creating new output directory {}", args.output.display()))?;
    fs::write(args.output.join("model.json"), metadata_json)?;
    fs::write(args.output.join("model.bin"), weights)?;
    write_json(&summary)
}
