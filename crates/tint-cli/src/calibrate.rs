use std::{fs, path::PathBuf};

use anyhow::{Context, Result, ensure};
use burn::tensor::backend::Backend;
use clap::{Args, ValueEnum};
use serde::Serialize;
use tint_core::SyntaxClass;
use tint_model::{ModelMetadata, WindowModel};

use crate::{
    RuntimeArgs,
    data::{Dataset, read_bounded},
    metrics::Counts,
    write_json,
};

#[derive(Args)]
pub struct CalibrateArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value = "macro-f1")]
    metric: CalibrateMetric,
    #[command(flatten)]
    pub runtime: RuntimeArgs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum CalibrateMetric {
    MacroF1,
    Agreement,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct CalibrationReport {
    format_version: u32,
    metric: CalibrateMetric,
    source_weights_sha256: String,
    weights_sha256: String,
    data_sha256: String,
    backend: String,
    batch_size: usize,
    offsets: Vec<f32>,
    before: f64,
    after: f64,
    evaluated_tokens: u64,
    selection: String,
}

fn score(rows: &[(usize, [f32; 9])], offsets: &[f64; 9], metric: CalibrateMetric) -> (f64, u64) {
    let mut counts = Counts::default();
    for &(truth, logits) in rows {
        let mut best = 0;
        let mut maximum = f64::NEG_INFINITY;
        for (class, logit) in logits.iter().enumerate() {
            let value = f64::from(*logit) + offsets[class];
            if value > maximum {
                maximum = value;
                best = class;
            }
        }
        counts.observe(
            false,
            SyntaxClass::try_from(truth).ok(),
            SyntaxClass::try_from(best).unwrap_or(SyntaxClass::Plain),
        );
    }
    let finished = counts.finish();
    let value = match metric {
        CalibrateMetric::MacroF1 => finished.macro_f1,
        CalibrateMetric::Agreement => finished.agreement,
    };
    (value, finished.evaluated_tokens)
}

fn search(rows: &[(usize, [f32; 9])], metric: CalibrateMetric) -> ([f64; 9], f64) {
    const GRID: [f64; 13] = [
        0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0,
    ];
    let mut offsets = [0.0; 9];
    let (mut best, _) = score(rows, &offsets, metric);
    for _ in 0..4 {
        let mut improved = false;
        for class in 0..9 {
            for candidate in GRID {
                let mut trial = offsets;
                trial[class] = candidate;
                let (value, _) = score(rows, &trial, metric);
                if value > best {
                    best = value;
                    offsets = trial;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    (offsets, best)
}

pub async fn calibrate<B: Backend>(args: CalibrateArgs, device: &B::Device) -> Result<()> {
    match fs::symlink_metadata(&args.output) {
        Ok(_) => anyhow::bail!(
            "output {} already exists; choose a new directory",
            args.output.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("checking output directory"),
    }
    let path = args.model.join("model.json");
    let metadata: ModelMetadata = serde_json::from_slice(&read_bounded(&path, 64 * 1024)?)
        .with_context(|| format!("parsing {}", path.display()))?;
    metadata.validate()?;
    let weights = read_bounded(&args.model.join("model.bin"), tint_model::MAX_WEIGHTS_BYTES)?;
    let model: WindowModel<B> = metadata.load(&weights, device)?;
    let dataset = Dataset::load(&args.data)?;
    let mut rows = Vec::new();
    for document in &dataset.documents {
        let states = metadata
            .config
            .context_state
            .then_some(document.states.as_slice());
        let logits = tint_model::predict_logits(
            &model,
            &metadata.config,
            &document.tokens,
            states,
            args.runtime.batch_size,
            device,
        )
        .await?;
        for ((token, label), logit) in document.tokens.iter().zip(&document.labels).zip(logits) {
            if token.is_whitespace() {
                continue;
            }
            if let (Some(truth), Some(logit)) = (label, logit) {
                rows.push((truth.id(), logit));
            }
        }
    }
    ensure!(!rows.is_empty(), "dataset has no scorable rows");
    let (before, tokens) = score(&rows, &[0.0; 9], args.metric);
    let (offsets, after) = search(&rows, args.metric);
    ensure!(
        after >= before,
        "calibration search regressed its own metric"
    );
    let mut calibrated = weights.clone();
    let bias = calibrated.len() - SyntaxClass::ALL.len() * 4;
    for (class, offset) in offsets.iter().enumerate() {
        let slot = &mut calibrated[bias + class * 4..bias + (class + 1) * 4];
        let value = f32::from_le_bytes(slot.try_into().unwrap()) + (*offset as f32);
        ensure!(value.is_finite(), "calibrated bias is non-finite");
        slot.copy_from_slice(&value.to_le_bytes());
    }
    let calibrated_metadata = ModelMetadata::new(metadata.config.clone(), &calibrated)?;
    let report = CalibrationReport {
        format_version: 2,
        metric: args.metric,
        source_weights_sha256: metadata.weights_sha256.clone(),
        weights_sha256: calibrated_metadata.weights_sha256.clone(),
        data_sha256: dataset.sha256.clone(),
        backend: format!("{:?}", args.runtime.backend).to_lowercase(),
        batch_size: args.runtime.batch_size,
        offsets: offsets.iter().map(|offset| *offset as f32).collect(),
        before,
        after,
        evaluated_tokens: tokens,
        selection: "coordinate ascent on the calibration split only; strict improvement keeps smaller absolute offsets first".into(),
    };
    fs::create_dir(&args.output)
        .with_context(|| format!("creating new output directory {}", args.output.display()))?;
    fs::write(args.output.join("model.bin"), &calibrated)?;
    fs::write(
        args.output.join("model.json"),
        serde_json::to_vec_pretty(&calibrated_metadata)?,
    )?;
    fs::write(
        args.output.join("calibration.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    write_json(&report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<(usize, [f32; 9])> {
        let mut rows = Vec::new();
        for _ in 0..20 {
            let mut plain = [0.0; 9];
            plain[0] = 1.0;
            rows.push((0, plain));
        }
        for _ in 0..10 {
            let mut keyword = [0.0; 9];
            keyword[0] = 0.4;
            keyword[4] = 0.0;
            rows.push((4, keyword));
        }
        rows
    }

    #[test]
    fn search_lifts_keyword_recall_without_hurting_plain() {
        let rows = rows();
        let (before, _) = score(&rows, &[0.0; 9], CalibrateMetric::MacroF1);
        let (offsets, after) = search(&rows, CalibrateMetric::MacroF1);
        assert!(after > before);
        let (value, _) = score(&rows, &offsets, CalibrateMetric::Agreement);
        assert!((value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn agreement_metric_is_supported() {
        let rows = rows();
        let (value, tokens) = score(&rows, &[0.0; 9], CalibrateMetric::Agreement);
        assert_eq!(tokens, 30);
        assert!((value - 20.0 / 30.0).abs() < 1e-12);
    }
}
