mod calibrate;
mod data;
mod metrics;
mod quantization;
mod training;

use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use burn::{
    backend::{Autodiff, NdArray},
    tensor::backend::Backend,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use tint_model::{ModelMetadata, WindowModel};

use data::{Dataset, MAX_DATA_BYTES, overlaps, read_bounded};
use metrics::{Counts, Evaluation};

#[derive(Parser)]
#[command(
    name = "tint",
    version,
    about = "Train and run a token-window syntax classifier",
    after_help = "JSONL limits: 64 MiB per file, 16 MiB per line, 500000 tokens across training splits. Sources: 4 MiB and 262144 tokens each. Metrics exclude whitespace and ambiguous labels; macro F1 averages classes with true support. Confusion rows are truth, columns are predictions, in model class order."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train with disjoint training and validation documents; save the best validation macro F1.
    Train(training::TrainArgs),
    /// Score labeled JSONL. A matching report rejects training overlap; validation reuse is allowed and reported.
    Evaluate(EvaluateArgs),
    /// Fit output-bias offsets on a labeled split; never on test data twice.
    Calibrate(calibrate::CalibrateArgs),
    /// Emit JSON spans with half-open UTF-8 byte or UTF-16 code-unit offsets.
    Highlight(HighlightArgs),
    /// Export checked Q4 weights without initializing a tensor backend.
    Quantize(quantization::ConvertArgs),
    /// Restore Q4 weights to a float artifact without a training report.
    Dequantize(quantization::ConvertArgs),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum BackendChoice {
    Cpu,
    Wgpu,
}

#[derive(Debug, Clone, Args)]
struct RuntimeArgs {
    #[arg(long, value_enum, default_value = "cpu")]
    backend: BackendChoice,
    #[arg(long, default_value = "128", value_parser = batch_size)]
    batch_size: usize,
}

fn batch_size(value: &str) -> std::result::Result<usize, String> {
    let size = value
        .parse::<usize>()
        .map_err(|_| "batch size must be an integer".to_owned())?;
    if (1..=tint_model::MAX_BATCH_SIZE).contains(&size) {
        Ok(size)
    } else {
        Err("batch size must be in 1..=4096".into())
    }
}

#[derive(Args)]
struct EvaluateArgs {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[command(flatten)]
    runtime: RuntimeArgs,
}

#[derive(Args)]
struct HighlightArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    utf16: bool,
    #[command(flatten)]
    runtime: RuntimeArgs,
}

fn main() {
    if let Err(error) = pollster::block_on(run(Cli::parse())) {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<()> {
    let choice = match &cli.command {
        Command::Train(args) => args.runtime.backend,
        Command::Evaluate(args) => args.runtime.backend,
        Command::Calibrate(args) => args.runtime.backend,
        Command::Highlight(args) => args.runtime.backend,
        Command::Quantize(args) => return quantization::export(args),
        Command::Dequantize(args) => return quantization::restore(args),
    };
    match choice {
        BackendChoice::Cpu => dispatch::<NdArray<f32>>(cli.command, &Default::default()).await,
        BackendChoice::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                dispatch::<burn::backend::Wgpu>(cli.command, &Default::default()).await
            }
            #[cfg(not(feature = "wgpu"))]
            {
                bail!(
                    "WGPU is not enabled in this binary; rebuild with `cargo build -p tint-cli --features wgpu` or use --backend cpu"
                )
            }
        }
    }
}

async fn dispatch<B: Backend>(command: Command, device: &B::Device) -> Result<()> {
    match command {
        Command::Quantize(_) | Command::Dequantize(_) => {
            unreachable!("conversion precedes backend selection")
        }
        Command::Train(args) => write_json(&training::train::<Autodiff<B>>(args, device).await?),
        Command::Calibrate(args) => calibrate::calibrate::<B>(args, device).await,
        Command::Evaluate(args) => {
            let dataset = Dataset::load(&args.data)?;
            let (metadata, model) = load_model::<B>(&args.model, device)?;
            let report_path = args.model.join("report.json");
            let (leakage_check, validation_overlap) = match std::fs::symlink_metadata(&report_path)
            {
                Ok(_) => {
                    let report: training::Report =
                        serde_json::from_slice(&read_bounded(&report_path, MAX_DATA_BYTES)?)
                            .with_context(|| format!("parsing {}", report_path.display()))?;
                    ensure!(
                        report.format_version == 1
                            && report.weights_sha256 == metadata.weights_sha256
                            && report.config == metadata.config,
                        "report does not match the model artifact"
                    );
                    if let Some(reason) = overlaps(&report.train.documents, &dataset.fingerprints) {
                        bail!("evaluation overlaps training data: {reason}");
                    }
                    (
                        "checked_against_training",
                        Some(
                            overlaps(&report.validation.documents, &dataset.fingerprints).is_some(),
                        ),
                    )
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    eprintln!("warning: report.json is absent; training leakage cannot be checked");
                    ("unavailable", None)
                }
                Err(error) => return Err(error).context("reading report metadata"),
            };
            let metrics = evaluate(
                &model,
                &metadata.config,
                &dataset,
                args.runtime.batch_size,
                device,
            )
            .await?;
            write_json(
                &serde_json::json!({ "format_version": 1, "backend": args.runtime.backend, "data_sha256": dataset.sha256, "weights_sha256": metadata.weights_sha256, "leakage_check": leakage_check, "validation_overlap": validation_overlap, "metrics": metrics }),
            )
        }
        Command::Highlight(args) => {
            let source = String::from_utf8(read_bounded(&args.input, tint_core::MAX_SOURCE_BYTES)?)
                .with_context(|| format!("{} must be UTF-8", args.input.display()))?;
            let (metadata, model) = load_model::<B>(&args.model, device)?;
            let mut spans = tint_model::highlight(
                &model,
                &metadata.config,
                &source,
                args.runtime.batch_size,
                device,
            )
            .await?;
            if args.utf16 {
                spans = tint_core::utf16_spans(&source, &spans)?;
            }
            write_json(
                &serde_json::json!({ "offset_encoding": if args.utf16 { "utf16" } else { "utf8" }, "spans": spans }),
            )
        }
    }
}

fn load_model<B: Backend>(
    directory: &Path,
    device: &B::Device,
) -> Result<(ModelMetadata, WindowModel<B>)> {
    let path = directory.join("model.json");
    let metadata: ModelMetadata = serde_json::from_slice(&read_bounded(&path, 64 * 1024)?)
        .with_context(|| format!("parsing {}", path.display()))?;
    metadata.validate()?;
    let weights = read_bounded(&directory.join("model.bin"), tint_model::MAX_WEIGHTS_BYTES)?;
    let model = metadata.load(&weights, device)?;
    Ok((metadata, model))
}

async fn evaluate<B: Backend>(
    model: &WindowModel<B>,
    config: &tint_model::ModelConfig,
    dataset: &Dataset,
    batch_size: usize,
    device: &B::Device,
) -> Result<Evaluation> {
    let mut overall = Counts::default();
    let mut languages = BTreeMap::<String, Counts>::new();
    for document in &dataset.documents {
        let states = config.context_state.then_some(document.states.as_slice());
        let predictions =
            tint_model::predict(model, config, &document.tokens, states, batch_size, device)
                .await?;
        ensure!(
            predictions.len() == document.tokens.len(),
            "prediction count differs from token count"
        );
        let language = languages.entry(document.language.clone()).or_default();
        for ((token, &label), prediction) in document
            .tokens
            .iter()
            .zip(&document.labels)
            .zip(predictions)
        {
            overall.observe(token.is_whitespace(), label, prediction);
            language.observe(token.is_whitespace(), label, prediction);
        }
    }
    Ok(Evaluation {
        overall: overall.finish(),
        per_language: languages
            .into_iter()
            .map(|(language, counts)| (language, counts.finish()))
            .collect(),
    })
}

fn write_json(value: &impl Serialize) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, value)?;
    writeln!(stdout)?;
    Ok(())
}
