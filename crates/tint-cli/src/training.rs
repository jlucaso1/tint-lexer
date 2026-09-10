use std::{fs, path::PathBuf};

use anyhow::{Context, Result, ensure};
use burn::{
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
};
use clap::{Args, ValueEnum};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use tint_core::{STATE_INPUTS, SyntaxClass, window_features};
use tint_model::{ModelConfig, ModelMetadata, WindowModel, input_tensor};

use crate::{
    BackendChoice, RuntimeArgs,
    data::{Dataset, Fingerprint, MAX_DATA_BYTES, MAX_TOTAL_TOKENS, overlaps},
    evaluate,
    metrics::Evaluation,
};

#[derive(Args)]
pub struct TrainArgs {
    #[arg(long)]
    train: PathBuf,
    #[arg(long)]
    validation: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "20", value_parser = clap::value_parser!(u32).range(1..=1000))]
    epochs: u32,
    #[arg(long, default_value = "0.003", value_parser = learning_rate)]
    learning_rate: f64,
    #[arg(long, default_value = "42")]
    seed: u64,
    #[arg(long, default_value = "4")]
    radius: usize,
    #[arg(long, default_value = "24")]
    embedding_dim: usize,
    #[arg(long, default_value = "64")]
    hidden_dim: usize,
    /// Snap weights to the Q4 block grid after each optimizer step.
    #[arg(long, default_value_t = false)]
    qat: bool,
    /// Per-class loss multipliers: none, or inverse-sqrt train support.
    #[arg(long, value_enum, default_value = "none")]
    class_weights: ClassWeightMode,
    /// Exponent on the inverse-support weights (0.5 reproduces inverse-sqrt).
    #[arg(long, default_value = "0.5", value_parser = nonnegative_rate)]
    class_weight_power: f64,
    /// Label smoothing alpha in [0, 1).
    #[arg(long, default_value = "0", value_parser = unit_rate)]
    label_smoothing: f64,
    /// L2 mean penalty on the embedding matrix for Q4 compressibility.
    #[arg(long, default_value = "0", value_parser = nonnegative_rate)]
    embedding_penalty: f64,
    /// Feed center-token document-context bits to the hidden layer (v2).
    #[arg(long, default_value_t = false)]
    context_state: bool,
    #[command(flatten)]
    pub runtime: RuntimeArgs,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum ClassWeightMode {
    #[default]
    None,
    InverseSqrt,
}

fn default_class_weight_power() -> f64 {
    0.5
}

fn unit_rate(value: &str) -> std::result::Result<f64, String> {
    let rate = value
        .parse::<f64>()
        .map_err(|_| "rate must be a number".to_owned())?;
    if rate.is_finite() && (0.0..1.0).contains(&rate) {
        Ok(rate)
    } else {
        Err("rate must be finite and in [0, 1)".into())
    }
}

fn nonnegative_rate(value: &str) -> std::result::Result<f64, String> {
    let rate = value
        .parse::<f64>()
        .map_err(|_| "rate must be a number".to_owned())?;
    if rate.is_finite() && rate >= 0.0 && (rate as f32).is_finite() {
        Ok(rate)
    } else {
        Err("rate must be finite, nonnegative, and representable as f32".into())
    }
}

fn learning_rate(value: &str) -> std::result::Result<f64, String> {
    let rate = value
        .parse::<f64>()
        .map_err(|_| "learning rate must be a number".to_owned())?;
    if rate.is_finite() && rate > 0.0 && (rate as f32).is_finite() && rate as f32 > 0.0 {
        Ok(rate)
    } else {
        Err("learning rate must be finite, positive, and representable as f32".into())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputReport {
    path: PathBuf,
    sha256: String,
    pub documents: Vec<Fingerprint>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Epoch {
    epoch: u32,
    train_loss: f64,
    validation: Evaluation,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub format_version: u32,
    pub weights_sha256: String,
    pub config: ModelConfig,
    seed: u64,
    epochs: u32,
    batch_size: usize,
    learning_rate: f64,
    backend: BackendChoice,
    optimizer: String,
    selection: String,
    metric_definition: String,
    best_epoch: u32,
    #[serde(default)]
    pub qat: bool,
    #[serde(default)]
    class_weight_mode: ClassWeightMode,
    #[serde(default = "default_class_weight_power")]
    class_weight_power: f64,
    #[serde(default)]
    class_weights: Vec<f32>,
    #[serde(default)]
    label_smoothing: f64,
    #[serde(default)]
    embedding_penalty: f64,
    pub train: InputReport,
    pub validation: InputReport,
    best_train_metrics: Evaluation,
    best_validation_metrics: Evaluation,
    history: Vec<Epoch>,
}

pub async fn train<B: AutodiffBackend>(args: TrainArgs, device: &B::Device) -> Result<Report> {
    match fs::symlink_metadata(&args.output) {
        Ok(_) => anyhow::bail!(
            "output {} already exists; choose a new directory",
            args.output.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("checking output directory"),
    }
    let config = ModelConfig {
        radius: args.radius,
        embedding_dim: args.embedding_dim,
        hidden_dim: args.hidden_dim,
        context_state: args.context_state,
    };
    config.validate()?;
    let train = Dataset::load(&args.train)?;
    let validation = Dataset::load(&args.validation)?;
    ensure!(
        train.token_count + validation.token_count <= MAX_TOTAL_TOKENS,
        "training and validation together exceed {MAX_TOTAL_TOKENS} tokens"
    );
    if let Some(reason) = overlaps(&train.fingerprints, &validation.fingerprints) {
        anyhow::bail!("split leakage: {reason}");
    }
    let mut samples = Vec::new();
    for (document_index, document) in train.documents.iter().enumerate() {
        for (token_index, label) in document.labels.iter().enumerate() {
            if let Some(label) = label {
                samples.push((document_index, token_index, label.id() as i32));
            }
        }
    }
    B::seed(device, args.seed);
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut model = config.init::<B>(device)?;
    let mut optimizer = AdamConfig::new().init::<B, WindowModel<B>>();
    let class_weights = match args.class_weights {
        ClassWeightMode::None => Vec::new(),
        ClassWeightMode::InverseSqrt => {
            let power = args.class_weight_power;
            ensure!(
                power.is_finite() && power >= 0.0,
                "class weight power must be finite and nonnegative"
            );
            let mut support = [0u64; SyntaxClass::ALL.len()];
            for &(_, _, label) in &samples {
                support[label as usize] += 1;
            }
            let total = samples.len() as f64;
            let mut weights: Vec<f32> = support
                .iter()
                .map(|&count| {
                    if count > 0 {
                        // sqrt() for the default keeps legacy weights bit-identical;
                        // powf(0.5) can differ in the last ulp.
                        if power == 0.5 {
                            (total / count as f64).sqrt() as f32
                        } else {
                            (total / count as f64).powf(power) as f32
                        }
                    } else {
                        1.0
                    }
                })
                .collect();
            let mean = weights.iter().sum::<f32>() / weights.len() as f32;
            for weight in &mut weights {
                *weight /= mean;
            }
            ensure!(
                weights
                    .iter()
                    .all(|weight| weight.is_finite() && *weight > 0.0),
                "class weights must be finite and positive"
            );
            weights
        }
    };
    let mut loss_config = CrossEntropyLossConfig::new();
    if !class_weights.is_empty() {
        loss_config.weights = Some(class_weights.clone());
    }
    if args.label_smoothing > 0.0 {
        loss_config.smoothing = Some(args.label_smoothing as f32);
    }
    let loss_fn = loss_config.init::<B>(device);
    let penalty_weight = (args.embedding_penalty > 0.0)
        .then(|| Tensor::<B, 1>::from_floats([(args.embedding_penalty) as f32], device));
    let mut best = None;
    let mut best_score = -1.0;
    let mut best_epoch = 0;
    let mut history = Vec::new();
    let mut history_bytes = 0;
    for epoch in 1..=args.epochs {
        samples.shuffle(&mut rng);
        let mut loss_sum = 0.0;
        for batch in samples.chunks(args.runtime.batch_size) {
            let mut features = Vec::with_capacity(batch.len() * config.input_width());
            let mut labels = Vec::with_capacity(batch.len());
            for &(document, token, label) in batch {
                features.extend(window_features(
                    &train.documents[document].tokens,
                    token,
                    config.radius,
                ));
                labels.push(label);
            }
            let input = input_tensor(features, batch.len(), &config, device)?;
            let targets =
                Tensor::<B, 1, Int>::from_data(TensorData::new(labels, [batch.len()]), device);
            let batch_states = if config.context_state {
                let mut flat = Vec::with_capacity(batch.len() * STATE_INPUTS);
                for &(document, token, _) in batch {
                    let states = &train.documents[document].states;
                    let previous = token
                        .checked_sub(1)
                        .map(|index| states[index] & 0b111)
                        .unwrap_or(0);
                    let next = states.get(token + 1).copied().unwrap_or(0) & 0b111;
                    flat.extend_from_slice(&tint_model::expand_state(
                        states[token],
                        previous,
                        next,
                    ));
                }
                Some(Tensor::<B, 2>::from_data(
                    TensorData::new(flat, [batch.len(), STATE_INPUTS]),
                    device,
                ))
            } else {
                None
            };
            if args.qat {
                let (_, dequantized) = tint_model::dequantized_bytes(&model, &config)
                    .await
                    .map_err(|error| anyhow::anyhow!("QAT fake quantization failed: {error}"))?;
                let quantum: Vec<f32> = dequantized
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect();
                let simulated = tint_model::fake_quantized(&model, &config, &quantum, device)
                    .map_err(|error| anyhow::anyhow!("QAT graph failed: {error}"))?;
                let cross_entropy =
                    loss_fn.forward(simulated.forward(input, batch_states), targets);
                let loss = if let Some(penalty) = penalty_weight.as_ref() {
                    cross_entropy + model.embedding_l2() * penalty.clone()
                } else {
                    cross_entropy
                };
                let data = loss
                    .clone()
                    .into_data_async()
                    .await
                    .map_err(|error| anyhow::anyhow!("loss readback failed: {error:?}"))?;
                let value = data.iter::<f32>().next().context("empty loss tensor")? as f64;
                ensure!(
                    value.is_finite(),
                    "non-finite loss at epoch {epoch}; reduce --learning-rate"
                );
                loss_sum += value * batch.len() as f64;
                // The simulated copy is forward-only; its intermediates drain
                // into the float leaves, so gradients come from the model.
                let gradients = GradientsParams::from_grads(loss.backward(), &model);
                model = optimizer.step(args.learning_rate, model, gradients);
                continue;
            }
            let cross_entropy = loss_fn.forward(model.forward(input, batch_states), targets);
            let loss = if let Some(penalty) = penalty_weight.as_ref() {
                cross_entropy + model.embedding_l2() * penalty.clone()
            } else {
                cross_entropy
            };
            let data = loss
                .clone()
                .into_data_async()
                .await
                .map_err(|error| anyhow::anyhow!("loss readback failed: {error:?}"))?;
            let value = data.iter::<f32>().next().context("empty loss tensor")? as f64;
            ensure!(
                value.is_finite(),
                "non-finite loss at epoch {epoch}; reduce --learning-rate"
            );
            loss_sum += value * batch.len() as f64;
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(args.learning_rate, model, gradients);
        }
        // QAT validates the dequantized grid weights so epoch selection
        // rewards the artifact that actually ships, not the float shadow.
        let (valid, best_candidate) = if args.qat {
            let (metadata, dequantized) = tint_model::dequantized_bytes(&model, &config)
                .await
                .map_err(|error| anyhow::anyhow!("QAT validation failed: {error}"))?;
            let grid = metadata.load::<B::InnerBackend>(&dequantized, device)?;
            let float = model.valid().weights().await?;
            (grid, float)
        } else {
            let valid = model.valid();
            let bytes = valid.weights().await?;
            (valid, bytes)
        };
        let metrics = evaluate(
            &valid,
            &config,
            &validation,
            args.runtime.batch_size,
            device,
        )
        .await?;
        let score = metrics.overall.macro_f1;
        if score > best_score {
            best = Some(best_candidate);
            best_score = score;
            best_epoch = epoch;
        }
        let train_loss = loss_sum / samples.len() as f64;
        eprintln!(
            "epoch {epoch}/{} train_loss={train_loss:.6} validation_macro_f1={score:.6} validation_agreement={:.6}",
            args.epochs, metrics.overall.agreement
        );
        let entry = Epoch {
            epoch,
            train_loss,
            validation: metrics,
        };
        history_bytes += serde_json::to_vec(&entry)?.len();
        ensure!(
            history_bytes <= MAX_DATA_BYTES / 2,
            "training history exceeds 32 MiB; reduce epochs or language count"
        );
        history.push(entry);
    }
    let weights = best.context("training produced no best model")?;
    let metadata = ModelMetadata::new(config.clone(), &weights)?;
    let best_model = metadata.load::<B::InnerBackend>(&weights, device)?;
    let best_train_metrics = evaluate(
        &best_model,
        &config,
        &train,
        args.runtime.batch_size,
        device,
    )
    .await?;
    let best_validation_metrics = history[(best_epoch - 1) as usize].validation.clone();
    let report = Report {
        format_version: 1, weights_sha256: metadata.weights_sha256.clone(), config,
        seed: args.seed, epochs: args.epochs, batch_size: args.runtime.batch_size,
        learning_rate: args.learning_rate, backend: args.runtime.backend,
        optimizer: if args.qat {
            "Adam, Burn 0.21 defaults, QAT straight-through q4-block64-v1".into()
        } else {
            "Adam, Burn 0.21 defaults".into()
        },
        selection: "maximum validation macro_f1; earliest epoch wins ties".into(),
        metric_definition: "Token agreement and per-class precision/recall/F1 exclude whitespace and ambiguous labels. Macro F1 averages only classes with true support; zero denominators yield zero. Confusion rows are truth, columns are prediction, in model class order.".into(),
        best_epoch,
        qat: args.qat,
        class_weight_mode: args.class_weights,
        class_weight_power: args.class_weight_power,
        class_weights: class_weights.clone(),
        label_smoothing: args.label_smoothing,
        embedding_penalty: args.embedding_penalty,
        train: InputReport { path: args.train, sha256: train.sha256, documents: train.fingerprints },
        validation: InputReport { path: args.validation, sha256: validation.sha256, documents: validation.fingerprints },
        best_train_metrics, best_validation_metrics, history,
    };
    let report_bytes = serde_json::to_vec(&report)?;
    ensure!(
        report_bytes.len() <= MAX_DATA_BYTES,
        "report exceeds 64 MiB"
    );
    let metadata_bytes = serde_json::to_vec_pretty(&metadata)?;
    fs::create_dir(&args.output)
        .with_context(|| format!("creating new output directory {}", args.output.display()))?;
    let save = (|| -> Result<()> {
        fs::write(args.output.join("model.bin"), &weights)?;
        fs::write(args.output.join("model.json"), metadata_bytes)?;
        fs::write(args.output.join("report.json"), report_bytes)?;
        Ok(())
    })();
    save.with_context(|| format!("saving {}; output may be incomplete", args.output.display()))?;
    Ok(report)
}
