mod artifact;
pub mod quantization;

pub use artifact::{
    MAX_WEIGHTS_BYTES, ModelMetadata, dequantized_bytes, fake_quantized, project_q4,
};

use burn::{
    module::{Initializer, Module},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{Int, Tensor, TensorData, activation::tanh, backend::Backend},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tint_core::{
    FEATURE_COUNT, FEATURE_VOCAB_SIZE, MAX_SOURCE_BYTES, STATE_INPUTS, Span, SyntaxClass, Token,
    merge_spans, token_context, tokenize, window_features,
};

pub const ARCHITECTURE: &str = "window-mlp-v1";
pub const ARCHITECTURE_V2: &str = "window-mlp-v2";
pub const MAX_BATCH_SIZE: usize = 4096;
pub const MAX_TOKENS: usize = 262_144;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub radius: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    /// Center-token context bits. Skipped in JSON when false so version-1
    /// artifacts keep their exact serialized shape.
    #[serde(default, skip_serializing_if = "is_false")]
    pub context_state: bool,
}

fn is_false(value: &bool) -> bool {
    !value
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            radius: 4,
            embedding_dim: 24,
            hidden_dim: 64,
            context_state: false,
        }
    }
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.radius > 16
            || !(1..=128).contains(&self.embedding_dim)
            || !(1..=256).contains(&self.hidden_dim)
        {
            return Err(ModelError::InvalidConfig);
        }
        Ok(())
    }

    pub fn window_size(&self) -> usize {
        self.radius * 2 + 1
    }

    pub fn input_width(&self) -> usize {
        self.window_size() * FEATURE_COUNT
    }

    /// Extra center-token context inputs appended to the hidden-layer input.
    pub fn state_width(&self) -> usize {
        if self.context_state { STATE_INPUTS } else { 0 }
    }

    pub fn hidden_input_width(&self) -> usize {
        self.window_size() * self.embedding_dim + self.state_width()
    }

    pub fn parameter_count(&self) -> usize {
        FEATURE_VOCAB_SIZE * self.embedding_dim
            + self.hidden_input_width() * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * SyntaxClass::ALL.len()
            + SyntaxClass::ALL.len()
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<WindowModel<B>, ModelError> {
        self.validate()?;
        Ok(WindowModel {
            embedding: EmbeddingConfig::new(FEATURE_VOCAB_SIZE, self.embedding_dim)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.1,
                })
                .init(device),
            hidden: LinearConfig::new(self.hidden_input_width(), self.hidden_dim).init(device),
            output: LinearConfig::new(self.hidden_dim, SyntaxClass::ALL.len()).init(device),
        })
    }
}

#[derive(Module, Debug)]
pub struct WindowModel<B: Backend> {
    embedding: Embedding<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> WindowModel<B> {
    /// Input rows contain flattened categorical features for one token window.
    /// Version-2 models take a second `[batch, 3]` float tensor with the
    /// center-token context bits; version-1 models take `None` instead.
    pub fn forward(&self, input: Tensor<B, 2, Int>, states: Option<Tensor<B, 2>>) -> Tensor<B, 2> {
        let [batch, width] = input.dims();
        let dim = self.embedding.weight.dims()[1];
        let window = width / FEATURE_COUNT;
        if let Some(states) = &states {
            assert_eq!(
                states.dims(),
                [batch, STATE_INPUTS],
                "state tensor must hold one row per batch row"
            );
        }
        let features = self
            .embedding
            .forward(input)
            .reshape([batch, window, FEATURE_COUNT, dim])
            .mean_dim(2)
            .reshape([batch, window * dim]);
        let hidden_input = match states {
            Some(states) => Tensor::cat(vec![features, states], 1),
            None => features,
        };
        self.output.forward(tanh(self.hidden.forward(hidden_input)))
    }

    /// Mean of squared embedding weights, for compressibility penalties.
    /// Stays on the float leaves so gradients reach the trained parameters.
    pub fn embedding_l2(&self) -> Tensor<B, 1> {
        self.embedding.weight.val().square().mean()
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("model requires radius <= 16, embedding dimension 1..=128, hidden dimension 1..=256")]
    InvalidConfig,
    #[error("batch size must be in 1..=4096")]
    InvalidBatchSize,
    #[error("invalid feature shape or categorical ID")]
    InvalidFeatures,
    #[error("source exceeds the 4 MiB or 262144-token limit")]
    SourceTooLarge,
    #[error("unsupported model metadata or feature schema")]
    InvalidMetadata,
    #[error("weight size, checksum, or values are invalid")]
    InvalidWeights,
    #[error("tensor readback failed: {0}")]
    Readback(String),
    #[error(transparent)]
    Core(#[from] tint_core::CoreError),
}

pub fn input_tensor<B: Backend>(
    features: Vec<i32>,
    batch_size: usize,
    config: &ModelConfig,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>, ModelError> {
    config.validate()?;
    if !(1..=MAX_BATCH_SIZE).contains(&batch_size) {
        return Err(ModelError::InvalidBatchSize);
    }
    if features.len() != batch_size * config.input_width()
        || features
            .iter()
            .any(|&id| !(0..FEATURE_VOCAB_SIZE as i32).contains(&id))
    {
        return Err(ModelError::InvalidFeatures);
    }
    Ok(Tensor::from_data(
        TensorData::new(features, [batch_size, config.input_width()]),
        device,
    ))
}

pub async fn predict<B: Backend>(
    model: &WindowModel<B>,
    config: &ModelConfig,
    tokens: &[Token],
    states: Option<&[u8]>,
    batch_size: usize,
    device: &B::Device,
) -> Result<Vec<SyntaxClass>, ModelError> {
    config.validate()?;
    if model.embedding.weight.dims() != [FEATURE_VOCAB_SIZE, config.embedding_dim]
        || model.hidden.weight.dims() != [config.hidden_input_width(), config.hidden_dim]
    {
        return Err(ModelError::InvalidConfig);
    }
    if !(1..=MAX_BATCH_SIZE).contains(&batch_size) {
        return Err(ModelError::InvalidBatchSize);
    }
    if tokens.len() > MAX_TOKENS {
        return Err(ModelError::SourceTooLarge);
    }
    let state_rows = check_states(config, tokens, states)?;
    let mut labels = vec![SyntaxClass::Plain; tokens.len()];
    let indices: Vec<_> = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| (!token.is_whitespace()).then_some(index))
        .collect();
    for batch in indices.chunks(batch_size) {
        let mut features = Vec::with_capacity(batch.len() * config.input_width());
        for &index in batch {
            features.extend(window_features(tokens, index, config.radius));
        }
        let input = input_tensor(features, batch.len(), config, device)?;
        let result = model
            .forward(input, state_batch(&state_rows, batch, device))
            .argmax(1)
            .into_data_async()
            .await
            .map_err(|error| ModelError::Readback(format!("{error:?}")))?;
        for (&index, label) in batch.iter().zip(result.iter::<i32>()) {
            labels[index] = SyntaxClass::try_from(label as usize)?;
        }
    }
    Ok(labels)
}

/// Per-token logits for calibration. `None` marks whitespace, scored like
/// [`predict`]: every other token carries all nine class logits in order.
pub async fn predict_logits<B: Backend>(
    model: &WindowModel<B>,
    config: &ModelConfig,
    tokens: &[Token],
    states: Option<&[u8]>,
    batch_size: usize,
    device: &B::Device,
) -> Result<Vec<Option<[f32; 9]>>, ModelError> {
    config.validate()?;
    if model.embedding.weight.dims() != [FEATURE_VOCAB_SIZE, config.embedding_dim]
        || model.hidden.weight.dims() != [config.hidden_input_width(), config.hidden_dim]
    {
        return Err(ModelError::InvalidConfig);
    }
    if !(1..=MAX_BATCH_SIZE).contains(&batch_size) {
        return Err(ModelError::InvalidBatchSize);
    }
    if tokens.len() > MAX_TOKENS {
        return Err(ModelError::SourceTooLarge);
    }
    let state_rows = check_states(config, tokens, states)?;
    let mut logits: Vec<Option<[f32; 9]>> = vec![None; tokens.len()];
    let indices: Vec<_> = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| (!token.is_whitespace()).then_some(index))
        .collect();
    for batch in indices.chunks(batch_size) {
        let mut features = Vec::with_capacity(batch.len() * config.input_width());
        for &index in batch {
            features.extend(window_features(tokens, index, config.radius));
        }
        let input = input_tensor(features, batch.len(), config, device)?;
        let result = model
            .forward(input, state_batch(&state_rows, batch, device))
            .into_data_async()
            .await
            .map_err(|error| ModelError::Readback(format!("{error:?}")))?;
        let values: Vec<f32> = result.iter().collect();
        if values.len() != batch.len() * SyntaxClass::ALL.len() {
            return Err(ModelError::Readback(
                "logit row width differs from class count".into(),
            ));
        }
        for (position, &index) in batch.iter().enumerate() {
            let mut row = [0.0; 9];
            row.copy_from_slice(&values[position * 9..(position + 1) * 9]);
            logits[index] = Some(row);
        }
    }
    Ok(logits)
}

/// Expands one packed context byte plus neighboring masks to the nine
/// [`tint_core::STATE_INPUTS`] floats, in documented order.
pub fn expand_state(center: u8, previous: u8, next: u8) -> [f32; STATE_INPUTS] {
    let mask = center & 0b111;
    let quote = (center >> 3) & 0b11;
    let distance = (center >> 5) & 0b11;
    [
        f32::from(mask & 0b001 != 0),
        f32::from(mask & 0b010 != 0),
        f32::from(mask & 0b100 != 0),
        f32::from(mask != previous),
        f32::from(mask != next),
        f32::from(quote & 0b01 != 0),
        f32::from(quote & 0b10 != 0),
        f32::from(distance & 0b01 != 0),
        f32::from(distance & 0b10 != 0),
    ]
}

/// Expands packed context bytes to one float row per token in
/// [`tint_core::STATE_INPUTS`] order, or `None` for version-1 configs.
/// Caller and config must agree on the presence of states.
fn check_states(
    config: &ModelConfig,
    tokens: &[Token],
    states: Option<&[u8]>,
) -> Result<Option<Vec<[f32; STATE_INPUTS]>>, ModelError> {
    match (config.context_state, states) {
        (false, None) => Ok(None),
        (true, Some(states)) => {
            if states.len() != tokens.len() {
                return Err(ModelError::InvalidFeatures);
            }
            let masks: Vec<u8> = states.iter().map(|byte| byte & 0b111).collect();
            Ok(Some(
                states
                    .iter()
                    .enumerate()
                    .map(|(index, byte)| {
                        let previous = if index > 0 { masks[index - 1] } else { 0 };
                        let next = masks.get(index + 1).copied().unwrap_or(0);
                        expand_state(*byte, previous, next)
                    })
                    .collect(),
            ))
        }
        _ => Err(ModelError::InvalidFeatures),
    }
}

fn state_batch<B: Backend>(
    rows: &Option<Vec<[f32; STATE_INPUTS]>>,
    batch: &[usize],
    device: &B::Device,
) -> Option<Tensor<B, 2>> {
    rows.as_ref().map(|rows| {
        let mut flat = Vec::with_capacity(batch.len() * STATE_INPUTS);
        for &index in batch {
            flat.extend_from_slice(&rows[index]);
        }
        Tensor::from_data(TensorData::new(flat, [batch.len(), STATE_INPUTS]), device)
    })
}

pub async fn highlight<B: Backend>(
    model: &WindowModel<B>,
    config: &ModelConfig,
    source: &str,
    batch_size: usize,
    device: &B::Device,
) -> Result<Vec<Span>, ModelError> {
    if source.len() > MAX_SOURCE_BYTES {
        return Err(ModelError::SourceTooLarge);
    }
    let tokens = tokenize(source);
    let owned;
    let states = if config.context_state {
        owned = token_context(source, &tokens);
        Some(owned.as_slice())
    } else {
        None
    };
    let labels = predict(model, config, &tokens, states, batch_size, device).await?;
    Ok(merge_spans(&tokens, &labels)?)
}
