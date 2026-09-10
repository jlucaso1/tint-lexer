use burn::{
    module::Param,
    nn::{Embedding, Linear},
    tensor::{Tensor, TensorData, backend::Backend},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tint_core::{FEATURE_VERSION, FEATURE_VERSION_V2, FEATURE_VOCAB_SIZE, SyntaxClass};

use crate::{
    ARCHITECTURE, ARCHITECTURE_V2, ModelConfig, ModelError, WindowModel, quantization::quantize,
};

pub const MAX_WEIGHTS_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelMetadata {
    pub format_version: u32,
    pub feature_version: u32,
    pub architecture: String,
    pub config: ModelConfig,
    pub classes: Vec<SyntaxClass>,
    pub weights_sha256: String,
}

impl ModelMetadata {
    pub fn new(config: ModelConfig, weights: &[u8]) -> Result<Self, ModelError> {
        let (feature_version, architecture) = if config.context_state {
            (FEATURE_VERSION_V2, ARCHITECTURE_V2.to_owned())
        } else {
            (FEATURE_VERSION, ARCHITECTURE.to_owned())
        };
        let metadata = Self {
            format_version: 1,
            feature_version,
            architecture,
            config,
            classes: SyntaxClass::ALL.to_vec(),
            weights_sha256: format!("{:x}", Sha256::digest(weights)),
        };
        metadata.validate_weights(weights)?;
        Ok(metadata)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        self.config.validate()?;
        let versioned = (self.feature_version == FEATURE_VERSION
            && self.architecture == ARCHITECTURE
            && !self.config.context_state)
            || (self.feature_version == FEATURE_VERSION_V2
                && self.architecture == ARCHITECTURE_V2
                && self.config.context_state);
        if self.format_version != 1
            || !versioned
            || self.classes != SyntaxClass::ALL
            || self.weights_sha256.len() != 64
            || !self
                .weights_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(ModelError::InvalidMetadata);
        }
        Ok(())
    }

    pub fn validate_weights(&self, weights: &[u8]) -> Result<(), ModelError> {
        self.validate()?;
        if weights.len() > MAX_WEIGHTS_BYTES
            || weights.len() != self.config.parameter_count() * 4
            || format!("{:x}", Sha256::digest(weights)) != self.weights_sha256
            || weights
                .as_chunks::<4>()
                .0
                .iter()
                .any(|bytes| !f32::from_le_bytes(*bytes).is_finite())
        {
            return Err(ModelError::InvalidWeights);
        }
        Ok(())
    }

    pub fn load<B: Backend>(
        &self,
        weights: &[u8],
        device: &B::Device,
    ) -> Result<WindowModel<B>, ModelError> {
        self.validate_weights(weights)?;
        let values: Vec<f32> = weights
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        let mut offset = 0;
        let mut take = |count: usize| {
            let slice = values[offset..offset + count].to_vec();
            offset += count;
            slice
        };
        let config = &self.config;
        let input_dim = config.hidden_input_width();
        let classes = SyntaxClass::ALL.len();
        Ok(WindowModel {
            embedding: Embedding {
                weight: Param::from_tensor(Tensor::from_data(
                    TensorData::new(
                        take(FEATURE_VOCAB_SIZE * config.embedding_dim),
                        [FEATURE_VOCAB_SIZE, config.embedding_dim],
                    ),
                    device,
                )),
            },
            hidden: Linear {
                weight: Param::from_tensor(Tensor::from_data(
                    TensorData::new(
                        take(input_dim * config.hidden_dim),
                        [input_dim, config.hidden_dim],
                    ),
                    device,
                )),
                bias: Some(Param::from_tensor(Tensor::from_data(
                    TensorData::new(take(config.hidden_dim), [config.hidden_dim]),
                    device,
                ))),
            },
            output: Linear {
                weight: Param::from_tensor(Tensor::from_data(
                    TensorData::new(
                        take(config.hidden_dim * classes),
                        [config.hidden_dim, classes],
                    ),
                    device,
                )),
                bias: Some(Param::from_tensor(Tensor::from_data(
                    TensorData::new(take(classes), [classes]),
                    device,
                ))),
            },
        })
    }
}

impl<B: Backend> WindowModel<B> {
    /// Row-major f32, little-endian: embedding, hidden weight/bias, output weight/bias.
    pub async fn weights(&self) -> Result<Vec<u8>, ModelError> {
        let tensors = [
            self.embedding.weight.val().flatten::<1>(0, 1),
            self.hidden.weight.val().flatten::<1>(0, 1),
            self.hidden
                .bias
                .as_ref()
                .expect("hidden bias is required")
                .val(),
            self.output.weight.val().flatten::<1>(0, 1),
            self.output
                .bias
                .as_ref()
                .expect("output bias is required")
                .val(),
        ];
        let mut bytes = Vec::new();
        for tensor in tensors {
            let data = tensor
                .into_data_async()
                .await
                .map_err(|error| ModelError::Readback(format!("{error:?}")))?;
            for value in data.iter::<f32>() {
                if !value.is_finite() {
                    return Err(ModelError::InvalidWeights);
                }
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(bytes)
    }
}

/// Snaps live weights to the Q4 block grid while preserving parameter IDs.
///
/// Quantization-aware training calls this after each optimizer step so Adam
/// momentum survives. The dequantized values replace the parameter contents
/// through `Param::map`, which keeps each ID stable.
pub async fn project_q4<B: Backend>(
    model: WindowModel<B>,
    config: &ModelConfig,
    device: &B::Device,
) -> Result<WindowModel<B>, ModelError> {
    config.validate()?;
    let flat = model.weights().await?;
    let metadata = ModelMetadata::new(config.clone(), &flat)?;
    let (quantized, bytes) = quantize(&metadata, &flat)?;
    let (_, restored) = quantized.dequantize(&bytes)?;
    let values: Vec<f32> = restored
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    let mut offset = 0;
    let mut take = |count: usize| {
        let slice = values[offset..offset + count].to_vec();
        offset += count;
        slice
    };
    let embedding_dim = config.embedding_dim;
    let hidden_dim = config.hidden_dim;
    let input_dim = config.hidden_input_width();
    let classes = SyntaxClass::ALL.len();
    let embedding_values = take(FEATURE_VOCAB_SIZE * embedding_dim);
    let hidden_weight_values = take(input_dim * hidden_dim);
    let hidden_bias_values = take(hidden_dim);
    let output_weight_values = take(hidden_dim * classes);
    let output_bias_values = take(classes);
    debug_assert_eq!(offset, values.len());
    let WindowModel {
        embedding,
        hidden,
        output,
    } = model;
    let embedding = Embedding {
        weight: embedding.weight.map(|_| {
            Tensor::from_data(
                TensorData::new(embedding_values, [FEATURE_VOCAB_SIZE, embedding_dim]),
                device,
            )
        }),
    };
    let hidden =
        Linear {
            weight: hidden.weight.map(|_| {
                Tensor::from_data(
                    TensorData::new(hidden_weight_values, [input_dim, hidden_dim]),
                    device,
                )
            }),
            bias: Some(hidden.bias.expect("hidden bias is required").map(|_| {
                Tensor::from_data(TensorData::new(hidden_bias_values, [hidden_dim]), device)
            })),
        };
    let output =
        Linear {
            weight: output.weight.map(|_| {
                Tensor::from_data(
                    TensorData::new(output_weight_values, [hidden_dim, classes]),
                    device,
                )
            }),
            bias: Some(output.bias.expect("output bias is required").map(|_| {
                Tensor::from_data(TensorData::new(output_bias_values, [classes]), device)
            })),
        };
    Ok(WindowModel {
        embedding,
        hidden,
        output,
    })
}

/// Runs the exact Q4 artifact quantizer on live weights and returns globals.
///
/// Both training-time fake quantization and epoch validation share this so
/// the simulated grid matches the final exported artifact bit for bit.
pub async fn dequantized_bytes<B: Backend>(
    model: &WindowModel<B>,
    config: &ModelConfig,
) -> Result<(ModelMetadata, Vec<u8>), ModelError> {
    config.validate()?;
    let flat = model.weights().await?;
    let metadata = ModelMetadata::new(config.clone(), &flat)?;
    let (quantized, bytes) = quantize(&metadata, &flat)?;
    quantized.dequantize(&bytes)
}

/// Builds a straight-through fake-quantized copy of `model`.
///
/// Forward evaluates with the globally dequantized Q4 values. The grid sum is
/// a tracked intermediate, so `backward` through the copy accumulates the
/// straight-through gradient on the original float leaves. Collect gradients
/// from the float model, never from the copy.
pub fn fake_quantized<B: Backend>(
    model: &WindowModel<B>,
    config: &ModelConfig,
    dequantized: &[f32],
    device: &B::Device,
) -> Result<WindowModel<B>, ModelError> {
    config.validate()?;
    if dequantized.len() != config.parameter_count() {
        return Err(ModelError::InvalidWeights);
    }
    let mut offset = 0;
    let mut take = |count: usize| {
        let slice = dequantized[offset..offset + count].to_vec();
        offset += count;
        slice
    };
    let embedding_dim = config.embedding_dim;
    let hidden_dim = config.hidden_dim;
    let input_dim = config.hidden_input_width();
    let classes = SyntaxClass::ALL.len();
    let combine = |quantum: Vec<f32>, live: Tensor<B, 1>| {
        Tensor::from_data(TensorData::new(quantum, [live.dims()[0]]), device) + live.clone()
            - live.detach()
    };
    let combine2 = |quantum: Vec<f32>, live: Tensor<B, 2>| {
        let [rows, cols] = live.dims();
        Tensor::from_data(TensorData::new(quantum, [rows, cols]), device) + live.clone()
            - live.detach()
    };
    let embedding = Embedding {
        weight: Param::initialized(
            model.embedding.weight.id,
            combine2(
                take(FEATURE_VOCAB_SIZE * embedding_dim),
                model.embedding.weight.val(),
            ),
        ),
    };
    let hidden = Linear {
        weight: Param::initialized(
            model.hidden.weight.id,
            combine2(take(input_dim * hidden_dim), model.hidden.weight.val()),
        ),
        bias: Some(Param::initialized(
            model
                .hidden
                .bias
                .as_ref()
                .expect("hidden bias is required")
                .id,
            combine(
                take(hidden_dim),
                model
                    .hidden
                    .bias
                    .as_ref()
                    .expect("hidden bias is required")
                    .val(),
            ),
        )),
    };
    let output = Linear {
        weight: Param::initialized(
            model.output.weight.id,
            combine2(take(hidden_dim * classes), model.output.weight.val()),
        ),
        bias: Some(Param::initialized(
            model
                .output
                .bias
                .as_ref()
                .expect("output bias is required")
                .id,
            combine(
                take(classes),
                model
                    .output
                    .bias
                    .as_ref()
                    .expect("output bias is required")
                    .val(),
            ),
        )),
    };
    debug_assert_eq!(offset, dequantized.len());
    Ok(WindowModel {
        embedding,
        hidden,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::quantize;

    #[test]
    fn qat_projection_preserves_ids_and_snaps_to_grid() {
        use burn::backend::NdArray;
        let device = Default::default();
        let config = ModelConfig {
            radius: 0,
            embedding_dim: 2,
            hidden_dim: 1,
            context_state: false,
        };
        let model: WindowModel<NdArray<f32>> = config.init(&device).unwrap();
        let ids_before = [
            model.embedding.weight.id,
            model.hidden.weight.id,
            model.hidden.bias.as_ref().unwrap().id,
            model.output.weight.id,
            model.output.bias.as_ref().unwrap().id,
        ];
        let flat = pollster::block_on(model.weights()).unwrap();
        let projected = pollster::block_on(project_q4(model, &config, &device)).unwrap();
        let ids_after = [
            projected.embedding.weight.id,
            projected.hidden.weight.id,
            projected.hidden.bias.as_ref().unwrap().id,
            projected.output.weight.id,
            projected.output.bias.as_ref().unwrap().id,
        ];
        assert_eq!(ids_before, ids_after);
        let metadata = ModelMetadata::new(config, &flat).unwrap();
        let (quantized, bytes) = quantize(&metadata, &flat).unwrap();
        let (_, expected) = quantized.dequantize(&bytes).unwrap();
        assert_eq!(pollster::block_on(projected.weights()).unwrap(), expected);
    }

    #[test]
    fn fake_quantized_matches_grid_forward_and_keeps_ids() {
        use burn::backend::NdArray;
        let device = Default::default();
        let config = ModelConfig {
            radius: 0,
            embedding_dim: 2,
            hidden_dim: 1,
            context_state: false,
        };
        let model: WindowModel<NdArray<f32>> = config.init(&device).unwrap();
        let (_, dequantized) = pollster::block_on(dequantized_bytes(&model, &config)).unwrap();
        let quantum: Vec<f32> = dequantized
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        let simulated = fake_quantized(&model, &config, &quantum, &device).unwrap();
        assert_eq!(simulated.embedding.weight.id, model.embedding.weight.id);
        let restored = ModelMetadata::new(config.clone(), &dequantized)
            .unwrap()
            .load::<NdArray<f32>>(&dequantized, &device)
            .unwrap();
        let features = vec![1; config.input_width()];
        let input =
            crate::input_tensor::<NdArray<f32>>(features.clone(), 1, &config, &device).unwrap();
        let again = crate::input_tensor::<NdArray<f32>>(features, 1, &config, &device).unwrap();
        let expected: Vec<f32> = restored.forward(input, None).into_data().iter().collect();
        let actual: Vec<f32> = simulated.forward(again, None).into_data().iter().collect();
        // The straight-through sum reconstructs the grid values up to f32
        // rounding of (deq + live) - live, so compare with a tight tolerance.
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() <= 1e-6, "{a} differs from {b}");
        }
    }

    #[test]
    fn ste_gradients_update_float_model() {
        use burn::backend::{Autodiff, NdArray};
        use burn::module::AutodiffModule;
        use burn::nn::loss::CrossEntropyLossConfig;
        use burn::optim::{AdamConfig, GradientsParams, Optimizer};
        use burn::tensor::{Int, Tensor, TensorData};
        type A = Autodiff<NdArray<f32>>;
        let device = Default::default();
        let config = ModelConfig {
            radius: 0,
            embedding_dim: 2,
            hidden_dim: 1,
            context_state: false,
        };
        let mut model: WindowModel<A> = config.init(&device).unwrap();
        let before = pollster::block_on(model.weights()).unwrap();
        let features = vec![1; 4 * config.input_width()];
        let input = crate::input_tensor::<A>(features, 4, &config, &device).unwrap();
        let targets =
            Tensor::<A, 1, Int>::from_data(TensorData::new(vec![0, 1, 2, 3], [4]), &device);
        let (_, dequantized) = pollster::block_on(dequantized_bytes(&model, &config)).unwrap();
        let quantum: Vec<f32> = dequantized
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        let simulated = fake_quantized(&model, &config, &quantum, &device).unwrap();
        let loss_fn = CrossEntropyLossConfig::new().init::<A>(&device);
        let loss = loss_fn.forward(simulated.forward(input, None), targets);
        // The copy is forward-only; its intermediates drain into the float
        // leaves, so gradients are collected from the float model.
        // The copy is forward-only; its intermediates drain into the float
        // leaves, so gradients are collected from the float model.
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        assert_eq!(gradients.len(), 5);
        let mut optimizer = AdamConfig::new().init::<A, WindowModel<A>>();
        model = optimizer.step(0.003, model, gradients);
        let after = pollster::block_on(model.weights()).unwrap();
        assert_ne!(before, after);
        let _ = model.valid();
    }
}
