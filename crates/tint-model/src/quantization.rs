//! Q4 preserves the float artifact's parameter order across consecutive blocks of 64.
//! Each block stores an f32 LE scale followed by low-nibble-first signed codes plus 8.
//! The final unused high nibble is zero. Source checksums record provenance, not authenticity.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tint_core::SyntaxClass;

use crate::{MAX_WEIGHTS_BYTES, ModelConfig, ModelError, ModelMetadata};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizedMetadata {
    pub format_version: u32,
    pub feature_version: u32,
    pub architecture: String,
    pub config: ModelConfig,
    pub classes: Vec<SyntaxClass>,
    pub quantization: String,
    pub weights_sha256: String,
    pub source_weights_sha256: String,
}

impl QuantizedMetadata {
    pub fn validate(&self) -> Result<(), ModelError> {
        ModelMetadata {
            format_version: self.format_version,
            feature_version: self.feature_version,
            architecture: self.architecture.clone(),
            config: self.config.clone(),
            classes: self.classes.clone(),
            weights_sha256: self.weights_sha256.clone(),
        }
        .validate()?;
        if self.quantization != "q4-block64-v1"
            || self.config.parameter_count() > MAX_WEIGHTS_BYTES / 4
            || self.source_weights_sha256.len() != 64
            || !self
                .source_weights_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(ModelError::InvalidMetadata);
        }
        Ok(())
    }

    pub fn dequantize(&self, bytes: &[u8]) -> Result<(ModelMetadata, Vec<u8>), ModelError> {
        self.validate()?;
        let count = self.config.parameter_count();
        if bytes.len() != count.div_ceil(64) * 4 + count.div_ceil(2)
            || format!("{:x}", Sha256::digest(bytes)) != self.weights_sha256
        {
            return Err(ModelError::InvalidWeights);
        }
        let mut weights = Vec::with_capacity(count * 4);
        let mut offset = 0;
        for start in (0..count).step_by(64) {
            let n = (count - start).min(64);
            let scale = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += 4;
            if !scale.is_finite() || scale <= 0.0 {
                return Err(ModelError::InvalidWeights);
            }
            for i in 0..n {
                let code = (bytes[offset + i / 2] >> (4 * (i % 2))) & 15;
                if code == 0 {
                    return Err(ModelError::InvalidWeights);
                }
                let value = (i32::from(code) - 8) as f32 * scale;
                if !value.is_finite() {
                    return Err(ModelError::InvalidWeights);
                }
                weights.extend_from_slice(&value.to_le_bytes());
            }
            if !n.is_multiple_of(2) && bytes[offset + n / 2] >> 4 != 0 {
                return Err(ModelError::InvalidWeights);
            }
            offset += n.div_ceil(2);
        }
        Ok((ModelMetadata::new(self.config.clone(), &weights)?, weights))
    }
}

/// Uses maxabs / 7, clamped to MIN_POSITIVE, or 1 for an all-zero block.
/// Division and rounding use f32. Halfway values round away from zero, then
/// clamp to -7..=7. Rejects any reconstruction that would overflow f32.
pub fn quantize(
    metadata: &ModelMetadata,
    weights: &[u8],
) -> Result<(QuantizedMetadata, Vec<u8>), ModelError> {
    metadata.validate_weights(weights)?;
    let count = metadata.config.parameter_count();
    let mut bytes = Vec::with_capacity(count.div_ceil(64) * 4 + count.div_ceil(2));
    for block in weights.as_chunks::<4>().0.chunks(64) {
        let maxabs = block.iter().fold(0.0_f32, |max, value| {
            max.max(f32::from_le_bytes(*value).abs())
        });
        let scale = if maxabs == 0.0 {
            1.0
        } else {
            (maxabs / 7.0).max(f32::MIN_POSITIVE)
        };
        bytes.extend_from_slice(&scale.to_le_bytes());
        for (i, value) in block.iter().enumerate() {
            let q = (f32::from_le_bytes(*value) / scale)
                .round()
                .clamp(-7.0, 7.0);
            if !(q * scale).is_finite() {
                return Err(ModelError::InvalidWeights);
            }
            let code = (q as i8 + 8) as u8;
            if i.is_multiple_of(2) {
                bytes.push(code);
            } else {
                *bytes.last_mut().unwrap() |= code << 4;
            }
        }
    }
    let quantized = QuantizedMetadata {
        format_version: metadata.format_version,
        feature_version: metadata.feature_version,
        architecture: metadata.architecture.clone(),
        config: metadata.config.clone(),
        classes: metadata.classes.clone(),
        quantization: "q4-block64-v1".to_owned(),
        weights_sha256: format!("{:x}", Sha256::digest(&bytes)),
        source_weights_sha256: metadata.weights_sha256.clone(),
    };
    quantized.validate()?;
    Ok((quantized, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(values: &[f32], embedding_dim: usize) -> (ModelMetadata, Vec<u8>) {
        let config = ModelConfig {
            radius: 0,
            embedding_dim,
            hidden_dim: 1,
            context_state: false,
        };
        let weights = (0..config.parameter_count())
            .flat_map(|i| values[i % values.len()].to_le_bytes())
            .collect::<Vec<_>>();
        (ModelMetadata::new(config, &weights).unwrap(), weights)
    }

    fn rehash(metadata: &mut QuantizedMetadata, bytes: &[u8]) {
        metadata.weights_sha256 = format!("{:x}", Sha256::digest(bytes));
    }

    #[test]
    fn roundtrip_and_error_bounds() {
        for values in [
            vec![0.0, -0.0],
            (-7..=7).map(|x| x as f32).collect(),
            vec![-7.0, -0.5, 0.5, 7.0, 0.123, -2.73],
            vec![f32::from_bits(1), -f32::MIN_POSITIVE, 0.0],
            vec![f32::MAX / 2.0, -f32::MAX / 2.0],
        ] {
            for dim in [1, 2, 64] {
                let (metadata, weights) = artifact(&values, dim);
                let (qmeta, bytes) = quantize(&metadata, &weights).unwrap();
                let (restored, decoded) = qmeta.dequantize(&bytes).unwrap();
                restored.validate_weights(&decoded).unwrap();
                assert_eq!(qmeta.source_weights_sha256, metadata.weights_sha256);
                assert_eq!(restored.config, metadata.config);
                assert_eq!(restored.classes, metadata.classes);
                assert_eq!(weights.len(), decoded.len());
                for (i, (a, b)) in weights
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(decoded.as_chunks::<4>().0)
                    .enumerate()
                {
                    let offset = i / 64 * 36;
                    let scale = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                    let error = (f64::from(f32::from_le_bytes(*a))
                        - f64::from(f32::from_le_bytes(*b)))
                    .abs();
                    assert!(error <= f64::from(scale) * 0.501);
                }
            }
        }
    }

    #[test]
    fn exact_codes_rounding_and_block_boundary() {
        let (metadata, weights) = artifact(&[-7.0, 7.0, -0.5, 0.5], 2);
        let (qmeta, bytes) = quantize(&metadata, &weights).unwrap();
        assert_eq!(&bytes[..6], &[0, 0, 128, 63, 0xf1, 0x97]);
        assert_eq!(&bytes[36..42], &bytes[..6]);
        assert_eq!(metadata.config.parameter_count() % 2, 1);
        assert_eq!(bytes.last().unwrap() >> 4, 0);
        let (restored, _) = qmeta.dequantize(&bytes).unwrap();
        assert_ne!(restored.weights_sha256, qmeta.source_weights_sha256);
        let (metadata, weights) = artifact(&[0.0], 1);
        let (_, bytes) = quantize(&metadata, &weights).unwrap();
        assert_eq!(&bytes[..5], &[0, 0, 128, 63, 0x88]);
    }

    #[test]
    fn even_and_full_final_blocks_with_extreme_values() {
        for dim in [1, 21] {
            let config = ModelConfig {
                radius: 0,
                embedding_dim: dim,
                hidden_dim: 2,
                context_state: false,
            };
            let count = config.parameter_count();
            assert_eq!(count % 2, 0);
            if dim == 21 {
                assert_eq!(count % 64, 0);
            }
            let weights: Vec<_> = (0..count)
                .flat_map(|i| {
                    (if i.is_multiple_of(2) {
                        f32::MAX
                    } else {
                        -f32::MAX
                    })
                    .to_le_bytes()
                })
                .collect();
            let metadata = ModelMetadata::new(config, &weights).unwrap();
            let (qmeta, bytes) = quantize(&metadata, &weights).unwrap();
            assert_eq!(bytes.len(), count.div_ceil(64) * 4 + count / 2);
            assert_eq!(*bytes.last().unwrap(), 0x1f);
            let (restored, decoded) = qmeta.dequantize(&bytes).unwrap();
            restored.validate_weights(&decoded).unwrap();
            assert_eq!(decoded, weights);
        }
    }

    #[test]
    fn corrupt_weights_and_scales() {
        let (metadata, weights) = artifact(&[7.0], 2);
        let (qmeta, bytes) = quantize(&metadata, &weights).unwrap();
        let mut corrupt = bytes.clone();
        corrupt[4] ^= 1;
        assert!(qmeta.dequantize(&corrupt).is_err());
        for scale in [
            0.0,
            -0.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
        ] {
            let mut corrupt = bytes.clone();
            corrupt[..4].copy_from_slice(&scale.to_le_bytes());
            let mut meta = qmeta.clone();
            rehash(&mut meta, &corrupt);
            assert!(meta.dequantize(&corrupt).is_err());
        }
        for kind in 0..5 {
            let mut corrupt = bytes.clone();
            match kind {
                0 => corrupt[4] &= 0xf0,
                1 => corrupt[4] &= 0x0f,
                2 => *corrupt.last_mut().unwrap() |= 0x10,
                3 => {
                    corrupt.pop();
                }
                _ => corrupt.push(0),
            }
            let mut meta = qmeta.clone();
            rehash(&mut meta, &corrupt);
            assert!(meta.dequantize(&corrupt).is_err());
        }
    }

    #[test]
    fn malformed_metadata_and_source() {
        let (metadata, weights) = artifact(&[1.0], 1);
        let (qmeta, bytes) = quantize(&metadata, &weights).unwrap();
        for field in [
            "format_version",
            "feature_version",
            "architecture",
            "config",
            "classes",
            "quantization",
            "weights_sha256",
            "source_weights_sha256",
        ] {
            let mut json = serde_json::to_value(&qmeta).unwrap();
            json[field] = match field {
                "format_version" | "feature_version" => serde_json::json!(2),
                "config" => {
                    serde_json::json!({"radius": usize::MAX, "embedding_dim": 1, "hidden_dim": 1})
                }
                "classes" => serde_json::json!([]),
                _ => serde_json::json!("invalid"),
            };
            let invalid: QuantizedMetadata = serde_json::from_value(json).unwrap();
            assert!(invalid.validate().is_err());
            assert!(invalid.dequantize(&bytes).is_err());
        }
        let mut json = serde_json::to_value(&qmeta).unwrap();
        json["extra"] = serde_json::json!(0);
        assert!(serde_json::from_value::<QuantizedMetadata>(json).is_err());
        let mut capped = qmeta.clone();
        capped.config = ModelConfig {
            radius: 16,
            embedding_dim: 128,
            hidden_dim: 256,
            context_state: false,
        };
        assert!(capped.config.parameter_count() * 4 <= MAX_WEIGHTS_BYTES);
        assert!(capped.validate().is_ok());
        let mut provenance = qmeta.clone();
        provenance.source_weights_sha256 = "0".repeat(64);
        assert!(provenance.dequantize(&bytes).is_ok());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid = weights.clone();
            invalid[..4].copy_from_slice(&value.to_le_bytes());
            let mut meta = metadata.clone();
            meta.weights_sha256 = format!("{:x}", Sha256::digest(&invalid));
            assert!(quantize(&meta, &invalid).is_err());
        }
        assert!(quantize(&metadata, &weights[..weights.len() - 1]).is_err());
        let mut invalid = metadata;
        invalid.classes.reverse();
        assert!(quantize(&invalid, &weights).is_err());
    }

    #[test]
    fn forward_class_agreement_report() {
        use burn::backend::NdArray;
        let (metadata, weights) = artifact(&[-0.7, 0.13, 0.4, -0.21], 2);
        let (qmeta, bytes) = quantize(&metadata, &weights).unwrap();
        let (decoded_meta, decoded) = qmeta.dequantize(&bytes).unwrap();
        let device = Default::default();
        let original = metadata.load::<NdArray<f32>>(&weights, &device).unwrap();
        let restored = decoded_meta
            .load::<NdArray<f32>>(&decoded, &device)
            .unwrap();
        let tokens = tint_core::tokenize("fn main() { let answer = 42; println!(\"hello\"); }");
        let a = pollster::block_on(crate::predict(
            &original,
            &metadata.config,
            &tokens,
            None,
            32,
            &device,
        ))
        .unwrap();
        let b = pollster::block_on(crate::predict(
            &restored,
            &metadata.config,
            &tokens,
            None,
            32,
            &device,
        ))
        .unwrap();
        assert_eq!(a.len(), b.len());
        let compared = tokens.iter().filter(|token| !token.is_whitespace()).count();
        let agreed = tokens
            .iter()
            .zip(a.iter().zip(&b))
            .filter(|(token, (a, b))| !token.is_whitespace() && a == b)
            .count();
        eprintln!("synthetic forward class agreement {agreed}/{compared}");
    }
}
