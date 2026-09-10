use burn::{backend::NdArray, tensor::backend::Backend};
use tint_core::{FEATURE_COUNT, FEATURE_VOCAB_SIZE, SyntaxClass, tokenize, window_features};
use tint_model::{ModelConfig, ModelError, ModelMetadata, highlight, input_tensor, predict};

type Cpu = NdArray<f32>;

#[test]
fn artifact_round_trip_preserves_logits() {
    pollster::block_on(async {
        let device = Default::default();
        Cpu::seed(&device, 17);
        let config = ModelConfig::default();
        let model = config.init::<Cpu>(&device).unwrap();
        let tokens = tokenize("let answer = 42;");
        let features = window_features(&tokens, 0, config.radius);
        let input = input_tensor(features, 1, &config, &device).unwrap();
        let expected = model.forward(input.clone(), None).into_data();
        let weights = model.weights().await.unwrap();
        assert_eq!(weights.len(), config.parameter_count() * 4);
        assert_eq!(config.parameter_count(), 47_233);
        let metadata = ModelMetadata::new(config, &weights).unwrap();
        let encoded = serde_json::to_string(&metadata).unwrap();
        let decoded: ModelMetadata = serde_json::from_str(&encoded).unwrap();
        let restored = decoded.load::<Cpu>(&weights, &device).unwrap();
        assert_eq!(expected, restored.forward(input, None).into_data());
        assert_eq!(weights, restored.weights().await.unwrap());
    });
}

#[test]
fn rejects_corrupt_or_incompatible_artifacts() {
    pollster::block_on(async {
        let config = ModelConfig::default();
        let model = config.init::<Cpu>(&Default::default()).unwrap();
        let weights = model.weights().await.unwrap();
        let metadata = ModelMetadata::new(config.clone(), &weights).unwrap();
        assert!(
            metadata
                .validate_weights(&weights[..weights.len() - 1])
                .is_err()
        );
        let mut corrupt = weights.clone();
        corrupt[0] ^= 1;
        assert!(metadata.validate_weights(&corrupt).is_err());
        corrupt[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(ModelMetadata::new(config.clone(), &corrupt).is_err());
        corrupt[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(ModelMetadata::new(config, &corrupt).is_err());
        let mut incompatible = metadata.clone();
        incompatible.feature_version += 1;
        assert!(incompatible.validate().is_err());
        incompatible = metadata.clone();
        incompatible.classes.swap(0, 1);
        assert!(incompatible.validate().is_err());
        incompatible = metadata;
        incompatible.config.radius = usize::MAX;
        assert!(incompatible.validate().is_err());
    });
}

#[test]
fn input_validation_precedes_tensor_creation() {
    let config = ModelConfig::default();
    let device = Default::default();
    for (features, batch) in [
        (vec![], 0),
        (vec![], 1),
        (vec![-1; config.input_width()], 1),
        (vec![FEATURE_VOCAB_SIZE as i32; config.input_width()], 1),
        (vec![], tint_model::MAX_BATCH_SIZE + 1),
    ] {
        assert!(input_tensor::<Cpu>(features, batch, &config, &device).is_err());
    }
    assert_eq!(config.input_width(), config.window_size() * FEATURE_COUNT);
}

#[test]
fn prediction_is_independent_of_batch_boundaries() {
    pollster::block_on(async {
        let device = Default::default();
        let config = ModelConfig::default();
        let model = config.init::<Cpu>(&device).unwrap();
        let source = "fn main() {\r\n let name = \"café 😀\";\n}\n";
        let tokens = tokenize(source);
        let one = predict(&model, &config, &tokens, None, 1, &device)
            .await
            .unwrap();
        let many = predict(&model, &config, &tokens, None, 128, &device)
            .await
            .unwrap();
        assert_eq!(one, many);
        for (token, class) in tokens.iter().zip(&one) {
            if token.is_whitespace() {
                assert_eq!(*class, SyntaxClass::Plain);
            }
        }
        let spans = highlight(&model, &config, source, 7, &device)
            .await
            .unwrap();
        let text: String = spans
            .iter()
            .map(|span| &source[span.start..span.end])
            .collect();
        assert_eq!(text, source);
        assert!(
            highlight(&model, &config, "", 1, &device)
                .await
                .unwrap()
                .is_empty()
        );
    });
}

#[test]
fn rejects_configuration_that_does_not_match_the_model() {
    pollster::block_on(async {
        let device = Default::default();
        let config = ModelConfig::default();
        let model = config.init::<Cpu>(&device).unwrap();
        let other = ModelConfig {
            radius: 1,
            ..config
        };
        assert!(matches!(
            predict(&model, &other, &tokenize("let x"), None, 2, &device).await,
            Err(ModelError::InvalidConfig)
        ));
    });
}

#[test]
fn version_two_state_path_counts_params_and_validates() {
    use tint_core::token_states;
    pollster::block_on(async {
        let device = Default::default();
        Cpu::seed(&device, 7);
        let config = ModelConfig {
            radius: 4,
            embedding_dim: 6,
            hidden_dim: 64,
            context_state: true,
        };
        assert_eq!(config.hidden_input_width(), 9 * 6 + 9);
        assert_eq!(config.parameter_count(), 8190 + 63 * 64 + 64 + 576 + 9);
        let model = config.init::<Cpu>(&device).unwrap();
        let weights = model.weights().await.unwrap();
        let metadata = ModelMetadata::new(config.clone(), &weights).unwrap();
        assert_eq!(metadata.feature_version, 2);
        assert_eq!(metadata.architecture, "window-mlp-v2");
        let encoded = serde_json::to_string(&metadata).unwrap();
        assert!(encoded.contains("context_state"));
        let plain = "/* gamma */ \"gamma\" gamma";
        let tokens = tokenize(plain);
        let states = token_states(plain, &tokens);
        let some = states.as_slice();
        let with = predict(&model, &config, &tokens, Some(some), 32, &device)
            .await
            .unwrap();
        assert_eq!(with.len(), tokens.len());
        assert!(
            predict(&model, &config, &tokens, None, 32, &device)
                .await
                .is_err()
        );
        let v1 = ModelConfig::default();
        assert!(
            predict(&model, &v1, &tokens, Some(some), 32, &device)
                .await
                .is_err()
        );
        let legacy = serde_json::json!({"radius": 4, "embedding_dim": 8, "hidden_dim": 64});
        let parsed: ModelConfig = serde_json::from_value(legacy).unwrap();
        assert!(!parsed.context_state);
    });
}
