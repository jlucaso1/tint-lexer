#![cfg(feature = "wgpu")]

use burn::{
    backend::{NdArray, Wgpu},
    tensor::backend::Backend,
};
use tint_core::{tokenize, window_features};
use tint_model::{ModelConfig, ModelMetadata, input_tensor, predict};

#[test]
#[ignore = "requires a working WGPU adapter"]
fn wgpu_matches_cpu_logits_and_classes() {
    pollster::block_on(async {
        type Cpu = NdArray<f32>;
        let cpu_device = Default::default();
        let gpu_device = Default::default();
        Cpu::seed(&cpu_device, 37);
        let config = ModelConfig::default();
        let cpu = config.init::<Cpu>(&cpu_device).unwrap();
        let weights = cpu.weights().await.unwrap();
        let metadata = ModelMetadata::new(config.clone(), &weights).unwrap();
        let gpu = metadata.load::<Wgpu>(&weights, &gpu_device).unwrap();
        let tokens = tokenize("fn greet(name: &str) { println!(\"hello {name}\"); }\r\n");
        let features: Vec<_> = (0..tokens.len())
            .flat_map(|index| window_features(&tokens, index, config.radius))
            .collect();
        let cpu_input = input_tensor(features.clone(), tokens.len(), &config, &cpu_device).unwrap();
        let gpu_input = input_tensor(features, tokens.len(), &config, &gpu_device).unwrap();
        let expected = cpu
            .forward(cpu_input, None)
            .into_data_async()
            .await
            .unwrap();
        let actual = gpu
            .forward(gpu_input, None)
            .into_data_async()
            .await
            .unwrap();
        let difference = expected
            .iter::<f32>()
            .zip(actual.iter::<f32>())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(difference < 1e-4, "maximum logit error {difference}");
        assert_eq!(
            predict(&cpu, &config, &tokens, None, 1, &cpu_device)
                .await
                .unwrap(),
            predict(&gpu, &config, &tokens, None, 7, &gpu_device)
                .await
                .unwrap()
        );
        assert_eq!(weights, gpu.weights().await.unwrap());
    });
}
