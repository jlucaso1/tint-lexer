#![cfg(target_arch = "wasm32")]

use burn::backend::wgpu::{Wgpu, WgpuDevice, graphics::AutoGraphicsApi, init_setup_async};
use tint_core::{MAX_SOURCE_BYTES, utf16_spans};
use tint_model::{MAX_WEIGHTS_BYTES, ModelConfig, ModelMetadata, WindowModel};
use wasm_bindgen::prelude::*;

type Backend = Wgpu<f32, i32>;

#[wasm_bindgen]
pub struct Highlighter {
    model: WindowModel<Backend>,
    config: ModelConfig,
    device: WgpuDevice,
}

#[wasm_bindgen]
impl Highlighter {
    #[wasm_bindgen]
    pub async fn load(metadata_json: String, weights: Vec<u8>) -> Result<Highlighter, JsValue> {
        if metadata_json.len() >= 16 * 1024 {
            return Err(JsValue::from_str(
                "Model metadata must be smaller than 16 KiB.",
            ));
        }
        if weights.len() > MAX_WEIGHTS_BYTES {
            return Err(JsValue::from_str("Model weights exceed 8 MiB."));
        }
        let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
            .map_err(|error| JsValue::from_str(&format!("Invalid model metadata: {error}")))?;
        metadata.validate_weights(&weights).map_err(js_error)?;
        let device = WgpuDevice::default();
        init_setup_async::<AutoGraphicsApi>(&device, Default::default()).await;
        let model = metadata.load(&weights, &device).map_err(js_error)?;
        Ok(Self {
            model,
            config: metadata.config,
            device,
        })
    }

    #[wasm_bindgen]
    pub async fn highlight(&self, source: String) -> Result<JsValue, JsValue> {
        if source.len() > MAX_SOURCE_BYTES {
            return Err(JsValue::from_str("Source exceeds 4 MiB."));
        }
        let spans = tint_model::highlight(&self.model, &self.config, &source, 256, &self.device)
            .await
            .map_err(js_error)?;
        let spans = utf16_spans(&source, &spans).map_err(js_error)?;
        serde_wasm_bindgen::to_value(&spans).map_err(js_error)
    }
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
