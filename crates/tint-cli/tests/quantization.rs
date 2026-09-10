use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

use serde_json::Value;
use tint_model::{ModelConfig, ModelMetadata, quantization::QuantizedMetadata};

fn convert(command: &str, model: &Path, output: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tint"))
        .arg(command)
        .arg("--model")
        .arg(model)
        .arg("--output")
        .arg(output)
        .env("WGPU_BACKEND", "invalid")
        .output()
        .unwrap()
}

fn success(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn export_restore_evaluate_and_refuse_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("float");
    let q4 = dir.path().join("q4");
    let restored = dir.path().join("restored");
    fs::create_dir(&source).unwrap();
    let config = ModelConfig {
        radius: 0,
        embedding_dim: 2,
        hidden_dim: 2,
        context_state: false,
    };
    let weights: Vec<_> = (0..config.parameter_count())
        .flat_map(|i| ((i % 31) as f32 / 17.0 - 0.7).to_le_bytes())
        .collect();
    let metadata = ModelMetadata::new(config, &weights).unwrap();
    let original_json = serde_json::to_vec(&metadata).unwrap();
    fs::write(source.join("model.json"), &original_json).unwrap();
    fs::write(source.join("model.bin"), &weights).unwrap();
    fs::write(source.join("report.json"), b"do not copy").unwrap();

    let summary = success(convert("quantize", &source, &q4));
    let disk: Value =
        serde_json::from_slice(&fs::read(q4.join("quantization.json")).unwrap()).unwrap();
    assert_eq!(summary, disk);
    assert_eq!(summary.as_object().unwrap().len(), 7);
    assert_eq!(summary["original_bytes"], weights.len());
    assert_eq!(
        summary["parameter_count"],
        metadata.config.parameter_count()
    );
    assert_eq!(summary["source_weights_sha256"], metadata.weights_sha256);
    let qmeta: QuantizedMetadata =
        serde_json::from_slice(&fs::read(q4.join("model.q4.json")).unwrap()).unwrap();
    let qbytes = fs::read(q4.join("model.q4.bin")).unwrap();
    assert_eq!(summary["quantized_bytes"], qbytes.len());
    assert_eq!(summary["weights_sha256"], qmeta.weights_sha256);
    let (expected_meta, expected_weights) = qmeta.dequantize(&qbytes).unwrap();
    let errors: Vec<_> = weights
        .as_chunks::<4>()
        .0
        .iter()
        .zip(expected_weights.as_chunks::<4>().0)
        .map(|(a, b)| (f64::from(f32::from_le_bytes(*a)) - f64::from(f32::from_le_bytes(*b))).abs())
        .collect();
    assert_eq!(
        summary["max_absolute_error"].as_f64().unwrap(),
        errors.iter().copied().fold(0.0_f64, f64::max)
    );
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    assert!((summary["rmse"].as_f64().unwrap() - rmse).abs() <= f64::EPSILON * rmse);
    let restored_summary = success(convert("dequantize", &q4, &restored));
    let restored_meta: ModelMetadata =
        serde_json::from_slice(&fs::read(restored.join("model.json")).unwrap()).unwrap();
    let restored_weights = fs::read(restored.join("model.bin")).unwrap();
    restored_meta.validate_weights(&restored_weights).unwrap();
    assert_eq!(restored_weights, expected_weights);
    assert_eq!(restored_meta.weights_sha256, expected_meta.weights_sha256);
    assert_ne!(restored_meta.weights_sha256, metadata.weights_sha256);
    assert_eq!(
        restored_summary["weights_sha256"],
        restored_meta.weights_sha256
    );
    assert_eq!(
        restored_summary["source_weights_sha256"],
        metadata.weights_sha256
    );
    assert_eq!(
        restored_summary["quantized_weights_sha256"],
        qmeta.weights_sha256
    );
    assert_eq!(fs::read_dir(&restored).unwrap().count(), 2);
    assert_eq!(fs::read_dir(&q4).unwrap().count(), 3);
    for (command, input, output) in [("quantize", &source, &q4), ("dequantize", &q4, &restored)] {
        assert!(!convert(command, input, output).status.success());
        assert!(!convert(command, input, input).status.success());
        let existing_file = dir.path().join(format!("{command}.file"));
        fs::write(&existing_file, b"keep").unwrap();
        assert!(!convert(command, input, &existing_file).status.success());
        assert_eq!(fs::read(existing_file).unwrap(), b"keep");
        assert!(
            !convert(command, input, &dir.path().join("missing/child"))
                .status
                .success()
        );
    }
    assert_eq!(fs::read(source.join("model.json")).unwrap(), original_json);
    assert_eq!(fs::read(source.join("model.bin")).unwrap(), weights);
    assert_eq!(fs::read(q4.join("model.q4.bin")).unwrap(), qbytes);
    assert_eq!(
        fs::read(restored.join("model.bin")).unwrap(),
        restored_weights
    );

    let data = dir.path().join("test.jsonl");
    fs::write(&data, br#"{"version":1,"id":"test","group":"test","language":"rust","source":"let","spans":[{"start":0,"end":3,"class":"keyword"}]}"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_tint"))
        .arg("evaluate")
        .arg("--data")
        .arg(data)
        .arg("--model")
        .arg(&restored)
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&output.stderr).contains("report.json is absent"));
    assert_eq!(success(output)["leakage_check"], "unavailable");

    let bad_output = dir.path().join("bad-output");
    fs::write(q4.join("model.q4.bin"), b"invalid").unwrap();
    assert!(!convert("dequantize", &q4, &bad_output).status.success());
    assert!(!bad_output.exists());
    fs::remove_file(source.join("model.bin")).unwrap();
    fs::create_dir(source.join("model.bin")).unwrap();
    assert!(!convert("quantize", &source, &bad_output).status.success());
    assert!(!bad_output.exists());
}

#[cfg(unix)]
#[test]
fn conversion_rejects_fifos_and_existing_symlinks() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");
    fs::create_dir(&source).unwrap();
    let config = ModelConfig {
        radius: 0,
        embedding_dim: 1,
        hidden_dim: 1,
        context_state: false,
    };
    let weights = vec![0; config.parameter_count() * 4];
    let metadata = ModelMetadata::new(config, &weights).unwrap();
    let (qmeta, qbytes) = tint_model::quantization::quantize(&metadata, &weights).unwrap();
    fs::write(
        source.join("model.json"),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
    fs::write(
        source.join("model.q4.json"),
        serde_json::to_vec(&qmeta).unwrap(),
    )
    .unwrap();
    for (command, name, bytes) in [
        ("quantize", "model.bin", weights),
        ("dequantize", "model.q4.bin", qbytes),
    ] {
        let path = source.join(name);
        assert!(
            Command::new("mkfifo")
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        let output = dir.path().join(command);
        assert!(!convert(command, &source, &output).status.success());
        assert!(!output.exists());
        fs::remove_file(&path).unwrap();
        fs::write(&path, bytes).unwrap();
        symlink(dir.path().join("absent"), &output).unwrap();
        assert!(!convert(command, &source, &output).status.success());
        assert!(
            fs::symlink_metadata(&output)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }
}
