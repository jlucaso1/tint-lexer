use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

use serde_json::{Value, json};

fn tint(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tint"))
        .args(args)
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

fn document(path: &Path, id: &str, group: &str, source: &str) {
    let doc = json!({ "version": 1, "id": id, "group": group, "language": "rust", "source": source, "spans": [{ "start": 0, "end": source.len(), "class": "keyword" }] });
    fs::write(path, serde_json::to_vec(&doc).unwrap()).unwrap();
}

#[test]
fn train_reload_evaluate_and_highlight() {
    let dir = tempfile::tempdir().unwrap();
    let train = dir.path().join("train.jsonl");
    let validation = dir.path().join("validation.jsonl");
    let test = dir.path().join("test.jsonl");
    let model = dir.path().join("model");
    let input = dir.path().join("source.txt");
    document(&train, "train", "train-group", "alpha beta");
    document(&validation, "validation", "validation-group", "gamma delta");
    document(&test, "test", "test-group", "epsilon zeta");
    let args = [
        "train",
        "--train",
        train.to_str().unwrap(),
        "--validation",
        validation.to_str().unwrap(),
        "--output",
        model.to_str().unwrap(),
        "--epochs",
        "2",
        "--batch-size",
        "2",
        "--radius",
        "0",
        "--embedding-dim",
        "2",
        "--hidden-dim",
        "4",
    ];
    let output = tint(&args);
    assert!(String::from_utf8_lossy(&output.stderr).contains("epoch 2/2"));
    let report = success(output);
    assert_eq!(report["history"].as_array().unwrap().len(), 2);
    assert_eq!(
        report["best_train_metrics"]["overall"]["evaluated_tokens"],
        2
    );
    assert_eq!(report["backend"], "cpu");
    let disk: Value =
        serde_json::from_slice(&fs::read(model.join("report.json")).unwrap()).unwrap();
    assert_eq!(report, disk);
    let metadata: Value =
        serde_json::from_slice(&fs::read(model.join("model.json")).unwrap()).unwrap();
    assert_eq!(metadata["weights_sha256"], report["weights_sha256"]);
    assert!(!tint(&args).status.success());

    let repeated = dir.path().join("repeated-model");
    let mut repeated_args = args;
    repeated_args[6] = repeated.to_str().unwrap();
    let repeated_report = success(tint(&repeated_args));
    assert_eq!(repeated_report, report);
    assert_eq!(
        fs::read(repeated.join("model.bin")).unwrap(),
        fs::read(model.join("model.bin")).unwrap()
    );

    let model_arg = model.to_str().unwrap();
    let evaluated = success(tint(&[
        "evaluate",
        "--data",
        test.to_str().unwrap(),
        "--model",
        model_arg,
    ]));
    assert_eq!(evaluated["leakage_check"], "checked_against_training");
    assert_eq!(evaluated["validation_overlap"], false);
    let evaluated = success(tint(&[
        "evaluate",
        "--data",
        validation.to_str().unwrap(),
        "--model",
        model_arg,
    ]));
    assert_eq!(evaluated["validation_overlap"], true);
    assert_eq!(evaluated["metrics"], report["best_validation_metrics"]);
    let leaked = tint(&[
        "evaluate",
        "--data",
        train.to_str().unwrap(),
        "--model",
        model_arg,
    ]);
    assert!(!leaked.status.success());
    assert!(String::from_utf8_lossy(&leaked.stderr).contains("overlaps training"));

    let report_path = model.join("report.json");
    let report_bytes = fs::read(&report_path).unwrap();
    let mut mismatched_report = report.clone();
    mismatched_report["weights_sha256"] = json!("0".repeat(64));
    fs::write(
        &report_path,
        serde_json::to_vec(&mismatched_report).unwrap(),
    )
    .unwrap();
    let output = tint(&[
        "evaluate",
        "--data",
        test.to_str().unwrap(),
        "--model",
        model_arg,
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("report does not match"));
    fs::remove_file(&report_path).unwrap();
    let output = tint(&[
        "evaluate",
        "--data",
        test.to_str().unwrap(),
        "--model",
        model_arg,
    ]);
    assert!(String::from_utf8_lossy(&output.stderr).contains("leakage cannot be checked"));
    let unchecked = success(output);
    assert_eq!(unchecked["leakage_check"], "unavailable");
    assert!(unchecked["validation_overlap"].is_null());
    fs::write(&report_path, report_bytes).unwrap();

    let source = "a\u{1f600} b\n";
    fs::write(&input, source).unwrap();
    for utf16 in [false, true] {
        let mut args = vec![
            "highlight",
            "--model",
            model_arg,
            "--input",
            input.to_str().unwrap(),
        ];
        if utf16 {
            args.push("--utf16");
        }
        let highlighted = success(tint(&args));
        assert_eq!(
            highlighted["offset_encoding"],
            if utf16 { "utf16" } else { "utf8" }
        );
        let spans = highlighted["spans"].as_array().unwrap();
        assert_eq!(spans[0]["start"], 0);
        assert_eq!(
            spans.last().unwrap()["end"],
            if utf16 {
                source.encode_utf16().count()
            } else {
                source.len()
            }
        );
        for pair in spans.windows(2) {
            assert_eq!(pair[0]["end"], pair[1]["start"]);
        }
    }
    fs::write(&input, "").unwrap();
    assert_eq!(
        success(tint(&[
            "highlight",
            "--model",
            model_arg,
            "--input",
            input.to_str().unwrap()
        ]))["spans"],
        json!([])
    );
    let mut weights = fs::read(model.join("model.bin")).unwrap();
    weights[0] ^= 1;
    fs::write(model.join("model.bin"), weights).unwrap();
    assert!(
        !tint(&[
            "highlight",
            "--model",
            model_arg,
            "--input",
            input.to_str().unwrap()
        ])
        .status
        .success()
    );
}

#[test]
fn reject_bad_arguments() {
    for (flag, value) in [
        ("--epochs", "0"),
        ("--epochs", "1001"),
        ("--batch-size", "0"),
        ("--batch-size", "4097"),
        ("--learning-rate", "NaN"),
        ("--learning-rate", "inf"),
        ("--learning-rate", "0"),
        ("--learning-rate", "1e-100"),
        ("--backend", "cuda"),
    ] {
        let output = tint(&[
            "train",
            "--train",
            "unused",
            "--validation",
            "unused",
            "--output",
            "unused",
            flag,
            value,
        ]);
        assert_eq!(output.status.code(), Some(2), "{flag}={value}");
        assert!(output.stdout.is_empty());
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let output = tint(&[
            "highlight",
            "--model",
            "unused",
            "--input",
            "unused",
            "--backend",
            "wgpu",
        ]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("--features wgpu"));
    }
}

#[test]
fn reject_split_leakage_and_report_line_context() {
    let dir = tempfile::tempdir().unwrap();
    let train = dir.path().join("train.jsonl");
    let validation = dir.path().join("validation.jsonl");
    let model = dir.path().join("model");
    document(&train, "train", "group", "alpha");
    let args = [
        "train",
        "--train",
        train.to_str().unwrap(),
        "--validation",
        validation.to_str().unwrap(),
        "--output",
        model.to_str().unwrap(),
        "--epochs",
        "1",
    ];
    for (id, group, source) in [
        ("train", "different", "beta"),
        ("validation", "group", "beta"),
        ("validation", "different", "alpha"),
    ] {
        document(&validation, id, group, source);
        let output = tint(&args);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("split leakage"));
        assert!(!model.exists());
    }
    fs::write(&validation, "{\"version\": 1, \"unknown\": true}\n").unwrap();
    let output = tint(&args);
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("validation.jsonl"));
    assert!(error.contains("line 1"));
    assert!(!model.exists());
}
