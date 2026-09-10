#[cfg(unix)]
#[test]
fn rejects_fifo_without_waiting_for_a_writer() {
    use std::{
        process::Command,
        thread,
        time::{Duration, Instant},
    };

    let directory = tempfile::tempdir().unwrap();
    let fifo = directory.path().join("source");
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    let mut child = Command::new(env!("CARGO_BIN_EXE_tint"))
        .args(["highlight", "--model"])
        .arg(directory.path())
        .arg("--input")
        .arg(fifo)
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(!status.success());
            let output = child.wait_with_output().unwrap();
            assert!(String::from_utf8_lossy(&output.stderr).contains("not a regular file"));
            break;
        }
        if start.elapsed() > Duration::from_secs(2) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("opening a FIFO blocked before regular-file validation");
        }
        thread::sleep(Duration::from_millis(10));
    }
}
