use std::{collections::HashSet, fs::OpenOptions, io::Read, path::Path};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tint_core::{Document, SyntaxClass, Token, align_labels, token_context, tokenize};

pub const MAX_DATA_BYTES: usize = 64 * 1024 * 1024;
const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_TOTAL_TOKENS: usize = 500_000;

pub fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    ensure!(
        std::fs::metadata(path)
            .with_context(|| format!("reading metadata for {}", path.display()))?
            .is_file(),
        "{} is not a regular file",
        path.display()
    );
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // A file replaced by a FIFO between metadata and open must not block.
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    ensure!(
        file.metadata()?.is_file(),
        "{} is not a regular file",
        path.display()
    );
    ensure!(
        file.metadata()?.len() <= limit as u64,
        "{} exceeds {limit} bytes",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= limit,
        "{} exceeds {limit} bytes",
        path.display()
    );
    Ok(bytes)
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fingerprint {
    pub id: String,
    pub group: String,
    pub source_sha256: String,
}

pub struct Example {
    pub language: String,
    pub tokens: Vec<Token>,
    pub states: Vec<u8>,
    pub labels: Vec<Option<SyntaxClass>>,
}

pub struct Dataset {
    pub documents: Vec<Example>,
    pub fingerprints: Vec<Fingerprint>,
    pub sha256: String,
    pub token_count: usize,
}

impl Dataset {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = read_bounded(path, MAX_DATA_BYTES)?;
        Self::parse(&bytes).with_context(|| path.display().to_string())
    }

    fn parse(bytes: &[u8]) -> Result<Self> {
        let mut dataset = Self {
            documents: Vec::new(),
            fingerprints: Vec::new(),
            sha256: sha256(bytes),
            token_count: 0,
        };
        let mut ids = HashSet::new();
        for (index, line) in bytes.split_inclusive(|&byte| byte == b'\n').enumerate() {
            let document = (|| -> Result<Document> {
                ensure!(line.len() <= MAX_LINE_BYTES, "JSONL line exceeds 16 MiB");
                let document: Document = serde_json::from_slice(line)?;
                document.validate()?;
                ensure!(
                    ids.insert(document.id.clone()),
                    "duplicate document ID {:?}",
                    document.id
                );
                Ok(document)
            })()
            .with_context(|| format!("line {}", index + 1))?;
            let tokens = tokenize(&document.source);
            dataset.token_count += tokens.len();
            ensure!(
                tokens.len() <= tint_model::MAX_TOKENS,
                "line {} exceeds 262144 tokens",
                index + 1
            );
            ensure!(
                dataset.token_count <= MAX_TOTAL_TOKENS,
                "line {} exceeds dataset limit of {MAX_TOTAL_TOKENS} tokens",
                index + 1
            );
            let labels = align_labels(&tokens, &document.spans);
            let states = token_context(&document.source, &tokens);
            dataset.fingerprints.push(Fingerprint {
                id: document.id,
                group: document.group,
                source_sha256: sha256(document.source.as_bytes()),
            });
            dataset.documents.push(Example {
                language: document.language,
                tokens,
                states,
                labels,
            });
        }
        ensure!(!dataset.documents.is_empty(), "dataset is empty");
        ensure!(
            dataset
                .documents
                .iter()
                .any(|doc| doc.labels.iter().any(Option::is_some)),
            "dataset has no unambiguous non-whitespace labels"
        );
        Ok(dataset)
    }
}

pub fn overlaps(left: &[Fingerprint], right: &[Fingerprint]) -> Option<String> {
    let ids: HashSet<_> = left.iter().map(|f| f.id.as_str()).collect();
    let groups: HashSet<_> = left.iter().map(|f| f.group.as_str()).collect();
    let hashes: HashSet<_> = left.iter().map(|f| f.source_sha256.as_str()).collect();
    for item in right {
        if ids.contains(item.id.as_str()) {
            return Some(format!("duplicate document ID {:?}", item.id));
        }
        if groups.contains(item.group.as_str()) {
            return Some(format!("group {:?} crosses splits", item.group));
        }
        if hashes.contains(item.source_sha256.as_str()) {
            return Some(format!(
                "source SHA256 {} crosses splits",
                item.source_sha256
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCUMENT: &str = r#"{"version":1,"id":"a","group":"g","language":"rust","source":"let","spans":[{"start":0,"end":3,"class":"keyword"}]}"#;

    #[test]
    fn strict_jsonl() {
        assert!(Dataset::parse(DOCUMENT.as_bytes()).is_ok());
        for input in [
            format!("{DOCUMENT}\n{DOCUMENT}"),
            format!("{DOCUMENT}\n\n"),
            DOCUMENT.replace("\"version\":1", "\"version\":2"),
            DOCUMENT.replace("\"version\":1", "\"unknown\":1"),
            DOCUMENT.replace("\"end\":3", "\"end\":2"),
        ] {
            let error = Dataset::parse(input.as_bytes()).err().unwrap();
            assert!(format!("{error:#}").contains("line"));
        }
    }

    #[test]
    fn leakage_checks_all_keys() {
        let a = Fingerprint {
            id: "a".into(),
            group: "g".into(),
            source_sha256: "hash".into(),
        };
        for b in [
            Fingerprint {
                id: "a".into(),
                group: "h".into(),
                source_sha256: "other".into(),
            },
            Fingerprint {
                id: "b".into(),
                ..a.clone()
            },
            Fingerprint {
                id: "b".into(),
                group: "h".into(),
                source_sha256: "hash".into(),
            },
        ] {
            assert!(overlaps(std::slice::from_ref(&a), &[b]).is_some());
        }
        assert!(
            overlaps(
                &[a],
                &[Fingerprint {
                    id: "b".into(),
                    group: "h".into(),
                    source_sha256: "other".into()
                }]
            )
            .is_none()
        );
    }

    #[test]
    fn file_and_token_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl");
        std::fs::write(&path, DOCUMENT).unwrap();
        assert!(read_bounded(&path, DOCUMENT.len()).is_ok());
        assert!(read_bounded(&path, DOCUMENT.len() - 1).is_err());
        assert!(read_bounded(dir.path(), MAX_DATA_BYTES).is_err());
        assert!(Dataset::parse(b"").is_err());
        assert!(Dataset::parse(DOCUMENT.replace("let", "   ").as_bytes()).is_err());
        let source = ";".repeat(tint_model::MAX_TOKENS + 1);
        let document = serde_json::json!({ "version": 1, "id": "large", "group": "g", "language": "rust", "source": source, "spans": [{ "start": 0, "end": source.len(), "class": "plain" }] });
        let error = Dataset::parse(&serde_json::to_vec(&document).unwrap())
            .err()
            .unwrap();
        assert!(format!("{error:#}").contains("262144 tokens"));
    }
}
