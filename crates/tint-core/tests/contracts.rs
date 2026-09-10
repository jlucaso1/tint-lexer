use tint_core::*;

fn span(start: usize, end: usize, class: SyntaxClass) -> Span {
    Span { start, end, class }
}

fn document(source: &str) -> Document {
    Document {
        version: 1,
        id: "sample".into(),
        group: "repository".into(),
        language: "unknown".into(),
        source: source.into(),
        spans: if source.is_empty() {
            vec![]
        } else {
            vec![span(0, source.len(), SyntaxClass::Plain)]
        },
    }
}

#[test]
fn class_ids_and_serialization_are_stable() {
    let names = [
        "plain", "comment", "string", "number", "keyword", "type", "function", "constant",
        "operator",
    ];
    for (id, class) in SyntaxClass::ALL.into_iter().enumerate() {
        assert_eq!(class.id(), id);
        assert_eq!(SyntaxClass::try_from(id), Ok(class));
        let encoded = serde_json::to_value(class).unwrap();
        assert_eq!(encoded, names[id]);
        assert_eq!(
            serde_json::from_value::<SyntaxClass>(encoded).unwrap(),
            class
        );
    }
    for id in [9, 10, usize::MAX] {
        assert_eq!(SyntaxClass::try_from(id), Err(CoreError::InvalidClass(id)));
    }
    assert_eq!(
        serde_json::to_value(span(0, 1, SyntaxClass::Type)).unwrap(),
        serde_json::json!({"start": 0, "end": 1, "class": "type"})
    );
}

#[test]
fn tokenizer_preserves_utf8_crlf_and_combining_marks() {
    let source = "foo_42\t \r\n\n\r+e\u{301}\u{1f600}\u{4e2d};\u{a0}x";
    let tokens = tokenize(source);
    let parts: Vec<_> = tokens.iter().map(|t| &source[t.start..t.end]).collect();
    assert_eq!(
        parts,
        [
            "foo_42",
            "\t ",
            "\r\n",
            "\n",
            "\r",
            "+",
            "e\u{301}\u{1f600}\u{4e2d}",
            ";",
            "\u{a0}",
            "x"
        ]
    );
    assert_eq!(parts.concat(), source);
    assert_eq!(
        tokens.iter().map(|t| t.kind).collect::<Vec<_>>(),
        [
            TokenKind::Word,
            TokenKind::Whitespace,
            TokenKind::Newline,
            TokenKind::Newline,
            TokenKind::Newline,
            TokenKind::Symbol,
            TokenKind::Word,
            TokenKind::Symbol,
            TokenKind::Whitespace,
            TokenKind::Word,
        ]
    );
    for (i, token) in tokens.iter().enumerate() {
        assert_eq!(token.start, if i == 0 { 0 } else { tokens[i - 1].end });
        assert!(token.start < token.end);
        assert!(source.is_char_boundary(token.start));
        assert!(source.is_char_boundary(token.end));
        assert_eq!(
            token.is_whitespace(),
            parts[i].chars().all(char::is_whitespace)
        );
    }
    assert!(tokenize("").is_empty());
}

#[test]
fn features_stay_in_disjoint_namespaces() {
    let source = "aA0_\u{301} \r\n+-\0\u{7f}\u{1f600} abcdefghijklmnopqrstuvwxyz";
    let ranges = [1..=4, 5..=20, 21..=84, 85..=212, 213..=340, 341..=1364];
    for token in tokenize(source) {
        for (feature, range) in token.features.into_iter().zip(&ranges) {
            assert!(range.contains(&feature), "{token:?}");
            assert!((feature as usize) < FEATURE_VOCAB_SIZE);
        }
    }
    assert_eq!(FEATURE_COUNT, 6);
    assert_eq!(FEATURE_VERSION, 1);
    assert_eq!(FEATURE_VOCAB_SIZE, 1365);
    assert_eq!(tokenize("a")[0].features, [1, 5, 22, 182, 310, 641]);
    assert_eq!(tokenize("aA0_\u{301}")[0].features[2], 68);
    assert_eq!(tokenize("\u{1f600}")[0].features[1..5], [5, 53, 212, 340]);
    assert_eq!(tokenize("abcdefghijklmnop")[0].features[1], 20);
    assert_eq!(tokenize("abcdefghijklmnopqrstuvwxyz")[0].features[1], 20);
    assert_eq!(tokenize(source), tokenize(source));
}

#[test]
fn windows_pad_both_sides_and_handle_out_of_range_indices() {
    let tokens = tokenize("a+b");
    assert_eq!(window_features(&tokens, 1, 0), tokens[1].features);
    let expected: Vec<i32> = [
        vec![0; 12],
        tokens[0].features.to_vec(),
        tokens[1].features.to_vec(),
        tokens[2].features.to_vec(),
    ]
    .concat();
    assert_eq!(window_features(&tokens, 0, 2), expected);
    let expected: Vec<i32> = [
        tokens[0].features.to_vec(),
        tokens[1].features.to_vec(),
        tokens[2].features.to_vec(),
        vec![0; 12],
    ]
    .concat();
    assert_eq!(window_features(&tokens, 2, 2), expected);
    assert_eq!(
        window_features(&tokens, 3, 1),
        [tokens[2].features.to_vec(), vec![0; 12]].concat()
    );
    assert_eq!(window_features(&[], 0, 2), vec![0; 30]);
    assert_eq!(window_features(&tokens, usize::MAX, 1), vec![0; 18]);
}

#[test]
#[should_panic(expected = "feature window exceeds addressable allocation size")]
fn impossible_window_size_is_rejected_before_allocation() {
    window_features(&[], 0, usize::MAX);
}

#[test]
fn utf16_conversion_matches_prefix_encoding_with_gaps() {
    let source = "a\u{1f600}e\u{301}\r\n\u{1f680}z";
    let boundaries: Vec<_> = source
        .char_indices()
        .map(|(i, _)| i)
        .chain([source.len()])
        .collect();
    let spans: Vec<_> = boundaries
        .windows(2)
        .map(|b| span(b[0], b[1], SyntaxClass::String))
        .collect();
    for input in [spans.clone(), spans.iter().step_by(2).copied().collect()] {
        let converted = utf16_spans(source, &input).unwrap();
        for (original, actual) in input.iter().zip(converted) {
            assert_eq!(
                actual.start,
                source[..original.start].encode_utf16().count()
            );
            assert_eq!(actual.end, source[..original.end].encode_utf16().count());
            assert_eq!(actual.class, original.class);
        }
    }
    assert_eq!(
        utf16_spans(source, &[span(1, 5, SyntaxClass::Plain)]).unwrap(),
        [span(1, 3, SyntaxClass::Plain)]
    );
    assert!(utf16_spans("", &[]).unwrap().is_empty());
    assert!(utf16_spans(source, &[]).unwrap().is_empty());
}

#[test]
fn invalid_offsets_are_rejected() {
    let source = "a\u{1f600}b";
    let invalid = [
        vec![span(0, 0, SyntaxClass::Plain)],
        vec![span(5, 1, SyntaxClass::Plain)],
        vec![span(0, 7, SyntaxClass::Plain)],
        vec![span(usize::MAX, usize::MAX, SyntaxClass::Plain)],
        vec![span(2, 5, SyntaxClass::Plain)],
        vec![span(0, 2, SyntaxClass::Plain)],
        vec![
            span(0, 5, SyntaxClass::Plain),
            span(1, 6, SyntaxClass::String),
        ],
        vec![
            span(5, 6, SyntaxClass::Plain),
            span(0, 1, SyntaxClass::String),
        ],
    ];
    for spans in invalid {
        assert!(matches!(
            utf16_spans(source, &spans),
            Err(CoreError::InvalidSpan(_))
        ));
        let mut doc = document(source);
        doc.spans = spans;
        assert!(doc.validate().is_err());
    }
}

#[test]
fn document_requires_complete_coverage_and_valid_metadata() {
    assert!(document("").validate().is_ok());
    assert!(document("a\u{1f600}\r\ne\u{301}").validate().is_ok());
    for spans in [
        vec![],
        vec![span(1, 3, SyntaxClass::Plain)],
        vec![span(0, 2, SyntaxClass::Plain)],
        vec![
            span(0, 1, SyntaxClass::Plain),
            span(2, 3, SyntaxClass::Plain),
        ],
    ] {
        let mut doc = document("abc");
        doc.spans = spans;
        assert_eq!(doc.validate(), Err(CoreError::IncompleteCoverage));
    }
    let mut doc = document("");
    doc.spans.push(span(0, 0, SyntaxClass::Plain));
    assert!(doc.validate().is_err());
    for version in [0, 2, u32::MAX] {
        let mut doc = document("");
        doc.version = version;
        assert_eq!(doc.validate(), Err(CoreError::UnsupportedVersion(version)));
    }
    for name in ["id", "group", "language"] {
        let mut doc = document("");
        match name {
            "id" => doc.id = " ".into(),
            "group" => doc.group.clear(),
            _ => doc.language.clear(),
        }
        assert_eq!(doc.validate(), Err(CoreError::EmptyField(name)));
    }
    let mut doc = document(&"a".repeat(MAX_SOURCE_BYTES));
    assert!(doc.validate().is_ok());
    doc.source.push('a');
    assert_eq!(
        doc.validate(),
        Err(CoreError::SourceTooLarge(MAX_SOURCE_BYTES + 1))
    );
}

#[test]
fn dataset_schema_is_strict_and_round_trips() {
    let doc = document("hello");
    let value = serde_json::to_value(&doc).unwrap();
    assert_eq!(
        serde_json::from_value::<Document>(value.clone()).unwrap(),
        doc
    );
    let mut unknown = value.clone();
    unknown["extra"] = true.into();
    assert!(serde_json::from_value::<Document>(unknown).is_err());
    let mut unknown_span = value.clone();
    unknown_span["spans"][0]["label"] = "plain".into();
    assert!(serde_json::from_value::<Document>(unknown_span).is_err());
    for field in ["version", "id", "group", "language", "source", "spans"] {
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<Document>(missing).is_err());
    }
    for class in [serde_json::json!("unknown"), serde_json::json!(0)] {
        let mut invalid = value.clone();
        invalid["spans"][0]["class"] = class;
        assert!(serde_json::from_value::<Document>(invalid).is_err());
    }
}

#[test]
fn alignment_accepts_same_class_splits_and_ignores_ambiguous_words() {
    use SyntaxClass::{Keyword, Plain, String};
    let tokens = tokenize("abcd + ef\r\n");
    let spans = [
        span(0, 2, Keyword),
        span(2, 4, Keyword),
        span(4, 7, Plain),
        span(7, 8, Keyword),
        span(8, 9, String),
        span(9, 11, String),
    ];
    assert_eq!(
        align_labels(&tokens, &spans),
        [Some(Keyword), None, Some(Plain), None, None, None]
    );
    assert_eq!(
        align_labels(&tokens, &[span(0, 11, String)]),
        [Some(String), None, Some(String), None, Some(String), None]
    );
    assert_eq!(align_labels(&tokens, &[]), vec![None; tokens.len()]);
    assert!(align_labels(&[], &spans).is_empty());
    assert_eq!(
        align_labels(&tokenize("abcd"), &[span(0, 2, Plain), span(3, 4, Plain)]),
        [None]
    );
    assert_eq!(
        align_labels(
            &tokenize("abcd+ef"),
            &[span(0, 2, Plain), span(2, 7, Keyword)]
        ),
        [None, Some(Keyword), Some(Keyword)]
    );
}

#[test]
fn merge_checks_partitions_and_coalesces_labels() {
    use SyntaxClass::{Keyword, Plain};
    let tokens = tokenize("let x=1;\r\n");
    let labels = [Keyword, Plain, Plain, Plain, Plain, Plain, Plain];
    let merged = merge_spans(&tokens, &labels).unwrap();
    assert_eq!(merged, [span(0, 3, Keyword), span(3, 10, Plain)]);
    assert!(merge_spans(&[], &[]).unwrap().is_empty());
    assert!(matches!(
        merge_spans(&tokens, &[]),
        Err(CoreError::LabelCount { .. })
    ));
    for (start, end) in [(1, 3), (0, 0), (3, 2)] {
        let mut invalid = tokens.clone();
        invalid[0].start = start;
        invalid[0].end = end;
        assert_eq!(
            merge_spans(&invalid, &labels),
            Err(CoreError::InvalidToken(0))
        );
    }
    for start in [2, 4] {
        let mut invalid = tokens.clone();
        invalid[1].start = start;
        assert_eq!(
            merge_spans(&invalid, &labels),
            Err(CoreError::InvalidToken(1))
        );
    }
}

#[test]
fn merge_and_alignment_round_trip_for_generated_inputs() {
    let pieces = [
        "a",
        "\u{301}",
        "\u{1f600}",
        "\u{4e2d}",
        "_0",
        " ",
        "\t",
        "\r",
        "\n",
        "+",
        "\0",
    ];
    for a in pieces {
        for b in pieces {
            for c in pieces {
                let source = [a, b, c].concat();
                let tokens = tokenize(&source);
                assert_eq!(
                    tokens
                        .iter()
                        .map(|t| &source[t.start..t.end])
                        .collect::<String>(),
                    source
                );
                let labels: Vec<_> = (0..tokens.len()).map(|i| SyntaxClass::ALL[i % 9]).collect();
                let spans = merge_spans(&tokens, &labels).unwrap();
                let mut doc = document(&source);
                doc.spans = spans.clone();
                doc.validate().unwrap();
                let expected: Vec<_> = tokens
                    .iter()
                    .zip(labels)
                    .map(|(t, label)| (!t.is_whitespace()).then_some(label))
                    .collect();
                assert_eq!(align_labels(&tokens, &spans), expected);
                let converted = utf16_spans(&source, &spans).unwrap();
                assert_eq!(converted.last().unwrap().end, source.encode_utf16().count());
            }
        }
    }
}
