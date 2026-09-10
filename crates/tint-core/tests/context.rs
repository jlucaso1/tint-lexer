use tint_core::{
    STATE_BLOCK_COMMENT, STATE_LINE_COMMENT, STATE_MASK, STATE_STRING, StateScanner,
    dilated_window_features, token_context, token_states, tokenize, window_features,
};

#[test]
fn local_windows_cannot_identify_distant_string_or_comment_delimiters() {
    let expression = "const result = alpha + beta + gamma + delta + epsilon;";
    let sources = [
        expression.to_owned(),
        format!("/* {expression} */"),
        format!("\"{expression}\";"),
    ];
    let windows: Vec<_> = sources
        .iter()
        .map(|source| {
            let tokens = tokenize(source);
            let index = tokens
                .iter()
                .position(|token| &source[token.start..token.end] == "gamma")
                .unwrap();
            window_features(&tokens, index, 4)
        })
        .collect();
    assert_eq!(windows[0], windows[1]);
    assert_eq!(windows[0], windows[2]);
}

#[test]
fn dilated_windows_observe_distant_delimiters() {
    let expression = "const result = alpha + beta + gamma + delta + epsilon;";
    let sources = [
        expression.to_owned(),
        format!("/* {expression} */"),
        format!("\"{expression}\";"),
    ];
    let windows: Vec<_> = sources
        .iter()
        .map(|source| {
            let tokens = tokenize(source);
            let index = tokens
                .iter()
                .position(|token| &source[token.start..token.end] == "gamma")
                .unwrap();
            dilated_window_features(&tokens, index)
        })
        .collect();
    assert_eq!(windows[0].len(), 17 * tint_core::FEATURE_COUNT);
    // The block-comment opener lands on the -16 anchor, but the string quote
    // sits 15 tokens out, between anchors. Fixed dilation has blind spots.
    assert_ne!(windows[0], windows[1]);
    assert_eq!(windows[0], windows[2]);
    // The dense radius-4 prefix stays identical in all three inputs.
    for window in &windows {
        assert_eq!(
            &window[..9 * tint_core::FEATURE_COUNT],
            &windows[0][..9 * tint_core::FEATURE_COUNT]
        );
    }
    assert_ne!(
        &windows[1][9 * tint_core::FEATURE_COUNT..],
        &windows[0][9 * tint_core::FEATURE_COUNT..]
    );
}

#[test]
fn prefix_state_distinguishes_all_three_delimiter_contexts() {
    let expression = "const result = alpha + beta + gamma + delta + epsilon;";
    let cases = [
        (expression.to_owned(), 0),
        (format!("/* {expression} */"), STATE_BLOCK_COMMENT),
        (format!("\"{expression}\";"), STATE_STRING),
        (format!("// {expression}"), STATE_LINE_COMMENT),
    ];
    for (source, expected) in cases {
        let tokens = tokenize(&source);
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == "gamma")
            .unwrap();
        let states = token_states(&source, &tokens);
        assert_eq!(states.len(), tokens.len());
        assert_eq!(states[index], expected, "source: {source}");
    }
}

#[test]
fn prefix_state_handles_escapes_and_closers() {
    let source = r#"const a = "x \" gamma"; /* done */ // tail gamma"#;
    let tokens = tokenize(source);
    let states = token_states(source, &tokens);
    let at = |text: &str| {
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == text)
            .unwrap();
        states[index]
    };
    assert_eq!(at("gamma"), STATE_STRING);
    assert_eq!(at("done"), STATE_BLOCK_COMMENT);
    assert_eq!(at("tail"), STATE_LINE_COMMENT);
    assert_eq!(at("const"), 0);
}

#[test]
fn streaming_scanner_matches_batched_states() {
    let source = "/* a */ const b = \"c \\\" d\"; // e\nf";
    let tokens = tokenize(source);
    let batched = token_states(source, &tokens);
    let mut scanner = StateScanner::new();
    let streamed: Vec<u8> = tokens
        .iter()
        .map(|token| scanner.advance(source, token.start, token.end) & STATE_MASK)
        .collect();
    assert_eq!(batched, streamed);
}
#[test]
fn apostrophes_in_prose_do_not_open_strings() {
    let cases = [
        ("// don't panic", "panic", STATE_LINE_COMMENT),
        ("// it's fine", "fine", STATE_LINE_COMMENT),
        ("let s = 'quoted';", "quoted", STATE_STRING),
        ("rock 'n' roll", "n", STATE_STRING),
        ("dogs' toys", "toys", STATE_STRING),
    ];
    for (source, probe, expected) in cases {
        let tokens = tokenize(source);
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == probe)
            .unwrap();
        let states = token_context(source, &tokens);
        assert_eq!(states[index] & STATE_MASK, expected, "source: {source}");
    }
}

#[test]
fn apostrophes_close_unconditionally() {
    // The symmetric variant (mid-word closes skipped) measured worse on
    // validation and test, so closing stays unconditional. 'don't' flickers
    // shut at the apostrophe and reopens; the model learns through it.
    let source = "let s = 'don''t x';";
    let tokens = tokenize(source);
    let states = token_context(source, &tokens);
    let at = |text: &str| {
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == text)
            .unwrap();
        states[index] & STATE_MASK
    };
    assert_eq!(at("don"), STATE_STRING);
    assert_eq!(at("s"), 0);
}

#[test]
fn line_markers_fire_with_guards() {
    let cases = [
        ("x = 1 # comment\ny = 2", "comment", STATE_LINE_COMMENT),
        ("# full line", "full", STATE_LINE_COMMENT),
        ("#!/usr/bin/env python", "usr", STATE_LINE_COMMENT),
        ("mov eax, 1 ; load", "load", 0),
        ("; load all\nmov eax, 1", "load", STATE_LINE_COMMENT),
        ("-- haskell comment", "haskell", STATE_LINE_COMMENT),
        ("% matlab comment", "matlab", STATE_LINE_COMMENT),
        ("! fortran comment", "fortran", STATE_LINE_COMMENT),
        ("alpha # beta", "beta", STATE_LINE_COMMENT),
        ("http://example.com/#frag", "frag", STATE_LINE_COMMENT),
    ];
    for (source, probe, expected) in cases {
        let tokens = tokenize(source);
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == probe)
            .unwrap();
        let states = token_context(source, &tokens);
        assert_eq!(states[index] & STATE_MASK, expected, "source: {source}");
    }
}

#[test]
fn marker_misfires_stay_quiet() {
    let cases = [
        ("#include <stdio.h>", "include", 0),
        ("#[derive(Debug)]", "derive", 0),
        ("let s = a#b;", "b", 0),
        ("x--;\ny", "y", 0),
        ("a = b % c;", "c", 0),
        ("if (!ready) {}", "ready", 0),
    ];
    for (source, probe, expected) in cases {
        let tokens = tokenize(source);
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == probe)
            .unwrap();
        let states = token_context(source, &tokens);
        assert_eq!(states[index] & STATE_MASK, expected, "source: {source}");
    }
}

#[test]
fn block_scalar_content_suppresses_hash_comments() {
    let source = "key_block: |\n  first # still text\n  second\n# real comment\n";
    let tokens = tokenize(source);
    let states = token_context(source, &tokens);
    let at = |text: &str| {
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == text)
            .unwrap();
        states[index] & STATE_MASK
    };
    assert_eq!(at("still"), 0);
    assert_eq!(at("second"), 0);
    assert_eq!(at("real"), STATE_LINE_COMMENT);
}

#[test]
fn block_scalar_headers_cover_indicators_and_exits() {
    // Folded, chomped, and nested `key: ` headers arm scalar content; a
    // dedented plain line closes it again. Only `: ` plus `|`/`>` arms:
    // `- |`, bare `>`, and `--- >` stay plain context (documented).
    let cases = [
        "a: >\n  folded # text\nb: 1\n  # after\n",
        "a: |-\n  kept # text\n# after\n",
        "outer:\n  inner: |3-\n    deep # text\n  sibling: 1\n    # after\n",
        "key: |2\n  indented # text\nother: 1\n  # after\n",
    ];
    for source in cases {
        let tokens = tokenize(source);
        let states = token_context(source, &tokens);
        let at = |text: &str| {
            let index = tokens
                .iter()
                .position(|token| &source[token.start..token.end] == text)
                .unwrap();
            states[index] & STATE_MASK
        };
        assert_eq!(at("text"), 0, "source: {source}");
        assert_eq!(at("after"), STATE_LINE_COMMENT, "source: {source}");
    }
}

#[test]
fn pipes_without_colons_are_not_scalar_headers() {
    // Shell-style continuations (`curl http://x |`) must not arm scalar
    // tracking: the indented `#` below stays a comment.
    let source = "curl http://example.com |\n  # real comment\n";
    let tokens = tokenize(source);
    let states = token_context(source, &tokens);
    let index = tokens
        .iter()
        .position(|token| &source[token.start..token.end] == "real")
        .unwrap();
    assert_eq!(states[index] & STATE_MASK, STATE_LINE_COMMENT);
    // A header with a trailing comment still arms tracking (the `|` follows
    // `: `), so the indented `#` below is scalar content, not a comment.
    let source = "key: | # comment\n  # content\n# real\n";
    let tokens = tokenize(source);
    let states = token_context(source, &tokens);
    let at = |text: &str| {
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == text)
            .unwrap();
        states[index] & STATE_MASK
    };
    assert_eq!(at("comment"), STATE_LINE_COMMENT);
    assert_eq!(at("content"), 0);
    assert_eq!(at("real"), STATE_LINE_COMMENT);
}

#[test]
fn trailing_dashdash_comments_fire_with_blanks() {
    let cases = [
        ("SELECT a, b -- strip spaces", "strip", STATE_LINE_COMMENT),
        ("f(...) --[[ long bracket ]]", "bracket", 0),
        ("x--", "x", 0),
        ("i-- ; j", "j", 0),
        ("x --y", "y", 0),
        ("a --- b", "b", 0),
        ("-- full line", "full", STATE_LINE_COMMENT),
    ];
    for (source, probe, expected) in cases {
        let tokens = tokenize(source);
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == probe)
            .unwrap();
        let states = token_context(source, &tokens);
        assert_eq!(states[index] & STATE_MASK, expected, "source: {source}");
    }
}

#[test]
fn aligned_trailing_semicolons_fire_double_blank_only() {
    let cases = [
        (
            "safeseh handler         ; register handler",
            "register",
            STATE_LINE_COMMENT,
        ),
        ("i32.store  ;; store value", "value", STATE_LINE_COMMENT),
        ("mov eax, 1 ; load", "load", 0),
        ("a = 1 ; b = 2", "b", 0),
        ("; full line", "full", STATE_LINE_COMMENT),
    ];
    for (source, probe, expected) in cases {
        let tokens = tokenize(source);
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == probe)
            .unwrap();
        let states = token_context(source, &tokens);
        assert_eq!(states[index] & STATE_MASK, expected, "source: {source}");
    }
}

#[test]
fn percent_needs_a_blank_or_block_marker() {
    let cases = [
        ("% matlab comment", "matlab", STATE_LINE_COMMENT),
        ("%{ block open", "block", STATE_LINE_COMMENT),
        ("%macro name", "name", 0),
        ("%let x = 1", "x", 0),
        ("%x[ ls ]", "ls", 0),
    ];
    for (source, probe, expected) in cases {
        let tokens = tokenize(source);
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == probe)
            .unwrap();
        let states = token_context(source, &tokens);
        assert_eq!(states[index] & STATE_MASK, expected, "source: {source}");
    }
}

#[test]
fn hash_dollar_opens_a_comment() {
    // Shell/Perl `#$var` interpolation is always comment text; `a#b`
    // (macro paste, fragment) still stays quiet.
    let cases = [
        ("#$o->{x} = 1", "o", STATE_LINE_COMMENT),
        ("let s = a#b;", "b", 0),
    ];
    for (source, probe, expected) in cases {
        let tokens = tokenize(source);
        let index = tokens
            .iter()
            .position(|token| &source[token.start..token.end] == probe)
            .unwrap();
        let states = token_context(source, &tokens);
        assert_eq!(states[index] & STATE_MASK, expected, "source: {source}");
    }
}

#[test]
fn context_bytes_carry_quote_and_distance() {
    let source = "\"abcdef ghij\"";
    let tokens = tokenize(source);
    let context = token_context(source, &tokens);
    assert_eq!(context.len(), tokens.len());
    let inner = context[1];
    assert_eq!(inner & STATE_MASK, STATE_STRING);
    assert_eq!((inner >> 3) & 3, 1);
    assert_eq!((inner >> 5) & 3, 1);
}
