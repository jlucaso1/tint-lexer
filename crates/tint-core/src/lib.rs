//! Language-independent tokenization and byte-offset annotation contracts.
//!
//! Tokenization is lossless, not a programming-language parser. All non-ASCII
//! non-whitespace characters belong to word runs, including combining marks,
//! emoji, and punctuation. Only CR and LF delimit newline tokens.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const FEATURE_COUNT: usize = 6;
/// Includes the padding ID, zero.
pub const FEATURE_VOCAB_SIZE: usize = 1365;
pub const FEATURE_VERSION: u32 = 1;
/// Version 2 keeps the six categorical features and adds three center-token
/// document-context bits from [`token_states`], packed into spare packing
/// bits. Nothing in the version-1 pipeline may depend on them.
pub const FEATURE_VERSION_V2: u32 = 2;
/// Center-token context bits consumed by version-2 models.
pub const STATE_COUNT: usize = 3;
pub const MAX_SOURCE_BYTES: usize = 4 * 1024 * 1024;
pub const DOCUMENT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum SyntaxClass {
    Plain,
    Comment,
    String,
    Number,
    Keyword,
    Type,
    Function,
    Constant,
    Operator,
}

impl SyntaxClass {
    pub const ALL: [Self; 9] = [
        Self::Plain,
        Self::Comment,
        Self::String,
        Self::Number,
        Self::Keyword,
        Self::Type,
        Self::Function,
        Self::Constant,
        Self::Operator,
    ];

    pub const fn id(self) -> usize {
        self as usize
    }
}

impl TryFrom<usize> for SyntaxClass {
    type Error = CoreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::ALL
            .get(value)
            .copied()
            .ok_or(CoreError::InvalidClass(value))
    }
}

/// Half-open byte offsets, except in the output of [`utf16_spans`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    #[serde(rename = "class")]
    pub class: SyntaxClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TokenKind {
    Word = 1,
    Whitespace = 2,
    Newline = 3,
    Symbol = 4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub start: usize,
    pub end: usize,
    pub features: [i32; FEATURE_COUNT],
    pub kind: TokenKind,
}

impl Token {
    pub const fn is_whitespace(&self) -> bool {
        matches!(self.kind, TokenKind::Whitespace | TokenKind::Newline)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CoreError {
    #[error("invalid syntax class ID {0}")]
    InvalidClass(usize),
    #[error("token count {tokens} differs from label count {labels}")]
    LabelCount { tokens: usize, labels: usize },
    #[error("invalid token offsets at index {0}")]
    InvalidToken(usize),
    #[error("invalid span offsets at index {0}")]
    InvalidSpan(usize),
    #[error("spans do not completely cover the source")]
    IncompleteCoverage,
    #[error("unsupported document version {0}")]
    UnsupportedVersion(u32),
    #[error("document field {0} must not be empty")]
    EmptyField(&'static str),
    #[error("source has {0} bytes, exceeding the source limit")]
    SourceTooLarge(usize),
}

fn kind(ch: char) -> TokenKind {
    if matches!(ch, '\r' | '\n') {
        TokenKind::Newline
    } else if ch.is_whitespace() {
        TokenKind::Whitespace
    } else if ch.is_ascii_alphanumeric() || ch == '_' || !ch.is_ascii() {
        TokenKind::Word
    } else {
        TokenKind::Symbol
    }
}

/// Partitions every byte of `source` into nonempty tokens on character boundaries.
/// CRLF is one token. Other CR and LF characters are individual tokens.
///
/// Feature version 1 uses kind IDs 1..=4, scalar length capped at 16 in
/// 5..=20, and a six-bit shape mask in 21..=84. Shape bits record lowercase,
/// uppercase, numeric, underscore, whitespace, and other characters, in order.
/// First and last scalar IDs occupy 85..=212 and 213..=340. Non-ASCII scalars
/// use 127, also the ASCII DEL value. UTF-8 FNV-1a with a wrapping 32-bit state,
/// reduced modulo 1024, occupies 341..=1364. Zero is reserved for padding.
pub fn tokenize(source: &str) -> Vec<Token> {
    iter_tokens(source).collect()
}

/// Produces the same tokens as [`tokenize`] without retaining previous tokens.
pub fn iter_tokens(source: &str) -> impl Iterator<Item = Token> + '_ {
    let mut chars = source.char_indices().peekable();
    std::iter::from_fn(move || {
        let (start, first) = chars.next()?;
        let token_kind = kind(first);
        let mut end = start + first.len_utf8();
        if first == '\r' && chars.peek().is_some_and(|&(_, ch)| ch == '\n') {
            chars.next();
            end += 1;
        } else if matches!(token_kind, TokenKind::Word | TokenKind::Whitespace) {
            while let Some(&(offset, ch)) = chars.peek() {
                if kind(ch) != token_kind {
                    break;
                }
                chars.next();
                end = offset + ch.len_utf8();
            }
        }
        let text = &source[start..end];
        let mut length = 0usize;
        let mut shape = 0i32;
        let mut last = first;
        for ch in text.chars() {
            length += 1;
            last = ch;
            let char_shape = i32::from(ch.is_lowercase())
                | (i32::from(ch.is_uppercase()) << 1)
                | (i32::from(ch.is_numeric()) << 2)
                | (i32::from(ch == '_') << 3)
                | (i32::from(ch.is_whitespace()) << 4);
            shape |= if char_shape == 0 { 32 } else { char_shape };
        }
        let hash = text.bytes().fold(2_166_136_261u32, |state, byte| {
            (state ^ u32::from(byte)).wrapping_mul(16_777_619)
        });
        let scalar_id = |ch: char| if ch.is_ascii() { ch as i32 } else { 127 };
        Some(Token {
            start,
            end,
            kind: token_kind,
            features: [
                token_kind as i32,
                5 + (length.min(16) - 1) as i32,
                21 + shape,
                85 + scalar_id(first),
                213 + scalar_id(last),
                341 + (hash % 1024) as i32,
            ],
        })
    })
}

/// Flattens a centered token window, padding missing tokens with zeroes.
/// An index outside `tokens` is allowed; any positions still in range are copied.
///
/// # Panics
/// Panics if the output size exceeds `isize::MAX` bytes. The caller must choose
/// a radius whose allocation fits available memory, as with other `Vec` APIs.
pub fn window_features(tokens: &[Token], index: usize, radius: usize) -> Vec<i32> {
    let length = radius
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_mul(FEATURE_COUNT))
        .filter(|&n| n <= isize::MAX as usize / size_of::<i32>())
        .expect("feature window exceeds addressable allocation size");
    let mut output = vec![0; length];
    for (slot, target) in output
        .as_chunks_mut::<FEATURE_COUNT>()
        .0
        .iter_mut()
        .enumerate()
    {
        let position = if slot < radius {
            index.checked_sub(radius - slot)
        } else {
            index.checked_add(slot - radius)
        };
        if let Some(token) = position.and_then(|position| tokens.get(position)) {
            target.copy_from_slice(&token.features);
        }
    }
    output
}

/// Sparse long-range anchors for the experimental dilated window.
/// Dense radius 4 covers offsets -4..=+4; these extend visibility to -32..+32
/// so distant delimiters fall inside the window without widening every slot.
pub const DILATED_OFFSETS: [i64; 8] = [-32, -16, -8, 8, 16, 32, -64, 64];
/// Total slots in [`dilated_window_features`]: nine dense plus eight sparse.
pub const DILATED_SLOTS: usize = 17;

/// Flattens a dilated token window: dense radius 4 plus [`DILATED_OFFSETS`].
/// Ordering is dense offsets -4..=+4 followed by [`DILATED_OFFSETS`] order.
/// Out-of-range positions contribute zeroes, as in [`window_features`].
/// Experimental: feature version 1 pipelines must keep using [`window_features`].
pub fn dilated_window_features(tokens: &[Token], index: usize) -> Vec<i32> {
    let mut output = vec![0; DILATED_SLOTS * FEATURE_COUNT];
    let index = index as i64;
    let last = tokens.len() as i64;
    let mut slot = 0;
    for delta in -4..=4 {
        let position = index + delta;
        if position >= 0 && position < last {
            output[slot * FEATURE_COUNT..(slot + 1) * FEATURE_COUNT]
                .copy_from_slice(&tokens[position as usize].features);
        }
        slot += 1;
    }
    for delta in DILATED_OFFSETS {
        let position = index + delta;
        if position >= 0 && position < last {
            output[slot * FEATURE_COUNT..(slot + 1) * FEATURE_COUNT]
                .copy_from_slice(&tokens[position as usize].features);
        }
        slot += 1;
    }
    debug_assert_eq!(slot, DILATED_SLOTS);
    output
}

/// Bit in [`token_states`] output marking a token inside `"`, `'`, or `` ` ``.
pub const STATE_STRING: u8 = 0b001;
/// Bit marking a token inside a `/* ... */` span. Delimiters do not nest.
pub const STATE_BLOCK_COMMENT: u8 = 0b010;
/// Bit marking a token inside a `//` span, ending at the next newline.
pub const STATE_LINE_COMMENT: u8 = 0b100;

/// Samples language-agnostic document context at each token start.
///
/// A single left-to-right character scan tracks whether the reader is inside
/// a quote, a `/* ... */` span, or a `//` span when each token begins. The
/// output holds one bitmask per token using [`STATE_STRING`],
/// [`STATE_BLOCK_COMMENT`], and [`STATE_LINE_COMMENT`].
///
/// Heuristics, not parsing: backslash escapes apply only inside quotes,
/// block spans do not nest, newlines never close quotes, and `//` inside a
/// quote or block span is ordinary text. Apostrophes in prose can misfire;
/// downstream models must treat these bits as soft hints. Runs in O(source
/// bytes + token count) time. Tokens must partition the source from byte zero.
pub fn token_states(source: &str, tokens: &[Token]) -> Vec<u8> {
    let mut scanner = StateScanner::new();
    tokens
        .iter()
        .map(|token| scanner.advance(source, token.start, token.end) & STATE_MASK)
        .collect()
}

/// Full per-token context byte from [`StateScanner::advance`].
/// Bits 0-2 hold the [`token_states`] mask, bits 3-4 the quote code
/// (0 none, 1 `"`, 2 `'`, 3 backtick), and bits 5-6 the opener-distance
/// bucket (0 outside, 1 for 1-2 tokens, 2 for 3-8, 3 for 9 or more).
pub fn token_context(source: &str, tokens: &[Token]) -> Vec<u8> {
    let mut scanner = StateScanner::new();
    tokens
        .iter()
        .map(|token| scanner.advance(source, token.start, token.end))
        .collect()
}

/// Mask of the three state bits inside a context byte.
pub const STATE_MASK: u8 = 0b0000_0111;
/// Expanded model inputs per center token, in order: three state bits, state
/// change from the previous token, state change to the next token, two quote
/// code bits, two opener-distance bits. Transition bits are derived from
/// neighboring masks, never packed, so streaming needs no lookahead.
pub const STATE_INPUTS: usize = 9;

/// Incremental version of [`token_states`] for streaming tokenizers.
/// Feed tokens in order with their byte offsets; each call returns the full
/// context byte (see [`token_context`]) at the token start, then scans the
/// token bytes. Zero lookahead: every output depends only on bytes seen.
///
/// Line markers beyond `//`: `#` fires anywhere when followed by space,
/// tab, newline, `!`, another `#`, `$`, or end of input. `;`, `--`, `%`, and
/// `!` fire only at line start after whitespace, with `!` needing a space,
/// tab, `!`, `>`, newline, or end-of-input follower and `%` needing a blank,
/// a line break, `%`, or a `{`/`}` block marker. `#[` never fires. A `key: `
/// line ending in `|` or `>` arms scalar tracking, which suppresses later
/// `#` fires on deeper-indented lines until a dedent. These are
/// heuristics: C preprocessor lines and markdown headers can misfire, and
/// single-blank `;` inline comments after code are missed on purpose.
#[derive(Debug, Clone)]
pub struct StateScanner {
    quote: Option<u8>,
    block_depth: u32,
    line: bool,
    line_start: bool,
    cursor: usize,
    tokens_seen: u64,
    entered_at: u64,
    prev_mask: u8,
    scalar: Option<usize>,
    line_indent: usize,
    line_text: bool,
}

impl Default for StateScanner {
    fn default() -> Self {
        Self {
            quote: None,
            block_depth: 0,
            line: false,
            line_start: true,
            cursor: 0,
            tokens_seen: 0,
            entered_at: 0,
            prev_mask: 0,
            scalar: None,
            line_indent: 0,
            line_text: false,
        }
    }
}

impl StateScanner {
    pub fn new() -> Self {
        Self::default()
    }

    fn track(&mut self, byte: u8) {
        if byte == b'\n' || byte == b'\r' {
            self.line_start = true;
            self.line_indent = 0;
            self.line_text = false;
        } else if byte != b' ' && byte != b'\t' {
            // The first text byte of a plain or comment line at or above the
            // scalar base closes the scalar. Lines inside quotes or block
            // spans never close, so a multiline string cannot end one early.
            if !self.line_text
                && self.quote.is_none()
                && self.block_depth == 0
                && self.scalar.is_some_and(|base| self.line_indent <= base)
            {
                self.scalar = None;
            }
            self.line_start = false;
            self.line_text = true;
        } else if !self.line_text {
            self.line_indent += 1;
        }
    }

    /// Whether the current line is block-scalar content, where `#` is text.
    fn in_scalar_content(&self) -> bool {
        self.scalar.is_some_and(|base| self.line_indent > base)
    }

    /// Whether `bytes[cursor]` (`|` or `>`) follows a `key: ` mapping header
    /// and so arms scalar tracking at the current line indent. Runs in the
    /// plain-text branch only, so quotes, block spans, and comments never
    /// arm it; `--- >` tripped the `--` rule first for the same reason.
    fn scalar_trigger(&self, bytes: &[u8]) -> bool {
        self.cursor >= 2
            && bytes[self.cursor - 2] == b':'
            && matches!(bytes[self.cursor - 1], b' ' | b'\t')
    }

    /// Whether `bytes[cursor]` starts a `--` trailing comment: one blank
    /// before (callers already matched the second dash) and a blank, a line
    /// break, or end of input after. `a--`, `x --y`, and `a --- b` stay
    /// operators; spaced decrement (`a -- b`) remains a known misfire.
    fn spaced_trailing(&self, bytes: &[u8]) -> bool {
        if self.cursor == 0 || !matches!(bytes[self.cursor - 1], b' ' | b'\t') {
            return false;
        }
        matches!(
            bytes.get(self.cursor + 2).copied(),
            Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') | None
        )
    }

    /// Whether `bytes[cursor]` (`;`) starts an aligned trailing comment:
    /// two blanks before and a blank, a line break, another `;`, or end of
    /// input after. Single-blank `;` stays code (C-style `a = 1 ; b = 2`).
    fn aligned_trailing(&self, bytes: &[u8]) -> bool {
        if self.cursor < 2
            || !matches!(bytes[self.cursor - 1], b' ' | b'\t')
            || !matches!(bytes[self.cursor - 2], b' ' | b'\t')
        {
            return false;
        }
        matches!(
            bytes.get(self.cursor + 1).copied(),
            Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') | Some(b';') | None
        )
    }

    pub fn advance(&mut self, source: &str, start: usize, end: usize) -> u8 {
        let bytes = source.as_bytes();
        let mut mask = 0;
        if self.quote.is_some() {
            mask |= STATE_STRING;
        }
        if self.block_depth > 0 {
            mask |= STATE_BLOCK_COMMENT;
        }
        if self.line {
            mask |= STATE_LINE_COMMENT;
        }
        if mask != 0 && self.prev_mask == 0 {
            self.entered_at = self.tokens_seen;
        }
        let quote_code = match self.quote {
            Some(b'"') => 1,
            Some(b'\'') => 2,
            Some(b'`') => 3,
            _ => 0,
        };
        let distance = if mask == 0 {
            0
        } else {
            match self.tokens_seen - self.entered_at + 1 {
                1..=2 => 1,
                3..=8 => 2,
                _ => 3,
            }
        };
        let output = mask | (quote_code << 3) | (distance << 5);
        self.prev_mask = mask;
        self.tokens_seen += 1;
        self.cursor = self.cursor.max(start);
        while self.cursor < end {
            let byte = bytes[self.cursor];
            let next = bytes.get(self.cursor + 1).copied();
            if self.line {
                if byte == b'\n' || byte == b'\r' {
                    self.line = false;
                }
                self.track(byte);
                self.cursor += 1;
            } else if let Some(open) = self.quote {
                if byte == b'\\' {
                    self.track(byte);
                    self.cursor += if next.is_some() { 2 } else { 1 };
                } else {
                    // Closing stays unconditional: the symmetric variant
                    // (skipping mid-word closes) measured worse on both
                    // validation and test, so prose apostrophes still close.
                    if byte == open {
                        self.quote = None;
                    }
                    self.track(byte);
                    self.cursor += 1;
                }
            } else if self.block_depth > 0 {
                if byte == b'*' && next == Some(b'/') {
                    self.block_depth -= 1;
                    self.track(byte);
                    self.cursor += 2;
                } else {
                    self.track(byte);
                    self.cursor += 1;
                }
            } else if byte == b'/' && next == Some(b'/') {
                self.line = true;
                self.track(byte);
                self.cursor += 2;
            } else if byte == b'/' && next == Some(b'*') {
                self.block_depth += 1;
                self.track(byte);
                self.cursor += 2;
            } else if byte == b'#'
                && !self.in_scalar_content()
                && matches!(
                    next,
                    Some(b' ')
                        | Some(b'\t')
                        | Some(b'\n')
                        | Some(b'\r')
                        | Some(b'!')
                        | Some(b'#')
                        | Some(b'$')
                        | None
                )
            {
                self.line = true;
                self.track(byte);
                self.cursor += 1;
            } else if byte == b';' && (self.line_start || self.aligned_trailing(bytes)) {
                // A `;` after two blanks ends an aligned trailing comment
                // (`op  ; comment`); single blanks stay code (`a = 1 ; b = 2`
                // in spaced styles would misfire, so they are left alone).
                self.line = true;
                self.track(byte);
                self.cursor += 1;
            } else if byte == b'-'
                && next == Some(b'-')
                && (self.line_start || self.spaced_trailing(bytes))
            {
                // A `--` wrapped in blanks is a trailing comment
                // (`x -- comment`); `a--` and `x --y` stay operators.
                self.line = true;
                self.track(byte);
                self.cursor += 2;
            } else if byte == b'%'
                && self.line_start
                && matches!(
                    next,
                    Some(b' ')
                        | Some(b'\t')
                        | Some(b'\n')
                        | Some(b'\r')
                        | Some(b'{')
                        | Some(b'}')
                        | Some(b'%')
                        | None
                )
            {
                // A line-start `%` needs a blank, a line break, a Matlab
                // `%{`/`%}` block marker, or an Erlang `%%` after it:
                // `%macro`, `%let`, and Ruby `%x` stay code, as does any
                // mid-line `%` (batch `%%a` loop variables included).
                self.line = true;
                self.track(byte);
                self.cursor += 1;
            } else if byte == b'!'
                && self.line_start
                && matches!(
                    next,
                    Some(b' ')
                        | Some(b'\t')
                        | Some(b'\n')
                        | Some(b'\r')
                        | Some(b'!')
                        | Some(b'>')
                        | None
                )
            {
                self.line = true;
                self.track(byte);
                self.cursor += 1;
            } else {
                if matches!(byte, b'"' | b'`') {
                    self.quote = Some(byte);
                } else if byte == b'\'' {
                    // An apostrophe between word characters is prose, not a
                    // string delimiter: don't, it's, dogs. All other quotes
                    // keep the old behavior, including trailing possessives.
                    let prev_word = self.cursor > 0 && is_word_byte(bytes[self.cursor - 1]);
                    let next_word = next.is_some_and(is_word_byte);
                    if !(prev_word && next_word) {
                        self.quote = Some(byte);
                    }
                } else if matches!(byte, b'|' | b'>') && self.scalar_trigger(bytes) {
                    self.scalar = Some(self.line_indent);
                }
                self.track(byte);
                self.cursor += 1;
            }
        }
        output
    }
}

/// ASCII word byte for the apostrophe rule: letters, digits, underscore.
fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Coalesces adjacent equal labels. Tokens must form a nonempty-length partition
/// starting at byte zero. Without the source, character boundaries cannot be checked.
pub fn merge_spans(tokens: &[Token], labels: &[SyntaxClass]) -> Result<Vec<Span>, CoreError> {
    if tokens.len() != labels.len() {
        return Err(CoreError::LabelCount {
            tokens: tokens.len(),
            labels: labels.len(),
        });
    }
    let mut spans: Vec<Span> = Vec::new();
    let mut end = 0;
    for (index, (token, &class)) in tokens.iter().zip(labels).enumerate() {
        if token.start != end || token.end <= token.start {
            return Err(CoreError::InvalidToken(index));
        }
        end = token.end;
        if let Some(previous) = spans.last_mut().filter(|span| span.class == class) {
            previous.end = end;
        } else {
            spans.push(Span {
                start: token.start,
                end,
                class,
            });
        }
    }
    Ok(spans)
}

fn validate_spans(source: &str, spans: &[Span], complete: bool) -> Result<(), CoreError> {
    let mut end = 0;
    for (index, span) in spans.iter().enumerate() {
        if span.start < end
            || span.end <= span.start
            || span.end > source.len()
            || !source.is_char_boundary(span.start)
            || !source.is_char_boundary(span.end)
        {
            return Err(CoreError::InvalidSpan(index));
        }
        if complete && span.start != end {
            return Err(CoreError::IncompleteCoverage);
        }
        end = span.end;
    }
    if complete && end != source.len() {
        return Err(CoreError::IncompleteCoverage);
    }
    Ok(())
}

/// Converts ordered, non-overlapping byte spans to UTF-16 code-unit offsets.
/// Gaps are allowed, but empty spans and invalid character boundaries are not.
/// Runs in O(source bytes + span count) time without an offset lookup table.
pub fn utf16_spans(source: &str, spans: &[Span]) -> Result<Vec<Span>, CoreError> {
    validate_spans(source, spans, false)?;
    let mut byte_offset = 0;
    let mut utf16_offset = 0;
    let mut output = Vec::with_capacity(spans.len());
    for span in spans {
        utf16_offset += source[byte_offset..span.start]
            .chars()
            .map(char::len_utf16)
            .sum::<usize>();
        let start = utf16_offset;
        utf16_offset += source[span.start..span.end]
            .chars()
            .map(char::len_utf16)
            .sum::<usize>();
        byte_offset = span.end;
        output.push(Span {
            start,
            end: utf16_offset,
            class: span.class,
        });
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    pub version: u32,
    pub id: String,
    pub group: String,
    pub language: String,
    pub source: String,
    pub spans: Vec<Span>,
}

impl Document {
    /// Checks the version, metadata, byte limit, and complete byte-span partition.
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.version != DOCUMENT_VERSION {
            return Err(CoreError::UnsupportedVersion(self.version));
        }
        for (name, value) in [
            ("id", &self.id),
            ("group", &self.group),
            ("language", &self.language),
        ] {
            if value.trim().is_empty() {
                return Err(CoreError::EmptyField(name));
            }
        }
        if self.source.len() > MAX_SOURCE_BYTES {
            return Err(CoreError::SourceTooLarge(self.source.len()));
        }
        validate_spans(&self.source, &self.spans, true)
    }
}

/// Assigns labels only to fully covered non-whitespace tokens with one class.
/// Multiple contiguous spans of the same class may cover a token.
/// Tokens must be ordered and non-overlapping, and spans must come from a
/// validated document. Under these preconditions the scan is O(tokens + spans).
pub fn align_labels(tokens: &[Token], spans: &[Span]) -> Vec<Option<SyntaxClass>> {
    let mut cursor = 0;
    tokens
        .iter()
        .map(|token| {
            if token.is_whitespace() || token.start >= token.end {
                return None;
            }
            while cursor < spans.len() && spans[cursor].end <= token.start {
                cursor += 1;
            }
            let class = spans.get(cursor)?.class;
            let mut covered = token.start;
            let mut consistent = true;
            while covered < token.end {
                let span = spans.get(cursor)?;
                if span.start > covered || span.end <= covered {
                    return None;
                }
                consistent &= span.class == class;
                covered = span.end.min(token.end);
                if span.end <= token.end {
                    cursor += 1;
                }
            }
            consistent.then_some(class)
        })
        .collect()
}
