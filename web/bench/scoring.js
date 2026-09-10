export const CLASSES = ['plain', 'comment', 'string', 'number', 'keyword', 'type', 'function', 'constant', 'operator'];

export function validateSpans(source, spans, field = 'class') {
  if (typeof source !== 'string' || !Array.isArray(spans)) throw new Error('Invalid source or spans');
  let cursor = 0;
  for (const span of spans) {
    if (!span || span.start !== cursor || !Number.isSafeInteger(span.end) || span.end <= cursor ||
        span.end > source.length || !CLASSES.includes(span[field])) throw new Error('Invalid span coverage or class');
    const code = source.charCodeAt(span.end);
    if (code >= 0xdc00 && code <= 0xdfff && source.charCodeAt(span.end - 1) >= 0xd800 && source.charCodeAt(span.end - 1) <= 0xdbff) {
      throw new Error('Span splits a Unicode code point');
    }
    cursor = span.end;
  }
  if (cursor !== source.length) throw new Error('Incomplete span coverage');
}

export function teacherSpans(source, spans) {
  if (typeof source !== 'string' || !Array.isArray(spans)) throw new Error('Invalid teacher input');
  const wanted = new Set([0]);
  for (const span of spans) {
    if (!span || !Number.isSafeInteger(span.start) || !Number.isSafeInteger(span.end)) throw new Error('Invalid UTF-8 teacher boundary');
    wanted.add(span.start); wanted.add(span.end);
  }
  const boundaries = new Map([[0, 0]]);
  let bytes = 0, units = 0;
  for (const character of source) {
    const code = character.codePointAt(0);
    if (code >= 0xd800 && code <= 0xdfff) throw new Error('Unpaired surrogate');
    bytes += code <= 0x7f ? 1 : code <= 0x7ff ? 2 : code <= 0xffff ? 3 : 4;
    units += character.length;
    if (wanted.has(bytes)) boundaries.set(bytes, units);
  }
  const converted = spans.map(span => {
    if (!span || !boundaries.has(span.start) || !boundaries.has(span.end)) throw new Error('Invalid UTF-8 teacher boundary');
    return { start: boundaries.get(span.start), end: boundaries.get(span.end), class: span.class };
  });
  validateSpans(source, converted);
  return converted;
}

export function emptyScore() {
  return { confusion: CLASSES.map(() => Array(10).fill(0)), excluded_whitespace: 0, excluded_truth_mixed: 0 };
}

export function scoreDocument(source, tokens, truth, predictions, field = 'class') {
  validateSpans(source, truth);
  validateSpans(source, predictions, field);
  if (!tokens || tokens.length % 4 !== 0) throw new Error('Invalid packed tokenizer output');
  const score = emptyScore();
  let start = 0, ti = 0, pi = 0;
  function label(spans, index, end, key) {
    const first = spans[index][key];
    let mixed = false;
    while (spans[index].end < end) {
      index++;
      if (spans[index][key] !== first) mixed = true;
    }
    return [mixed ? 9 : CLASSES.indexOf(first), index];
  }
  for (let i = 0; i < tokens.length; i += 4) {
    const end = tokens[i], kind = tokens[i + 1] & 2047;
    if (!Number.isSafeInteger(end) || end <= start || end > source.length ||
        (end < source.length && /[\uD800-\uDBFF]/.test(source[end - 1]) && /[\uDC00-\uDFFF]/.test(source[end]))) throw new Error('Invalid token boundary');
    while (truth[ti].end <= start) ti++;
    while (predictions[pi].end <= start) pi++;
    let expected, predicted;
    [expected, ti] = label(truth, ti, end, 'class');
    [predicted, pi] = label(predictions, pi, end, field);
    if (kind === 2 || kind === 3) score.excluded_whitespace++;
    else if (expected === 9) score.excluded_truth_mixed++;
    else score.confusion[expected][predicted]++;
    start = end;
  }
  if (start !== source.length) throw new Error('Incomplete tokenizer coverage');
  return score;
}

export function addScore(target, score) {
  for (let row = 0; row < 9; row++) for (let column = 0; column < 10; column++) target.confusion[row][column] += score.confusion[row][column];
  target.excluded_whitespace += score.excluded_whitespace;
  target.excluded_truth_mixed += score.excluded_truth_mixed;
  return target;
}

export function summarizeScore(score) {
  const classes = CLASSES.map((name, i) => {
    const support = score.confusion[i].reduce((a, b) => a + b, 0);
    const predicted = score.confusion.reduce((sum, row) => sum + row[i], 0);
    const tp = score.confusion[i][i];
    return { class: name, support, precision: predicted ? tp / predicted : 0, recall: support ? tp / support : 0,
      f1: support + predicted ? 2 * tp / (support + predicted) : 0 };
  });
  const supported = classes.filter(item => item.support);
  const total = classes.reduce((sum, item) => sum + item.support, 0);
  return { ...score, columns: [...CLASSES, 'mixed'], classes, scored_tokens: total,
    mixed_predictions: score.confusion.reduce((sum, row) => sum + row[9], 0),
    agreement: total ? score.confusion.reduce((sum, row, i) => sum + row[i], 0) / total : null,
    macro_f1: supported.length ? supported.reduce((sum, item) => sum + item.f1, 0) / supported.length : null };
}

export function timingSummary(samples) {
  if (!samples.length || samples.some(value => !Number.isFinite(value) || value < 0)) throw new Error('Invalid timing samples');
  const sorted = [...samples].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return { samples_ms: samples, median_ms: sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2,
    p95_ms: sorted[Math.ceil(sorted.length * 0.95) - 1] };
}
