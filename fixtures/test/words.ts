const MIN_LENGTH = 4;

export function frequencies(text: string): Record<string, number> {
  const result: Record<string, number> = {};
  // Punctuation does not count toward word length.
  for (const word of text.toLowerCase().match(/[a-z]+/g) ?? []) {
    if (word.length < MIN_LENGTH) continue;
    result[word] = (result[word] ?? 0) + 1;
  }
  return result;
}

const sentence = "Small birds circle the small tower.";
for (const [word, count] of Object.entries(frequencies(sentence)).sort()) {
  console.log(`${word} => ${count}`);
}
