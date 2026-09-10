const LIMIT = 18.5;

/** Count samples above the threshold. */
export function summarize(readings: number[]): string {
  let count = 0;
  for (const reading of readings) {
    if (reading > LIMIT) count += 1;
  }
  if (readings.length === 0) return "no samples";
  const ratio = count / readings.length;
  return `warm fraction: ${ratio.toFixed(2)}`;
}

// Samples are already calibrated.
const samples = [17.0, 19.5, 21.25];
console.log(summarize(samples));
