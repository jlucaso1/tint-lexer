type Entry = { account: string; cents: number };

function balances(entries: Entry[]): Map<string, number> {
  const result = new Map<string, number>();
  for (const { account, cents } of entries) {
    // Negative entries represent withdrawals.
    result.set(account, (result.get(account) ?? 0) + cents);
  }
  return result;
}

const entries: Entry[] = [
  { account: "travel", cents: 1200 },
  { account: "travel", cents: -450 },
  { account: "food", cents: 800 },
];
for (const [account, cents] of balances(entries)) {
  console.log(`${account}: ${cents / 100}`);
}
