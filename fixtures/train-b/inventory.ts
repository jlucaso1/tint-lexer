export class Inventory {
  items: Record<string, number> = {};

  add(name: string, amount = 1): void {
    if (amount < 0) throw new Error("negative delivery");
    this.items[name] = (this.items[name] ?? 0) + amount;
  }

  describe(): string[] {
    // Sort names so reports have a stable order.
    return Object.keys(this.items).sort().map(name => `${name}: ${this.items[name]}`);
  }
}

const stock = new Inventory();
stock.add("bolts", 16);
stock.add("washers", 32);
console.log(stock.describe().join("\n"));
