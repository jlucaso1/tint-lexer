const TAX: u32 = 7;

struct Receipt {
    subtotal: u32,
    label: String,
}

fn total(receipt: &Receipt) -> u32 {
    // Round down to whole units.
    receipt.subtotal + receipt.subtotal * TAX / 100
}

fn main() {
    let receipt = Receipt {
        subtotal: 240,
        label: String::from("market"),
    };
    println!("{}: {}", receipt.label, total(&receipt));
}
