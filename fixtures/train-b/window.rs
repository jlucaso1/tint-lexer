fn moving_sum(values: &[i64], width: usize) -> Vec<i64> {
    // A zero-width window has no samples.
    if width == 0 || width > values.len() {
        return Vec::new();
    }
    let mut output = Vec::new();
    for window in values.windows(width) {
        output.push(window.iter().sum());
    }
    output
}

fn main() {
    let samples = [4, -2, 8, 6, 3];
    let sums = moving_sum(&samples, 3);
    for value in sums {
        println!("window={value}");
    }
}
