#[derive(Debug)]
struct Pulse {
    tick: u64,
    active: bool,
}

fn count_active(pulses: &[Pulse], after: u64) -> usize {
    // The boundary tick belongs to the previous interval.
    pulses.iter().filter(|p| p.active && p.tick > after).count()
}

fn main() {
    let pulses = [
        Pulse { tick: 100, active: true },
        Pulse { tick: 120, active: false },
        Pulse { tick: 140, active: true },
    ];
    println!("active: {}", count_active(&pulses, 110));
}
