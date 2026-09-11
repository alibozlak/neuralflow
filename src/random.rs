use std::cell::Cell;

/// Seed used until `set_random_seed` is called, so a program gets the same
/// initial weights and the same batch order on every run.
const DEFAULT_SEED: u64 = 42;

thread_local! {
    static STATE: Cell<u64> = const { Cell::new(DEFAULT_SEED) };
}

/// Keras' `keras.utils.set_random_seed`: weight initialisation and shuffling
/// on this thread become reproducible from `seed`.
pub fn set_random_seed(seed: u64) {
    STATE.with(|state| state.set(seed));
}

/// SplitMix64: tiny and good enough for initial weights and batch order.
/// Not for cryptography.
fn next_u64() -> u64 {
    STATE.with(|state| {
        let next = state.get().wrapping_add(0x9E37_79B9_7F4A_7C15);
        state.set(next);

        let mut z = next;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    })
}

/// A uniform sample from [low, high).
pub(crate) fn uniform(low: f64, high: f64) -> f64 {
    // The top 53 bits fill an f64 mantissa exactly.
    let unit = (next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    low + (high - low) * unit
}

/// Fisher–Yates shuffle.
pub(crate) fn shuffle<T>(items: &mut [T]) {
    for i in (1..items.len()).rev() {
        let j = (next_u64() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_numbers() {
        set_random_seed(7);
        let first: Vec<f64> = (0..5).map(|_| uniform(-1., 1.)).collect();
        set_random_seed(7);
        let second: Vec<f64> = (0..5).map(|_| uniform(-1., 1.)).collect();

        assert_eq!(first, second);
        assert!(first.iter().all(|&value| (-1. ..1.).contains(&value)));
    }

    #[test]
    fn shuffle_keeps_every_item() {
        let mut items: Vec<usize> = (0..100).collect();
        shuffle(&mut items);
        assert_ne!(items, (0..100).collect::<Vec<_>>());

        items.sort();
        assert_eq!(items, (0..100).collect::<Vec<_>>());
    }
}
