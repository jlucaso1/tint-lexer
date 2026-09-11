#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
#[global_allocator]
static ALLOCATOR: talc::wasm::WasmDynamicTalc = talc::wasm::new_wasm_dynamic_allocator();

pub const MAX_SOURCE_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_TOKENS: usize = 4 * 1024 * 1024;

pub fn pack_into(source: &str, packed: &mut Vec<u32>) -> Result<(), &'static str> {
    if source.len() > MAX_SOURCE_BYTES {
        return Err("Source exceeds 16 MiB.");
    }
    packed.clear();
    packed.reserve((source.len() / 4 * 4).min(MAX_TOKENS * 4));
    let ascii = source.is_ascii();
    let mut end = 0u32;
    let mut scanner = tint_core::StateScanner::new();
    for (index, token) in tint_core::iter_tokens(source).enumerate() {
        if index == MAX_TOKENS {
            return Err("Source exceeds 4194304 tokens.");
        }
        if ascii {
            end = token.end as u32;
        } else {
            end += source[token.start..token.end].encode_utf16().count() as u32;
        }
        packed.push(end);
        let state = scanner.advance(source, token.start, token.end);
        debug_assert!(state < 128);
        for (pair_index, pair) in token.features.as_chunks::<2>().0.iter().enumerate() {
            let mut word = pair[0] as u32 | ((pair[1] as u32) << 11);
            if pair_index == 0 {
                word |= (state as u32) << 22;
            }
            packed.push(word);
        }
    }
    Ok(())
}

pub fn pack(source: &str) -> Result<Vec<u32>, &'static str> {
    let mut packed = Vec::with_capacity((source.len() / 4 * 4).min(MAX_TOKENS * 4));
    pack_into(source, &mut packed)?;
    Ok(packed)
}

pub const RADIUS: usize = 4;
pub const DIM: usize = 6;
pub const HIDDEN: usize = 64;
pub const BASE: usize = (RADIUS * 2 + 1) * DIM; // 54
pub const HW: usize = 1365 * DIM; // 8190
pub const HB: usize = HW + (BASE + 9) * HIDDEN; // 12222
pub const OW: usize = HB + HIDDEN; // 12286
pub const OB: usize = OW + HIDDEN * 9; // 12862
pub const WEIGHTS_COUNT: usize = OB + 9; // 12871

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn simd_tanh7(x: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;
    let c135135 = f32x4_splat(135135.0);
    let c17325 = f32x4_splat(17325.0);
    let c378 = f32x4_splat(378.0);
    let c62370 = f32x4_splat(62370.0);
    let c3150 = f32x4_splat(3150.0);
    let c28 = f32x4_splat(28.0);
    let c_pos1 = f32x4_splat(1.0);
    let c_neg1 = f32x4_splat(-1.0);

    let x2 = f32x4_mul(x, x);
    let num_inner = f32x4_add(c378, x2);
    let num_mid = f32x4_add(c17325, f32x4_mul(x2, num_inner));
    let num = f32x4_mul(x, f32x4_add(c135135, f32x4_mul(x2, num_mid)));

    let den_inner = f32x4_add(c3150, f32x4_mul(c28, x2));
    let den_mid = f32x4_add(c62370, f32x4_mul(x2, den_inner));
    let den = f32x4_add(c135135, f32x4_mul(x2, den_mid));

    let res = f32x4_div(num, den);
    f32x4_min(f32x4_max(res, c_neg1), c_pos1)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn infer_cpu_into(tokens: &[u32], weights: &[f32], labels: &mut Vec<u8>) {
    use core::arch::wasm32::*;
    let count = tokens.len() / 4;
    labels.clear();
    if count == 0 || weights.len() < WEIGHTS_COUNT {
        return;
    }
    labels.reserve(count);
    unsafe {
        labels.set_len(count);
    }

    let total_e = count + 2 * RADIUS;
    let mut e: Vec<f32> = Vec::with_capacity(total_e * DIM);
    unsafe {
        e.set_len(total_e * DIM);
    }

    let scale6 = f32x4_splat(0.16666667);
    for i in 0..total_e {
        let t = i as isize - RADIUS as isize;
        let e_offset = i * DIM;
        if t >= 0 && (t as usize) < count {
            let off = (t as usize) * 4;
            let w1 = tokens[off + 1];
            let w2 = tokens[off + 2];
            let w3 = tokens[off + 3];

            let id0 = ((w1 & 2047) as usize) * DIM;
            let id1 = (((w1 >> 11) & 2047) as usize) * DIM;
            let id2 = ((w2 & 2047) as usize) * DIM;
            let id3 = (((w2 >> 11) & 2047) as usize) * DIM;
            let id4 = ((w3 & 2047) as usize) * DIM;
            let id5 = (((w3 >> 11) & 2047) as usize) * DIM;

            unsafe {
                let v0 = v128_load(weights.as_ptr().add(id0) as *const v128);
                let v1 = v128_load(weights.as_ptr().add(id1) as *const v128);
                let v2 = v128_load(weights.as_ptr().add(id2) as *const v128);
                let v3 = v128_load(weights.as_ptr().add(id3) as *const v128);
                let v4 = v128_load(weights.as_ptr().add(id4) as *const v128);
                let v5 = v128_load(weights.as_ptr().add(id5) as *const v128);
                let s0 = f32x4_add(
                    f32x4_add(v0, v1),
                    f32x4_add(f32x4_add(v2, v3), f32x4_add(v4, v5)),
                );
                v128_store(
                    e.as_mut_ptr().add(e_offset) as *mut v128,
                    f32x4_mul(s0, scale6),
                );
            }
            e[e_offset + 4] = (weights[id0 + 4]
                + weights[id1 + 4]
                + weights[id2 + 4]
                + weights[id3 + 4]
                + weights[id4 + 4]
                + weights[id5 + 4])
                * 0.16666667;
            e[e_offset + 5] = (weights[id0 + 5]
                + weights[id1 + 5]
                + weights[id2 + 5]
                + weights[id3 + 5]
                + weights[id4 + 5]
                + weights[id5 + 5])
                * 0.16666667;
        } else {
            for d in 0..DIM {
                e[e_offset + d] = weights[d];
            }
        }
    }

    let out_bias = &weights[OB..OB + 9];

    // Pre-transpose output weights (9 x 64) for register dot-products
    let mut w_out_t = [0.0f32; 9 * 64];
    for d in 0..HIDDEN {
        for c in 0..9 {
            w_out_t[c * HIDDEN + d] = weights[OW + d * 9 + c];
        }
    }

    for t in 0..count {
        let off = t * 4;
        let w1 = tokens[off + 1];
        let kind = w1 & 2047;
        if kind == 2 || kind == 3 {
            labels[t] = 0;
            continue;
        }

        unsafe {
            let hb_ptr = weights.as_ptr().add(HB) as *const v128;
            let mut h0 = v128_load(hb_ptr.add(0));
            let mut h1 = v128_load(hb_ptr.add(1));
            let mut h2 = v128_load(hb_ptr.add(2));
            let mut h3 = v128_load(hb_ptr.add(3));
            let mut h4 = v128_load(hb_ptr.add(4));
            let mut h5 = v128_load(hb_ptr.add(5));
            let mut h6 = v128_load(hb_ptr.add(6));
            let mut h7 = v128_load(hb_ptr.add(7));
            let mut h8 = v128_load(hb_ptr.add(8));
            let mut h9 = v128_load(hb_ptr.add(9));
            let mut h10 = v128_load(hb_ptr.add(10));
            let mut h11 = v128_load(hb_ptr.add(11));
            let mut h12 = v128_load(hb_ptr.add(12));
            let mut h13 = v128_load(hb_ptr.add(13));
            let mut h14 = v128_load(hb_ptr.add(14));
            let mut h15 = v128_load(hb_ptr.add(15));

            let e_start = t * DIM;
            let e_window = &e[e_start..e_start + 54];
            for k in 0..54 {
                let ek = e_window[k];
                let ek4 = f32x4_splat(ek);
                let w_ptr = weights.as_ptr().add(HW + k * 64) as *const v128;
                h0 = f32x4_add(h0, f32x4_mul(ek4, v128_load(w_ptr.add(0))));
                h1 = f32x4_add(h1, f32x4_mul(ek4, v128_load(w_ptr.add(1))));
                h2 = f32x4_add(h2, f32x4_mul(ek4, v128_load(w_ptr.add(2))));
                h3 = f32x4_add(h3, f32x4_mul(ek4, v128_load(w_ptr.add(3))));
                h4 = f32x4_add(h4, f32x4_mul(ek4, v128_load(w_ptr.add(4))));
                h5 = f32x4_add(h5, f32x4_mul(ek4, v128_load(w_ptr.add(5))));
                h6 = f32x4_add(h6, f32x4_mul(ek4, v128_load(w_ptr.add(6))));
                h7 = f32x4_add(h7, f32x4_mul(ek4, v128_load(w_ptr.add(7))));
                h8 = f32x4_add(h8, f32x4_mul(ek4, v128_load(w_ptr.add(8))));
                h9 = f32x4_add(h9, f32x4_mul(ek4, v128_load(w_ptr.add(9))));
                h10 = f32x4_add(h10, f32x4_mul(ek4, v128_load(w_ptr.add(10))));
                h11 = f32x4_add(h11, f32x4_mul(ek4, v128_load(w_ptr.add(11))));
                h12 = f32x4_add(h12, f32x4_mul(ek4, v128_load(w_ptr.add(12))));
                h13 = f32x4_add(h13, f32x4_mul(ek4, v128_load(w_ptr.add(13))));
                h14 = f32x4_add(h14, f32x4_mul(ek4, v128_load(w_ptr.add(14))));
                h15 = f32x4_add(h15, f32x4_mul(ek4, v128_load(w_ptr.add(15))));
            }

            let cst = (w1 >> 22) & 7;
            let q = (w1 >> 25) & 3;
            let r = (w1 >> 27) & 3;
            let pst = if t > 0 {
                (tokens[(t - 1) * 4 + 1] >> 22) & 7
            } else {
                0
            };
            let nxt = if t + 1 < count {
                (tokens[(t + 1) * 4 + 1] >> 22) & 7
            } else {
                0
            };

            macro_rules! add_state_row {
                ($row_idx:expr) => {
                    let w_ptr = weights.as_ptr().add(HW + $row_idx * 64) as *const v128;
                    h0 = f32x4_add(h0, v128_load(w_ptr.add(0)));
                    h1 = f32x4_add(h1, v128_load(w_ptr.add(1)));
                    h2 = f32x4_add(h2, v128_load(w_ptr.add(2)));
                    h3 = f32x4_add(h3, v128_load(w_ptr.add(3)));
                    h4 = f32x4_add(h4, v128_load(w_ptr.add(4)));
                    h5 = f32x4_add(h5, v128_load(w_ptr.add(5)));
                    h6 = f32x4_add(h6, v128_load(w_ptr.add(6)));
                    h7 = f32x4_add(h7, v128_load(w_ptr.add(7)));
                    h8 = f32x4_add(h8, v128_load(w_ptr.add(8)));
                    h9 = f32x4_add(h9, v128_load(w_ptr.add(9)));
                    h10 = f32x4_add(h10, v128_load(w_ptr.add(10)));
                    h11 = f32x4_add(h11, v128_load(w_ptr.add(11)));
                    h12 = f32x4_add(h12, v128_load(w_ptr.add(12)));
                    h13 = f32x4_add(h13, v128_load(w_ptr.add(13)));
                    h14 = f32x4_add(h14, v128_load(w_ptr.add(14)));
                    h15 = f32x4_add(h15, v128_load(w_ptr.add(15)));
                };
            }

            if (cst & 1) != 0 {
                add_state_row!(54);
            }
            if ((cst >> 1) & 1) != 0 {
                add_state_row!(55);
            }
            if ((cst >> 2) & 1) != 0 {
                add_state_row!(56);
            }
            if cst != pst {
                add_state_row!(57);
            }
            if cst != nxt {
                add_state_row!(58);
            }
            if (q & 1) != 0 {
                add_state_row!(59);
            }
            if ((q >> 1) & 1) != 0 {
                add_state_row!(60);
            }
            if (r & 1) != 0 {
                add_state_row!(61);
            }
            if ((r >> 1) & 1) != 0 {
                add_state_row!(62);
            }

            // Activation: Order 7 Padé tanh
            h0 = simd_tanh7(h0);
            h1 = simd_tanh7(h1);
            h2 = simd_tanh7(h2);
            h3 = simd_tanh7(h3);
            h4 = simd_tanh7(h4);
            h5 = simd_tanh7(h5);
            h6 = simd_tanh7(h6);
            h7 = simd_tanh7(h7);
            h8 = simd_tanh7(h8);
            h9 = simd_tanh7(h9);
            h10 = simd_tanh7(h10);
            h11 = simd_tanh7(h11);
            h12 = simd_tanh7(h12);
            h13 = simd_tanh7(h13);
            h14 = simd_tanh7(h14);
            h15 = simd_tanh7(h15);

            // Output classification directly via SIMD register dot-products
            let mut best = 0u8;
            let mut max = f32::NEG_INFINITY;

            for c in 0..9 {
                let wt = w_out_t.as_ptr().add(c * HIDDEN) as *const v128;
                let mut acc = f32x4_mul(h0, v128_load(wt.add(0)));
                acc = f32x4_add(acc, f32x4_mul(h1, v128_load(wt.add(1))));
                acc = f32x4_add(acc, f32x4_mul(h2, v128_load(wt.add(2))));
                acc = f32x4_add(acc, f32x4_mul(h3, v128_load(wt.add(3))));
                acc = f32x4_add(acc, f32x4_mul(h4, v128_load(wt.add(4))));
                acc = f32x4_add(acc, f32x4_mul(h5, v128_load(wt.add(5))));
                acc = f32x4_add(acc, f32x4_mul(h6, v128_load(wt.add(6))));
                acc = f32x4_add(acc, f32x4_mul(h7, v128_load(wt.add(7))));
                acc = f32x4_add(acc, f32x4_mul(h8, v128_load(wt.add(8))));
                acc = f32x4_add(acc, f32x4_mul(h9, v128_load(wt.add(9))));
                acc = f32x4_add(acc, f32x4_mul(h10, v128_load(wt.add(10))));
                acc = f32x4_add(acc, f32x4_mul(h11, v128_load(wt.add(11))));
                acc = f32x4_add(acc, f32x4_mul(h12, v128_load(wt.add(12))));
                acc = f32x4_add(acc, f32x4_mul(h13, v128_load(wt.add(13))));
                acc = f32x4_add(acc, f32x4_mul(h14, v128_load(wt.add(14))));
                acc = f32x4_add(acc, f32x4_mul(h15, v128_load(wt.add(15))));

                let score = f32x4_extract_lane::<0>(acc)
                    + f32x4_extract_lane::<1>(acc)
                    + f32x4_extract_lane::<2>(acc)
                    + f32x4_extract_lane::<3>(acc)
                    + out_bias[c];

                if score > max {
                    max = score;
                    best = c as u8;
                }
            }

            labels[t] = best;
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[allow(clippy::needless_range_loop, clippy::manual_memcpy)]
pub fn infer_cpu_into(tokens: &[u32], weights: &[f32], labels: &mut Vec<u8>) {
    let count = tokens.len() / 4;
    labels.clear();
    if count == 0 || weights.len() < WEIGHTS_COUNT {
        return;
    }
    labels.resize(count, 0);

    // Compute embeddings e for (count + 2 * RADIUS) tokens
    let total_e = count + 2 * RADIUS;
    let mut e = vec![0.0f32; total_e * DIM];

    for i in 0..total_e {
        let t = i as isize - RADIUS as isize;
        let e_offset = i * DIM;
        if t >= 0 && (t as usize) < count {
            let off = (t as usize) * 4;
            let w1 = tokens[off + 1];
            let w2 = tokens[off + 2];
            let w3 = tokens[off + 3];

            let id0 = ((w1 & 2047) as usize) * DIM;
            let id1 = (((w1 >> 11) & 2047) as usize) * DIM;
            let id2 = ((w2 & 2047) as usize) * DIM;
            let id3 = (((w2 >> 11) & 2047) as usize) * DIM;
            let id4 = ((w3 & 2047) as usize) * DIM;
            let id5 = (((w3 >> 11) & 2047) as usize) * DIM;

            for d in 0..DIM {
                e[e_offset + d] = (weights[id0 + d]
                    + weights[id1 + d]
                    + weights[id2 + d]
                    + weights[id3 + d]
                    + weights[id4 + d]
                    + weights[id5 + d])
                    * 0.16666667;
            }
        } else {
            for d in 0..DIM {
                e[e_offset + d] = weights[d];
            }
        }
    }

    let mut h_acc = [0.0f32; HIDDEN];
    let mut h = [0.0f32; HIDDEN];

    let hidden_bias = &weights[HB..HB + HIDDEN];
    let out_bias = &weights[OB..OB + 9];

    for t in 0..count {
        let off = t * 4;
        let w1 = tokens[off + 1];
        let kind = w1 & 2047;
        if kind == 2 || kind == 3 {
            labels[t] = 0;
            continue;
        }

        h_acc.copy_from_slice(hidden_bias);

        let e_start = t * DIM;
        let e_window = &e[e_start..e_start + 54];
        for k in 0..54 {
            let ek = e_window[k];
            if ek != 0.0 {
                let w_off = HW + k * HIDDEN;
                let w_row = &weights[w_off..w_off + HIDDEN];
                for d in 0..HIDDEN {
                    h_acc[d] += ek * w_row[d];
                }
            }
        }

        let cst = (w1 >> 22) & 7;
        let q = (w1 >> 25) & 3;
        let r = (w1 >> 27) & 3;
        let pst = if t > 0 {
            (tokens[(t - 1) * 4 + 1] >> 22) & 7
        } else {
            0
        };
        let nxt = if t + 1 < count {
            (tokens[(t + 1) * 4 + 1] >> 22) & 7
        } else {
            0
        };

        if (cst & 1) != 0 {
            let w_row = &weights[HW + 54 * HIDDEN..HW + 55 * HIDDEN];
            for d in 0..HIDDEN {
                h_acc[d] += w_row[d];
            }
        }
        if ((cst >> 1) & 1) != 0 {
            let w_row = &weights[HW + 55 * HIDDEN..HW + 56 * HIDDEN];
            for d in 0..HIDDEN {
                h_acc[d] += w_row[d];
            }
        }
        if ((cst >> 2) & 1) != 0 {
            let w_row = &weights[HW + 56 * HIDDEN..HW + 57 * HIDDEN];
            for d in 0..HIDDEN {
                h_acc[d] += w_row[d];
            }
        }
        if cst != pst {
            let w_row = &weights[HW + 57 * HIDDEN..HW + 58 * HIDDEN];
            for d in 0..HIDDEN {
                h_acc[d] += w_row[d];
            }
        }
        if cst != nxt {
            let w_row = &weights[HW + 58 * HIDDEN..HW + 59 * HIDDEN];
            for d in 0..HIDDEN {
                h_acc[d] += w_row[d];
            }
        }
        if (q & 1) != 0 {
            let w_row = &weights[HW + 59 * HIDDEN..HW + 60 * HIDDEN];
            for d in 0..HIDDEN {
                h_acc[d] += w_row[d];
            }
        }
        if ((q >> 1) & 1) != 0 {
            let w_row = &weights[HW + 60 * HIDDEN..HW + 61 * HIDDEN];
            for d in 0..HIDDEN {
                h_acc[d] += w_row[d];
            }
        }
        if (r & 1) != 0 {
            let w_row = &weights[HW + 61 * HIDDEN..HW + 62 * HIDDEN];
            for d in 0..HIDDEN {
                h_acc[d] += w_row[d];
            }
        }
        if ((r >> 1) & 1) != 0 {
            let w_row = &weights[HW + 62 * HIDDEN..HW + 63 * HIDDEN];
            for d in 0..HIDDEN {
                h_acc[d] += w_row[d];
            }
        }

        for d in 0..HIDDEN {
            h[d] = h_acc[d].tanh();
        }

        let mut scores = [0.0f32; 9];
        scores.copy_from_slice(out_bias);

        for d in 0..HIDDEN {
            let hd = h[d];
            let row = &weights[OW + d * 9..OW + (d + 1) * 9];
            for c in 0..9 {
                scores[c] += hd * row[c];
            }
        }

        let mut best = 0u8;
        let mut max = scores[0];
        for c in 1..9 {
            if scores[c] > max {
                max = scores[c];
                best = c as u8;
            }
        }

        labels[t] = best;
    }
}

pub fn infer_cpu(tokens: &[u32], weights: &[f32]) -> Vec<u8> {
    let mut labels = Vec::new();
    infer_cpu_into(tokens, weights, &mut labels);
    labels
}

#[allow(dead_code)]
struct State {
    tokens: Vec<u32>,
    labels: Vec<u8>,
    weights: Vec<f32>,
}

#[cfg(target_arch = "wasm32")]
struct StateCell {
    inner: core::cell::UnsafeCell<State>,
}

// SAFETY: wasm32 without atomics is single-threaded, and the entry points
// below never re-enter the state (the raw ABI makes no JS calls at all).
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for StateCell {}

#[cfg(target_arch = "wasm32")]
static STATE: StateCell = StateCell {
    inner: core::cell::UnsafeCell::new(State {
        tokens: Vec::new(),
        labels: Vec::new(),
        weights: Vec::new(),
    }),
};

#[cfg(target_arch = "wasm32")]
fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    // SAFETY: see the Sync impl above.
    unsafe { f(&mut *STATE.inner.get()) }
}

// Raw numeric ABI consumed by the npm bundle. The module has zero imports:
// errors are negative codes so the JS glue needs no JsValue machinery.
#[cfg(target_arch = "wasm32")]
const ERR_TOO_BIG: i32 = -1;
#[cfg(target_arch = "wasm32")]
const ERR_TOO_MANY_TOKENS: i32 = -2;
#[cfg(target_arch = "wasm32")]
const ERR_INVALID_UTF8: i32 = -3;

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
/// # Safety
/// `size`/`align` must describe a layout the caller owns via a previous
/// `tint_malloc`/`tint_realloc` result when calling `tint_free`/`tint_realloc`.
pub unsafe extern "C" fn tint_malloc(size: usize, align: usize) -> *mut u8 {
    let Ok(layout) = core::alloc::Layout::from_size_align(size, align) else {
        return core::ptr::null_mut();
    };
    // SAFETY: layout is valid; ownership moves to the caller.
    unsafe { std::alloc::alloc(layout) }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
/// # Safety
/// Same contract as `tint_malloc`; `ptr`/`old_size`/`align` must match a live
/// allocation from `tint_malloc`/`tint_realloc`.
pub unsafe extern "C" fn tint_realloc(
    ptr: *mut u8,
    old_size: usize,
    align: usize,
    new_size: usize,
) -> *mut u8 {
    let (Ok(old), Ok(new)) = (
        core::alloc::Layout::from_size_align(old_size, align),
        core::alloc::Layout::from_size_align(new_size, align),
    ) else {
        return core::ptr::null_mut();
    };
    // SAFETY: ptr is a live allocation with the `old` layout.
    unsafe { std::alloc::realloc(ptr, old, new.size()) }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
/// # Safety
/// Same contract as `tint_malloc`.
pub unsafe extern "C" fn tint_free(ptr: *mut u8, size: usize, align: usize) {
    if ptr.is_null() {
        return;
    }
    if let Ok(layout) = core::alloc::Layout::from_size_align(size, align) {
        // SAFETY: ptr is a live allocation with this layout.
        unsafe { std::alloc::dealloc(ptr, layout) }
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
/// # Safety
/// `ptr` must point to `len` readable `f32` values.
pub unsafe extern "C" fn tint_set_weights(ptr: *const f32, len: usize) {
    // SAFETY: guaranteed by the caller (JS glue writes the buffer first).
    let src = unsafe { core::slice::from_raw_parts(ptr, len) };
    with_state(|state| {
        state.weights.clear();
        state.weights.extend_from_slice(src);
    });
}

#[cfg(target_arch = "wasm32")]
fn utf8_source(src_ptr: *const u8, src_len: usize) -> Result<&'static str, i32> {
    if src_len > MAX_SOURCE_BYTES {
        return Err(ERR_TOO_BIG);
    }
    // SAFETY: the byte range is readable; validity is checked below.
    let bytes = unsafe { core::slice::from_raw_parts(src_ptr, src_len) };
    core::str::from_utf8(bytes).map_err(|_| ERR_INVALID_UTF8)
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
/// # Safety
/// `src_ptr` must point to `src_len` readable bytes. Returns the token count,
/// or a negative `ERR_*` code.
pub unsafe extern "C" fn tint_tokenize(src_ptr: *const u8, src_len: usize) -> i32 {
    let src = match utf8_source(src_ptr, src_len) {
        Ok(src) => src,
        Err(code) => return code,
    };
    with_state(|state| match pack_into(src, &mut state.tokens) {
        Ok(()) => (state.tokens.len() / 4) as i32,
        // The byte cap is pre-checked above, so only the token cap can fail.
        Err(_) => ERR_TOO_MANY_TOKENS,
    })
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
/// # Safety
/// Same contract as `tint_tokenize`.
pub unsafe extern "C" fn tint_tokenize_and_infer(src_ptr: *const u8, src_len: usize) -> i32 {
    let src = match utf8_source(src_ptr, src_len) {
        Ok(src) => src,
        Err(code) => return code,
    };
    with_state(|state| {
        if pack_into(src, &mut state.tokens).is_err() {
            return ERR_TOO_MANY_TOKENS;
        }
        let count = state.tokens.len() / 4;
        infer_cpu_into(&state.tokens, &state.weights, &mut state.labels);
        count as i32
    })
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub fn tint_get_tokens_ptr() -> *const u32 {
    with_state(|state| state.tokens.as_ptr())
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub fn tint_get_labels_ptr() -> *const u8 {
    with_state(|state| state.labels.as_ptr())
}

// Legacy wasm-bindgen ABI for the gitignored compact playground
// (`web/compact/pkg`, rebuilt with `--features compat`). Untouched behavior.
#[cfg(all(target_arch = "wasm32", feature = "compat"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_weights(w: &[f32]) {
    with_state(|state| {
        state.weights.clear();
        state.weights.extend_from_slice(w);
    });
}

#[cfg(all(target_arch = "wasm32", feature = "compat"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tokenize_and_infer(source: &str) -> Result<usize, wasm_bindgen::JsValue> {
    with_state(|state| {
        let State {
            tokens,
            labels,
            weights,
        } = state;
        pack_into(source, tokens).map_err(wasm_bindgen::JsValue::from_str)?;
        let count = tokens.len() / 4;
        infer_cpu_into(tokens, weights, labels);
        Ok(count)
    })
}

#[cfg(all(target_arch = "wasm32", feature = "compat"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn get_tokens_ptr() -> *const u32 {
    with_state(|state| state.tokens.as_ptr())
}

#[cfg(all(target_arch = "wasm32", feature = "compat"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn get_labels_ptr() -> *const u8 {
    with_state(|state| state.labels.as_ptr())
}

#[cfg(all(target_arch = "wasm32", feature = "compat"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tokenize(source: &str) -> Result<Vec<u32>, wasm_bindgen::JsValue> {
    pack(source).map_err(wasm_bindgen::JsValue::from_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_features_and_utf16_ends_match_core() {
        for source in [
            "",
            "fn main() { 42 }",
            "\r\n\r\n\n\r",
            "a\u{1f600}e\u{301}\t\u{2003}b\r\n+",
        ] {
            let packed = pack(source).unwrap();
            let tokens = tint_core::tokenize(source);
            assert_eq!(packed.len(), tokens.len() * 4);
            for (row, token) in packed.as_chunks::<4>().0.iter().zip(tokens) {
                assert_eq!(row[0] as usize, source[..token.end].encode_utf16().count());
                for index in 0..6 {
                    assert_eq!(
                        (row[1 + index / 2] >> ((index % 2) * 11)) & 2047,
                        token.features[index] as u32
                    );
                }
            }
        }
    }

    #[test]
    fn ascii_fast_path_matches_slow_reference() {
        fn slow_pack(source: &str) -> Vec<u32> {
            let mut packed = Vec::new();
            let mut end = 0u32;
            let mut scanner = tint_core::StateScanner::new();
            for token in tint_core::iter_tokens(source) {
                end += source[token.start..token.end].encode_utf16().count() as u32;
                packed.push(end);
                let state = scanner.advance(source, token.start, token.end);
                for (pair_index, pair) in token.features.as_chunks::<2>().0.iter().enumerate() {
                    let mut word = pair[0] as u32 | ((pair[1] as u32) << 11);
                    if pair_index == 0 {
                        word |= (state as u32) << 22;
                    }
                    packed.push(word);
                }
            }
            packed
        }
        let repeated = "word + \"str\" // cmt\r\n".repeat(512);
        let ascii_all = (0u8..128).map(|byte| byte as char).collect::<String>();
        for source in [
            "",
            "a",
            "fn main() { 42 }",
            "\r\n\r\n\n\r",
            " \t\u{0B}\u{0C}",
            "AaZz09_ +/*\"'`//",
            "/* ab */ \"cd\" // ef",
            "a\u{1f600}e\u{301}\t\u{2003}b\r\n+",
            "\u{1f600}\u{1f600}\u{2003}",
            repeated.as_str(),
            ascii_all.as_str(),
        ] {
            assert_eq!(pack(source).unwrap(), slow_pack(source), "{source:?}");
            if source.is_ascii() {
                for (row, token) in pack(source)
                    .unwrap()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(tint_core::tokenize(source))
                {
                    assert_eq!(row[0], token.end as u32);
                }
            }
        }
    }

    #[test]
    fn state_bits_ride_spare_packing_bits() {
        let source = "/* ab */ \"cd\" // ef";
        let packed = pack(source).unwrap();
        let states: Vec<u32> = packed
            .as_chunks::<4>()
            .0
            .iter()
            .map(|row| (row[1] >> 22) & 127)
            .collect();
        let tokens = tint_core::tokenize(source);
        assert_eq!(states.len(), tokens.len());
        for (row, token) in packed.as_chunks::<4>().0.iter().zip(&tokens) {
            for index in 0..6 {
                assert_eq!(
                    (row[1 + index / 2] >> ((index % 2) * 11)) & 2047,
                    token.features[index] as u32
                );
            }
        }
        let at = |text: &str| {
            let tokens = tint_core::tokenize(source);
            let index = tokens
                .iter()
                .position(|token| &source[token.start..token.end] == text)
                .unwrap();
            states[index]
        };
        assert_eq!(at("ab") & 7, tint_core::STATE_BLOCK_COMMENT as u32);
        assert_eq!(at("cd") & 7, tint_core::STATE_STRING as u32);
        assert_eq!(at("ef") & 7, tint_core::STATE_LINE_COMMENT as u32);
        assert_eq!((at("cd") >> 3) & 3, 1);
        assert!((at("cd") >> 5) & 3 >= 1);
    }

    #[test]
    fn quote_distance_bytes_match_hand_trace() {
        let packed = pack("\"a\"").unwrap();
        let states: Vec<u32> = packed
            .as_chunks::<4>()
            .0
            .iter()
            .map(|row| (row[1] >> 22) & 127)
            .collect();
        assert_eq!(
            states,
            vec![0, 1 | (1 << 3) | (1 << 5), 1 | (1 << 3) | (1 << 5)]
        );
    }

    #[test]
    fn rejects_source_and_token_limits() {
        assert_eq!(
            pack(&"a".repeat(MAX_SOURCE_BYTES + 1)),
            Err("Source exceeds 16 MiB.")
        );
        assert_eq!(
            pack(&"+".repeat(MAX_TOKENS + 1)),
            Err("Source exceeds 4194304 tokens.")
        );
    }
}
