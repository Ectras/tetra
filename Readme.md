![tests](https://github.com/Ectras/tetra/actions/workflows/test.yml/badge.svg)

# Tetra: Tensor contraction library

A library for `Complex64` tensors, mainly focused on fast contraction.

## Key points
- Shared data ownership using `Arc`
- Zero-cost permutation until raw data access is needed
- Efficient transposition using [hptt](https://github.com/springer13/hptt)
- Optional: Efficient contractions using MKLs `zgemm3m`

### Crate features
- `mkl`: Use 64-bit MKL (otherwise falls back to a pure Rust implementation)
- `serde`: Serialization support
- `rand`: Generation of random tensors

## Example
```rust
use num_complex::Complex64;
use tetra::{all_close, contract, Layout, Tensor};

fn main() {
    // Define the input tensors (a matrix and a vector)
    let a = Tensor::new_from_flat(
        &[3, 2],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(0.0, 5.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(2.0, 0.0),
        ],
        Some(Layout::RowMajor),
    );

    let b = Tensor::new_from_flat(
        &[2],
        vec![Complex64::new(1.0, -1.0), Complex64::new(-2.0, 3.0)],
        None,
    );

    // Perform the contraction
    let out = contract(&[0], &[0, 1], a, &[1], b);

    // Compare against the expected result
    let expected = Tensor::new_from_flat(
        &[3],
        vec![
            Complex64::new(5.0, -7.0),
            Complex64::new(-17.0, -14.0),
            Complex64::new(-5.0, 5.0),
        ],
        None,
    );

    assert!(all_close(&out, &expected, 1e-10));
}
```

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed under the terms of both the Apache License, Version 2.0 and the MIT license without any additional terms or conditions.