use itertools::Itertools;
use num_complex::Complex64;
use rand::{distr::StandardUniform, Rng};

use crate::Tensor;

/// Generates a random tensor with the given shape from the standard normal
/// distribution.
pub fn random_tensor<R>(shape: &[usize], rng: &mut R) -> Tensor
where
    R: Rng + ?Sized,
{
    let data = rng
        .sample_iter(StandardUniform)
        .take(2 * Tensor::total_items(shape))
        .tuple_windows()
        .map(|(re, im)| Complex64::new(re, im))
        .collect();
    Tensor::new_from_flat(shape, data, None)
}
