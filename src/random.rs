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
        .chunks(2)
        .into_iter()
        .map(|chunk| {
            let [re, im] = chunk.collect_array().unwrap();
            Complex64::new(re, im)
        })
        .collect();
    Tensor::new_from_flat(shape, data, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rand_tensor() {
        let mut rng = rand::rng();
        let tensor = random_tensor(&[5, 1, 4], &mut rng);
        assert_eq!(tensor.len(), 20);
    }
}
