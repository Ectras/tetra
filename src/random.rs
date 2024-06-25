use rand::{distributions::Standard, Rng};

use crate::Tensor;

/// Generates a random tensor with the given shape from the standard normal
/// distribution.
pub fn random_tensor<R>(shape: &[u64], rng: &mut R) -> Tensor
where
    R: Rng + ?Sized,
{
    let data = rng
        .sample_iter(Standard)
        .take(Tensor::total_items(shape))
        .collect();
    Tensor::new_from_flat(shape, data, None)
}
