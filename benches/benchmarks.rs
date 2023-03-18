use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, black_box};
use num_complex::Complex64;
use rand::{
    distributions::{Distribution, Uniform},
    rngs::StdRng,
    SeedableRng,
};
use tetra::{contract, Tensor};

fn random_tensor(shape: &[u32]) -> Tensor {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(-10.0, 10.0);
    let number_elements: usize = shape.iter().product::<u32>().try_into().unwrap();
    let data = (0..number_elements)
        .map(|_| Complex64::new(range.sample(&mut rng), range.sample(&mut rng)))
        .collect::<Vec<_>>();
    Tensor::new_from_flat(shape, data)
}

#[inline]
fn consecutive_contraction(
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
    b_indices: &[u32],
    c_indices: &[u32],
    out1_indices: &[u32],
    d_indices: &[u32],
    out2_indices: &[u32],
) -> Tensor {
    let out1 = contract(out1_indices, b_indices, &b, c_indices, &c);
    let out2 = contract(out2_indices, d_indices, &d, out1_indices, &out1);
    out2
}

pub fn contraction_benchmark(criterion: &mut Criterion) {
    // Create tensors
    let b = random_tensor(&[4, 4, 3, 10]);
    let c = random_tensor(&[6, 3, 5, 4]);
    let d = random_tensor(&[6, 5, 4]);

    let mut group = criterion.benchmark_group("contractions");
    group.measurement_time(Duration::from_secs(10)).sample_size(200);
    group.bench_function("contraction", |bench| {
        bench.iter(|| {
            consecutive_contraction(
                &b,
                &c,
                &d,
                black_box(&[0, 1, 2, 3]),
                black_box(&[5, 2, 4, 1]),
                black_box(&[5, 3, 0, 4]),
                black_box(&[5, 4, 0]),
                black_box(&[3]),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, contraction_benchmark);
criterion_main!(benches);
