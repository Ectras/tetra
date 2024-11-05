use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;
use num_complex::Complex64;
use rand::{
    distributions::{Distribution, Uniform},
    rngs::StdRng,
    SeedableRng,
};
use tetra::{contract, Tensor};

fn random_tensor(shape: &[u64]) -> Tensor {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(-10.0, 10.0);
    let number_elements = Tensor::total_items(shape);
    let data = (0..number_elements)
        .map(|_| Complex64::new(range.sample(&mut rng), range.sample(&mut rng)))
        .collect_vec();
    Tensor::new_from_flat(shape, data, None)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn consecutive_contraction(
    b: Tensor,
    c: Tensor,
    d: Tensor,
    b_indices: &[usize],
    c_indices: &[usize],
    out1_indices: &[usize],
    d_indices: &[usize],
    out2_indices: &[usize],
) -> Tensor {
    let out1 = contract(out1_indices, b_indices, b, c_indices, c);
    contract(out2_indices, d_indices, d, out1_indices, out1)
}

pub fn contraction_benchmark(criterion: &mut Criterion) {
    // Create tensors
    let b = random_tensor(&[4, 4, 3, 10]);
    let c = random_tensor(&[6, 3, 5, 4]);
    let d = random_tensor(&[6, 5, 4]);

    let mut group = criterion.benchmark_group("contractions");
    group.bench_function("contraction", |bench| {
        bench.iter_batched(
            || (b.clone(), c.clone(), d.clone()),
            |(b, c, d)| {
                consecutive_contraction(
                    b,
                    c,
                    d,
                    black_box(&[0, 1, 2, 3]),
                    black_box(&[5, 2, 4, 1]),
                    black_box(&[5, 3, 0, 4]),
                    black_box(&[5, 4, 0]),
                    black_box(&[3]),
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, contraction_benchmark);
criterion_main!(benches);
