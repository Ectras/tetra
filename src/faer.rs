use faer::{linalg::matmul::matmul, Accum, MatMut, MatRef, Par};
use num_complex::Complex64;

/// Multiplies two complex matrices.
#[must_use]
pub fn matrix_matrix_multiplication(
    a_rows: usize,
    inner_dim: usize,
    b_cols: usize,
    a_data: &[Complex64],
    b_data: &[Complex64],
) -> Vec<Complex64> {
    assert_eq!(
        a_data.len(),
        a_rows * inner_dim,
        "Matrix A dimensions don't match data length"
    );
    assert_eq!(
        b_data.len(),
        inner_dim * b_cols,
        "Matrix B dimensions don't match data length"
    );
    let a_mat = MatRef::from_row_major_slice(a_data, a_rows, inner_dim);
    let b_mat = MatRef::from_row_major_slice(b_data, inner_dim, b_cols);
    let mut c = vec![Complex64::ZERO; a_rows * b_cols];
    let c_mat = MatMut::from_row_major_slice_mut(&mut c, a_rows, b_cols);
    matmul(
        c_mat,
        Accum::Replace,
        a_mat,
        b_mat,
        Complex64::ONE,
        Par::Seq,
    );
    c
}
