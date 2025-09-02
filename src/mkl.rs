use std::ptr;

use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use num_complex::Complex64;

extern crate intel_mkl_src;

mod ffi {
    use std::ffi::{c_int, c_longlong};

    use cblas_sys::{c_double_complex, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

    unsafe extern "C" {
        /// Matrix-matrix multiplication of two complex double matrices. This variant
        /// present in MKL uses less multiplications than the standard BLAS routine
        /// (ZGEMM).
        ///
        /// Reference: <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/cblas-gemm3m.html>.
        ///
        /// The function signature must match the C code for linking to work. For
        /// reference, look at the ZGEMM signature in `cblas-sys`. Note that MKL uses
        /// either 32-bit integers (`lp64`) or 64-bit integers (`ilp64`) for indexing.
        /// The integer types have to be changed accordingly
        /// (`c_int` or `c_longlong`).
        ///
        /// Also note that we can't use the `cblas-sys` interface to link this
        /// function, as the former assumes lp64.
        pub fn cblas_zgemm3m(
            layout: CBLAS_LAYOUT,
            transa: CBLAS_TRANSPOSE,
            transb: CBLAS_TRANSPOSE,
            m: c_longlong,
            n: c_longlong,
            k: c_longlong,
            alpha: *const c_double_complex,
            a: *const c_double_complex,
            lda: c_longlong,
            b: *const c_double_complex,
            ldb: c_longlong,
            beta: *const c_double_complex,
            c: *mut c_double_complex,
            ldc: c_longlong,
        );

        /// Returns the number of threads available to MKL.
        pub fn mkl_get_max_threads() -> c_int;
    }
}

/// Multiplies two complex matrices using MKL.
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
    let mut out = Vec::<Complex64>::with_capacity(a_rows * b_cols);
    let a_rows = a_rows.try_into().unwrap();
    let inner_dim = inner_dim.try_into().unwrap();
    let b_cols = b_cols.try_into().unwrap();
    unsafe {
        ffi::cblas_zgemm3m(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            a_rows,
            b_cols,
            inner_dim,
            ptr::from_ref(&Complex64::ONE).cast(),
            a_data.as_ptr().cast(),
            a_rows,
            b_data.as_ptr().cast(),
            inner_dim,
            ptr::from_ref(&Complex64::ZERO).cast(),
            out.as_mut_ptr().cast(),
            a_rows,
        );

        out.set_len(out.capacity());
    }
    out
}

/// Returns the maximum number of threads available to MKL.
#[inline]
#[must_use]
pub fn max_threads() -> u32 {
    unsafe { ffi::mkl_get_max_threads().try_into().unwrap() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threads_positive() {
        let max_threads = max_threads();
        assert!(max_threads > 0);
    }
}
