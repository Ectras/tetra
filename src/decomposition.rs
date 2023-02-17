use crate::Tensor;
extern crate openblas_src;
use lapack::{zgeqp3, zgesdd, zungqr};
use num_complex::{Complex64, ComplexFloat};
use std::cmp::{max, min};

pub trait Decomposition {
    fn qr(&mut self) -> (Tensor, Tensor);
    fn svd(&mut self) -> (Tensor, Tensor, Tensor);
}

impl Decomposition for Tensor {
    /// Implements QR decomposition. Returns (Q,R) where Q is a Hermitian matrix and R is an upper right triangular matrix
    fn qr(&mut self) -> (Tensor, Tensor) {
        assert!(self.ndim() == 2, "Only able to decompose matrices");
        // Get shape of input Tensor
        let m = self.shape[0];
        let n = self.shape[1];
        let min_dim = min(m, n) as usize;

        // Leading dimension of `self`
        let lda = max(1, m);

        let mut jpvt: Vec<i32> = (1..n + 1).collect();

        // The scalar factors of the elementary reflectors.
        let mut tau = vec![Complex64::new(0.0, 0.0); min_dim];

        // Set to -1 to query optimal scratch space
        let mut lwork = -1;
        // Complex work scratch space
        let mut work = vec![Complex64::new(0.0, 0.0); 1];

        // Double scratch space
        let mut rwork = vec![0.0; 2 * n as usize];

        // Return 0 if successful
        let mut info = 0;

        unsafe {
            zgeqp3(
                m,
                n,
                &mut self.data,
                lda,
                &mut jpvt,
                &mut tau,
                &mut work,
                lwork,
                &mut rwork,
                &mut info,
            );
        }

        // Get optimal work size
        lwork = work[0].re() as i32;

        // Resize work vector to optimal work size
        work.resize(lwork as usize, Complex64::new(0.0, 0.0));

        unsafe {
            zgeqp3(
                m,
                n,
                &mut self.data,
                lda,
                &mut jpvt,
                &mut tau,
                &mut work,
                lwork,
                &mut rwork,
                &mut info,
            );
        }

        assert!(info == 0, "QR decomposition did not converge ");
        let mut q_tensor = Tensor::new(&[m, n]);
        let mut r_tensor = Tensor::new(&[n, n]);

        // copy out upper right triangular matrix to taco tensor `r`
        for j in 0..n {
            for i in 0..min(min_dim as i32, j + 1) {
                r_tensor.insert(&[i, j], self.get(&[i, j]));
            }
        }

        unsafe {
            zungqr(
                m,
                min(m, n),
                min(m, n),
                &mut self.data,
                lda,
                &tau,
                &mut work,
                lwork,
                &mut info,
            );
        }

        for i in 0..m {
            for j in 0..n {
                q_tensor.insert(&[i, j], self.get(&[i, j]));
            }
        }
        (q_tensor, r_tensor)
    }
    fn svd(&mut self) -> (Tensor, Tensor, Tensor) {
        assert!(self.ndim() == 2, "Only able to decompose matrices");
        // Get shape of input Tensor
        let m = self.shape[0];
        let n = self.shape[1];
        let min_dim = min(m, n) as usize;
        let max_dim = max(m, n) as usize;

        // Leading dimension of `self`
        let lda = max(1, m);
        let ldu = m;
        let ldvt = min_dim as i32;

        let mut s = vec![0.0; min_dim];

        let mut u_tensor = Tensor::new(&[m, ldvt]);
        let mut vt_tensor = Tensor::new(&[ldvt, n]);
        let mut s_tensor = Tensor::new(&[ldvt, ldvt]);

        // Set to -1 to query optimal scratch space
        let mut lwork = -1;
        // Complex work scratch space
        let mut work = vec![Complex64::new(0.0, 0.0); 1];

        // Double scratch space
        let mut rwork = vec![
            0.0;
            max(
                5 * (min_dim << 2) + 5 * min_dim,
                2 * (max_dim << 2) + 2 * (min_dim << 2) + min_dim
            )
        ];

        // Integer scratch space
        let mut iwork = vec![0; 8 * min_dim];

        // Return 0 if successful
        let mut info = 0;
        unsafe {
            zgesdd(
                b'S',
                m,
                n,
                &mut self.data,
                lda,
                &mut s,
                &mut u_tensor.data,
                ldu,
                &mut vt_tensor.data,
                ldvt,
                &mut work,
                lwork,
                &mut rwork,
                &mut iwork,
                &mut info,
            )
        }

        // Get optimal work size
        lwork = work[0].re() as i32;

        // Resize work vector to optimal work size
        work.resize(lwork as usize, Complex64::new(0.0, 0.0));

        unsafe {
            zgesdd(
                b'S',
                m,
                n,
                &mut self.data,
                lda,
                &mut s,
                &mut u_tensor.data,
                ldu,
                &mut vt_tensor.data,
                ldvt,
                &mut work,
                lwork,
                &mut rwork,
                &mut iwork,
                &mut info,
            )
        }
        for i in 0..ldvt {
            s_tensor.insert(&[i, i], Complex64::new(s[i as usize], 0.0));
        }

        (u_tensor, s_tensor, vt_tensor)
    }
}

#[cfg(test)]
mod tests {
    use crate::decomposition::Decomposition;
    use crate::{contract, Tensor};
    use itertools::Itertools;
    use num_complex::Complex64;
    use rand::distributions::{Distribution, Uniform};
    use rand::Rng;

    #[test]
    fn test_qr() {
        let mut rng = rand::thread_rng();
        let die = Uniform::from(4..10);

        let tensor_dims = &[die.sample(&mut rng), die.sample(&mut rng)];
        let mut a = Tensor::new(tensor_dims);

        let t_ranges = tensor_dims
            .iter()
            .map(|e| (0..*e))
            .multi_cartesian_product();

        for dim in t_ranges.clone() {
            a.insert(&dim, Complex64::new(rng.gen(), 0.0));
        }
        let sol = a.clone();

        let (q, r) = a.qr();
        let out = contract(&[0, 2], &[0, 1], &q, &[1, 2], &r);
        for dim in t_ranges {
            assert!((sol.get(&dim) - out.get(&dim)).norm() < std::f64::EPSILON * 1e1);
        }
    }

    #[test]
    fn test_svd() {
        let mut rng = rand::thread_rng();
        let die = Uniform::from(4..10);

        let tensor_dims = &[die.sample(&mut rng), die.sample(&mut rng)];
        let mut a = Tensor::new(tensor_dims);

        let t_ranges = tensor_dims
            .iter()
            .map(|e| (0..*e))
            .multi_cartesian_product();

        for dim in t_ranges.clone() {
            a.insert(&dim, Complex64::new(rng.gen(), 0.0));
        }
        let sol = a.clone();

        let (u, s, vt) = a.svd();
        let us = contract(&[0, 2], &[0, 1], &u, &[1, 2], &s);
        let out = contract(&[0, 2], &[0, 1], &us, &[1, 2], &vt);
        for dim in t_ranges {
            assert!((sol.get(&dim) - out.get(&dim)).norm() < std::f64::EPSILON * 1e1);
        }
    }
}
