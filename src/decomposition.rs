use crate::Tensor;
use lapack::{zgeqp3, zgesdd, zungqr};
use num_complex::{Complex64, ComplexFloat};
use std::cmp::{max, min};

pub trait Decomposition {
    fn qr(self) -> (Tensor, Tensor);
    fn svd(self) -> (Tensor, Tensor, Tensor);
}

impl Decomposition for Tensor {
    /// Implements QR decomposition. Returns (Q,R) where Q is a Hermitian matrix and
    /// R is an upper right triangular matrix.
    ///
    /// # Panics
    /// - Panics if the tensor is not a matrix
    /// - Panics if the QR decomposition does not converge
    fn qr(mut self) -> (Tensor, Tensor) {
        // Get shape of input Tensor
        let [m, n] = self.shape()[..] else {
            panic!("Only able to decompose matrices")
        };
        let min_dim = min(m, n) as usize;

        // Leading dimension of `self`
        let lda = max(1, m);

        let mut jpvt = (1..=(n as i32)).collect::<Vec<i32>>();

        // The scalar factors of the elementary reflectors.
        let mut tau = Vec::with_capacity(min_dim);

        // Set to -1 to query optimal scratch space
        let mut lwork = -1;

        // Complex work scratch space
        let mut work = vec![Complex64::new(0.0, 0.0); 1];

        // Double scratch space
        let mut rwork = Vec::with_capacity(2 * n as usize);

        // Return 0 if successful
        let mut info = 0;

        unsafe {
            zgeqp3(
                m.try_into().unwrap(),
                n.try_into().unwrap(),
                &mut self.get_raw_data_mut(),
                lda.try_into().unwrap(),
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
                m.try_into().unwrap(),
                n.try_into().unwrap(),
                &mut self.get_raw_data_mut(),
                lda.try_into().unwrap(),
                &mut jpvt,
                &mut tau,
                &mut work,
                lwork,
                &mut rwork,
                &mut info,
            );
        }

        assert!(info == 0, "QR decomposition did not converge");
        let mut q_tensor = Self::new(&[m, n]);
        let mut r_tensor = Self::new(&[n, n]);

        // copy out upper right triangular matrix to tensor `r`
        for j in 0..n {
            for i in 0..min(min_dim as u64, j + 1) {
                r_tensor.insert(&[i, j], self.get(&[i, j]));
            }
        }

        unsafe {
            zungqr(
                m.try_into().unwrap(),
                min(m, n).try_into().unwrap(),
                min(m, n).try_into().unwrap(),
                &mut self.get_raw_data_mut(),
                lda.try_into().unwrap(),
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

    /// Implements SVD decomposition. Returns (U,S,Vt) where U and Vt are unitary
    /// matrices and S is a diagonal matrix of singular values.
    ///
    /// # Panics
    /// - Panics if the tensor is not a matrix
    fn svd(mut self) -> (Tensor, Tensor, Tensor) {
        // Get shape of input Tensor
        let [m, n] = self.shape()[..] else {
            panic!("Only able to decompose matrices")
        };
        let min_dim = min(m, n) as usize;
        let max_dim = max(m, n) as usize;

        // Leading dimension of `self`
        let lda = max(1, m);
        // Leading dimension of `u_tensor`
        let ldu = m;
        // Leading dimension of `vt_tensor`
        let ldvt = min_dim as u64;
        // Using double vector as stand in until other Tensor types defined
        let mut s = Vec::with_capacity(min_dim);

        // TODO: Add different Tensor types that allow for diagonal tensors
        let mut u_tensor = Self::new(&[m, ldvt]);
        let mut vt_tensor = Self::new(&[ldvt, n]);
        let mut s_tensor = Self::new(&[ldvt, ldvt]);

        // Set to -1 to query optimal scratch space
        let mut lwork = -1;

        // Complex work scratch space
        let mut work = vec![Complex64::new(0.0, 0.0); 1];

        // Double scratch space
        let mut rwork = Vec::with_capacity(max(
            5 * min_dim * min_dim + 5 * min_dim,
            2 * min_dim * max_dim + 2 * min_dim * min_dim + min_dim,
        ));
        // Integer scratch space
        let mut iwork = Vec::with_capacity(8 * min_dim);

        // Queries for optimal scratch space
        let mut info = 0;
        unsafe {
            zgesdd(
                b'S',
                m.try_into().unwrap(),
                n.try_into().unwrap(),
                &mut self.get_raw_data_mut(),
                lda.try_into().unwrap(),
                &mut s,
                &mut u_tensor.get_raw_data_mut(),
                ldu.try_into().unwrap(),
                &mut vt_tensor.get_raw_data_mut(),
                ldvt.try_into().unwrap(),
                &mut work,
                lwork,
                &mut rwork,
                &mut iwork,
                &mut info,
            );
        }

        // Get optimal work size
        lwork = work[0].re() as i32;

        // Resize work vector to optimal work size
        work.resize(lwork as usize, Complex64::new(0.0, 0.0));

        unsafe {
            zgesdd(
                b'S',
                m.try_into().unwrap(),
                n.try_into().unwrap(),
                &mut self.get_raw_data_mut(),
                lda.try_into().unwrap(),
                &mut s,
                &mut u_tensor.get_raw_data_mut(),
                ldu.try_into().unwrap(),
                &mut vt_tensor.get_raw_data_mut(),
                ldvt.try_into().unwrap(),
                &mut work,
                lwork,
                &mut rwork,
                &mut iwork,
                &mut info,
            );
            s.set_len(s.capacity());
        }

        // Fill in s_tensor
        for i in 0..ldvt {
            s_tensor.insert(&[i, i], Complex64::new(s[i as usize], 0.0));
        }

        (u_tensor, s_tensor, vt_tensor)
    }
}

#[cfg(test)]
mod tests {
    use crate::decomposition::Decomposition;
    use crate::{all_close, contract, Tensor};
    use itertools::Itertools;
    use num_complex::Complex64;
    use rand::distributions::{Distribution, Uniform};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_qr() {
        let mut rng = StdRng::seed_from_u64(23);
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

        assert!(all_close(&out, &sol, 1e-14));
    }

    #[test]
    fn test_svd() {
        let mut rng = StdRng::seed_from_u64(23);
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

        assert!(all_close(&out, &sol, 1e-14));
    }
}
