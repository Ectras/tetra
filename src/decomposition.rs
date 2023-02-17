use crate::Tensor;
extern crate openblas_src;
use lapack::{zgeqp3, zungqr};
use num_complex::{Complex64, ComplexFloat};
use std::cmp::{max, min};

trait Decomposition {
    fn qr(&mut self) -> (Tensor, Tensor);
    fn svd(&mut self) -> (Tensor, Tensor, Tensor);
}

impl Decomposition for Tensor {
    fn qr(&mut self) -> (Tensor, Tensor) {
        assert!(self.ndim() == 2, "Only able to decompose matrices");
        let m = self.shape[0];
        let n = self.shape[1];
        let lda = max(1, m);
        let mut jpvt = vec![0; n as usize];
        let mut tau = vec![Complex64::new(0.0, 0.0); min(m, n) as usize];
        let mut lwork = -1;
        let mut work = vec![Complex64::new(0.0, 0.0); n as usize];
        let mut rwork = vec![0.0; 2 * n as usize];
        let mut info = 0;

        unsafe {
            zgeqp3(
                m,
                n,
                self.data.as_mut_slice(),
                lda,
                &mut jpvt,
                &mut tau,
                &mut work,
                lwork,
                &mut rwork,
                &mut info,
            );
        }
        lwork = work[0].re() as i32;
        work.resize(lwork as usize, Complex64::new(0.0, 0.0));

        unsafe {
            zgeqp3(
                m,
                n,
                self.data.as_mut_slice(),
                lda,
                &mut jpvt.as_mut_slice(),
                &mut tau.as_mut_slice(),
                &mut work.as_mut_slice(),
                lwork,
                &mut rwork.as_mut_slice(),
                &mut info,
            );
        }
        let mut r_tensor = Tensor::new(&[n, n]);
        let mut q_tensor = Tensor::new(&[m, n]);
        // copy out upper right triangular matrix to taco tensor `r`
        for i in 0..n {
            for j in 0..i + 1 {
                r_tensor.insert(&[j, i], self.get(&[j, i]));
            }
        }

        lwork = -1;

        unsafe {
            zungqr(
                m,
                n,
                min(m, n),
                self.data.as_mut_slice(),
                lda,
                &tau,
                &mut work,
                lwork,
                &mut info,
            );
        }
        lwork = work[0].re() as i32;
        work.resize(lwork as usize, Complex64::new(0.0, 0.0));
        unsafe {
            // recover unitary hermitian matrix `q` from reflections
            zungqr(
                m,
                n,
                min(m, n),
                self.data.as_mut_slice(),
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
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::decomposition::Decomposition;
    use crate::Tensor;
    use itertools::Itertools;
    use num_complex::Complex64;

    #[test]
    fn test_qr() {
        let tensor_dims = &[6, 2];
        let mut a = Tensor::new(tensor_dims);

        let data = [
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(7.0, 0.0),
            Complex64::new(9.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(8.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        let t_ranges = tensor_dims
            .iter()
            .map(|e| (0..*e))
            .multi_cartesian_product();
        let mut data_iter = data.iter();
        for dim in t_ranges {
            println!("{dim:?}");
            a.insert(&dim, *data_iter.next().unwrap());
        }
        let _b = Tensor::new(tensor_dims);

        let (q, r) = a.qr();
    }
}
