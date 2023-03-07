use std::collections::HashSet;

extern crate openblas_src;
use cblas::{zgemm, Layout, Transpose};
use hptt_sys::{inv_permute, permute, transpose_simple};
use num_complex::Complex64;

/// A tensor of arbitrary dimensions containing complex64 values.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// The shape of the tensor.
    shape: Vec<i32>,

    /// The current permutation of axes.
    permutation: Vec<i32>,

    /// The tensor data in column-major order.
    data: Vec<Complex64>,
}

impl Tensor {
    /// Creates a new tensor of the given dimensions.
    /// The tensor is initialized with zeros.
    #[must_use]
    pub fn new(dimensions: &[i32]) -> Self {
        // Validity checks
        assert!(!dimensions.is_empty());
        for dim in dimensions {
            assert!(0 <= *dim);
        }

        // Construct tensor
        let total_items = dimensions.iter().product::<i32>() as usize;
        Self {
            shape: dimensions.to_vec(),
            permutation: (0..dimensions.len() as i32).collect(),
            data: vec![Complex64::new(0.0, 0.0); total_items],
        }
    }

    /// Actually transposes the underlying data according to the current axis permutation.
    fn materialize_transpose(&mut self) {
        self.data = transpose_simple(&self.permutation, &self.data, &self.shape);
        self.shape = permute(&self.permutation, &self.shape);
        self.permutation = (0..self.shape.len() as i32).collect();
    }

    /// Computes the flat index given the accessed coordinates.
    /// Assumes column-major ordering.
    ///
    /// # Panics
    /// Panics if the coordinates are invalid.
    fn compute_index(&self, coordinates: &[i32]) -> usize {
        // Get the unpermuted coordinates
        let dims = inv_permute(&self.permutation, coordinates);

        // Validate coordinates
        assert_eq!(dims.len(), self.shape.len());
        for i in 0..dims.len() {
            assert!(0 <= dims[i] && dims[i] < self.shape[i]);
        }

        // Compute index
        let mut idx = dims[dims.len() - 1];
        for i in (0..dims.len() - 1).rev() {
            idx = dims[i] + self.shape[i] * idx;
        }
        idx as usize
    }

    /// Inserts a value at the given position.
    pub fn insert(&mut self, coordinates: &[i32], value: Complex64) {
        let idx = self.compute_index(coordinates);
        self.data[idx] = value;
    }

    /// Gets the value at the given position.
    #[must_use]
    pub fn get(&self, coordinates: &[i32]) -> Complex64 {
        let idx = self.compute_index(coordinates);
        self.data[idx]
    }

    /// Returns a copy of the current shape.
    /// 
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// let mut t = Tensor::new(&[3, 2, 5, 4, 1]);
    /// assert_eq!(t.shape(), vec![3, 2, 5, 4, 1]);
    /// t.transpose(&[3, 1, 4, 0, 2]);
    /// assert_eq!(t.shape(), vec![4, 2, 1, 3, 5]);
    /// ```
    pub fn shape(&self) -> Vec<i32> {
        permute(&self.permutation, &self.shape)
    }

    /// Returns the size of a single axis or of the whole tensor.
    /// 
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// let t = Tensor::new(&[1, 3, 5]);
    /// assert_eq!(t.size(None), 15);
    /// assert_eq!(t.size(Some(1)), 3);
    /// assert_eq!(t.size(Some(2)), 5);
    /// ```
    pub fn size(&self, axis: Option<usize>) -> i32 {
        if let Some(axis) = axis {
            self.shape[self.permutation[axis] as usize]
        } else {
            self.data.len() as i32
        }
    }

    /// Gets the number of legs.
    ///
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// assert_eq!(Tensor::new(&[1, 1]).leg_count(), 2);
    /// assert_eq!(Tensor::new(&[1, 1, 2, 3]).leg_count(), 4);
    /// ```
    pub fn leg_count(&self) -> i32 {
        self.shape.len() as i32
    }

    /// Gets the dimension of the specified leg.
    ///
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// let t = Tensor::new(&[1, 3, 2, 4]);
    /// assert_eq!(t.leg_dimension(0), 1);
    /// assert_eq!(t.leg_dimension(1), 3);
    /// assert_eq!(t.leg_dimension(2), 2);
    /// ```
    pub fn leg_dimension(&self, leg: i32) -> i32 {
        self.shape[self.permutation[leg as usize] as usize] as i32
    }

    /// Transposes the tensor axes according to the permutation.
    /// This method does not modify the data but only the view, hence it's zero cost.
    pub fn transpose(&mut self, permutation: &[i32]) {
        self.permutation = permute(permutation, &self.permutation);
    }
}

/// Contracts two tensors a and b, writing the result to the out tensor.
/// The indices specify which legs are to be contracted (like einsum notation). So if
/// two tensors share an index, the corresponding dimension is contracted.
#[must_use]
pub fn contract(
    out_indices: &[i32],
    a_indices: &[i32],
    a: &Tensor,
    b_indices: &[i32],
    b: &Tensor,
) -> Tensor {
    assert_eq!(a_indices.len(), a.shape.len());
    assert_eq!(b_indices.len(), b.shape.len());

    // Find contracted indices
    let a_legs = a_indices.iter().copied().collect::<HashSet<_>>();
    let b_legs = b_indices.iter().copied().collect::<HashSet<_>>();
    let contracted = a_legs
        .intersection(&b_legs)
        .copied()
        .collect::<HashSet<_>>();

    let mut remaining =
        Vec::with_capacity(a_indices.len() + b_indices.len() - 2 * contracted.len());

    // Compute permutation, total size of contracted dimensions and total size of remaining dimensions for A
    let mut a_contracted = 0;
    let mut a_remaining = 0;
    let mut a_contracted_size = 1;
    let mut a_remaining_size = 1;
    let mut a_perm = vec![0i32; a_indices.len()];
    let mut contract_order = vec![0; contracted.len()];
    for (i, idx) in a_indices.iter().enumerate() {
        if contracted.contains(idx) {
            a_perm[(a_indices.len() - contracted.len()) + a_contracted] = i as i32;
            contract_order[a_contracted] = *idx;
            a_contracted_size *= a.size(Some(i));
            a_contracted += 1;
        } else {
            a_perm[a_remaining] = i as i32;
            a_remaining += 1;
            a_remaining_size *= a.size(Some(i));
            remaining.push(*idx);
        }
    }

    // Get transposed A
    let mut a_transposed = a.clone();
    a_transposed.transpose(&a_perm);

    // Compute permutation, total size of contracted dimensions and total size of remaining dimensions for B
    let mut b_remaining = 0;
    let mut b_contracted_size = 1;
    let mut b_remaining_size = 1;
    let mut b_perm = vec![0i32; b_indices.len()];
    for (i, idx) in b_indices.iter().enumerate() {
        if contracted.contains(idx) {
            b_perm[contract_order.iter().position(|e| *e == *idx ).unwrap()] = i as i32;
            b_contracted_size *= b.size(Some(i));
        } else {
            b_perm[contracted.len() + b_remaining] = i as i32;
            b_remaining += 1;
            b_remaining_size *= b.size(Some(i));
            remaining.push(*idx);
        }
    }

    // Get transposed B
    let mut b_transposed = b.clone();
    b_transposed.transpose(&b_perm);

    // Make sure the connecting matrix dimensions match
    assert_eq!(a_contracted_size, b_contracted_size);

    // Compute the shape of C based on the remaining indices
    let mut c_shape = Vec::with_capacity(remaining.len());
    for r in remaining.iter() {
        let mut found = false;
        for (i, s) in a_indices.iter().enumerate() {
            if *r == *s {
                c_shape.push(a.size(Some(i)));
                found = true;
                break;
            }
        }

        if !found {
            for (i, s) in b_indices.iter().enumerate() {
                if *r == *s {
                    c_shape.push(b.size(Some(i)));
                    break;
                }
            }
        }
    }

    // Create output tensor
    let mut out = Tensor::new(&c_shape);

    // Compute ZGEMM
    a_transposed.materialize_transpose();
    b_transposed.materialize_transpose();
    unsafe {
        zgemm(
            Layout::ColumnMajor,
            Transpose::None,
            Transpose::None,
            a_remaining_size,
            b_remaining_size,
            b_contracted_size,
            Complex64::new(1.0, 0.0),
            &a_transposed.data,
            a_remaining_size,
            &b_transposed.data,
            b_contracted_size,
            Complex64::new(0.0, 0.0),
            &mut out.data,
            a_remaining_size,
        );
    }

    // Find permutation for output tensor
    let mut c_perm = vec![0; remaining.len()];
    for i in 0..remaining.len() {
        for j in 0..remaining.len() {
            if out_indices[i] == remaining[j] {
                c_perm[i] = j as i32;
                break;
            }
        }
    }

    // Return transposed output tensor
    out.transpose(&c_perm);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_computation() {
        let t = Tensor::new(&[2, 4, 5, 1]);
        assert_eq!(t.compute_index(&[0, 0, 0, 0]), 0);
        assert_eq!(t.compute_index(&[1, 0, 0, 0]), 1);
        assert_eq!(t.compute_index(&[0, 1, 0, 0]), 2);
        assert_eq!(t.compute_index(&[0, 1, 1, 0]), 10);
        assert_eq!(t.compute_index(&[0, 1, 2, 0]), 18);
        assert_eq!(t.compute_index(&[1, 3, 4, 0]), 39);
    }

    #[test]
    fn test_single_transpose() {
        let mut a = Tensor::new(&[2, 3, 4]);
        a.insert(&[0, 0, 0], Complex64::new(1.0, 2.0));
        a.insert(&[0, 1, 3], Complex64::new(0.0, -1.0));
        a.insert(&[1, 2, 1], Complex64::new(-5.0, 0.0));

        a.transpose(&[1, 2, 0]);
        assert_eq!(a.shape(), vec![3, 4, 2]);
        assert_eq!(a.permutation, vec![1, 2, 0]);
        assert_eq!(a.get(&[0, 0, 0]), Complex64::new(1.0, 2.0));
        assert_eq!(a.get(&[1, 3, 0]), Complex64::new(0.0, -1.0));
        assert_eq!(a.get(&[2, 1, 1]), Complex64::new(-5.0, 0.0));

        a.transpose(&[1, 2, 0]);
        assert_eq!(a.shape(), vec![4, 2, 3]);
        assert_eq!(a.permutation, vec![2, 0, 1]);
        assert_eq!(a.get(&[0, 0, 0]), Complex64::new(1.0, 2.0));
        assert_eq!(a.get(&[3, 0, 1]), Complex64::new(0.0, -1.0));
        assert_eq!(a.get(&[1, 1, 2]), Complex64::new(-5.0, 0.0));
    }

    #[test]
    fn test_materialize_transpose() {
        let mut a = Tensor::new(&[2, 3, 4, 5]);
        a.insert(&[0, 0, 0, 1], Complex64::new(1.0, 2.0));
        a.insert(&[0, 1, 3, 2], Complex64::new(0.0, -1.0));
        a.insert(&[1, 2, 1, 4], Complex64::new(-5.0, 0.0));

        a.transpose(&[1, 2, 0, 3]);
        assert_eq!(a.permutation, [1, 2, 0, 3]);
        assert_eq!(a.shape(), vec![3, 4, 2, 5]);
        assert_eq!(a.get(&[0, 0, 0, 1]), Complex64::new(1.0, 2.0));
        assert_eq!(a.get(&[1, 3, 0, 2]), Complex64::new(0.0, -1.0));
        assert_eq!(a.get(&[2, 1, 1, 4]), Complex64::new(-5.0, 0.0));
        a.materialize_transpose();
        assert_eq!(a.permutation, [0, 1, 2, 3]);
        assert_eq!(a.shape(), vec![3, 4, 2, 5]);
    }

    #[test]
    fn toy_contraction() {
        // Create tensors
        let mut b = Tensor::new(&[2, 3, 4]);
        let mut c = Tensor::new(&[4]);

        // Insert data into B and C
        b.insert(&[0, 0, 0], Complex64::new(1.0, 0.0));
        b.insert(&[1, 2, 0], Complex64::new(2.0, 0.0));
        b.insert(&[1, 2, 1], Complex64::new(3.0, 0.0));
        c.insert(&[0], Complex64::new(4.0, 0.0));
        c.insert(&[1], Complex64::new(5.0, 0.0));

        // Contract the tensors
        let a = contract(&[0, 1], &[0, 1, 2], &b, &[2], &c);

        // Check result in A
        assert_eq!(a.get(&[0, 0]), Complex64::new(4.0, 0.0));
        assert_eq!(a.get(&[0, 1]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[0, 2]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[1, 0]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[1, 1]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[1, 2]), Complex64::new(23.0, 0.0));
    }

    #[test]
    fn toy_contraction_transposed() {
        // Create tensors
        let mut b = Tensor::new(&[2, 3, 4]);
        let mut c = Tensor::new(&[4]);

        // Insert data into B and C
        b.insert(&[0, 0, 0], Complex64::new(1.0, 0.0));
        b.insert(&[1, 2, 0], Complex64::new(2.0, 0.0));
        b.insert(&[1, 2, 1], Complex64::new(3.0, 0.0));
        c.insert(&[0], Complex64::new(4.0, 0.0));
        c.insert(&[1], Complex64::new(5.0, 0.0));

        // Contract the tensors
        let a = contract(&[1, 0], &[0, 1, 2], &b, &[2], &c);

        // Check result in A
        assert_eq!(a.get(&[0, 0]), Complex64::new(4.0, 0.0));
        assert_eq!(a.get(&[1, 0]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[2, 0]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[0, 1]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[1, 1]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[2, 1]), Complex64::new(23.0, 0.0));
    }
}
