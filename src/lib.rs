use hptt_sys::{permute, transpose_simple};
use num_complex::Complex64;

/// A tensor of arbitrary dimensions containing complex64 values.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// The shape of the tensor.
    shape: Vec<i32>,

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
            data: vec![Complex64::new(0.0, 0.0); total_items],
        }
    }

    /// Computes the flat index given the accessed coordinates.
    /// Assumes column-major ordering.
    ///
    /// # Panics
    /// Panics if the coordinates are invalid.
    fn compute_index(&self, dimensions: &[i32]) -> usize {
        // Validate coordinates
        assert_eq!(dimensions.len(), self.shape.len());
        for i in 0..dimensions.len() {
            assert!(0 <= dimensions[i] && dimensions[i] < self.shape[i]);
        }

        // Compute index
        let mut idx = dimensions[dimensions.len() - 1];
        for i in (0..dimensions.len() - 1).rev() {
            idx = dimensions[i] + self.shape[i] * idx;
        }
        idx as usize
    }

    /// Inserts a value at the given position.
    pub fn insert(&mut self, dimensions: &[i32], value: Complex64) {
        let idx = self.compute_index(dimensions);
        self.data[idx] = value;
    }

    /// Gets the value at the given position.
    #[must_use]
    pub fn get(&self, dimensions: &[i32]) -> Complex64 {
        let idx = self.compute_index(dimensions);
        self.data[idx]
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
    /// assert_eq!(Tensor::new(&[1, 3, 2, 4]).leg_dimension(0), 1);
    /// assert_eq!(Tensor::new(&[1, 3, 2, 4]).leg_dimension(1), 3);
    /// assert_eq!(Tensor::new(&[1, 3, 2, 4]).leg_dimension(2), 2);
    /// ```
    pub fn leg_dimension(&self, leg: i32) -> i32 {
        self.shape[leg as usize] as i32
    }

    /// Returns a tensor with the axes transposed according to the permutation.
    #[must_use]
    pub fn transpose(&self, permutation: &[i32]) -> Self {
        let b_data = transpose_simple(permutation, &self.data, &self.shape);
        let b_shape = permute(permutation, &self.shape);
        Self {
            shape: b_shape,
            data: b_data,
        }
    }
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
    fn test_transpose() {
        let mut a = Tensor::new(&[2, 3]);
        a.insert(&[0, 0], Complex64::new(1.0, 2.0));
        a.insert(&[0, 1], Complex64::new(0.0, -1.0));
        a.insert(&[1, 2], Complex64::new(-5.0, 0.0));

        let b = a.transpose(&[1, 0]);
        assert_eq!(b.leg_count(), 2);
        assert_eq!(b.leg_dimension(0), 3);
        assert_eq!(b.leg_dimension(1), 2);
        assert_eq!(b.get(&[0, 0]), Complex64::new(1.0, 2.0));
        assert_eq!(b.get(&[1, 0]), Complex64::new(0.0, -1.0));
        assert_eq!(b.get(&[2, 1]), Complex64::new(-5.0, 0.0));
    }
}
