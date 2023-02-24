extern crate openblas_src;
use hptt_sys::{permute, transpose_simple};
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
        self.permutation = (0..self.shape.len() as i32).collect();
        self.shape = permute(&self.permutation, &self.shape);
    }

    /// Computes the flat index given the accessed coordinates.
    /// Assumes column-major ordering.
    ///
    /// # Panics
    /// Panics if the coordinates are invalid.
    fn compute_index(&self, coordinates: &[i32]) -> usize {
        // Get the unpermuted coordinates
        let dims = permute(&self.permutation, coordinates);

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
    pub fn shape(&self) -> Vec<i32> {
        permute(&self.permutation, &self.shape)
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
        self.shape[self.permutation[leg as usize] as usize] as i32
    }

    /// Transposes the tensor axes according to the permutation.
    /// This method does not modify the data but only the view, hence it's zero cost.
    pub fn transpose(&mut self, permutation: &[i32]) {
        self.permutation = permute(permutation, &self.permutation);
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

        a.transpose(&[1, 0]);
        assert_eq!(a.shape(), vec![3, 2]);
        assert_eq!(a.get(&[0, 0]), Complex64::new(1.0, 2.0));
        assert_eq!(a.get(&[1, 0]), Complex64::new(0.0, -1.0));
        assert_eq!(a.get(&[2, 1]), Complex64::new(-5.0, 0.0));
    }
}
