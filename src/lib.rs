use num_complex::Complex64;

/// A tensor of arbitrary dimensions containing complex64 values.
#[derive(Clone, Debug)]
pub struct Tensor {
    shape: Vec<i32>,
    data: Vec<Complex64>,
}

impl Tensor {
    /// Creates a new tensor of the given dimensions.
    /// The tensor is initialized with zeros.
    #[must_use]
    pub fn new(dimensions: &[i32]) -> Self {
        let total_items: usize = dimensions.iter().product::<i32>() as usize;
        Self {
            shape: dimensions.to_vec(),
            data: vec![Complex64::new(0.0, 0.0); total_items],
        }
    }

    /// Computes the flat index given the accessed coordinates.
    fn compute_index(&self, dimensions: &[i32]) -> usize {
        assert_eq!(dimensions.len(), self.shape.len());

        let mut idx = dimensions[0];
        for i in 1..dimensions.len() {
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
    pub fn get(&self, dimensions: &[i32]) -> Complex64 {
        let idx = self.compute_index(dimensions);
        self.data[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_computation() {
        let t = Tensor::new(&[2, 4, 5, 1]);
        assert_eq!(t.compute_index(&[0, 1, 1, 0]), 6);
        assert_eq!(t.compute_index(&[0, 1, 2, 0]), 7);
        assert_eq!(t.compute_index(&[1, 3, 4, 0]), 39);
    }
}
