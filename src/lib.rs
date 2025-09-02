use std::{borrow::Cow, sync::Arc};

use float_cmp::{approx_eq, ApproxEq, F64Margin};
use hptt::transpose_simple;
use itertools::Itertools;
use num_complex::Complex64;
use permutation::Permutation;

use crate::{mkl::matrix_matrix_multiplication, utils::wrap};

#[cfg(feature = "serde")]
pub mod serde;

#[cfg(feature = "rand")]
pub mod random;

mod mkl;
mod utils;

pub use mkl::max_threads;

// pub mod decomposition;

/// The data layout of a tensor. For row-major, the last index is the fastest running
/// one.
///
/// This does not necessarily correspond to the underlying memory layout, as it
/// can also be realized through permutating accesses.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Layout {
    RowMajor,
    ColumnMajor,
}

/// A tensor of arbitrary dimensions containing [`Complex64`] values.
#[allow(clippy::len_without_is_empty)]
#[derive(Clone, Debug)]
pub struct Tensor {
    /// The shape of the tensor.
    shape: Vec<usize>,

    /// The current permutation of dimensions.
    permutation: Permutation,

    /// The tensor data in column-major order.
    data: Arc<Vec<Complex64>>,
}

impl Tensor {
    /// Creates a new `Tensor` of the given dimensions.
    /// The tensor is initialized with zeros.
    /// For a scalar, pass an empty slice (or use [`Tensor::new_scalar`]).
    ///
    /// # Panics
    /// - Panics if any dimension is zero
    ///
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// let scalar = Tensor::new(&[]);
    /// let vector = Tensor::new(&[5]);
    /// let matrix = Tensor::new(&[3, 4]);
    /// ```
    #[must_use]
    pub fn new(dimensions: &[usize]) -> Self {
        // Validity checks
        assert!(dimensions.iter().all(|&x| x > 0));

        // Construct tensor
        let identity = Permutation::one(dimensions.len());
        let total_items = Tensor::total_items(dimensions);
        let zeros = vec![Complex64::default(); total_items];
        Self {
            shape: dimensions.to_vec(),
            permutation: identity,
            data: Arc::new(zeros),
        }
    }

    /// Creates a new [`Tensor`] with the given dimensions and the corresponding data.
    /// Assumes data is column-major unless otherwise specified.
    ///
    /// # Panics
    /// - Panics if any dimension is zero
    /// - Panics if the length of the data does not match with the dimensions given
    ///
    /// # Examples
    /// ```
    /// # use num_complex::Complex64;
    /// # use tetra::{Layout, Tensor, all_close};
    /// let row_major = Tensor::new_from_flat(&[2, 2], vec![
    ///     Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
    ///     Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)
    /// ], Some(Layout::RowMajor));
    /// let col_major = Tensor::new_from_flat(&[2, 2], vec![
    ///     Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0),
    ///     Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)
    /// ], Some(Layout::ColumnMajor));
    /// assert!(all_close(&row_major, &col_major, 1e-12));
    /// ```
    #[must_use]
    pub fn new_from_flat(
        dimensions: &[usize],
        data: Vec<Complex64>,
        layout: Option<Layout>,
    ) -> Self {
        // Validity checks
        assert!(dimensions.iter().all(|&x| x > 0));
        assert_eq!(Tensor::total_items(dimensions), data.len());

        // Get the permutation based on the requested layout
        let (permutation, shape) = match layout.unwrap_or(Layout::ColumnMajor) {
            Layout::RowMajor => {
                let perm_line = (0..dimensions.len()).rev().collect_vec();
                let perm = Permutation::oneline(perm_line);
                let mut dims = dimensions.to_vec();
                dims.reverse();
                (perm, dims)
            }
            Layout::ColumnMajor => (Permutation::one(dimensions.len()), dimensions.to_vec()),
        };

        // Construct tensor
        Self {
            shape,
            permutation,
            data: Arc::new(data),
        }
    }

    /// Creates a new `Tensor` with a single scalar value.
    #[inline]
    #[must_use]
    pub fn new_scalar(value: Complex64) -> Self {
        Self::new_from_flat(&[], vec![value], None)
    }

    /// Computes the total number of items specified by `dimensions`.
    ///
    /// # Panics
    /// - Panics if the result overflows (in debug *and* release mode)
    ///
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// assert_eq!(Tensor::total_items(&[2, 3, 4]), 24);
    /// assert_eq!(Tensor::total_items(&[1, 1]), 1);
    ///
    /// // A scalar has 1 item:
    /// assert_eq!(Tensor::total_items(&[]), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn total_items(dimensions: &[usize]) -> usize {
        dimensions
            .iter()
            .fold(1, |acc, &x| acc.checked_mul(x).unwrap())
    }

    /// Computes the flat index given the accessed coordinates.
    /// Assumes column-major ordering.
    ///
    /// # Panics
    /// - Panics if the coordinates are invalid
    #[must_use]
    fn compute_index(&self, coordinates: &[usize]) -> usize {
        // Borrow the data
        assert_eq!(coordinates.len(), self.shape.len());

        // Get the unpermuted coordinates
        let coords = self.permutation.apply_inv_slice(coordinates);

        // Compute index
        let mut idx = 0;
        for i in (0..coords.len()).rev() {
            // Check coordinate
            assert!(coords[i] < self.shape[i]);

            // Accumulate index
            let c = coords[i];
            let s = self.shape[i];
            if i == coords.len() - 1 {
                idx = c;
            } else {
                idx = s * idx + c;
            }
        }
        idx
    }

    /// Sets the value at the given position.
    pub fn set(&mut self, coordinates: &[usize], value: Complex64) {
        let idx = self.compute_index(coordinates);
        let data = Arc::make_mut(&mut self.data);
        data[idx] = value;
    }

    /// Gets the value at the given position.
    #[must_use]
    pub fn get(&self, coordinates: &[usize]) -> Complex64 {
        let idx = self.compute_index(coordinates);
        self.data[idx]
    }

    /// Returns a copy of the current shape.
    ///
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// # use permutation::Permutation;
    /// let mut t = Tensor::new(&[3, 2, 5, 4, 1]);
    /// assert_eq!(t.shape(), vec![3, 2, 5, 4, 1]);
    /// t.transpose(&Permutation::oneline([3, 1, 4, 0, 2]));
    /// assert_eq!(t.shape(), vec![4, 2, 1, 3, 5]);
    /// ```
    #[inline]
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.permutation.apply_slice(&self.shape)
    }

    /// Returns the size of a single dimension.
    ///
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// # use permutation::Permutation;
    /// let mut t = Tensor::new(&[1, 3, 5]);
    /// assert_eq!(t.len_of(1), 3);
    /// assert_eq!(t.len_of(2), 5);
    /// assert_eq!(t.len_of(0), 1);
    /// t.transpose(&Permutation::oneline([1, 2, 0]));
    /// assert_eq!(t.len_of(0), 5);
    /// assert_eq!(t.len_of(1), 1);
    /// assert_eq!(t.len_of(2), 3);
    /// ```
    #[inline]
    #[must_use]
    pub fn len_of(&self, dimension: usize) -> usize {
        self.shape[self.permutation.apply_inv_idx(dimension)]
    }

    /// Returns the size of the tensor, that is, the total number of elements of all
    /// dimensions.
    ///
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// let t1 = Tensor::new(&[1, 3, 5]);
    /// assert_eq!(t1.len(), 15);
    /// let t2 = Tensor::new(&[4, 2]);
    /// assert_eq!(t2.len(), 8);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of dimensions of the tensor.
    ///
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// assert_eq!(Tensor::new(&[1, 2]).ndim(), 2);
    /// assert_eq!(Tensor::new(&[1, 3, 6, 5]).ndim(), 4);
    /// ```
    #[inline]
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Transposes the tensor axes according to the permutation.
    /// This method does not modify the data but only the view, hence it is zero
    /// cost.
    #[inline]
    pub fn transpose(&mut self, permutation: &Permutation) {
        self.permutation = permutation * &self.permutation;
    }

    /// Computes the transposed data based on the current permutation.
    fn compute_transposed_data(&self, data: &[Complex64]) -> Vec<Complex64> {
        // Get the permutation as [i32]
        let raw_perm = (0..i32::try_from(self.permutation.len()).unwrap()).collect_vec();
        let raw_perm = self.permutation.apply_slice(raw_perm);

        // Get the shape as [i32]
        let shape = self
            .shape
            .iter()
            .map(|x| (*x).try_into().unwrap())
            .collect_vec();

        // Transpose data and shape
        transpose_simple(&raw_perm, data, &shape)
    }

    /// Returns whether the data is laid out contiguous in memory, i.e., the logical
    /// order of elements matches the physical order in memory. The order is
    /// cloumn-major.
    #[inline]
    fn is_contiguous(&self) -> bool {
        self.permutation == Permutation::one(self.permutation.len())
    }

    /// Returns the items of the tensor as a flat vector. The elements correspond to
    /// the logical order of elements: If the data is contiguous, a borrowed
    /// reference is returned, otherwise a contiguous copy is made and returned.
    pub fn elements(&self) -> Cow<Vec<Complex64>> {
        if self.is_contiguous() {
            Cow::Borrowed(&*self.data)
        } else {
            Cow::Owned(self.compute_transposed_data(&self.data))
        }
    }

    /// Extracts the data from the tensor. If the data is contiguous, it is returned
    /// as-is. Otherwise, a contiguous copy is made and returned.
    fn into_elements(self) -> Arc<Vec<Complex64>> {
        if self.is_contiguous() {
            self.data
        } else {
            Arc::new(self.compute_transposed_data(&self.data))
        }
    }

    /// Gets a mutable reference to the raw (i.e. flat) vector data. If the data is
    /// shared, it is cloned, so modifications are not reflected in other tensors.
    /// The data is not guaranteed to be contiguous.
    #[inline]
    pub fn raw_data_mut(&mut self) -> &mut Vec<Complex64> {
        Arc::make_mut(&mut self.data)
    }

    /// Conjugates the tensor in-place. If the data is shared, it will be copied
    /// first.
    ///
    /// # Examples
    /// ```
    /// # use num_complex::Complex64;
    /// # use tetra::{Tensor, all_close};
    /// let mut tensor = Tensor::new_from_flat(&[2, 2], vec![
    ///     Complex64::new(0.0, 0.0), Complex64::new(3.0, 0.0),
    ///     Complex64::new(2.0, 2.0), Complex64::new(0.0, 4.0)
    /// ], None);
    /// tensor.conjugate();
    ///
    /// let reference = Tensor::new_from_flat(&[2, 2], vec![
    ///     Complex64::new(0.0, 0.0), Complex64::new(3.0, 0.0),
    ///     Complex64::new(2.0, -2.0), Complex64::new(0.0, -4.0)
    /// ], None);
    ///
    /// assert!(all_close(&tensor, &reference, 1e-12))
    /// ```
    pub fn conjugate(&mut self) {
        let owned_data = Arc::make_mut(&mut self.data);
        for val in owned_data {
            *val = val.conj();
        }
    }

    /// Slices the tensor along the given `axis` at the given `index`. Returns the
    /// slice as a new tensor.
    ///
    /// Examples for numpy notation:
    /// - `slice(0, 1)` is equivalent to `tensor[1, :, :]`
    /// - `slice(1, 0)` is equivalent to `tensor[:, 0, :]`
    /// - `slice(2, 2)` is equivalent to `tensor[:, :, 2]`
    ///
    /// # Examples
    /// ```
    /// # use num_complex::Complex64;
    /// # use tetra::Tensor;
    /// let mut tensor = Tensor::new_from_flat(&[2, 2], vec![
    ///     Complex64::new(0.0, 0.0), Complex64::new(3.0, 0.0),
    ///     Complex64::new(2.0, 2.0), Complex64::new(0.0, 4.0)
    /// ], None);
    /// let slice = tensor.slice(0, 1); // in numpy notation this would be tensor[1, :]
    /// ```
    pub fn slice(&self, axis: usize, index: usize) -> Self {
        assert!(self.ndim() > 0, "Cannot slice a scalar");

        // Get permutation to make the axis the last one
        let last_index = self.ndim() - 1;
        let initial_perm = (0..self.ndim())
            .map(|i| match i.cmp(&axis) {
                std::cmp::Ordering::Less => i,
                std::cmp::Ordering::Equal => last_index,
                std::cmp::Ordering::Greater => i - 1,
            })
            .collect_vec();
        let perm_move_back = Permutation::oneline(initial_perm);

        // Get a transposed copy of the tensor (data is shared)
        let mut transposed = self.clone();
        transposed.transpose(&perm_move_back);

        // Get the slice
        let new_shape = &transposed.shape()[..last_index];
        let data = transposed.elements();
        let slice_size = Tensor::total_items(new_shape);
        let slice_data = data[index * slice_size..(index + 1) * slice_size].to_vec();

        Tensor::new_from_flat(new_shape, slice_data, None)
    }
}

/// Helper struct containing information about a contraction, like the required
/// permutations and resulting shape.
#[derive(Debug)]
struct ContractionPermutationData {
    /// The uncontracted labels.
    uncontracted: Vec<usize>,
    /// Permutation of `a`, moving uncontracted dimensions to the front and contracted
    /// dimensions to the back.
    a_permutation: Vec<usize>,
    /// Permutation of `b`, moving contracted dimensions to the front (in the same
    /// order as they appear in `a`) and uncontracted dimensions to the back.
    b_permutation: Vec<usize>,
    /// The size of the contracted dimensions.
    contracted_size: usize,
    /// The size of the uncontracted dimensions of `a`.
    a_uncontracted_size: usize,
    /// The size of the uncontracted dimensions of `b`.
    b_uncontracted_size: usize,
    /// The shape of the resulting tensor.
    c_shape: Vec<usize>,
}

#[must_use]
fn compute_contraction_permutation(
    a_labels: &[usize],
    a_shape: &[usize],
    b_labels: &[usize],
    b_shape: &[usize],
) -> ContractionPermutationData {
    let mut a_permutation = vec![0; a_labels.len()];
    let mut b_permutation = vec![0; b_labels.len()];

    let mut uncontracted = Vec::with_capacity(a_labels.len() + b_labels.len());
    let mut contracted = 0;
    let mut a_uncontracted_size = 1;
    let mut contracted_size = 1;
    let mut b_uncontracted_size = 1;
    let mut contracted_mask_b = vec![false; b_labels.len()];
    let mut c_shape = Vec::with_capacity(uncontracted.capacity());

    for (i, (label_a, dim_a)) in a_labels.iter().zip(a_shape).enumerate() {
        if let Some(j) = b_labels.iter().position(|label_b| label_a == label_b) {
            // This is a contracted index, move to back in `a` and front in `b`
            b_permutation[contracted] = j;
            contracted_mask_b[j] = true;
            contracted += 1;
            a_permutation[a_labels.len() - contracted] = i;
            contracted_size *= dim_a;
        } else {
            // This is an uncontracted index of `a`
            a_uncontracted_size *= dim_a;
            a_permutation[uncontracted.len()] = i;
            uncontracted.push(*label_a);
            c_shape.push(*dim_a);
        }
    }

    // permutation of contracted indices is reverse to order in `contracted` vec
    a_permutation[a_labels.len() - contracted..].reverse();

    let mut uncontracted_b = 0;
    for (i, (label_b, dim_b)) in b_labels.iter().zip(b_shape).enumerate() {
        if !contracted_mask_b[i] {
            // This is an uncontracted index of `b`
            b_uncontracted_size *= dim_b;
            b_permutation[contracted + uncontracted_b] = i;
            uncontracted.push(*label_b);
            c_shape.push(*dim_b);
            uncontracted_b += 1;
        }
    }

    ContractionPermutationData {
        uncontracted,
        a_permutation,
        b_permutation,
        contracted_size,
        a_uncontracted_size,
        b_uncontracted_size,
        c_shape,
    }
}

/// Contracts two tensors, returning the resulting tensor.
///
/// The indices specify which legs are to be contracted (like einsum notation). So if
/// two tensors share an index, the corresponding dimension is contracted.
///
/// # Examples
/// The following is equal to a matrix-matrix multiplication
/// ```
/// # use tetra::Tensor;
/// # use tetra::contract;
/// let a = Tensor::new(&[2, 3]);
/// let b = Tensor::new(&[3, 4]);
/// let c = contract(&[0, 2], &[0, 1], a, &[1, 2], b);
/// assert_eq!(c.shape(), vec![2, 4]);
/// ```
/// The following is equal to a scalar product of two vectors
/// ```
/// # use tetra::Tensor;
/// # use tetra::contract;
/// let a = Tensor::new(&[3]);
/// let b = Tensor::new(&[3]);
/// let c = contract(&[], &[0], a, &[0], b);
/// ```
#[must_use]
pub fn contract(
    out_indices: &[usize],
    a_indices: &[usize],
    mut a: Tensor,
    b_indices: &[usize],
    mut b: Tensor,
) -> Tensor {
    assert_eq!(a_indices.len(), a.ndim());
    assert_eq!(b_indices.len(), b.ndim());

    let ContractionPermutationData {
        uncontracted,
        a_permutation,
        b_permutation,
        a_uncontracted_size,
        b_uncontracted_size,
        contracted_size,
        c_shape,
    } = compute_contraction_permutation(a_indices, &a.shape(), b_indices, &b.shape());

    // Get transposed A
    a.transpose(&Permutation::oneline(a_permutation).inverse());
    let a_data = a.into_elements();

    // Get transposed B
    b.transpose(&Permutation::oneline(b_permutation).inverse());
    let b_data = b.into_elements();

    // Compute GEMM
    let out_data = matrix_matrix_multiplication(
        a_uncontracted_size,
        contracted_size,
        b_uncontracted_size,
        &a_data,
        &b_data,
    );

    // Create output tensor
    let mut out = Tensor::new_from_flat(&c_shape, out_data, None);

    // Find permutation for output tensor
    let remaining_to_sorted = permutation::sort(&uncontracted);
    let sorted_to_out_indices = permutation::sort(out_indices).inverse();
    let c_perm = &sorted_to_out_indices * &remaining_to_sorted;

    // Return transposed output tensor
    out.transpose(&c_perm);
    out
}

impl ApproxEq for &Tensor {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin = margin.into();
        if self.shape() != other.shape() {
            return false;
        }

        let self_elements = self.elements();
        let other_elements = other.elements();

        let self_elements = wrap(&self_elements);
        let other_elements = wrap(&other_elements);

        self_elements.approx_eq(other_elements, margin)
    }
}

/// Compares two tensors for approximate equality.
/// The tensors are considered equal if their shapes are equal and all their elements
/// are approximately equal.
#[must_use]
pub fn all_close(a: &Tensor, b: &Tensor, epsilon: f64) -> bool {
    approx_eq!(&Tensor, a, b, epsilon = epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_computation_scalar() {
        let t = Tensor::new(&[]);
        assert_eq!(t.compute_index(&[]), 0);
    }

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
    fn test_new_from_flat() {
        let index = (0..3).map(|_e| 0..3).multi_cartesian_product();
        let mut col_data = Vec::new();
        let mut row_data = Vec::new();
        let index_size = [1, 3, 9];
        let dimensions = [3, 3, 3];
        for mut dims in index {
            col_data.push(Complex64::new(
                dims.iter()
                    .zip(index_size.iter())
                    .map(|(i, size)| f64::from(i * size))
                    .product::<f64>(),
                0.0,
            ));
            dims.reverse();
            row_data.push(Complex64::new(
                dims.iter()
                    .zip(index_size.iter())
                    .map(|(i, size)| f64::from(i * size))
                    .product::<f64>(),
                0.0,
            ));
        }
        let col_tensor = Tensor::new_from_flat(&dimensions, col_data, Some(Layout::ColumnMajor));
        let row_tensor = Tensor::new_from_flat(&dimensions, row_data, Some(Layout::RowMajor));

        assert!(all_close(&col_tensor, &row_tensor, 1e-12));
    }

    #[test]
    fn test_scalar_get_set() {
        let mut t = Tensor::new(&[]);
        t.set(&[], Complex64::new(1.0, 2.0));
        assert_eq!(t.get(&[]), Complex64::new(1.0, 2.0));
    }

    #[test]
    fn test_scalar_shape() {
        let t = Tensor::new(&[]);
        assert_eq!(t.shape(), vec![]);
        assert_eq!(t.len(), 1);
        assert_eq!(t.ndim(), 0);
    }

    #[test]
    fn test_single_transpose() {
        let mut a = Tensor::new(&[2, 3, 4, 5]);
        a.set(&[0, 0, 0, 1], Complex64::new(1.0, 2.0));
        a.set(&[0, 1, 3, 2], Complex64::new(0.0, -1.0));
        a.set(&[1, 2, 1, 4], Complex64::new(-5.0, 0.0));

        a.transpose(&Permutation::oneline([2, 0, 1, 3]));
        assert_eq!(a.shape(), vec![3, 4, 2, 5]);
        assert_eq!(a.get(&[0, 0, 0, 1]), Complex64::new(1.0, 2.0));
        assert_eq!(a.get(&[1, 3, 0, 2]), Complex64::new(0.0, -1.0));
        assert_eq!(a.get(&[2, 1, 1, 4]), Complex64::new(-5.0, 0.0));
    }

    #[test]
    fn test_transpose_twice() {
        let mut a = Tensor::new(&[2, 3, 4]);
        a.set(&[0, 0, 0], Complex64::new(1.0, 2.0));
        a.set(&[0, 1, 3], Complex64::new(0.0, -1.0));
        a.set(&[1, 2, 1], Complex64::new(-5.0, 0.0));

        a.transpose(&Permutation::oneline([2, 0, 1]));
        assert_eq!(a.shape(), vec![3, 4, 2]);
        assert_eq!(a.len_of(0), 3);
        assert_eq!(a.len_of(1), 4);
        assert_eq!(a.len_of(2), 2);
        assert_eq!(a.get(&[0, 0, 0]), Complex64::new(1.0, 2.0));
        assert_eq!(a.get(&[1, 3, 0]), Complex64::new(0.0, -1.0));
        assert_eq!(a.get(&[2, 1, 1]), Complex64::new(-5.0, 0.0));

        a.transpose(&Permutation::oneline([2, 0, 1]));
        assert_eq!(a.shape(), vec![4, 2, 3]);
        assert_eq!(a.len_of(0), 4);
        assert_eq!(a.len_of(1), 2);
        assert_eq!(a.len_of(2), 3);
        assert_eq!(a.get(&[0, 0, 0]), Complex64::new(1.0, 2.0));
        assert_eq!(a.get(&[3, 0, 1]), Complex64::new(0.0, -1.0));
        assert_eq!(a.get(&[1, 1, 2]), Complex64::new(-5.0, 0.0));
    }

    #[test]
    fn test_raw_data_mut() {
        let mut a = Tensor::new(&[2, 3, 4]);
        a.set(&[0, 0, 0], Complex64::new(1.0, 2.0));
        a.set(&[0, 1, 3], Complex64::new(0.0, -1.0));
        a.set(&[1, 2, 1], Complex64::new(-5.0, 0.0));
        let mut b = a.clone();

        // Get the mutable reference to the raw data
        let a_data = a.raw_data_mut();
        let mut ref_a_data = vec![Complex64::ZERO; 24];
        ref_a_data[0] = Complex64::new(1.0, 2.0);
        ref_a_data[11] = Complex64::new(-5.0, 0.0);
        ref_a_data[20] = Complex64::new(0.0, -1.0);
        assert_eq!(*a_data, ref_a_data);

        // Change a value
        a_data[15] = Complex64::new(3.0, 4.33);

        // Check that B is not affected
        let b_data = b.raw_data_mut();
        let mut ref_b_data = vec![Complex64::ZERO; 24];
        ref_b_data[0] = Complex64::new(1.0, 2.0);
        ref_b_data[11] = Complex64::new(-5.0, 0.0);
        ref_b_data[20] = Complex64::new(0.0, -1.0);
        assert_eq!(*b_data, ref_b_data);
    }

    #[test]
    fn test_elements() {
        let ref_data = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(-5.0, 0.0),
        ];
        let a = Tensor::new_from_flat(&[1, 3, 1], ref_data.clone(), None);

        assert!(a.is_contiguous());
        let elements = a.elements();
        let Cow::Borrowed(data) = elements else {
            panic!("Expected borrowed data")
        };
        assert_eq!(data, &ref_data);
    }

    #[test]
    fn test_elements_after_transpose() {
        let ref_data = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(-5.0, 0.0),
            Complex64::new(3.0, 4.0),
        ];
        let transposed_ref_data = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-5.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(3.0, 4.0),
        ];
        let mut a = Tensor::new_from_flat(&[2, 1, 2], ref_data.clone(), None);

        // Without transpose, the data is still contiguous and can be borrowed
        assert!(a.is_contiguous());
        let elements = a.elements();
        let Cow::Borrowed(data) = elements else {
            panic!("Expected borrowed data")
        };
        assert_eq!(data, &ref_data);

        a.transpose(&Permutation::oneline(vec![2, 0, 1]));

        // After transpose, the data is no longer contiguous and must be cloned
        assert!(!a.is_contiguous());
        let elements = a.elements();
        let Cow::Owned(data) = elements else {
            panic!("Expected owned data")
        };
        assert_eq!(data, transposed_ref_data);
    }

    #[test]
    fn test_raw_data_mut_reflects_changes() {
        let mut a = Tensor::new(&[4, 2]);

        assert_eq!(a.get(&[1, 1]), Complex64::ZERO);
        a.raw_data_mut()[5] = Complex64::new(2.0, -1.0);
        assert_eq!(a.get(&[1, 1]), Complex64::new(2.0, -1.0));
    }

    #[test]
    fn test_raw_data_mut_scalar_reflects_changes() {
        let mut a = Tensor::new(&[]);
        a.raw_data_mut()[0] = Complex64::new(2.0, -3.0);
        assert_eq!(a.get(&[]), Complex64::new(2.0, -3.0));
    }

    #[test]
    fn test_into_elements() {
        let a_data = vec![
            Complex64::new(3.0, 2.0),
            Complex64::new(-4.0, -1.0),
            Complex64::new(-5.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let a = Tensor::new_from_flat(&[2, 3], a_data.clone(), None);
        assert_eq!(*a.into_elements(), a_data);
    }

    #[test]
    fn test_into_elements_transposed() {
        let a_data = vec![
            Complex64::new(3.0, 2.0),
            Complex64::new(-4.0, -1.0),
            Complex64::new(-5.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let a_transposed = vec![
            a_data[0], a_data[3], a_data[1], a_data[4], a_data[2], a_data[5],
        ];
        let a = Tensor::new_from_flat(&[2, 3], a_data, Some(Layout::RowMajor));
        assert_eq!(*a.into_elements(), a_transposed);
    }

    #[test]
    fn test_into_elements_shared() {
        let data = vec![
            Complex64::new(3.0, 2.0),
            Complex64::new(-4.0, -1.0),
            Complex64::new(-5.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let a = Tensor::new_from_flat(&[2, 3], data.clone(), None);
        let b = a.clone();
        let view = Arc::downgrade(&a.data);

        // a and b share the data
        assert_eq!(Arc::strong_count(&a.data), 2);

        // Now consume a and drop the reference
        {
            let a_data = a.into_elements();
            assert_eq!(*a_data, data);
        }

        // Only b holds the data now
        assert_eq!(Arc::strong_count(&b.data), 1);

        // Now consume b and drop the reference
        {
            let b_data = b.into_elements();
            assert_eq!(*b_data, data);
        }

        // The data is now deallocated
        assert!(view.upgrade().is_none());
    }

    fn int_to_complex(x: Vec<i32>) -> Vec<Complex64> {
        x.into_iter()
            .map(|x| Complex64::new(x.into(), 0.0))
            .collect()
    }

    #[test]
    fn test_slice() {
        let data = int_to_complex(vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        ]);
        let matrix = Tensor::new_from_flat(&[2, 3, 4], data, None);

        // Slicing axis 0, index 0 => data[0, :, :]
        let ref_data_slice_0 = int_to_complex(vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]);
        let slice = matrix.slice(0, 0);
        assert_eq!(slice.shape(), vec![3, 4]);
        assert_eq!(*slice.into_elements(), ref_data_slice_0);

        // Slicing axis 1, index 2 => data[:, 2, :]
        let ref_data_slice_1 = int_to_complex(vec![4, 5, 10, 11, 16, 17, 22, 23]);
        let slice = matrix.slice(1, 2);
        assert_eq!(slice.shape(), vec![2, 4]);
        assert_eq!(*slice.into_elements(), ref_data_slice_1);

        // Slicing axis 2, index 1 => data[:, :, 1]
        let ref_data_slice_2 = int_to_complex(vec![6, 7, 8, 9, 10, 11]);
        let slice = matrix.slice(2, 1);
        assert_eq!(slice.shape(), vec![2, 3]);
        assert_eq!(*slice.into_elements(), ref_data_slice_2);
    }

    #[test]
    fn test_slice_rowmajor() {
        let data = int_to_complex(vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        ]);
        let matrix = Tensor::new_from_flat(&[2, 3, 4], data, Some(Layout::RowMajor));

        // Slicing axis 0, index 0 => data[0, :, :]
        let ref_data_slice_0 = int_to_complex(vec![0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
        let slice = matrix.slice(0, 0);
        assert_eq!(slice.shape(), vec![3, 4]);
        assert_eq!(*slice.into_elements(), ref_data_slice_0);

        // Slicing axis 1, index 2 => data[:, 2, :]
        let ref_data_slice_1 = int_to_complex(vec![8, 20, 9, 21, 10, 22, 11, 23]);
        let slice = matrix.slice(1, 2);
        assert_eq!(slice.shape(), vec![2, 4]);
        assert_eq!(*slice.into_elements(), ref_data_slice_1);

        // Slicing axis 2, index 1 => data[:, :, 1]
        let ref_data_slice_2 = int_to_complex(vec![1, 13, 5, 17, 9, 21]);
        let slice = matrix.slice(2, 1);
        assert_eq!(slice.shape(), vec![2, 3]);
        assert_eq!(*slice.into_elements(), ref_data_slice_2);
    }

    #[test]
    fn test_compute_contraction_permutation() {
        // Contracted axes is 2, 5, 4
        // Uncontracted axes is 8, 9, 10, 11, 3, 1, 6, 7
        let data = compute_contraction_permutation(
            &[8, 2, 9, 5, 10, 11, 4],
            &[2, 3, 4, 5, 6, 7, 8],
            &[3, 1, 6, 5, 2, 7, 4],
            &[2, 3, 4, 5, 3, 6, 8],
        );

        assert_eq!(data.uncontracted, &[8, 9, 10, 11, 3, 1, 6, 7]);
        assert_eq!(data.a_permutation, &[0, 2, 4, 5, 1, 3, 6]);
        assert_eq!(data.b_permutation, &[4, 3, 6, 0, 1, 2, 5]);
        assert_eq!(data.a_uncontracted_size, 2 * 4 * 6 * 7);
        assert_eq!(data.b_uncontracted_size, 2 * 3 * 4 * 6);
        assert_eq!(data.contracted_size, 3 * 5 * 8);
        assert_eq!(data.c_shape, &[2, 4, 6, 7, 2, 3, 4, 6]);
    }

    #[test]
    fn toy_contraction() {
        // Create tensors
        let mut b = Tensor::new(&[2, 3, 4]);
        let mut c = Tensor::new(&[4]);

        // Insert data into B and C
        b.set(&[0, 0, 0], Complex64::new(1.0, 0.0));
        b.set(&[1, 2, 0], Complex64::new(2.0, 0.0));
        b.set(&[1, 2, 1], Complex64::new(3.0, 0.0));
        c.set(&[0], Complex64::new(4.0, 0.0));
        c.set(&[1], Complex64::new(5.0, 0.0));

        // Contract the tensors
        let a = contract(&[0, 1], &[0, 1, 2], b, &[2], c);

        // Check result in A
        assert_eq!(a.get(&[0, 0]), Complex64::new(4.0, 0.0));
        assert_eq!(a.get(&[0, 1]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[0, 2]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[1, 0]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[1, 1]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[1, 2]), Complex64::new(23.0, 0.0));
    }

    #[test]
    fn test_contraction_to_scalar() {
        let b = Tensor::new_from_flat(
            &[2],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            None,
        );
        let c = Tensor::new_from_flat(
            &[2],
            vec![Complex64::new(4.0, 0.0), Complex64::new(5.0, 0.0)],
            None,
        );

        let a = contract(&[], &[0], b, &[0], c);

        let sol = Tensor::new_scalar(Complex64::new(14.0, 0.0));
        assert!(all_close(&a, &sol, 1e-12));
    }

    #[test]
    fn test_contraction_big_to_scalar() {
        let a_shape = [2, 3, 4];
        let b_shape = [4, 2, 3];
        let a_data = vec![
            Complex64::new(-5.2, 5.9),
            Complex64::new(-7.2, -7.8),
            Complex64::new(-4.9, 4.3),
            Complex64::new(-3.5, -6.8),
            Complex64::new(2.4, -2.3),
            Complex64::new(-9.0, -6.9),
            Complex64::new(3.5, 6.5),
            Complex64::new(6.4, -2.8),
            Complex64::new(3.4, 1.6),
            Complex64::new(7.0, -0.2),
            Complex64::new(6.6, 1.1),
            Complex64::new(-1.7, -3.9),
            Complex64::new(3.9, 2.7),
            Complex64::new(4.6, 1.4),
            Complex64::new(10.0, -3.6),
            Complex64::new(8.2, -7.4),
            Complex64::new(-1.7, -9.1),
            Complex64::new(-8.8, 1.1),
            Complex64::new(-8.7, 2.3),
            Complex64::new(-5.7, -7.5),
            Complex64::new(2.1, -9.2),
            Complex64::new(-3.2, -6.6),
            Complex64::new(-3.7, 1.2),
            Complex64::new(6.4, 5.2),
        ];
        let b_data = vec![
            Complex64::new(6.6, 8.1),
            Complex64::new(-5.7, 1.7),
            Complex64::new(5.0, -6.1),
            Complex64::new(-0.7, -4.3),
            Complex64::new(1.4, -2.2),
            Complex64::new(3.4, -6.2),
            Complex64::new(4.7, -1.8),
            Complex64::new(4.7, -7.1),
            Complex64::new(-9.3, 2.6),
            Complex64::new(-4.9, 1.1),
            Complex64::new(4.2, -9.0),
            Complex64::new(7.0, -3.1),
            Complex64::new(8.0, 1.9),
            Complex64::new(1.8, -8.4),
            Complex64::new(6.0, 1.0),
            Complex64::new(4.7, -0.6),
            Complex64::new(-4.3, 8.7),
            Complex64::new(-5.3, 8.5),
            Complex64::new(2.3, 1.7),
            Complex64::new(8.1, 7.8),
            Complex64::new(-1.9, 0.9),
            Complex64::new(3.4, 9.7),
            Complex64::new(-1.4, -6.7),
            Complex64::new(4.9, 5.8),
        ];

        let sol = Tensor::new_scalar(Complex64::new(-160.09, 54.36));
        let a = Tensor::new_from_flat(&a_shape, a_data, Some(Layout::RowMajor));
        let b = Tensor::new_from_flat(&b_shape, b_data, Some(Layout::RowMajor));
        let c = contract(&[], &[2, 0, 1], a, &[1, 2, 0], b);

        assert!(all_close(&c, &sol, 1e-12));
    }

    #[test]
    fn test_contraction_scalars_only() {
        let a = Tensor::new_scalar(Complex64::new(5.0, 3.0));
        let b = Tensor::new_scalar(Complex64::new(-2.0, 4.0));
        let c = contract(&[], &[], a, &[], b);

        let sol = Tensor::new_scalar(Complex64::new(-22.0, 14.0));
        assert!(all_close(&c, &sol, 1e-12));
    }

    #[test]
    fn test_contraction_with_scalar() {
        let a_data = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-3.0, -1.0),
            Complex64::new(0.0, 5.0),
        ];
        let sol_data = vec![
            Complex64::new(3.0, 1.0),
            Complex64::new(-4.0, 2.0),
            Complex64::new(-7.0, 1.0),
            Complex64::new(5.0, 10.0),
        ];
        let a = Tensor::new_from_flat(&[2, 2], a_data, None);
        let b = Tensor::new_scalar(Complex64::new(2.0, -1.0));

        let sol = Tensor::new_from_flat(&[2, 2], sol_data, None);
        let c = contract(&[0, 1], &[0, 1], a, &[], b);
        assert!(all_close(&c, &sol, 1e-12));
    }

    #[test]
    fn toy_contraction_transposed() {
        // Create tensors
        let mut b = Tensor::new(&[2, 3, 4]);
        let mut c = Tensor::new(&[4]);

        // Insert data into B and C
        b.set(&[0, 0, 0], Complex64::new(1.0, 0.0));
        b.set(&[1, 2, 0], Complex64::new(2.0, 0.0));
        b.set(&[1, 2, 1], Complex64::new(3.0, 0.0));
        c.set(&[0], Complex64::new(4.0, 0.0));
        c.set(&[1], Complex64::new(5.0, 0.0));

        // Contract the tensors
        let a = contract(&[1, 0], &[0, 1, 2], b, &[2], c);

        // Check result in A
        assert_eq!(a.get(&[0, 0]), Complex64::new(4.0, 0.0));
        assert_eq!(a.get(&[1, 0]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[2, 0]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[0, 1]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[1, 1]), Complex64::new(0.0, 0.0));
        assert_eq!(a.get(&[2, 1]), Complex64::new(23.0, 0.0));
    }

    #[test]
    fn simple_contraction() {
        // Create tensors
        let solution_data = vec![
            Complex64::new(1.1913917228026232, -3.7863595014806157),
            Complex64::new(1.5884274662744466, 1.1478771890194843),
        ];
        let b_data = vec![
            Complex64::new(1.764052345967664, -0.10321885179355784),
            Complex64::new(1.8675579901499675, 0.7610377251469934),
            Complex64::new(0.9787379841057392, 0.144043571160878),
            Complex64::new(0.9500884175255894, 0.44386323274542566),
            Complex64::new(0.4001572083672233, 0.41059850193837233),
            Complex64::new(-0.977277879876411, 0.12167501649282841),
            Complex64::new(2.240893199201458, 1.454273506962975),
            Complex64::new(-0.1513572082976979, 0.33367432737426683),
        ];
        let c_data = vec![
            Complex64::new(1.4940790731576061, -2.5529898158340787),
            Complex64::new(0.31306770165090136, 0.8644361988595057),
            Complex64::new(-0.20515826376580087, 0.6536185954403606),
            Complex64::new(-0.8540957393017248, -0.7421650204064419),
        ];

        let solution = Tensor::new_from_flat(&[2], solution_data, None);
        let b = Tensor::new_from_flat(&[2, 2, 2], b_data, None);
        let c = Tensor::new_from_flat(&[2, 2], c_data, None);

        // Contract the tensors
        let out = contract(&[2], &[1, 0, 2], b, &[0, 1], c);

        assert!(all_close(&out, &solution, 1e-12));
    }

    #[test]
    fn consecutive_contraction() {
        // Create tensors
        let solution_data = vec![
            Complex64::new(-10.98182728872986, -17.01067985192062),
            Complex64::new(-21.9947494831352, -1.467445780361377),
            Complex64::new(25.117181643463176, 19.919494278086834),
            Complex64::new(16.08302122450077, -30.446009864589634),
        ];
        let b_data = vec![
            Complex64::new(0.44122748688504143, 0.996439826913362),
            Complex64::new(-0.3588289470012431, 0.19766009104249851),
            Complex64::new(0.10960984157818278, 0.0032888429341100755),
            Complex64::new(1.1513910094871702, -0.3058530211666308),
            Complex64::new(0.18760322583703548, -0.0061949084857593475),
            Complex64::new(-0.9806078852186219, 0.269612406446701),
            Complex64::new(-0.33087015189408764, 0.7124212708765678),
            Complex64::new(0.6034716026094954, 1.3348485742415819),
            Complex64::new(1.5824811170615634, -0.10593044205742323),
            Complex64::new(1.8573310072313118, -0.47773141727821256),
            Complex64::new(-0.32986995777935924, -0.10106761180924467),
            Complex64::new(-0.8568531547160899, 1.2919633833879631),
            Complex64::new(2.43077118700778, 0.059144243219039896),
            Complex64::new(-1.6647885294716944, -0.08687560627763552),
            Complex64::new(-0.9092324048562419, 0.7930533194619698),
            Complex64::new(-1.5111795576883658, 0.1007381887528521),
            Complex64::new(-1.192764612421806, -0.05230815085185874),
            Complex64::new(-0.8718791832556535, 1.1393429788252842),
            Complex64::new(-0.2520921296030769, -0.3633108784819174),
            Complex64::new(-0.7001790376899514, 1.5615322934488904),
            Complex64::new(-0.5916366579302884, -0.6315716297922155),
            Complex64::new(0.6448475108927784, 0.3554384723493521),
            Complex64::new(-0.2048765105875873, 0.24921765856490757),
            Complex64::new(-0.4225079291623943, 0.49444039812108825),
        ];
        let c_data = vec![
            Complex64::new(-0.3363362591365529, 0.8196799200762225),
            Complex64::new(-0.5983599334221348, -0.9286265130171387),
            Complex64::new(-1.2323861148032735, -2.7745833607708863),
            Complex64::new(0.7471525922096975, -0.8747737971072579),
            Complex64::new(-0.3010051283841473, -0.1837994315203068),
            Complex64::new(0.9101105639068093, -0.20345360402062487),
            Complex64::new(0.6566194702604272, -1.491128858818253),
            Complex64::new(-0.7607603085956881, -0.07006702505266953),
            Complex64::new(-1.4812592014760095, 0.3460834685556485),
            Complex64::new(-0.8894207341116367, -0.6806540653541923),
            Complex64::new(1.2221705568780823, -0.12050664401555156),
            Complex64::new(-0.3679445685512342, -1.454693871103971),
            Complex64::new(-2.859687966622556, -0.693799346816814),
            Complex64::new(0.49355766095142994, 1.0032095598180708),
            Complex64::new(-1.6936750408421892, -1.3803322297834448),
            Complex64::new(-1.2796231822645914, -0.9472255560382364),
            Complex64::new(-1.2640833431434955, -0.5494973323633916),
            Complex64::new(-0.6233722375263352, 0.7905936128970772),
            Complex64::new(1.4133980179217838, -0.341615035898709),
            Complex64::new(2.088514687597715, 0.8596459517628046),
            Complex64::new(0.04595522410229037, -0.8756859169243153),
            Complex64::new(1.7384488142376797, -0.13230195045629967),
            Complex64::new(0.49852362361856384, -0.9342785949904113),
            Complex64::new(0.49961804386532443, 0.06540955928016469),
            Complex64::new(-0.4357039190284899, -0.375707409780949),
            Complex64::new(1.1415077374346698, 0.6261926072814115),
            Complex64::new(1.4507335435185753, 0.42716401996388864),
            Complex64::new(1.4021666203758465, 1.140495938093696),
            Complex64::new(0.9562618573807691, 0.42386759329166346),
            Complex64::new(-1.0295734942351702, 1.6169788055097944),
            Complex64::new(-1.8777408839472627, -0.1452001865620599),
            Complex64::new(0.7924226173066341, 0.048944630112707777),
            Complex64::new(-0.4089923380377764, 0.4258991675344542),
            Complex64::new(-0.6463565855238054, 1.258073606676455),
            Complex64::new(0.24981731592122017, -0.6973458057904679),
            Complex64::new(-0.1225300926941241, -0.8970939975073631),
            Complex64::new(-1.3107731333797457, -0.13421699428097064),
            Complex64::new(0.7457269500324553, 0.7149499462374368),
            Complex64::new(0.8161323611566395, -1.2782156990100408),
            Complex64::new(1.212283405544346, -0.1877937061962787),
            Complex64::new(1.443881103376141, 2.3342400950999207),
            Complex64::new(-0.6105164006371435, 0.6837433172190976),
            Complex64::new(-0.24071114226406937, 0.04507815808386408),
            Complex64::new(-0.07915136118314284, -1.302621431749788),
            Complex64::new(-0.37562084407857, 0.6528930147774489),
            Complex64::new(-1.0650326193820066, -0.5302052748979336),
            Complex64::new(-0.14312641765714437, -0.8568681483200288),
            Complex64::new(-0.6105673577910354, 1.343946134660979),
            Complex64::new(1.8213647385840686, 0.6962219882267049),
            Complex64::new(-1.7537408604770464, -0.0419363501262477),
            Complex64::new(-0.4075191652021827, 0.5629625132005759),
            Complex64::new(0.21054245602828903, 0.27736011743265165),
            Complex64::new(0.8359439567722043, 1.639212334793982),
            Complex64::new(-0.23255622381537228, 0.5351551526217296),
            Complex64::new(-0.5775132332314081, -0.017353268952074445),
            Complex64::new(1.0184211331879478, -0.5404819579907136),
            Complex64::new(-2.3356218209439037, 1.1434278876345871),
            Complex64::new(-1.59969850388824, -3.283915447626658),
            Complex64::new(0.4677052146287663, -0.6470684773540814),
            Complex64::new(0.19443737779969736, 1.7543363242448964),
            Complex64::new(0.568132717730888, -0.31313850719663144),
            Complex64::new(-0.5934027710431155, 1.4429500100967874),
            Complex64::new(-1.7811512980070203, -0.4759508306463407),
            Complex64::new(0.3056001722083279, -0.36987212667207214),
            Complex64::new(-1.03849524010936, -0.1658160181188383),
            Complex64::new(-0.3493081255486548, -0.9105193539261216),
            Complex64::new(-1.164701910115395, 1.2083780731519589),
            Complex64::new(0.49832921470025815, -1.500859400279106),
            Complex64::new(0.9275162079169716, 0.8008188507530136),
            Complex64::new(-0.567051168267193, -0.24043683490700307),
            Complex64::new(-0.9955962989131678, 1.513645767997689),
            Complex64::new(-0.4464567266925109, 0.6849859143781624),
            Complex64::new(-0.1071639848241478, -1.807916624216005),
            Complex64::new(-0.7104664469959762, 0.7118892605186535),
            Complex64::new(0.45765807158631955, 0.895000938790005),
            Complex64::new(-0.09383244866604305, 0.9503447080453289),
            Complex64::new(-0.05935198087781362, 0.14082715272159635),
            Complex64::new(0.9325605063169753, 0.3758147854114012),
            Complex64::new(1.189060725824544, -1.809551228823682),
            Complex64::new(-0.44542999149860524, 0.6288803314609249),
            Complex64::new(-0.36840953098509416, 1.3592009708924977),
            Complex64::new(2.2332708137448996, -0.045670297058469644),
            Complex64::new(0.4763461763468124, -0.24946987656467687),
            Complex64::new(-1.1212876008978485, 1.7763221291484148),
            Complex64::new(0.8726546212518307, -0.16636061136497707),
            Complex64::new(-0.8464686168782807, -0.7150812419779358),
            Complex64::new(2.0736155286567595, 2.0636706641081846),
            Complex64::new(-0.510256056076286, -1.6296783988972459),
            Complex64::new(0.024210744386125548, -1.0620979588883213),
            Complex64::new(-0.3685447790808469, -0.5774299344746378),
            Complex64::new(-0.10061434630710828, 0.5316337242935001),
            Complex64::new(-0.24341970124625367, -0.03547248694616136),
            Complex64::new(0.7269532606897998, 1.0786281312529078),
            Complex64::new(1.4630954796733577, -0.5570793806175268),
            Complex64::new(-1.1184064312813717, 1.2147377284041538),
            Complex64::new(0.6178447508392777, -1.0075475280783048),
            Complex64::new(0.12480682533019469, -1.2821074800619094),
            Complex64::new(-0.7111632283050703, 0.29822712172625787),
            Complex64::new(-0.01694531772769253, -0.5063101292887338),
            Complex64::new(-1.3009514516955427, -0.6245033904425045),
            Complex64::new(-0.8139120077465551, -0.5788257825218013),
            Complex64::new(2.111488404373246, -0.14453465978140453),
            Complex64::new(0.7893664041799795, 0.419197757640167),
            Complex64::new(0.5004873274970344, -0.7781776086936343),
            Complex64::new(0.30364846530823975, 2.101776683209765),
            Complex64::new(0.03654264148725312, -0.8269874183593646),
            Complex64::new(-0.5732155560138283, 0.37108695058486196),
            Complex64::new(-1.0807019522462529, 1.1315139737358737),
            Complex64::new(0.2212541228509997, -0.93090048052683),
            Complex64::new(0.34691932708774675, -0.28749660960720785),
            Complex64::new(-0.48713264551119473, -0.5381093191951732),
            Complex64::new(1.4652048844331407, -0.3792249852036208),
            Complex64::new(-0.706093869937341, 0.9118624130029944),
            Complex64::new(-1.1515442452754336, 1.2864457382920482),
            Complex64::new(0.9721793096791724, 0.03464387576779198),
            Complex64::new(-0.5017555471945, 0.46743205826926665),
            Complex64::new(0.606870319141853, 2.5885695856409807),
            Complex64::new(0.4651009937136952, -0.24772894321806252),
            Complex64::new(-0.6385105554205018, -0.387331363349994),
            Complex64::new(-1.3362803092215407, -1.7310536281125464),
            Complex64::new(1.5387561454090368, 0.9638012998820848),
            Complex64::new(0.1707644524655671, -2.026401888498568),
            Complex64::new(-0.754830590411252, 0.1292002290299064),
            Complex64::new(0.5485678368565737, -1.1388902582429323),
            Complex64::new(1.1472020816149164, -1.2633011631661237),
            Complex64::new(-0.8077430955509538, 0.6327118619482441),
            Complex64::new(-0.689565232048181, 0.8325936130243294),
            Complex64::new(0.7769075911230111, -0.8034526008152093),
            Complex64::new(-0.28143012121166766, -0.9927694461542396),
            Complex64::new(-0.6346524996166139, 0.3154661502200849),
            Complex64::new(0.44295626086393586, 0.21653195567532169),
            Complex64::new(-1.7038854075200234, 0.826244795935599),
            Complex64::new(-0.8241234534171207, -0.7637468928463446),
            Complex64::new(-0.6928263420058225, -0.5700896497000028),
            Complex64::new(-1.4219245490984462, 0.5058397918952761),
            Complex64::new(0.390420608044887, 1.8557144042983549),
            Complex64::new(-0.2241898325725227, 0.8153138930730877),
            Complex64::new(0.5246942624732844, 0.26153957389632215),
            Complex64::new(-0.42703138590481854, 0.8994054623758969),
            Complex64::new(0.6302964765895014, 1.0745427793405484),
            Complex64::new(-0.8126247611063044, 1.0967647243543643),
            Complex64::new(0.3465017504759862, -1.1944459644165344),
            Complex64::new(0.2874036450384292, -1.2401446359099324),
            Complex64::new(1.3351503366259017, -0.47415241031291133),
            Complex64::new(1.1522047703756748, -0.12765822195358442),
            Complex64::new(1.0613514386873795, 0.5461774263683895),
            Complex64::new(-1.167278449710173, -0.5090725984592255),
            Complex64::new(0.877152812186743, -0.7786932527232613),
            Complex64::new(0.10134479014204936, 0.47870605481716055),
            Complex64::new(0.028241248478617587, 0.21456398452907421),
            Complex64::new(0.012758316706164157, -0.6069895355762209),
            Complex64::new(0.7882379435396791, -1.2439951342219098),
            Complex64::new(-0.7479057873130808, 1.010908739659707),
            Complex64::new(0.5218494903163309, 0.684246816180101),
            Complex64::new(-0.17170904843929477, 2.6453534453939627),
            Complex64::new(-0.4407384626484675, 0.06546949075634184),
            Complex64::new(-1.3970740246682614, 0.6932537043123401),
            Complex64::new(1.018137609564609, 1.1857444344717476),
            Complex64::new(1.6399540662187597, -0.22308326678466056),
            Complex64::new(0.41367880834311616, -0.03720826581577405),
            Complex64::new(0.9068894675659355, 0.14105656656999402),
            Complex64::new(-0.04978868431154955, -0.36119418525022573),
            Complex64::new(2.2601067737202705, 0.9939689768716928),
            Complex64::new(-0.2151878024236405, 1.1971544902820535),
            Complex64::new(2.2379656117362696, -0.17620336617922988),
            Complex64::new(-0.05567103468791388, 0.5630013013692811),
            Complex64::new(-2.386697744379377, -0.20999860102176093),
            Complex64::new(0.21204331876967614, -0.22598416550092423),
            Complex64::new(-0.07359331874884148, -0.25551774227104695),
            Complex64::new(-0.48212018670721907, -0.42559212626742415),
            Complex64::new(-2.2491181284434787, -1.7020899666109865),
            Complex64::new(-0.034702101230658505, 0.05078664982703555),
            Complex64::new(-0.414178265587903, -0.7173378949371145),
            Complex64::new(1.4728447297202483, -0.5731331883747963),
            Complex64::new(-0.2021181776204691, 1.3531131105069851),
            Complex64::new(2.5208076341549916, -0.051238525075944674),
            Complex64::new(0.7097978597938108, 0.012623501414499815),
            Complex64::new(0.5172593547162142, 1.1348633753477662),
            Complex64::new(-0.23998173469104078, 1.66638040156906),
            Complex64::new(-0.1917395684517187, -1.2347294991323687),
        ];
        let d_data = vec![
            Complex64::new(0.38547989325078796, -0.9329844355645641),
            Complex64::new(-0.7892390153216955, 1.2968208278933389),
            Complex64::new(0.1580651440379482, 0.20232289958002841),
            Complex64::new(1.6943283234312494, -0.2862162334955482),
            Complex64::new(1.3587072302870447, -1.2345043335155528),
            Complex64::new(-1.2761651609652445, -1.0246845857826614),
            Complex64::new(-0.4291899924545752, -0.34072388649508545),
            Complex64::new(0.7299598177268846, 0.7171924982042366),
            Complex64::new(-1.2032406653183243, 0.6786995494112171),
            Complex64::new(-0.19894722418706615, -0.2516369659275783),
            Complex64::new(0.6044376841910644, -0.29138373415184593),
            Complex64::new(-0.24030333213159447, 0.8898821682587077),
            Complex64::new(-0.5589262707450762, -0.07013112933872342),
            Complex64::new(-0.8138818675836827, 0.6977711073598102),
            Complex64::new(1.9535886796522515, 1.3425118797394124),
            Complex64::new(-0.683525678795807, -1.2084468614330979),
            Complex64::new(-1.069415624506323, 0.12135718150184839),
            Complex64::new(2.2409535697615475, -0.18598246728492965),
            Complex64::new(-1.1489999774005422, -1.5055231525834694),
            Complex64::new(1.4448594976242957, -0.8044978415709303),
            Complex64::new(-1.4406334980112179, -0.9993307344363753),
            Complex64::new(-0.018998122559880832, -0.06779508465536141),
            Complex64::new(-0.6741897995854045, -0.41143385602725513),
            Complex64::new(-1.0574619153949179, 0.3722697794487287),
            Complex64::new(-1.36515577626505, -0.10507983333344491),
            Complex64::new(0.40825945846606926, -0.1450517808176537),
            Complex64::new(0.5340751094696318, 1.6995480900305686),
            Complex64::new(0.9156625953165883, -0.2281859843793368),
            Complex64::new(0.20340805274695156, 0.7792603519294811),
            Complex64::new(1.1605590114481037, -1.8451851363285414),
        ];

        let solution = Tensor::new_from_flat(&[4], solution_data, None);
        let b = Tensor::new_from_flat(&[1, 2, 3, 4], b_data, None);
        let c = Tensor::new_from_flat(&[6, 3, 5, 2], c_data, None);
        let d = Tensor::new_from_flat(&[6, 5, 1], d_data, None);

        // Contract the tensors
        let out1 = contract(&[5, 3, 0, 4], &[0, 1, 2, 3], b, &[5, 2, 4, 1], c);
        let out2 = contract(&[3], &[5, 4, 0], d, &[5, 3, 0, 4], out1);

        assert!(all_close(&out2, &solution, 1e-12));
    }
}
