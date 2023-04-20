use std::collections::HashSet;

extern crate openblas_src;
use cblas_sys::{cblas_zgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use hptt_sys::transpose_simple;
use itertools::Itertools;
use num_complex::Complex64;
use permutation::Permutation;
use std::iter::zip;

pub mod permutation;

pub mod decomposition;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layout {
    RowMajor,
    ColumnMajor,
}

/// A tensor of arbitrary dimensions containing complex64 values.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// The shape of the tensor.
    shape: Vec<u32>,

    /// The current inverse permutation. Using the inverse is easier, because it
    /// maps from a given output element to the corresponding input element. E.g.,
    /// getting the size of a permutated axis is just `shape[inv_permutation[axis]]`.
    /// The original permutation, on the other hand, maps a given input element to
    /// the corresponding output output element. The above example would be similar
    /// to `shape[permutation.index_of(axis)]`.
    inv_permutation: Permutation,

    /// The tensor data in column-major order.
    data: Vec<Complex64>,
}

impl Tensor {
    /// Creates a new tensor of the given dimensions.
    /// The tensor is initialized with zeros.
    ///
    /// # Panics
    /// - Panics if the dimensions are empty
    #[must_use]
    pub fn new(dimensions: &[u32]) -> Self {
        // Validity checks
        assert!(!dimensions.is_empty());

        // Construct tensor
        let total_items = dimensions.iter().product::<u32>();
        Self {
            shape: dimensions.to_vec(),
            inv_permutation: Permutation::identity(dimensions.len()),
            data: vec![Complex64::new(0.0, 0.0); total_items.try_into().unwrap()],
        }
    }

    /// Creates a new tensor with the given dimensions and the corresponding data. Assumes data is
    /// column major unless otherwise specified.
    ///
    /// # Panics
    /// - Panics if the dimensions are empty
    /// - Panics if the length of the data does not match with the dimensions given
    #[must_use]
    pub fn new_from_flat(dimensions: &[u32], data: Vec<Complex64>, layout: Option<Layout>) -> Self {
        // Validity checks
        assert!(!dimensions.is_empty());
        let total_items: usize = dimensions.iter().product::<u32>().try_into().unwrap();
        assert_eq!(total_items, data.len());

        let inv_permutation = if let Some(layout) = layout {
            if layout == Layout::RowMajor {
                Permutation::new((0..dimensions.len()).rev().collect())
            } else {
                Permutation::identity(dimensions.len())
            }
        } else {
            Permutation::identity(dimensions.len())
        };

        // Construct tensor
        Self {
            shape: dimensions.to_vec(),
            inv_permutation,
            data,
        }
    }

    /// Creates a new tensor without actual data.
    ///
    /// # Panics
    /// - Panics if the dimensions are empty
    fn new_uninitialized(dimensions: &[u32]) -> Self {
        // Validity checks
        assert!(!dimensions.is_empty());

        // Construct tensor
        let total_items = dimensions.iter().product::<u32>();
        Self {
            shape: dimensions.to_vec(),
            inv_permutation: Permutation::identity(dimensions.len()),
            data: Vec::with_capacity(total_items.try_into().unwrap()),
        }
    }

    /// Computes the flat index given the accessed coordinates.
    /// Assumes column-major ordering.
    ///
    /// # Panics
    /// Panics if the coordinates are invalid.
    fn compute_index(&self, coordinates: &[u32]) -> usize {
        // Get the unpermuted coordinates
        let dims = self.inv_permutation.apply(coordinates);

        // Validate coordinates
        assert_eq!(dims.len(), self.shape.len());
        for (i, &dim_i) in dims.iter().enumerate() {
            assert!(dim_i < self.shape[i]);
        }

        // Compute index
        let mut idx = dims[dims.len() - 1];
        for i in (0..dims.len() - 1).rev() {
            idx = dims[i] + self.shape[i] * idx;
        }
        idx.try_into().unwrap()
    }

    /// Inserts a value at the given position.
    pub fn insert(&mut self, coordinates: &[u32], value: Complex64) {
        let idx = self.compute_index(coordinates);
        self.data[idx] = value;
    }

    /// Gets the value at the given position.
    #[must_use]
    pub fn get(&self, coordinates: &[u32]) -> Complex64 {
        let idx = self.compute_index(coordinates);
        self.data[idx]
    }

    /// Returns a copy of the current shape.
    ///
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// # use tetra::permutation::Permutation;
    /// let mut t = Tensor::new(&[3, 2, 5, 4, 1]);
    /// assert_eq!(t.shape(), vec![3, 2, 5, 4, 1]);
    /// t.transpose(&Permutation::new(vec![3, 1, 4, 0, 2]));
    /// assert_eq!(t.shape(), vec![4, 2, 1, 3, 5]);
    /// ```
    #[must_use]
    pub fn shape(&self) -> Vec<u32> {
        self.inv_permutation.apply_inverse(&self.shape)
    }

    /// Returns the size of a single axis or of the whole tensor.
    ///
    /// # Panics
    /// Panics if the total size is requested and it is larger than u32.
    ///
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// let t = Tensor::new(&[1, 3, 5]);
    /// assert_eq!(t.size(None), 15);
    /// assert_eq!(t.size(Some(1)), 3);
    /// assert_eq!(t.size(Some(2)), 5);
    /// ```
    #[must_use]
    pub fn size(&self, axis: Option<usize>) -> u32 {
        if let Some(axis) = axis {
            self.shape[self.inv_permutation[axis]]
        } else {
            self.data.len().try_into().unwrap()
        }
    }

    /// Returns the number of dimensions / axes of the tensor.
    ///
    /// # Examples
    /// ```
    /// # use tetra::Tensor;
    /// assert_eq!(Tensor::new(&[1, 2]).ndim(), 2);
    /// assert_eq!(Tensor::new(&[1, 3, 6, 5]).ndim(), 4);
    /// ```
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Transposes the tensor axes according to the permutation.
    /// This method does not modify the data but only the view, hence it's zero cost.
    /// The permutation is interpreted as an inverse permutation wich matches the
    /// numpy convention.
    pub fn transpose(&mut self, inv_permutation: &Permutation) {
        self.inv_permutation = &self.inv_permutation * inv_permutation;
    }

    /// Computes the transposed data based on the current permutation.
    fn compute_transposed_data(&self, permutation: &Permutation) -> Vec<Complex64> {
        // Get the permutation as [i32]
        let perm: Vec<_> = permutation
            .order()
            .iter()
            .map(|x| (*x).try_into().unwrap())
            .collect();

        // Get the shape as [i32]
        let shape: Vec<_> = self
            .shape
            .iter()
            .map(|x| (*x).try_into().unwrap())
            .collect();

        // Transpose data and shape
        transpose_simple(&perm, &self.data, &shape)
    }

    /// Actually transposes the underlying data according to the current axis permutation.
    /// This should not affect the tensor as observable from the outside (e.g. shape(),
    /// size(), get() and similar should show no difference).
    fn materialize_transpose(&mut self) {
        self.data = self.compute_transposed_data(&self.inv_permutation);
        self.shape = self.inv_permutation.apply_inverse(&self.shape);
        self.inv_permutation = Permutation::identity(self.shape.len());
    }

    /// Creates the transposed tensor. Performs a full data copy.
    /// The permutation is interpreted as an inverse permutation wich matches the
    /// numpy convention.
    #[must_use]
    pub fn transposed(&self, inv_permutation: &Permutation) -> Self {
        let perm = &self.inv_permutation * inv_permutation;
        let data = self.compute_transposed_data(&perm);
        let shape = perm.apply_inverse(&self.shape);
        Self::new_from_flat(&shape, data, None)
    }
}

/// Contracts two tensors a and b, writing the result to the out tensor.
/// The indices specify which legs are to be contracted (like einsum notation). So if
/// two tensors share an index, the corresponding dimension is contracted.
///
/// # Panics
/// - Panics if contracted sizes don't match
#[must_use]
pub fn contract(
    out_indices: &[u32],
    a_indices: &[u32],
    a: &Tensor,
    b_indices: &[u32],
    b: &Tensor,
) -> Tensor {
    assert_eq!(a_indices.len(), a.shape.len());
    assert_eq!(b_indices.len(), b.shape.len());

    // Find contracted indices
    let b_legs = b_indices.iter().copied().collect::<HashSet<_>>();
    let contracted = a_indices
        .iter()
        .filter(|idx| b_legs.contains(idx))
        .copied()
        .collect::<HashSet<_>>();

    // Find hyperedges
    let hyperedges = contracted
        .iter()
        .filter(|idx| out_indices.contains(idx))
        .copied()
        .collect::<HashSet<_>>();

    let mut remaining = Vec::with_capacity(
        a_indices.len() + b_indices.len() - 2 * contracted.len() + hyperedges.len(),
    );

    // Keeps track of order of contracted edges
    let mut contract_order = Vec::with_capacity(contracted.len() - hyperedges.len());

    // Keeps track of order of hyperedges
    let mut hyperedge_order = Vec::with_capacity(hyperedges.len());
    let mut hyperedge_size = Vec::with_capacity(hyperedges.len());

    // Compute permutation, total size of contracted dimensions and total size of remaining dimensions for A
    let mut a_contracted = 0;
    let mut a_remaining = 0;
    let mut a_hyperedges = 0;
    let mut a_contracted_size = 1;
    let mut a_remaining_size = 1;
    let mut a_perm = vec![0; a_indices.len()];

    for (i, idx) in a_indices.iter().enumerate() {
        if hyperedges.contains(idx) {
            a_perm[a_indices.len() - hyperedges.len() + a_hyperedges] = i;
            a_hyperedges += 1;
            hyperedge_order.push(*idx);
            hyperedge_size.push(a.size(Some(i)));
        } else if contracted.contains(idx) {
            a_perm[(a_indices.len() - contracted.len()) + a_contracted] = i;
            contract_order.push(*idx);
            a_contracted_size *= a.size(Some(i));
            a_contracted += 1;
        } else {
            a_perm[a_remaining] = i;
            a_remaining += 1;
            a_remaining_size *= a.size(Some(i));
            remaining.push(*idx);
        }
    }

    // Get transposed A
    let a_transposed = a.transposed(&Permutation::new(a_perm));

    // Compute permutation, total size of contracted dimensions and total size of remaining dimensions for B
    let mut b_remaining = 0;
    let mut b_contracted_size = 1;
    let mut b_remaining_size = 1;
    let mut b_perm = vec![0; b_indices.len()];
    for (i, idx) in b_indices.iter().enumerate() {
        if hyperedges.contains(idx) {
            b_perm[hyperedge_order.iter().position(|e| *e == *idx).unwrap() + b_indices.len()
                - hyperedges.len()] = i;
        } else if contracted.contains(idx) {
            b_perm[contract_order.iter().position(|e| *e == *idx).unwrap()] = i;
            b_contracted_size *= b.size(Some(i));
        } else {
            b_perm[contracted.len() + b_remaining] = i;
            b_remaining += 1;
            b_remaining_size *= b.size(Some(i));
            remaining.push(*idx);
        }
    }

    // Get transposed B
    let b_transposed = b.transposed(&Permutation::new(b_perm));

    // Make sure the connecting matrix dimensions match
    assert_eq!(a_contracted_size, b_contracted_size);

    // Compute the shape of C based on the remaining indices
    let mut c_shape = Vec::with_capacity(remaining.len() + hyperedges.len());
    for r in &remaining {
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

    // Determine chunk size when performing hyperedge contraction
    let a_chunk_size = (a_contracted_size * a_remaining_size) as usize;
    let b_chunk_size = (b_contracted_size * b_remaining_size) as usize;
    let c_chunk_size = c_shape.iter().product::<u32>() as usize;

    for (hyperedge_size, hyperedge_index) in zip(&hyperedge_size, &hyperedge_order) {
        c_shape.push(*hyperedge_size);
        remaining.push(*hyperedge_index);
    }
    // Create output tensor
    let mut out = Tensor::new_uninitialized(&c_shape);

    let hyperedge_iter = if hyperedge_size.is_empty() {
        [0..1].into_iter().multi_cartesian_product()
    } else {
        hyperedge_size
            .clone()
            .iter()
            .map(|&e| 0..e)
            .multi_cartesian_product()
    };

    for dim in hyperedge_iter {
        let mut index: usize = 0;
        for (i, size) in zip(dim, &hyperedge_size) {
            index = (index + i as usize) * (*size) as usize;
        }
        if !hyperedge_size.is_empty() {
            index /= hyperedge_size[hyperedge_size.len() - 1] as usize;
        }

        // Compute ZGEMM
        unsafe {
            let out_start = out.data.as_mut_ptr();
            let out_chunk_start = out_start.add(index * c_chunk_size);

            // Make sure that we are not writing past the allocated memory
            let out_chunk_end = out_chunk_start.add(c_chunk_size);
            assert!(
                usize::try_from(out_chunk_end.offset_from(out_start)).unwrap()
                    <= out.data.capacity()
            );

            // Perform matrix-matrix multiplication
            cblas_zgemm(
                CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                a_remaining_size.try_into().unwrap(),
                b_remaining_size.try_into().unwrap(),
                b_contracted_size.try_into().unwrap(),
                &Complex64::new(1.0, 0.0) as *const _ as *const _,
                a_transposed.data[index * a_chunk_size..(index + 1) * a_chunk_size].as_ptr()
                    as *const _,
                a_remaining_size.try_into().unwrap(),
                b_transposed.data[index * b_chunk_size..(index + 1) * b_chunk_size].as_ptr()
                    as *const _,
                b_contracted_size.try_into().unwrap(),
                &Complex64::new(0.0, 0.0) as *const _ as *const _,
                out_chunk_start as *mut _,
                a_remaining_size.try_into().unwrap(),
            );
        }
    }

    // Update length as full vector is now initialized
    unsafe {
        out.data.set_len(out.data.capacity());
    }

    // Find permutation for output tensor
    let c_perm = Permutation::between(&remaining, out_indices);

    // Return transposed output tensor
    out.transpose(&c_perm);
    out
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use float_cmp::assert_approx_eq;

    use super::*;

    fn assert_tensors_equal(left: &mut Tensor, right: &mut Tensor) {
        assert_eq!(left.data.len(), right.data.len());
        assert_eq!(left.shape(), right.shape());

        left.materialize_transpose();
        right.materialize_transpose();
        for (va, vb) in zip(&left.data, &right.data) {
            assert_approx_eq!(f64, va.re, vb.re, epsilon = 1e-14);
            assert_approx_eq!(f64, va.im, vb.im, epsilon = 1e-14);
        }
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
    fn test_single_transpose() {
        let mut a = Tensor::new(&[2, 3, 4]);
        a.insert(&[0, 0, 0], Complex64::new(1.0, 2.0));
        a.insert(&[0, 1, 3], Complex64::new(0.0, -1.0));
        a.insert(&[1, 2, 1], Complex64::new(-5.0, 0.0));

        a.transpose(&Permutation::new(vec![1, 2, 0]));
        assert_eq!(a.shape(), vec![3, 4, 2]);
        assert_eq!(a.get(&[0, 0, 0]), Complex64::new(1.0, 2.0));
        assert_eq!(a.get(&[1, 3, 0]), Complex64::new(0.0, -1.0));
        assert_eq!(a.get(&[2, 1, 1]), Complex64::new(-5.0, 0.0));

        a.transpose(&Permutation::new(vec![1, 2, 0]));
        assert_eq!(a.shape(), vec![4, 2, 3]);
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

        a.transpose(&Permutation::new(vec![1, 2, 0, 3]));
        assert_eq!(a.shape(), vec![3, 4, 2, 5]);
        assert_eq!(a.get(&[0, 0, 0, 1]), Complex64::new(1.0, 2.0));
        assert_eq!(a.get(&[1, 3, 0, 2]), Complex64::new(0.0, -1.0));
        assert_eq!(a.get(&[2, 1, 1, 4]), Complex64::new(-5.0, 0.0));
        a.materialize_transpose();
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

        let mut solution = Tensor::new_from_flat(&[2], solution_data, None);
        let b = Tensor::new_from_flat(&[2, 2, 2], b_data, None);
        let c = Tensor::new_from_flat(&[2, 2], c_data, None);

        // Contract the tensors
        let mut out = contract(&[2], &[1, 0, 2], &b, &[0, 1], &c);

        assert_tensors_equal(&mut out, &mut solution);
    }

    #[test]
    fn contraction_one_hyperedge_one_normal() {
        let b_data = vec![
            Complex64::new(0.44122748688504143, -0.9092324048562419),
            Complex64::new(2.43077118700778, 0.18760322583703548),
            Complex64::new(0.10960984157818278, -1.192764612421806),
            Complex64::new(-0.33087015189408764, -0.5916366579302884),
            Complex64::new(-0.2520921296030769, -0.32986995777935924),
            Complex64::new(1.5824811170615634, -0.2048765105875873),
        ];
        let c_data = vec![
            Complex64::new(-0.3588289470012431, 0.7930533194619698),
            Complex64::new(-1.5111795576883658, 0.19766009104249851),
            Complex64::new(0.996439826913362, 0.1007381887528521),
            Complex64::new(-0.7001790376899514, -0.10106761180924467),
            Complex64::new(-0.8568531547160899, 1.5615322934488904),
            Complex64::new(-0.3633108784819174, 1.2919633833879631),
            Complex64::new(0.6034716026094954, -0.6315716297922155),
            Complex64::new(0.6448475108927784, 1.3348485742415819),
            Complex64::new(0.7124212708765678, 0.3554384723493521),
            Complex64::new(1.1513910094871702, -0.05230815085185874),
            Complex64::new(-0.8718791832556535, -0.3058530211666308),
            Complex64::new(0.0032888429341100755, 1.1393429788252842),
            Complex64::new(-1.6647885294716944, -0.0061949084857593475),
            Complex64::new(-0.9806078852186219, -0.08687560627763552),
            Complex64::new(0.059144243219039896, 0.269612406446701),
            Complex64::new(1.8573310072313118, 0.24921765856490757),
            Complex64::new(-0.4225079291623943, -0.47773141727821256),
            Complex64::new(-0.10593044205742323, 0.49444039812108825),
        ];
        let solution_data = vec![
            Complex64::new(0.734617622804381, 1.1238676714086437),
            Complex64::new(1.2778747597962676, -1.7886157314230977),
            Complex64::new(2.8003838167444983, 1.998880565131441),
            Complex64::new(0.7203118822499492, 1.4515077983070295),
            Complex64::new(1.4359570078447461, 3.7303968153219964),
            Complex64::new(-0.9775931892780798, 0.4906729660620053),
            Complex64::new(1.4158326901458151, -1.0740710092306176),
            Complex64::new(2.0400576231610157, 0.7093378375520341),
            Complex64::new(0.2617332277917399, 0.7631522656907458),
        ];

        let mut solution = Tensor::new_from_flat(&[3, 3, 1], solution_data, None);
        let b = Tensor::new_from_flat(&[1, 3, 2, 1], b_data, None);
        let c = Tensor::new_from_flat(&[3, 2, 1, 3], c_data, None);

        let mut out = contract(&[1, 4, 0], &[0, 1, 2, 3], &b, &[4, 2, 3, 1], &c);

        assert_tensors_equal(&mut out, &mut solution);
    }

    #[test]
    fn contraction_three_hyperedges_only() {
        let b_data = vec![
            Complex64::new(0.0525795517940905, 0.5940587423078663),
            Complex64::new(0.7525040444615576, 0.8038480520516861),
            Complex64::new(0.1128332539895021, 0.4446153013757889),
            Complex64::new(0.0785807572180043, 0.9935954700500143),
            Complex64::new(0.8185816938998588, 0.96707376127355),
            Complex64::new(0.9472854477212035, 0.3758739972788238),
            Complex64::new(0.9777265602588574, 0.4480678490565591),
            Complex64::new(0.5623477769917304, 0.9065030582228698),
            Complex64::new(0.7653505709096636, 0.504100201624668),
            Complex64::new(0.1532635445497977, 0.8871822684786365),
            Complex64::new(0.2734451624358636, 0.2314999123816396),
            Complex64::new(0.110870616468761, 0.7427655071692136),
            Complex64::new(0.1222575838728678, 0.4014008258759437),
            Complex64::new(0.1930475575523624, 0.7990057760304885),
            Complex64::new(0.8725944878654819, 0.2241634833297647),
            Complex64::new(0.2154670526575549, 0.8110019216680409),
        ];

        let c_data = vec![
            Complex64::new(0.8445169236441942, 0.3025958478792655),
            Complex64::new(0.1329375757528354, 0.696460648419754),
            Complex64::new(0.520511455016131, 0.5238746346974953),
            Complex64::new(0.8246339363179194, 0.8959130730289593),
            Complex64::new(0.5946174680932, 0.8125272231801355),
            Complex64::new(0.4113544242326473, 0.7506886864236262),
            Complex64::new(0.6973939119766687, 0.0727243012099578),
            Complex64::new(0.5015356755349253, 0.2851927366478476),
        ];

        let solution_data = vec![
            Complex64::new(-0.1353553874910031, 0.5176030155740229),
            Complex64::new(-0.1741916773808778, 0.2905378371570276),
            Complex64::new(-0.2056963759446933, 1.089412115120795),
            Complex64::new(-0.017456794562414, 0.698642162437705),
            Complex64::new(-0.5647085529036102, 0.6986703787159183),
            Complex64::new(0.4048366584551033, 1.2454499612885606),
            Complex64::new(0.1075057026532834, 0.8657339001527011),
            Complex64::new(0.0235093842648876, 0.6150211251483878),
            Complex64::new(0.4938128817272317, 0.6573130564137721),
            Complex64::new(0.02105440733518, 0.2637493408107594),
            Complex64::new(-0.6297265642504097, 0.6520648764877406),
            Complex64::new(0.0233033904706731, 0.5260631308334801),
            Complex64::new(-0.2633072526483266, 0.1385088488354639),
            Complex64::new(0.5187400321269859, 0.9666216247686549),
            Complex64::new(-0.5203936295667466, 0.4735931783538535),
            Complex64::new(-0.1232274436570458, 0.4681960350487574),
        ];

        let mut solution =
            Tensor::new_from_flat(&[2, 2, 2, 2], solution_data, Some(Layout::RowMajor));
        let b = Tensor::new_from_flat(&[2, 2, 2, 2], b_data, Some(Layout::RowMajor));
        let c = Tensor::new_from_flat(&[2, 2, 2], c_data, Some(Layout::RowMajor));

        let mut out = contract(&[1, 3, 2, 0], &[1, 3, 0, 2], &b, &[2, 0, 3], &c);
        assert_tensors_equal(&mut out, &mut solution);
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

        let mut solution = Tensor::new_from_flat(&[4], solution_data, None);
        let b = Tensor::new_from_flat(&[1, 2, 3, 4], b_data, None);
        let c = Tensor::new_from_flat(&[6, 3, 5, 2], c_data, None);
        let d = Tensor::new_from_flat(&[6, 5, 1], d_data, None);

        // Contract the tensors
        let out1 = contract(&[5, 3, 0, 4], &[0, 1, 2, 3], &b, &[5, 2, 4, 1], &c);
        let mut out2 = contract(&[3], &[5, 4, 0], &d, &[5, 3, 0, 4], &out1);

        assert_tensors_equal(&mut out2, &mut solution);
    }
}
