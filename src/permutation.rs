use std::collections::HashSet;
use std::mem::{self, MaybeUninit};
use std::ops::Mul;
use std::{iter::zip, ops::Index};

/// Represents a permutation of elements. `perm[i] = j` means that the element at
/// position `i` of the input will be mapped by this permutation to position `j`
/// in the output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permutation {
    /// Specifies where each element of the input maps to, i.e. `out[order[i]] = in[i]`.
    order: Vec<usize>,
}

impl Permutation {
    /// Creates a permutation with the given order.
    ///
    /// # Panics
    /// Panics in debug if the order is not a consecutive sequence starting from 0.
    ///
    /// # Example
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let p = Permutation::new(vec![2, 0, 1]);
    /// assert_eq!(p[0], 2);
    /// assert_eq!(p[1], 0);
    /// assert_eq!(p[2], 1);
    /// ```
    #[must_use]
    pub fn new(order: Vec<usize>) -> Self {
        // Check validity
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert!(
            sorted == (0..order.len()).collect::<Vec<_>>(),
            "The order must be a consecutive sequence starting from 0"
        );

        // Construct permutation
        Self { order }
    }

    /// Creates a identity permutation of given size.
    ///
    /// # Example
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let p = Permutation::identity(3);
    /// assert_eq!(p[0], 0);
    /// assert_eq!(p[1], 1);
    /// assert_eq!(p[2], 2);
    /// ```
    #[must_use]
    pub fn identity(size: usize) -> Self {
        Self {
            order: (0..size).collect(),
        }
    }

    /// Gets a reference to the permutations order.
    #[must_use]
    pub const fn order(&self) -> &Vec<usize> {
        &self.order
    }

    /// Checks if the permutation is an identity.
    ///
    /// # Example
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let p1 = Permutation::identity(3);
    /// assert_eq!(p1.is_identity(), true);
    /// let p2 = Permutation::new(vec![0, 1, 2]);
    /// assert_eq!(p2.is_identity(), true);
    /// let p3 = Permutation::new(vec![2, 0, 1]);
    /// assert_eq!(p3.is_identity(), false);
    /// ```
    #[must_use]
    pub fn is_identity(&self) -> bool {
        if self.is_empty() {
            return true;
        }

        let mut previous = self[0];
        for i in 1..self.len() {
            if self[i] <= previous {
                return false;
            }
            previous = self[i];
        }
        true
    }

    /// Returns the size of the permutation.
    ///
    /// # Example
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let p = Permutation::identity(5);
    /// assert_eq!(p.len(), 5);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Returns whether the permutation is an empty function (defined on the empty
    /// set) or not.
    ///
    /// # Example
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let p = Permutation::new(Vec::new());
    /// assert!(p.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Creates the permuted version of an array, i.e. `out[perm[i]] = in[i]`.
    ///
    /// # Panics
    /// Panics if the arrays length does not match the permutations length.
    ///
    /// # Example
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let data = &[2, 4, 3, 1];
    /// let perm = Permutation::new(vec![3, 2, 0, 1]);
    /// let data_p = perm.apply(data);
    /// assert_eq!(data_p, &[3, 1, 4, 2]);
    /// ```
    #[must_use]
    pub fn apply<T>(&self, array: &[T]) -> Vec<T>
    where
        T: Copy,
    {
        assert_eq!(
            self.len(),
            array.len(),
            "The arrays length must match the length of the permutation"
        );

        // Create a vector of uninitialized elements
        let mut out: Vec<MaybeUninit<T>> = vec![MaybeUninit::uninit(); array.len()];

        // Scattered write of array to the vector
        for i in 0..array.len() {
            out[self[i]].write(array[i]);
        }

        // Everything is written. Transmute vector to the initialized type
        unsafe { mem::transmute::<_, Vec<T>>(out) }
    }

    /// Applies the permutation to the array in-place. Uses O(n) extra memory.
    ///
    /// # Panics
    /// Panics if the arrays length does not match the length of the permutation.
    ///
    /// # Example
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let mut data = [2, 4, 3, 1];
    /// let perm = Permutation::new(vec![3, 2, 0, 1]);
    /// perm.apply_inplace(&mut data);
    /// assert_eq!(data, [3, 1, 4, 2]);
    /// ```
    pub fn apply_inplace<T>(&self, array: &mut [T]) {
        assert_eq!(
            self.len(),
            array.len(),
            "The arrays length must match the length of the permutation"
        );

        let mut done = HashSet::with_capacity(array.len());
        for i in 0..array.len() {
            if done.contains(&i) {
                continue;
            }
            let mut idx = self[i];
            while idx != i {
                array.swap(i, idx);
                done.insert(idx);
                idx = self[idx];
            }
        }
    }

    /// Creates the original version of an permuted array, i.e. `out[i] = in[perm[i]]`.
    ///
    /// # Panics
    /// Panics if the arrays length does not match the length of the permutation.
    ///
    /// # Example
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let data = &[3, 1, 4, 2];
    /// let perm = Permutation::new(vec![3, 2, 0, 1]);
    /// let data_p = perm.apply_inverse(data);
    /// assert_eq!(data_p, &[2, 4, 3, 1]);
    /// ```
    #[must_use]
    pub fn apply_inverse<T>(&self, array: &[T]) -> Vec<T>
    where
        T: Copy,
    {
        assert_eq!(
            self.len(),
            array.len(),
            "The arrays length must match the length of the permutation"
        );
        (0..array.len()).map(|i| array[self[i]]).collect()
    }

    /// Applies the inverse permutation to the array in-place, without extra memory
    /// allocation.
    ///
    /// # Panics
    /// Panics if the arrays length does not match the length of the permutation.
    ///
    /// # Example
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let mut data = [3, 1, 4, 2];
    /// let perm = Permutation::new(vec![3, 2, 0, 1]);
    /// perm.apply_inverse_inplace(&mut data);
    /// assert_eq!(data, [2, 4, 3, 1]);
    /// ```
    pub fn apply_inverse_inplace<T>(&self, array: &mut [T]) {
        assert_eq!(
            self.len(),
            array.len(),
            "The arrays length must match the length of the permutation"
        );
        for i in 0..array.len() {
            let mut index = self[i];
            while index < i {
                index = self[index];
            }
            array.swap(i, index);
        }
    }

    /// Creates the inverse permutation. Applying the inverse permutation is the same
    /// as using the inverse apply of the original permutation.
    ///
    /// # Example
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let p = Permutation::new(vec![3, 2, 0, 1]);
    /// let p_inv = p.inverse();
    /// assert_eq!(p_inv[0], 2);
    /// assert_eq!(p_inv[1], 3);
    /// assert_eq!(p_inv[2], 1);
    /// assert_eq!(p_inv[3], 0);
    /// ```
    #[must_use]
    pub fn inverse(&self) -> Self {
        let identity: Vec<_> = (0..self.len()).collect();
        let new_order = self.apply(&identity);
        Self { order: new_order }
    }

    /// Finds the permutation that is needed to transform `b` to `a`. It is assumed
    /// that both slices contain the same elements.
    ///
    /// # Panics
    /// Panics if the slices have different sizes.
    ///
    /// # Examples
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let a = &[1, 2, 0, 3];
    /// let b = &[0, 3, 2, 1];
    /// let perm = Permutation::between(a, b);
    /// assert_eq!(perm[0], 2);
    /// assert_eq!(perm[1], 3);
    /// assert_eq!(perm[2], 1);
    /// assert_eq!(perm[3], 0);
    /// ```
    #[must_use]
    pub fn between<T>(a: &[T], b: &[T]) -> Self
    where
        T: Ord + Copy,
    {
        // Check validity
        assert_eq!(a.len(), b.len());
        let mut a_sorted = a.to_vec();
        let mut b_sorted = b.to_vec();
        a_sorted.sort();
        b_sorted.sort();
        assert!(
            a_sorted == b_sorted,
            "The arrays must contain the same elements"
        );

        let mut perm1 = (0..a.len()).collect::<Vec<_>>();
        let mut perm2 = perm1.clone();
        perm1.sort_by_key(|i| a[*i]);
        perm2.sort_by_key(|i| b[*i]);
        let mut out = vec![0; a.len()];
        for (i, j) in zip(perm1, perm2) {
            out[j] = i;
        }
        Self::new(out)
    }
}

impl Index<usize> for Permutation {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.order[index]
    }
}

impl Mul<&Permutation> for &Permutation {
    type Output = Permutation;

    /// Multiplication of two permutations. This is like a composition,
    /// meaning the permutation of `perm1 * perm2` will act as
    /// `out[perm2[perm1[i]]] = in[i]`.
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let p1 = Permutation::new(vec![2, 0, 3, 1]);
    /// let p2 = Permutation::new(vec![1, 0, 3, 2]);
    /// let p = &p1 * &p2;
    /// assert_eq!(p[0], 3);
    /// assert_eq!(p[1], 1);
    /// assert_eq!(p[2], 2);
    /// assert_eq!(p[3], 0);
    /// ```
    fn mul(self, rhs: &Permutation) -> Self::Output {
        Self::Output {
            order: self.apply_inverse(&rhs.order),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Permutation;

    #[test]
    fn apply_and_apply_inverse_is_identity() {
        let data = &['a', 'b', 'c', 'd', 'e', 'f'];
        let p1 = Permutation::new(vec![3, 4, 2, 0, 5, 1]);
        let p2 = Permutation::new(vec![5, 2, 0, 4, 1, 3]);
        let p3 = Permutation::new(vec![5, 0, 4, 2, 3, 1]);
        let p4 = Permutation::new(vec![2, 3, 0, 1, 5, 4]);
        assert_eq!(p1.apply_inverse(&p1.apply(data)), data);
        assert_eq!(p2.apply_inverse(&p2.apply(data)), data);
        assert_eq!(p3.apply(&p3.apply_inverse(data)), data);
        assert_eq!(p4.apply(&p4.apply_inverse(data)), data);
    }

    #[test]
    fn permute_between_and_apply() {
        let data1 = &[4, 1, 0, 2, 3];
        let data2 = &[3, 4, 1, 0, 2];
        let data3 = &[3, 1, 4, 2, 0];
        let p12 = Permutation::between(data1, data2);
        let p13 = Permutation::between(data1, data3);
        let p23 = Permutation::between(data2, data3);
        assert_eq!(p12.apply(data2), data1);
        assert_eq!(p13.apply(data3), data1);
        assert_eq!(p23.apply(data3), data2);
    }

    #[test]
    fn multiplication() {
        let data = ['a', 'b', 'c', 'd', 'e'];
        let p1 = Permutation::new(vec![3, 0, 2, 4, 1]);
        let p2 = Permutation::new(vec![1, 4, 0, 2, 3]);
        let data1 = p1.apply(&data);
        let data2 = p2.apply(&data1);
        let data12 = (&p1 * &p2).apply(&data);
        assert_eq!(data12, data2);
    }

    #[test]
    fn apply_inverse_and_inverse() {
        let data = &['a', 'b', 'c', 'd', 'e', 'f'];
        let p = Permutation::new(vec![4, 0, 2, 3, 5, 1]);
        let p_inv = p.inverse();

        assert_eq!(p_inv.apply(data), p.apply_inverse(data));
        assert_eq!(p_inv.apply_inverse(data), p.apply(data));
    }

    #[test]
    fn multiply_with_inverse_is_identity() {
        let p1 = Permutation::new(vec![3, 4, 2, 0, 5, 1]);
        let p2 = Permutation::new(vec![5, 2, 0, 4, 1, 3]);
        assert!((&p1 * &p1.inverse()).is_identity());
        assert!((&p1.inverse() * &p1).is_identity());
        assert!((&p2 * &p2.inverse()).is_identity());
        assert!((&p2.inverse() * &p2).is_identity());
    }

    #[test]
    fn multiply_with_identity_is_self() {
        let p1 = Permutation::new(vec![0, 3, 4, 2, 5, 1]);
        let p2 = Permutation::new(vec![5, 2, 1, 4, 0, 3]);
        let pid = Permutation::identity(6);
        assert_eq!(&p1 * &pid, p1);
        assert_eq!(&pid * &p1, p1);
        assert_eq!(&p2 * &pid, p2);
        assert_eq!(&pid * &p2, p2);
    }

    #[test]
    fn inverse_inverse() {
        let p1 = Permutation::new(vec![5, 1, 3, 0, 2, 4]);
        let p2 = Permutation::new(vec![0, 3, 4, 1, 2, 5]);
        assert_eq!(p1.inverse().inverse(), p1);
        assert_eq!(p2.inverse().inverse(), p2);
    }

    #[test]
    fn apply_and_inplace() {
        let data = &['a', 'b', 'c', 'd', 'e', 'f'];
        let p1 = Permutation::new(vec![3, 4, 2, 0, 5, 1]);
        let p2 = Permutation::new(vec![5, 2, 0, 4, 1, 3]);
        let p3 = Permutation::new(vec![5, 0, 4, 2, 3, 1]);
        let p4 = Permutation::new(vec![2, 3, 0, 1, 5, 4]);

        let mut cpy = *data;
        p1.apply_inplace(&mut cpy);
        assert_eq!(cpy.to_vec(), p1.apply(data));

        cpy = *data;
        p2.apply_inplace(&mut cpy);
        assert_eq!(cpy.to_vec(), p2.apply(data));

        cpy = *data;
        p3.apply_inplace(&mut cpy);
        assert_eq!(cpy.to_vec(), p3.apply(data));

        cpy = *data;
        p4.apply_inplace(&mut cpy);
        assert_eq!(cpy.to_vec(), p4.apply(data));
    }

    #[test]
    fn apply_inverse_and_inplace() {
        let data = &['a', 'b', 'c', 'd', 'e', 'f'];
        let p1 = Permutation::new(vec![3, 4, 2, 0, 5, 1]);
        let p2 = Permutation::new(vec![5, 2, 0, 4, 1, 3]);
        let p3 = Permutation::new(vec![5, 0, 4, 2, 3, 1]);
        let p4 = Permutation::new(vec![2, 3, 0, 1, 5, 4]);

        let mut cpy = *data;
        p1.apply_inverse_inplace(&mut cpy);
        assert_eq!(cpy.to_vec(), p1.apply_inverse(data));

        cpy = *data;
        p2.apply_inverse_inplace(&mut cpy);
        assert_eq!(cpy.to_vec(), p2.apply_inverse(data));

        cpy = *data;
        p3.apply_inverse_inplace(&mut cpy);
        assert_eq!(cpy.to_vec(), p3.apply_inverse(data));

        cpy = *data;
        p4.apply_inverse_inplace(&mut cpy);
        assert_eq!(cpy.to_vec(), p4.apply_inverse(data));
    }
}
