use std::mem::{self, MaybeUninit};
use std::ops::Mul;
use std::{iter::zip, ops::Index};

/// Represents a permutation of elements. `perm[i] = j` means that the element at
/// position `i` of the input will be mapped by this permutation to position `j`
/// in the output.
#[derive(Clone, Debug, PartialEq)]
pub struct Permutation {
    /// Specifies where each element of the input maps to, i.e. `out[order[i]] = in[i]`.
    order: Vec<usize>,
}

impl Permutation {
    /// Creates a permutation with the given order.
    ///
    /// # Example
    /// ```
    /// # use tetra::permutation::Permutation;
    /// let p = Permutation::new(vec![2, 0, 1]);
    /// assert_eq!(p[0], 2);
    /// assert_eq!(p[1], 0);
    /// assert_eq!(p[2], 1);
    /// ```
    pub fn new(order: Vec<usize>) -> Self {
        // Check validity
        if cfg!(debug_assertions) {
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert!(
                sorted == (0..order.len()).collect::<Vec<_>>(),
                "The order must be a consecutive sequence starting from 0"
            );
        }

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
    pub fn identity(size: usize) -> Self {
        Self {
            order: (0..size).collect(),
        }
    }

    /// Gets a reference to the permutations order.
    pub fn order(&self) -> &Vec<usize> {
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
    pub fn is_identity(&self) -> bool {
        if self.len() == 0 {
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
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Creates the permuted version of an array, i.e. `out[perm[i]] = in[i]`.
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

    /// Creates the original version of an permuted array, i.e. `out[i] = in[perm[i]]`.
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

    /// Finds the permutation that is needed to transform `b` to `a`. It is assumed that both
    /// slices contain the same elements.
    ///
    /// # Panics
    /// - Panics if the slices have different sizes
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
        if cfg!(debug_assertions) {
            let mut asorted = a.to_vec();
            let mut bsorted = b.to_vec();
            asorted.sort();
            bsorted.sort();
            assert!(
                asorted == bsorted,
                "The arrays must contain the same elements"
            );
        }

        let mut perm1 = (0..a.len()).collect::<Vec<_>>();
        let mut perm2 = perm1.clone();
        perm1.sort_by_key(|i| a[*i as usize]);
        perm2.sort_by_key(|i| b[*i as usize]);
        let mut out = vec![0; a.len()];
        for (i, j) in zip(perm1, perm2) {
            out[j as usize] = i;
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
    fn permute_and_inverse() {
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
    fn permute_between() {
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
    fn composition() {
        let data = ['a', 'b', 'c', 'd', 'e'];
        let p1 = Permutation::new(vec![3, 0, 2, 4, 1]);
        let p2 = Permutation::new(vec![1, 4, 0, 2, 3]);
        let data1 = p1.apply(&data);
        let data2 = p2.apply(&data1);
        let data12 = (&p1 * &p2).apply(&data);
        assert_eq!(data12, data2);
    }
}
