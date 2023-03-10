use std::iter::zip;

/// Creates the permuted version of an array.
///
/// # Example
/// ```
/// # use tetra::permutation::permute;
/// let arr = &[2, 4, 3, 1];
/// let perm = &[3, 2, 0, 1];
/// let arr_p = permute(perm, arr);
/// assert_eq!(arr_p, vec![1, 3, 2, 4]);
/// ```
#[must_use]
#[inline]
pub fn permute<T>(perm: &[i32], arr: &[T]) -> Vec<T>
where
    T: Copy,
{
    (0..arr.len()).map(|i| arr[perm[i] as usize]).collect()
}

/// Creates the original version of an permuted array.
///
/// # Example
/// ```
/// # use tetra::permutation::permute;
/// # use tetra::permutation::inv_permute;
/// let arr = &[2, 4, 3, 1];
/// let perm = &[3, 2, 0, 1];
/// let arr_p = permute(perm, arr);
/// let arr2 = inv_permute(perm, &arr_p);
/// assert_eq!(arr2, arr);
/// ```
#[must_use]
#[inline]
pub fn inv_permute<T>(perm: &[i32], arr: &[T]) -> Vec<T>
where
    T: Copy,
{
    let mut indices = (0..perm.len()).collect::<Vec<_>>();
    indices.sort_unstable_by_key(|&i| &perm[i]);
    (0..arr.len()).map(|i| arr[indices[i]]).collect()
}

/// Finds the permutation that is needed to transform a to b. It is assumed that both
/// slices contain the same elements.
///
/// # Panics
/// - Panics if the slices have different sizes
///
/// # Examples
/// ```
/// # use tetra::permutation::permutation_between;
/// # use tetra::permutation::permute;
/// let a = &[1, 2, 0, 5, 4];
/// let b = &[0, 1, 2, 4, 5];
/// let perm = permutation_between(a, b);
/// assert_eq!(perm, &[2, 0, 1, 4, 3]);
/// assert_eq!(permute(&perm, a), b);
/// ```
#[must_use]
#[inline]
pub fn permutation_between<T>(a: &[T], b: &[T]) -> Vec<i32>
where
    T: Ord + Copy,
{
    assert_eq!(a.len(), b.len());
    let mut perm1 = (0..a.len() as i32).collect::<Vec<_>>();
    let mut perm2 = perm1.clone();
    perm1.sort_by_key(|i| a[*i as usize]);
    perm2.sort_by_key(|i| b[*i as usize]);
    let mut out = vec![0; a.len()];
    for (i, j) in zip(perm1, perm2) {
        out[j as usize] = i;
    }
    out
}
