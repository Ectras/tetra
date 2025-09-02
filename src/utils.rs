use float_cmp::{ApproxEq, F64Margin};
use num_complex::Complex64;

/// A transparent wrapper around a [`Complex64`] to provide it with an [`ApproxEq`]
/// implementation.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Complex64ApproxEq(Complex64);

impl ApproxEq for Complex64ApproxEq {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin = margin.into();
        self.0.re.approx_eq(other.0.re, margin) && self.0.im.approx_eq(other.0.im, margin)
    }
}
