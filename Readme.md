# Tetra: Tensor contraction library

The documentation can be found [here](https://quantum-research.pages.gitlab.lrz.de/tensornetworksimulation/tetra).

### Choosing a different BLAS implementation

By default, the crate will try to link against a system installation of OpenBLAS. If this is not wanted (e.g., because Intel MKL should be used), add the crate without default features along with the wanted [source crate](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki):

```toml
[dependencies]
tetra = { version = "...", default-features = false }
intel-mkl-src = { version = "...", features = ["mkl-static-lp64-iomp"]}
```

This might also require adding a `extern crate` statement to your code (such that the corresponding crate is included in the build even if not directly used). For example, put the following in `lib.rs`:

```rust
extern crate intel_mkl_src;
```