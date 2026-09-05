//! The two CBLAS entry points this crate calls, declared directly.
//!
//! Declaring them here rather than depending on a bindings crate keeps the
//! `blas` feature free of any vendoring machinery — see `build.rs` for how the
//! system library is located.
//!
//! # Why CBLAS and not Fortran BLAS
//!
//! The Fortran interface (`sdot_`) is the classic trap: several BLAS builds,
//! notably anything derived from `f2c`, return `double` from single-precision
//! functions, so calling `sdot_` through an `f32`-returning signature reads
//! garbage.  The `cblas_*` interface is a C API with a specified `float`
//! return, so it is well-defined everywhere.
//!
//! Note that Netlib's `libblas` is Fortran-only and does **not** export these
//! symbols; `libcblas` or OpenBLAS does.
//!
//! # Integer width
//!
//! These declarations use `c_int`, matching a standard LP64 build.  BLAS
//! libraries built with 64-bit indices (`INTERFACE64=1`, or MKL's ILP64
//! layer) take 64-bit integers and are **not** compatible — link the LP64
//! variant (`mkl_rt` defaults to LP64).  The counts this crate passes are
//! bounded by vector dimension, so `c_int` is never the limiting factor.

use std::ffi::c_int;

/// CBLAS `layout` values.
pub(crate) const CBLAS_ROW_MAJOR: c_int = 101;
/// CBLAS `transpose` values.
pub(crate) const CBLAS_NO_TRANS: c_int = 111;

extern "C" {
    /// `x · y` over `n` elements with the given strides.
    pub(crate) fn cblas_sdot(
        n: c_int,
        x: *const f32,
        incx: c_int,
        y: *const f32,
        incy: c_int,
    ) -> f32;

    /// Euclidean norm of `x` over `n` elements.
    pub(crate) fn cblas_snrm2(n: c_int, x: *const f32, incx: c_int) -> f32;

    /// `y := alpha * op(A) * x + beta * y` for a dense `m x n` matrix.
    ///
    /// This is the one place BLAS clearly beats the SIMD kernels: scoring a
    /// query against many *contiguous* candidates at once amortises the call
    /// over the whole matrix instead of paying it per vector.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cblas_sgemv(
        layout: c_int,
        trans: c_int,
        m: c_int,
        n: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        x: *const f32,
        incx: c_int,
        beta: f32,
        y: *mut f32,
        incy: c_int,
    );
}
