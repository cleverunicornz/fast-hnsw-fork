//! Distance metrics.
//!
//! All five metrics run on the crate's vectorised kernels, which
//! dispatch to NEON or AVX2 and fall back to a portable fold elsewhere.
//!
//! # The `blas` feature
//!
//! With `blas` enabled, [`DotProduct`] and [`Cosine`] call the system BLAS
//! (`sdot` / `snrm2`) instead.  Both are *exact* substitutions — BLAS computes
//! the same quantity — but **measure before enabling it: on Apple silicon it
//! is currently slower than the built-in SIMD path.**
//!
//! Apple silicon, Accelerate:
//!
//! | dim  | `dot` SIMD | `dot` BLAS | `Cosine` SIMD | `Cosine` BLAS |
//! |------|-----------|-----------|---------------|---------------|
//! | 128  | 6.1 ns    | 20.0 ns   | 15.0 ns       | 33.5 ns       |
//! | 768  | 36.7 ns   | 39.4 ns   | 98.3 ns       | 200.8 ns      |
//! | 1536 | 74.3 ns   | 76.1 ns   | 263 ns        | 419 ns        |
//!
//! x86-64, OpenBLAS 0.3.32 (i9-12900H):
//!
//! | dim  | `dot` SIMD | `dot` BLAS | `Cosine` SIMD | `Cosine` BLAS |
//! |------|-----------|-----------|---------------|---------------|
//! | 128  | 5.2 ns    | 13.2 ns   | 16.2 ns       | 66.5 ns       |
//! | 768  | 26.9 ns   | 22.8 ns   | 84.4 ns       | 252.2 ns      |
//! | 1536 | 56.1 ns   | 44.5 ns   | 174 ns        | 496 ns        |
//!
//! A standalone `cblas_sdot` really is faster than a portable scalar fold, and
//! that is what motivated the feature.  It loses here for a structural reason:
//! the SIMD kernel **inlines** into the caller, while a BLAS call is an
//! opaque FFI boundary that cannot be inlined — and [`Cosine`] pays that cost
//! three times over (`sdot` plus two `snrm2`).  At these vector sizes the call
//! overhead is comparable to the arithmetic.
//!
//! The feature is kept because it may still pay off elsewhere — a threaded
//! OpenBLAS or MKL on x86, or much larger dimensions — but it is off by
//! default and should be justified by a measurement on your own hardware.
//!
//! [`Euclidean`], [`SquaredEuclidean`] and [`Manhattan`] never use BLAS, even
//! when the feature is on, because BLAS has no squared-difference or
//! absolute-difference primitive.  The only way to express L2 in BLAS is the
//! expansion `‖a−b‖² = ‖a‖² + ‖b‖² − 2·a·b`, and both forms of it were
//! measured and rejected:
//!
//! * **Three `sdot` calls** (norms computed per call) runs at 0.56–0.72× the
//!   portable fold — slower, for worse accuracy.
//! * **One `sdot` with both squared norms cached** does reach ~2.0× the
//!   portable fold, but the
//!   expansion subtracts two large nearly-equal quantities. Measured relative
//!   error against an `f64` reference, for vectors differing by ~0.1%:
//!
//!   | element scale | direct fold | norm expansion |
//!   |---------------|-------------|----------------|
//!   | 1             | 6.3e-8      | 2.05           |
//!   | 100           | 1.5e-8      | 1.00           |
//!   | 10 000        | 0.0         | 1.00           |
//!
//!   That is 100–205% error, and it is worst for *near-identical* vectors —
//!   exactly the nearest neighbours whose relative ordering decides the
//!   top-k.  A 2× kernel speed-up is not worth silently misranking the
//!   results the index exists to return.
//!
//!   The direct kernel is also what the SIMD path computes, so this accuracy
//!   argument stands independently of which backend is faster.
//!
//! The backend is resolved at build time — the Accelerate framework on Apple
//! platforms, `pkg-config` elsewhere, overridable with `FAST_HNSW_BLAS_LIB`.
//! Nothing is vendored or downloaded; see the crate-level docs.

/// A distance (or dissimilarity) between two vectors.
/// Lower is closer.
pub trait Distance: Send + Sync + 'static {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// Stable identifier recorded in saved indexes so that reopening one with
    /// the wrong metric is rejected rather than silently answered.
    ///
    /// A graph's structure encodes the metric it was built with, so scoring it
    /// with a different one returns confident, wrong neighbours — the same
    /// failure mode as a mismatched query dimension, and just as invisible.
    ///
    /// `0` means *unspecified*. An index saved by such a metric records no
    /// identity and can afterwards be opened by anything, so existing custom
    /// implementations keep working unchanged — this method has a default and
    /// their own snapshots stay readable.
    ///
    /// The reverse is not permitted: a metric returning `0` cannot open an
    /// index that *did* record an identity. It cannot be shown to match, and
    /// assuming it does is the silent-wrong-answer case this exists to
    /// prevent. Declare compatibility by returning that index's id.
    ///
    /// Give a custom metric a non-zero id to have it checked. Pick one at or
    /// above [`RESERVED_METRIC_IDS`], or deliberately return a built-in's id to
    /// declare compatibility with indexes it wrote.
    fn metric_id() -> u32
    where
        Self: Sized,
    {
        0
    }

    /// Human-readable name used in mismatch errors.
    fn metric_name() -> &'static str
    where
        Self: Sized,
    {
        "custom"
    }
}

/// Ids below this are reserved for metrics shipped with this crate.
///
/// A custom [`Distance`] that wants to be identity-checked should return
/// something at or above this value from [`Distance::metric_id`], so a future
/// built-in metric cannot collide with it.
pub const RESERVED_METRIC_IDS: u32 = 1024;

/// Map a recorded id back to a name, for error messages.
pub(crate) fn metric_name_for_id(id: u32) -> &'static str {
    match id {
        0 => "unspecified",
        1 => "SquaredEuclidean",
        2 => "Euclidean",
        3 => "Cosine",
        4 => "DotProduct",
        5 => "Manhattan",
        _ => "custom",
    }
}

// ─── Dot product and norm ────────────────────────────────────────────────────
//
// These two are the only kernels the `blas` feature substitutes, because they
// are the only ones BLAS can express exactly (`sdot` / `snrm2`).  Without the
// feature they use the crate's vectorised kernels.

/// Dot product over the common prefix of `a` and `b`.
#[cfg(not(feature = "blas"))]
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot(a, b)
}

/// Dot product via the system BLAS (`sdot`).
///
/// An exact drop-in for the vectorised kernel: same quantity, same `zip`
/// truncation to the common prefix.
#[cfg(feature = "blas")]
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    if n > i32::MAX as usize {
        return crate::simd::dot(a, b);
    }
    // SAFETY: both pointers are valid for `n` contiguous `f32` reads, and the
    // guard above keeps the `i32` cast well-defined.
    unsafe { crate::blas::cblas_sdot(n as i32, a.as_ptr(), 1, b.as_ptr(), 1) }
}

/// L2 norm of a single vector.
#[cfg(not(feature = "blas"))]
#[inline(always)]
fn norm(v: &[f32]) -> f32 {
    crate::simd::norm_squared(v).sqrt()
}

/// L2 norm of a single vector via BLAS (`snrm2`).
#[cfg(feature = "blas")]
#[inline(always)]
fn norm(v: &[f32]) -> f32 {
    if v.is_empty() || v.len() > i32::MAX as usize {
        return crate::simd::norm_squared(v).sqrt();
    }
    // SAFETY: `v` is valid for `v.len()` contiguous `f32` reads.
    unsafe { crate::blas::cblas_snrm2(v.len() as i32, v.as_ptr(), 1) }
}

// ─── Built-in metrics ────────────────────────────────────────────────────────

/// Squared Euclidean distance  (avoids a sqrt; preserves nearest-neighbour order).
#[derive(Clone, Copy, Debug, Default)]
pub struct SquaredEuclidean;

impl Distance for SquaredEuclidean {
    fn metric_id() -> u32 { 1 }
    fn metric_name() -> &'static str { "SquaredEuclidean" }

    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        crate::simd::l2_squared(a, b)
    }
}

/// True Euclidean (L2) distance.
#[derive(Clone, Copy, Debug, Default)]
pub struct Euclidean;

impl Distance for Euclidean {
    fn metric_id() -> u32 { 2 }
    fn metric_name() -> &'static str { "Euclidean" }

    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        crate::simd::l2_squared(a, b).sqrt()
    }
}

/// Cosine distance  = 1 − cosine_similarity ∈ [0, 2].
/// Works correctly only for non-zero vectors.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cosine;

impl Distance for Cosine {
    fn metric_id() -> u32 { 3 }
    fn metric_name() -> &'static str { "Cosine" }

    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot = dot(a, b);
        // Norms deliberately cover each slice in full (not the common prefix),
        // preserving the previous behaviour for unequal-length inputs.
        let na = norm(a);
        let nb = norm(b);
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        // clamp to [−1, 1] before subtracting to absorb float rounding
        let cos = (dot / (na * nb)).clamp(-1.0, 1.0);
        1.0 - cos
    }
}

/// Inner-product distance  = 1 − dot(a, b).
/// Useful for pre-normalised embeddings (equals cosine distance when ‖a‖=‖b‖=1).
#[derive(Clone, Copy, Debug, Default)]
pub struct DotProduct;

impl Distance for DotProduct {
    fn metric_id() -> u32 { 4 }
    fn metric_name() -> &'static str { "DotProduct" }

    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - dot(a, b)
    }
}

/// Manhattan (L1) distance.
#[derive(Clone, Copy, Debug, Default)]
pub struct Manhattan;

impl Distance for Manhattan {
    fn metric_id() -> u32 { 5 }
    fn metric_name() -> &'static str { "Manhattan" }

    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        crate::simd::l1(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Cosine, Distance, DotProduct, Euclidean, Manhattan, SquaredEuclidean,
    };

    /// Deterministic non-trivial values so lane folding cannot accidentally
    /// agree with the reference by symmetry.
    fn sample(len: usize, offset: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i + offset) as f32 * 0.37).sin() * 3.0 + 0.5)
            .collect()
    }

    #[test]
    fn dot_product_handles_scalar_tail_and_zip_length() {
        let a = [1.0, -2.0, 3.0, 4.0, 0.5, 99.0];
        let b = [2.0, 3.0, -1.0, 0.25, 8.0];
        let expected_dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let actual_dot = 1.0 - DotProduct.distance(&a, &b);
        assert!((actual_dot - expected_dot).abs() <= f32::EPSILON * 8.0);
    }

    /// The lane-folded kernels must agree with the straightforward serial fold
    /// across every length class: below one lane block, exact multiples, and
    /// multiples plus a scalar tail.
    #[test]
    fn lane_folded_kernels_match_serial_reference() {
        for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 31, 64, 128, 129] {
            let a = sample(len, 0);
            let b = sample(len, 11);

            let l2_squared: f32 =
                a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            let l1: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
            let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

            // Relative tolerance: the vector kernels fuse multiply-add, which
            // rounds once where this serial reference rounds twice.
            let tolerance = 1e-5 * l2_squared.abs().max(l1.abs()).max(1.0);
            assert!(
                (SquaredEuclidean.distance(&a, &b) - l2_squared).abs() <= tolerance,
                "SquaredEuclidean mismatch at len {len}"
            );
            assert!(
                (Euclidean.distance(&a, &b) - l2_squared.sqrt()).abs() <= tolerance,
                "Euclidean mismatch at len {len}"
            );
            assert!(
                (Manhattan.distance(&a, &b) - l1).abs() <= tolerance,
                "Manhattan mismatch at len {len}"
            );
            assert!(
                ((1.0 - DotProduct.distance(&a, &b)) - dot).abs() <= tolerance,
                "DotProduct mismatch at len {len}"
            );
        }
    }

    /// Unequal lengths must still pair elements the way `zip` did, including
    /// when the common prefix ends mid-lane-block.
    #[test]
    fn unequal_lengths_pair_over_the_common_prefix() {
        for (left, right) in [(6usize, 5usize), (20, 9), (9, 20), (17, 8), (8, 17)] {
            let a = sample(left, 2);
            let b = sample(right, 23);
            let l2_squared: f32 =
                a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            let l1: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
            let tolerance = 1e-5 * l2_squared.abs().max(l1.abs()).max(1.0);

            assert!(
                (SquaredEuclidean.distance(&a, &b) - l2_squared).abs() <= tolerance,
                "SquaredEuclidean mismatch for lengths {left}/{right}"
            );
            assert!(
                (Manhattan.distance(&a, &b) - l1).abs() <= tolerance,
                "Manhattan mismatch for lengths {left}/{right}"
            );
        }
    }

    /// Cosine norms cover each slice in full rather than the common prefix.
    #[test]
    fn cosine_matches_full_length_norms() {
        let a = sample(20, 3);
        let b = sample(9, 31);
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let expected = 1.0 - (dot / (na * nb)).clamp(-1.0, 1.0);
        assert!((Cosine.distance(&a, &b) - expected).abs() <= f32::EPSILON * 64.0);
    }

    #[test]
    fn identical_vectors_are_exactly_zero_distance() {
        let a = sample(128, 5);
        assert_eq!(SquaredEuclidean.distance(&a, &a), 0.0);
        assert_eq!(Euclidean.distance(&a, &a), 0.0);
        assert_eq!(Manhattan.distance(&a, &a), 0.0);
    }

    #[test]
    fn zero_norm_cosine_is_neutral() {
        let zero = vec![0.0f32; 16];
        let other = sample(16, 7);
        assert_eq!(Cosine.distance(&zero, &other), 1.0);
        assert_eq!(Cosine.distance(&other, &zero), 1.0);
    }
}
