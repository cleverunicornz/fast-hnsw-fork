/// A distance (or dissimilarity) between two vectors.
/// Lower is closer.
pub trait Distance: Send + Sync + 'static {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Dot product with independent accumulators so modern compilers can exploit
/// instruction-level parallelism without a platform-specific SIMD dependency.
///
/// The scalar tail preserves the `zip` semantics of the original
/// implementation for unequal or non-multiple-of-four dimensions.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let common = a.len().min(b.len());
    let chunks = common / 4;
    let mut sums = [0.0; 4];
    for chunk in 0..chunks {
        let offset = chunk * 4;
        sums[0] += a[offset] * b[offset];
        sums[1] += a[offset + 1] * b[offset + 1];
        sums[2] += a[offset + 2] * b[offset + 2];
        sums[3] += a[offset + 3] * b[offset + 3];
    }
    let mut sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for index in chunks * 4..common {
        sum += a[index] * b[index];
    }
    sum
}

// ─── Built-in metrics ────────────────────────────────────────────────────────

/// Squared Euclidean distance  (avoids a sqrt; preserves nearest-neighbour order).
#[derive(Clone, Copy, Debug, Default)]
pub struct SquaredEuclidean;

impl Distance for SquaredEuclidean {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }
}

/// True Euclidean (L2) distance.
#[derive(Clone, Copy, Debug, Default)]
pub struct Euclidean;

impl Distance for Euclidean {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }
}

/// Cosine distance  = 1 − cosine_similarity ∈ [0, 2].
/// Works correctly only for non-zero vectors.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cosine;

impl Distance for Cosine {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot = dot(a, b);
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
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
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - dot(a, b)
    }
}

/// Manhattan (L1) distance.
#[derive(Clone, Copy, Debug, Default)]
pub struct Manhattan;

impl Distance for Manhattan {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::{Distance, DotProduct};

    #[test]
    fn dot_product_handles_scalar_tail_and_zip_length() {
        let a = [1.0, -2.0, 3.0, 4.0, 0.5, 99.0];
        let b = [2.0, 3.0, -1.0, 0.25, 8.0];
        let expected_dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let actual_dot = 1.0 - DotProduct.distance(&a, &b);
        assert!((actual_dot - expected_dot).abs() <= f32::EPSILON * 8.0);
    }
}
