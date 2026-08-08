//! Ordered `(distance, node-id)` pair shared by the HNSW search routines.
//!
//! HNSW needs two kinds of priority queue: a *min*-heap for the candidate set
//! `C` (pop returns the closest element) and a bounded *max*-heap for the
//! dynamic nearest-neighbour list `W` (pop returns the farthest).  Both are
//! built directly on [`std::collections::BinaryHeap`] inside
//! [`Scratch`](crate::hnsw) so the two heaps and the sorted output buffer can
//! share one reusable allocation; `BinaryHeap` is a max-heap, so the candidate
//! side wraps entries in [`std::cmp::Reverse`].
//!
//! This module supplies the element type those heaps order.

use std::cmp::Ordering;

// ─── Ordered pair ────────────────────────────────────────────────────────────

/// A (distance, node-id) pair with a total order on `distance`.
/// NaN distances are treated as larger than any finite value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistId {
    pub dist: f32,
    pub id: usize,
}

impl DistId {
    #[inline]
    pub fn new(dist: f32, id: usize) -> Self {
        Self { dist, id }
    }
}

impl Eq for DistId {}

impl PartialOrd for DistId {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Max-heap order: larger distance = higher priority.
impl Ord for DistId {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // total_cmp gives a proper total order including NaN (NaN > everything).
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    #[test]
    fn ordering_is_by_distance_then_id() {
        assert!(DistId::new(1.0, 9) < DistId::new(2.0, 0));
        assert!(DistId::new(1.0, 0) < DistId::new(1.0, 1));
        assert_eq!(DistId::new(1.0, 0), DistId::new(1.0, 0));
    }

    #[test]
    fn nan_sorts_above_every_finite_distance() {
        // `total_cmp` keeps the heap a valid total order even if a metric
        // returns NaN, so a poisoned distance is pushed to the far end rather
        // than corrupting the heap invariant.
        assert!(DistId::new(f32::NAN, 0) > DistId::new(f32::MAX, 0));
    }

    #[test]
    fn reverse_wrapper_yields_a_min_heap() {
        let mut candidates = BinaryHeap::new();
        for dist in [3.0, 1.0, 2.0] {
            candidates.push(Reverse(DistId::new(dist, dist as usize)));
        }
        let popped: Vec<f32> = std::iter::from_fn(|| candidates.pop())
            .map(|Reverse(entry)| entry.dist)
            .collect();
        assert_eq!(popped, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn plain_heap_yields_a_max_heap() {
        let mut results = BinaryHeap::new();
        for dist in [3.0, 1.0, 2.0] {
            results.push(DistId::new(dist, dist as usize));
        }
        let popped: Vec<f32> = std::iter::from_fn(|| results.pop())
            .map(|entry| entry.dist)
            .collect();
        assert_eq!(popped, [3.0, 2.0, 1.0]);
    }
}
