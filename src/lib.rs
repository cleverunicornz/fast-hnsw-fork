//! # fast-hnsw
//!
//! A pure-Rust implementation of **Hierarchical Navigable Small World** (HNSW)
//! approximate nearest-neighbour search, following the algorithm from:
//!
//! > Malkov & Yashunin, *"Efficient and robust approximate nearest neighbor
//! > search using Hierarchical Navigable Small World graphs"*,
//! > IEEE TPAMI 2018.
//!
//! ## Quick start
//!
//! ```rust
//! use fast_hnsw::{Builder, Hnsw, SearchResult};
//! use fast_hnsw::distance::Euclidean;
//!
//! // Build an index.
//! let mut index: Hnsw<Euclidean> = Builder::new()
//!     .m(16)
//!     .ef_construction(200)
//!     .seed(42)
//!     .build(Euclidean);
//!
//! // Insert vectors.
//! for i in 0..100_u32 {
//!     index.insert(vec![i as f32, (i * i) as f32]);
//! }
//!
//! // Query: find 5 nearest neighbours with ef=50.
//! let results: Vec<SearchResult> = index.search(&[10.0, 101.0], 5, 50);
//! assert_eq!(results[0].id, 10);
//! ```
//!
//! ## Distance metrics
//!
//! | Type                        | Description                         |
//! |-----------------------------|-------------------------------------|
//! | [`distance::Euclidean`]     | True L2 distance                    |
//! | [`distance::SquaredEuclidean`] | L2² (faster, same NN order)      |
//! | [`distance::Cosine`]        | 1 − cosine similarity               |
//! | [`distance::DotProduct`]    | 1 − dot product                     |
//! | [`distance::Manhattan`]     | L1 / taxicab distance               |
//!
//! Custom metrics are easy to add by implementing the [`distance::Distance`] trait.
//!
//! ## Feature flags
//! *(none yet — this crate is dependency-light by design)*

pub mod builder;
pub mod distance;
pub(crate) mod heap;
pub mod hnsw;
pub mod labeled;
pub mod paired;
pub mod payload;
pub mod persist;

pub use builder::Builder;
pub use hnsw::{Config, Hnsw, IndexStats, PruneStrategy, SearchResult, SearchWorkspace};
pub use labeled::{LabeledIndex, MappedLabeledIndex, MappedLabeledResult};
pub use paired::PairedIndex;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use distance::{Cosine, Euclidean, Manhattan, SquaredEuclidean};
    use labeled::LabeledIndex;
    use paired::PairedIndex;
    use crate::persist;

    // ── helpers ──────────────────────────────────────────────────────────

    fn build_index(n: usize, dim: usize, seed: u64) -> Hnsw<Euclidean> {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed + 1_000);
        let mut index = Builder::new()
            .m(16)
            .ef_construction(200)
            .seed(seed)
            .build(Euclidean);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.insert(v);
        }
        index
    }

    /// Brute-force exact k-NN.
    fn exact_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
        let mut dists: Vec<(f32, usize)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                (d, i)
            })
            .collect();
        dists.sort_by(|a, b| a.0.total_cmp(&b.0));
        dists.iter().take(k).map(|(_, i)| *i).collect()
    }

    // ── unit tests ────────────────────────────────────────────────────────

    #[test]
    fn empty_index_returns_nothing() {
        let index: Hnsw<Euclidean> = Builder::new().build(Euclidean);
        assert!(index.search(&[1.0, 2.0], 5, 20).is_empty());
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn single_vector_always_returned() {
        let mut index = Builder::new().seed(0).build(Euclidean);
        index.insert(vec![1.0, 2.0, 3.0]);
        let res = index.search(&[0.0, 0.0, 0.0], 1, 10);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, 0);
    }

    #[test]
    fn ids_are_assigned_sequentially() {
        let mut index = Builder::new().seed(1).build(Euclidean);
        for i in 0..20 {
            let id = index.insert(vec![i as f32]);
            assert_eq!(id, i);
        }
        assert_eq!(index.len(), 20);
    }

    #[test]
    fn nearest_of_two_is_correct() {
        let mut index = Builder::new().seed(2).build(Euclidean);
        index.insert(vec![0.0, 0.0]); // id=0
        index.insert(vec![10.0, 0.0]); // id=1
        // Query very close to id=0
        let res = index.search(&[0.1, 0.0], 1, 10);
        assert_eq!(res[0].id, 0);
        // Query very close to id=1
        let res = index.search(&[9.9, 0.0], 1, 10);
        assert_eq!(res[0].id, 1);
    }

    #[test]
    fn distances_are_non_negative_and_ordered() {
        let index = build_index(200, 16, 3);
        let query: Vec<f32> = vec![0.5; 16];
        let results = index.search(&query, 10, 50);
        assert_eq!(results.len(), 10);
        for w in results.windows(2) {
            assert!(w[0].distance >= 0.0);
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn k_larger_than_index_returns_all() {
        let index = build_index(30, 4, 4);
        let query = vec![0.5f32; 4];
        let res = index.search(&query, 100, 200);
        assert_eq!(res.len(), 30);
    }

    #[test]
    fn filtered_search_applies_filter_before_top_k() {
        let mut index = Builder::new()
            .m(16)
            .ef_construction(100)
            .seed(44)
            .build(Euclidean);
        for id in 0..100 {
            index.insert(vec![id as f32]);
        }

        let results = index.search_filtered(&[12.1], 3, 100, |id| id % 10 == 0);
        assert_eq!(
            results.iter().map(|result| result.id).collect::<Vec<_>>(),
            [10, 20, 0]
        );
    }

    #[test]
    fn filtered_search_navigates_through_rejected_nodes() {
        let mut index = Builder::new()
            .m(16)
            .ef_construction(100)
            .seed(45)
            .build(Euclidean);
        for id in 0..100 {
            index.insert(vec![id as f32]);
        }

        let results = index.search_filtered(&[0.0], 1, 100, |id| id == 99);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 99);
    }

    #[test]
    fn filtered_search_handles_empty_eligibility_set() {
        let index = build_index(100, 4, 46);
        let results = index.search_filtered(&[0.5; 4], 10, 100, |_| false);
        assert!(results.is_empty());
    }

    #[test]
    fn reusable_workspace_preserves_search_and_filter_results() {
        let index = build_index(200, 16, 47);
        let query = [0.5; 16];
        let expected = index.search(&query, 10, 50);
        let expected_filtered = index.search_filtered(&query, 10, 50, |id| id % 3 == 0);

        let mut workspace = SearchWorkspace::default();
        assert_eq!(
            index.search_with_workspace(&query, 10, 50, &mut workspace),
            expected
        );
        assert_eq!(
            index.search_filtered_with_workspace(
                &query,
                10,
                50,
                |id| id % 3 == 0,
                &mut workspace,
            ),
            expected_filtered
        );
        assert_eq!(
            index.search_with_workspace(&query, 10, 100, &mut workspace),
            index.search(&query, 10, 100)
        );
    }

    #[test]
    fn stored_vectors_are_retrievable() {
        let mut index = Builder::new().seed(5).build(Euclidean);
        let vecs = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        for v in &vecs {
            index.insert(v.clone());
        }
        for (i, v) in vecs.iter().enumerate() {
            assert_eq!(index.get_vector(i), v.as_slice());
        }
    }

    #[test]
    fn dim_is_tracked() {
        let mut index = Builder::new().seed(6).build(Euclidean);
        assert_eq!(index.dim(), None);
        index.insert(vec![1.0, 2.0, 3.0]);
        assert_eq!(index.dim(), Some(3));
    }

    #[test]
    #[should_panic(expected = "expected 3")]
    fn wrong_dimension_panics() {
        let mut index = Builder::new().seed(7).build(Euclidean);
        index.insert(vec![1.0, 2.0, 3.0]);
        index.insert(vec![1.0, 2.0]); // wrong dim → panic
    }

    // ── recall tests ──────────────────────────────────────────────────────

    fn recall(index: &Hnsw<Euclidean>, vectors: &[Vec<f32>], k: usize, ef: usize, n_queries: usize) -> f64 {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(99_999);
        let dim = vectors[0].len();

        let mut hits = 0usize;
        let mut total = 0usize;

        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            let exact = exact_knn(vectors, &query, k);
            let approx: Vec<usize> = index.search(&query, k, ef).iter().map(|r| r.id).collect();
            let exact_set: std::collections::HashSet<usize> = exact.into_iter().collect();
            for id in &approx {
                if exact_set.contains(id) {
                    hits += 1;
                }
            }
            total += k;
        }

        hits as f64 / total as f64
    }

    #[test]
    fn recall_128d_is_acceptable() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(77);
        let dim = 128;
        let n = 1_000;

        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut index = Builder::new()
            .m(16)
            .ef_construction(200)
            .seed(42)
            .build(Euclidean);

        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.insert(v.clone());
            vectors.push(v);
        }

        let r = recall(&index, &vectors, 10, 100, 100);
        println!("Recall@10 (128d, 1k vectors, ef=100): {:.2}%", r * 100.0);
        // Expect ≥ 90 % recall with these parameters.
        assert!(r >= 0.90, "recall {:.2}% is too low", r * 100.0);
    }

    #[test]
    fn recall_32d_high_ef_is_near_perfect() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(55);
        let dim = 32;
        let n = 500;

        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut index = Builder::new()
            .m(32)
            .ef_construction(400)
            .seed(13)
            .build(Euclidean);

        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.insert(v.clone());
            vectors.push(v);
        }

        let r = recall(&index, &vectors, 10, 500, 50);
        println!("Recall@10 (32d, 500 vectors, ef=500): {:.2}%", r * 100.0);
        assert!(r >= 0.98, "recall {:.2}% is too low", r * 100.0);
    }

    // ── distance metric tests ─────────────────────────────────────────────

    #[test]
    fn squared_euclidean_finds_correct_neighbour() {
        let mut index = Builder::new().seed(10).build(SquaredEuclidean);
        index.insert(vec![0.0, 0.0]); // id=0
        index.insert(vec![1.0, 0.0]); // id=1
        index.insert(vec![5.0, 0.0]); // id=2
        let res = index.search(&[0.2, 0.0], 1, 10);
        assert_eq!(res[0].id, 0);
    }

    #[test]
    fn cosine_distance_orthogonal_vectors() {
        let mut index = Builder::new().seed(11).build(Cosine);
        index.insert(vec![1.0, 0.0]); // id=0
        index.insert(vec![0.0, 1.0]); // id=1  orthogonal
        index.insert(vec![0.9, 0.1]); // id=2  close to id=0
        let res = index.search(&[1.0, 0.0], 1, 10);
        assert_eq!(res[0].id, 0);
    }

    #[test]
    fn manhattan_metric_correct_order() {
        let mut index = Builder::new().seed(12).build(Manhattan);
        index.insert(vec![0.0]);  // id=0, dist=1.0 from query 1.0
        index.insert(vec![10.0]); // id=1, dist=9.0 from query 1.0
        index.insert(vec![1.5]);  // id=2, dist=0.5 from query 1.0
        let res = index.search(&[1.0], 1, 10);
        assert_eq!(res[0].id, 2);
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn two_identical_vectors() {
        let mut index = Builder::new().seed(20).build(Euclidean);
        index.insert(vec![1.0, 1.0]); // id=0
        index.insert(vec![1.0, 1.0]); // id=1  duplicate
        let res = index.search(&[1.0, 1.0], 2, 10);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].distance, 0.0);
        assert_eq!(res[1].distance, 0.0);
    }

    #[test]
    fn one_dimensional_vectors() {
        let mut index = Builder::new().seed(21).build(Euclidean);
        for i in 0..50_u32 {
            index.insert(vec![i as f32]);
        }
        let res = index.search(&[25.0], 3, 30);
        let ids: Vec<usize> = res.iter().map(|r| r.id).collect();
        assert!(ids.contains(&25));
    }

    #[test]
    fn large_dimension_does_not_panic() {
        let mut index = Builder::new().m(8).ef_construction(50).seed(22).build(Euclidean);
        let dim: usize = 1024;
        for i in 0..50_u32 {
            let v: Vec<f32> = (0..dim).map(|j| (i as usize + j) as f32).collect();
            index.insert(v);
        }
        let query: Vec<f32> = vec![1.0; dim];
        let res = index.search(&query, 5, 20);
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn simple_neighbour_selection_fallback() {
        let mut index = Builder::new()
            .m(16)
            .ef_construction(100)
            .heuristic(false) // use simple selection
            .seed(30)
            .build(Euclidean);
        for i in 0..100_u32 {
            index.insert(vec![i as f32, 0.0]);
        }
        let res = index.search(&[50.0, 0.0], 3, 30);
        // Should include 50
        assert!(res.iter().any(|r| r.id == 50));
    }

    // ── stats ─────────────────────────────────────────────────────────────

    #[test]
    fn stats_are_consistent() {
        let index = build_index(500, 32, 50);
        let stats = index.stats();
        assert_eq!(stats.num_vectors, 500);
        // Layer 0 must contain all nodes.
        assert_eq!(stats.layer_counts[0], 500);
        // Edge count must be even (undirected).
        assert_eq!(stats.layer_edges[0] % 2, 0);
        println!("{}", stats);
    }

    // ── Persistence tests ─────────────────────────────────────────────────

    fn make_hnsw(n: usize, dim: usize, seed: u64) -> (Hnsw<Euclidean>, Vec<Vec<f32>>) {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed + 5_000);
        let mut index = Builder::new().m(16).ef_construction(200).seed(seed).build(Euclidean);
        let mut corpus = Vec::with_capacity(n);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.insert(v.clone());
            corpus.push(v);
        }
        (index, corpus)
    }

    #[test]
    fn persist_save_load_round_trip() {
        let (orig, _) = make_hnsw(200, 16, 300);
        let dir = tempdir();
        let path = dir.join("test.hnsw");
        persist::save(&orig, &path).expect("save failed");

        let loaded = persist::load(&path, Euclidean).expect("load failed");
        assert_eq!(orig.len(), loaded.len());
        assert_eq!(orig.dim(), loaded.dim());
        // Vectors must be identical.
        for i in 0..orig.len() {
            assert_eq!(orig.get_vector(i), loaded.get_vector(i),
                       "vector {i} differs after load");
        }
        // Search results must be identical (same graph topology).
        let q = vec![0.5f32; 16];
        let r_orig   = orig.search(&q, 5, 50);
        let r_loaded = loaded.search(&q, 5, 50);
        assert_eq!(r_orig.len(), r_loaded.len());
        for (a, b) in r_orig.iter().zip(r_loaded.iter()) {
            assert_eq!(a.id, b.id, "search result id differs");
            assert!((a.distance - b.distance).abs() < 1e-6,
                    "distance differs: {} vs {}", a.distance, b.distance);
        }
    }

    #[test]
    fn persist_mmap_load_round_trip() {
        let (orig, _) = make_hnsw(200, 16, 301);
        let dir  = tempdir();
        let path = dir.join("mmap_test.hnsw");
        persist::save(&orig, &path).expect("save failed");

        let mmap = persist::load_mmap(&path, Euclidean).expect("mmap load failed");
        assert!(matches!(
            &mmap.graph,
            crate::hnsw::GraphStore::Mapped(_)
        ));
        assert_eq!(orig.len(), mmap.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get_vector(i), mmap.get_vector(i),
                       "mmap vector {i} differs");
        }
        let q = vec![0.3f32; 16];
        let r_orig = orig.search(&q, 5, 50);
        let r_mmap = mmap.search(&q, 5, 50);
        for (a, b) in r_orig.iter().zip(r_mmap.iter()) {
            assert_eq!(a.id, b.id);
        }
        let filtered_orig = orig.search_filtered(&q, 5, 200, |id| id % 7 == 0);
        let filtered_mmap = mmap.search_filtered(&q, 5, 200, |id| id % 7 == 0);
        assert_eq!(filtered_orig, filtered_mmap);
    }

    #[test]
    fn compact_snapshot_is_smaller_and_search_equivalent() {
        let (orig, _) = make_hnsw(200, 16, 303);
        let dir = tempdir();
        let v1_path = dir.join("full-v1.hnsw");
        let v2_path = dir.join("compact-v2.hnsw");
        persist::save(&orig, &v1_path).expect("v1 save failed");

        // Re-encoding an existing mmap snapshot is an important production
        // path: compacting does not require rebuilding the graph.
        let v1_mmap =
            persist::load_mmap(&v1_path, Euclidean).expect("v1 mmap load failed");
        persist::save_compact(&v1_mmap, &v2_path).expect("v2 save failed");

        let mut edge_count = 0;
        for node in 0..orig.graph.node_count() {
            for layer in 0..orig.graph.level_count(node) {
                edge_count += orig.graph.neighbour_count(node, layer);
            }
        }
        let v1_bytes = std::fs::metadata(&v1_path).expect("v1 metadata failed").len();
        let v2_bytes = std::fs::metadata(&v2_path).expect("v2 metadata failed").len();
        assert_eq!(v1_bytes - v2_bytes, edge_count as u64 * 4);

        let compact =
            persist::load_mmap(&v2_path, Euclidean).expect("v2 mmap load failed");
        assert_eq!(orig.len(), compact.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get_vector(i), compact.get_vector(i));
        }

        let query = vec![0.3f32; 16];
        assert_eq!(
            orig.search(&query, 10, 100),
            compact.search(&query, 10, 100)
        );
        assert_eq!(
            orig.search_filtered(&query, 10, 200, |id| id % 7 == 0),
            compact.search_filtered(&query, 10, 200, |id| id % 7 == 0)
        );
    }

    #[test]
    fn compact_snapshot_rejects_owned_load() {
        let (index, _) = make_hnsw(20, 8, 304);
        let dir = tempdir();
        let path = dir.join("compact-owned.hnsw");
        persist::save_compact(&index, &path).expect("compact save failed");

        let error = persist::load(&path, Euclidean)
            .err()
            .expect("owned compact load should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("read-only"));
        assert!(error.to_string().contains("load_mmap"));
    }

    #[test]
    fn compact_snapshot_supports_variable_and_fixed_payloads() {
        let (index, _) = make_hnsw(30, 8, 305);
        let dir = tempdir();

        let string_path = dir.join("compact-strings.hnsw");
        let strings: Vec<String> = (0..index.len()).map(|id| format!("item-{id}")).collect();
        persist::save_compact_with_payload(&index, &strings, &string_path)
            .expect("compact variable payload save failed");
        let (string_index, loaded_strings) =
            persist::load_mmap_with_payload::<_, String>(&string_path, Euclidean)
                .expect("compact variable payload mmap load failed");
        assert_eq!(loaded_strings, strings);
        assert_eq!(string_index.len(), index.len());

        let fixed_path = dir.join("compact-u32.hnsw");
        let labels: Vec<u32> = (0..index.len() as u32).map(|id| id * 10).collect();
        persist::save_compact_with_payload(&index, &labels, &fixed_path)
            .expect("compact fixed payload save failed");
        let (fixed_index, mapped_labels) =
            persist::load_mmap_with_fixed_payload::<_, u32>(&fixed_path, Euclidean)
                .expect("compact fixed payload mmap load failed");
        assert_eq!(fixed_index.len(), index.len());
        for (id, expected) in labels.iter().copied().enumerate() {
            assert_eq!(mapped_labels.get(id).expect("payload decode failed"), expected);
        }
    }

    #[test]
    fn compact_snapshot_rejects_truncated_adjacency() {
        let (index, _) = make_hnsw(20, 8, 306);
        let dir = tempdir();
        let path = dir.join("compact-truncated.hnsw");
        persist::save_compact(&index, &path).expect("compact save failed");

        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open failed");
        let len = file.metadata().expect("metadata failed").len();
        file.set_len(len - 17).expect("truncate failed");
        drop(file);

        let error = persist::load_mmap(&path, Euclidean)
            .err()
            .expect("truncated compact adjacency should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("connection list"));
    }

    #[test]
    fn compact_snapshot_rejects_out_of_range_neighbour() {
        use std::io::{Read, Seek, SeekFrom, Write};

        let n = 20;
        let dim = 8;
        let (index, _) = make_hnsw(n, dim, 307);
        let dir = tempdir();
        let path = dir.join("compact-invalid-neighbour.hnsw");
        persist::save_compact(&index, &path).expect("compact save failed");

        let offsets_start = 256 + n * dim * 4 + n * 4;
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open failed");
        file.seek(SeekFrom::Start(offsets_start as u64))
            .expect("seek failed");
        let mut offset = [0; 8];
        file.read_exact(&mut offset).expect("offset read failed");
        let first_record = u64::from_le_bytes(offset);
        file.seek(SeekFrom::Start(first_record))
            .expect("record seek failed");
        let mut count = [0; 4];
        file.read_exact(&mut count).expect("count read failed");
        assert!(u32::from_le_bytes(count) > 0);
        file.write_all(&(n as u32).to_le_bytes())
            .expect("neighbour write failed");
        drop(file);

        let error = persist::load_mmap(&path, Euclidean)
            .err()
            .expect("out-of-range neighbour should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("invalid neighbour id"));
    }

    #[test]
    fn persist_mmap_rejects_invalid_graph_offset() {
        use std::io::{Seek, SeekFrom, Write};

        let n = 20;
        let dim = 8;
        let (index, _) = make_hnsw(n, dim, 302);
        let dir = tempdir();
        let path = dir.join("bad_graph_offset.hnsw");
        persist::save(&index, &path).expect("save failed");

        let offsets_start = 256 + n * dim * 4 + n * 4;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open failed");
        file.seek(SeekFrom::Start(offsets_start as u64))
            .expect("seek failed");
        file.write_all(&0u64.to_le_bytes()).expect("write failed");
        drop(file);

        let error = persist::load_mmap(&path, Euclidean)
            .err()
            .expect("invalid graph offset should be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("invalid connection offset"));
    }

    #[test]
    fn persist_empty_index() {
        let empty: Hnsw<Euclidean> = Builder::new().build(Euclidean);
        let dir  = tempdir();
        let path = dir.join("empty.hnsw");
        persist::save(&empty, &path).expect("save empty failed");
        let loaded = persist::load(&path, Euclidean).expect("load empty failed");
        assert_eq!(loaded.len(), 0);
        assert!(loaded.search(&[0.0, 1.0], 5, 10).is_empty());
    }

    // ── LabeledIndex tests ────────────────────────────────────────────────

    #[test]
    fn labeled_insert_and_search_u32() {
        let mut idx: LabeledIndex<Euclidean, u32> =
            Builder::new().seed(400).build_labeled(Euclidean);
        idx.insert(vec![0.0, 0.0], 10_u32);
        idx.insert(vec![1.0, 0.0], 20_u32);
        idx.insert(vec![0.0, 1.0], 30_u32);

        let hits = idx.search(&[0.1, 0.0], 1, 20);
        assert_eq!(hits[0].payload, &10_u32);
        assert_eq!(hits[0].id, 0);

        let filtered = idx.search_filtered(&[0.1, 0.0], 2, 20, |_id, payload| *payload >= 20);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|result| *result.payload >= 20));
    }

    #[test]
    fn labeled_insert_and_search_string() {
        let mut idx: LabeledIndex<Euclidean, String> =
            Builder::new().seed(401).build_labeled(Euclidean);
        idx.insert(vec![1.0, 0.0], "cat".to_string());
        idx.insert(vec![0.0, 1.0], "dog".to_string());
        idx.insert(vec![0.5, 0.5], "rabbit".to_string());

        let hits = idx.search(&[0.9, 0.1], 1, 20);
        assert_eq!(hits[0].payload, "cat");
        assert_eq!(hits[0].embedding, &[1.0f32, 0.0]);
    }

    #[test]
    fn labeled_search_returns_embedding() {
        let mut idx: LabeledIndex<Euclidean, ()> =
            Builder::new().seed(402).build_labeled(Euclidean);
        let v = vec![3.0f32, 4.0];
        idx.insert(v.clone(), ());
        let hits = idx.search(&[3.0, 4.0], 1, 10);
        assert_eq!(hits[0].embedding, v.as_slice());
    }

    #[test]
    fn labeled_save_load_u32() {
        let mut idx: LabeledIndex<Euclidean, u32> =
            Builder::new().seed(410).build_labeled(Euclidean);
        for i in 0..50_u32 {
            idx.insert(vec![i as f32, (i * 2) as f32], i * 10);
        }
        let dir  = tempdir();
        let path = dir.join("labeled_u32.hnsw");
        idx.save(&path).expect("save failed");

        let loaded = LabeledIndex::<Euclidean, u32>::load(&path, Euclidean)
            .expect("load failed");
        assert_eq!(loaded.len(), 50);
        for i in 0..50_usize {
            assert_eq!(loaded.get_payload(i), &(i as u32 * 10));
            assert_eq!(loaded.get_embedding(i), &[i as f32, (i * 2) as f32]);
        }
        let hits = loaded.search(&[25.0, 50.0], 1, 30);
        assert_eq!(hits[0].id, 25);
        assert_eq!(hits[0].payload, &250_u32);
    }

    #[test]
    fn labeled_save_load_string() {
        let labels = ["alpha", "beta", "gamma", "delta", "epsilon"];
        let mut idx: LabeledIndex<Euclidean, String> =
            Builder::new().seed(411).build_labeled(Euclidean);
        for (i, &s) in labels.iter().enumerate() {
            idx.insert(vec![i as f32], s.to_string());
        }
        let dir  = tempdir();
        let path = dir.join("labeled_str.hnsw");
        idx.save(&path).expect("save failed");

        let loaded = LabeledIndex::<Euclidean, String>::load(&path, Euclidean)
            .expect("load failed");
        for (i, &s) in labels.iter().enumerate() {
            assert_eq!(loaded.get_payload(i), s);
        }
    }

    #[test]
    fn labeled_save_load_vec_f32_payload() {
        // Payload is a secondary embedding (variable-width)
        let mut idx: LabeledIndex<Euclidean, Vec<f32>> =
            Builder::new().seed(412).build_labeled(Euclidean);
        let primary = vec![1.0f32, 0.0];
        let secondary = vec![0.0f32, 0.0, 1.0]; // different dim
        idx.insert(primary.clone(), secondary.clone());
        let dir  = tempdir();
        let path = dir.join("labeled_vecf32.hnsw");
        idx.save(&path).expect("save failed");

        let loaded = LabeledIndex::<Euclidean, Vec<f32>>::load(&path, Euclidean)
            .expect("load failed");
        assert_eq!(loaded.get_payload(0), &secondary);
    }

    #[test]
    fn labeled_mmap_load() {
        let mut idx: LabeledIndex<Euclidean, u32> =
            Builder::new().seed(420).build_labeled(Euclidean);
        for i in 0..30_u32 {
            idx.insert(vec![i as f32], i);
        }
        let dir  = tempdir();
        let path = dir.join("labeled_mmap.hnsw");
        idx.save(&path).expect("save failed");

        let mmap = LabeledIndex::<Euclidean, u32>::load_mmap(&path, Euclidean)
            .expect("mmap load failed");
        assert_eq!(mmap.len(), 30);
        for i in 0..30_usize {
            assert_eq!(mmap.get_payload(i), &(i as u32));
        }
    }

    #[test]
    fn labeled_fixed_payloads_stay_mapped() {
        let mut idx: LabeledIndex<Euclidean, u32> =
            Builder::new().seed(421).build_labeled(Euclidean);
        for i in 0..30_u32 {
            idx.insert(vec![i as f32], i * 10);
        }
        let dir = tempdir();
        let path = dir.join("labeled_fixed_mmap.hnsw");
        idx.save(&path).expect("save failed");

        let mmap = LabeledIndex::<Euclidean, u32>::load_mmap_fixed(&path, Euclidean)
            .expect("fixed mmap load failed");
        assert!(matches!(
            &mmap.inner.graph,
            crate::hnsw::GraphStore::Mapped(_)
        ));
        assert_eq!(mmap.len(), 30);
        for i in 0..30_usize {
            assert_eq!(mmap.get_payload(i).expect("payload decode failed"), i as u32 * 10);
        }
        assert_eq!(
            mmap.get_payload(30).expect_err("out-of-range id should fail").kind(),
            std::io::ErrorKind::InvalidInput
        );

        let hits = mmap.search(&[12.0], 1, 20).expect("search failed");
        assert_eq!(hits[0].id, 12);
        assert_eq!(hits[0].payload, 120);
        assert_eq!(hits[0].embedding, &[12.0]);

        let filtered = mmap
            .search_filtered(&[12.0], 3, 30, |_id, payload| *payload % 40 == 0)
            .expect("filtered search failed");
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0].id, 12);
        assert!(filtered.iter().all(|result| result.payload % 40 == 0));
    }

    #[test]
    fn labeled_mapped_payloads_reject_variable_width_types() {
        let mut idx: LabeledIndex<Euclidean, String> =
            Builder::new().seed(422).build_labeled(Euclidean);
        idx.insert(vec![1.0], "one".to_string());
        let dir = tempdir();
        let path = dir.join("labeled_variable_mmap.hnsw");
        idx.save(&path).expect("save failed");

        let error = LabeledIndex::<Euclidean, String>::load_mmap_fixed(&path, Euclidean)
            .err()
            .expect("variable-width mapped payload should be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn labeled_mapped_payloads_reject_truncated_columns() {
        let mut idx: LabeledIndex<Euclidean, u64> =
            Builder::new().seed(423).build_labeled(Euclidean);
        idx.insert(vec![1.0], 10);
        idx.insert(vec![2.0], 20);
        let dir = tempdir();
        let path = dir.join("labeled_truncated_mmap.hnsw");
        idx.save(&path).expect("save failed");

        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open failed");
        let truncated_len = file.metadata().expect("metadata failed").len() - 1;
        file.set_len(truncated_len).expect("truncate failed");
        drop(file);

        let error = LabeledIndex::<Euclidean, u64>::load_mmap_fixed(&path, Euclidean)
            .err()
            .expect("truncated mapped payload should be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("payload column"));
    }

    // ── PairedIndex tests ─────────────────────────────────────────────────

    #[test]
    fn paired_insert_and_search_both_sides() {
        let mut idx: PairedIndex<Euclidean, Euclidean> = Builder::new()
            .m(16).ef_construction(50).seed(500)
            .build_paired(Euclidean, Euclidean);

        // Three items: each has a 2-D A-embedding and 3-D B-embedding.
        idx.insert(vec![1.0, 0.0],       vec![0.9, 0.1, 0.0]);   // id=0
        idx.insert(vec![0.0, 1.0],       vec![0.1, 0.8, 0.1]);   // id=1
        idx.insert(vec![0.5, 0.5],       vec![0.3, 0.3, 0.4]);   // id=2

        // Search A-space: query near item 0
        let hits_a = idx.search_by_a(&[0.9, 0.1], 1, 20);
        assert_eq!(hits_a[0].id, 0);
        assert_eq!(hits_a[0].emb_b, &[0.9f32, 0.1, 0.0]);

        // Search B-space: query near item 1
        let hits_b = idx.search_by_b(&[0.1, 0.9, 0.0], 1, 20);
        assert_eq!(hits_b[0].id, 1);
        assert_eq!(hits_b[0].emb_a, &[0.0f32, 1.0]);
    }

    #[test]
    fn paired_len_consistent() {
        let mut idx: PairedIndex<Euclidean, Euclidean> =
            PairedIndex::new(Default::default(), Euclidean, Default::default(), Euclidean);
        assert_eq!(idx.len(), 0);
        for i in 0..10_u32 {
            idx.insert(vec![i as f32], vec![i as f32, i as f32]);
            assert_eq!(idx.len(), i as usize + 1);
        }
    }

    #[test]
    fn paired_cross_side_retrieval() {
        let mut idx: PairedIndex<Euclidean, Euclidean> = Builder::new()
            .m(16).ef_construction(100).seed(501)
            .build_paired(Euclidean, Euclidean);
        // 20 items
        for i in 0..20_u32 {
            idx.insert(vec![i as f32, 0.0], vec![0.0, i as f32]);
        }
        // Search by A near item 10 → get B embedding of item 10
        let hits = idx.search_by_a(&[10.0, 0.0], 1, 30);
        assert_eq!(hits[0].id, 10);
        assert_eq!(hits[0].emb_b, &[0.0f32, 10.0]);
        // Confirm: searching by B near item 10 → get A embedding of item 10
        let hits2 = idx.search_by_b(&[0.0, 10.0], 1, 30);
        assert_eq!(hits2[0].id, 10);
        assert_eq!(hits2[0].emb_a, &[10.0f32, 0.0]);
    }

    #[test]
    fn paired_save_load() {
        let mut idx: PairedIndex<Euclidean, Euclidean> = Builder::new()
            .m(16).ef_construction(100).seed(510)
            .build_paired(Euclidean, Euclidean);
        for i in 0..50_u32 {
            idx.insert(vec![i as f32], vec![i as f32, i as f32]);
        }
        let dir = tempdir();
        let base = dir.join("paired");
        idx.save(&base).expect("save failed");

        let loaded = PairedIndex::<Euclidean, Euclidean>::load(&base, Euclidean, Euclidean)
            .expect("load failed");
        assert_eq!(loaded.len(), 50);
        for i in 0..50_usize {
            assert_eq!(loaded.get_emb_a(i), &[i as f32][..]);
            assert_eq!(loaded.get_emb_b(i), &[i as f32, i as f32][..]);
        }
        let hits = loaded.search_by_a(&[25.0], 1, 30);
        assert_eq!(hits[0].id, 25);
    }

    #[test]
    fn paired_mmap_load() {
        let mut idx: PairedIndex<Euclidean, Euclidean> = Builder::new()
            .seed(520).build_paired(Euclidean, Euclidean);
        for i in 0..30_u32 {
            idx.insert(vec![i as f32, 0.0], vec![0.0, i as f32, 1.0]);
        }
        let dir  = tempdir();
        let base = dir.join("paired_mmap");
        idx.save(&base).expect("save failed");

        let m = PairedIndex::<Euclidean, Euclidean>::load_mmap(&base, Euclidean, Euclidean)
            .expect("mmap load failed");
        assert_eq!(m.len(), 30);
        // Spot-check a few vectors
        for i in [0, 15, 29] {
            assert_eq!(m.get_emb_a(i), &[i as f32, 0.0f32][..]);
            assert_eq!(m.get_emb_b(i), &[0.0f32, i as f32, 1.0][..]);
        }
    }

    // Helper: create a temp directory that lives for the duration of the test.
    fn tempdir() -> std::path::PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
        let dir = std::env::temp_dir().join(format!("hnsw_test_{ts}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // ── PruneStrategy tests ───────────────────────────────────────────────

    /// Helper: build an index with a given prune strategy and return
    /// (index, corpus) so callers can run recall checks.
    fn build_with_prune(n: usize, dim: usize, seed: u64, ps: PruneStrategy)
        -> (Hnsw<Euclidean>, Vec<Vec<f32>>)
    {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed + 2_000);
        let mut index = Builder::new()
            .m(16)
            .ef_construction(200)
            .prune_strategy(ps)
            .seed(seed)
            .build(Euclidean);
        let mut corpus = Vec::with_capacity(n);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.insert(v.clone());
            corpus.push(v);
        }
        (index, corpus)
    }

    #[test]
    fn prune_strategy_default_is_simple() {
        // Ensure Config::default() picks Simple so users get the fastest
        // behaviour out of the box without any builder call.
        assert_eq!(Config::default().prune_strategy, PruneStrategy::Simple);
        // Building via Builder without calling .prune_strategy() must also
        // default to Simple.
        let mut index = Builder::new().seed(0).build(Euclidean);
        index.insert(vec![1.0, 2.0]);
        // The index built successfully — no panics, correct result.
        assert_eq!(index.search(&[1.0, 2.0], 1, 10)[0].id, 0);
    }

    #[test]
    fn prune_strategy_simple_gives_acceptable_recall() {
        let (index, corpus) = build_with_prune(500, 32, 101, PruneStrategy::Simple);
        let r = recall(&index, &corpus, 10, 200, 50);
        println!("Simple recall@10 (32d 500v ef=200): {:.2}%", r * 100.0);
        assert!(r >= 0.95, "Simple recall {:.2}% too low", r * 100.0);
    }

    #[test]
    fn prune_strategy_heuristic_gives_acceptable_recall() {
        let (index, corpus) = build_with_prune(500, 32, 101, PruneStrategy::Heuristic);
        let r = recall(&index, &corpus, 10, 200, 50);
        println!("Heuristic recall@10 (32d 500v ef=200): {:.2}%", r * 100.0);
        assert!(r >= 0.95, "Heuristic recall {:.2}% too low", r * 100.0);
    }

    #[test]
    fn prune_strategy_heuristic_recall_ge_simple() {
        // Heuristic must not be worse than Simple (it does strictly more work
        // to preserve diversity).  Run both on the same data and seed.
        let (idx_s, corpus) = build_with_prune(500, 128, 202, PruneStrategy::Simple);
        let (idx_h, _)      = build_with_prune(500, 128, 202, PruneStrategy::Heuristic);
        let r_s = recall(&idx_s, &corpus, 10, 100, 50);
        let r_h = recall(&idx_h, &corpus, 10, 100, 50);
        println!("Simple {:.2}%  Heuristic {:.2}%", r_s * 100.0, r_h * 100.0);
        // Allow up to 1 pp slack for statistical noise in the random queries.
        assert!(r_h + 0.01 >= r_s,
            "Heuristic recall ({:.2}%) should be ≥ Simple ({:.2}%)",
            r_h * 100.0, r_s * 100.0);
    }

    #[test]
    fn max_level_grows_with_more_inserts() {
        let index_small = build_index(10, 4, 60);
        let index_large = build_index(10_000, 4, 60);
        // With many more nodes the entry-point level is likely higher.
        // This is probabilistic but almost certain with 10 000 vs 10 nodes.
        let l_small = index_small.max_level().unwrap_or(0);
        let l_large = index_large.max_level().unwrap_or(0);
        println!("small max_level={l_small}, large max_level={l_large}");
        assert!(l_large >= l_small);
    }
}
