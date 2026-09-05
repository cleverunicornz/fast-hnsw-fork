# fast-hnsw-quantized

An optional mmap-native 2–4 bit vector sidecar for `fast-hnsw`. It keeps HNSW
node ordering and application metadata outside the codec, prepares each query
once, and scores packed rows during graph traversal without reconstructing
`Vec<f32>` candidates.

The default is 4 bits per dimension. A 384-dimensional row occupies 196 bytes
(192 bytes of codes plus one `f32` scale), compared with 1,536 bytes for the
original `f32` row. Exact vectors may remain in a separate mmap for final
candidate reranking.

## Quick start

```rust,no_run
use fast_hnsw::distance::Cosine;
use fast_hnsw::Builder;
use fast_hnsw_quantized::{
    MappedQuantizedVectors, QuantizedConfig, QuantizedHnsw, QuantizedMetric,
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]];
let mut index = Builder::new().seed(42).build(Cosine)?;
for vector in &vectors {
    index.insert(vector.clone())?;
}

let file = tempfile::NamedTempFile::new()?;
MappedQuantizedVectors::save(
    file.path(),
    2,
    QuantizedMetric::Cosine,
    QuantizedConfig::default(),
    &vectors,
)?;

// SAFETY: keep the sidecar inode immutable while the mapping is alive.
let mapped = unsafe { MappedQuantizedVectors::open_mmap_verified(file.path())? };
let serving = QuantizedHnsw::new(&index, &mapped)?;
let hits = serving.search(&[0.9, 0.1], 2, 32)?;
assert_eq!(hits[0].id, 0);
# Ok(())
# }
```

Disable default features to use the codec without depending on `fast-hnsw`:

```toml
[dependencies]
fast-hnsw-quantized = { version = "0.1", default-features = false }
```

That mode is suitable for another graph engine or an exact quantized scan.
`write_to` and `open_mmap_region` support embedding the self-contained sidecar
inside a larger application snapshot.

## Format and lifecycle

Version 1 stores a fixed header, packed row-major codes, one little-endian
`f32` reconstruction scale per row, and a CRC32 checksum. Dimensions that do
not evenly tile a supported FWHT block are zero-padded to the next multiple of
eight; callers continue to use the original dimension.

`open_mmap` performs structural and scale validation without scanning packed
code pages. `open_mmap_verified` additionally checks the complete CRC. Neither
method makes mutable files safe: a mapped inode must remain immutable. For
atomic installation, write a temporary sibling, verify it, then rename it.

The crate intentionally does not own labels, tenant or ACL filters, update
policy, exact reranking, or memory-pressure adaptation. `QuantizedHnsw`
delegates eligibility to fast-hnsw's in-traversal predicate so rejected nodes
remain navigable but cannot enter the result heap.

## Benchmark

Run the paired exact/quantized traversal and Recall@k check with:

```shell
cargo bench -p fast-hnsw-quantized --bench quantized
```

`FHQ_BENCH_ROWS`, `FHQ_BENCH_DIMENSIONS`, `FHQ_BENCH_QUERIES`,
`FHQ_BENCH_K`, `FHQ_BENCH_EF`, and `FHQ_BENCH_BITS` override the defaults.
The output reports graph build time, QPS and exact-scan recall for both HNSW
paths, and quantized payload bytes relative to the original `f32` matrix.
