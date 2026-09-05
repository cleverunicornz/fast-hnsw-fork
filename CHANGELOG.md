# Changelog

## 2.0.0

### Breaking: no public API panics

Every operation that could fail on user input now returns
`Result<_, fast_hnsw::Error>` instead of panicking. Previously a wrong-length
query, a zero `k`, an out-of-range id, an invalid `Config`, or an insert into a
read-only mapped index all aborted the calling thread — impossible for a server
to handle per-request.

`Error` converts to and from `std::io::Error`, so `?` works in either
direction and the persistence layer keeps its `io::Result` signatures.

Affected: `Hnsw::{new, new_with_seed, insert, search*, exact_search,
get_vector, compacted}`, `Builder::{build, build_labeled, build_paired,
build_parallel}`, `LabeledIndex::{new, from_builder, insert, search,
search_filtered, get_payload, get_embedding, compacted}`,
`PairedIndex::{new, from_builder, insert, search_by_a, search_by_b,
get_emb_a, get_emb_b, compacted}`.

`Hnsw::try_insert` and `InsertError` are removed; `insert` is now the fallible
form and reports `Error::DimensionMismatch` / `Error::ReadOnly`.

Migration is mechanical — add `?` or `.unwrap()` at the call site.

### Breaking: two silent-wrong-answer bugs now fail loudly

Both previously returned confident, incorrect results:

- **Mismatched query dimension.** Metrics fold pairwise, so a short query was
  silently truncated. Now `Error::DimensionMismatch`.
- **Mismatched metric on load.** The file format did not record which metric
  built the graph, so a Cosine index could be opened as Euclidean and answer
  with different neighbours. The metric is now recorded in the header and
  every load path rejects a mismatch.

Snapshots written by earlier versions record no metric and still load with any
metric; files written by 2.0 remain readable by older versions, which treat the
new header bytes as padding.

### Added

- **Deletion.** `remove` / `restore` / `is_deleted` / `deleted_count` /
  `live_len` / `compacted` on `Hnsw`, `LabeledIndex` and `PairedIndex`.
  Tombstones keep deleted nodes as navigation waypoints so the graph stays
  connected. Saving a tombstoned index is refused, since the format cannot
  record tombstones and writing it would resurrect deleted vectors.
- **`Hnsw::exact_search`** — exhaustive k-NN, for ground truth and small
  indexes. Uses BLAS `sgemv` for inner-product metrics when `blas` is enabled.
- **`PreadIndex`** (`pread` module) — graph in memory, vectors read from disk
  with positional reads. I/O errors surface as values rather than `SIGBUS`;
  substantially slower than mmap, so choose it for the operational properties.
- **`parallel` feature** — `Builder::build_parallel`, ~8x on 14 cores, at a
  few points of recall.
- **`simd` feature (default)** — hand-written NEON / AVX2 kernels, ~2x on
  AArch64 and ~3x on x86-64 over the auto-vectorised fold.
- **`avx512` feature** — 512-bit kernels; ~10-15% over AVX2 at 512+ dimensions.
  Raises the effective MSRV to 1.89.
- **`blas` feature** — routes `DotProduct`/`Cosine` through a system BLAS.
  Vendors nothing: `build.rs` emits a link directive only. Measure first; it is
  slower than the built-in SIMD in most configurations.
- **`serde` feature** — derives on `Config`, `SearchResult`, `IndexStats`,
  `PruneStrategy`. Indexes themselves are not serde types; use `persist`.
- **`Distance::metric_id` / `metric_name`** — defaulted, so existing custom
  metrics are unaffected.

### Fixed

- **Soundness:** `get_vector` on a memory-mapped index performed unchecked
  pointer arithmetic, so an out-of-range id from safe code was undefined
  behaviour. Now bounds-checked.
- The owned `load` path did not validate neighbour ids, offset tables, node
  levels, or section bounds, while `load_mmap` did. A corrupt file loaded
  "successfully" and failed later, far from the cause.
- The variable-width payload offset table was trusted, so a corrupt file could
  drive an unbounded allocation.
- `PairedIndex::load_mmap` did not check that both sides have equal length.

### Performance

- `search()` reuses a per-thread workspace instead of allocating an O(n)
  visited set per call — 2.5x at one million vectors.
- Distance kernels fold across independent accumulators and dispatch to SIMD.

### Removed

- Dead `CandidateHeap` / `ResultHeap` (never used; `Scratch` uses `BinaryHeap`).
