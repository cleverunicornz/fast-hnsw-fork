//! Positional-read index: graph in memory, vectors left on disk.
//!
//! [`load_mmap`](crate::persist::load_mmap) maps the whole file and lets the
//! kernel decide what stays resident.  This module is the explicit-I/O
//! alternative: the graph — a few percent of the file — is read into memory
//! once, and each vector is fetched with a positional read (`pread` on Unix,
//! `seek_read` on Windows) at the moment it is scored.
//!
//! # Why choose this over mmap
//!
//! * **Errors are values.**  A truncated or vanished file surfaces as
//!   [`io::Error`] from the search call.  Through a mapping the same situation
//!   is `SIGBUS` — a fault the process cannot catch or attribute.
//! * **Bounded, predictable residency.**  Nothing but the graph is charged to
//!   the process, and the OS page cache is the only thing holding vector data.
//!   A mapped index instead grows its resident set as queries touch pages.
//! * **Network and fuse filesystems.**  `mmap` over NFS or a userspace
//!   filesystem ranges from slow to unsafe; positional reads are ordinary I/O.
//!
//! Memory is dominated by the graph rather than the vectors.  At `M = 16`, a
//! node's adjacency costs on the order of a hundred bytes, while a 768-dim
//! `f32` vector costs 3 KiB — so this typically holds a few percent of what a
//! fully resident index would.
//!
//! # What it costs
//!
//! One syscall per distance computation.  A cached mmap read is a load from
//! the page cache; a `pread` of the same cached page still traps into the
//! kernel.  Graph traversal performs hundreds of distance computations per
//! query, so with a warm cache this is **much** slower than the mapped path:
//!
//! | index | vectors off-heap | mmap | `pread` | |
//! |-------|------------------|------|---------|--|
//! | 50 000 × 128 | 26 MB | 51 µs/query | 876 µs/query | 17× slower |
//! | 20 000 × 768 | 61 MB | 182 µs/query | 1 111 µs/query | 6× slower |
//!
//! (`k = 10`, `ef = 100`, both returning identical results.)  The gap narrows
//! as vectors grow, because the fixed syscall cost is amortised over more
//! bytes, and narrows again when reads genuinely reach the device rather than
//! the page cache.
//!
//! **Choose this for the operational properties, not for throughput.**  If
//! queries per second is what matters and the data fits,
//! [`load_mmap`](crate::persist::load_mmap) is the right call.
//!
//! # Example
//!
//! ```no_run
//! use fast_hnsw::pread::PreadIndex;
//! use fast_hnsw::distance::Euclidean;
//!
//! let index = PreadIndex::open("index.hnsw", Euclidean)?;
//! for hit in index.search(&[0.1, 0.2], 10, 100)? {
//!     println!("id={} distance={}", hit.id, hit.distance);
//! }
//! # Ok::<(), std::io::Error>(())
//! ```

use std::cell::RefCell;
use std::fs::File;
use std::io;
use std::path::Path;

use crate::distance::Distance;
use crate::hnsw::{Hnsw, SearchResult, SearchWorkspace};

/// A file region holding `count` vectors of `dim` `f32`s, read on demand.
pub struct PreadVectors {
    file: File,
    /// Byte offset of vector 0 within the file.
    data_offset: u64,
    count: usize,
    dim: usize,
}

impl PreadVectors {
    /// Read vector `id` into `buf`, resizing it to the index dimension.
    ///
    /// Reusing one buffer across calls keeps the read allocation-free.
    pub fn read_into(&self, id: usize, buf: &mut Vec<f32>) -> io::Result<()> {
        if id >= self.count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("vector id {id} out of bounds for {} vectors", self.count),
            ));
        }
        buf.clear();
        buf.resize(self.dim, 0.0);

        let offset = self.data_offset + (id as u64) * (self.dim as u64) * 4;
        // Read straight into the destination's bytes: `f32` has no padding and
        // no invalid bit patterns, so any 4 bytes decode to some `f32`. The
        // file stores little-endian, which is handled below.
        let bytes = {
            let slice: &mut [f32] = buf.as_mut_slice();
            // SAFETY: `slice` covers `dim * 4` initialised bytes owned by
            // `buf`, and `u8` has weaker alignment than `f32`, so the
            // reinterpretation is valid for the lifetime of this borrow.
            unsafe {
                std::slice::from_raw_parts_mut(
                    slice.as_mut_ptr() as *mut u8,
                    self.dim * std::mem::size_of::<f32>(),
                )
            }
        };
        read_exact_at(&self.file, bytes, offset)?;

        // The on-disk format is little-endian; swap only where that differs
        // from the host so the common case costs nothing.
        #[cfg(target_endian = "big")]
        for value in buf.iter_mut() {
            *value = f32::from_le_bytes(value.to_ne_bytes());
        }
        Ok(())
    }

    /// Read vector `id` into a fresh `Vec`.
    pub fn read(&self, id: usize) -> io::Result<Vec<f32>> {
        let mut buf = Vec::new();
        self.read_into(id, &mut buf)?;
        Ok(buf)
    }

    /// Number of vectors on disk.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the region holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Positional read of exactly `buf.len()` bytes at `offset`.
///
/// `read_at`/`seek_read` may return short, so this loops. Unlike
/// `Seek` + `Read`, it does not disturb the file cursor, which is what makes
/// one `File` safely shareable across concurrent queries.
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        #[cfg(unix)]
        let read = {
            use std::os::unix::fs::FileExt;
            file.read_at(buf, offset)
        };
        #[cfg(windows)]
        let read = {
            use std::os::windows::fs::FileExt;
            file.seek_read(buf, offset)
        };
        #[cfg(not(any(unix, windows)))]
        let read: io::Result<usize> = Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "positional reads are not supported on this platform",
        ));

        match read {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "vector section ended early; file truncated?",
                ))
            }
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// An index whose graph is resident and whose vectors are read from disk.
///
/// See the [module docs](self) for the trade-off against
/// [`load_mmap`](crate::persist::load_mmap).
pub struct PreadIndex<D: Distance> {
    inner: Hnsw<D>,
    vectors: PreadVectors,
}

impl<D: Distance> PreadIndex<D> {
    /// Open `path`, loading only the graph into memory.
    ///
    /// Accepts both the mutable (v1) and compact (v2) formats.
    pub fn open(path: impl AsRef<Path>, metric: D) -> io::Result<Self> {
        crate::persist::load_pread(path, metric)
    }

    pub(crate) fn from_parts(inner: Hnsw<D>, vectors: PreadVectors) -> Self {
        Self { inner, vectors }
    }

    /// The `k` approximate nearest neighbours of `query`.
    ///
    /// Fails if `query.len()` differs from the index dimension, if `k` is
    /// zero, or if a vector read fails.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> io::Result<Vec<SearchResult>> {
        let mut workspace = SearchWorkspace::default();
        self.search_with_workspace(query, k, ef, &mut workspace)
    }

    /// Search reusing caller-owned traversal storage.
    pub fn search_with_workspace(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        workspace: &mut SearchWorkspace,
    ) -> io::Result<Vec<SearchResult>> {
        self.inner.check_query_dim(query)?;
        let state = ReadState::new();
        let results = self.inner.search_with_distance_and_workspace(
            k,
            ef,
            |id| state.distance(&self.vectors, &self.inner, query, id),
            workspace,
        )?;
        state.into_result(results)
    }

    /// Search with a filter-before-top-k eligibility predicate over ids.
    pub fn search_filtered<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accepts: F,
    ) -> io::Result<Vec<SearchResult>>
    where
        F: Fn(usize) -> bool,
    {
        self.inner.check_query_dim(query)?;
        let state = ReadState::new();
        let results = self.inner.search_filtered_with_distance(
            k,
            ef,
            |id| state.distance(&self.vectors, &self.inner, query, id),
            accepts,
        )?;
        state.into_result(results)
    }

    /// Read one stored vector from disk.
    pub fn get_vector(&self, id: usize) -> io::Result<Vec<f32>> {
        self.vectors.read(id)
    }

    /// The on-disk vector region, for direct reads.
    pub fn vectors(&self) -> &PreadVectors {
        &self.vectors
    }

    /// The underlying graph. Its vector accessors return
    /// [`Error::VectorsNotResident`](crate::Error::VectorsNotResident) — the
    /// vectors live on disk; use [`PreadIndex::get_vector`] instead.
    pub fn graph(&self) -> &Hnsw<D> {
        &self.inner
    }

    pub fn len(&self) -> usize {
        self.vectors.count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> Option<usize> {
        self.inner.dim()
    }
}

/// Per-query read buffer plus the first I/O error encountered.
///
/// The traversal callback must return an `f32`, so a failed read cannot
/// propagate directly. It is recorded here and re-raised once the search
/// returns; the failing node scores as infinitely far so it simply loses,
/// rather than corrupting the result ordering.
struct ReadState {
    buf: RefCell<Vec<f32>>,
    error: RefCell<Option<io::Error>>,
}

impl ReadState {
    fn new() -> Self {
        Self {
            buf: RefCell::new(Vec::new()),
            error: RefCell::new(None),
        }
    }

    fn distance<D: Distance>(
        &self,
        vectors: &PreadVectors,
        index: &Hnsw<D>,
        query: &[f32],
        id: usize,
    ) -> f32 {
        let mut buf = self.buf.borrow_mut();
        match vectors.read_into(id, &mut buf) {
            Ok(()) => index.metric().distance(query, &buf),
            Err(error) => {
                let mut slot = self.error.borrow_mut();
                if slot.is_none() {
                    *slot = Some(error);
                }
                f32::INFINITY
            }
        }
    }

    fn into_result(self, results: Vec<SearchResult>) -> io::Result<Vec<SearchResult>> {
        match self.error.into_inner() {
            Some(error) => Err(error),
            None => Ok(results),
        }
    }
}

impl PreadVectors {
    pub(crate) fn new(file: File, data_offset: u64, count: usize, dim: usize) -> Self {
        Self {
            file,
            data_offset,
            count,
            dim,
        }
    }
}
