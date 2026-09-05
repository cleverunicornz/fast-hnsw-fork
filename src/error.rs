//! The error type returned by fallible operations.
//!
//! No public operation in this crate panics on bad input. Anything that can be
//! wrong about a call — a mismatched query, an out-of-range id, a read-only
//! index, an invalid configuration — comes back as an [`Error`] describing
//! what was expected and what arrived.
//!
//! Persistence keeps returning [`std::io::Result`], because those failures are
//! genuinely I/O-shaped (a missing file, a truncated read, a corrupt header)
//! and callers already handle them as such. [`Error`] converts into
//! [`std::io::Error`] and back, so a single `?` works in either direction:
//!
//! ```
//! use fast_hnsw::{Builder, Hnsw};
//! use fast_hnsw::distance::Euclidean;
//!
//! fn build_and_save(path: &str) -> std::io::Result<()> {
//!     let mut index: Hnsw<Euclidean> = Builder::new().build(Euclidean)?; // Error -> io::Error
//!     index.insert(vec![1.0, 0.0])?;
//!     fast_hnsw::persist::save(&index, path)                             // io::Error
//! }
//! # let dir = std::env::temp_dir().join("fast_hnsw_error_doc.hnsw");
//! # build_and_save(dir.to_str().unwrap()).unwrap();
//! # std::fs::remove_file(dir).ok();
//! ```

use std::fmt;

/// Result alias for operations that can fail with an [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Something was wrong with a call into this crate.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A vector or query did not match the index's established dimension.
    ///
    /// Every metric folds pairwise, so a mismatched length would otherwise be
    /// silently truncated and produce confident, wrong answers.
    DimensionMismatch {
        /// The dimension the index stores.
        expected: usize,
        /// The dimension that was supplied.
        actual: usize,
    },

    /// `k` was zero, so there is no meaningful result to return.
    ZeroK,

    /// The index is backed by a read-only memory mapping and cannot be
    /// mutated. Rebuild it from an owned load if you need to insert.
    ReadOnly,

    /// The operation needs the vectors in memory, but this index defers them
    /// to disk. Read them through
    /// [`PreadIndex`](crate::pread::PreadIndex) instead.
    VectorsNotResident,

    /// An id was outside the range of stored vectors.
    IdOutOfBounds {
        /// The id that was requested.
        id: usize,
        /// The number of slots that exist.
        len: usize,
    },

    /// A build parameter was outside its supported range.
    InvalidConfig {
        /// Which parameter.
        parameter: &'static str,
        /// Why it was rejected.
        reason: String,
    },

    /// The two halves of a [`PairedIndex`](crate::paired::PairedIndex) carry
    /// different tombstones, so an operation that must renumber them together
    /// cannot proceed.
    ///
    /// Only reachable by deleting through the public `index_a` / `index_b`
    /// fields individually, which bypasses the paired API.
    DesynchronizedPair,

    /// A worker thread panicked while holding a lock during a parallel build,
    /// leaving the graph in an unknown state.
    Poisoned,

    /// An underlying I/O or persistence failure.
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "dimension mismatch: this index stores {expected}-dimensional \
                 vectors but {actual} components were supplied",
            ),
            Self::ZeroK => f.write_str("k must be greater than zero"),
            Self::ReadOnly => f.write_str(
                "this index is backed by a read-only memory mapping and cannot be modified",
            ),
            Self::VectorsNotResident => f.write_str(
                "this index keeps its vectors on disk; read them through PreadIndex",
            ),
            Self::IdOutOfBounds { id, len } => {
                write!(f, "id {id} is out of bounds for an index with {len} slots")
            }
            Self::InvalidConfig { parameter, reason } => {
                write!(f, "invalid configuration for `{parameter}`: {reason}")
            }
            Self::DesynchronizedPair => f.write_str(
                "the two sides of this paired index carry different tombstones, so they \
                 would be renumbered differently; delete through PairedIndex::remove \
                 rather than through the individual sides",
            ),
            Self::Poisoned => {
                f.write_str("a worker thread panicked during the parallel build")
            }
            Self::Io(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<Error> for std::io::Error {
    /// Lets persistence code, which returns [`std::io::Result`], propagate
    /// these with `?`.
    fn from(error: Error) -> Self {
        match error {
            Error::Io(inner) => inner,
            other => {
                let kind = match other {
                    Error::IdOutOfBounds { .. }
                    | Error::DimensionMismatch { .. }
                    | Error::ZeroK
                    | Error::InvalidConfig { .. } => std::io::ErrorKind::InvalidInput,
                    _ => std::io::ErrorKind::Other,
                };
                std::io::Error::new(kind, other.to_string())
            }
        }
    }
}
