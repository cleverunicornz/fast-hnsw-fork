//! Binary file format and memory-mapped loading for [`Hnsw`] indexes.
//!
//! ## File layout
//!
//! All multi-byte integers are **little-endian**.  The layout is designed so
//! that the vector section sits at a fixed, known offset (`VECTORS_OFFSET =
//! 256`) — making it directly mmap-able as a `&[f32]` slice without any
//! reformatting.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────── offset 0
//! │  HEADER  (256 bytes, zero-padded)
//! │    0..8   magic     b"HNSWNDX\0"
//! │    8..12  version   u32 = 1 (mutable) or 2 (compact mmap-only)
//! │   12..20  n         u64  — number of vectors
//! │   20..28  dim       u64  — vector dimension
//! │   28..36  m         u64
//! │   36..44  m0        u64  — resolved (m0.unwrap_or(2*m))
//! │   44..52  ef        u64  — ef_construction
//! │   52..60  ep_id     u64  — entry-point id; u64::MAX = None
//! │   60..68  ep_level  u64
//! │   68      flags     use_heuristic | extend_candidates | keep_pruned | prune_strategy
//! │   69..256 padding   (zeros)
//! ├─────────────────────────────────────────────────────────── VECTORS_OFFSET = 256
//! │  VECTORS  (n × dim × 4 bytes)
//! │    f32 values, row-major, directly mmap-able
//! ├─────────────────────────────────────────────────────────── VECTORS_OFFSET + n*dim*4
//! │  LEVELS   (n × 4 bytes)
//! │    u32 per node — its maximum layer index (0 = only in layer 0)
//! ├─────────────────────────────────────────────────────────── after LEVELS
//! │  CONN OFFSETS  (n × 8 bytes)
//! │    u64 per node — absolute byte offset of that node's CONN DATA record
//! ├─────────────────────────────────────────────────────────── (variable)
//! │  CONN DATA  (variable, stored sequentially node 0 … n-1)
//! │    For node i (at conn_offsets[i]):
//! │      For each layer 0 ..= levels[i]:
//! │        n_conns : u32
//! │        v1: [(id: u32, dist: f32); n_conns]   8 bytes each
//! │        v2: [id: u32; n_conns]                 4 bytes each
//! ├─────────────────────────────────────────────────────────── (variable)
//! │  PAYLOAD HEADER  (16 bytes)
//! │    payload_count  : u64  — number of payload entries (= n, or 0)
//! │    payload_stride : u64  — bytes per entry; 0 = variable-width
//! ├─────────────────────────────────────────────────────────── (if stride == 0)
//! │  PAYLOAD VAR-OFFSETS  (payload_count × 8 bytes)
//! │    u64 absolute offsets into PAYLOAD DATA for each entry
//! ├─────────────────────────────────────────────────────────── (variable)
//! │  PAYLOAD DATA  (raw encoded bytes)
//! └───────────────────────────────────────────────────────────
//! ```
//!
//! ## I/O optimisations
//!
//! ### Write path
//! * **Pre-allocated disk space** (`save` / `save_with_payload`): the exact
//!   file size is computed before the first byte is written and passed to
//!   [`File::set_len`].  On most file systems this causes the kernel to
//!   allocate a contiguous extent up front, removing incremental `fallocate`
//!   calls and reducing metadata overhead.
//! * **Single-pass connection offset table** (`write_hnsw`): connection data
//!   byte offsets are calculated arithmetically from the known file layout
//!   (all section sizes are known before any bytes are written).  The offset
//!   table is written first, then the connection data in one forward pass —
//!   no seek-back required.
//! * **Bulk connection-pair encoding** (`write_hnsw`): all `(id, dist)` pairs
//!   for one layer are packed into a single `Vec<u8>` scratch buffer and
//!   written with one `write_all` call, reducing per-pair write overhead.
//! * **Variable-width payload offsets without seek-back** (`write_payloads`):
//!   all payloads are encoded into temporary buffers first; byte offsets are
//!   pre-computed; the offset table and payload bytes are then written in one
//!   forward pass.
//!
//! ### Read / mmap path
//! * **Bulk level and offset reads** (`read_graph`): instead of `n` individual
//!   4-byte / 8-byte `read_exact` calls, all `n` levels are read in a single
//!   `n × 4` byte read, and all `n` connection offsets in a single `n × 8`
//!   byte read.
//! * **Sequential connection data reads** (`read_graph`): connection data is
//!   stored sequentially (node 0, node 1, …, node n-1).  The per-node seek to
//!   `conn_offsets[i]` in the old code was a no-op in practice but added
//!   overhead.  It is eliminated; data is read straight through.
//! * **Bulk connection-pair reads** (`read_graph`): all `n_conns` `(id, dist)`
//!   pairs for a layer are read in one `n_conns × 8` byte `read_exact` call.
//! * **`MADV_RANDOM` on mmap** (`read_hnsw_mmap`): advises the kernel not to
//!   read-ahead pages linearly; ANN search accesses vectors in random order
//!   and read-ahead wastes I/O bandwidth and evicts useful cache lines.
//! * **Mmap reuse for payload loading** (`load_mmap_with_payload`): instead of
//!   opening a second file handle and seeking to the payload section, the
//!   already-mapped bytes are sliced directly into an [`io::Cursor`] — zero
//!   extra syscalls.
//! * **Fixed-stride payload bulk read** (`read_payloads`): all `n × stride`
//!   payload bytes are read in one `read_exact` call and decoded from the
//!   contiguous in-memory buffer, instead of `n` separate reads.

use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use crate::distance::Distance;
use crate::hnsw::{Config, GraphStore, Hnsw, OwnedGraph, PruneStrategy, VecStore};
use crate::payload::Payload;

/// Lazy, read-only view over a fixed-width payload column in an index mmap.
///
/// Values are decoded on access from their little-endian wire representation;
/// the full payload column is never copied into a `Vec<L>`.
pub struct MappedPayloads<L: Payload> {
    mmap: Arc<memmap2::Mmap>,
    data_offset: usize,
    count: usize,
    stride: usize,
    marker: PhantomData<L>,
}

impl<L: Payload> MappedPayloads<L> {
    /// Number of payload entries.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the payload column is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Decode one payload directly from the mapped file.
    pub fn get(&self, id: usize) -> io::Result<L> {
        if id >= self.count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("payload id {id} out of bounds for {} entries", self.count),
            ));
        }
        let start = self.data_offset + id * self.stride;
        let end = start + self.stride;
        let (payload, consumed) = L::decode(&self.mmap[start..end])
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if consumed != self.stride {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "fixed payload decoder consumed {consumed} bytes; expected {}",
                    self.stride
                ),
            ));
        }
        Ok(payload)
    }
}

// ─── Layout constants ─────────────────────────────────────────────────────────

pub(crate) const MAGIC:           &[u8; 8] = b"HNSWNDX\0";
pub(crate) const VERSION:         u32       = 1;
pub(crate) const COMPACT_VERSION: u32       = 2;
const FULL_EDGE_STRIDE:           usize     = 8;
const COMPACT_EDGE_STRIDE:        usize     = 4;
/// Byte offset where the vector data begins (fixed, so callers can mmap it).
pub(crate) const VECTORS_OFFSET:  usize     = 256;

// Byte positions within the fixed header.
const OFF_VERSION:  usize =  8;
const OFF_N:        usize = 12;
const OFF_DIM:      usize = 20;
const OFF_M:        usize = 28;
const OFF_M0:       usize = 36;
const OFF_EF:       usize = 44;
const OFF_EP_ID:    usize = 52;
const OFF_EP_LEVEL: usize = 60;
const OFF_FLAGS:    usize = 68;
// Byte 68: use_heuristic  69: extend_candidates  70: keep_pruned  71: prune_strategy

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
fn read_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

fn u32_le(v: u32) -> [u8; 4] { v.to_le_bytes() }
fn u64_le(v: u64) -> [u8; 8] { v.to_le_bytes() }
fn f32_le(v: f32) -> [u8; 4] { v.to_le_bytes() }

// ─── Size estimation ──────────────────────────────────────────────────────────

/// Compute the exact byte size of the HNSW section of the file (header +
/// vectors + levels + connection-offset table + connection data).
///
/// Used to pre-allocate the file on disk before the first write, which
/// allows the OS to reserve a contiguous extent and avoids incremental
/// metadata updates.
fn hnsw_section_bytes<D: Distance>(index: &Hnsw<D>, edge_stride: usize) -> u64 {
    let n   = index.vec_store.len();
    let dim = index.dim.unwrap_or(0);

    // Sum the connection-data bytes: for each node, for each layer,
    // 4 bytes (n_conns) + n_conns × 8 bytes (id + dist pairs).
    let mut conn_bytes = 0u64;
    for node in 0..index.graph.node_count() {
        for layer in 0..index.graph.level_count(node) {
            conn_bytes +=
                4 + (index.graph.neighbour_count(node, layer) as u64) * edge_stride as u64;
        }
    }

    VECTORS_OFFSET as u64          // fixed header
    + (n as u64) * (dim as u64) * 4   // vectors
    + (n as u64) * 4               // levels (u32 per node)
    + (n as u64) * 8               // connection-offset table (u64 per node)
    + conn_bytes                   // connection data
}

/// Compute the exact byte size of a fixed-stride payload section.
/// Returns `None` if the payload is variable-width.
fn fixed_payload_section_bytes<L: Payload>(n: usize) -> Option<u64> {
    L::fixed_stride().map(|stride| {
        16                          // payload header (count + stride)
        + (n as u64) * (stride as u64)
    })
}

// ─── Public save ──────────────────────────────────────────────────────────────

/// Serialize `index` to `path` with no payload section.
///
/// The file can be loaded back with [`load`] or [`load_mmap`].
pub fn save<D: Distance>(index: &Hnsw<D>, path: impl AsRef<Path>) -> io::Result<()> {
    let file = File::create(path)?;

    // Pre-allocate the exact file size so the OS can assign a contiguous
    // disk extent in one shot.  Errors are silently ignored — this is a
    // pure performance hint and does not affect correctness.
    let total = hnsw_section_bytes(index, FULL_EDGE_STRIDE)
        + 16; // empty payload header (payload_count=0, stride=0)
    let _ = file.set_len(total);

    let mut w = BufWriter::new(file);
    write_hnsw(index, &mut w)?;
    write_empty_payload(&mut w)?;
    w.flush()
}

/// Serialize `index` together with a payload slice to `path`.
///
/// `payloads.len()` must equal `index.len()`.
pub fn save_with_payload<D, L>(
    index:    &Hnsw<D>,
    payloads: &[L],
    path:     impl AsRef<Path>,
) -> io::Result<()>
where
    D: Distance,
    L: Payload,
{
    assert_eq!(
        payloads.len(), index.len(),
        "payload count ({}) must match index size ({})",
        payloads.len(), index.len()
    );
    let file = File::create(path)?;

    // Pre-allocate for fixed-stride payloads (exact size known without
    // encoding).  Variable-width payloads require encoding to know sizes;
    // we skip pre-allocation there to avoid the extra encoding pass.
    let graph_bytes = hnsw_section_bytes(index, FULL_EDGE_STRIDE);
    if let Some(payload_bytes) = fixed_payload_section_bytes::<L>(payloads.len()) {
        let _ = file.set_len(graph_bytes + payload_bytes);
    }

    let mut w = BufWriter::new(file);
    write_hnsw(index, &mut w)?;
    write_payloads(payloads, &mut w)?;
    w.flush()
}

/// Serialize a compact, read-only snapshot with ID-only adjacency records.
///
/// Compact snapshots can be opened with [`load_mmap`] and are suitable for
/// serving and archival. They intentionally cannot be opened with [`load`]
/// because the omitted build-time edge distances are required for mutation.
pub fn save_compact<D: Distance>(
    index: &Hnsw<D>,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let file = File::create(path)?;
    let total = hnsw_section_bytes(index, COMPACT_EDGE_STRIDE) + 16;
    let _ = file.set_len(total);

    let mut w = BufWriter::new(file);
    write_hnsw_format(index, &mut w, COMPACT_VERSION, COMPACT_EDGE_STRIDE)?;
    write_empty_payload(&mut w)?;
    w.flush()
}

/// Serialize a compact, read-only snapshot together with payloads.
pub fn save_compact_with_payload<D, L>(
    index: &Hnsw<D>,
    payloads: &[L],
    path: impl AsRef<Path>,
) -> io::Result<()>
where
    D: Distance,
    L: Payload,
{
    assert_eq!(
        payloads.len(),
        index.len(),
        "payload count ({}) must match index size ({})",
        payloads.len(),
        index.len()
    );
    let file = File::create(path)?;
    let graph_bytes = hnsw_section_bytes(index, COMPACT_EDGE_STRIDE);
    if let Some(payload_bytes) = fixed_payload_section_bytes::<L>(payloads.len()) {
        let _ = file.set_len(graph_bytes + payload_bytes);
    }

    let mut w = BufWriter::new(file);
    write_hnsw_format(index, &mut w, COMPACT_VERSION, COMPACT_EDGE_STRIDE)?;
    write_payloads(payloads, &mut w)?;
    w.flush()
}

// ─── Public load (owned) ─────────────────────────────────────────────────────

/// Load an index from `path`, copying vector data into an owned `Vec<f32>`.
///
/// Works for any index size.  Use [`load_mmap`] for very large indexes where
/// you want the OS page cache to manage memory.
pub fn load<D: Distance>(path: impl AsRef<Path>, metric: D) -> io::Result<Hnsw<D>> {
    let mut file = File::open(path)?;
    let (index, _) = read_hnsw_owned(&mut file, metric, false)?;
    Ok(index)
}

/// Load an index **and** its payload from `path`.
///
/// Returns `(index, payloads)`.  Fails with `InvalidData` if the file
/// contains no payload or the type does not match what was written.
pub fn load_with_payload<D, L>(
    path:   impl AsRef<Path>,
    metric: D,
) -> io::Result<(Hnsw<D>, Vec<L>)>
where
    D: Distance,
    L: Payload,
{
    let mut file = File::open(path)?;
    let (index, payload_start) = read_hnsw_owned(&mut file, metric, true)?;
    let payloads = read_payloads::<L, _>(&mut file, index.len(), payload_start)?;
    Ok((index, payloads))
}

// ─── Public load (mmap) ──────────────────────────────────────────────────────

/// Load an index from `path`, keeping vectors and graph **memory-mapped**.
///
/// Vectors, levels, connection offsets, and adjacency records remain in the
/// read-only file mapping. The OS page cache manages which pages are resident;
/// search decodes little-endian edge records directly as it traverses them.
/// No graph-sized heap allocation or adjacency-list deserialization occurs.
///
/// ## Behaviour
/// * Inserts into a mmap-backed index will panic (it is read-only).
/// * The file must not be deleted or truncated while the index is in use.
pub fn load_mmap<D: Distance>(path: impl AsRef<Path>, metric: D) -> io::Result<Hnsw<D>> {
    let file = File::open(path.as_ref())?;
    read_hnsw_mmap_inner(file, metric).map(|(idx, _, _)| idx)
}

/// Mmap-load an index **and** its payload.
///
/// Vectors and graph stay memory-mapped (no heap copy). Payload entries are
/// decoded from the existing mapping — no second `File::open` or seek.
pub fn load_mmap_with_payload<D, L>(
    path:   impl AsRef<Path>,
    metric: D,
) -> io::Result<(Hnsw<D>, Vec<L>)>
where
    D: Distance,
    L: Payload,
{
    let file = File::open(path.as_ref())?;
    let (index, payload_start, mmap) = read_hnsw_mmap_inner(file, metric)?;

    // Reuse the live mapping — no second open/seek required.
    //
    // Important: we must wrap the FULL mmap (not a slice starting at
    // payload_start) because the variable-width payload offset table stores
    // ABSOLUTE byte positions within the file.  If we sliced the mmap and
    // used a cursor over the slice, seeking to an absolute offset would land
    // at the wrong position (offset + payload_start bytes into the mmap).
    let mut cursor = io::Cursor::new(mmap.as_ref() as &[u8]);
    let payloads = read_payloads::<L, _>(&mut cursor, index.len(), payload_start)?;
    Ok((index, payloads))
}

/// Mmap-load an index and retain a fixed-width payload column in the mapping.
///
/// Unlike [`load_mmap_with_payload`], this function does not allocate or
/// decode a `Vec<L>` at open. Use [`MappedPayloads::get`] to decode individual
/// values. Variable-width payload types are rejected.
pub fn load_mmap_with_fixed_payload<D, L>(
    path: impl AsRef<Path>,
    metric: D,
) -> io::Result<(Hnsw<D>, MappedPayloads<L>)>
where
    D: Distance,
    L: Payload,
{
    let file = File::open(path.as_ref())?;
    let (index, payload_start, mmap) = read_hnsw_mmap_inner(file, metric)?;
    let payloads = map_fixed_payloads::<L>(
        Arc::clone(&mmap),
        payload_start,
        index.len(),
    )?;
    Ok((index, payloads))
}

// ─── Core write ──────────────────────────────────────────────────────────────

/// Write the Hnsw index to `w` (header + vectors + levels + conn-offsets +
/// conn-data).  Does **not** write a payload section.
///
/// The entire write is a single forward pass — no seeks or
/// `stream_position()` calls required.  All connection-data byte offsets are
/// pre-computed from the known file layout before any data is written.
pub(crate) fn write_hnsw<D: Distance, W: Write>(
    index: &Hnsw<D>,
    w:     &mut W,
) -> io::Result<()> {
    write_hnsw_format(index, w, VERSION, FULL_EDGE_STRIDE)
}

fn write_hnsw_format<D: Distance, W: Write>(
    index: &Hnsw<D>,
    w: &mut W,
    version: u32,
    edge_stride: usize,
) -> io::Result<()> {
    let n   = index.vec_store.len();
    let dim = index.dim.unwrap_or(0);
    let cfg = &index.config;

    // ── Header (256 bytes, zero-padded) ──────────────────────────────────
    let mut hdr = [0u8; VECTORS_OFFSET];
    hdr[..8].copy_from_slice(MAGIC);
    hdr[OFF_VERSION..OFF_VERSION + 4].copy_from_slice(&u32_le(version));
    hdr[OFF_N..OFF_N + 8].copy_from_slice(&u64_le(n as u64));
    hdr[OFF_DIM..OFF_DIM + 8].copy_from_slice(&u64_le(dim as u64));
    hdr[OFF_M..OFF_M + 8].copy_from_slice(&u64_le(cfg.m as u64));
    hdr[OFF_M0..OFF_M0 + 8].copy_from_slice(&u64_le(cfg.m0() as u64));
    hdr[OFF_EF..OFF_EF + 8].copy_from_slice(&u64_le(cfg.ef_construction as u64));

    let (ep_id, ep_level) = index.entry_point.unwrap_or((usize::MAX, 0));
    hdr[OFF_EP_ID..OFF_EP_ID + 8].copy_from_slice(&u64_le(ep_id as u64));
    hdr[OFF_EP_LEVEL..OFF_EP_LEVEL + 8].copy_from_slice(&u64_le(ep_level as u64));
    hdr[OFF_FLAGS]     = cfg.use_heuristic as u8;
    hdr[OFF_FLAGS + 1] = cfg.extend_candidates as u8;
    hdr[OFF_FLAGS + 2] = cfg.keep_pruned as u8;
    hdr[OFF_FLAGS + 3] = match cfg.prune_strategy {
        PruneStrategy::Simple    => 0,
        PruneStrategy::Heuristic => 1,
    };
    w.write_all(&hdr)?;

    // ── Vectors ───────────────────────────────────────────────────────────
    // Always starts at VECTORS_OFFSET (= 256) so callers can mmap the file
    // and do pointer arithmetic without reading a table.
    w.write_all(index.vec_store.as_bytes())?;

    // ── Levels ────────────────────────────────────────────────────────────
    for node in 0..index.graph.node_count() {
        let level = (index.graph.level_count(node) as u32).saturating_sub(1);
        w.write_all(&u32_le(level))?;
    }

    // ── Connection offsets (single-pass, no seek-back) ────────────────────
    //
    // All section sizes are known before writing begins, so we can compute
    // every node's absolute byte offset arithmetically:
    //
    //   conn_data_base = 256 (header)
    //                  + n*dim*4 (vectors)
    //                  + n*4 (levels)
    //                  + n*8 (this offset table)
    //
    // Then for each node i:
    //   offsets[i] = conn_data_base + sum_{j<i}( size_of_node_j_conn_data )
    //   size_of_node_j = sum_{layer} (4 + n_conns_layer * 8)
    //
    // We write the computed offset table first, then stream the connection
    // data — a single forward pass with no seek required.
    let conn_data_base: u64 = VECTORS_OFFSET as u64
        + (n as u64) * (dim as u64) * 4
        + (n as u64) * 4
        + (n as u64) * 8;

    let mut running_off = conn_data_base;
    for node in 0..index.graph.node_count() {
        w.write_all(&u64_le(running_off))?;
        for layer in 0..index.graph.level_count(node) {
            // 4 bytes for n_conns header + 8 bytes per (id, dist) pair.
            running_off +=
                4 + (index.graph.neighbour_count(node, layer) as u64) * edge_stride as u64;
        }
    }

    // ── Connection data (bulk-encoded per layer) ───────────────────────────
    //
    // All (id, dist) pairs for one layer are packed into a single byte buffer
    // and written with one write_all call — one syscall per layer instead of
    // three (n_conns header + id + dist) per pair.
    let mut conn_buf: Vec<u8> = Vec::new();
    for node in 0..index.graph.node_count() {
        for layer in 0..index.graph.level_count(node) {
            let n_conns = index.graph.neighbour_count(node, layer);
            // Reserve: 4 bytes for n_conns + 8 bytes per pair.
            conn_buf.clear();
            conn_buf.reserve(4 + n_conns * edge_stride);
            conn_buf.extend_from_slice(&u32_le(n_conns as u32));
            for (id, dist) in index.graph.neighbours(node, layer).into_iter().flatten() {
                conn_buf.extend_from_slice(&u32_le(id));
                if edge_stride == FULL_EDGE_STRIDE {
                    conn_buf.extend_from_slice(&f32_le(dist));
                }
            }
            w.write_all(&conn_buf)?;
        }
    }

    Ok(())
}

// ─── Core read (owned) ───────────────────────────────────────────────────────

/// Read header + vectors + graph into owned heap structures.
/// Returns the index and the byte position of the payload-header section
/// (so the caller can continue reading payloads if desired).
fn read_hnsw_owned<D: Distance, R: Read + Seek>(
    r:               &mut R,
    metric:          D,
    _expect_payload: bool,
) -> io::Result<(Hnsw<D>, u64)> {
    let (version, cfg, n, dim, ep, vec_offset, _file_size) = read_header(r)?;
    if version == COMPACT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "compact v2 snapshots are read-only; open with load_mmap",
        ));
    }

    // ── Vectors (owned copy) ──────────────────────────────────────────────
    r.seek(SeekFrom::Start(vec_offset as u64))?;
    let n_floats = n * dim;
    let mut raw = vec![0u8; n_floats * 4];
    r.read_exact(&mut raw)?;
    // Interpret raw bytes as f32 LE values.
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut vs = VecStore::new(dim, n);
    vs.data = floats;

    // ── Graph ─────────────────────────────────────────────────────────────
    let (connections, payload_pos) = read_graph(r, n)?;

    let index = Hnsw::from_parts(
        cfg,
        metric,
        vs,
        GraphStore::from_owned(connections),
        ep,
        if dim == 0 { None } else { Some(dim) },
    );
    Ok((index, payload_pos))
}

// ─── Core read (mmap) ────────────────────────────────────────────────────────

/// Open `file` and keep its vectors and graph in one read-only mapping.
///
/// Returns `(index, payload_section_start, arc_mmap)`.  The `Arc<Mmap>` is
/// provided so the caller can read the payload section from the existing
/// mapping without opening a second file handle.
fn read_hnsw_mmap_inner<D: Distance>(
    file:   File,
    metric: D,
) -> io::Result<(Hnsw<D>, u64, Arc<memmap2::Mmap>)> {
    // SAFETY: We open the file read-only and never write through the mapping.
    // The `Arc<Mmap>` stored inside `VecStore::MmapBacking` keeps the mapping
    // alive for the full lifetime of the index.
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file)? });

    // Advise the OS that we will access pages in random order (ANN search
    // pattern), so it should not waste I/O bandwidth on linear read-ahead.
    // Failure is silently ignored — this is a performance hint only.
    #[cfg(unix)]
    let _ = mmap.advise(memmap2::Advice::Random);

    let mut cursor = io::Cursor::new(mmap.as_ref() as &[u8]);
    let (version, cfg, n, dim, ep, vec_offset, _file_size) =
        read_header(&mut cursor)?;
    let edge_stride = edge_stride_for_version(version)?;

    // Bounds check: vector section must fit inside the mapping.
    let vec_bytes = n * dim * 4;
    if vec_offset + vec_bytes > mmap.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file too short: vector section extends past end of file",
        ));
    }

    let vs = VecStore::from_mmap(Arc::clone(&mmap), vec_offset, n, dim);

    // Levels, offsets, and adjacency records remain in the mapping. Validate
    // their bounds once at open, then let search decode edge pairs lazily.
    let levels_offset = vec_offset + vec_bytes;
    let payload_pos = validate_mapped_graph(&mmap, levels_offset, n, edge_stride)?;
    let graph =
        GraphStore::from_mmap(Arc::clone(&mmap), levels_offset, n, edge_stride);

    let index = Hnsw::from_parts(
        cfg,
        metric,
        vs,
        graph,
        ep,
        if dim == 0 { None } else { Some(dim) },
    );
    Ok((index, payload_pos, mmap))
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

/// Parse the 256-byte fixed header.  Returns
/// `(version, config, n, dim, entry_point, vec_section_byte_offset,
/// file_size_hint)`.
fn read_header<R: Read + Seek>(
    r: &mut R,
) -> io::Result<(u32, Config, usize, usize, Option<(usize, usize)>, usize, u64)> {
    let mut hdr = [0u8; VECTORS_OFFSET];
    r.read_exact(&mut hdr)?;

    if &hdr[..8] != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid magic bytes — expected {:?}", MAGIC),
        ));
    }
    let version = read_u32(&hdr, OFF_VERSION);
    if version != VERSION && version != COMPACT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported file version {version} (expected {VERSION} or {COMPACT_VERSION})"
            ),
        ));
    }

    let n        = read_u64(&hdr, OFF_N)  as usize;
    let dim      = read_u64(&hdr, OFF_DIM) as usize;
    let m        = read_u64(&hdr, OFF_M)  as usize;
    let m0       = read_u64(&hdr, OFF_M0) as usize;
    let ef       = read_u64(&hdr, OFF_EF) as usize;
    let ep_id    = read_u64(&hdr, OFF_EP_ID) as usize;
    let ep_level = read_u64(&hdr, OFF_EP_LEVEL) as usize;

    let use_heuristic      = hdr[OFF_FLAGS] != 0;
    let extend_candidates  = hdr[OFF_FLAGS + 1] != 0;
    let keep_pruned        = hdr[OFF_FLAGS + 2] != 0;
    let prune_strategy     = match hdr[OFF_FLAGS + 3] {
        0 => PruneStrategy::Simple,
        1 => PruneStrategy::Heuristic,
        b => return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown prune_strategy byte {b}"),
        )),
    };

    let entry_point = if ep_id == usize::MAX {
        None
    } else {
        Some((ep_id, ep_level))
    };

    let config = Config {
        m,
        m0: Some(m0),
        ef_construction: ef,
        use_heuristic,
        extend_candidates,
        keep_pruned,
        prune_strategy,
        capacity: n, // use saved n as pre-alloc hint
    };

    // Estimate file size from the current position
    let pos = r.stream_position()?;
    let end = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(pos))?;

    Ok((
        version,
        config,
        n,
        dim,
        entry_point,
        VECTORS_OFFSET,
        end,
    ))
}

fn edge_stride_for_version(version: u32) -> io::Result<usize> {
    match version {
        VERSION => Ok(FULL_EDGE_STRIDE),
        COMPACT_VERSION => Ok(COMPACT_EDGE_STRIDE),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported file version {version}"),
        )),
    }
}

/// Read the levels + conn-offsets + conn-data sections from the current
/// position of `r`.  Returns `(connections, payload_section_start_pos)`.
///
/// ## I/O strategy
/// All `n` levels are read in a single `n × 4` byte read; all `n` connection
/// offsets in a single `n × 8` byte read.  Connection data is stored
/// sequentially (node 0, node 1, …), so we read it in one forward pass
/// without seeking to each node's recorded offset (the offsets table is
/// read but used only as a bounds / random-access aid; full-load is
/// sequential).  Each layer's `(id, dist)` pairs are read in a single
/// `n_conns × 8` byte read.
fn read_graph<R: Read + Seek>(
    r: &mut R,
    n: usize,
) -> io::Result<(OwnedGraph, u64)> {
    // ── Levels: one bulk read of n × 4 bytes ─────────────────────────────
    let mut raw_levels = vec![0u8; n * 4];
    r.read_exact(&mut raw_levels)?;
    let levels: Vec<u32> = raw_levels
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // ── Connection offsets: one bulk read of n × 8 bytes ─────────────────
    // We read the offset table to advance past it; for a full load the
    // connection data is then read sequentially (no per-node seek needed).
    let mut raw_offsets = vec![0u8; n * 8];
    r.read_exact(&mut raw_offsets)?;
    // (offsets decoded only if needed for future partial-load features;
    //  here we read sequentially so the values are not used.)
    let _ = raw_offsets; // explicitly acknowledge we've consumed the bytes

    // ── Connection data: sequential forward read ───────────────────────────
    //
    // The writer stores connection data for node 0, node 1, …, node n-1 in
    // order, so we read straight through without seeking.  Each layer's
    // (id, dist) pairs are bulk-read into a single byte buffer and decoded
    // in one pass.
    let mut connections: OwnedGraph = Vec::with_capacity(n);
    let mut pair_buf: Vec<u8> = Vec::new();
    let mut buf4 = [0u8; 4];

    for &level in &levels {
        let n_layers = level as usize + 1;
        let mut node_conn: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            // Read n_conns (4 bytes).
            r.read_exact(&mut buf4)?;
            let n_conns = u32::from_le_bytes(buf4) as usize;

            // Bulk-read all (id, dist) pairs for this layer in one call.
            let byte_count = n_conns * 8;
            pair_buf.resize(byte_count, 0);
            if byte_count > 0 {
                r.read_exact(&mut pair_buf[..byte_count])?;
            }

            let layer_conn: Vec<(u32, f32)> = pair_buf[..byte_count]
                .chunks_exact(8)
                .map(|c| {
                    let id   = u32::from_le_bytes(c[0..4].try_into().unwrap());
                    let dist = f32::from_le_bytes(c[4..8].try_into().unwrap());
                    (id, dist)
                })
                .collect();

            node_conn.push(layer_conn);
        }
        connections.push(node_conn);
    }

    // The payload header begins right after the connection data.
    let payload_pos = r.stream_position()?;
    Ok((connections, payload_pos))
}

/// Validate the memory-mapped graph tables and records without materializing
/// adjacency lists. Returns the byte offset of the payload section.
fn validate_mapped_graph(
    mmap: &[u8],
    levels_offset: usize,
    n: usize,
    edge_stride: usize,
) -> io::Result<u64> {
    let levels_bytes = n.checked_mul(4).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "levels table size overflow")
    })?;
    let offsets_bytes = n.checked_mul(8).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "offset table size overflow")
    })?;
    let offsets_offset = levels_offset.checked_add(levels_bytes).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "offset table position overflow")
    })?;
    let data_offset = offsets_offset.checked_add(offsets_bytes).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "graph data position overflow")
    })?;
    if data_offset > mmap.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file too short: graph tables extend past end of file",
        ));
    }

    let mut position = data_offset;
    for node in 0..n {
        let level_pos = levels_offset + node * 4;
        let level = u32::from_le_bytes(
            mmap[level_pos..level_pos + 4].try_into().unwrap(),
        ) as usize;

        let offset_pos = offsets_offset + node * 8;
        let recorded = u64::from_le_bytes(
            mmap[offset_pos..offset_pos + 8].try_into().unwrap(),
        );
        if recorded != position as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid connection offset for node {node}: {recorded} != {position}"
                ),
            ));
        }

        for _ in 0..=level {
            let count_end = position.checked_add(4).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "connection position overflow")
            })?;
            let count_bytes = mmap.get(position..count_end).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "file too short: missing connection count",
                )
            })?;
            let count = u32::from_le_bytes(count_bytes.try_into().unwrap()) as usize;
            let edge_bytes = count.checked_mul(edge_stride).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "connection list size overflow")
            })?;
            let edge_end = count_end.checked_add(edge_bytes).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "connection position overflow")
            })?;
            let edges = mmap.get(count_end..edge_end).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "file too short: connection list extends past end of file",
                )
            })?;
            for edge in edges.chunks_exact(edge_stride) {
                let neighbour = u32::from_le_bytes(edge[..4].try_into().unwrap()) as usize;
                if neighbour >= n {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "invalid neighbour id {neighbour} for index with {n} nodes"
                        ),
                    ));
                }
            }
            position = edge_end;
        }
    }

    Ok(position as u64)
}

// ─── Payload I/O ─────────────────────────────────────────────────────────────

fn map_fixed_payloads<L: Payload>(
    mmap: Arc<memmap2::Mmap>,
    payload_section_pos: u64,
    expected_count: usize,
) -> io::Result<MappedPayloads<L>> {
    let payload_section_pos = usize::try_from(payload_section_pos).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "payload section position does not fit in memory",
        )
    })?;
    let header_end = payload_section_pos.checked_add(16).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "payload header position overflow")
    })?;
    let header = mmap.get(payload_section_pos..header_end).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "file too short: missing payload header",
        )
    })?;
    let count = usize::try_from(read_u64(header, 0)).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "payload count does not fit in memory",
        )
    })?;
    let stride = usize::try_from(read_u64(header, 8)).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "payload stride does not fit in memory",
        )
    })?;

    if count != expected_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("payload count {count} != index size {expected_count}"),
        ));
    }
    let expected_stride = L::fixed_stride().filter(|stride| *stride > 0).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "mapped payload access requires a non-zero fixed-width Payload",
        )
    })?;
    if stride != expected_stride {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("payload stride {stride} != type stride {expected_stride}"),
        ));
    }

    let payload_bytes = count.checked_mul(stride).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "payload column size overflow")
    })?;
    let data_end = header_end.checked_add(payload_bytes).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "payload column position overflow")
    })?;
    if data_end > mmap.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file too short: fixed payload column extends past end of file",
        ));
    }

    Ok(MappedPayloads {
        mmap,
        data_offset: header_end,
        count,
        stride,
        marker: PhantomData,
    })
}

/// Write a zero-entry payload section (marker for "no payload").
pub(crate) fn write_empty_payload<W: Write>(w: &mut W) -> io::Result<()> {
    // payload_count = 0, payload_stride = 0
    w.write_all(&u64_le(0))?;
    w.write_all(&u64_le(0))?;
    Ok(())
}

/// Serialize `payloads` and write the payload section.
///
/// ## Fixed-stride payloads
/// Written sequentially — no offset table, no seek.
///
/// ## Variable-width payloads
/// All payloads are encoded into temporary buffers first so their sizes are
/// known.  Byte offsets are then pre-computed and written in a single forward
/// pass — no seek-back to patch a placeholder table.
pub(crate) fn write_payloads<L: Payload, W: Write + Seek>(
    payloads: &[L],
    w:        &mut W,
) -> io::Result<()> {
    let n      = payloads.len();
    let stride = L::fixed_stride().unwrap_or(0) as u64;

    // Payload header (16 bytes).
    w.write_all(&u64_le(n as u64))?;
    w.write_all(&u64_le(stride))?;

    if stride > 0 {
        // ── Fixed-stride: sequential write, no offset table ───────────────
        let mut buf = Vec::with_capacity(stride as usize);
        for p in payloads {
            buf.clear();
            p.encode(&mut buf);
            debug_assert_eq!(buf.len(), stride as usize);
            w.write_all(&buf)?;
        }
    } else {
        // ── Variable-width: encode all first, then write in one pass ───────
        //
        // Encoding upfront lets us compute every absolute byte offset before
        // writing the offset table — no seek-back required.
        let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(n);
        let mut buf = Vec::new();
        for p in payloads {
            buf.clear();
            p.encode(&mut buf);
            encoded.push(buf.clone());
        }

        // Current position (= start of offset table, after the 16-byte header
        // already written above).
        let offsets_table_start = w.stream_position()?;
        // Data begins right after the n × 8 byte offset table.
        let data_start: u64 = offsets_table_start + (n as u64) * 8;

        // Compute absolute offsets for each entry.
        let mut offsets: Vec<u64> = Vec::with_capacity(n);
        let mut cur = data_start;
        for enc in &encoded {
            offsets.push(cur);
            cur += enc.len() as u64;
        }

        // Write offset table.
        for &off in &offsets {
            w.write_all(&u64_le(off))?;
        }
        // Write payload data.
        for enc in &encoded {
            w.write_all(enc)?;
        }
    }

    Ok(())
}

/// Deserialize `n` payload entries starting at `payload_section_pos`.
pub(crate) fn read_payloads<L: Payload, R: Read + Seek>(
    r:                   &mut R,
    n:                   usize,
    payload_section_pos: u64,
) -> io::Result<Vec<L>> {
    r.seek(SeekFrom::Start(payload_section_pos))?;

    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8)?;
    let payload_count = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8)?;
    let stride = u64::from_le_bytes(buf8) as usize;

    if payload_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file contains no payload section",
        ));
    }
    if payload_count != n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("payload count {payload_count} != index size {n}"),
        ));
    }

    let mut payloads = Vec::with_capacity(n);

    if stride > 0 {
        // ── Fixed-stride: one bulk read of all n × stride bytes ───────────
        //
        // A single read_exact call pulls all payload bytes into a contiguous
        // buffer; we then decode each stride-sized chunk in sequence.
        // This reduces the number of read_exact calls from n to 1.
        let total_bytes = n * stride;
        let mut raw = vec![0u8; total_bytes];
        r.read_exact(&mut raw)?;
        for chunk in raw.chunks_exact(stride) {
            let (p, _) = L::decode(chunk).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, e.to_string())
            })?;
            payloads.push(p);
        }
    } else {
        // ── Variable-width: read offset table, then seek to each entry ────
        let mut offsets = vec![0u64; n];
        for off in &mut offsets {
            r.read_exact(&mut buf8)?;
            *off = u64::from_le_bytes(buf8);
        }
        let mut buf = Vec::new();
        for i in 0..n {
            r.seek(SeekFrom::Start(offsets[i]))?;
            let end = if i + 1 < n {
                offsets[i + 1]
            } else {
                r.seek(SeekFrom::End(0))?
            };
            let byte_len = (end - offsets[i]) as usize;
            buf.resize(byte_len, 0);
            r.seek(SeekFrom::Start(offsets[i]))?;
            r.read_exact(&mut buf)?;
            let (p, _) = L::decode(&buf).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, e.to_string())
            })?;
            payloads.push(p);
        }
    }

    Ok(payloads)
}
