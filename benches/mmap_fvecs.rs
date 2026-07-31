//! Process-isolated mmap lifecycle benchmark.
//!
//! Build a retained snapshot with `fvecs`, then run this benchmark under the
//! platform RSS tool (`/usr/bin/time -l` on macOS, `/usr/bin/time -v` on
//! Linux). Only a bounded query sample is loaded into heap memory.

use fast_hnsw::distance::DotProduct;
use fast_hnsw::persist;
use fast_hnsw::SearchWorkspace;
use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const K: usize = 10;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout);
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct Args {
    index: PathBuf,
    fvecs: PathBuf,
    queries: usize,
    ef_search: usize,
    filter_modulo: Option<usize>,
}

fn main() -> io::Result<()> {
    let args = parse_args()?;

    let started = Instant::now();
    let index = persist::load_mmap(&args.index, DotProduct)?;
    let mmap_open = started.elapsed();
    let queries = load_queries(&args.fvecs, index.len(), args.queries)?;

    let started = Instant::now();
    let first = search(&index, &queries[0], args.ef_search, args.filter_modulo);
    let first_query = started.elapsed();
    if first.is_empty() {
        return Err(io::Error::other("mmap index returned no results"));
    }

    let mut timings = Vec::with_capacity(queries.len());
    reset_allocations();
    for query in &queries {
        let started = Instant::now();
        let result = search(&index, query, args.ef_search, args.filter_modulo);
        timings.push(started.elapsed());
        if result.is_empty() {
            return Err(io::Error::other("mmap index returned no results"));
        }
    }
    let allocated_search = allocation_snapshot();

    let mut workspace = SearchWorkspace::new(index.len(), args.ef_search);
    let _ = search_with_workspace(
        &index,
        &queries[0],
        args.ef_search,
        args.filter_modulo,
        &mut workspace,
    );
    let mut reused_timings = Vec::with_capacity(queries.len());
    reset_allocations();
    for query in &queries {
        let started = Instant::now();
        let result = search_with_workspace(
            &index,
            query,
            args.ef_search,
            args.filter_modulo,
            &mut workspace,
        );
        reused_timings.push(started.elapsed());
        if result.is_empty() {
            return Err(io::Error::other("mmap index returned no results"));
        }
    }
    let allocated_reused_search = allocation_snapshot();
    let directed_edges = index.stats().layer_edges.into_iter().sum::<usize>();
    let snapshot_bytes = args.index.metadata()?.len();
    let compact_id_only_bytes =
        snapshot_bytes.saturating_sub((directed_edges as u64).saturating_mul(4));

    println!("{{");
    println!("  \"index\": {:?},", args.index.display().to_string());
    println!("  \"rows\": {},", index.len());
    println!("  \"dimensions\": {},", index.dim().unwrap_or(0));
    println!("  \"queries\": {},", queries.len());
    println!("  \"ef_search\": {},", args.ef_search);
    match args.filter_modulo {
        Some(modulo) => println!("  \"filter_modulo\": {modulo},"),
        None => println!("  \"filter_modulo\": null,"),
    }
    println!("  \"mmap_open_ms\": {:.6},", milliseconds(mmap_open));
    println!(
        "  \"first_query_us\": {:.6},",
        microseconds(first_query)
    );
    println!(
        "  \"query_p50_us\": {:.6},",
        microseconds(percentile(&timings, 0.50))
    );
    println!(
        "  \"query_p99_us\": {:.6},",
        microseconds(percentile(&timings, 0.99))
    );
    println!(
        "  \"reused_query_p50_us\": {:.6},",
        microseconds(percentile(&reused_timings, 0.50))
    );
    println!(
        "  \"reused_query_p99_us\": {:.6},",
        microseconds(percentile(&reused_timings, 0.99))
    );
    println!(
        "  \"query_allocations\": {},",
        allocated_search.allocations
    );
    println!(
        "  \"query_allocated_bytes\": {},",
        allocated_search.bytes
    );
    println!(
        "  \"reused_query_allocations\": {},",
        allocated_reused_search.allocations
    );
    println!(
        "  \"reused_query_allocated_bytes\": {},",
        allocated_reused_search.bytes
    );
    println!("  \"directed_edges\": {directed_edges},");
    println!("  \"snapshot_bytes\": {snapshot_bytes},");
    println!("  \"estimated_id_only_edge_snapshot_bytes\": {compact_id_only_bytes}");
    println!("}}");
    Ok(())
}

fn search(
    index: &fast_hnsw::Hnsw<DotProduct>,
    query: &[f32],
    ef_search: usize,
    filter_modulo: Option<usize>,
) -> Vec<fast_hnsw::SearchResult> {
    match filter_modulo {
        Some(modulo) => index.search_filtered(query, K, ef_search, |id| id % modulo == 0),
        None => index.search(query, K, ef_search),
    }
}

fn search_with_workspace(
    index: &fast_hnsw::Hnsw<DotProduct>,
    query: &[f32],
    ef_search: usize,
    filter_modulo: Option<usize>,
    workspace: &mut SearchWorkspace,
) -> Vec<fast_hnsw::SearchResult> {
    match filter_modulo {
        Some(modulo) => index.search_filtered_with_workspace(
            query,
            K,
            ef_search,
            |id| id % modulo == 0,
            workspace,
        ),
        None => index.search_with_workspace(query, K, ef_search, workspace),
    }
}

struct AllocationSnapshot {
    allocations: usize,
    bytes: usize,
}

fn reset_allocations() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn allocation_snapshot() -> AllocationSnapshot {
    AllocationSnapshot {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

fn load_queries(path: &Path, rows: usize, count: usize) -> io::Result<Vec<Vec<f32>>> {
    if count == 0 {
        return Err(invalid_input("--queries must be greater than zero"));
    }
    if rows < 2 {
        return Err(invalid_input("index must contain at least two rows"));
    }

    let mut reader = BufReader::new(File::open(path)?);
    let dimensions = read_dimensions(&mut reader)?;
    let row_bytes = (dimensions + 1)
        .checked_mul(4)
        .ok_or_else(|| invalid_input("fvecs row size overflow"))?;
    let mut queries = Vec::with_capacity(count);
    for query_index in 0..count {
        let target = query_index.wrapping_mul(7_919) % rows;
        let noise = (target + rows / 2 + query_index + 1) % rows;
        let target_vector = read_row(&mut reader, target, dimensions, row_bytes)?;
        let noise_vector = read_row(&mut reader, noise, dimensions, row_bytes)?;
        let mut query = target_vector
            .iter()
            .zip(noise_vector)
            .map(|(value, noise)| value * 0.85 + noise * 0.15)
            .collect::<Vec<_>>();
        normalize(&mut query)?;
        queries.push(query);
    }
    Ok(queries)
}

fn read_dimensions(reader: &mut BufReader<File>) -> io::Result<usize> {
    reader.seek(SeekFrom::Start(0))?;
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    let dimensions = i32::from_le_bytes(bytes);
    if dimensions <= 0 {
        return Err(invalid_input("invalid fvecs dimension"));
    }
    Ok(dimensions as usize)
}

fn read_row(
    reader: &mut BufReader<File>,
    row: usize,
    dimensions: usize,
    row_bytes: usize,
) -> io::Result<Vec<f32>> {
    let offset = row
        .checked_mul(row_bytes)
        .ok_or_else(|| invalid_input("fvecs row offset overflow"))?;
    reader.seek(SeekFrom::Start(offset as u64))?;
    let mut dimension_bytes = [0; 4];
    reader.read_exact(&mut dimension_bytes)?;
    if i32::from_le_bytes(dimension_bytes) != dimensions as i32 {
        return Err(invalid_input("mixed or truncated fvecs dimensions"));
    }

    let mut vector = Vec::with_capacity(dimensions);
    for _ in 0..dimensions {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        vector.push(f32::from_le_bytes(bytes));
    }
    normalize(&mut vector)?;
    Ok(vector)
}

fn normalize(vector: &mut [f32]) -> io::Result<()> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(invalid_input("cannot normalize zero or non-finite vector"));
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

fn percentile(samples: &[Duration], fraction: f64) -> Duration {
    let mut samples = samples.to_vec();
    samples.sort_unstable();
    let index = ((samples.len() - 1) as f64 * fraction).ceil() as usize;
    samples[index]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn microseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

fn parse_args() -> io::Result<Args> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let filter_modulo = parse_arg(&args, "--filter-modulo")?;
    if filter_modulo == Some(0) {
        return Err(invalid_input("--filter-modulo must be greater than zero"));
    }
    Ok(Args {
        index: PathBuf::from(required_arg(&args, "--index")?),
        fvecs: PathBuf::from(required_arg(&args, "--fvecs")?),
        queries: parse_arg(&args, "--queries")?.unwrap_or(100),
        ef_search: parse_arg(&args, "--ef-search")?.unwrap_or(512),
        filter_modulo,
    })
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> io::Result<Option<T>> {
    let Some(position) = args.iter().position(|argument| argument == flag) else {
        return Ok(None);
    };
    let value = args
        .get(position + 1)
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")))?;
    value
        .parse()
        .map(Some)
        .map_err(|_| invalid_input(format!("invalid {flag} value")))
}

fn required_arg<'a>(args: &'a [String], flag: &str) -> io::Result<&'a str> {
    let position = args
        .iter()
        .position(|argument| argument == flag)
        .ok_or_else(|| invalid_input(format!("missing {flag}")))?;
    args.get(position + 1)
        .map(String::as_str)
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
