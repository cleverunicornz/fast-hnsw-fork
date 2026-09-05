//! Real-embedding recall and lifecycle benchmark for `.fvecs` corpora.
//!
//! Example:
//!   cargo bench --bench fvecs -- \
//!     --fvecs /path/to/minilm.fvecs --rows 10000 --queries 100 \
//!     --snapshot /tmp/minilm.hnsw

use fast_hnsw::distance::{Distance, DotProduct};
use fast_hnsw::{Builder, Hnsw, persist};
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const K: usize = 10;
const GRAPH_COUNT: usize = 4;

struct Args {
    fvecs: PathBuf,
    rows: usize,
    queries: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    seed: u64,
    snapshot: Option<PathBuf>,
}

fn main() -> io::Result<()> {
    let args = parse_args()?;
    let mut vectors = load_fvecs(&args.fvecs, args.rows)?;
    if vectors.len() < K {
        return Err(invalid_input(format!(
            "fvecs corpus must contain at least {K} rows"
        )));
    }
    for vector in &mut vectors {
        normalize(vector)?;
    }
    let rows = vectors.len();
    let dimensions = vectors[0].len();
    let queries = build_queries(&vectors, args.queries);
    let exact = exact_results(&vectors, &queries);
    let exact_filtered = exact_filtered_results(&vectors, &queries);

    let mut index: Hnsw<DotProduct> = Builder::new()
        .m(args.m)
        .ef_construction(args.ef_construction)
        .capacity(rows)
        .seed(args.seed)
        .build(DotProduct).unwrap();
    let started = Instant::now();
    for vector in &vectors {
        index.insert(vector.clone()).unwrap();
    }
    let build = started.elapsed();

    for query in queries.iter().take(3) {
        let _ = index.search(query, K, args.ef_search).unwrap();
    }
    let mut search_times = Vec::with_capacity(queries.len());
    let mut approximate = Vec::with_capacity(queries.len());
    let mut filtered_search_times = Vec::with_capacity(queries.len());
    let mut approximate_filtered = Vec::with_capacity(queries.len());
    for query in &queries {
        let started = Instant::now();
        let result = index.search(query, K, args.ef_search).unwrap();
        search_times.push(started.elapsed());
        approximate.push(result.into_iter().map(|item| item.id).collect::<Vec<_>>());

        let started = Instant::now();
        let result =
            index.search_filtered(query, K, args.ef_search, |id| id % GRAPH_COUNT == 0).unwrap();
        filtered_search_times.push(started.elapsed());
        approximate_filtered.push(
            result
                .into_iter()
                .map(|item| item.id)
                .collect::<Vec<_>>(),
        );
    }

    let temporary_snapshot = args.snapshot.is_none();
    let snapshot = args.snapshot.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!(
            "fast-hnsw-fvecs-{}-{}.hnsw",
            std::process::id(),
            args.seed
        ))
    });
    let started = Instant::now();
    persist::save(&index, &snapshot)?;
    let save = started.elapsed();
    let disk_bytes = snapshot.metadata()?.len();
    drop(index);
    let started = Instant::now();
    let reopened = persist::load_mmap(&snapshot, DotProduct)?;
    let mmap_open = started.elapsed();
    let started = Instant::now();
    let reopened_result = reopened.search(&queries[0], K, args.ef_search).unwrap();
    let first_query = started.elapsed();
    if reopened_result.is_empty() {
        return Err(io::Error::other("reopened index returned no results"));
    }
    if temporary_snapshot {
        std::fs::remove_file(&snapshot)?;
    }

    println!("{{");
    println!("  \"corpus\": {:?},", args.fvecs.display().to_string());
    println!("  \"rows\": {rows},");
    println!("  \"dimensions\": {dimensions},");
    println!("  \"queries\": {},", queries.len());
    println!("  \"k\": {K},");
    println!("  \"m\": {},", args.m);
    println!("  \"ef_construction\": {},", args.ef_construction);
    println!("  \"ef_search\": {},", args.ef_search);
    println!("  \"build_ms\": {:.6},", milliseconds(build));
    println!(
        "  \"query_p50_us\": {:.6},",
        microseconds(percentile(&search_times, 0.50))
    );
    println!(
        "  \"query_p99_us\": {:.6},",
        microseconds(percentile(&search_times, 0.99))
    );
    println!(
        "  \"recall_at_10\": {:.6},",
        mean_recall(&exact, &approximate)
    );
    println!(
        "  \"filtered_query_p50_us\": {:.6},",
        microseconds(percentile(&filtered_search_times, 0.50))
    );
    println!(
        "  \"filtered_query_p99_us\": {:.6},",
        microseconds(percentile(&filtered_search_times, 0.99))
    );
    println!(
        "  \"filtered_recall_at_10\": {:.6},",
        mean_recall(&exact_filtered, &approximate_filtered)
    );
    println!("  \"save_ms\": {:.6},", milliseconds(save));
    println!("  \"mmap_open_ms\": {:.6},", milliseconds(mmap_open));
    println!(
        "  \"first_query_us\": {:.6},",
        microseconds(first_query)
    );
    if temporary_snapshot {
        println!("  \"disk_bytes\": {disk_bytes}");
    } else {
        println!("  \"disk_bytes\": {disk_bytes},");
        println!(
            "  \"snapshot\": {:?}",
            snapshot.display().to_string()
        );
    }
    println!("}}");
    Ok(())
}

fn load_fvecs(path: &Path, limit: usize) -> io::Result<Vec<Vec<f32>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut vectors = Vec::new();
    let mut dimensions = None;
    while vectors.len() < limit {
        let mut dimension_bytes = [0; 4];
        match reader.read_exact(&mut dimension_bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }
        let row_dimensions = i32::from_le_bytes(dimension_bytes);
        if row_dimensions <= 0 {
            return Err(invalid_input("invalid fvecs dimension"));
        }
        let row_dimensions = row_dimensions as usize;
        if dimensions.get_or_insert(row_dimensions) != &row_dimensions {
            return Err(invalid_input("mixed fvecs dimensions"));
        }
        let mut vector = Vec::with_capacity(row_dimensions);
        for _ in 0..row_dimensions {
            let mut bytes = [0; 4];
            reader.read_exact(&mut bytes)?;
            let value = f32::from_le_bytes(bytes);
            if !value.is_finite() {
                return Err(invalid_input("non-finite fvecs value"));
            }
            vector.push(value);
        }
        vectors.push(vector);
    }
    if vectors.is_empty() {
        return Err(invalid_input("fvecs corpus is empty"));
    }
    Ok(vectors)
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

fn build_queries(vectors: &[Vec<f32>], queries: usize) -> Vec<Vec<f32>> {
    (0..queries)
        .map(|query_index| {
            let target = query_index.wrapping_mul(7_919) % vectors.len();
            let noise = (target + vectors.len() / 2 + query_index + 1) % vectors.len();
            let mut query = vectors[target]
                .iter()
                .zip(&vectors[noise])
                .map(|(value, noise)| value * 0.85 + noise * 0.15)
                .collect::<Vec<_>>();
            normalize(&mut query).expect("finite normalized source vectors produce a valid query");
            query
        })
        .collect()
}

fn exact_results(vectors: &[Vec<f32>], queries: &[Vec<f32>]) -> Vec<Vec<usize>> {
    exact_results_where(vectors, queries, |_| true)
}

fn exact_filtered_results(vectors: &[Vec<f32>], queries: &[Vec<f32>]) -> Vec<Vec<usize>> {
    exact_results_where(vectors, queries, |id| id % GRAPH_COUNT == 0)
}

fn exact_results_where<F>(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    accepts: F,
) -> Vec<Vec<usize>>
where
    F: Fn(usize) -> bool,
{
    let metric = DotProduct;
    queries
        .iter()
        .map(|query| {
            let mut distances = vectors
                .iter()
                .enumerate()
                .filter(|(id, _)| accepts(*id))
                .map(|(id, vector)| (metric.distance(query, vector), id))
                .collect::<Vec<_>>();
            if distances.len() > K {
                distances.select_nth_unstable_by(K, |left, right| {
                    left.0
                        .total_cmp(&right.0)
                        .then_with(|| left.1.cmp(&right.1))
                });
                distances.truncate(K);
            }
            distances.sort_unstable_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
            distances.iter().map(|(_, id)| *id).collect()
        })
        .collect()
}

fn mean_recall(expected: &[Vec<usize>], actual: &[Vec<usize>]) -> f64 {
    expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| {
            let expected = expected.iter().copied().collect::<HashSet<_>>();
            actual
                .iter()
                .take(K)
                .filter(|id| expected.contains(id))
                .count() as f64
                / expected.len() as f64
        })
        .sum::<f64>()
        / expected.len() as f64
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
    Ok(Args {
        fvecs: PathBuf::from(required_arg(&args, "--fvecs")?),
        rows: parse_arg(&args, "--rows")?.unwrap_or(usize::MAX),
        queries: parse_arg(&args, "--queries")?.unwrap_or(100),
        m: parse_arg(&args, "--m")?.unwrap_or(32),
        ef_construction: parse_arg(&args, "--ef-construction")?.unwrap_or(256),
        ef_search: parse_arg(&args, "--ef-search")?.unwrap_or(512),
        seed: parse_arg(&args, "--seed")?.unwrap_or(0x4e4f_5641),
        snapshot: optional_arg(&args, "--snapshot")?.map(PathBuf::from),
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

fn optional_arg<'a>(args: &'a [String], flag: &str) -> io::Result<Option<&'a str>> {
    let Some(position) = args.iter().position(|argument| argument == flag) else {
        return Ok(None);
    };
    args.get(position + 1)
        .map(String::as_str)
        .map(Some)
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
