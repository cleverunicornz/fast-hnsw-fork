use fast_hnsw::distance::Cosine;
use fast_hnsw::{Builder, SearchResult};
use fast_hnsw_quantized::{
    MappedQuantizedVectors, QuantizedConfig, QuantizedHnsw, QuantizedMetric,
};
use std::collections::HashSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows = setting("FHQ_BENCH_ROWS", 5_000).max(1);
    let dimensions = setting("FHQ_BENCH_DIMENSIONS", 384).max(1);
    let queries = setting("FHQ_BENCH_QUERIES", 50).max(1).min(rows);
    let k = setting("FHQ_BENCH_K", 10).max(1).min(rows);
    let ef = setting("FHQ_BENCH_EF", 256).max(k);
    let bits = setting("FHQ_BENCH_BITS", 4) as u8;

    let vectors = corpus(rows, dimensions);
    let mut index = Builder::new()
        .m(16)
        .ef_construction(200)
        .capacity(rows)
        .seed(42)
        .build(Cosine);
    let build_started = Instant::now();
    for vector in &vectors {
        index.insert(vector.clone());
    }
    let build_time = build_started.elapsed();

    let file = tempfile::NamedTempFile::new()?;
    MappedQuantizedVectors::save(
        file.path(),
        dimensions,
        QuantizedMetric::Cosine,
        QuantizedConfig::with_bits(bits)?,
        &vectors,
    )?;
    // SAFETY: the benchmark leaves the completed temporary file immutable.
    let mapped = unsafe { MappedQuantizedVectors::open_mmap_verified(file.path())? };
    let quantized = QuantizedHnsw::new(&index, &mapped)?;
    let query_rows = (0..queries)
        .map(|query| query * rows / queries)
        .collect::<Vec<_>>();
    let ground_truth = query_rows
        .iter()
        .map(|row| exact_top_k(&vectors, &vectors[*row], k))
        .collect::<Vec<_>>();

    for row in &query_rows {
        black_box(index.search(&vectors[*row], k, ef));
        black_box(quantized.search(&vectors[*row], k, ef)?);
    }
    let (exact_time, exact_hits) = measure(&query_rows, |row| {
        index.search(&vectors[row], k, ef)
    });
    let (quantized_time, quantized_hits) = measure(&query_rows, |row| {
        quantized
            .search(&vectors[row], k, ef)
            .expect("validated benchmark query")
    });

    println!("rows={rows} dimensions={dimensions} queries={queries} k={k} ef={ef} bits={bits}");
    println!("build_ms={:.3}", milliseconds(build_time));
    println!(
        "exact_hnsw_qps={:.1} recall_at_k={:.4}",
        queries as f64 / exact_time.as_secs_f64(),
        recall(&exact_hits, &ground_truth, k)
    );
    println!(
        "quantized_hnsw_qps={:.1} recall_at_k={:.4}",
        queries as f64 / quantized_time.as_secs_f64(),
        recall(&quantized_hits, &ground_truth, k)
    );
    println!(
        "f32_vector_bytes={} quantized_payload_bytes={} ratio={:.4}",
        rows * dimensions * size_of::<f32>(),
        mapped.packed_bytes(),
        mapped.packed_bytes() as f64 / (rows * dimensions * size_of::<f32>()) as f64
    );
    Ok(())
}

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn corpus(rows: usize, dimensions: usize) -> Vec<Vec<f32>> {
    let mut state = 0x1234_5678_9abc_def0_u64;
    (0..rows)
        .map(|_| {
            let mut vector = (0..dimensions)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect::<Vec<_>>();
            let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
            for value in &mut vector {
                *value /= norm;
            }
            vector
        })
        .collect()
}

fn exact_top_k(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<SearchResult> {
    let mut rows = vectors
        .iter()
        .enumerate()
        .map(|(id, vector)| SearchResult {
            id,
            distance: 1.0
                - query
                    .iter()
                    .zip(vector)
                    .map(|(left, right)| left * right)
                    .sum::<f32>(),
        })
        .collect::<Vec<_>>();
    rows.select_nth_unstable_by(k - 1, |left, right| {
        left.distance.total_cmp(&right.distance)
    });
    rows.truncate(k);
    rows.sort_by(|left, right| left.distance.total_cmp(&right.distance));
    rows
}

fn measure(
    queries: &[usize],
    mut search: impl FnMut(usize) -> Vec<SearchResult>,
) -> (Duration, Vec<Vec<SearchResult>>) {
    let started = Instant::now();
    let hits = queries
        .iter()
        .map(|row| black_box(search(*row)))
        .collect();
    (started.elapsed(), hits)
}

fn recall(actual: &[Vec<SearchResult>], expected: &[Vec<SearchResult>], k: usize) -> f64 {
    let overlap = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            let expected = expected.iter().map(|hit| hit.id).collect::<HashSet<_>>();
            actual
                .iter()
                .filter(|hit| expected.contains(&hit.id))
                .count()
        })
        .sum::<usize>();
    overlap as f64 / (actual.len() * k) as f64
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
