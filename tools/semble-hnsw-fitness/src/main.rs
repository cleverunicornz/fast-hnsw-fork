use std::cell::Cell;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fast_hnsw::distance::Cosine;
use fast_hnsw::{persist, Builder, Hnsw, PruneStrategy, SearchWorkspace};
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use serde_json::json;

use semble_hnsw_fitness::{
    control_agreement, load_fixture, recall_metrics, reconstruct_hybrid, sha256_file,
    target_metrics, verify_hybrid_controls, ControlAgreement, Fixture, RecallMetrics,
    TargetMetrics, EXPECTED_DIMENSION, RECEIPT_SCHEMA_VERSION,
};

const EFS: [usize; 5] = [50, 100, 200, 400, 800];
const WORKERS: [usize; 4] = [1, 2, 3, 4];
const M: usize = 32;
const M0: usize = 64;
const EF_CONSTRUCTION: usize = 400;
const SEED: u64 = 20_260_905;

type AnyError = Box<dyn std::error::Error>;
type AnyResult<T> = Result<T, AnyError>;

#[derive(Debug)]
struct RunOptions {
    fixture_dir: PathBuf,
    output_dir: PathBuf,
    query_repeats: usize,
    warmup: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BuildReceipt {
    status: String,
    hnsw_ready: bool,
    fast_hnsw_git_sha: String,
    config: BuildConfig,
    phase_timings_ms: PhaseTimings,
    artifact: ArtifactReceipt,
    peak_rss_bytes: Option<u64>,
    graph: GraphReceipt,
    evaluations: Vec<EfEvaluation>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BuildConfig {
    mode: String,
    workers: usize,
    determinism: String,
    m: usize,
    m0: usize,
    ef_construction: usize,
    prune_strategy: String,
    seed: u64,
    use_heuristic: bool,
    extend_candidates: bool,
    keep_pruned: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PhaseTimings {
    fixture_load: f64,
    graph_build: f64,
    persist: f64,
    mmap_open: f64,
    mmap_validation: f64,
    query_evaluation: f64,
    child_total: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ArtifactReceipt {
    file: String,
    format: String,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct GraphReceipt {
    vectors: usize,
    dimension: usize,
    max_level: usize,
    layer_node_counts: Vec<usize>,
    layer_directed_edges: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct EfEvaluation {
    ef: usize,
    dense: DenseEvaluation,
    filtered: Vec<FilteredEvaluation>,
    hybrid: HybridEvaluation,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DenseEvaluation {
    aggregate: RecallMetrics,
    latency_us: LatencySummary,
    queries: Vec<DenseQueryReceipt>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DenseQueryReceipt {
    query_id: String,
    recall: RecallMetrics,
    latency_us: LatencySummary,
    exact_top_10: Vec<usize>,
    hnsw_top_10: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FilteredEvaluation {
    shadow_set: String,
    shadow_file_count: usize,
    aggregate: RecallMetrics,
    latency_us: LatencySummary,
    rejected_nodes_observed: usize,
    shadowed_file_emitted: bool,
    queries: Vec<FilteredQueryReceipt>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FilteredQueryReceipt {
    query_id: String,
    recall: RecallMetrics,
    rejected_nodes_observed: usize,
    exact_top_10: Vec<usize>,
    hnsw_top_10: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct HybridEvaluation {
    control_target: TargetMetrics,
    candidate_target: TargetMetrics,
    control_agreement: ControlAgreement,
    reconstruction_latency_us: LatencySummary,
    queries: Vec<HybridQueryReceipt>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct HybridQueryReceipt {
    query_id: String,
    control_target: TargetMetrics,
    candidate_target: TargetMetrics,
    control_agreement: ControlAgreement,
    reconstruction_us: f64,
    control_top_10: Vec<usize>,
    candidate_top_10: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
struct LatencySummary {
    samples: usize,
    min: f64,
    p50: f64,
    p95: f64,
    mean: f64,
    max: f64,
}

#[derive(Clone, Debug, Serialize)]
struct HardwareFacts {
    os: String,
    kernel: String,
    architecture: String,
    cpu_model: Option<String>,
    logical_cpus: usize,
    memory_total_bytes: Option<u64>,
    output_filesystem_bytes: Option<u64>,
    output_filesystem_available_bytes: Option<u64>,
    output_filesystem_source: Option<String>,
    output_filesystem_type: Option<String>,
    runner_label: String,
    target_cpu_match: bool,
    target_memory_match: bool,
}

fn main() {
    if let Err(error) = real_main() {
        eprintln!("semble-hnsw-fitness failed: {error}");
        std::process::exit(1);
    }
}

fn real_main() -> AnyResult<()> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let command = arguments
        .first()
        .ok_or_else(|| invalid("expected `run` or `internal-build`"))?;
    match command.as_str() {
        "run" => run_parent(parse_run_options(&arguments[1..])?),
        "internal-build" => run_child(&arguments[1..]),
        other => Err(invalid(format!("unknown command {other:?}"))),
    }
}

fn parse_run_options(arguments: &[String]) -> AnyResult<RunOptions> {
    let fixture_dir = required_path(arguments, "--fixture-dir")?;
    let output_dir = required_path(arguments, "--output-dir")?;
    let query_repeats = optional_usize(arguments, "--query-repeats", 20)?;
    let warmup = optional_usize(arguments, "--warmup", 3)?;
    if query_repeats == 0 {
        return Err(invalid("--query-repeats must be greater than zero"));
    }
    Ok(RunOptions {
        fixture_dir,
        output_dir,
        query_repeats,
        warmup,
    })
}

fn run_parent(options: RunOptions) -> AnyResult<()> {
    fs::create_dir_all(&options.output_dir)?;
    for stale in ["receipt.json", "summary.md"] {
        let path = options.output_dir.join(stale);
        if path.exists() {
            fs::remove_file(path)?;
        }
    }

    let loaded = load_fixture(&options.fixture_dir)?;
    verify_hybrid_controls(&loaded.fixture).map_err(invalid)?;
    let fixture = loaded.fixture;
    drop(loaded.vectors);
    drop(loaded.query_vectors);

    let repository_root = repository_root();
    let fast_hnsw_git_sha = git_output(repository_root, &["rev-parse", "HEAD"])?;
    let fast_hnsw_branch = git_output(repository_root, &["branch", "--show-current"])?;
    let worktree_status = git_output(
        repository_root,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    if env::var_os("SEMBLE_HNSW_REQUIRE_CLEAN").is_some() && !worktree_status.is_empty() {
        return Err(invalid("fast-hnsw worktree is not clean"));
    }

    let hardware = hardware_facts(&options.output_dir)?;
    if env::var_os("SEMBLE_HNSW_REQUIRE_TARGET").is_some()
        && (!hardware.target_cpu_match || !hardware.target_memory_match)
    {
        return Err(invalid(format!(
            "runner does not expose the required 4 CPUs and 30 GiB RAM: CPUs={}, RAM={:?}",
            hardware.logical_cpus, hardware.memory_total_bytes
        )));
    }

    let executable = env::current_exe()?;
    let mut builds = Vec::new();
    for workers in WORKERS {
        let shard = options.output_dir.join(format!("build-{workers}.partial.json"));
        let artifact = options
            .output_dir
            .join(format!("semble-baseline-w{workers}.compact.hnsw"));
        if shard.exists() {
            fs::remove_file(&shard)?;
        }
        let status = Command::new(&executable)
            .arg("internal-build")
            .arg("--fixture-dir")
            .arg(&options.fixture_dir)
            .arg("--shard")
            .arg(&shard)
            .arg("--artifact")
            .arg(&artifact)
            .arg("--workers")
            .arg(workers.to_string())
            .arg("--query-repeats")
            .arg(options.query_repeats.to_string())
            .arg("--warmup")
            .arg(options.warmup.to_string())
            .status()?;
        if !status.success() {
            return Err(invalid(format!(
                "{workers}-worker HNSW child failed with {status}"
            )));
        }
        let build: BuildReceipt = serde_json::from_slice(&fs::read(&shard)?)?;
        validate_build_receipt(
            &build,
            &fixture,
            &fast_hnsw_git_sha,
            &artifact,
            workers,
        )?;
        fs::remove_file(shard)?;
        builds.push(build);
    }

    let summary = render_summary(&fixture, &hardware, &builds, &fast_hnsw_git_sha)?;
    write_atomic(&options.output_dir.join("summary.md"), summary.as_bytes())?;
    let fixture_sha256 = sha256_file(&options.fixture_dir.join("fixture.json"))?;
    let generated_unix_seconds = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let query_definitions: Vec<_> = fixture
        .queries
        .iter()
        .map(|query| {
            json!({
                "id": query.id.clone(),
                "kind": query.kind.clone(),
                "text": query.text.clone(),
                "targets": query.targets.clone(),
            })
        })
        .collect();
    let receipt = json!({
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "generated_unix_seconds": generated_unix_seconds,
        "fast_hnsw": {
            "repository": "cleverunicornz/fast-hnsw-fork",
            "git_sha": fast_hnsw_git_sha,
            "branch": fast_hnsw_branch,
            "crate_version": "2.0.0",
            "worktree_clean": worktree_status.is_empty(),
        },
        "semble": fixture.generator,
        "model": fixture.model,
        "corpus": fixture.corpus,
        "query_fixture": fixture.query_fixture,
        "fixture": {
            "schema_version": fixture.schema_version,
            "metadata_file": "fixture/fixture.json",
            "fixture_json_sha256": fixture_sha256,
            "vectors": fixture.vectors,
            "query_vectors": fixture.query_vectors,
        },
        "hardware": hardware,
        "execution": {
            "primary_runner": "cvu-agent-code-x64",
            "production_shape": "r3-32",
            "vcpus": 4,
            "memory_gib": 32,
            "local_nvme_gib": 100,
            "gbp_per_hour_ex_vat": 0.1139,
            "query_efs": EFS,
            "query_repeats": options.query_repeats,
            "warmup_queries": options.warmup,
            "hnsw": {
                "m": M,
                "m0": M0,
                "ef_construction": EF_CONSTRUCTION,
                "prune_strategy": "simple",
                "seed": SEED,
                "use_heuristic": true,
                "extend_candidates": false,
                "keep_pruned": true,
            },
        },
        "controls": fixture.controls,
        "ranking": fixture.ranking,
        "fixture_generation_timings_ms": fixture.timings_ms,
        "fixture_environment": fixture.environment,
        "queries": query_definitions,
        "shadow_sets": fixture.shadow_sets,
        "builds": builds,
        "gates": {
            "owned_semble_source_verified": true,
            "model_and_dimension_verified": true,
            "corpus_revision_and_cleanliness_verified": true,
            "fixture_binary_checksums_verified": true,
            "hybrid_control_reconstructed": true,
            "all_worker_builds_persisted_and_mmap_opened": true,
            "all_results_complete": true,
            "accept_all_filter_matches_dense_hnsw": true,
            "no_shadowed_file_emitted": true,
            "rejected_nodes_observed_during_filtered_navigation": true,
        },
    });
    let encoded = serde_json::to_vec_pretty(&receipt)?;
    write_atomic(&options.output_dir.join("receipt.json"), &encoded)?;
    println!(
        "wrote {} and {}",
        options.output_dir.join("receipt.json").display(),
        options.output_dir.join("summary.md").display()
    );
    Ok(())
}

fn run_child(arguments: &[String]) -> AnyResult<()> {
    let child_started = Instant::now();
    let fixture_dir = required_path(arguments, "--fixture-dir")?;
    let shard = required_path(arguments, "--shard")?;
    let artifact = required_path(arguments, "--artifact")?;
    let workers = required_usize(arguments, "--workers")?;
    let query_repeats = required_usize(arguments, "--query-repeats")?;
    let warmup = required_usize(arguments, "--warmup")?;
    if !WORKERS.contains(&workers) || query_repeats == 0 {
        return Err(invalid("invalid child worker or repetition count"));
    }

    let fixture_started = Instant::now();
    let loaded = load_fixture(&fixture_dir)?;
    verify_hybrid_controls(&loaded.fixture).map_err(invalid)?;
    let fixture_load = elapsed_ms(fixture_started.elapsed());
    let fixture = loaded.fixture;
    let vectors = loaded.vectors;
    let query_vectors = loaded.query_vectors;

    let build_started = Instant::now();
    let builder = Builder::new()
        .m(M)
        .m0(M0)
        .ef_construction(EF_CONSTRUCTION)
        .heuristic(true)
        .extend_candidates(false)
        .keep_pruned(true)
        .prune_strategy(PruneStrategy::Simple)
        .capacity(vectors.len())
        .seed(SEED);
    let index: Hnsw<Cosine> = if workers == 1 {
        let mut index = builder.build(Cosine)?;
        for vector in vectors {
            index.insert(vector)?;
        }
        index
    } else {
        let pool = ThreadPoolBuilder::new().num_threads(workers).build()?;
        pool.install(move || builder.build_parallel(Cosine, vectors))?
    };
    let graph_build = elapsed_ms(build_started.elapsed());
    if index.len() != fixture.chunks.len() || index.dim() != Some(EXPECTED_DIMENSION) {
        return Err(invalid("constructed graph shape does not match the fixture"));
    }
    let stats = index.stats();
    let graph = GraphReceipt {
        vectors: stats.num_vectors,
        dimension: EXPECTED_DIMENSION,
        max_level: stats.max_level,
        layer_node_counts: stats.layer_counts,
        layer_directed_edges: stats.layer_edges,
    };

    if artifact.exists() {
        fs::remove_file(&artifact)?;
    }
    let persist_started = Instant::now();
    persist::save_compact(&index, &artifact)?;
    let persist_ms = elapsed_ms(persist_started.elapsed());
    drop(index);

    let mmap_started = Instant::now();
    let mapped = persist::load_mmap(&artifact, Cosine)?;
    let mmap_open = elapsed_ms(mmap_started.elapsed());
    if mapped.len() != fixture.chunks.len() || mapped.dim() != Some(EXPECTED_DIMENSION) {
        return Err(invalid("mmap-opened graph shape does not match the fixture"));
    }
    let mmap_validation_started = Instant::now();
    let mapped_stats = mapped.stats();
    if mapped_stats.num_vectors != graph.vectors
        || mapped_stats.max_level != graph.max_level
        || mapped_stats.layer_counts != graph.layer_node_counts
        || mapped_stats.layer_edges != graph.layer_directed_edges
    {
        return Err(invalid(
            "mmap-opened graph statistics do not match the completed build",
        ));
    }
    let mmap_validation = elapsed_ms(mmap_validation_started.elapsed());

    let evaluation_started = Instant::now();
    let evaluations = EFS
        .iter()
        .map(|ef| evaluate_ef(&mapped, &fixture, &query_vectors, *ef, query_repeats, warmup))
        .collect::<AnyResult<Vec<_>>>()?;
    let query_evaluation = elapsed_ms(evaluation_started.elapsed());
    let artifact_bytes = fs::metadata(&artifact)?.len();
    let artifact_sha256 = sha256_file(&artifact)?;
    let fast_hnsw_git_sha = git_output(repository_root(), &["rev-parse", "HEAD"])?;
    let file = artifact
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("artifact path has no UTF-8 filename"))?
        .to_owned();
    let peak_rss_bytes = peak_rss_bytes()?;
    if cfg!(target_os = "linux") && peak_rss_bytes.is_none() {
        return Err(invalid("Linux child could not read VmHWM peak RSS"));
    }

    let receipt = BuildReceipt {
        status: "complete".into(),
        hnsw_ready: true,
        fast_hnsw_git_sha,
        config: BuildConfig {
            mode: if workers == 1 {
                "sequential".into()
            } else {
                "parallel".into()
            },
            workers,
            determinism: if workers == 1 {
                "seeded insertion order and graph bytes are deterministic".into()
            } else {
                "seeded levels and fixed inputs; concurrent edge ordering is scheduler-dependent"
                    .into()
            },
            m: M,
            m0: M0,
            ef_construction: EF_CONSTRUCTION,
            prune_strategy: "simple".into(),
            seed: SEED,
            use_heuristic: true,
            extend_candidates: false,
            keep_pruned: true,
        },
        phase_timings_ms: PhaseTimings {
            fixture_load,
            graph_build,
            persist: persist_ms,
            mmap_open,
            mmap_validation,
            query_evaluation,
            child_total: elapsed_ms(child_started.elapsed()),
        },
        artifact: ArtifactReceipt {
            file,
            format: "fast-hnsw compact v2, mmap-ready".into(),
            bytes: artifact_bytes,
            sha256: artifact_sha256,
        },
        peak_rss_bytes,
        graph,
        evaluations,
    };
    let encoded = serde_json::to_vec_pretty(&receipt)?;
    write_atomic(&shard, &encoded)?;
    Ok(())
}

fn evaluate_ef(
    index: &Hnsw<Cosine>,
    fixture: &Fixture,
    query_vectors: &[Vec<f32>],
    ef: usize,
    query_repeats: usize,
    warmup: usize,
) -> AnyResult<EfEvaluation> {
    let candidate_count = fixture.ranking.candidate_count();
    let mut dense_queries = Vec::with_capacity(fixture.queries.len());
    let mut all_dense_latencies = Vec::new();
    let mut dense_results = Vec::with_capacity(fixture.queries.len());

    for (query, vector) in fixture.queries.iter().zip(query_vectors) {
        let mut workspace = SearchWorkspace::new(index.len(), ef);
        for _ in 0..warmup {
            let warm = index.search_with_workspace(vector, candidate_count, ef, &mut workspace)?;
            if warm.len() != candidate_count {
                return Err(invalid(format!("query {} warmup result is incomplete", query.id)));
            }
        }
        let mut samples = Vec::with_capacity(query_repeats);
        let mut results = Vec::new();
        for _ in 0..query_repeats {
            let started = Instant::now();
            results = index.search_with_workspace(vector, candidate_count, ef, &mut workspace)?;
            samples.push(elapsed_us(started.elapsed()));
        }
        if results.len() != candidate_count {
            return Err(invalid(format!("query {} dense result is incomplete", query.id)));
        }
        let candidate_ids: Vec<usize> = results.iter().map(|result| result.id).collect();
        let exact_ids: Vec<usize> = query
            .exact_dense
            .iter()
            .map(|result| result.chunk_index)
            .collect();
        let recall = recall_metrics(&candidate_ids, &exact_ids);
        all_dense_latencies.extend_from_slice(&samples);
        dense_queries.push(DenseQueryReceipt {
            query_id: query.id.clone(),
            recall,
            latency_us: latency_summary(&samples),
            exact_top_10: exact_ids.into_iter().take(10).collect(),
            hnsw_top_10: candidate_ids.iter().take(10).copied().collect(),
        });
        dense_results.push(candidate_ids);
    }
    let dense = DenseEvaluation {
        aggregate: mean_recall(dense_queries.iter().map(|query| query.recall)),
        latency_us: latency_summary(&all_dense_latencies),
        queries: dense_queries,
    };

    let mut filtered = Vec::with_capacity(fixture.shadow_sets.len());
    for shadow in &fixture.shadow_sets {
        let mut shadow_mask = vec![false; fixture.chunks.len()];
        for id in &shadow.chunk_indices {
            shadow_mask[*id] = true;
        }
        let mut query_receipts = Vec::with_capacity(fixture.queries.len());
        let mut latency_samples = Vec::with_capacity(fixture.queries.len());
        let mut rejected_total = 0usize;
        let mut emitted = false;
        for (query_index, (query, vector)) in fixture
            .queries
            .iter()
            .zip(query_vectors)
            .enumerate()
        {
            let rejected = Cell::new(0usize);
            let mut workspace = SearchWorkspace::new(index.len(), ef);
            let started = Instant::now();
            let results = index.search_filtered_with_workspace(
                vector,
                candidate_count,
                ef,
                |id| {
                    let accepted = !shadow_mask[id];
                    if !accepted {
                        rejected.set(rejected.get() + 1);
                    }
                    accepted
                },
                &mut workspace,
            )?;
            latency_samples.push(elapsed_us(started.elapsed()));
            if results.len() != candidate_count {
                return Err(invalid(format!(
                    "query {} filtered result {} is incomplete",
                    query.id, shadow.name
                )));
            }
            let candidate_ids: Vec<usize> = results.iter().map(|result| result.id).collect();
            if shadow.file_count == 0 && candidate_ids != dense_results[query_index] {
                return Err(invalid(format!(
                    "query {} accept-all filtered traversal drifted from dense HNSW",
                    query.id
                )));
            }
            if candidate_ids.iter().any(|id| shadow_mask[*id]) {
                emitted = true;
            }
            let exact = query.filtered_exact.get(&shadow.name).ok_or_else(|| {
                invalid(format!("query {} is missing {} control", query.id, shadow.name))
            })?;
            let exact_ids: Vec<usize> = exact.iter().map(|result| result.chunk_index).collect();
            let recall = recall_metrics(&candidate_ids, &exact_ids);
            rejected_total += rejected.get();
            query_receipts.push(FilteredQueryReceipt {
                query_id: query.id.clone(),
                recall,
                rejected_nodes_observed: rejected.get(),
                exact_top_10: exact_ids.into_iter().take(10).collect(),
                hnsw_top_10: candidate_ids.into_iter().take(10).collect(),
            });
        }
        if emitted {
            return Err(invalid(format!("HNSW emitted a file from {}", shadow.name)));
        }
        if shadow.file_count > 0 && rejected_total == 0 {
            return Err(invalid(format!(
                "{} did not observe rejected nodes during traversal",
                shadow.name
            )));
        }
        filtered.push(FilteredEvaluation {
            shadow_set: shadow.name.clone(),
            shadow_file_count: shadow.file_count,
            aggregate: mean_recall(query_receipts.iter().map(|query| query.recall)),
            latency_us: latency_summary(&latency_samples),
            rejected_nodes_observed: rejected_total,
            shadowed_file_emitted: emitted,
            queries: query_receipts,
        });
    }

    let mut hybrid_queries = Vec::with_capacity(fixture.queries.len());
    let mut reconstruction_latencies = Vec::with_capacity(fixture.queries.len());
    for (query, dense_ids) in fixture.queries.iter().zip(&dense_results) {
        let started = Instant::now();
        let candidate = reconstruct_hybrid(fixture, query, dense_ids).map_err(invalid)?;
        let reconstruction_us = elapsed_us(started.elapsed());
        reconstruction_latencies.push(reconstruction_us);
        if candidate.len() != fixture.ranking.top_k {
            return Err(invalid(format!(
                "query {} hybrid candidate is incomplete",
                query.id
            )));
        }
        let candidate_ids: Vec<usize> = candidate.iter().map(|hit| hit.chunk_index).collect();
        let control_ids: Vec<usize> = query
            .hybrid_control
            .iter()
            .map(|hit| hit.chunk_index)
            .collect();
        hybrid_queries.push(HybridQueryReceipt {
            query_id: query.id.clone(),
            control_target: target_metrics(&control_ids, &fixture.chunks, &query.targets),
            candidate_target: target_metrics(&candidate_ids, &fixture.chunks, &query.targets),
            control_agreement: control_agreement(&candidate_ids, &control_ids),
            reconstruction_us,
            control_top_10: control_ids,
            candidate_top_10: candidate_ids,
        });
    }
    let hybrid = HybridEvaluation {
        control_target: mean_target(hybrid_queries.iter().map(|query| query.control_target)),
        candidate_target: mean_target(hybrid_queries.iter().map(|query| query.candidate_target)),
        control_agreement: mean_agreement(
            hybrid_queries.iter().map(|query| query.control_agreement),
        ),
        reconstruction_latency_us: latency_summary(&reconstruction_latencies),
        queries: hybrid_queries,
    };

    Ok(EfEvaluation {
        ef,
        dense,
        filtered,
        hybrid,
    })
}

fn validate_build_receipt(
    build: &BuildReceipt,
    fixture: &Fixture,
    expected_git_sha: &str,
    artifact_path: &Path,
    expected_workers: usize,
) -> AnyResult<()> {
    if build.status != "complete"
        || !build.hnsw_ready
        || build.fast_hnsw_git_sha != expected_git_sha
        || build.config.workers != expected_workers
        || (build.config.workers == 1) != (build.config.mode == "sequential")
        || build.config.determinism.is_empty()
        || build.config.m != M
        || build.config.m0 != M0
        || build.config.ef_construction != EF_CONSTRUCTION
        || build.config.prune_strategy != "simple"
        || build.config.seed != SEED
        || !build.config.use_heuristic
        || build.config.extend_candidates
        || !build.config.keep_pruned
        || build.graph.vectors != fixture.chunks.len()
        || build.graph.dimension != EXPECTED_DIMENSION
        || build.graph.layer_node_counts.len() != build.graph.layer_directed_edges.len()
        || build.graph.layer_node_counts.is_empty()
        || build.evaluations.len() != EFS.len()
        || !artifact_path.is_file()
        || artifact_path.file_name().and_then(|name| name.to_str())
            != Some(build.artifact.file.as_str())
        || fs::metadata(artifact_path)?.len() != build.artifact.bytes
        || sha256_file(artifact_path)? != build.artifact.sha256
        || build.artifact.bytes == 0
        || build.artifact.sha256.len() != 64
        || ![
            build.phase_timings_ms.fixture_load,
            build.phase_timings_ms.graph_build,
            build.phase_timings_ms.persist,
            build.phase_timings_ms.mmap_open,
            build.phase_timings_ms.mmap_validation,
            build.phase_timings_ms.query_evaluation,
            build.phase_timings_ms.child_total,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
    {
        return Err(invalid(format!(
            "{}-worker build receipt is incomplete",
            build.config.workers
        )));
    }
    for (evaluation, expected_ef) in build.evaluations.iter().zip(EFS) {
        if evaluation.ef != expected_ef
            || evaluation.dense.queries.len() != fixture.queries.len()
            || evaluation.hybrid.queries.len() != fixture.queries.len()
            || evaluation.filtered.len() != fixture.shadow_sets.len()
            || evaluation.filtered.iter().any(|scenario| {
                scenario.shadowed_file_emitted
                    || scenario.queries.len() != fixture.queries.len()
                    || (scenario.shadow_file_count > 0 && scenario.rejected_nodes_observed == 0)
            })
        {
            return Err(invalid(format!(
                "{}-worker ef={} evaluation is incomplete",
                build.config.workers, expected_ef
            )));
        }
    }
    Ok(())
}

fn render_summary(
    fixture: &Fixture,
    hardware: &HardwareFacts,
    builds: &[BuildReceipt],
    fast_hnsw_git_sha: &str,
) -> AnyResult<String> {
    let mut output = String::new();
    writeln!(output, "# Semble / fast-hnsw fitness receipt")?;
    writeln!(output)?;
    writeln!(
        output,
        "Status: **complete**. Every graph was built, persisted, reopened with mmap, \
         and queried as HNSW; no exact fallback is a candidate result."
    )?;
    writeln!(output)?;
    writeln!(output, "- fast-hnsw: `{fast_hnsw_git_sha}` (`2.0.0`)")?;
    writeln!(
        output,
        "- Semble: `{}` at `{}` (`{}`)",
        fixture.generator.semble_tag,
        fixture.generator.semble_git_sha,
        fixture.generator.semble_repository
    )?;
    writeln!(output, "- Model: `{}`, {} dimensions", fixture.model.identifier, fixture.model.dimension)?;
    writeln!(
        output,
        "- Corpus: `{}` at `{}`; {} files, {} chunks",
        fixture.corpus.repository,
        fixture.corpus.git_sha,
        fixture.corpus.indexed_file_count,
        fixture.corpus.chunk_count
    )?;
    writeln!(
        output,
        "- Queries: {} grounded definitions, fixture `{}`",
        fixture.queries.len(),
        fixture.query_fixture.sha256
    )?;
    writeln!(
        output,
        "- Target: `cvu-agent-code-x64` / `r3-32`, 4 vCPU, 32 GiB RAM, \
         100 GiB local NVMe, GBP 0.1139/hour ex VAT"
    )?;
    writeln!(
        output,
        "- Observed: {} logical CPUs, {} RAM, `{}` / `{}`",
        hardware.logical_cpus,
        format_bytes(hardware.memory_total_bytes),
        hardware.os,
        hardware.architecture
    )?;
    writeln!(output)?;
    writeln!(output, "## Controls")?;
    writeln!(output)?;
    writeln!(output, "- Dense control: {}", fixture.controls.dense_control)?;
    writeln!(output, "- Dense candidate: {}", fixture.controls.dense_candidate)?;
    writeln!(output, "- Hybrid control: {}", fixture.controls.hybrid_control)?;
    writeln!(output, "- Hybrid candidate: {}", fixture.controls.hybrid_candidate)?;
    writeln!(output, "- Scope: {}", fixture.controls.delta_scope)?;
    writeln!(output)?;
    writeln!(output, "## Construction")?;
    writeln!(output)?;
    writeln!(
        output,
        "| workers | mode | build ms | persist ms | mmap ms | verify ms | artifact | peak RSS | ready |"
    )?;
    writeln!(output, "|---:|---|---:|---:|---:|---:|---:|---:|:---:|")?;
    for build in builds {
        writeln!(
            output,
            "| {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {} | {} | yes |",
            build.config.workers,
            build.config.mode,
            build.phase_timings_ms.graph_build,
            build.phase_timings_ms.persist,
            build.phase_timings_ms.mmap_open,
            build.phase_timings_ms.mmap_validation,
            format_bytes(Some(build.artifact.bytes)),
            format_bytes(build.peak_rss_bytes),
        )?;
    }
    writeln!(output)?;
    writeln!(
        output,
        "Configuration for every row: `M={M}`, `M0={M0}`, \
         `ef_construction={EF_CONSTRUCTION}`, prune strategy `simple`, seed `{SEED}`, \
         heuristic selection on, extend-candidates off, keep-pruned on."
    )?;
    writeln!(
        output,
        "The sequential graph is seed-reproducible. Parallel rows keep fixed inputs, \
         parameters, and seeded levels, while concurrent edge order remains \
         scheduler-dependent; each receipt hashes the graph actually measured."
    )?;
    writeln!(output)?;
    writeln!(output, "## Quality And Latency")?;
    writeln!(output)?;
    writeln!(
        output,
        "| workers | ef | dense R@1 | R@5 | R@10 | dense p50 us | \
         Semble target R@10 | HNSW target R@10 | Semble MRR@10 | HNSW MRR@10 | \
         Semble agreement R@10 | \
         shadow-10 R@10 | shadow-50 R@10 |"
    )?;
    writeln!(
        output,
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )?;
    for build in builds {
        for evaluation in &build.evaluations {
            let shadow_10 = filtered_named(evaluation, "shadow-10")?;
            let shadow_50 = filtered_named(evaluation, "shadow-50")?;
            writeln!(
                output,
                "| {} | {} | {:.4} | {:.4} | {:.4} | {:.1} | {:.4} | {:.4} | \
                 {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
                build.config.workers,
                evaluation.ef,
                evaluation.dense.aggregate.recall_at_1,
                evaluation.dense.aggregate.recall_at_5,
                evaluation.dense.aggregate.recall_at_10,
                evaluation.dense.latency_us.p50,
                evaluation.hybrid.control_target.recall_at_10,
                evaluation.hybrid.candidate_target.recall_at_10,
                evaluation.hybrid.control_target.mrr_at_10,
                evaluation.hybrid.candidate_target.mrr_at_10,
                evaluation.hybrid.control_agreement.recall_at_10,
                shadow_10.aggregate.recall_at_10,
                shadow_50.aggregate.recall_at_10,
            )?;
        }
    }
    writeln!(output)?;
    writeln!(
        output,
        "Dense latency is mmap HNSW traversal after warmup. Hybrid quality reuses \
         fixture-recorded Semble BM25 ranks and reconstructs its RRF and reranking; \
         the JSON keeps Semble control timings, HNSW timings, and reconstruction \
         timings separate rather than presenting their sum as a directly measured \
         end-to-end latency."
    )?;
    writeln!(
        output,
        "Filtered searches use deterministic nested 0-, 10-, and 50-file exclusions. \
         Rejected nodes were observed during traversal, and no excluded file was emitted."
    )?;
    Ok(output)
}

fn filtered_named<'a>(evaluation: &'a EfEvaluation, name: &str) -> AnyResult<&'a FilteredEvaluation> {
    evaluation
        .filtered
        .iter()
        .find(|scenario| scenario.shadow_set == name)
        .ok_or_else(|| invalid(format!("missing filtered scenario {name}")))
}

fn mean_recall(values: impl Iterator<Item = RecallMetrics>) -> RecallMetrics {
    let values: Vec<_> = values.collect();
    let count = values.len() as f64;
    RecallMetrics {
        recall_at_1: values.iter().map(|value| value.recall_at_1).sum::<f64>() / count,
        recall_at_5: values.iter().map(|value| value.recall_at_5).sum::<f64>() / count,
        recall_at_10: values.iter().map(|value| value.recall_at_10).sum::<f64>() / count,
    }
}

fn mean_target(values: impl Iterator<Item = TargetMetrics>) -> TargetMetrics {
    let values: Vec<_> = values.collect();
    let count = values.len() as f64;
    TargetMetrics {
        recall_at_1: values.iter().map(|value| value.recall_at_1).sum::<f64>() / count,
        recall_at_5: values.iter().map(|value| value.recall_at_5).sum::<f64>() / count,
        recall_at_10: values.iter().map(|value| value.recall_at_10).sum::<f64>() / count,
        mrr_at_10: values.iter().map(|value| value.mrr_at_10).sum::<f64>() / count,
    }
}

fn mean_agreement(values: impl Iterator<Item = ControlAgreement>) -> ControlAgreement {
    let values: Vec<_> = values.collect();
    let count = values.len() as f64;
    ControlAgreement {
        recall_at_1: values.iter().map(|value| value.recall_at_1).sum::<f64>() / count,
        recall_at_5: values.iter().map(|value| value.recall_at_5).sum::<f64>() / count,
        recall_at_10: values.iter().map(|value| value.recall_at_10).sum::<f64>() / count,
        control_top1_mrr_at_10: values
            .iter()
            .map(|value| value.control_top1_mrr_at_10)
            .sum::<f64>()
            / count,
    }
}

fn latency_summary(samples: &[f64]) -> LatencySummary {
    if samples.is_empty() {
        return LatencySummary::default();
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let p50 = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);
    LatencySummary {
        samples: sorted.len(),
        min: sorted[0],
        p50,
        p95,
        mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
        max: *sorted.last().expect("nonempty samples"),
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let rank = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[rank]
}

fn hardware_facts(output_dir: &Path) -> AnyResult<HardwareFacts> {
    let logical_cpus = std::thread::available_parallelism()?.get();
    let memory_total_bytes = linux_memory_total_bytes()?;
    let cpu_model = linux_cpu_model()?;
    let os = linux_pretty_name()?.unwrap_or_else(|| env::consts::OS.to_owned());
    let kernel = command_output("uname", &["-sr"])?;
    let (filesystem_bytes, filesystem_available) = filesystem_capacity(output_dir)?;
    let (filesystem_source, filesystem_type) = filesystem_identity(output_dir)?;
    Ok(HardwareFacts {
        os,
        kernel,
        architecture: env::consts::ARCH.to_owned(),
        cpu_model,
        logical_cpus,
        memory_total_bytes,
        output_filesystem_bytes: filesystem_bytes,
        output_filesystem_available_bytes: filesystem_available,
        output_filesystem_source: filesystem_source,
        output_filesystem_type: filesystem_type,
        runner_label: "cvu-agent-code-x64".into(),
        target_cpu_match: logical_cpus == 4,
        target_memory_match: memory_total_bytes.is_some_and(|bytes| bytes >= 30 * 1024_u64.pow(3)),
    })
}

fn linux_memory_total_bytes() -> io::Result<Option<u64>> {
    let Ok(contents) = fs::read_to_string("/proc/meminfo") else {
        return Ok(None);
    };
    let value = contents.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        match (fields.next(), fields.next()) {
            (Some("MemTotal:"), Some(value)) => value.parse::<u64>().ok(),
            _ => None,
        }
    });
    Ok(value.map(|kib| kib * 1024))
}

fn linux_cpu_model() -> io::Result<Option<String>> {
    let Ok(contents) = fs::read_to_string("/proc/cpuinfo") else {
        return Ok(None);
    };
    Ok(contents.lines().find_map(|line| {
        line.strip_prefix("model name\t:")
            .or_else(|| line.strip_prefix("Hardware\t:"))
            .map(str::trim)
            .map(str::to_owned)
    }))
}

fn linux_pretty_name() -> io::Result<Option<String>> {
    let Ok(contents) = fs::read_to_string("/etc/os-release") else {
        return Ok(None);
    };
    Ok(contents.lines().find_map(|line| {
        line.strip_prefix("PRETTY_NAME=")
            .map(|value| value.trim_matches('"').to_owned())
    }))
}

fn peak_rss_bytes() -> io::Result<Option<u64>> {
    let Ok(contents) = fs::read_to_string("/proc/self/status") else {
        return Ok(None);
    };
    let value = contents.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        match (fields.next(), fields.next()) {
            (Some("VmHWM:"), Some(value)) => value.parse::<u64>().ok(),
            _ => None,
        }
    });
    Ok(value.map(|kib| kib * 1024))
}

fn filesystem_capacity(path: &Path) -> AnyResult<(Option<u64>, Option<u64>)> {
    let output = Command::new("df").arg("-Pk").arg(path).output()?;
    if !output.status.success() {
        return Ok((None, None));
    }
    let text = String::from_utf8(output.stdout)?;
    let Some(line) = text.lines().last() else {
        return Ok((None, None));
    };
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 6 {
        return Ok((None, None));
    }
    let total = fields[1].parse::<u64>().ok().map(|kib| kib * 1024);
    let available = fields[3].parse::<u64>().ok().map(|kib| kib * 1024);
    Ok((total, available))
}

fn filesystem_identity(path: &Path) -> AnyResult<(Option<String>, Option<String>)> {
    let output = Command::new("findmnt")
        .args(["-n", "-o", "SOURCE,FSTYPE", "-T"])
        .arg(path)
        .output();
    let Ok(output) = output else {
        return Ok((None, None));
    };
    if !output.status.success() {
        return Ok((None, None));
    }
    let text = String::from_utf8(output.stdout)?;
    let mut fields = text.split_whitespace();
    Ok((fields.next().map(str::to_owned), fields.next().map(str::to_owned)))
}

fn format_bytes(value: Option<u64>) -> String {
    value.map_or_else(
        || "unknown".into(),
        |bytes| format!("{:.2} GiB", bytes as f64 / 1024_f64.powi(3)),
    )
}

fn command_output(program: &str, arguments: &[&str]) -> AnyResult<String> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(invalid(format!("{program} failed with {}", output.status)));
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn git_output(repository: &Path, arguments: &[&str]) -> AnyResult<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()?;
    if !output.status.success() {
        return Err(invalid(format!(
            "git {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn repository_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("benchmark support crate is nested under repository tools")
}

fn write_atomic(path: &Path, contents: &[u8]) -> io::Result<()> {
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, contents)?;
    fs::rename(temporary, path)
}

fn required_path(arguments: &[String], name: &str) -> AnyResult<PathBuf> {
    Ok(PathBuf::from(required_value(arguments, name)?))
}

fn required_usize(arguments: &[String], name: &str) -> AnyResult<usize> {
    required_value(arguments, name)?
        .parse()
        .map_err(|_| invalid(format!("{name} must be an unsigned integer")))
}

fn optional_usize(arguments: &[String], name: &str, default: usize) -> AnyResult<usize> {
    match option_value(arguments, name) {
        Some(value) => value
            .parse()
            .map_err(|_| invalid(format!("{name} must be an unsigned integer"))),
        None => Ok(default),
    }
}

fn required_value<'a>(arguments: &'a [String], name: &str) -> AnyResult<&'a str> {
    option_value(arguments, name).ok_or_else(|| invalid(format!("missing required option {name}")))
}

fn option_value<'a>(arguments: &'a [String], name: &str) -> Option<&'a str> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].as_str())
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn elapsed_us(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

fn invalid(message: impl Into<String>) -> AnyError {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}
