use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::{self, Read};
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const FIXTURE_SCHEMA_VERSION: u32 = 2;
pub const RECEIPT_SCHEMA_VERSION: u32 = 2;
pub const EXPECTED_SEMBLE_VERSION: &str = "0.7.0";
pub const EXPECTED_SEMBLE_REPOSITORY: &str = "cleverunicornz/semble";
pub const EXPECTED_SEMBLE_TAG: &str = "v0.7.0";
pub const EXPECTED_SEMBLE_SHA: &str = "444a8bde49a9656856ac457d17b2b8ddbe0cd074";
pub const EXPECTED_MODEL: &str = "minishlab/potion-code-16M-v2";
pub const EXPECTED_DIMENSION: usize = 256;
pub const EXPECTED_CORPUS_REPOSITORY: &str = "cleverunicornz/yeet-code";
pub const EXPECTED_CORPUS_SHA: &str = "951dd74fd6cdbe050cb451dc9ab0448836728dbb";
pub const EXPECTED_QUERY_FIXTURE_SHA256: &str =
    "0c94e0d1995fd40e03c8cb6ef1835667f959748acf81fa3854fc6d9f9c26f89d";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Fixture {
    pub schema_version: u32,
    pub generator: GeneratorIdentity,
    pub model: ModelIdentity,
    pub corpus: CorpusIdentity,
    pub query_fixture: QueryFixtureIdentity,
    pub oracle_checks: OracleChecks,
    pub controls: ControlDefinitions,
    pub ranking: RankingRecipe,
    pub vectors: MatrixFile,
    pub query_vectors: MatrixFile,
    pub chunks: Vec<ChunkFixture>,
    pub queries: Vec<QueryFixture>,
    pub shadow_sets: Vec<ShadowSet>,
    pub timings_ms: BTreeMap<String, f64>,
    pub environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GeneratorIdentity {
    pub name: String,
    pub semble_repository: String,
    pub semble_tag: String,
    pub semble_git_sha: String,
    pub semble_version: String,
    pub source_hashes: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelIdentity {
    pub identifier: String,
    pub dimension: usize,
    pub vector_dtype: String,
    pub probe_sha256: String,
    pub model2vec_version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CorpusIdentity {
    pub repository: String,
    pub git_sha: String,
    pub git_tree_sha: String,
    pub clean: bool,
    pub tracked_file_count: usize,
    pub tracked_bytes: u64,
    pub tracked_content_sha256: String,
    pub indexed_file_count: usize,
    pub chunk_count: usize,
    pub indexed_files_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct QueryFixtureIdentity {
    pub file: String,
    pub sha256: String,
    pub query_count: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OracleChecks {
    pub installed_semble_source_verified: bool,
    pub model_identity_verified: bool,
    pub corpus_identity_verified: bool,
    pub dense_backend: String,
    pub dense_backend_verified: bool,
    pub brute_force_query_id: String,
    pub brute_force_top_k: usize,
    pub brute_force_rank_order_equal: bool,
    pub brute_force_top_k_set_equal: bool,
    pub brute_force_top_1_equal: bool,
    pub brute_force_scores_match: bool,
    pub brute_force_max_score_delta: f64,
    pub brute_force_score_tolerance: f64,
    pub shadow_membership_verified: bool,
}

impl OracleChecks {
    pub fn all_passed(&self) -> bool {
        self.installed_semble_source_verified
            && self.model_identity_verified
            && self.corpus_identity_verified
            && self.dense_backend_verified
            && self.brute_force_rank_order_equal
            && self.brute_force_top_k_set_equal
            && self.brute_force_top_1_equal
            && self.brute_force_scores_match
            && self.brute_force_max_score_delta.is_finite()
            && self.brute_force_score_tolerance.is_finite()
            && self.brute_force_score_tolerance > 0.0
            && self.brute_force_max_score_delta <= self.brute_force_score_tolerance
            && self.shadow_membership_verified
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ControlDefinitions {
    pub dense_control: String,
    pub dense_candidate: String,
    pub hybrid_control: String,
    pub hybrid_candidate: String,
    pub delta_scope: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RankingRecipe {
    pub top_k: usize,
    pub candidate_multiplier: usize,
    pub rrf_k: usize,
    pub file_coherence_boost_fraction: f64,
    pub file_saturation_threshold: usize,
    pub file_saturation_decay: f64,
}

impl RankingRecipe {
    pub fn candidate_count(&self) -> usize {
        self.top_k * self.candidate_multiplier
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MatrixFile {
    pub file: String,
    pub rows: usize,
    pub columns: usize,
    pub bytes: u64,
    pub dtype: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChunkFixture {
    pub index: usize,
    pub id: String,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: Option<String>,
    pub content_sha256: String,
    pub file_sha256: String,
    pub path_penalty: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct QueryFixture {
    pub id: String,
    pub kind: String,
    pub text: String,
    pub targets: Vec<String>,
    pub query_vector_row: usize,
    pub alpha: f64,
    pub exact_dense: Vec<RankedHit>,
    pub bm25: Vec<RankedHit>,
    pub hybrid_control: Vec<RankedHit>,
    pub filtered_exact: BTreeMap<String, Vec<RankedHit>>,
    pub control_candidate_order: Vec<usize>,
    pub query_boost_existing: Vec<BoostCoefficient>,
    pub query_boost_injected: Vec<BoostCoefficient>,
    pub timings_us: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RankedHit {
    pub chunk_index: usize,
    pub score: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BoostCoefficient {
    pub chunk_index: usize,
    pub coefficient: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShadowSet {
    pub name: String,
    pub file_count: usize,
    pub files: Vec<String>,
    pub chunk_indices: Vec<usize>,
}

#[derive(Debug)]
pub struct LoadedFixture {
    pub fixture: Fixture,
    pub vectors: Vec<Vec<f32>>,
    pub query_vectors: Vec<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct RecallMetrics {
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct TargetMetrics {
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr_at_10: f64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct ControlAgreement {
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub control_top1_mrr_at_10: f64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ReturnedCountSummary {
    pub requested: usize,
    pub min: usize,
    pub p50: usize,
    pub mean: f64,
    pub max: usize,
    pub short_queries: usize,
}

pub fn load_fixture(fixture_dir: &Path) -> Result<LoadedFixture, Box<dyn std::error::Error>> {
    let raw = fs::read(fixture_dir.join("fixture.json"))?;
    let fixture: Fixture = serde_json::from_slice(&raw)?;
    validate_fixture(&fixture)?;
    let vectors = read_matrix(fixture_dir, &fixture.vectors)?;
    let query_vectors = read_matrix(fixture_dir, &fixture.query_vectors)?;
    validate_matrix_rows(&vectors, "chunk vectors")?;
    validate_matrix_rows(&query_vectors, "query vectors")?;
    Ok(LoadedFixture {
        fixture,
        vectors,
        query_vectors,
    })
}

pub fn validate_fixture(fixture: &Fixture) -> io::Result<()> {
    require(
        fixture.schema_version == FIXTURE_SCHEMA_VERSION,
        format!(
            "fixture schema {} does not match {}",
            fixture.schema_version, FIXTURE_SCHEMA_VERSION
        ),
    )?;
    require(
        fixture.generator.semble_repository == EXPECTED_SEMBLE_REPOSITORY,
        "fixture was not generated by the owned Semble repository",
    )?;
    require(
        fixture.generator.semble_tag == EXPECTED_SEMBLE_TAG,
        "fixture Semble tag drifted",
    )?;
    require(
        fixture.generator.semble_git_sha == EXPECTED_SEMBLE_SHA,
        "fixture Semble commit drifted",
    )?;
    require(
        fixture.generator.semble_version == EXPECTED_SEMBLE_VERSION,
        "fixture Semble version drifted",
    )?;
    require(
        fixture.model.identifier == EXPECTED_MODEL,
        "fixture embedding model drifted",
    )?;
    require(
        fixture.model.dimension == EXPECTED_DIMENSION,
        "fixture embedding dimension drifted",
    )?;
    require(
        fixture.model.vector_dtype == "float32-le",
        "fixture model dtype must be float32-le",
    )?;
    require(
        fixture.corpus.repository == EXPECTED_CORPUS_REPOSITORY,
        "fixture corpus repository drifted",
    )?;
    require(
        fixture.corpus.git_sha == EXPECTED_CORPUS_SHA,
        "fixture corpus commit drifted",
    )?;
    require(fixture.corpus.clean, "fixture corpus was not clean")?;
    require(
        fixture.query_fixture.sha256 == EXPECTED_QUERY_FIXTURE_SHA256,
        "query fixture bytes drifted",
    )?;
    require(
        fixture.query_fixture.query_count == 36 && fixture.queries.len() == 36,
        "fixture must contain all 36 authoritative queries",
    )?;
    require(
        fixture.oracle_checks.all_passed(),
        "fixture oracle evidence contains a failed check",
    )?;
    require(
        fixture.oracle_checks.dense_backend == "semble.index.dense.SelectableBasicBackend"
            && fixture.oracle_checks.brute_force_query_id == "y01"
            && fixture.oracle_checks.brute_force_top_k == 50,
        "fixture dense exactness evidence drifted",
    )?;
    require(
        fixture.corpus.chunk_count == fixture.chunks.len(),
        "corpus chunk count does not match chunk metadata",
    )?;
    require(
        fixture.vectors.rows == fixture.chunks.len()
            && fixture.vectors.columns == EXPECTED_DIMENSION,
        "chunk-vector matrix shape drifted",
    )?;
    require(
        fixture.query_vectors.rows == fixture.queries.len()
            && fixture.query_vectors.columns == EXPECTED_DIMENSION,
        "query-vector matrix shape drifted",
    )?;
    require(
        fixture.vectors.dtype == "float32-le" && fixture.query_vectors.dtype == "float32-le",
        "binary matrices must use float32-le",
    )?;
    require(
        fixture.ranking.top_k == 10
            && fixture.ranking.candidate_multiplier == 5
            && fixture.ranking.rrf_k == 60,
        "Semble candidate or RRF parameters drifted",
    )?;
    require(
        (fixture.ranking.file_coherence_boost_fraction - 0.2).abs() <= f64::EPSILON
            && fixture.ranking.file_saturation_threshold == 1
            && (fixture.ranking.file_saturation_decay - 0.5).abs() <= f64::EPSILON,
        "Semble reranking parameters drifted",
    )?;

    for (index, chunk) in fixture.chunks.iter().enumerate() {
        require(chunk.index == index, "chunk indices must be contiguous")?;
        require(
            chunk.start_line <= chunk.end_line,
            format!("chunk {index} has an invalid line range"),
        )?;
        require(
            chunk.path_penalty.is_finite() && chunk.path_penalty > 0.0 && chunk.path_penalty <= 1.0,
            format!("chunk {index} has an invalid path penalty"),
        )?;
    }

    let candidate_count = fixture.ranking.candidate_count();
    for (index, query) in fixture.queries.iter().enumerate() {
        require(
            query.id == format!("y{:02}", index + 1),
            "query ids must be y01 through y36 in order",
        )?;
        require(
            query.kind == "nl" || query.kind == "sym",
            format!("query {} has an invalid kind", query.id),
        )?;
        require(
            query.query_vector_row == index,
            format!("query {} vector row drifted", query.id),
        )?;
        require(
            !query.text.trim().is_empty() && !query.targets.is_empty(),
            format!("query {} is incomplete", query.id),
        )?;
        require(
            query.alpha.is_finite() && (0.0..=1.0).contains(&query.alpha),
            format!("query {} has an invalid hybrid alpha", query.id),
        )?;
        let expected_alpha = if query.kind == "sym" { 0.3 } else { 0.5 };
        require(
            (query.alpha - expected_alpha).abs() <= f64::EPSILON,
            format!("query {} auto-selected alpha drifted", query.id),
        )?;
        require(
            query.exact_dense.len() == candidate_count,
            format!("query {} exact dense control is incomplete", query.id),
        )?;
        require(
            query.hybrid_control.len() == fixture.ranking.top_k,
            format!("query {} hybrid control is incomplete", query.id),
        )?;
        validate_hits(&query.exact_dense, fixture.chunks.len(), &query.id)?;
        validate_hits(&query.bm25, fixture.chunks.len(), &query.id)?;
        validate_hits(&query.hybrid_control, fixture.chunks.len(), &query.id)?;
        for (shadow_name, hits) in &query.filtered_exact {
            require(
                hits.len() == candidate_count,
                format!(
                    "query {} filtered control {shadow_name} is incomplete",
                    query.id
                ),
            )?;
            validate_hits(hits, fixture.chunks.len(), &query.id)?;
        }
        for boost in query
            .query_boost_existing
            .iter()
            .chain(query.query_boost_injected.iter())
        {
            require(
                boost.chunk_index < fixture.chunks.len()
                    && boost.coefficient.is_finite()
                    && boost.coefficient >= 0.0,
                format!("query {} has an invalid boost coefficient", query.id),
            )?;
        }
    }

    let expected_shadows = [("shadow-0", 0usize), ("shadow-10", 10), ("shadow-50", 50)];
    require(
        fixture.shadow_sets.len() == expected_shadows.len(),
        "fixture must contain 0-, 10-, and 50-file shadow sets",
    )?;
    for (shadow, (name, count)) in fixture.shadow_sets.iter().zip(expected_shadows) {
        require(
            shadow.name == name && shadow.file_count == count && shadow.files.len() == count,
            format!("shadow set {name} is incomplete"),
        )?;
        require(
            fixture
                .queries
                .iter()
                .all(|query| query.filtered_exact.contains_key(name)),
            format!("filtered exact controls are missing {name}"),
        )?;
        require(
            shadow
                .chunk_indices
                .iter()
                .all(|index| *index < fixture.chunks.len()),
            format!("shadow set {name} has an invalid chunk index"),
        )?;
        let expected_indices: Vec<usize> = fixture
            .chunks
            .iter()
            .filter(|chunk| shadow.files.contains(&chunk.file_path))
            .map(|chunk| chunk.index)
            .collect();
        require(
            shadow.chunk_indices == expected_indices,
            format!("shadow set {name} chunk membership drifted"),
        )?;
        let rejected: HashSet<usize> = shadow.chunk_indices.iter().copied().collect();
        require(
            fixture.queries.iter().all(|query| {
                query.filtered_exact[name]
                    .iter()
                    .all(|hit| !rejected.contains(&hit.chunk_index))
            }),
            format!("filtered exact control emitted a member of {name}"),
        )?;
    }
    let mut expected_file_order = Vec::new();
    let mut seen_files = HashSet::new();
    for rank in 0..candidate_count {
        for query in &fixture.queries {
            let path = fixture.chunks[query.exact_dense[rank].chunk_index]
                .file_path
                .clone();
            if seen_files.insert(path.clone()) {
                expected_file_order.push(path);
            }
        }
    }
    let mut remaining: Vec<String> = fixture
        .chunks
        .iter()
        .map(|chunk| chunk.file_path.clone())
        .filter(|path| seen_files.insert(path.clone()))
        .collect();
    remaining.sort();
    expected_file_order.extend(remaining);
    require(
        expected_file_order.len() >= 50,
        "fixture has fewer than 50 deterministically selectable files",
    )?;
    for shadow in &fixture.shadow_sets {
        require(
            shadow.files == expected_file_order[..shadow.file_count],
            format!("shadow set {} selection is not deterministic", shadow.name),
        )?;
    }
    Ok(())
}

fn validate_hits(hits: &[RankedHit], chunk_count: usize, query_id: &str) -> io::Result<()> {
    let mut seen = HashSet::new();
    for hit in hits {
        require(
            hit.chunk_index < chunk_count && hit.score.is_finite(),
            format!("query {query_id} has an invalid ranked hit"),
        )?;
        require(
            seen.insert(hit.chunk_index),
            format!("query {query_id} has duplicate ranked hits"),
        )?;
    }
    Ok(())
}

fn read_matrix(fixture_dir: &Path, spec: &MatrixFile) -> io::Result<Vec<Vec<f32>>> {
    require(
        is_plain_filename(&spec.file),
        "matrix file must be a plain filename",
    )?;
    let bytes = fs::read(fixture_dir.join(&spec.file))?;
    let expected_bytes = spec
        .rows
        .checked_mul(spec.columns)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| invalid_data("matrix byte length overflow"))?;
    require(
        bytes.len() == expected_bytes && spec.bytes == expected_bytes as u64,
        format!("matrix {} byte length drifted", spec.file),
    )?;
    require(
        sha256_hex(&bytes) == spec.sha256,
        format!("matrix {} checksum drifted", spec.file),
    )?;
    decode_f32_matrix(&bytes, spec.rows, spec.columns)
}

pub fn decode_f32_matrix(bytes: &[u8], rows: usize, columns: usize) -> io::Result<Vec<Vec<f32>>> {
    let expected = rows
        .checked_mul(columns)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| invalid_data("matrix dimensions overflow"))?;
    require(
        bytes.len() == expected,
        "matrix dimensions do not match its bytes",
    )?;
    let mut matrix = Vec::with_capacity(rows);
    for row in bytes.chunks_exact(columns * 4) {
        let decoded = row
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte f32")))
            .collect();
        matrix.push(decoded);
    }
    Ok(matrix)
}

fn validate_matrix_rows(matrix: &[Vec<f32>], label: &str) -> io::Result<()> {
    for (row, vector) in matrix.iter().enumerate() {
        require(
            vector.iter().all(|value| value.is_finite()),
            format!("{label} row {row} contains a non-finite value"),
        )?;
        let norm_squared: f64 = vector.iter().map(|value| f64::from(*value).powi(2)).sum();
        require(
            norm_squared > 0.0 && norm_squared.is_finite(),
            format!("{label} row {row} has an invalid norm"),
        )?;
    }
    Ok(())
}

fn is_plain_filename(value: &str) -> bool {
    let mut components = Path::new(value).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn recall_metrics(candidate: &[usize], exact: &[usize]) -> RecallMetrics {
    RecallMetrics {
        recall_at_1: recall_at(candidate, exact, 1),
        recall_at_5: recall_at(candidate, exact, 5),
        recall_at_10: recall_at(candidate, exact, 10),
    }
}

pub fn recall_at(candidate: &[usize], exact: &[usize], k: usize) -> f64 {
    let denominator = exact.len().min(k);
    if denominator == 0 {
        return if candidate.is_empty() { 1.0 } else { 0.0 };
    }
    let exact: HashSet<usize> = exact.iter().take(k).copied().collect();
    let matches = candidate
        .iter()
        .take(k)
        .filter(|id| exact.contains(id))
        .count();
    matches as f64 / denominator as f64
}

pub fn returned_count_summary(counts: &[usize], requested: usize) -> ReturnedCountSummary {
    if counts.is_empty() {
        return ReturnedCountSummary {
            requested,
            ..ReturnedCountSummary::default()
        };
    }
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    ReturnedCountSummary {
        requested,
        min: sorted[0],
        p50: sorted[(sorted.len() - 1) / 2],
        mean: sorted.iter().sum::<usize>() as f64 / sorted.len() as f64,
        max: *sorted.last().expect("nonempty returned counts"),
        short_queries: sorted.iter().filter(|count| **count < requested).count(),
    }
}

pub fn validate_filtered_candidates(
    candidate: &[usize],
    excluded: &[bool],
) -> Result<usize, String> {
    if candidate.is_empty() {
        return Err("filtered HNSW returned zero eligible candidates".into());
    }
    let mut seen = HashSet::new();
    for id in candidate {
        if *id >= excluded.len() {
            return Err(format!("filtered HNSW returned out-of-range id {id}"));
        }
        if excluded[*id] {
            return Err(format!("filtered HNSW emitted excluded id {id}"));
        }
        if !seen.insert(*id) {
            return Err(format!("filtered HNSW returned duplicate id {id}"));
        }
    }
    Ok(candidate.len())
}

pub fn target_metrics(
    candidate: &[usize],
    chunks: &[ChunkFixture],
    targets: &[String],
) -> TargetMetrics {
    let targets: BTreeSet<&str> = targets.iter().map(String::as_str).collect();
    let recall = |k: usize| {
        let found: BTreeSet<&str> = candidate
            .iter()
            .take(k)
            .filter_map(|id| chunks.get(*id))
            .map(|chunk| chunk.file_path.as_str())
            .filter(|path| targets.contains(path))
            .collect();
        if targets.is_empty() {
            0.0
        } else {
            found.len() as f64 / targets.len() as f64
        }
    };
    let reciprocal_rank = candidate
        .iter()
        .take(10)
        .position(|id| {
            chunks
                .get(*id)
                .is_some_and(|chunk| targets.contains(chunk.file_path.as_str()))
        })
        .map_or(0.0, |rank| 1.0 / (rank + 1) as f64);
    TargetMetrics {
        recall_at_1: recall(1),
        recall_at_5: recall(5),
        recall_at_10: recall(10),
        mrr_at_10: reciprocal_rank,
    }
}

pub fn control_agreement(candidate: &[usize], control: &[usize]) -> ControlAgreement {
    let control_top1_mrr_at_10 = control.first().map_or(0.0, |expected| {
        candidate
            .iter()
            .take(10)
            .position(|id| id == expected)
            .map_or(0.0, |rank| 1.0 / (rank + 1) as f64)
    });
    ControlAgreement {
        recall_at_1: recall_at(candidate, control, 1),
        recall_at_5: recall_at(candidate, control, 5),
        recall_at_10: recall_at(candidate, control, 10),
        control_top1_mrr_at_10,
    }
}

pub fn reconstruct_hybrid(
    fixture: &Fixture,
    query: &QueryFixture,
    dense_ids: &[usize],
) -> Result<Vec<RankedHit>, String> {
    let candidate_count = fixture.ranking.candidate_count();
    if dense_ids.len() < candidate_count {
        return Err(format!(
            "query {} dense candidate has {} results; expected at least {candidate_count}",
            query.id,
            dense_ids.len()
        ));
    }

    let mut scores: BTreeMap<usize, f64> = BTreeMap::new();
    let mut dense_seen = HashSet::new();
    for (rank, id) in dense_ids.iter().take(candidate_count).enumerate() {
        if *id >= fixture.chunks.len() || !dense_seen.insert(*id) {
            return Err(format!(
                "query {} has invalid dense candidate ids",
                query.id
            ));
        }
        scores.insert(*id, query.alpha / (fixture.ranking.rrf_k + rank + 1) as f64);
    }
    for (rank, hit) in query.bm25.iter().take(candidate_count).enumerate() {
        *scores.entry(hit.chunk_index).or_default() +=
            (1.0 - query.alpha) / (fixture.ranking.rrf_k + rank + 1) as f64;
    }

    let control_order: HashMap<usize, usize> = query
        .control_candidate_order
        .iter()
        .enumerate()
        .map(|(rank, id)| (*id, rank))
        .collect();
    let mut order: Vec<usize> = scores.keys().copied().collect();
    order.sort_by_key(|id| {
        (
            fixture.chunks[*id].start_line,
            control_order.get(id).copied().unwrap_or(usize::MAX),
            *id,
        )
    });

    apply_file_coherence(&mut scores, &order, fixture)?;
    let max_score = scores.values().copied().fold(0.0, f64::max);
    let existing: HashMap<usize, f64> = query
        .query_boost_existing
        .iter()
        .map(|boost| (boost.chunk_index, boost.coefficient))
        .collect();
    for id in &order {
        if let Some(coefficient) = existing.get(id) {
            *scores.get_mut(id).expect("ordered score") += coefficient * max_score;
        }
    }
    for boost in &query.query_boost_injected {
        if let std::collections::btree_map::Entry::Vacant(entry) = scores.entry(boost.chunk_index) {
            entry.insert(boost.coefficient * max_score);
            order.push(boost.chunk_index);
        }
    }

    rerank(
        &scores,
        &order,
        &fixture.chunks,
        &fixture.ranking,
        query.alpha < 1.0,
    )
}

fn apply_file_coherence(
    scores: &mut BTreeMap<usize, f64>,
    order: &[usize],
    fixture: &Fixture,
) -> Result<(), String> {
    if scores.is_empty() {
        return Ok(());
    }
    let max_score = scores.values().copied().fold(0.0, f64::max);
    if max_score == 0.0 {
        return Ok(());
    }
    let mut file_sum: HashMap<&str, f64> = HashMap::new();
    let mut best_chunk: HashMap<&str, usize> = HashMap::new();
    for id in order {
        let chunk = fixture
            .chunks
            .get(*id)
            .ok_or_else(|| format!("invalid chunk id {id}"))?;
        let score = *scores
            .get(id)
            .ok_or_else(|| format!("missing score for chunk {id}"))?;
        *file_sum.entry(&chunk.file_path).or_default() += score;
        match best_chunk.get(chunk.file_path.as_str()) {
            Some(current) if scores[current] >= score => {}
            _ => {
                best_chunk.insert(&chunk.file_path, *id);
            }
        }
    }
    let max_file_sum = file_sum.values().copied().fold(0.0, f64::max);
    let boost_unit = max_score * fixture.ranking.file_coherence_boost_fraction;
    for (file, id) in best_chunk {
        *scores.get_mut(&id).expect("best chunk has score") +=
            boost_unit * file_sum[file] / max_file_sum;
    }
    Ok(())
}

fn rerank(
    scores: &BTreeMap<usize, f64>,
    order: &[usize],
    chunks: &[ChunkFixture],
    recipe: &RankingRecipe,
    penalise_paths: bool,
) -> Result<Vec<RankedHit>, String> {
    let mut ranked = Vec::with_capacity(scores.len());
    let order_rank: HashMap<usize, usize> = order
        .iter()
        .enumerate()
        .map(|(rank, id)| (*id, rank))
        .collect();
    for (id, score) in scores {
        let chunk = chunks
            .get(*id)
            .ok_or_else(|| format!("invalid reranking chunk {id}"))?;
        let penalty = if penalise_paths {
            chunk.path_penalty
        } else {
            1.0
        };
        ranked.push((*id, score * penalty));
    }
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| order_rank[&left.0].cmp(&order_rank[&right.0]))
    });

    let mut file_selected: HashMap<&str, usize> = HashMap::new();
    let mut selected: Vec<(usize, f64)> = Vec::new();
    let mut min_selected = f64::INFINITY;
    for (id, penalised_score) in ranked {
        if selected.len() >= recipe.top_k && penalised_score <= min_selected {
            break;
        }
        let file = chunks[id].file_path.as_str();
        let already_selected = *file_selected.get(file).unwrap_or(&0);
        let mut effective_score = penalised_score;
        if already_selected >= recipe.file_saturation_threshold {
            let excess = already_selected - recipe.file_saturation_threshold + 1;
            effective_score *= recipe.file_saturation_decay.powi(excess as i32);
        }
        selected.push((id, effective_score));
        file_selected.insert(file, already_selected + 1);
        if selected.len() >= recipe.top_k {
            min_selected = selected
                .iter()
                .map(|(_, score)| *score)
                .fold(f64::INFINITY, f64::min);
        }
    }
    selected.sort_by(|left, right| right.1.total_cmp(&left.1));
    Ok(selected
        .into_iter()
        .take(recipe.top_k)
        .map(|(chunk_index, score)| RankedHit { chunk_index, score })
        .collect())
}

pub fn verify_hybrid_controls(fixture: &Fixture) -> Result<(), String> {
    for query in &fixture.queries {
        let dense: Vec<usize> = query
            .exact_dense
            .iter()
            .map(|hit| hit.chunk_index)
            .collect();
        let reconstructed = reconstruct_hybrid(fixture, query, &dense)?;
        if reconstructed.len() != query.hybrid_control.len() {
            return Err(format!("query {} hybrid control length drifted", query.id));
        }
        for (actual, expected) in reconstructed.iter().zip(&query.hybrid_control) {
            if actual.chunk_index != expected.chunk_index
                || (actual.score - expected.score).abs() > 1e-10
            {
                return Err(format!(
                    "query {} hybrid control drifted: reconstructed {:?}, expected {:?}",
                    query.id, actual, expected
                ));
            }
        }
    }
    Ok(())
}

fn require(condition: bool, message: impl Into<String>) -> io::Result<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid_data(message))
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use fast_hnsw::distance::Euclidean;
    use fast_hnsw::Builder;

    use super::*;

    #[test]
    fn fixture_json_round_trip_parses() {
        let fixture = sample_fixture();
        let encoded = serde_json::to_vec(&fixture).unwrap();
        let decoded: Fixture = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.schema_version, FIXTURE_SCHEMA_VERSION);
        assert_eq!(decoded.queries[0].id, "y01");
        assert_eq!(decoded.chunks[1].file_path, "src/b.rs");
    }

    #[test]
    fn little_endian_matrix_parser_is_exact() {
        let bytes = [1.0f32, -2.5, 3.25, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            decode_f32_matrix(&bytes, 2, 2).unwrap(),
            vec![vec![1.0, -2.5], vec![3.25, 4.0]]
        );
        assert!(decode_f32_matrix(&bytes, 1, 3).is_err());
    }

    #[test]
    fn recall_uses_the_exact_prefix_as_ground_truth() {
        let exact = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let candidate = [1, 8, 2, 12, 13, 3, 4, 5, 6, 7];
        let metrics = recall_metrics(&candidate, &exact);
        assert_eq!(metrics.recall_at_1, 1.0);
        assert_eq!(metrics.recall_at_5, 2.0 / 5.0);
        assert_eq!(metrics.recall_at_10, 8.0 / 10.0);
    }

    #[test]
    fn recall_keeps_exact_denominator_when_candidate_is_short() {
        let exact = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let candidate = [1, 2];
        assert_eq!(recall_at(&candidate, &exact, 10), 2.0 / 10.0);
    }

    #[test]
    fn filtered_candidate_validation_accepts_short_nonempty_results() {
        let excluded = [false, false, false, true];
        assert_eq!(validate_filtered_candidates(&[0, 2], &excluded), Ok(2));
        assert!(validate_filtered_candidates(&[], &excluded).is_err());
        assert!(validate_filtered_candidates(&[0, 3], &excluded).is_err());
    }

    #[test]
    fn returned_count_summary_exposes_short_queries() {
        let summary = returned_count_summary(&[50, 7, 20, 50], 50);
        assert_eq!(summary.requested, 50);
        assert_eq!(summary.min, 7);
        assert_eq!(summary.p50, 20);
        assert_eq!(summary.max, 50);
        assert_eq!(summary.short_queries, 2);
        assert_eq!(summary.mean, 31.75);
    }

    #[test]
    fn oracle_checks_only_pass_when_every_proof_passes() {
        let mut checks = sample_oracle_checks();
        assert!(checks.all_passed());
        checks.brute_force_scores_match = false;
        assert!(!checks.all_passed());
    }

    #[test]
    fn filtered_search_traverses_rejected_nodes_but_never_emits_them() {
        let mut index = Builder::new()
            .m(16)
            .ef_construction(100)
            .seed(44)
            .build(Euclidean)
            .unwrap();
        for id in 0..100 {
            index.insert(vec![id as f32]).unwrap();
        }

        let rejected_seen = Cell::new(0usize);
        let results = index
            .search_filtered(&[12.1], 3, 100, |id| {
                let accepted = id % 10 == 0;
                if !accepted {
                    rejected_seen.set(rejected_seen.get() + 1);
                }
                accepted
            })
            .unwrap();
        assert!(rejected_seen.get() > 0);
        assert_eq!(
            results.iter().map(|result| result.id).collect::<Vec<_>>(),
            [10, 20, 0]
        );
        assert!(results.iter().all(|result| result.id % 10 == 0));
    }

    #[test]
    fn hybrid_reconstruction_applies_rrf_and_query_boosts() {
        let mut fixture = sample_fixture();
        fixture.ranking.top_k = 2;
        fixture.ranking.candidate_multiplier = 1;
        fixture.queries[0].alpha = 0.5;
        fixture.queries[0].bm25 = vec![RankedHit {
            chunk_index: 1,
            score: 9.0,
        }];
        fixture.queries[0].control_candidate_order = vec![0, 1];
        fixture.queries[0].query_boost_existing = vec![BoostCoefficient {
            chunk_index: 1,
            coefficient: 1.0,
        }];
        let result = reconstruct_hybrid(&fixture, &fixture.queries[0], &[0, 1]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].chunk_index, 1);
        assert!(result[0].score > result[1].score);
    }

    fn sample_fixture() -> Fixture {
        let chunks = vec![
            ChunkFixture {
                index: 0,
                id: "c0".into(),
                file_path: "src/a.rs".into(),
                start_line: 1,
                end_line: 2,
                language: Some("rust".into()),
                content_sha256: "a".repeat(64),
                file_sha256: "b".repeat(64),
                path_penalty: 1.0,
            },
            ChunkFixture {
                index: 1,
                id: "c1".into(),
                file_path: "src/b.rs".into(),
                start_line: 3,
                end_line: 4,
                language: Some("rust".into()),
                content_sha256: "c".repeat(64),
                file_sha256: "d".repeat(64),
                path_penalty: 1.0,
            },
        ];
        Fixture {
            schema_version: FIXTURE_SCHEMA_VERSION,
            generator: GeneratorIdentity {
                name: "test".into(),
                semble_repository: EXPECTED_SEMBLE_REPOSITORY.into(),
                semble_tag: EXPECTED_SEMBLE_TAG.into(),
                semble_git_sha: EXPECTED_SEMBLE_SHA.into(),
                semble_version: EXPECTED_SEMBLE_VERSION.into(),
                source_hashes: BTreeMap::new(),
            },
            model: ModelIdentity {
                identifier: EXPECTED_MODEL.into(),
                dimension: 2,
                vector_dtype: "float32-le".into(),
                probe_sha256: "e".repeat(64),
                model2vec_version: "test".into(),
            },
            corpus: CorpusIdentity {
                repository: EXPECTED_CORPUS_REPOSITORY.into(),
                git_sha: EXPECTED_CORPUS_SHA.into(),
                git_tree_sha: "f".repeat(40),
                clean: true,
                tracked_file_count: 2,
                tracked_bytes: 2,
                tracked_content_sha256: "1".repeat(64),
                indexed_file_count: 2,
                chunk_count: 2,
                indexed_files_sha256: "2".repeat(64),
            },
            query_fixture: QueryFixtureIdentity {
                file: "queries.json".into(),
                sha256: EXPECTED_QUERY_FIXTURE_SHA256.into(),
                query_count: 1,
            },
            oracle_checks: sample_oracle_checks(),
            controls: ControlDefinitions {
                dense_control: "exact".into(),
                dense_candidate: "hnsw".into(),
                hybrid_control: "exact hybrid".into(),
                hybrid_candidate: "hnsw hybrid".into(),
                delta_scope: "excluded".into(),
            },
            ranking: RankingRecipe {
                top_k: 2,
                candidate_multiplier: 1,
                rrf_k: 60,
                file_coherence_boost_fraction: 0.2,
                file_saturation_threshold: 1,
                file_saturation_decay: 0.5,
            },
            vectors: MatrixFile {
                file: "vectors.f32le".into(),
                rows: 2,
                columns: 2,
                bytes: 16,
                dtype: "float32-le".into(),
                sha256: "3".repeat(64),
            },
            query_vectors: MatrixFile {
                file: "queries.f32le".into(),
                rows: 1,
                columns: 2,
                bytes: 8,
                dtype: "float32-le".into(),
                sha256: "4".repeat(64),
            },
            chunks,
            queries: vec![QueryFixture {
                id: "y01".into(),
                kind: "nl".into(),
                text: "query".into(),
                targets: vec!["src/a.rs".into()],
                query_vector_row: 0,
                alpha: 0.5,
                exact_dense: vec![
                    RankedHit {
                        chunk_index: 0,
                        score: 1.0,
                    },
                    RankedHit {
                        chunk_index: 1,
                        score: 0.5,
                    },
                ],
                bm25: Vec::new(),
                hybrid_control: Vec::new(),
                filtered_exact: BTreeMap::new(),
                control_candidate_order: vec![0, 1],
                query_boost_existing: Vec::new(),
                query_boost_injected: Vec::new(),
                timings_us: BTreeMap::new(),
            }],
            shadow_sets: Vec::new(),
            timings_ms: BTreeMap::new(),
            environment: BTreeMap::new(),
        }
    }

    fn sample_oracle_checks() -> OracleChecks {
        OracleChecks {
            installed_semble_source_verified: true,
            model_identity_verified: true,
            corpus_identity_verified: true,
            dense_backend: "semble.index.dense.SelectableBasicBackend".into(),
            dense_backend_verified: true,
            brute_force_query_id: "y01".into(),
            brute_force_top_k: 50,
            brute_force_rank_order_equal: true,
            brute_force_top_k_set_equal: true,
            brute_force_top_1_equal: true,
            brute_force_scores_match: true,
            brute_force_max_score_delta: 0.0,
            brute_force_score_tolerance: 5e-4,
            shadow_membership_verified: true,
        }
    }
}
