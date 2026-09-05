#!/usr/bin/env python3
"""Generate the Semble-owned half of the HNSW fitness fixture.

This script intentionally runs against the runner's installed Semble. It
refuses any package whose behavior-bearing source files do not match the owned
v0.7.0 release, then records the exact vectors and control-stage results used
by the Rust benchmark support binary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import struct
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

FIXTURE_SCHEMA_VERSION = 2
SEMBLE_REPOSITORY = "cleverunicornz/semble"
SEMBLE_TAG = "v0.7.0"
SEMBLE_GIT_SHA = "444a8bde49a9656856ac457d17b2b8ddbe0cd074"
SEMBLE_VERSION = "0.7.0"
MODEL_IDENTIFIER = "minishlab/potion-code-16M-v2"
MODEL_DIMENSION = 256
CORPUS_REPOSITORY = "cleverunicornz/yeet-code"
CORPUS_GIT_SHA = "951dd74fd6cdbe050cb451dc9ab0448836728dbb"
QUERY_FIXTURE_SHA256 = (
    "0c94e0d1995fd40e03c8cb6ef1835667f959748acf81fa3854fc6d9f9c26f89d"
)
TOP_K = 10
CANDIDATE_MULTIPLIER = 5
DIRECT_COSINE_SCORE_TOLERANCE = 5e-4
CANDIDATE_COUNT = TOP_K * CANDIDATE_MULTIPLIER

# These hashes bind installed code to cleverunicornz/semble@v0.7.0. They are
# hashes of the raw files at SEMBLE_GIT_SHA, not hashes from upstream MinishLab.
SEMBLE_SOURCE_HASHES = {
    "version.py": "881b9935aea9e8846157519a516acc4b706910e6d75fb2a2e117bf93a8cb4d88",
    "index/dense.py": "cadaecf677186892665765a772b873580ff3fda8b8519b1ca76f4f665460f460",
    "index/create.py": "0b7a987baf70797545006eec743620f3a9324b576bc3ddb4882cb35056746b3b",
    "index/index.py": "b8f586cd184edeac79ae82d2eff8a9500f809fa86f088b944ea170a196fb51c2",
    "search.py": "0d1d8d9db837500b021b91a7176e46dc53efe3dbedae6b629f977ddbadc96309",
    "ranking/boosting.py": "1f65e73a9e406ade3e62e17719482d7eb9a63e4265024f65cca045c3d64faeec",
    "ranking/penalties.py": "5ade8634f8c7aef5f310a79a6fc3b35bcc40543a36f3b03563173c3c2bd0b0d8",
    "ranking/weighting.py": "c0032c400db8d5671ef42c9595cfc15b26bd778f8f2547d4b303bce62ed7db36",
    "chunking/chunking.py": "5f2cbd2fda5a7d386e3eca6c368ed60fc2c24c7fe5a2db93af1bb36c97e789c2",
    "chunking/core.py": "ca00470e0c3eb310cd75baff3930f90fc1237d7f22e07cd1e85dc044ed1aadb1",
    "index/bm25.py": "5c4b56bfc91f0e5af0dfafbdbb2c41243d5b9066e68f411284d31a31ed630e01",
    "index/file_walker.py": "e3c2c0a6fb217b1c15d179cc33a1f2afdfdeeb1e9e741cf67868d6c4c5678a33",
    "index/files.py": "7886184881f1c392e5c71d65c01748f691b1c84c1d66bf1d0834188952d0d4c1",
    "index/sparse.py": "b1ba36afa0770c6dd871ee095de28d9662c99a19a38728992d1f61dac30bacb5",
    "index/types.py": "675f87bc4ed5ff918a9f6738d745a67cc0c209e9447caf8286064bbf08f0dfe2",
    "tokens.py": "d9406b36c59468e7e8ad86e939c3bd8acafa50e640f4d5404d152be841d0009e",
    "types.py": "a59339f3b09e76d8675073d3bf021a070ed2f50cb7503cbeaca1b4fd5cac1ad2",
    "utils.py": "266c8ae25f27a496385b9f45b9b95f895b69a50ff98625b9611ea062607c1931",
}

T = TypeVar("T")


@dataclass
class QueryDraft:
    definition: dict[str, Any]
    prepared: Any
    exact_dense: list[Any]
    bm25: list[Any]
    hybrid_control: list[Any]
    control_candidate_order: list[int]
    query_boost_existing: list[dict[str, Any]]
    query_boost_injected: list[dict[str, Any]]
    timings_us: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timed(function: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter_ns()
    result = function()
    elapsed_us = (time.perf_counter_ns() - started) / 1_000.0
    return result, elapsed_us


def git(corpus: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(corpus), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_installed_semble() -> dict[str, str]:
    spec = importlib.util.find_spec("semble")
    if spec is None or not spec.submodule_search_locations:
        fail("the runner's pinned Semble package is not installed")
    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    observed: dict[str, str] = {}
    for relative, expected in SEMBLE_SOURCE_HASHES.items():
        source = package_root / relative
        if not source.is_file():
            fail(f"installed Semble source is missing {relative}")
        actual = sha256_bytes(source.read_bytes())
        observed[relative] = actual
        if actual != expected:
            fail(
                f"installed Semble source drift for {relative}: "
                f"expected {expected}, found {actual}"
            )
    return observed


def verify_queries(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != QUERY_FIXTURE_SHA256:
        fail(
            f"authoritative query fixture drift: expected {QUERY_FIXTURE_SHA256}, found {digest}"
        )
    queries = json.loads(raw)
    if not isinstance(queries, list) or len(queries) != 36:
        fail("the authoritative query fixture must contain exactly 36 entries")
    for index, query in enumerate(queries, 1):
        expected_id = f"y{index:02d}"
        if query.get("id") != expected_id:
            fail(
                f"query fixture must contain ordered ids y01 through y36; missing {expected_id}"
            )
        if query.get("kind") not in {"nl", "sym"}:
            fail(f"query {expected_id} has an invalid kind")
        if not query.get("query") or not query.get("targets"):
            fail(f"query {expected_id} is incomplete")
    return queries, digest


def verify_corpus(corpus: Path) -> dict[str, Any]:
    if not corpus.is_dir():
        fail(f"corpus is not a directory: {corpus}")
    revision = git(corpus, "rev-parse", "HEAD")
    if revision != CORPUS_GIT_SHA:
        fail(f"wrong corpus revision: expected {CORPUS_GIT_SHA}, found {revision}")
    status = git(corpus, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        fail(f"corpus worktree is not clean:\n{status}")

    tree = git(corpus, "rev-parse", "HEAD^{tree}")
    names_raw = subprocess.run(
        ["git", "-C", str(corpus), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    names = [name.decode("utf-8") for name in names_raw.split(b"\0") if name]
    digest = hashlib.sha256()
    digest.update(b"semble-hnsw-corpus-worktree-v1\0")
    total_bytes = 0
    for name in sorted(names):
        path = corpus / name
        if path.is_symlink():
            data = os.readlink(path).encode("utf-8")
        else:
            data = path.read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<Q", len(data)))
        digest.update(data)
        total_bytes += len(data)
    return {
        "repository": CORPUS_REPOSITORY,
        "git_sha": revision,
        "git_tree_sha": tree,
        "clean": True,
        "tracked_file_count": len(names),
        "tracked_bytes": total_bytes,
        "tracked_content_sha256": digest.hexdigest(),
    }


def matrix_bytes(matrix: np.ndarray, expected_columns: int) -> bytes:
    array = np.asarray(matrix, dtype=np.float32, order="C")
    if array.ndim != 2 or array.shape[1] != expected_columns:
        fail(
            f"matrix shape drift: expected (*, {expected_columns}), found {array.shape}"
        )
    if not np.isfinite(array).all():
        fail("matrix contains non-finite values")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        fail("matrix contains a zero or invalid vector")
    return array.astype("<f4", copy=False).tobytes(order="C")


def verify_direct_cosine_parity(
    corpus_bytes: bytes,
    query_bytes: bytes,
    recorded: list[Any],
    chunk_to_index: dict[Any, int],
) -> dict[str, Any]:
    vectors = np.frombuffer(corpus_bytes, dtype="<f4").reshape(-1, MODEL_DIMENSION)
    queries = np.frombuffer(query_bytes, dtype="<f4").reshape(-1, MODEL_DIMENSION)
    query = queries[0]
    vector_norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query)
    if (
        np.any(vector_norms <= 0)
        or not np.isfinite(vector_norms).all()
        or query_norm <= 0
    ):
        fail("direct cosine parity encountered an invalid exported vector norm")
    similarities = vectors.dot(query) / (vector_norms * query_norm)
    direct_indices = np.argsort(-similarities, kind="stable")[:CANDIDATE_COUNT]
    recorded_indices = np.array(
        [chunk_to_index[result.chunk] for result in recorded],
        dtype=np.int64,
    )
    rank_order_equal = np.array_equal(direct_indices, recorded_indices)
    top_k_set_equal = set(map(int, direct_indices)) == set(map(int, recorded_indices))
    top_1_equal = int(direct_indices[0]) == int(recorded_indices[0])
    scores_match = all(
        np.isclose(
            float(result.score),
            float(similarities[chunk_to_index[result.chunk]]),
            rtol=DIRECT_COSINE_SCORE_TOLERANCE,
            atol=DIRECT_COSINE_SCORE_TOLERANCE,
        )
        for result in recorded
    )
    direct_set = set(map(int, direct_indices))
    recorded_set = set(map(int, recorded_indices))
    score_deltas = [
        abs(float(result.score) - float(similarities[chunk_to_index[result.chunk]]))
        for result in recorded
    ]
    if (
        not rank_order_equal
        or not top_k_set_equal
        or not top_1_equal
        or not scores_match
    ):
        fail(
            "Semble exact dense control did not match direct brute-force cosine "
            "over the exported vectors for y01: "
            f"set_equal={top_k_set_equal}, top_1_equal={top_1_equal}, "
            f"rank_order_equal={rank_order_equal}, "
            f"scores_match={scores_match}, "
            f"direct_only={sorted(direct_set - recorded_set)[:10]}, "
            f"recorded_only={sorted(recorded_set - direct_set)[:10]}, "
            f"max_score_delta={max(score_deltas, default=0.0):.9g}"
        )
    return {
        "brute_force_query_id": "y01",
        "brute_force_top_k": CANDIDATE_COUNT,
        "brute_force_rank_order_equal": rank_order_equal,
        "brute_force_top_k_set_equal": top_k_set_equal,
        "brute_force_top_1_equal": top_1_equal,
        "brute_force_scores_match": scores_match,
        "brute_force_max_score_delta": max(score_deltas, default=0.0),
        "brute_force_score_tolerance": DIRECT_COSINE_SCORE_TOLERANCE,
    }


def ranked_hits(
    results: list[Any], chunk_to_index: dict[Any, int]
) -> list[dict[str, Any]]:
    return [
        {"chunk_index": chunk_to_index[result.chunk], "score": float(result.score)}
        for result in results
    ]


def boost_recipe(
    query: str,
    chunks: list[Any],
    chunk_to_index: dict[Any, int],
    apply_query_boost: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unit_scores = {chunk: 1.0 for chunk in chunks}
    boosted = apply_query_boost(unit_scores, query, chunks)
    existing = [
        {"chunk_index": chunk_to_index[chunk], "coefficient": float(score - 1.0)}
        for chunk, score in boosted.items()
        if score != 1.0
    ]

    injected_by_index: dict[int, float] = {}
    for sentinel in chunks[:2]:
        initial = {sentinel: 1.0}
        observed = apply_query_boost(initial, query, chunks)
        for chunk, score in observed.items():
            if chunk not in initial:
                injected_by_index[chunk_to_index[chunk]] = float(score)
    injected = [
        {"chunk_index": index, "coefficient": coefficient}
        for index, coefficient in sorted(injected_by_index.items())
    ]
    existing.sort(key=lambda item: item["chunk_index"])
    return existing, injected


def select_shadow_files(drafts: list[QueryDraft], chunks: list[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for rank in range(CANDIDATE_COUNT):
        for draft in drafts:
            path = draft.exact_dense[rank].chunk.file_path
            if path not in seen:
                seen.add(path)
                ordered.append(path)
    for path in sorted({chunk.file_path for chunk in chunks}):
        if path not in seen:
            ordered.append(path)
    if len(ordered) < 50:
        fail("corpus has fewer than 50 indexed files; shadow scenarios are incomplete")
    return ordered


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    corpus = args.corpus.resolve()
    queries_path = args.queries.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == corpus or corpus in output_dir.parents:
        fail("fixture output must be outside the corpus worktree")
    if os.environ.get("PYTHONHASHSEED") != "0":
        fail("PYTHONHASHSEED must be exactly 0 for deterministic Semble tie behavior")
    configured_model = os.environ.get("SEMBLE_MODEL_NAME", MODEL_IDENTIFIER)
    if configured_model != MODEL_IDENTIFIER:
        fail(
            f"SEMBLE_MODEL_NAME drift: expected {MODEL_IDENTIFIER}, found {configured_model}"
        )
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        fail("HF_HUB_OFFLINE must be 1 so the pinned runner model cannot be replaced")

    total_started = time.perf_counter_ns()
    observed_source_hashes = verify_installed_semble()

    from semble.index.create import create_index_from_path
    from semble.index.dense import SelectableBasicBackend, load_model
    from semble.index.index import SembleIndex
    from semble.ranking.boosting import _FILE_COHERENCE_BOOST_FRAC, apply_query_boost
    from semble.ranking.penalties import (
        _FILE_SATURATION_DECAY,
        _FILE_SATURATION_THRESHOLD,
        _file_path_penalty,
    )
    from semble.search import (
        _RRF_K,
        _search_bm25_prepared,
        _search_semantic_prepared,
    )
    from semble.types import ContentType
    from semble.version import __version__
    from vicinity.utils import normalize

    distribution_version = importlib.metadata.version("semble")
    if __version__ != SEMBLE_VERSION:
        fail(
            f"installed Semble version drift: expected {SEMBLE_VERSION}, found {__version__}"
        )
    if distribution_version != SEMBLE_VERSION:
        fail(
            f"installed Semble distribution drift: expected {SEMBLE_VERSION}, "
            f"found {distribution_version}"
        )
    if _RRF_K != 60:
        fail(f"Semble RRF constant drift: expected 60, found {_RRF_K}")
    if _FILE_COHERENCE_BOOST_FRAC != 0.2:
        fail("Semble file-coherence boost drifted")
    if _FILE_SATURATION_THRESHOLD != 1 or _FILE_SATURATION_DECAY != 0.5:
        fail("Semble file-saturation behavior drifted")

    query_definitions, query_digest = verify_queries(queries_path)
    corpus_identity, corpus_hash_us = timed(lambda: verify_corpus(corpus))

    (model, resolved_model), model_load_us = timed(lambda: load_model(MODEL_IDENTIFIER))
    if resolved_model != MODEL_IDENTIFIER:
        fail(f"Semble resolved a different model: {resolved_model}")
    if int(model.dim) != MODEL_DIMENSION:
        fail(f"model dimension drift: expected {MODEL_DIMENSION}, found {model.dim}")

    def create_index() -> tuple[Any, Any, list[Any], Any]:
        return create_index_from_path(
            corpus,
            model,
            content=(ContentType.CODE,),
            display_root=corpus,
        )

    (bm25_index, semantic_index, chunks, manifest), semble_index_us = timed(
        create_index
    )
    dense_backend = (
        f"{type(semantic_index).__module__}.{type(semantic_index).__qualname__}"
    )
    dense_backend_verified = type(semantic_index) is SelectableBasicBackend
    if not dense_backend_verified:
        fail(
            "Semble dense oracle must be semble.index.dense.SelectableBasicBackend; "
            f"found {dense_backend}"
        )
    index = SembleIndex(
        model,
        bm25_index,
        semantic_index,
        chunks,
        MODEL_IDENTIFIER,
        root=corpus,
        content=(ContentType.CODE,),
        manifest=manifest,
    )
    if len(chunks) < CANDIDATE_COUNT:
        fail(
            f"corpus produced only {len(chunks)} chunks; at least {CANDIDATE_COUNT} are required"
        )

    chunk_to_index = {chunk: index for index, chunk in enumerate(chunks)}
    if len(chunk_to_index) != len(chunks):
        fail("Semble produced duplicate chunk identities")
    indexed_paths = sorted({chunk.file_path for chunk in chunks})
    indexed_path_set = set(indexed_paths)
    for query in query_definitions:
        missing = sorted(set(query["targets"]) - indexed_path_set)
        if missing:
            fail(f"query {query['id']} target paths are not indexed: {missing}")

    file_hashes: dict[str, str] = {}
    indexed_files_digest = hashlib.sha256()
    indexed_files_digest.update(b"semble-indexed-files-v1\0")
    for relative in indexed_paths:
        data = (corpus / relative).read_bytes()
        digest = sha256_bytes(data)
        file_hashes[relative] = digest
        encoded = relative.encode("utf-8")
        indexed_files_digest.update(struct.pack("<Q", len(encoded)))
        indexed_files_digest.update(encoded)
        indexed_files_digest.update(bytes.fromhex(digest))

    path_penalties = [_file_path_penalty(chunk.file_path) for chunk in chunks]
    chunk_metadata = []
    for chunk_index, chunk in enumerate(chunks):
        content_bytes = chunk.content.encode("utf-8")
        identity = hashlib.sha256()
        identity.update(b"semble-chunk-v1\0")
        identity.update(struct.pack("<Q", chunk_index))
        identity.update(chunk.file_path.encode("utf-8"))
        identity.update(b"\0")
        identity.update(struct.pack("<QQ", chunk.start_line, chunk.end_line))
        identity.update(content_bytes)
        chunk_metadata.append(
            {
                "index": chunk_index,
                "id": identity.hexdigest(),
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language,
                "content_sha256": sha256_bytes(content_bytes),
                "file_sha256": file_hashes[chunk.file_path],
                "path_penalty": float(path_penalties[chunk_index]),
            }
        )

    corpus_vectors = np.asarray(semantic_index.vectors, dtype=np.float32, order="C")
    if corpus_vectors.shape != (len(chunks), MODEL_DIMENSION):
        fail(f"Semble vector matrix shape drift: {corpus_vectors.shape}")

    drafts: list[QueryDraft] = []
    query_vector_rows: list[np.ndarray] = []
    for definition in query_definitions:
        prepared, prepare_us = timed(
            lambda definition=definition: index.prepare_query(definition["query"])
        )
        embedding = np.asarray(normalize(prepared.embedding), dtype=np.float32).reshape(
            -1
        )
        if embedding.shape != (MODEL_DIMENSION,):
            fail(f"query {definition['id']} embedding shape drift: {embedding.shape}")
        query_vector_rows.append(embedding)

        exact_dense, exact_dense_us = timed(
            lambda prepared=prepared: _search_semantic_prepared(
                prepared.embedding,
                semantic_index,
                chunks,
                CANDIDATE_COUNT,
                None,
            )
        )
        bm25, bm25_us = timed(
            lambda prepared=prepared: _search_bm25_prepared(
                prepared.tokens,
                bm25_index,
                chunks,
                CANDIDATE_COUNT,
                None,
            )
        )
        hybrid_control, hybrid_us = timed(
            lambda prepared=prepared: index.search_prepared(
                prepared,
                TOP_K,
                rerank=True,
                record_stats=False,
            )
        )
        if len(exact_dense) != CANDIDATE_COUNT or len(hybrid_control) != TOP_K:
            fail(f"query {definition['id']} produced incomplete controls")

        semantic_chunks = {result.chunk for result in exact_dense}
        bm25_chunks = {result.chunk for result in bm25 if result.score}
        control_candidate_order = [
            chunk_to_index[chunk]
            for chunk in sorted(
                semantic_chunks | bm25_chunks, key=lambda chunk: chunk.start_line
            )
        ]
        existing, injected = boost_recipe(
            definition["query"], chunks, chunk_to_index, apply_query_boost
        )
        drafts.append(
            QueryDraft(
                definition=definition,
                prepared=prepared,
                exact_dense=exact_dense,
                bm25=bm25,
                hybrid_control=hybrid_control,
                control_candidate_order=control_candidate_order,
                query_boost_existing=existing,
                query_boost_injected=injected,
                timings_us={
                    "prepare_query": prepare_us,
                    "exact_dense": exact_dense_us,
                    "bm25": bm25_us,
                    "hybrid_control": hybrid_us,
                },
            )
        )

    shadow_file_order = select_shadow_files(drafts, chunks)
    shadow_sets = []
    for count in (0, 10, 50):
        files = shadow_file_order[:count]
        file_set = set(files)
        chunk_indices = sorted(
            chunk_to_index[chunk] for chunk in chunks if chunk.file_path in file_set
        )
        shadow_sets.append(
            {
                "name": f"shadow-{count}",
                "file_count": count,
                "files": files,
                "chunk_indices": chunk_indices,
            }
        )

    shadow_exclusions: dict[str, Any] = {}
    shadow_membership_verified = True
    for shadow in shadow_sets:
        oracle_excluded = (
            index.indices_for_paths(set(shadow["files"])) if shadow["files"] else None
        )
        oracle_indices = (
            []
            if oracle_excluded is None
            else sorted(int(value) for value in oracle_excluded)
        )
        serialized_indices = sorted(int(value) for value in shadow["chunk_indices"])
        membership_equal = oracle_indices == serialized_indices
        shadow_membership_verified = shadow_membership_verified and membership_equal
        if not membership_equal:
            fail(
                f"Semble indices_for_paths disagrees with serialized membership for "
                f"{shadow['name']}"
            )
        shadow_exclusions[shadow["name"]] = oracle_excluded

    query_metadata = []
    for query_index, draft in enumerate(drafts):
        filtered_exact: dict[str, list[dict[str, Any]]] = {}
        filtered_total_us = 0.0
        for shadow in shadow_sets:
            excluded = shadow_exclusions[shadow["name"]]
            results, elapsed_us = timed(
                lambda draft=draft, excluded=excluded: _search_semantic_prepared(
                    draft.prepared.embedding,
                    semantic_index,
                    chunks,
                    CANDIDATE_COUNT,
                    None,
                    excluded,
                )
            )
            if len(results) != CANDIDATE_COUNT:
                fail(
                    f"query {draft.definition['id']} filtered control {shadow['name']} is incomplete"
                )
            shadowed = set(shadow["files"])
            if any(result.chunk.file_path in shadowed for result in results):
                fail(
                    f"Semble exact control emitted a shadowed file for {shadow['name']}"
                )
            filtered_exact[shadow["name"]] = ranked_hits(results, chunk_to_index)
            filtered_total_us += elapsed_us
        timings = dict(draft.timings_us)
        timings["filtered_exact_total"] = filtered_total_us
        query_metadata.append(
            {
                "id": draft.definition["id"],
                "kind": draft.definition["kind"],
                "text": draft.definition["query"],
                "targets": draft.definition["targets"],
                "query_vector_row": query_index,
                "alpha": float(draft.prepared.alpha_weight),
                "exact_dense": ranked_hits(draft.exact_dense, chunk_to_index),
                "bm25": ranked_hits(draft.bm25, chunk_to_index),
                "hybrid_control": ranked_hits(draft.hybrid_control, chunk_to_index),
                "filtered_exact": filtered_exact,
                "control_candidate_order": draft.control_candidate_order,
                "query_boost_existing": draft.query_boost_existing,
                "query_boost_injected": draft.query_boost_injected,
                "timings_us": timings,
            }
        )

    corpus_bytes = matrix_bytes(corpus_vectors, MODEL_DIMENSION)
    query_matrix = np.vstack(query_vector_rows)
    query_bytes = matrix_bytes(query_matrix, MODEL_DIMENSION)
    direct_cosine_checks, direct_cosine_us = timed(
        lambda: verify_direct_cosine_parity(
            corpus_bytes,
            query_bytes,
            drafts[0].exact_dense,
            chunk_to_index,
        )
    )
    probe = np.asarray(
        model.encode(
            [
                "Semble HNSW fitness model identity probe",
                "BranchLeaseManager",
                "hybrid code search with reciprocal rank fusion",
            ]
        ),
        dtype=np.float32,
    )
    probe_sha256 = sha256_bytes(matrix_bytes(probe, MODEL_DIMENSION))

    if git(corpus, "status", "--porcelain=v1", "--untracked-files=all"):
        fail("Semble fixture generation modified the corpus worktree")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(output_dir / "vectors.f32le", corpus_bytes)
    write_atomic(output_dir / "queries.f32le", query_bytes)
    total_ms = (time.perf_counter_ns() - total_started) / 1_000_000.0
    corpus_identity.update(
        {
            "indexed_file_count": len(indexed_paths),
            "chunk_count": len(chunks),
            "indexed_files_sha256": indexed_files_digest.hexdigest(),
        }
    )
    fixture = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "generator": {
            "name": "generate_fixture.py",
            "semble_repository": SEMBLE_REPOSITORY,
            "semble_tag": SEMBLE_TAG,
            "semble_git_sha": SEMBLE_GIT_SHA,
            "semble_version": SEMBLE_VERSION,
            "source_hashes": SEMBLE_SOURCE_HASHES,
        },
        "model": {
            "identifier": MODEL_IDENTIFIER,
            "dimension": MODEL_DIMENSION,
            "vector_dtype": "float32-le",
            "probe_sha256": probe_sha256,
            "model2vec_version": importlib.metadata.version("model2vec"),
        },
        "corpus": corpus_identity,
        "query_fixture": {
            "file": queries_path.name,
            "sha256": query_digest,
            "query_count": len(query_definitions),
        },
        "oracle_checks": {
            "installed_semble_source_verified": (
                observed_source_hashes == SEMBLE_SOURCE_HASHES
            ),
            "model_identity_verified": (
                resolved_model == MODEL_IDENTIFIER and int(model.dim) == MODEL_DIMENSION
            ),
            "corpus_identity_verified": (
                corpus_identity["git_sha"] == CORPUS_GIT_SHA
                and corpus_identity["clean"]
            ),
            "dense_backend": dense_backend,
            "dense_backend_verified": dense_backend_verified,
            **direct_cosine_checks,
            "shadow_membership_verified": shadow_membership_verified,
        },
        "controls": {
            "dense_control": "Semble 0.7.0 exact cosine top-50 over its baseline chunk vectors; BM25 disabled",
            "dense_candidate": "fast-hnsw 2.0 cosine HNSW top-50 over the identical baseline vectors; BM25 disabled",
            "hybrid_control": (
                "Semble 0.7.0 exact dense plus its BM25, weighted RRF, query boosts, "
                "path penalties, and file saturation"
            ),
            "hybrid_candidate": (
                "fast-hnsw baseline dense plus the same recorded Semble 0.7.0 BM25 "
                "and reconstructed ranking stages"
            ),
            "delta_scope": (
                "No delta index is present; exclusions only model deterministic "
                "filtered baseline shadow scenarios"
            ),
        },
        "ranking": {
            "top_k": TOP_K,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
            "rrf_k": _RRF_K,
            "file_coherence_boost_fraction": _FILE_COHERENCE_BOOST_FRAC,
            "file_saturation_threshold": _FILE_SATURATION_THRESHOLD,
            "file_saturation_decay": _FILE_SATURATION_DECAY,
        },
        "vectors": {
            "file": "vectors.f32le",
            "rows": len(chunks),
            "columns": MODEL_DIMENSION,
            "bytes": len(corpus_bytes),
            "dtype": "float32-le",
            "sha256": sha256_bytes(corpus_bytes),
        },
        "query_vectors": {
            "file": "queries.f32le",
            "rows": len(query_definitions),
            "columns": MODEL_DIMENSION,
            "bytes": len(query_bytes),
            "dtype": "float32-le",
            "sha256": sha256_bytes(query_bytes),
        },
        "chunks": chunk_metadata,
        "queries": query_metadata,
        "shadow_sets": shadow_sets,
        "timings_ms": {
            "corpus_identity": corpus_hash_us / 1_000.0,
            "model_load": model_load_us / 1_000.0,
            "semble_index_build": semble_index_us / 1_000.0,
            "direct_cosine_parity": direct_cosine_us / 1_000.0,
            "fixture_generation_total": total_ms,
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "semble_distribution": distribution_version,
            "numpy": np.__version__,
            "vicinity": importlib.metadata.version("vicinity"),
            "orjson": importlib.metadata.version("orjson"),
            "pathspec": importlib.metadata.version("pathspec"),
            "semble_grammars": importlib.metadata.version("semble-grammars"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "pythonhashseed": os.environ["PYTHONHASHSEED"],
            "hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
        },
    }
    encoded_fixture = (
        json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    write_atomic(output_dir / "fixture.json", encoded_fixture)
    print(
        json.dumps(
            {
                "fixture": str(output_dir / "fixture.json"),
                "chunks": len(chunks),
                "queries": len(query_definitions),
                "corpus_sha": corpus_identity["git_sha"],
                "semble_sha": SEMBLE_GIT_SHA,
                "model": MODEL_IDENTIFIER,
                "dimension": MODEL_DIMENSION,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"fixture generation failed: {error}", file=sys.stderr)
        raise
