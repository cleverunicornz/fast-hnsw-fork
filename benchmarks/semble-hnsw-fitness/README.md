# Semble / HNSW fitness benchmark

This is an offline benchmark harness. It does not integrate HNSW into Semble,
change Semble's embedding model, or add Semble-specific types to the
`fast-hnsw` production API.

## Pinned inputs

- Semble: `cleverunicornz/semble@v0.7.1`, peeled commit
  `21d885145c8724b94122fa0718988b0c7bf8e902`.
- Model: `minishlab/potion-code-16M-v2` at revision
  `e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b`, 256 dimensions.
- Corpus: `cleverunicornz/yeet-code` commit
  `951dd74fd6cdbe050cb451dc9ab0448836728dbb`.
- Queries: the 36 grounded definitions in `queries.json`, copied byte-for-byte
  from the infrastructure fixture. Its required SHA-256 is
  `0c94e0d1995fd40e03c8cb6ef1835667f959748acf81fa3854fc6d9f9c26f89d`.
- Primary machine: `cvu-agent-code-x64`, production `r3-32` shape (4 vCPU,
  32 GiB RAM, 100 GiB local NVMe, GBP 0.1139/hour ex VAT).

`generate_fixture.py` uses the runner's installed Semble. Before importing its
benchmark behavior, it hashes the installed source files that define indexing,
exact dense retrieval, exclusions, RRF, boosting, and reranking. Every hash must
match the owned release above. A package that merely reports version `0.7.1`
but contains different code is rejected.

The `cvu-agent-code-x64` image is intentionally required to pre-provision that
Semble release and the pinned model snapshot. The workflow does not install or
download either one and sets `HF_HUB_OFFLINE=1`; fixture generation is the
authoritative enforcement point for distribution version, owned source hashes,
model revision and file hashes, model dimension, and model-output identity.

## Controls

The generator constructs a fresh code-only Semble index from a clean corpus
checkout, bypassing cache reuse. It writes chunk and query vectors as compact
row-major little-endian `f32` matrices and writes identities, controls, BM25
ranks, and the ranking recipe to JSON.

- Dense control: Semble exact cosine top-50 over its baseline vectors. BM25 is
  absent from this comparison.
- Dense candidate: fast-hnsw cosine HNSW top-50 over those exact bytes.
- Hybrid control: Semble exact dense plus Semble BM25, weighted RRF, query
  boosts, path penalties, and file saturation.
- Hybrid candidate: HNSW replaces only the baseline dense list. The recorded
  Semble BM25 list and data-derived Semble ranking recipe are unchanged.
- Delta: no delta index is constructed. Deterministic exclusions are used only
  for the filtered baseline scenarios.

The Rust harness first reconstructs every exact-dense hybrid control and
requires exact chunk order plus a `1e-10` score tolerance. That gate catches a
ranking contract mismatch before any HNSW result can be reported.

Fixture generation also requires the dense index to be exactly
`semble.index.dense.SelectableBasicBackend`. For query `y01`, it decodes the
same little-endian bytes written to the fixture, performs direct brute-force
cosine against every row, and requires identical top-50 rank order. Raw
scores may differ by at most `5e-4` because Semble normalizes the model's
`float16` query before the fixture exports that effective vector as `float32`.
Each shadow set independently
serializes path membership and must exactly match Semble's
`indices_for_paths(...)` oracle before filtered controls are recorded.

## Measurements

Separate child processes build sequential, 2-worker, 3-worker, and 4-worker
graphs with `M=32`, `M0=64`, `ef_construction=400`, simple reverse pruning,
seed `20260905`, heuristic selection enabled, extend-candidates disabled, and
keep-pruned enabled. A child records graph build, compact persistence, mmap
open, query evaluation, artifact bytes/hash, graph statistics, and Linux
`VmHWM` whole-child peak RSS. The high-water mark includes fixture residency,
graph construction, persistence, mmap validation, and mmap query page faults;
it is not construction-only RSS. A graph becomes ready only after the
persisted graph is reopened and all checks finish; exact search is never a
candidate fallback.

The sequential graph is byte-reproducible from the fixed seed and insertion
order. The library's parallel builder deliberately does not promise identical
edge order across scheduler interleavings. Those rows still pin inputs,
parameters, seeded node levels, and worker count, and the receipt hashes the
exact graph whose quality was measured.

Each graph is queried at `ef=50,100,200,400,800`. The JSON contains every
query's dense Recall@1/@5/@10 against exact dense, plus aggregate mmap search
latency. Hybrid metrics include target-file Recall@1/@5/@10 and MRR@10 for both
Semble control and HNSW candidate. Control agreement reports prefix overlap and
the reciprocal rank of Semble's top-1 result in the candidate top-10.

Filtered search uses nested 0-, 10-, and 50-file shadow sets selected
deterministically from exact-dense neighborhoods. It compares HNSW with exact
cosine over the same eligible chunk IDs, fails on any excluded emission, and
records rejected nodes observed during traversal. Filtered latency uses the
same warmup and repeated timing counts as dense latency. A nonempty short
result is quality evidence, not a harness failure: every query records
`accepted_returned`, and Recall@k keeps the exact eligible top-k as its
denominator. A small unit test also locks the graph property that rejected
nodes can relay navigation without entering the result heap.

BM25 generation and current Semble hybrid timings remain recorded separately
from HNSW traversal and Rust reconstruction timings. The summary does not add
independently sampled phases and label the result as direct end-to-end latency.

## Local validation

Run from the `fast-hnsw-fork` root with the owned Semble 0.7.1 environment
already active:

```bash
git clone https://github.com/cleverunicornz/yeet-code.git benchmark-corpus
git -C benchmark-corpus checkout --detach 951dd74fd6cdbe050cb451dc9ab0448836728dbb
test "$(git -C benchmark-corpus rev-parse HEAD)" = "951dd74fd6cdbe050cb451dc9ab0448836728dbb"
test -z "$(git -C benchmark-corpus status --porcelain=v1 --untracked-files=all)"

export PYTHONHASHSEED=0
export SEMBLE_MODEL_NAME=minishlab/potion-code-16M-v2
export HF_HUB_OFFLINE=1
python3 benchmarks/semble-hnsw-fitness/generate_fixture.py \
  --corpus benchmark-corpus \
  --queries benchmarks/semble-hnsw-fitness/queries.json \
  --output-dir benchmark-artifacts/fixture

cargo test --locked -p semble-hnsw-fitness
cargo run --locked --release -p semble-hnsw-fitness -- \
  run \
  --fixture-dir benchmark-artifacts/fixture \
  --output-dir benchmark-artifacts \
  --query-repeats 25 \
  --warmup 3
```

The expensive run is deliberately outside unit tests. Successful output is one
`benchmark-artifacts/receipt.json`, one concise `summary.md`, four mmap-ready
HNSW artifacts, and the generated fixture directory.

## Workflow dispatch

```bash
gh workflow run semble-hnsw-fitness.yml \
  --repo cleverunicornz/fast-hnsw-fork \
  --ref benchmark-semble-hnsw-fitness
gh run list \
  --repo cleverunicornz/fast-hnsw-fork \
  --workflow semble-hnsw-fitness.yml \
  --limit 1
```

The manual workflow is bound to `cvu-agent-code-x64`, uses
`secrets.GH_PAT_REPOS` only for the exact corpus checkout, enforces 4 exposed
CPUs and at least 30 GiB RAM, appends the Markdown result to
`GITHUB_STEP_SUMMARY`, and uploads the receipt, fixture, and HNSW files. Receipt
gate booleans are computed from oracle evidence, checksums, build artifacts,
and per-query records; no successful gate is emitted as an unconditional
literal.
