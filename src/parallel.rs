//! Multi-threaded index construction (`parallel` feature).
//!
//! Sequential [`Hnsw::insert`](crate::Hnsw::insert) processes one vector at a
//! time, so building a large index is bounded by a single core.  This module
//! builds the same graph with a thread pool.
//!
//! # How it works
//!
//! Construction splits into two phases:
//!
//! 1. **Sequential setup.**  All vectors are copied into the flat store and
//!    every node's layer is drawn from the seeded RNG.  Doing this up front
//!    means the vector store is immutable during phase 2 (so it can be shared
//!    by reference) and the layer assignment is reproducible regardless of how
//!    many threads run later.
//! 2. **Parallel insertion.**  Nodes are inserted concurrently.  Each node's
//!    adjacency list sits behind its own [`RwLock`], so traversals read
//!    neighbour lists while other threads append to unrelated nodes.
//!
//! Node 0 is inserted first and seeds the entry point; every other node then
//! has somewhere to start its descent.
//!
//! # Locking
//!
//! A worker never holds two node locks at once.  The forward edge `q → nb` and
//! the reverse edge `nb → q` are written under separate, sequential lock
//! acquisitions, and pruning `nb` needs only `nb`'s lock because the vector
//! data it reads is immutable.  With no thread ever waiting on a second lock,
//! the usual lock-ordering deadlock cannot arise.
//!
//! # Determinism
//!
//! **A parallel build is not byte-reproducible, even with a fixed seed.**  Node
//! layers are deterministic, but the order in which concurrent inserts reach a
//! shared neighbour decides that neighbour's edge ordering, and therefore which
//! edge pruning discards.  Use [`Builder::build`](crate::Builder::build) with a
//! seed when you need reproducible output.
//!
//! # Speed and recall
//!
//! Measured on 20 000 uniform random vectors, `M=16`, `ef_construction=200`,
//! on a 14-core machine:
//!
//! | threads | build speed-up | recall@10, ef=100 | recall@10, ef=400 |
//! |---------|----------------|-------------------|-------------------|
//! | 1 (sequential) | 1.0×      | 80.4%             | 94.6%             |
//! | 2       | 1.7×           | 76.6%             | 93.8%             |
//! | 4       | 3.8×           | 75.0%             | 92.6%             |
//! | 8       | 6.9×           | 75.4%             | 92.0%             |
//! | 14      | 7.8×           | 74.2%             | 89.6%             |
//!
//! **A parallel build trades a few points of recall for the speed-up.**  A
//! sequential insert searches a graph whose every existing node is fully
//! connected; a concurrent one may search while a dozen of its neighbours are
//! still mid-insert, so it picks from a partially-built neighbourhood and
//! selects slightly worse edges.  The gap is visible even at two threads and
//! widens gently with thread count.
//!
//! The effect is dimension-dependent — at 32 dimensions the same measurement
//! shows 99.8% against 100.0%, essentially parity — and most of it is bought
//! back by raising query-time `ef`, which is far cheaper than the build time
//! saved.  Use a sequential build when recall at a fixed low `ef` is the
//! binding constraint.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::distance::Distance;
use crate::error::{Error, Result};
use crate::heap::DistId;
use crate::hnsw::{
    Config, Edge, GraphStore, Hnsw, PruneStrategy, VecStore, VisitedTracker,
};
use crate::Builder;

/// Adjacency lists behind one lock per node.
///
/// Locking per node rather than per graph is what makes the build scale: two
/// inserts touching different neighbourhoods never contend.
struct ConcurrentGraph {
    nodes: Vec<RwLock<Vec<Vec<Edge>>>>,
}

impl ConcurrentGraph {
    fn new(levels: &[u32], config: &Config) -> Self {
        let nodes = levels
            .iter()
            .map(|&level| {
                let layers = (0..=level as usize)
                    .map(|layer| Vec::with_capacity(config.max_links(layer)))
                    .collect();
                RwLock::new(layers)
            })
            .collect();
        Self { nodes }
    }

    /// Snapshot one node's neighbour ids at `layer`.
    ///
    /// The lock is released before the caller computes any distances, so a
    /// slow metric never blocks another thread's insert.
    fn neighbours(&self, node: usize, layer: usize, out: &mut Vec<u32>) {
        out.clear();
        let guard = self.nodes[node].read().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(edges) = guard.get(layer) {
            out.extend(edges.iter().map(|&(id, _)| id));
        }
    }

    fn into_owned(self) -> Vec<Vec<Vec<Edge>>> {
        self.nodes
            .into_iter()
            .map(|node| node.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner()))
            .collect()
    }
}

/// Per-thread traversal storage, reused across every node a worker handles.
struct Workspace {
    visited: VisitedTracker,
    candidates: std::collections::BinaryHeap<std::cmp::Reverse<DistId>>,
    results: std::collections::BinaryHeap<DistId>,
    out: Vec<DistId>,
    entry: Vec<DistId>,
    neighbour_ids: Vec<u32>,
    selected: Vec<(usize, f32)>,
    discarded: Vec<(usize, f32)>,
}

impl Workspace {
    fn new(node_count: usize, ef: usize) -> Self {
        Self {
            visited: VisitedTracker::new(node_count),
            candidates: std::collections::BinaryHeap::with_capacity(ef * 2 + 1),
            results: std::collections::BinaryHeap::with_capacity(ef + 1),
            out: Vec::with_capacity(ef),
            entry: Vec::with_capacity(ef),
            neighbour_ids: Vec::with_capacity(64),
            selected: Vec::with_capacity(64),
            discarded: Vec::with_capacity(64),
        }
    }
}

/// Greedy best-first search of one layer, writing results closest-first into
/// `workspace.out`.
#[allow(clippy::too_many_arguments)]
fn search_layer<D: Distance>(
    vec_store: &VecStore,
    graph: &ConcurrentGraph,
    metric: &D,
    query: &[f32],
    ef: usize,
    layer: usize,
    workspace: &mut Workspace,
) {
    workspace.visited.begin();
    workspace.candidates.clear();
    workspace.results.clear();

    for &entry in &workspace.entry {
        if workspace.visited.visit(entry.id) {
            workspace.candidates.push(std::cmp::Reverse(entry));
            workspace.results.push(entry);
            if workspace.results.len() > ef {
                workspace.results.pop();
            }
        }
    }

    while let Some(std::cmp::Reverse(candidate)) = workspace.candidates.pop() {
        let Some(worst) = workspace.results.peek().map(|entry| entry.dist) else {
            break;
        };
        if candidate.dist > worst {
            break;
        }

        graph.neighbours(candidate.id, layer, &mut workspace.neighbour_ids);
        for index in 0..workspace.neighbour_ids.len() {
            let neighbour = workspace.neighbour_ids[index] as usize;
            if !workspace.visited.visit(neighbour) {
                continue;
            }
            let distance = metric.distance(query, vec_store.get(neighbour));
            let worst = workspace
                .results
                .peek()
                .map(|entry| entry.dist)
                .unwrap_or(f32::INFINITY);
            if distance < worst || workspace.results.len() < ef {
                let entry = DistId::new(distance, neighbour);
                workspace.candidates.push(std::cmp::Reverse(entry));
                workspace.results.push(entry);
                if workspace.results.len() > ef {
                    workspace.results.pop();
                }
            }
        }
    }

    workspace.out.clear();
    while let Some(entry) = workspace.results.pop() {
        workspace.out.push(entry);
    }
    workspace.out.reverse();
}

/// Algorithm 4 diversity selection over `candidates` (already closest-first),
/// writing the kept entries into `workspace.selected`.
fn select_neighbours<D: Distance>(
    vec_store: &VecStore,
    metric: &D,
    candidates: &[(usize, f32)],
    m: usize,
    keep_pruned: bool,
    selected: &mut Vec<(usize, f32)>,
    discarded: &mut Vec<(usize, f32)>,
) {
    selected.clear();
    discarded.clear();

    if candidates.len() <= m {
        selected.extend_from_slice(candidates);
        return;
    }

    for &(candidate_id, candidate_dist) in candidates {
        if selected.len() >= m {
            break;
        }
        let mut accept = true;
        for &(kept_id, kept_dist) in selected.iter() {
            // Triangle-inequality shortcut, as in the sequential path.
            if kept_dist > 2.0 * candidate_dist {
                continue;
            }
            let between = metric.distance(vec_store.get(candidate_id), vec_store.get(kept_id));
            if between <= candidate_dist {
                accept = false;
                break;
            }
        }
        if accept {
            selected.push((candidate_id, candidate_dist));
        } else if keep_pruned {
            discarded.push((candidate_id, candidate_dist));
        }
    }

    if keep_pruned {
        let needed = m.saturating_sub(selected.len());
        selected.extend(discarded.iter().take(needed).copied());
    }
}

/// Reusable buffers for [`prune_node`], kept per worker.
struct PruneScratch {
    sorted: Vec<(usize, f32)>,
    selected: Vec<(usize, f32)>,
    discarded: Vec<(usize, f32)>,
}

impl PruneScratch {
    fn new() -> Self {
        Self {
            sorted: Vec::new(),
            selected: Vec::new(),
            discarded: Vec::new(),
        }
    }
}

/// Shrink `node`'s over-full connection list back to `m_max`.
///
/// Holds only `node`'s write lock; the vector data the heuristic reads is
/// immutable during the parallel phase.
#[allow(clippy::too_many_arguments)]
fn prune_node<D: Distance>(
    vec_store: &VecStore,
    graph: &ConcurrentGraph,
    metric: &D,
    node: usize,
    layer: usize,
    m_max: usize,
    strategy: PruneStrategy,
    keep_pruned: bool,
    scratch: &mut PruneScratch,
) {
    let mut guard = graph.nodes[node].write().unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(edges) = guard.get_mut(layer) else {
        return;
    };
    if edges.len() <= m_max {
        return;
    }

    match strategy {
        PruneStrategy::Simple => {
            edges.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            edges.truncate(m_max);
        }
        PruneStrategy::Heuristic => {
            scratch.sorted.clear();
            scratch
                .sorted
                .extend(edges.iter().map(|&(id, dist)| (id as usize, dist)));
            scratch.sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            let (sorted, selected, discarded) = (
                &scratch.sorted,
                &mut scratch.selected,
                &mut scratch.discarded,
            );
            select_neighbours(
                vec_store,
                metric,
                sorted,
                m_max,
                keep_pruned,
                selected,
                discarded,
            );
            edges.clear();
            edges.extend(selected.iter().map(|&(id, dist)| (id as u32, dist)));
        }
    }
}

/// Insert one node into the shared graph.
#[allow(clippy::too_many_arguments)]
fn insert_node<D: Distance>(
    vec_store: &VecStore,
    graph: &ConcurrentGraph,
    metric: &D,
    config: &Config,
    entry_point: &RwLock<(usize, usize)>,
    levels: &[u32],
    node: usize,
    workspace: &mut Workspace,
) {
    let node_level = levels[node] as usize;
    let node_vec = vec_store.get(node);

    let (mut current_id, entry_level) = *entry_point.read().unwrap_or_else(|poisoned| poisoned.into_inner());
    if current_id == node {
        return;
    }

    workspace.entry.clear();
    workspace
        .entry
        .push(DistId::new(metric.distance(node_vec, vec_store.get(current_id)), current_id));

    // Phase 1: greedy descent through the layers above this node.
    for layer in (node_level + 1..=entry_level).rev() {
        search_layer(vec_store, graph, metric, node_vec, 1, layer, workspace);
        if workspace.out.is_empty() {
            break;
        }
        std::mem::swap(&mut workspace.entry, &mut workspace.out);
    }
    if let Some(nearest) = workspace.entry.first() {
        current_id = nearest.id;
    }
    let _ = current_id;

    // Phase 2: connect at every layer this node participates in.
    let mut prune_scratch = PruneScratch::new();
    let mut edges: Vec<Edge> = Vec::new();

    for layer in (0..=node_level.min(entry_level)).rev() {
        let m_max = config.max_links(layer);
        search_layer(
            vec_store,
            graph,
            metric,
            node_vec,
            config.ef_construction,
            layer,
            workspace,
        );

        // Exclude this node from its own candidate set. Sequentially it has no
        // in-edges yet and so is unreachable from its own search; under
        // concurrency another thread may already have linked to it, at which
        // point the search finds it and it would select itself as a neighbour.
        let candidates: Vec<(usize, f32)> = workspace
            .out
            .iter()
            .filter(|entry| entry.id != node)
            .map(|entry| (entry.id, entry.dist))
            .collect();

        if config.use_heuristic {
            let (mut selected, mut discarded) = (
                std::mem::take(&mut workspace.selected),
                std::mem::take(&mut workspace.discarded),
            );
            select_neighbours(
                vec_store,
                metric,
                &candidates,
                m_max,
                config.keep_pruned,
                &mut selected,
                &mut discarded,
            );
            workspace.selected = selected;
            workspace.discarded = discarded;
        } else {
            workspace.selected.clear();
            workspace
                .selected
                .extend(candidates.iter().take(m_max).copied());
        }

        edges.clear();
        edges.extend(
            workspace
                .selected
                .iter()
                .map(|&(id, dist)| (id as u32, dist)),
        );

        // Forward edges: one lock acquisition, released before the reverse
        // edges are written, so no worker ever holds two node locks.
        {
            let mut guard = graph.nodes[node].write().unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(list) = guard.get_mut(layer) {
                list.extend(edges.iter().copied());
            }
        }

        // Prune this node's own list too. Sequentially a node always writes its
        // own edges before any reverse edge can reach it, so its list starts at
        // `m_max` and only later reverse edges (which prune) can grow it. Under
        // concurrency a higher-numbered node may insert first and attach
        // reverse edges here beforehand, so the extend above can overshoot
        // `m_max` with nothing else scheduled to trim it.
        prune_node(
            vec_store,
            graph,
            metric,
            node,
            layer,
            m_max,
            config.prune_strategy,
            config.keep_pruned,
            &mut prune_scratch,
        );

        for &(neighbour, distance) in &edges {
            let neighbour = neighbour as usize;
            {
                let mut guard = graph.nodes[neighbour].write().unwrap_or_else(|poisoned| poisoned.into_inner());
                match guard.get_mut(layer) {
                    Some(list) => list.push((node as u32, distance)),
                    None => continue,
                }
            }
            prune_node(
                vec_store,
                graph,
                metric,
                neighbour,
                layer,
                m_max,
                config.prune_strategy,
                config.keep_pruned,
                &mut prune_scratch,
            );
        }

        std::mem::swap(&mut workspace.entry, &mut workspace.out);
    }
}

/// Build an index from `vectors` using a rayon thread pool.
///
/// See the [module docs](self) for the locking scheme and the determinism
/// caveat.
///
/// Fails if the vectors do not all share one dimension, or the configuration
/// is invalid.
pub fn build_parallel<D: Distance>(
    config: Config,
    metric: D,
    vectors: Vec<Vec<f32>>,
    seed: Option<u64>,
) -> Result<Hnsw<D>> {
    config.validate()?;

    if vectors.is_empty() {
        return match seed {
            Some(seed) => Hnsw::new_with_seed(config, metric, seed),
            None => Hnsw::new(config, metric),
        };
    }

    let dim = vectors[0].len();
    for vector in vectors.iter() {
        if vector.len() != dim {
            return Err(Error::DimensionMismatch { expected: dim, actual: vector.len() });
        }
    }
    let count = vectors.len();

    // ── Phase 1 (sequential): vectors and layer assignment ────────────────
    let mut vec_store = VecStore::new(dim, count);
    for vector in vectors {
        vec_store.push(vector);
    }

    let mut rng = match seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => rand::make_rng(),
    };
    let m_l = config.m_l();
    let levels: Vec<u32> = (0..count)
        .map(|_| {
            let uniform: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
            (-uniform.ln() * m_l).floor() as u32
        })
        .collect();

    let graph = ConcurrentGraph::new(&levels, &config);

    // Node 0 seeds the entry point so every other insert has a starting point.
    let entry_point = RwLock::new((0usize, levels[0] as usize));
    let max_level = AtomicUsize::new(levels[0] as usize);

    // ── Phase 2 (parallel): insert the remaining nodes ────────────────────
    let ef = config.ef_construction;
    (1..count).into_par_iter().for_each_init(
        || Workspace::new(count, ef),
        |workspace, node| {
            insert_node(
                &vec_store,
                &graph,
                &metric,
                &config,
                &entry_point,
                &levels,
                node,
                workspace,
            );

            // Promote the entry point if this node reaches a new top layer.
            let node_level = levels[node] as usize;
            if node_level > max_level.load(Ordering::Acquire) {
                let mut guard = entry_point.write().unwrap_or_else(|poisoned| poisoned.into_inner());
                if node_level > guard.1 {
                    *guard = (node, node_level);
                    max_level.store(node_level, Ordering::Release);
                }
            }
        },
    );

    let entry = *entry_point.read().unwrap_or_else(|poisoned| poisoned.into_inner());
    Ok(Hnsw::from_parts(
        config,
        metric,
        vec_store,
        GraphStore::from_owned(graph.into_owned()),
        Some(entry),
        Some(dim),
    ))
}

impl Builder {
    /// Build an index from `vectors` across a rayon thread pool.
    ///
    /// Equivalent to calling [`Builder::build`] and then inserting every
    /// vector, except that insertion runs concurrently.  The resulting graph
    /// is **not** byte-reproducible even with [`Builder::seed`]; see the
    /// [module docs](crate::parallel).
    ///
    /// ```no_run
    /// use fast_hnsw::{Builder, Hnsw};
    /// use fast_hnsw::distance::Euclidean;
    ///
    /// let vectors: Vec<Vec<f32>> = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    /// let index: Hnsw<Euclidean> = Builder::new()
    ///     .m(16)
    ///     .ef_construction(200)
    ///     .build_parallel(Euclidean, vectors).unwrap();
    /// ```
    pub fn build_parallel<D: Distance>(
        self,
        metric: D,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Hnsw<D>> {
        let seed = self.seed_value();
        build_parallel(self.into_config(), metric, vectors, seed)
    }
}
