#!/usr/bin/env python3
"""
plot_bench.py  —  generate figures from figures/bench.jsonl

Quickstart
----------
    cargo bench --bench bench                  # default workloads (≤ 50k)
    cargo bench --bench bench -- --full        # scale up to 1 M
    python3 figures/plot_bench.py

Output files (same directory as this script)
--------------------------------------------
    bench_fig1_insert_throughput.png   — vecs/s per workload (bar)
    bench_fig2_insert_latency.png      — µs/insert per workload (bar)
    bench_fig3_search_qps.png          — QPS per workload × ef (grouped bar)
    bench_fig4_scaling.png             — throughput vs n line chart
                                         (only rendered when ≥ 3 n values share
                                          the same dim, i.e. after --full)
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HERE  = pathlib.Path(__file__).parent
JSONL = HERE / "bench.jsonl"

COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860"]

# ── Load ───────────────────────────────────────────────────────────────────────

def load_data():
    if not JSONL.exists():
        sys.exit(
            f"[error] {JSONL} not found\n"
            "  run:  cargo bench --bench bench\n"
            "  (add -- --full to include workloads up to 1 M)"
        )

    inserts, searches = [], []
    with open(JSONL) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            (inserts if rec["type"] == "insert" else searches).append(rec)

    print(f"Reading {JSONL} …")
    print(f"  insert records : {len(inserts)}")
    print(f"  search records : {len(searches)}")
    return inserts, searches

# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_n(n):
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:g}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:g}k"
    return str(n)

def workload_label(r):
    return f"n={fmt_n(r['n'])}\ndim={r['dim']}"

def save(fig, name):
    path = HERE / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ── Figure 1 — insert throughput (vecs / s) ────────────────────────────────────

def fig_insert_throughput(inserts):
    labels   = [workload_label(r) for r in inserts]
    vecs_s   = [1e6 / r["per_insert_us"] for r in inserts]
    x        = np.arange(len(inserts))
    width    = min(0.6, 4.0 / max(len(inserts), 1))

    fig, ax = plt.subplots(figsize=(max(6, len(inserts) * 1.1), 5))
    bars = ax.bar(x, vecs_s, width=width, color=COLORS[0])
    ax.bar_label(bars, [f"{v:,.0f}" for v in vecs_s], padding=4, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Vectors inserted / second  (higher = better)")
    ax.set_title(
        "Insert throughput — ours (this repo)\n"
        "M=16 · ef_construction=200"
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(0, max(vecs_s) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "bench_fig1_insert_throughput.png")

# ── Figure 2 — insert latency (µs / insert) ────────────────────────────────────

def fig_insert_latency(inserts):
    labels  = [workload_label(r) for r in inserts]
    lat_us  = [r["per_insert_us"] for r in inserts]
    x       = np.arange(len(inserts))
    width   = min(0.6, 4.0 / max(len(inserts), 1))

    fig, ax = plt.subplots(figsize=(max(6, len(inserts) * 1.1), 5))
    bars = ax.bar(x, lat_us, width=width, color=COLORS[1])
    ax.bar_label(
        bars,
        [f"{v:,.0f} µs" if v >= 1 else f"{v:.1f} µs" for v in lat_us],
        padding=4, fontsize=8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("µs per insert  (lower = better)")
    ax.set_title(
        "Insert latency — ours (this repo)\n"
        "M=16 · ef_construction=200"
    )
    ax.set_ylim(0, max(lat_us) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "bench_fig2_insert_latency.png")

# ── Figure 3 — search QPS (grouped by ef) ──────────────────────────────────────

def fig_search_qps(searches):
    # Collect ordered unique keys.
    nd_pairs = list(dict.fromkeys((r["n"], r["dim"]) for r in searches))
    efs      = sorted(set(r["ef"] for r in searches))
    lookup   = {(r["n"], r["dim"], r["ef"]): r["qps"] for r in searches}

    x     = np.arange(len(nd_pairs))
    w     = min(0.8 / max(len(efs), 1), 0.35)
    offsets = np.linspace(-(len(efs)-1)/2, (len(efs)-1)/2, len(efs)) * w

    fig, ax = plt.subplots(figsize=(max(8, len(nd_pairs) * 1.4), 5))
    for i, ef in enumerate(efs):
        vals = [lookup.get((n, d, ef), 0) for n, d in nd_pairs]
        bars = ax.bar(
            x + offsets[i], vals,
            width=w * 0.9, color=COLORS[i % len(COLORS)], label=f"ef={ef}",
        )
        ax.bar_label(
            bars,
            [f"{v:,.0f}" if v > 0 else "" for v in vals],
            padding=3, fontsize=7, rotation=90 if len(nd_pairs) > 4 else 0,
        )

    xlabels = [f"n={fmt_n(n)}\ndim={d}" for n, d in nd_pairs]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Queries per second  (higher = better)")
    ax.set_title(
        "Search throughput — ours (this repo)\n"
        "M=16 · ef_construction=200 · K=10"
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(0, max(r["qps"] for r in searches) * 1.35)
    ax.legend(title="beam width", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "bench_fig3_search_qps.png")

# ── Figure 4 — scaling with n (line chart, only for --full data) ───────────────

def fig_scaling(inserts, searches):
    """
    Rendered only when there are ≥ 3 distinct n values sharing a dim,
    which happens after `cargo bench --bench bench -- --full`.
    """
    # dim=128 insert scaling
    ins128   = sorted((r for r in inserts  if r["dim"] == 128), key=lambda r: r["n"])
    sea_ef50 = sorted((r for r in searches if r["dim"] == 128 and r["ef"] == 50),
                      key=lambda r: r["n"])

    enough_ins = len(set(r["n"] for r in ins128))   >= 3
    enough_sea = len(set(r["n"] for r in sea_ef50)) >= 3

    if not enough_ins and not enough_sea:
        print("  bench_fig4_scaling.png — skipped "
              "(need ≥ 3 n values at dim=128; run with --full)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    def style_ax(ax, xs, title, ylabel):
        use_log = max(xs) / max(min(xs), 1) >= 20
        if use_log:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: fmt_n(int(v)))
            )
        else:
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: fmt_n(int(v)))
            )
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
        )
        ax.set_xlabel("Index size  n")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    # Left: insert throughput vs n
    ax = axes[0]
    if enough_ins:
        ns  = [r["n"] for r in ins128]
        thr = [1e6 / r["per_insert_us"] for r in ins128]
        ax.plot(ns, thr, "o-", color=COLORS[0], lw=2, ms=6)
        ax.fill_between(ns, thr, alpha=0.12, color=COLORS[0])
        style_ax(ax, ns,
                 "Insert throughput scaling\ndim=128 · M=16 · ef_construction=200",
                 "Vectors / second")
    else:
        ax.set_visible(False)

    # Right: search QPS vs n (ef=50)
    ax = axes[1]
    if enough_sea:
        ns  = [r["n"] for r in sea_ef50]
        qps = [r["qps"] for r in sea_ef50]
        ax.plot(ns, qps, "s-", color=COLORS[1], lw=2, ms=6)
        ax.fill_between(ns, qps, alpha=0.12, color=COLORS[1])
        style_ax(ax, ns,
                 "Search QPS scaling\ndim=128 · K=10 · ef=50",
                 "Queries / second")
    else:
        ax.set_visible(False)

    fig.tight_layout()
    save(fig, "bench_fig4_scaling.png")

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    inserts, searches = load_data()
    print("Generating bench figures …")
    fig_insert_throughput(inserts)
    fig_insert_latency(inserts)
    if searches:
        fig_search_qps(searches)
    fig_scaling(inserts, searches)
    print("Done.")
