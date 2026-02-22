"""
Benchmark charts: pure-Rust HNSW vs. hnsw_rs and hnsw v0.11
============================================================
Data source
-----------
Reads `figures/compare.jsonl` produced by `cargo bench --bench compare`.
Each line is one JSON object covering one (workload × library) combination:

    {"n":1000,"dim":32,"lib":"ours","ins_per_s":18702.1,...}

Libraries present in the file:
    ours      — this repository
    hnsw_rs   — v0.3.3, Jean-Pierre Both
    hnsw_v0   — v0.11.0, Geordon Worley (rust-cv)

Run:
    cargo bench --bench compare   # produces figures/compare.jsonl
    python3 figures/plot_benchmarks.py
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

plt.style.use("seaborn-v0_8-whitegrid")

HERE       = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(HERE, "compare.jsonl")

# ── Palette ───────────────────────────────────────────────────────────────────
LIB_META = {
    "ours":    {"color": "#1565C0", "label": "ours (this repo)",           "ls": "-",  "marker": "o"},
    "hnsw_rs": {"color": "#BF360C", "label": "hnsw_rs v0.3.3",            "ls": "--", "marker": "s"},
    "hnsw_v0": {"color": "#2E7D32", "label": "hnsw v0.11 (rust-cv)",      "ls": ":",  "marker": "^"},
}
ALPHA = 0.88

# ── Load data from JSONL ──────────────────────────────────────────────────────

def load_data(path: str):
    """
    Returns:
      workloads  – ordered list of axis label strings per unique (n, dim)
      w_short    – short workload labels like "1k/32"
      libs       – ordered list of library names found in the file
      ins        – {lib: [ins_per_s, ...]}
      qps        – {ef: {lib: [qps, ...]}}
      recall     – {ef: {lib: [recall_pct, ...]}}
    """
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: {path} not found.\n"
            "Run 'cargo bench --bench compare' first to generate it."
        )

    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Collect ordered unique (n, dim) pairs and library names.
    seen_wl: list[tuple] = []
    seen_lib: list[str]  = []
    for rec in records:
        key = (rec["n"], rec["dim"])
        if key not in seen_wl:
            seen_wl.append(key)
        if rec["lib"] not in seen_lib:
            seen_lib.append(rec["lib"])

    def label(n, dim):
        nl = f"{n // 1000}k" if n >= 1000 else str(n)
        return f"n={nl}\ndim={dim}"

    def short(n, dim):
        nl = f"{n // 1000}k" if n >= 1000 else str(n)
        return f"{nl}/{dim}"

    workloads = [label(n, d) for n, d in seen_wl]
    w_short   = [short(n, d) for n, d in seen_wl]

    # Index by (n, dim, lib)
    by_key: dict[tuple, dict] = {}
    for rec in records:
        by_key[(rec["n"], rec["dim"], rec["lib"])] = rec

    def series(lib: str, field: str) -> list:
        return [by_key[(n, d, lib)][field] for n, d in seen_wl]

    ins = {lib: series(lib, "ins_per_s") for lib in seen_lib}

    efs = [50, 200, 500]
    qps    = {ef: {lib: series(lib, f"ef{ef}_qps")        for lib in seen_lib} for ef in efs}
    recall = {ef: {lib: series(lib, f"ef{ef}_recall_pct") for lib in seen_lib} for ef in efs}

    return workloads, w_short, seen_lib, ins, qps, recall


WORKLOADS, W_SHORT, LIBS, INS, QPS, RECALL = load_data(JSONL_PATH)

# ── Helpers ───────────────────────────────────────────────────────────────────

def legend_handles(libs=None):
    if libs is None:
        libs = LIBS
    return [
        mpatches.Patch(
            color=LIB_META[lib]["color"], alpha=ALPHA,
            label=LIB_META[lib]["label"]
        )
        for lib in libs
        if lib in LIB_META
    ]

def _fmt_k(v):
    if v >= 10_000: return f"{v/1000:.0f}k"
    if v >= 1_000:  return f"{v/1000:.1f}k"
    return f"{v:.0f}"

def grouped_bars(ax, series_list, labels, ylabel, title):
    """
    series_list: list of (data_values, color, bar_label)
    Each data_values is a list aligned with labels.
    """
    n_series = len(series_list)
    x        = np.arange(len(labels))
    w        = 0.72 / n_series
    offsets  = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * w

    for (vals, color, blabel), off in zip(series_list, offsets):
        bars = ax.bar(x + off, vals, w, color=color, alpha=ALPHA, label=blabel, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.04,
                    _fmt_k(val), ha="center", va="bottom", fontsize=6.5,
                    color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Insert throughput
# ─────────────────────────────────────────────────────────────────────────────

def fig_insert():
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)

    series = [
        (INS[lib], LIB_META[lib]["color"], LIB_META[lib]["label"])
        for lib in LIBS if lib in LIB_META
    ]
    grouped_bars(ax, series, WORKLOADS, "Vectors inserted / second  (higher = better)",
                 "Insert Throughput")

    ax.legend(handles=legend_handles(), fontsize=9, loc="upper right")
    fig.suptitle("M=16 · ef_construction=200 · single-threaded · release build",
                 fontsize=8, color="grey", y=1.01)
    fig.tight_layout()
    path = os.path.join(HERE, "fig1_insert_throughput.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Search throughput
# ─────────────────────────────────────────────────────────────────────────────

def fig_search():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=150,
                             sharey=False, constrained_layout=True)
    for ax, ef in zip(axes, [50, 200, 500]):
        series = [
            (QPS[ef][lib], LIB_META[lib]["color"], LIB_META[lib]["label"])
            for lib in LIBS if lib in LIB_META
        ]
        grouped_bars(ax, series, WORKLOADS,
                     "Queries / second  (higher = better)" if ef == 50 else "",
                     f"Search QPS  (ef = {ef})")

    axes[0].legend(handles=legend_handles(), fontsize=8, loc="upper right")
    fig.suptitle(
        "Search throughput at three beam widths · M=16 · ef_construction=200 · single-threaded",
        fontsize=9, color="grey")
    path = os.path.join(HERE, "fig2_search_throughput.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Recall@10
# ─────────────────────────────────────────────────────────────────────────────

def fig_recall():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=150,
                             sharey=True, constrained_layout=True)
    for ax, ef in zip(axes, [50, 200, 500]):
        series = [
            (RECALL[ef][lib], LIB_META[lib]["color"], LIB_META[lib]["label"])
            for lib in LIBS if lib in LIB_META
        ]
        grouped_bars(ax, series, WORKLOADS,
                     "Recall@10  (%)" if ef == 50 else "",
                     f"Recall@10  (ef = {ef})")
        ax.set_ylim(30, 110)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))

    axes[0].legend(handles=legend_handles(), fontsize=8, loc="lower right")
    fig.suptitle("Recall@10 · ground truth = brute-force L2", fontsize=9, color="grey")
    path = os.path.join(HERE, "fig3_recall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Recall vs. QPS tradeoff  (one panel per library)
# ─────────────────────────────────────────────────────────────────────────────

def fig_tradeoff():
    EFS      = [50, 200, 500]
    N_WL     = len(WORKLOADS)
    WL_COLORS = ["#1565C0", "#00796B", "#6A1B9A", "#E65100", "#37474F"][:N_WL]

    fig, axes = plt.subplots(1, len(LIBS), figsize=(7 * len(LIBS), 5.5), dpi=150,
                              constrained_layout=True)
    if len(LIBS) == 1:
        axes = [axes]

    for ax, lib in zip(axes, LIBS):
        meta = LIB_META.get(lib, {"color": "grey", "label": lib, "marker": "o"})
        for wi, (wl_short, color) in enumerate(zip(W_SHORT, WL_COLORS)):
            xs = [QPS[ef][lib][wi]    for ef in EFS]
            ys = [RECALL[ef][lib][wi] for ef in EFS]
            ax.plot(xs, ys, marker=meta["marker"], color=color,
                    linewidth=1.8, markersize=6, label=wl_short, zorder=4)
            for ef, x, y in zip(EFS, xs, ys):
                offset = (-14, 5) if ef == EFS[0] else (4, -12) if ef == EFS[-1] else (4, 5)
                ax.annotate(f"ef={ef}", xy=(x, y), xytext=offset,
                            textcoords="offset points", fontsize=6.5,
                            color=color, alpha=0.85)

        ax.set_xscale("log")
        ax.set_xlabel("Search throughput  (queries / second)", fontsize=10)
        ax.set_ylabel("Recall@10  (%)", fontsize=10)
        ax.set_title(meta["label"], fontsize=11, fontweight="bold")
        ax.set_ylim(30, 106)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))
        ax.legend(title="workload\n(n / dim)", fontsize=8, title_fontsize=8,
                  loc="lower right")
        ax.tick_params(labelsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Recall vs. Throughput tradeoff  (ef ∈ {50, 200, 500}) · "
        "M=16 · ef_construction=200 · K=10",
        fontsize=9, color="grey")
    path = os.path.join(HERE, "fig4_recall_vs_qps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Speedup heatmap  (ours vs. each other library)
# ─────────────────────────────────────────────────────────────────────────────

def fig_speedup_heatmap():
    other_libs = [lib for lib in LIBS if lib != "ours"]
    metrics    = ["Insert"] + [f"Search ef={ef}" for ef in [50, 200, 500]]
    row_labels = []
    speedup_rows = []

    for olib in other_libs:
        olabel = LIB_META.get(olib, {}).get("label", olib)
        for metric_label, ef_or_none in zip(metrics, [None, 50, 200, 500]):
            row_labels.append(f"vs {olabel}\n{metric_label}")
            row = []
            for wi in range(len(WORKLOADS)):
                if ef_or_none is None:
                    ratio = INS["ours"][wi] / INS[olib][wi]
                else:
                    ratio = QPS[ef_or_none]["ours"][wi] / QPS[ef_or_none][olib][wi]
                row.append(ratio)
            speedup_rows.append(row)

    speedup_matrix = np.array(speedup_rows)
    log_mat        = np.log2(speedup_matrix)
    vmax           = max(abs(log_mat).max() + 0.1, 1.0)

    n_rows = len(row_labels)
    fig, ax = plt.subplots(figsize=(11, 0.65 * n_rows + 1.5), dpi=150,
                            constrained_layout=True)
    im = ax.imshow(log_mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(WORKLOADS)))
    ax.set_xticklabels(W_SHORT, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Workload  (n vectors / dimension)", fontsize=10)

    for r in range(n_rows):
        for c in range(len(WORKLOADS)):
            s  = speedup_matrix[r, c]
            ls = log_mat[r, c]
            txt = f"▲ {s:.2f}×" if s >= 1.0 else f"▼ {1/s:.2f}×"
            fc  = "white" if abs(ls) > 0.5 else "black"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=8.5, color=fc, fontweight="bold")

    # Draw horizontal separators between library blocks
    n_metrics = len(metrics)
    for i in range(1, len(other_libs)):
        ax.axhline(i * n_metrics - 0.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(
        "log₂(ours / other)\n▶ blue = ours faster  ▶ red = ours slower",
        fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(
        "Speedup  (ours vs. each competitor) — ▲ blue = ours faster, ▼ red = ours slower",
        fontsize=11, fontweight="bold", pad=8)
    path = os.path.join(HERE, "fig5_speedup_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – Before / after optimisation stages
#
# v0 / v1 / Heuristic series are historical snapshots (fixed).
# v2 (Simple default), hnsw_rs and hnsw_v0 are taken from live JSONL data.
# ─────────────────────────────────────────────────────────────────────────────

def fig_before_after():
    # Historical data — fixed development snapshots, not re-measured each run.
    INS_V0   = [ 4884,  1588, 2858,  865,  616]
    INS_V1   = [ 6462,  1786, 3960,  953,  761]
    INS_HEUR = [ 8271,  2073, 5014, 1085,  843]

    REC_V1   = [100.0, 100.0, 100.0, 96.7, 78.7]
    REC_HEUR = [100.0, 100.0, 100.0, 96.6, 78.7]

    # Live data from compare.jsonl
    INS_V2     = INS["ours"]
    INS_RS     = INS.get("hnsw_rs", [0] * len(WORKLOADS))
    INS_V0_EXT = INS.get("hnsw_v0", [0] * len(WORKLOADS))
    REC_V2     = RECALL[200]["ours"]
    REC_RS     = RECALL[200].get("hnsw_rs", [0] * len(WORKLOADS))
    REC_V0_EXT = RECALL[200].get("hnsw_v0", [0] * len(WORKLOADS))

    C0    = "#90A4AE"
    C1    = "#64B5F6"
    C2    = LIB_META["ours"]["color"]
    CHEUR = "#F39C12"
    C_RS  = LIB_META["hnsw_rs"]["color"]
    C_V0  = LIB_META["hnsw_v0"]["color"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.5), dpi=150,
                                    constrained_layout=True)
    x    = np.arange(len(WORKLOADS))
    w    = 0.13
    offs = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * w

    for vals, color, label, off in [
        (INS_V0,     C0,    "v0 naïve",                                   offs[0]),
        (INS_V1,     C1,    "v1 VecStore / VisitedTracker / …",           offs[1]),
        (INS_V2,     C2,    "v2 PruneStrategy::Simple (default) ★",      offs[2]),
        (INS_HEUR,   CHEUR, "PruneStrategy::Heuristic (+recall, slower)", offs[3]),
        (INS_RS,     C_RS,  "hnsw_rs v0.3.3",                             offs[4]),
        (INS_V0_EXT, C_V0,  "hnsw v0.11 (rust-cv)",                       offs[5]),
    ]:
        bars = ax1.bar(x + off, vals, w, color=color, alpha=0.88, label=label, zorder=3)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.025,
                     _fmt_k(val), ha="center", va="bottom", fontsize=6,
                     color=color, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(WORKLOADS, fontsize=9)
    ax1.set_ylabel("Vectors inserted / second  (higher = better)", fontsize=10)
    ax1.set_title("Insert Throughput — Optimisation Stages",
                  fontsize=11, fontweight="bold")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=7.5, loc="upper right")

    for vals, color, label, off in [
        (REC_V1,     C1,    "v1",                                          offs[1]),
        (REC_V2,     C2,    "PruneStrategy::Simple (default) ★",          offs[2]),
        (REC_HEUR,   CHEUR, "PruneStrategy::Heuristic",                    offs[3]),
        (REC_RS,     C_RS,  "hnsw_rs v0.3.3",                              offs[4]),
        (REC_V0_EXT, C_V0,  "hnsw v0.11 (rust-cv)",                        offs[5]),
    ]:
        bars = ax2.bar(x + off, vals, w, color=color, alpha=0.88, label=label, zorder=3)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=6,
                     color=color, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(WORKLOADS, fontsize=9)
    ax2.set_ylabel("Recall@10  (%) at ef=200  (higher = better)", fontsize=10)
    ax2.set_title("Recall@10 at ef=200 — Quality Across Strategies",
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(60, 108)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(
        "Default PruneStrategy::Simple beats both competitors on insert speed  ·  "
        "hnsw v0.11 shows higher recall at large n (better graph construction quality)",
        fontsize=8, color="grey")
    path = os.path.join(HERE, "fig6_before_after.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 – Recall vs QPS: all three on one chart, colour = workload
# ─────────────────────────────────────────────────────────────────────────────

def fig_all_tradeoff():
    """
    Single Recall-vs-QPS chart with all three libraries overlaid.
    Each workload gets its own colour; each library gets a different linestyle.
    """
    EFS      = [50, 200, 500]
    N_WL     = len(WORKLOADS)
    WL_COLORS = ["#1565C0", "#00796B", "#6A1B9A", "#E65100", "#37474F"][:N_WL]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150, constrained_layout=True)

    for wi, (wl_short, color) in enumerate(zip(W_SHORT, WL_COLORS)):
        for lib in LIBS:
            if lib not in LIB_META:
                continue
            meta = LIB_META[lib]
            xs = [QPS[ef][lib][wi]    for ef in EFS]
            ys = [RECALL[ef][lib][wi] for ef in EFS]
            ax.plot(xs, ys,
                    marker=meta["marker"], linestyle=meta["ls"],
                    color=color, linewidth=1.6, markersize=5,
                    alpha=0.85, zorder=4,
                    label=f"{wl_short} / {meta['label']}" if wi == 0 else "_nolegend_")

    # Build a two-part legend: workloads (coloured lines) + libraries (linestyles)
    wl_handles  = [mpatches.Patch(color=c, label=s)
                   for s, c in zip(W_SHORT, WL_COLORS)]
    lib_handles = [plt.Line2D([0], [0],
                              linestyle=LIB_META[lib]["ls"],
                              marker=LIB_META[lib]["marker"],
                              color="black", linewidth=1.5, markersize=6,
                              label=LIB_META[lib]["label"])
                   for lib in LIBS if lib in LIB_META]

    leg1 = ax.legend(handles=wl_handles,  title="Workload (n/dim)",
                     fontsize=8, title_fontsize=8, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=lib_handles, title="Library",
              fontsize=8, title_fontsize=8, loc="lower right")

    ax.set_xscale("log")
    ax.set_xlabel("Search throughput  (queries / second)", fontsize=11)
    ax.set_ylabel("Recall@10  (%)", fontsize=11)
    ax.set_ylim(30, 107)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_title(
        "Recall vs. Throughput — all three libraries (ef ∈ {50, 200, 500})\n"
        "M=16 · ef_construction=200 · K=10",
        fontsize=10, fontweight="bold")

    path = os.path.join(HERE, "fig7_all_tradeoff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Reading {JSONL_PATH} …")
    print(f"  workloads : {W_SHORT}")
    print(f"  libraries : {LIBS}")
    print("Generating benchmark figures …")
    fig_insert()
    fig_search()
    fig_recall()
    fig_tradeoff()
    fig_speedup_heatmap()
    fig_before_after()
    fig_all_tradeoff()
    print("Done.")
