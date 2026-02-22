"""
plot_persist.py – generate persistence benchmark figures.

Data source
-----------
Reads `figures/persist.csv` produced by `cargo bench --bench persist`.
The CSV has the header:
    workload,type,file_bytes,save_us,load_us,mmap_us

Run:
    cargo bench --bench persist   # produces figures/persist.csv
    python3 figures/plot_persist.py

Figures written to figures/:
  fig7_save_throughput.png   – save MB/s by workload and index type
  fig8_mmap_speedup.png      – mmap-load × speedup vs owned load
  fig9_load_latency.png      – owned vs mmap load time (log scale)
  fig10_file_sizes.png       – file size vs n for each index type
"""

import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "persist.csv")

# ── Load data from CSV ────────────────────────────────────────────────────────

def load_data(path: str) -> dict:
    """
    Returns DATA[workload_key][type_label] = (file_bytes, save_us, load_us, mmap_us).
    """
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: {path} not found.\n"
            "Run 'cargo bench --bench persist' first to generate it."
        )

    data: dict = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            wl  = row["workload"]
            typ = row["type"]
            tup = (
                int(row["file_bytes"]),
                int(row["save_us"]),
                int(row["load_us"]),
                int(row["mmap_us"]),
            )
            data.setdefault(wl, {})[typ] = tup
    return data


DATA = load_data(CSV_PATH)

# ── Derive ordered workload / type lists from the data ───────────────────────
# Preserve insertion order (Python 3.7+) which matches the benchmark order.

WL_KEYS = list(DATA.keys())   # e.g. ["1000×32", "1000×128", …]
TYPES   = list(next(iter(DATA.values())).keys())   # e.g. ["hnsw", "+u32", …]

# Pretty labels for axis ticks
def wl_short(wk: str) -> str:
    """'1000×32' → 'n=1k/32'"""
    n_str, dim_str = wk.replace("×", "x").split("x")
    n = int(n_str)
    n_label = f"{n // 1000}k" if n >= 1000 else str(n)
    return f"n={n_label}/{dim_str}"

WORKLOAD_LABELS = [wl_short(wk) for wk in WL_KEYS]

TYPE_LABELS = {
    "hnsw":      "Bare Hnsw\n(no payload)",
    "+u32":      "+u32\n(class label)",
    "+String":   "+String\n(text tag)",
    "+Vec<f32>": "+Vec<f32>\n(32-d secondary)",
    "paired":    "PairedIndex\n(2 × HNSW)",
}
COLORS = {
    "hnsw":      "#1565C0",
    "+u32":      "#1976D2",
    "+String":   "#F39C12",
    "+Vec<f32>": "#E67E22",
    "paired":    "#BF360C",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def mb(b: int) -> float:
    return b / (1024 ** 2)

def mbs(b: int, us: int) -> float:
    return mb(b) / (us / 1e6)

# ── Fig 7 · Save throughput (MB/s) ───────────────────────────────────────────

def fig_save_throughput():
    fig, ax = plt.subplots(figsize=(13, 5), dpi=150, constrained_layout=True)

    x       = np.arange(len(WL_KEYS))
    n_types = len(TYPES)
    w       = 0.15
    offsets = np.linspace(-(n_types - 1) / 2, (n_types - 1) / 2, n_types) * w

    for i, typ in enumerate(TYPES):
        vals = [mbs(DATA[wk][typ][0], DATA[wk][typ][1]) for wk in WL_KEYS]
        bars = ax.bar(x + offsets[i], vals, w,
                      color=COLORS[typ], alpha=0.88,
                      label=TYPE_LABELS[typ].replace("\n", " "), zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.025,
                    f"{v:.0f}", ha="center", va="bottom",
                    fontsize=6.5, color=COLORS[typ], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOAD_LABELS, fontsize=9)
    ax.set_ylabel("Save throughput  (MB/s, higher = better)", fontsize=10)
    ax.set_title(
        "Fig 7 — Save Throughput by Index Type and Workload\n"
        "Fixed-payload types (no payload, u32, paired) achieve higher MB/s "
        "because they write sequentially without an offset table.",
        fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=5, loc="upper right")

    path = os.path.join(HERE, "fig7_save_throughput.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ── Fig 8 · mmap speedup heatmap ─────────────────────────────────────────────

def fig_mmap_speedup():
    speedups = np.zeros((len(TYPES), len(WL_KEYS)))
    for j, wk in enumerate(WL_KEYS):
        for i, typ in enumerate(TYPES):
            fb, su, lu, mu = DATA[wk][typ]
            speedups[i, j] = lu / mu

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150, constrained_layout=True)
    im = ax.imshow(speedups, cmap="Blues", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="mmap speedup  (load_time / mmap_time)", pad=0.01)

    ax.set_xticks(range(len(WL_KEYS)))
    ax.set_xticklabels(WORKLOAD_LABELS, fontsize=9)
    ax.set_yticks(range(len(TYPES)))
    ax.set_yticklabels(
        [TYPE_LABELS[t].replace("\n", " ") for t in TYPES], fontsize=9)

    for i in range(len(TYPES)):
        for j in range(len(WL_KEYS)):
            v = speedups[i, j]
            ax.text(j, i, f"{v:.0f}×",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if v > speedups.max() * 0.6 else "#1a1a2e")

    ax.set_title(
        "Fig 8 — mmap-Load Speedup vs Owned Load\n"
        "Vector bytes are NOT read during mmap load — they are faulted in lazily on first access.\n"
        "Speedup = owned_load_time / mmap_load_time.  Higher = faster mmap.",
        fontsize=10, fontweight="bold")

    path = os.path.join(HERE, "fig8_mmap_speedup.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ── Fig 9 · Load latency: owned vs mmap ──────────────────────────────────────

def fig_load_latency():
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150, constrained_layout=True)

    for ax_idx, (ax, metric_idx, title) in enumerate(zip(
        axes,
        [2, 3],   # index into (file_bytes, save_us, load_us, mmap_us)
        ["Owned load  (reads all bytes into RAM)",
         "mmap load  (maps file; vector bytes NOT read until accessed)"],
    )):
        x       = np.arange(len(WL_KEYS))
        n_types = len(TYPES)
        w       = 0.15
        offsets = np.linspace(-(n_types - 1) / 2, (n_types - 1) / 2, n_types) * w

        for i, typ in enumerate(TYPES):
            vals = [DATA[wk][typ][metric_idx] / 1_000 for wk in WL_KEYS]  # µs → ms
            bars = ax.bar(x + offsets[i], vals, w,
                          color=COLORS[typ], alpha=0.88,
                          label=TYPE_LABELS[typ].replace("\n", " "), zorder=3)
            for bar, v in zip(bars, vals):
                if v < 0.5:
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.025,
                        f"{v:.0f}" if v >= 1 else f"{v:.1f}",
                        ha="center", va="bottom",
                        fontsize=6, color=COLORS[typ], fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOAD_LABELS, fontsize=9)
        ax.set_ylabel("Load time  (ms, lower = better)", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        if ax_idx == 0:
            ax.legend(fontsize=8, ncol=2, loc="upper left")

    fig.suptitle(
        "Fig 9 — Load Latency: Owned vs Memory-Mapped\n"
        "mmap load is dominated by graph deserialization; "
        "owned load additionally reads all vector bytes off disk.",
        fontsize=10, color="#333333")

    path = os.path.join(HERE, "fig9_load_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ── Fig 10 · File sizes ───────────────────────────────────────────────────────

def fig_file_sizes():
    # Identify workloads with dim=128 for the line chart
    wk_128 = [wk for wk in WL_KEYS if wk.endswith("×128")]
    n_vals  = [int(wk.split("×")[0]) for wk in wk_128]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150,
                                    constrained_layout=True)

    # Left: line chart — file size vs n for dim=128
    for typ in TYPES:
        sizes_mb = [mb(DATA[wk][typ][0]) for wk in wk_128]
        ax1.plot(n_vals, sizes_mb, marker="o", linewidth=2,
                 color=COLORS[typ], label=TYPE_LABELS[typ].replace("\n", " "))
        ax1.annotate(f"{sizes_mb[-1]:.1f} MiB",
                     (n_vals[-1], sizes_mb[-1]),
                     textcoords="offset points", xytext=(6, 0),
                     fontsize=8, color=COLORS[typ], fontweight="bold")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(n_vals)
    ax1.set_xticklabels([f"{n // 1000}k" if n >= 1000 else str(n) for n in n_vals])
    ax1.set_xlabel("n (number of vectors, dim=128)", fontsize=10)
    ax1.set_ylabel("File size (MiB)", fontsize=10)
    ax1.set_title("File size vs n  (dim=128, log-log)", fontsize=10, fontweight="bold")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=8)

    # Right: stacked bar breakdown for the largest workload (last in wk_128)
    wk_large    = wk_128[-1]
    base_hnsw   = mb(DATA[wk_large]["hnsw"][0])

    bars_base    = []
    bars_payload = []
    labels       = []

    for typ in TYPES:
        total = mb(DATA[wk_large][typ][0])
        extra = max(total - base_hnsw, 0.0)
        bars_base.append(base_hnsw)
        bars_payload.append(extra)
        labels.append(TYPE_LABELS[typ].replace("\n", " "))

    x  = np.arange(len(TYPES))
    w  = 0.55
    b1 = ax2.bar(x, bars_base,    w, label="Hnsw vectors + graph",
                 color="#1565C0", alpha=0.85, zorder=3)
    b2 = ax2.bar(x, bars_payload, w, bottom=bars_base,
                 label="Payload / 2nd graph overhead",
                 color="#F39C12", alpha=0.88, zorder=3)

    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h / 2,
                 f"{h:.1f}", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")
    for bar, base, extra in zip(b2, bars_base, bars_payload):
        if extra < 0.3:
            continue
        ax2.text(bar.get_x() + bar.get_width() / 2, base + extra / 2,
                 f"+{extra:.1f}", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")

    totals = [a + b for a, b in zip(bars_base, bars_payload)]
    for i, (tot, xp) in enumerate(zip(totals, x)):
        ax2.text(xp, tot + 0.5, f"{tot:.1f} MiB",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    n_large = int(wk_large.split("×")[0])
    dim_large = wk_large.split("×")[1]
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("File size (MiB)", fontsize=10)
    ax2.set_title(
        f"File size breakdown  (n={n_large // 1000}k, dim={dim_large})",
        fontsize=10, fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "Fig 10 — File Sizes\n"
        "Vector data dominates; payload adds modest overhead for fixed types.\n"
        "Vec<f32> and paired indexes have the largest overhead.",
        fontsize=10, color="#333333")

    path = os.path.join(HERE, "fig10_file_sizes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Reading {CSV_PATH} …")
    print(f"  workloads : {WL_KEYS}")
    print(f"  types     : {TYPES}")
    print("Generating persistence benchmark figures …")
    fig_save_throughput()
    fig_mmap_speedup()
    fig_load_latency()
    fig_file_sizes()
    print("Done.")
