#!/usr/bin/env python3
"""Fig. S2 — network construction workflow and parameter sensitivity."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from fig_style import (ACCENT_RED, ACCENT_BLUE, GREY, BLACK,
                       panel_letter, save_cns_figure)

# Paths resolve relative to this package; override with RIF_DATA / RIF_DATA2 / RIF_DATA3 / RIF_FIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
D = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "figS"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
CT = ["Stromal", "Epithelial", "T cell"]
grid = pd.read_csv(f"{D}/s2_grid.csv")
met = pd.read_csv(f"{D}/s2_metric.csv")
thr = pd.read_csv(f"{D}/s2_threshold.csv")

fig = plt.figure(figsize=(6.98, 6.05))
gs = GridSpec(3, 2, figure=fig, height_ratios=[0.50, 1.02, 0.92],
              hspace=0.50, wspace=0.42, left=0.075, right=0.93,
              top=0.97, bottom=0.075)

# ---------------- A: workflow schematic ----------------
axA = fig.add_subplot(gs[0, :]); axA.set_axis_off()
axA.set_xlim(0, 1); axA.set_ylim(0, 1)
boxes = [
    ("11 cell types\n149,590 cells", "#E8EEF6"),
    ("MAGIC imputation\nkNN = 5, t = 3", "#DCE6F2"),
    ("Within-group\nPearson → Fisher z", "#DCE6F2"),
    ("Δz = z$_{RIF}$ − z$_{Fertile}$\nsigned top-50 edges", "#F6E3E3"),
    ("Gene score s\n→ z per cell type", "#DCE6F2"),
    ("Stouffer across types\n→ 8 modules", "#E8EEF6"),
]
n = len(boxes); w = 0.128; gap = (1 - n * w) / (n + 1)
y0, hgt = 0.30, 0.46
for i, (txt, col) in enumerate(boxes):
    x = gap + i * (w + gap)
    axA.add_patch(FancyBboxPatch((x, y0), w, hgt, boxstyle="round,pad=0.012,rounding_size=0.025",
                                 fc=col, ec="#8A8A8A", lw=0.5))
    axA.text(x + w / 2, y0 + hgt / 2, txt, ha="center", va="center", fontsize=5.6,
             color=BLACK, linespacing=1.3)
    if i < n - 1:
        axA.add_patch(FancyArrowPatch((x + w + 0.004, y0 + hgt / 2),
                                      (x + w + gap - 0.004, y0 + hgt / 2),
                                      arrowstyle="-|>", mutation_scale=6,
                                      lw=0.6, color="#555555"))
axA.text(gap, 0.12, "per group (RIF vs Fertile); genes detected in ≥5% of cells in both groups",
         fontsize=5.4, color="#555555", ha="left")
axA.text(1 - gap, 0.12, "reference parameters shown; sensitivity in B–D",
         fontsize=5.4, color="#555555", ha="right")
axA.text(gap, 0.02, "MAGIC + Pearson/Fisher-z approximation of scDNS; not the scDNS model",
         fontsize=5.4, color=ACCENT_RED, ha="left")
panel_letter(axA, "A", x=-0.012, y=1.03)

# ---------------- B: parameter grid heatmaps ----------------
gsB = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], wspace=0.30)
cmap_r = LinearSegmentedColormap.from_list("r", ["#F7FBFF", "#9ECAE1", "#2171B5", "#08306B"])
cmap_j = LinearSegmentedColormap.from_list("j", ["#FFFFFF", "#FDDBC7", "#EF8A62", "#B2182B"])
cols = [(k, t) for k in [3, 5, 8] for t in [2, 3, 4]]
labels = [f"k{k} t{t}" for k, t in cols]
for j, (key, cmap, vmax, title) in enumerate([
        ("spearman_vs_ref", cmap_r, 1.0, "Gene-level z: Spearman ρ vs reference"),
        ("top50_jaccard_vs_ref", cmap_j, 1.0, "Top-50 divergence: Jaccard vs reference")]):
    ax = fig.add_subplot(gsB[0, j])
    M = np.full((len(CT), len(cols)), np.nan)
    for i, ct in enumerate(CT):
        for c, (k, t) in enumerate(cols):
            v = grid[(grid.cell_type == ct) & (grid.knn == k) & (grid.t == t)][key]
            if len(v):
                M[i, c] = float(v.iloc[0])
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    for i in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.text(c, i, f"{M[i, c]:.2f}", ha="center", va="center", fontsize=5.4,
                    color="white" if M[i, c] > 0.72 else BLACK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, fontsize=5.4, rotation=90)
    ax.set_yticks(range(len(CT))); ax.set_yticklabels(CT, fontsize=5.6)
    ax.tick_params(length=0, pad=2)
    ax.set_title(title, fontsize=6, pad=3)
    # reference cell k5/t3
    ref_c = cols.index((5, 3))
    ax.add_patch(plt.Rectangle((ref_c - 0.5, -0.5), 1, len(CT), fill=False,
                               ec=BLACK, lw=0.9))
    for y in np.arange(-0.5, len(CT), 1):
        ax.axhline(y, color="white", lw=0.5)
    for x in np.arange(-0.5, len(cols), 1):
        ax.axvline(x, color="white", lw=0.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=5.4, length=1.5)
    cb.outline.set_linewidth(0.3)
    if j == 0:
        cb.set_label("ρ", fontsize=5.4, labelpad=1)
        panel_letter(ax, "B", x=-0.32, y=1.10)
    else:
        cb.set_label("Jaccard", fontsize=5.4, labelpad=1)

# ---------------- C: |combined_z| threshold stability ----------------
axC = fig.add_subplot(gs[2, 0])
x = np.arange(len(thr))
axC.bar(x, thr.n_modules, width=0.55, color="#D9D9D9", edgecolor="#8A8A8A",
        lw=0.4, label="modules (left)", zorder=1)
axC.set_ylim(0, 14)
axC.set_yticks([0, 4, 8, 12])
axC.set_ylabel("Modules", fontsize=6, labelpad=1)
axC.set_xticks(x)
axC.set_xticklabels([f"{c:g}\n({int(g)})" for c, g in zip(thr.cz_min, thr.n_genes)],
                    fontsize=5.4)
axC.set_xlabel("|combined z| cutoff (genes retained)", fontsize=6, labelpad=2)
axC.tick_params(labelsize=5.4, length=2, pad=1)
ax2 = axC.twinx()
ax2.plot(x, -np.log10(thr.min_mhc_q), color=ACCENT_RED, marker="o", ms=3.4, lw=1.2,
         label="MHC-I q (right)", zorder=3)
ax2.set_ylim(3.5, 6.0)
ax2.set_ylabel("−log$_{10}$ MHC-I $q$", fontsize=6, color=ACCENT_RED, labelpad=2)
ax2.tick_params(labelsize=5.4, length=2, pad=1, colors=ACCENT_RED)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(ACCENT_RED)
for xi, (q, jac) in enumerate(zip(thr.min_mhc_q, thr.mean_matched_jaccard)):
    axC.text(xi, 0.35, f"J={jac:.2f}", ha="center", va="bottom", fontsize=5.4,
             color="#444444")
handles = [plt.Rectangle((0, 0), 1, 1, fc="#D9D9D9", ec="#8A8A8A", lw=0.4),
           Line2D([0], [0], color=ACCENT_RED, marker="o", ms=3, lw=1.2)]
axC.legend(handles, ["n modules", "MHC-I hypergeom q"], fontsize=5.4, loc="upper left",
           handlelength=1.0, borderpad=0.15, labelspacing=0.25, frameon=False)
axC.set_title("Module robustness to effect-size cutoff", fontsize=6, pad=3)
panel_letter(axC, "C", x=-0.18, y=1.08)

# ---------------- D: correlation-metric sensitivity ----------------
gsD = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 1], hspace=0.48)
alt = ["MAGIC + Spearman", "No MAGIC + Pearson"]
colors = {"Stromal": "#1B7837", "Epithelial": "#2166AC", "T cell": "#B2182B"}
for j, (metric, ylab, ylim) in enumerate([
        ("spearman_vs_ref", "ρ vs reference", (0, 1.0)),
        ("top50_jaccard_vs_ref", "Jaccard vs reference", (0, 1.0))]):
    ax = fig.add_subplot(gsD[j])
    wbar = 0.24
    for i, ct in enumerate(CT):
        vals = [1.0] + [float(met[(met.cell_type == ct) & (met.metric == m)][metric].iloc[0])
                        for m in alt]
        ax.bar(np.arange(3) + (i - 1) * wbar, vals, width=wbar, color=colors[ct],
               edgecolor="white", lw=0.4, label=ct, zorder=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["MAGIC +\nPearson", "MAGIC +\nSpearman", "No MAGIC +\nPearson"],
                       fontsize=5.4)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylab, fontsize=5.6, labelpad=1)
    ax.tick_params(labelsize=5.4, length=2, pad=1)
    if j == 0:
        ax.legend(fontsize=5.4, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 0.96),
                  handlelength=0.9, columnspacing=0.8, borderpad=0.1,
                  handletextpad=0.4)
        ax.set_title("Two alternative pipelines (reference = 1.0)", fontsize=6, pad=12)
        panel_letter(ax, "D", x=-0.18, y=1.28)
    else:
        ax.set_xlabel("Network construction variant", fontsize=5.8, labelpad=2)

save_cns_figure(fig, f"{FIG}/FigS2_method_sensitivity")
print("FIG_S2_DONE")
