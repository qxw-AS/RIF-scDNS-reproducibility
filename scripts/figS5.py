#!/usr/bin/env python3
"""Fig. S5 — DEG concordance, per-donor consistency, power, pseudobulk counts."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from fig_style import (ACCENT_RED, ACCENT_BLUE, GREY, BLACK,
                       panel_letter, save_cns_figure)

# Paths resolve relative to this package; override with RIF_DATA / RIF_DATA2 / RIF_DATA3 / RIF_FIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
D = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "figS"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
CT = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
      "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]

deg = pd.read_csv(f"{D}/s5_deg_merged.csv.gz")
con = pd.read_csv(f"{D}/s5_concordance.csv").set_index("cell_type")
don = pd.read_csv(f"{D}/s5_donor_consistency.csv")
pw = pd.read_csv(f"{D}/s5_power.csv")

fig = plt.figure(figsize=(6.95, 7.0))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.52, 0.92, 0.50],
              width_ratios=[0.47, 0.53], hspace=0.46, wspace=0.30,
              left=0.085, right=0.975, top=0.955, bottom=0.07)

# ---------------- A: cell-level vs pseudobulk log2FC facets ----------------
gsA = GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[0, :], hspace=0.55, wspace=0.32)
axesA = []
lim = 2.6
for k, ct in enumerate(CT):
    ax = fig.add_subplot(gsA[k // 4, k % 4])
    d = deg[deg.cell_type == ct]
    ns = ~d.sig_cell
    ax.scatter(d.log2FC[ns], d.logfoldchanges[ns], s=1.0, c="#C9C9C9",
               alpha=0.22, lw=0, rasterized=True, zorder=2)
    ax.scatter(d.log2FC[d.sig_cell], d.logfoldchanges[d.sig_cell], s=1.0,
               c=ACCENT_RED, alpha=0.28, lw=0, rasterized=True, zorder=3)
    ax.plot([-lim, lim], [-lim, lim], color="#555555", lw=0.4, ls=(0, (3, 2)), zorder=4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=5.4, length=1.8, pad=1)
    r = con.loc[ct, "spearman_log2FC"]
    ax.text(0.04, 0.96, f"ρ = {r:.2f}\nsig = {int(con.loc[ct,'n_sig_cell']):,}",
            transform=ax.transAxes, fontsize=5.4, va="top", ha="left")
    ax.set_title(ct, fontsize=5.4, pad=2)
    axesA.append(ax)
ax_leg = fig.add_subplot(gsA[2, 3])
ax_leg.set_axis_off()
ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
ax_leg.scatter([0.10], [0.72], s=10, c=ACCENT_RED, alpha=0.6, lw=0)
ax_leg.scatter([0.10], [0.54], s=10, c="#C9C9C9", alpha=0.8, lw=0)
ax_leg.text(0.20, 0.72, "cell-level FDR < 0.05", fontsize=5.4, va="center")
ax_leg.text(0.20, 0.54, "not significant", fontsize=5.4, va="center")
ax_leg.text(0.0, 0.30, "Axes clipped at ±2.6;\n93016 gene–type pairs;\n"
            "dashed line: y = x", fontsize=5.4, va="top", color="#555555")
panel_letter(axesA[0], "A", x=-0.30, y=1.22)
fig.text(0.035, 0.705, "Pseudobulk log$_2$FC (RIF vs Fertile)", rotation=90,
         fontsize=7, va="center")
fig.canvas.draw()
_ybot = axesA[8].get_position().y0
fig.text(0.505, _ybot - 0.043, "Cell-level Wilcoxon log$_2$FC (RIF vs Fertile)",
         fontsize=7, ha="center")

# ---------------- B: per-donor direction consistency ----------------
axB = fig.add_subplot(gs[1, 0])
med = don.groupby("cell_type").frac_RIF_same_dir.median().sort_values()
cy = {ct: i for i, ct in enumerate(med.index)}
for ct, d in don.groupby("cell_type"):
    y = cy[ct]
    axB.hlines(y, d.frac_RIF_same_dir.min(), d.frac_RIF_same_dir.max(),
               color="#D5D5D5", lw=0.8, zorder=1)
    axB.scatter(d.frac_RIF_same_dir, np.full(len(d), y), s=17,
                c=[ACCENT_RED if v > 0 else ACCENT_BLUE for v in d.overall_diff],
                edgecolor="white", lw=0.3, zorder=3)
    axB.scatter([med[ct]], [y], marker="|", s=50, color="#333333", lw=0.9, zorder=4)
    low = d[d.frac_RIF_same_dir <= 0.5].sort_values("frac_RIF_same_dir")
    for i, (_, r) in enumerate(low.iterrows()):
        axB.annotate(f"{r.gene} {r.frac_RIF_same_dir:.1f}",
                     (r.frac_RIF_same_dir, y), xytext=(3, 4 + 9 * i),
                     textcoords="offset points", fontsize=5.4, color=ACCENT_RED,
                     va="bottom")
axB.axvline(0.5, color="#777777", lw=0.5, ls=(0, (3, 2)))
axB.set_yticks(range(len(med))); axB.set_yticklabels(med.index, fontsize=5.4)
axB.set_xlim(0.30, 1.06); axB.set_ylim(-0.6, len(med) - 0.4)
axB.set_xlabel("Fraction of RIF donors matching the group-mean direction",
               fontsize=5.4, labelpad=2)
axB.tick_params(axis="x", labelsize=5.4, length=2, pad=1)
axB.tick_params(axis="y", length=0, pad=2)
axB.set_title("Per-donor consistency (44 gene–type pairs; 10 RIF, 6 fertile; tick = median)",
              fontsize=5.4, pad=3)
panel_letter(axB, "B", x=-0.30, y=1.06)

# ---------------- C: downsampling power ----------------
axC = fig.add_subplot(gs[1, 1])
colors = {"Stromal": "#1B7837", "Epithelial": ACCENT_BLUE, "T cell": ACCENT_RED}
for ct in ["Stromal", "Epithelial", "T cell"]:
    d = pw[pw.cell_type == ct].sort_values("N_per_group")
    axC.plot(d.N_per_group, d.n_sig_padj05, marker="o", ms=3.2, lw=1.1,
             color=colors[ct], label=ct)
    axC.plot(d.N_per_group, d.n_sig_padj01, marker="s", ms=2.4, lw=0.8,
             ls=(0, (2, 2)), color=colors[ct], alpha=0.85)
axC.set_xscale("log")
axC.set_xticks([100, 300, 500, 1000])
axC.set_xticklabels(["100", "300", "500", "1,000"])
axC.set_xlabel("Cells sampled per group", fontsize=5.4, labelpad=2)
axC.set_ylabel("DEGs (of 2,000 genes)", fontsize=5.4, labelpad=1)
axC.tick_params(labelsize=5.4, length=2, pad=1)
axC.set_ylim(0, 680)
axC.legend(fontsize=5.4, loc="upper left", handlelength=1.1, borderpad=0.15,
           labelspacing=0.25)
axC.text(0.985, 0.03, "solid: FDR < 0.05\ndashed: FDR < 0.01",
         transform=axC.transAxes, fontsize=5.4, ha="right", va="bottom",
         color="#555555")
axC.set_title("Downsampling power (2,000-gene network panel)", fontsize=5.6, pad=3)
panel_letter(axC, "C", x=-0.20, y=1.06)

# ---------------- D: DEG counts cell-level vs pseudobulk ----------------
axD = fig.add_subplot(gs[2, :])
cc = con[["n_sig_cell", "n_sig_pb"]].sort_values("n_sig_cell")
y = np.arange(len(cc))
h = 0.36
axD.barh(y + h / 2, cc.n_sig_cell, height=h, color=ACCENT_RED, edgecolor="white",
         lw=0.3, label="Cell-level Wilcoxon")
axD.barh(y - h / 2, cc.n_sig_pb, height=h, color="#9E9E9E", edgecolor="white",
         lw=0.3, label="Pseudobulk")
for yi, v in zip(y, cc.n_sig_cell):
    axD.text(v + 60, yi + h / 2, f"{int(v):,}", va="center", fontsize=5.4)
for yi, v in zip(y, cc.n_sig_pb):
    axD.text(12, yi - h / 2, f"{int(v)}", va="center", fontsize=5.4, color="#555555")
axD.set_yticks(y); axD.set_yticklabels(cc.index, fontsize=5.4)
axD.set_xlim(0, 7400)
axD.set_xlabel("Number of genes with FDR < 0.05", fontsize=5.4, labelpad=2)
axD.tick_params(axis="x", labelsize=5.4, length=2, pad=1)
axD.tick_params(axis="y", length=0, pad=2)
axD.legend(fontsize=5.4, ncol=2, loc="lower right", handlelength=1.0,
           borderpad=0.15, columnspacing=1.0)
axD.set_title("Cell-level vs pseudobulk DEG counts", fontsize=5.6, pad=3)
axD.text(0.985, 0.55, "93,016 pairs: 1 DEG (ZNF552, q = 0.025)\n"
                      "12,822 pairs (|log2FC| ≥ 0.5): 2 DEGs\n"
                      "(+ PIK3AP1, T cell, q = 0.034)",
         transform=axD.transAxes, fontsize=5.4, ha="right", va="center",
         color="#555555", linespacing=1.35)
panel_letter(axD, "D", x=-0.065, y=1.10)

save_cns_figure(fig, f"{FIG}/FigS5_DEG_robustness")
print("FIG_S5_DONE", f"median_frac={don.frac_RIF_same_dir.median():.3f}",
      f"frac>=0.8={(don.frac_RIF_same_dir >= 0.8).mean():.2f}")
