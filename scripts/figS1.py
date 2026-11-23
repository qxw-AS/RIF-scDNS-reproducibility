#!/usr/bin/env python3
"""Fig. S1 — cohort structure, QC, composition, and annotation robustness."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from fig_style import (CATEGORICAL, DIVERGING, ACCENT_RED, ACCENT_BLUE, GREY,
                       BLACK, panel_letter, save_cns_figure)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
D = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "figS"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
os.makedirs(FIG, exist_ok=True)

CT = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
      "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]
CT_COLORS = {
    "Epithelial": "#2166AC", "Glandular": "#4393C3", "Ciliated": "#9970AB",
    "Stromal": "#1B7837", "Endothelial": "#F1A340", "Lymphatic EC": "#B35806",
    "T cell": "#B2182B", "NK cell": "#D6604D", "B/Plasma": "#762A83",
    "Myeloid": "#666666", "Mast": "#222222", "Smooth muscle": "#5AAE61",
}
GROUP_COLORS = {"Fertile": "#2166AC", "RIF": "#B2182B"}
SAMPLES = [f"RIF_{i}" for i in range(1, 11)] + ["LH7_1", "LH7_2", "LH7_3",
                                                "LH7_PGT_1", "LH7_PGT_2", "LH7_PGT_3"]

cnt = pd.read_csv(f"{D}/s1_donor_counts.csv")
qc = pd.read_csv(f"{D}/s1_qc_by_sample.csv").set_index("sample").loc[SAMPLES]
comp = pd.read_csv(f"{D}/s1_composition_stats.csv")
cl = pd.read_csv(f"{D}/s1_cluster_qc.csv", index_col=0)
ari = pd.read_csv(f"{D}/s1_subsample_ari.csv")
ncl = pd.read_csv(f"{D}/s1_resolution_clusters.csv")

fig = plt.figure(figsize=(7.05, 4.7))
gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.95],
              width_ratios=[1.0, 1.0, 0.52, 0.84], wspace=0.55, hspace=0.62,
              left=0.065, right=0.975, top=0.94, bottom=0.10)

# ---------------- A: donor x cell type ----------------
axA = fig.add_subplot(gs[0, 0:2])
piv = (cnt.pivot_table(index="sample", columns="cell_type", values="n", aggfunc="sum")
       .reindex(index=SAMPLES, columns=CT).fillna(0))
bottom = np.zeros(len(piv))
x = np.arange(len(piv))
for ct in CT:
    axA.bar(x, piv[ct].to_numpy(), bottom=bottom, width=0.78,
            color=CT_COLORS[ct], edgecolor="white", linewidth=0.35, label=ct)
    bottom += piv[ct].to_numpy()
axA.set_xticks(x)
axA.set_xticklabels(SAMPLES, rotation=90, fontsize=5.4)
for t, s in zip(axA.get_xticklabels(), SAMPLES):
    t.set_color(GROUP_COLORS["RIF" if s.startswith("RIF") else "Fertile"])
axA.set_ylabel("Annotated cells")
axA.set_xlim(-0.7, len(piv) - 0.3)
axA.axvline(9.5, color=GREY, lw=0.6, ls=":")
axA.text(4.5, axA.get_ylim()[1] * 1.04, "RIF (10 donors)", ha="center", va="bottom",
         fontsize=5.6, color=GROUP_COLORS["RIF"])
axA.text(12.5, axA.get_ylim()[1] * 1.04, "Fertile (6 donors)", ha="center", va="bottom",
         fontsize=5.6, color=GROUP_COLORS["Fertile"])
axA.set_ylim(0, axA.get_ylim()[1] * 1.12)
handles = [Line2D([0], [0], marker="s", linestyle="", markersize=4,
                  markerfacecolor=CT_COLORS[c], markeredgecolor="none", label=c) for c in CT]
axLeg = fig.add_subplot(gs[0, 2])
axLeg.axis("off")
axLeg.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.05, 0.5),
             fontsize=5.4, handletextpad=0.3, labelspacing=0.38, borderpad=0.1)
panel_letter(axA, "A", x=-0.09, y=1.13)

# ---------------- B: QC ----------------
gsB = gs[0, 3].subgridspec(2, 1, hspace=0.10)
axB1 = fig.add_subplot(gsB[0])
axB2 = fig.add_subplot(gsB[1], sharex=axB1)
for ax, col, lab in [(axB1, "n_genes_median", "Genes/cell"), (axB2, "pct_mt_median", "Mitochondrial %")]:
    vals = [qc.loc[s, col] for s in SAMPLES]
    cols = [GROUP_COLORS["RIF" if s.startswith("RIF") else "Fertile"] for s in SAMPLES]
    ax.scatter(np.arange(len(SAMPLES)), vals, s=15, c=cols, linewidths=0, zorder=3)
    ax.set_ylabel(lab, fontsize=6, labelpad=1)
    ax.tick_params(axis="y", labelsize=5.5, length=2, pad=1)
    ax.set_xlim(-0.6, len(SAMPLES) - 0.4)
axB2.set_xticks(np.arange(len(SAMPLES)))
axB2.set_xticklabels(SAMPLES, rotation=90, fontsize=5.4)
for t, s in zip(axB2.get_xticklabels(), SAMPLES):
    t.set_color(GROUP_COLORS["RIF" if s.startswith("RIF") else "Fertile"])
axB1.tick_params(axis="x", labelbottom=False, length=0)
axB1.set_title("Per-donor QC", fontsize=6, pad=2)
panel_letter(axB1, "B", x=-0.30, y=1.12)

# ---------------- C: composition effect ----------------
axC = fig.add_subplot(gs[1, 0])
comp = comp.set_index("cell_type").loc[CT]
y = np.arange(len(CT))[::-1]
for yy, (ct, r) in zip(y, comp.iterrows()):
    c = ACCENT_RED if r.log2_ratio > 0.4 else (ACCENT_BLUE if r.log2_ratio < -0.4 else GREY)
    axC.plot([0, r.log2_ratio], [yy, yy], color=c, lw=0.8, alpha=0.7)
    axC.scatter(r.log2_ratio, yy, s=18, color=c, edgecolor="white", linewidth=0.3, zorder=3)
axC.axvline(0, color=BLACK, lw=0.6)
axC.set_yticks(y)
axC.set_yticklabels(CT, fontsize=5.4)
axC.set_xlabel("log2 fraction ratio (RIF / fertile)", fontsize=6, labelpad=1)
axC.tick_params(axis="x", labelsize=5.4, length=2, pad=1)
axC.tick_params(axis="y", length=0, pad=2)
axC.set_xlim(-1.6, 1.6)
axC.text(0.98, 0.03, "donor-level Wilcoxon\nall BH q > 0.8",
         transform=axC.transAxes, fontsize=5.4, color="#555555",
         va="bottom", ha="right")
panel_letter(axC, "C", x=-0.30, y=1.04)

# ---------------- D: cluster x donor composition ----------------
axD = fig.add_subplot(gs[1, 1])
cm = pd.read_csv(f"{D}/s1_cluster_sample_matrix.csv", index_col=0)
sm = pd.read_csv(f"{D}/s1_sample_group.csv", index_col=0)


def _skey(s):
    pre = "RIF" if s.startswith("RIF") else ("PGT" if "PGT" in s else "LH7")
    return (pre, int(s.split("_")[-1]))


order_s = (sorted([s for s in sm.index if sm.loc[s, "group"] == "RIF"], key=_skey) +
           sorted([s for s in sm.index if sm.loc[s, "group"] == "Fertile"], key=_skey))
cm = cm[order_s]
imD = axD.imshow(cm.to_numpy(), aspect="auto", interpolation="nearest",
                 cmap=LinearSegmentedColormap.from_list("frac", ["#FFFFFF", "#FDDBC7",
                                                                 "#B2182B"]),
                 vmin=0, vmax=1)
axD.set_xticks(range(len(order_s)))
axD.set_xticklabels([s.replace("LH7_PGT", "PGT") for s in order_s], rotation=90, fontsize=5.4)
axD.set_yticks(range(0, len(cm), 5))
axD.set_yticklabels([str(i) for i in range(0, len(cm), 5)], fontsize=5.4)
axD.set_ylabel("Leiden cluster", fontsize=5.6, labelpad=1)
axD.tick_params(length=1.5, pad=1)
for x in range(len(order_s) + 1):
    axD.axvline(x - 0.5, color="white", lw=0.3)
axD.axvline(len(order_s) - sum(1 for s in order_s if sm.loc[s, "group"] == "Fertile") - 0.5,
            color="#333333", lw=0.8)
axD.text(0.02, 1.03, "RIF (10 donors)", transform=axD.transAxes, fontsize=5.4,
         color="#B2182B", ha="left", va="bottom")
axD.text(0.98, 1.03, "Fertile (6)", transform=axD.transAxes, fontsize=5.4,
         color="#2166AC", ha="right", va="bottom")
axD.set_title("Leiden clusters × donors", fontsize=6, pad=13)
axD.text(0.02, -0.50, "14/53 clusters >50% single donor\n(9.6% of cells, 14,409/149,590)",
         transform=axD.transAxes, fontsize=5.4, color=ACCENT_RED, va="top")
caxD = axD.inset_axes([0.72, -0.34, 0.26, 0.035])
cbD = fig.colorbar(imD, cax=caxD, orientation="horizontal")
cbD.ax.tick_params(labelsize=5.4, length=1, pad=0.5)
cbD.set_label("fraction of cluster", fontsize=5.4, labelpad=0.5)
cbD.outline.set_linewidth(0.25)
panel_letter(axD, "D", x=-0.22, y=1.05)

# ---------------- E: ARI ----------------
axE = fig.add_subplot(gs[1, 2:4])
for rep, g in ari.groupby("repeat"):
    axE.plot(g.resolution, g.ari, color=GREY, lw=0.6, alpha=0.7, marker="o", ms=2)
mean = ari.groupby("resolution").ari.mean()
axE.plot(mean.index, mean.values, color=BLACK, lw=1.0, marker="o", ms=3.5, zorder=4)
axE.set_xlabel("Leiden resolution", fontsize=6, labelpad=1)
axE.set_ylabel("ARI vs full data", fontsize=6, labelpad=1)
axE.set_ylim(0.50, 0.90)
axE.set_xlim(0.85, 2.65)
axE.tick_params(labelsize=5.4, length=2, pad=1)
for _, r in ncl.iterrows():
    axE.text(r.resolution, 0.885, f"{int(r.n_clusters_full)}", ha="center", va="top",
             fontsize=5.4, color="#555555")
axE.text(0.03, 0.06, "80% subsampling, 3 repeats", transform=axE.transAxes,
         fontsize=5.4, color="#555555")
panel_letter(axE, "E", x=-0.22, y=1.04)

save_cns_figure(fig, f"{FIG}/FigS1_cohort_QC")
print("FIG_S1_DONE")
