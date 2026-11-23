#!/usr/bin/env python3
"""Fig. S6 — H1 decision robustness: per-type curves, p-values, cross-type overlap, scope."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from fig_style import (ACCENT_RED, ACCENT_BLUE, GREY, BLACK, DIVERGING,
                       panel_letter, save_cns_figure)

# Paths resolve relative to this package; override with RIF_DATA / RIF_DATA2 / RIF_DATA3 / RIF_FIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
D2 = os.environ.get("RIF_DATA2", os.path.join(_ROOT, "data", "fig2"))
D = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "figS"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))

CT = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
      "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]
cur = pd.read_csv(f"{D}/../fig3/jaccard_curves.csv")
glob = pd.read_csv(f"{D}/../fig3/jaccard_global.csv")
J = pd.read_csv(f"{D}/s6_cross_type_jaccard.csv", index_col=0).loc[CT, CT]
P = pd.read_csv(f"{D}/s6_pvalue_matrix.csv", index_col=0).loc[CT]
env = pd.read_csv(f"{D}/s6_scope_envelope.csv")

fig = plt.figure(figsize=(7.05, 4.75))
gs = GridSpec(2, 2, figure=fig, wspace=0.50, hspace=0.58,
              left=0.075, right=0.975, top=0.94, bottom=0.10)

# ---------------- A: per-type curves ----------------
axA = fig.add_subplot(gs[0, 0])
for ct, g in cur.groupby("cell_type"):
    g = g.sort_values("N")
    axA.plot(g.N, g.jaccard, color="#B9B9B9", lw=0.6, alpha=0.9)
med = cur.groupby("N").jaccard.median()
nul = cur.groupby("N")[["null_mean", "null_lo", "null_hi"]].mean()
axA.fill_between(nul.index, nul.null_lo, nul.null_hi, color="#D9D9D9", alpha=0.7, lw=0)
axA.plot(nul.index, nul.null_mean, color="#777777", lw=0.7, ls="--")
axA.plot(med.index, med.values, color=ACCENT_RED, lw=1.4, marker="o", ms=3,
         label="median (11 types)")
axA.axhline(0.5, color=ACCENT_RED, lw=0.9, ls=":", alpha=0.8)
axA.text(48, 0.515, "pre-specified 0.5", fontsize=5.4, color=ACCENT_RED, ha="left")
axA.set_xlabel("Top-N genes per method", fontsize=6, labelpad=1)
axA.set_ylabel("Jaccard (divergence ∩ DEG)", fontsize=6, labelpad=1)
axA.set_xlim(40, 900)
axA.set_ylim(0, 0.56)
axA.tick_params(labelsize=5.5, length=2, pad=1)
axA.text(0.03, 0.62, "per cell type (n=11)\n95% permutation null\n"
         f"median J$_{{50}}$ = {med.loc[50]:.2f}",
         transform=axA.transAxes, fontsize=5.4, color="#555555",
         va="top", ha="left", linespacing=1.35)
panel_letter(axA, "A", x=-0.13, y=1.04)

# ---------------- B: p-value heatmap ----------------
axB = fig.add_subplot(gs[0, 1])
M = -np.log10(P.to_numpy())
cmap = LinearSegmentedColormap.from_list("p", ["#FFFFFF", "#DEEBF7", "#6BAED6", "#08306B"])
im = axB.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=4)
axB.set_xticks(range(P.shape[1]))
axB.set_xticklabels([int(c) for c in P.columns], fontsize=5.5)
axB.set_yticks(range(len(CT)))
axB.set_yticklabels(CT, fontsize=5.4)
axB.tick_params(length=0, pad=2)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if P.to_numpy()[i, j] < 0.05:
            axB.text(j, i, "*", ha="center", va="center", fontsize=6.5,
                     color="white" if M[i, j] > 2.2 else BLACK)
axB.set_xlabel("Top-N", fontsize=6, labelpad=1)
axB.set_title("Permutation p (upper tail, uncorrected)", fontsize=6, pad=3)
for x in np.arange(-0.5, P.shape[1], 1):
    axB.axvline(x, color="white", lw=0.6)
for y in np.arange(-0.5, len(CT), 1):
    axB.axhline(y, color="white", lw=0.5)
caxB = axB.inset_axes([0.0, -0.16, 0.45, 0.035])
cb = fig.colorbar(im, cax=caxB, orientation="horizontal")
cb.set_label("−log10 p", fontsize=5.4, labelpad=1)
cb.ax.tick_params(labelsize=5.4, length=1.5, pad=1)
cb.outline.set_linewidth(0.3)
panel_letter(axB, "B", x=-0.22, y=1.05)

# ---------------- C: cross-type Jaccard ----------------
axC = fig.add_subplot(gs[1, 0])
cmap2 = LinearSegmentedColormap.from_list("j", ["#FFFFFF", "#C6DBEF", "#2171B5", "#08306B"])
im2 = axC.imshow(J.to_numpy(), aspect="auto", cmap=cmap2, vmin=0, vmax=0.6)
axC.set_xticks(range(len(CT)))
axC.set_xticklabels(CT, rotation=90, fontsize=5.4)
axC.set_yticks(range(len(CT)))
axC.set_yticklabels(CT, fontsize=5.4)
axC.tick_params(length=0, pad=2)
for x in np.arange(-0.5, len(CT), 1):
    axC.axvline(x, color="white", lw=0.5)
for y in np.arange(-0.5, len(CT), 1):
    axC.axhline(y, color="white", lw=0.5)
axC.set_title("Top-50 divergence overlap across cell types", fontsize=6, pad=3)
caxC = axC.inset_axes([1.02, 0.0, 0.028, 1.0])
cb2 = fig.colorbar(im2, cax=caxC)
cb2.set_label("Jaccard", fontsize=5.4, labelpad=1)
cb2.ax.tick_params(labelsize=5.4, length=1.5, pad=1)
cb2.outline.set_linewidth(0.3)
panel_letter(axC, "C", x=-0.24, y=1.06)

# ---------------- D: scope envelope ----------------
axD = fig.add_subplot(gs[1, 1])
q1 = cur.groupby("N").jaccard.quantile(0.25)
q3 = cur.groupby("N").jaccard.quantile(0.75)
axD.fill_between(q1.index, q1.values, q3.values, color="#D9D9D9", alpha=0.7, lw=0)
axD.plot(med.index, med.values, color=ACCENT_RED, lw=1.4, marker="o", ms=3,
         label="cell-type median")
axD.plot(glob.N, glob.jaccard, color=ACCENT_BLUE, lw=1.2, marker="s", ms=2.6,
         label="global min-padj")
axD.axhline(0.5, color=ACCENT_RED, lw=0.9, ls=":", alpha=0.8)
axD.set_xscale("log")
axD.minorticks_off()
axD.set_xticks([50, 100, 200, 400, 800])
axD.set_xticklabels(["50", "100", "200", "400", "800"])
axD.set_xlabel("Top-N genes per method", fontsize=6, labelpad=1)
axD.set_ylabel("Jaccard", fontsize=6, labelpad=1)
axD.set_ylim(0, 0.56)
axD.tick_params(labelsize=5.5, length=2, pad=1)
axD.legend(loc="upper left", fontsize=5.4, handlelength=1.2, labelspacing=0.3,
           borderpad=0.1)
axD.text(0.97, 0.06, "global N=50: J=0.010\nnetwork-only=0", transform=axD.transAxes,
         fontsize=5.4, color=ACCENT_BLUE, ha="right", va="bottom")
panel_letter(axD, "D", x=-0.16, y=1.05)

save_cns_figure(fig, f"{FIG}/FigS6_H1_robustness")
print("FIG_S6_DONE")
