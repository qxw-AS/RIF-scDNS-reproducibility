#!/usr/bin/env python3
"""Fig. S9 (v2 numbering) — external single-cell expression in GSE248784 (not replicated)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from fig_style import ACCENT_RED, ACCENT_BLUE, GREY, BLACK, panel_letter, save_cns_figure

# Paths resolve relative to this package; override with RIF_DATA / RIF_DATA2 / RIF_DATA3 / RIF_FIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
D = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "external", "gse248784", "processed"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
t = pd.read_csv(f"{D}/s10_tests.csv")
g = pd.read_csv(f"{D}/s10_m2_genes.csv")
ph = pd.read_csv(f"{D}/s10_phase.csv")

CT = ["Epithelial", "Endothelial", "Myeloid", "NK cell", "Stromal", "T cell"]
CT_LAB = ["Epithelial", "Endothelial", "Myeloid", "NK", "Stromal", "T cell"]
FEAT = ["M2", "HLA_score", "IFN_score"]
FEAT_LAB = ["M2 module", "HLA score", "IFN score"]

fig = plt.figure(figsize=(7.2, 5.7))
gs = GridSpec(2, 2, figure=fig, width_ratios=[0.47, 0.53],
              height_ratios=[0.92, 1.0], hspace=0.44, wspace=0.30,
              left=0.085, right=0.965, top=0.955, bottom=0.10)

# ---------------- A: workflow + donor QC ----------------
gsA = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], height_ratios=[0.62, 0.38],
                              hspace=0.42)
axA = fig.add_subplot(gsA[0]); axA.set_axis_off()
axA.set_xlim(0, 1); axA.set_ylim(0, 1)
boxes = [
    ("GSE248784\nscRNA-seq\n21 donors", "#E8EEF6"),
    ("11 RIF\n10 Normal\nLH+7 not annotated", "#DCE6F2"),
    ("73,500 cells\n3,500/donor\n8 cell types", "#DCE6F2"),
    ("Donor × type\npseudobulk\nMWU + BH", "#F6E3E3"),
]
for i, (txt, col) in enumerate(boxes):
    x = 0.03 + (i % 2) * 0.49
    y = 0.53 - (i // 2) * 0.47
    axA.add_patch(FancyBboxPatch((x, y), 0.45, 0.42,
                                 boxstyle="round,pad=0.010,rounding_size=0.03",
                                 fc=col, ec="#8A8A8A", lw=0.5))
    axA.text(x + 0.225, y + 0.21, txt, ha="center", va="center", fontsize=5.4,
             color=BLACK, linespacing=1.3)
    if i % 2 == 0:
        axA.add_patch(FancyArrowPatch((x + 0.455, y + 0.21), (x + 0.485, y + 0.21),
                                      arrowstyle="-|>", mutation_scale=5.5, lw=0.55, color="#555555"))
    if i < 2:
        axA.add_patch(FancyArrowPatch((x + 0.225, y - 0.01), (x + 0.225, y - 0.075),
                                      arrowstyle="-|>", mutation_scale=5.5, lw=0.55, color="#555555"))
axA.set_title("Cell-type-resolved external validation", fontsize=5.8, pad=3)
panel_letter(axA, "A", x=-0.02, y=1.06)

axAq = fig.add_subplot(gsA[1])
qc = pd.read_csv(f"{D}/external_sample_QC.csv")
qc = qc.sort_values(["group", "gsm"], ascending=[True, True])
xq = np.arange(len(qc))
axAq.bar(xq, qc.median_genes / 1000, width=0.72,
         color=[ACCENT_RED if g == "RIF" else ACCENT_BLUE for g in qc.group],
         edgecolor="none")
axAq.set_ylim(0, 3.4)
axAq.set_xlim(-0.7, len(qc) - 0.3)
axAq.set_ylabel("Median genes\n(×1,000)", fontsize=5.4, labelpad=1)
axAq.set_xticks([])
axAq.tick_params(axis="y", labelsize=5.4, length=1.5, pad=1)
axAq.set_title("Donor QC: genes p = 0.46; UMI p = 0.55; mt% p = 0.11", fontsize=5.4, pad=2)

# ---------------- B: donor-level effect heatmap ----------------
axB = fig.add_subplot(gs[0, 1])
M = np.full((len(FEAT), len(CT)), np.nan)
P = np.full_like(M, np.nan)
for i, f in enumerate(FEAT):
    for j, ct in enumerate(CT):
        r = t[(t.feature == f) & (t.cell_type == ct)]
        if len(r):
            M[i, j] = r.delta.iloc[0]
            P[i, j] = r.p.iloc[0]
cmap = LinearSegmentedColormap.from_list("bwr", ["#2166AC", "#F7F7F7", "#B2182B"])
vmax = max(abs(np.nanmin(M)), abs(np.nanmax(M)))
im = axB.imshow(M, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax))
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        star = "*" if P[i, j] < 0.05 else ""
        axB.text(j, i, f"{M[i, j]:+.2f}{star}", ha="center", va="center", fontsize=5.4,
                 color="white" if abs(M[i, j]) > 0.62 * vmax else BLACK)
axB.set_xticks(range(len(CT))); axB.set_xticklabels(CT_LAB, rotation=40, ha="right", fontsize=5.4)
axB.set_yticks(range(len(FEAT))); axB.set_yticklabels(FEAT_LAB, fontsize=5.4)
axB.set_xticks(np.arange(-0.5, len(CT), 1), minor=True)
axB.set_yticks(np.arange(-0.5, len(FEAT), 1), minor=True)
axB.grid(which="minor", color="white", lw=0.6)
axB.tick_params(which="both", length=0)
axB.set_title("No module/HLA/IFN shift survives BH\n(all q = 1.00; * nominal p < 0.05)", fontsize=5.8, pad=3)
panel_letter(axB, "B", x=-0.14, y=1.08)
cb = fig.colorbar(im, ax=axB, fraction=0.028, pad=0.02)
cb.ax.tick_params(labelsize=5.4, length=2, pad=1)
cb.outline.set_linewidth(0.4)
cb.set_label("Δ (RIF − Normal), z", fontsize=5.4, labelpad=1)

# ---------------- C: M2 gene concordance by compartment ----------------
gsC = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0], wspace=0.42)
comps = [("all", "All cells"), ("Epithelial", "Epithelial"), ("Stromal", "Stromal")]
for k, (comp, lab) in enumerate(comps):
    ax = fig.add_subplot(gsC[0, k])
    s = g[g.compartment == comp]
    ax.axhline(0, color="#999999", lw=0.4, ls=(0, (3, 2)))
    ax.axvline(0, color="#999999", lw=0.4, ls=(0, (3, 2)))
    cols = [ACCENT_RED if x in ["HLA-A", "HLA-B", "HLA-C"] else "#8C8C8C" for x in s.gene]
    ax.scatter(s.discovery_combined_z, s.external_log2FC, s=7, c=cols,
               edgecolor="white", lw=0.2, zorder=3)
    ax.set_title(lab, fontsize=5.4, pad=2)
    ax.tick_params(labelsize=5.4, length=1.5, pad=1)
    ax.set_ylim(-0.85, 0.85)
    ax.set_xlim(1.1, 5.9)
    ax.set_xticks([2, 3, 4, 5])
    rho = s[["discovery_combined_z", "external_log2FC"]].corr(method="spearman").iloc[0, 1]
    ax.text(0.5, 0.02, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=5.4, color="#555555")
    if k == 0:
        ax.set_ylabel("External log2FC\n(RIF − Normal)", fontsize=5.4, labelpad=1)
    if k == 1:
        ax.set_xlabel("Discovery M2 combined z", fontsize=5.4, labelpad=1)
panel_letter(fig.axes[-3], "C", x=-0.30, y=1.12)
fig.text(0.105, 0.018, "M2 gene-level Spearman ρ = 0.32 (all), 0.37 (epithelial), 0.29 (stromal);\n"
                       "mean log2FC remains negative and sign concordance is 39% — no directional replication",
         fontsize=5.4, color="#555555", ha="left", va="bottom", linespacing=1.3)

# ---------------- D: cycle-phase proxy ----------------
axD = fig.add_subplot(gs[1, 1])
rng = np.random.default_rng(20261117)
for gi, (grp, col) in enumerate([("Normal", ACCENT_BLUE), ("RIF", ACCENT_RED)]):
    v = ph.loc[ph.group == grp, "phase_score"].values
    jit = rng.uniform(-0.11, 0.11, len(v))
    axD.scatter(np.full(len(v), gi) + jit, v, s=16, color=col, alpha=0.75,
                edgecolor="white", lw=0.3, zorder=3)
    axD.hlines(np.median(v), gi - 0.22, gi + 0.22, color=BLACK, lw=0.9, zorder=4)
axD.axhline(0, color="#999999", lw=0.4, ls=(0, (3, 2)))
axD.set_xticks([0, 1]); axD.set_xticklabels(["Normal\n(n = 10)", "RIF\n(n = 11)"], fontsize=5.4)
axD.set_ylabel("Secretory − proliferative score", fontsize=5.4, labelpad=1)
axD.set_xlim(-0.45, 1.45)
axD.tick_params(axis="y", labelsize=5.4, length=2, pad=1)
axD.tick_params(axis="x", length=0, pad=2)
axD.set_title("Cycle-phase proxy differs between groups", fontsize=5.8, pad=3)
axD.text(0.02, 0.96, "Mann-Whitney p = 0.022\nsecretory p = 0.062\nproliferative p = 0.218",
         transform=axD.transAxes, fontsize=5.4, va="top", ha="left", color="#555555",
         linespacing=1.3)
axD.set_ylim(-4.6, 2.35)
axD.text(0.02, 0.03, "GEO metadata report neither LH+7 nor\ncycle phase — phase imbalance cannot be excluded",
         transform=axD.transAxes, fontsize=5.4, ha="left", va="bottom",
         color=ACCENT_RED, linespacing=1.3)
panel_letter(axD, "D", x=-0.16, y=1.08)

save_cns_figure(fig, f"{FIG}/FigS9_external_GSE248784")
print("FIG_S10_DONE")
