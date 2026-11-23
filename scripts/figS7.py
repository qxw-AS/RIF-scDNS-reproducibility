#!/usr/bin/env python3
"""Fig. S8 (v2 numbering) — external bulk comparison, GSE106602 (inconclusive)."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from fig_style import ACCENT_RED, ACCENT_BLUE, GREY, BLACK, panel_letter, save_cns_figure

# Paths resolve relative to this package; override with RIF_DATA / RIF_DATA2 / RIF_DATA3 / RIF_FIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
D = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "figS"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
mod = pd.read_csv(f"{D}/s7_external_modules.csv")
g = pd.read_csv(f"{D}/s7_external_m2_genes.csv")
feat = pd.read_csv(f"{D}/s7_external_features.csv")
summary = json.load(open(f"{D}/s7_external_summary.json"))

fig = plt.figure(figsize=(7.2, 5.6))
gs = GridSpec(2, 2, figure=fig, width_ratios=[0.46, 0.54],
              height_ratios=[0.95, 1.0], hspace=0.42, wspace=0.30,
              left=0.085, right=0.965, top=0.955, bottom=0.10)

# ---------------- A: workflow / cohort ----------------
axA = fig.add_subplot(gs[0, 0]); axA.set_axis_off()
axA.set_xlim(0, 1); axA.set_ylim(0, 1)
boxes = [
    ("GSE106602\nbulk RNA-seq\n70 samples", "#E8EEF6"),
    ("LH+7 mid-secretory\n19 IVF failure\n16 healthy", "#DCE6F2"),
    ("63,677 genes\n458/470 module\ngenes mapped", "#DCE6F2"),
    ("Direction consistency\nonly; no cell-type\nresolution", "#F6E3E3"),
]
for i, (txt, col) in enumerate(boxes):
    x = 0.03 + (i % 2) * 0.49
    y = 0.56 - (i // 2) * 0.46
    axA.add_patch(FancyBboxPatch((x, y), 0.45, 0.36,
                                 boxstyle="round,pad=0.012,rounding_size=0.03",
                                 fc=col, ec="#8A8A8A", lw=0.5))
    axA.text(x + 0.225, y + 0.18, txt, ha="center", va="center", fontsize=5.4,
             color=BLACK, linespacing=1.35)
    if i % 2 == 0:
        axA.add_patch(FancyArrowPatch((x + 0.455, y + 0.18), (x + 0.485, y + 0.18),
                                      arrowstyle="-|>", mutation_scale=6, lw=0.6, color="#555555"))
    if i < 2:
        axA.add_patch(FancyArrowPatch((x + 0.225, y - 0.01), (x + 0.225, y - 0.095),
                                      arrowstyle="-|>", mutation_scale=6, lw=0.6, color="#555555"))
axA.set_title("External cohort is bulk tissue, not cell-type-resolved", fontsize=5.8, pad=3)
panel_letter(axA, "A", x=-0.02, y=1.03)

# ---------------- B: module direction consistency ----------------
axB = fig.add_subplot(gs[0, 1])
axB.axhline(0, color="#888888", lw=0.5, ls=(0, (3, 2)))
axB.axvline(0, color="#888888", lw=0.5, ls=(0, (3, 2)))
for _, r in mod.iterrows():
    col = ACCENT_BLUE if r.direction_concordant else ACCENT_RED
    axB.scatter(r.discovery_combined_z, r.external_delta, s=30, color=col,
                edgecolor="white", lw=0.4, zorder=3)
    axB.annotate(r.module, (r.discovery_combined_z, r.external_delta),
                 xytext=(4, 2), textcoords="offset points", fontsize=5.4, color=col,
                 fontweight="bold" if r.module == "M2" else "normal")
axB.set_xlabel("Discovery module mean combined z\n(network divergence direction)", fontsize=5.4, labelpad=1)
axB.set_ylabel("External module expression Δ\n(IVF failure − healthy)", fontsize=5.4, labelpad=1)
axB.tick_params(labelsize=5.4, length=2, pad=1)
axB.set_title(f"Module direction: {summary['module_direction_concordant']}/8 concordant; M2, M3, M5 discordant",
              fontsize=5.8, pad=3)
axB.text(0.03, 0.04, "bulk expression\n≠ network rewiring", transform=axB.transAxes,
         fontsize=5.4, color="#555555", va="bottom")
panel_letter(axB, "B", x=-0.12, y=1.05)

# ---------------- C: M2 gene-level concordance ----------------
axC = fig.add_subplot(gs[1, 0])
m2 = g.copy()
cols = [ACCENT_RED if x in ["HLA-A", "HLA-B", "HLA-C"] else "#B8B8B8" for x in m2.gene]
axC.axhline(0, color="#888888", lw=0.5, ls=(0, (3, 2)))
axC.axvline(0, color="#888888", lw=0.5, ls=(0, (3, 2)))
axC.scatter(m2.combined_z, m2.external_log2FC, s=18, c=cols, edgecolor="white", lw=0.3, zorder=3)
for _, r in m2[m2.gene.isin(["HLA-A", "HLA-B", "HLA-C"])].iterrows():
    axC.annotate(r.gene, (r.combined_z, r.external_log2FC), xytext=(4, 2),
                 textcoords="offset points", fontsize=5.4, color=ACCENT_RED)
axC.set_xlabel("Discovery M2 combined z (network score)", fontsize=5.4, labelpad=1)
axC.set_ylabel("External log2FC\n(IVF failure − healthy)", fontsize=5.4, labelpad=1)
axC.tick_params(labelsize=5.4, length=2, pad=1)
axC.set_title("M2 gene expression direction does not replicate", fontsize=5.8, pad=3)
axC.text(0.97, 0.96, f"Spearman ρ = {summary['m2_spearman_rho']:.2f} (p = {summary['m2_spearman_p']:.2f})\n"
                     f"sign concordance = {summary['m2_sign_concordance']*100:.0f}%  (n = {summary['m2_genes_mapped']})",
         transform=axC.transAxes, fontsize=5.4, va="top", ha="right", color="#555555")
axC.set_xlim(m2.combined_z.min() - 0.4, m2.combined_z.max() + 0.75)
panel_letter(axC, "C", x=-0.20, y=1.05)

# ---------------- D: raw vs adjusted effects ----------------
axD = fig.add_subplot(gs[1, 1])
order = ["M2", "HLA_score", "IFN_score"]
lab = ["M2 module", "HLA score", "IFN score"]
f = feat.set_index("feature")
m2raw = mod.set_index("module").loc["M2"]
raw_map = {"M2": (float(m2raw.external_delta), float(m2raw.q)), }
nd = pd.read_csv(os.path.join(_ROOT, "data", "external", "negative_diagnostics.csv"))
adj = (nd[nd.dataset == "GSE106602"]
       .set_index("check")
       .loc[[f"adj group (phase+det) {k}" for k in order]])
for i, k in enumerate(order):
    yc = len(order) - 1 - i
    if k == "M2":
        raw, q = raw_map["M2"]
    else:
        raw, q = f.loc[k, "delta"], f.loc[k, "q"]
    b = float(adj.loc[f"adj group (phase+det) {k}", "value"].split("=")[1].split()[0])
    p = float(adj.loc[f"adj group (phase+det) {k}", "p"])
    axD.plot([raw, b], [yc + 0.17, yc - 0.17], color="#CCCCCC", lw=0.7, zorder=1)
    axD.scatter(raw, yc + 0.17, s=24, color=ACCENT_RED if q < 0.05 else "#999999",
                edgecolor="white", lw=0.4, zorder=3)
    axD.scatter(b, yc - 0.17, s=24, marker="s", color="#555555",
                edgecolor="white", lw=0.4, zorder=3)
    axD.text(raw + (-0.04 if raw < 0 else 0.04), yc + 0.17, f"q = {q:.3f}",
             va="center", ha="right" if raw < 0 else "left", fontsize=5.4)
    axD.text(b + (-0.04 if b < 0 else 0.04), yc - 0.17, f"p = {p:.2f}",
             va="center", ha="right" if b < 0 else "left", fontsize=5.4, color="#555555")
axD.axvline(0, color="#888888", lw=0.5, ls=(0, (3, 2)))
axD.set_yticks(range(len(order)))
axD.set_yticklabels(lab[::-1], fontsize=5.4)
axD.set_ylim(-0.6, len(order) - 0.4)
axD.set_xlim(-1.05, 0.12)
axD.set_xlabel("Δ expression (IVF failure − healthy)", fontsize=5.4, labelpad=1)
axD.tick_params(axis="x", labelsize=5.4, length=2, pad=1)
axD.tick_params(axis="y", length=0, pad=2)
axD.set_title("Effects attenuate after technical adjustment", fontsize=5.8, pad=3)
axD.text(0.02, 0.97, "raw (circle) vs adjusted for phase +\n"
                     "genes detected (square); no adjusted p < 0.05",
         transform=axD.transAxes, fontsize=5.4, color="#555555",
         va="top", ha="left", linespacing=1.3)
panel_letter(axD, "D", x=-0.16, y=1.05)

save_cns_figure(fig, f"{FIG}/FigS8_external_GSE106602")
print("FIG_S7_DONE")
