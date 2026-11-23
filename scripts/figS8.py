#!/usr/bin/env python3
"""Fig. S11 (v2 numbering) — H2 bridge condition: network-only genes vs Cao 2025 dynamic sets (phase untestable)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import blended_transform_factory
from fig_style import (ACCENT_RED, ACCENT_BLUE, GREY, BLACK, CATEGORICAL,
                       panel_letter, save_cns_figure)

# Paths resolve relative to this package; override with RIF_DATA / RIF_DATA2 / RIF_DATA3 / RIF_FIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
D = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "figS"))
D2 = os.environ.get("RIF_DATA2", os.path.join(_ROOT, "data", "fig2"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
TYPES = ["Epithelial", "Glandular", "Stromal"]

stats = pd.read_csv(f"{D}/s8_bridge_stats.csv")
bg = pd.read_csv(f"{D}/s8_cao_dynamic.csv")
br = pd.read_csv(f"{D}/s8_bridge_genes.csv")
Z = pd.read_csv(f"{D2}/gene_z_by_type.csv", index_col=0)

# deduplicate bridge rows: keep the compartment with the highest MI contribution
br_max = br.sort_values("cao_MI", ascending=False).drop_duplicates(["cell_type", "gene"])
br_max = br_max.set_index(["cell_type", "gene"])
br_cnt = br.groupby(["cell_type", "gene"]).cao_sheet.nunique()

fig = plt.figure(figsize=(7.2, 6.35))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.28, 0.92], width_ratios=[0.38, 0.62],
              hspace=0.52, wspace=0.34, left=0.13, right=0.965, top=0.94, bottom=0.08)

# ---------------- A: network-only and bridge counts ----------------
axA = fig.add_subplot(gs[0, 0])
x = np.arange(len(stats))
w = 0.36
axA.bar(x - w / 2, stats.n_network_only, width=w, color="#BDBDBD",
        edgecolor="#7A7A7A", lw=0.3, label="network-only")
axA.bar(x + w / 2, stats.n_bridge, width=w, color=ACCENT_RED, edgecolor="#7A2A2A",
        lw=0.3, label="∩ Cao dynamic")
for xi, (n, k, p) in enumerate(zip(stats.n_network_only, stats.n_bridge, stats.hypergeom_p)):
    axA.text(xi - w / 2, n + 0.7, str(n), ha="center", fontsize=5.4)
    axA.text(xi + w / 2, k + 0.7, str(k), ha="center", fontsize=5.4, color=ACCENT_RED,
             fontweight="bold")
    axA.text(xi, -6.2, f"p = {p:.3f}", ha="center", fontsize=5.4, color="#444444")
axA.axhline(3, color="#555555", lw=0.6, ls=(0, (3, 2)))
axA.text(0.42, 3.7, "criterion ≥3", fontsize=5.4, ha="center", color="#555555")
axA.set_xticks(x); axA.set_xticklabels(stats.cell_type, fontsize=5.4)
axA.set_ylabel("Genes (top-50 divergence)", fontsize=5.6, labelpad=1)
axA.set_ylim(0, 42)
axA.tick_params(axis="y", labelsize=5.4, length=2, pad=1)
axA.tick_params(axis="x", length=2, pad=1)
axA.legend(fontsize=5.4, loc="upper left", handlelength=1.0, borderpad=0.15)
axA.set_title("Bridge counts (hypergeometric p)", fontsize=5.8, pad=3)
panel_letter(axA, "A", x=-0.28, y=1.06)

# ---------------- B: bridge gene dot table ----------------
axB = fig.add_subplot(gs[0, 1])
rows = []
for ct in TYPES:
    d = br_max.loc[ct].reset_index()
    d["cell_type"] = ct
    rows.append(d.sort_values("combined_z"))
tbl = pd.concat(rows, ignore_index=True)
yt, ylab, y = [], [], 0
trans = blended_transform_factory(axB.transAxes, axB.transData)
for ct in TYPES:
    d = tbl[tbl.cell_type == ct]
    y0 = y
    for _, r in d.iterrows():
        z = Z.loc[r.gene, ct] if r.gene in Z.index else np.nan
        cl = int(r.cao_cluster)
        axB.scatter(z, y, s=10 + 55 * r.cao_MI, c=CATEGORICAL[(cl - 1) % 6],
                    edgecolor="white", lw=0.35, zorder=3)
        ylab.append(r.gene); yt.append(y)
        num = "—" if pd.isna(r.combined_z) else f"{r.combined_z:+.2f}"
        axB.text(4.75, y, num, fontsize=5.4, ha="right", va="center", color="#444444")
        y += 1
    axB.axhline(y - 0.5, color="#B0B0B0", lw=0.5)
    axB.text(-0.135, (y0 + y - 1) / 2, ct, fontsize=5.4, rotation=90,
             ha="center", va="center", color="#333333", fontweight="bold",
             transform=trans, clip_on=False)
axB.axvline(0, color="#888888", lw=0.45, ls=(0, (2, 2)))
axB.set_yticks(yt); axB.set_yticklabels(ylab, fontsize=5.4, style="italic")
axB.set_ylim(y - 0.5, -0.5)
axB.set_xlim(-2.7, 4.9)
axB.set_xticks([-2, -1, 0, 1, 2, 3, 4])
axB.set_xlabel("Type-specific divergence z", fontsize=5.4, labelpad=1)
axB.tick_params(axis="x", labelsize=5.4, length=2, pad=1)
axB.tick_params(axis="y", length=0, pad=2)
axB.set_title("Bridge genes: divergence z and Cao dynamic cluster", fontsize=5.6, pad=3)
axB.text(4.88, -0.85, "combined z", fontsize=5.4, ha="right", va="bottom",
         color="#444444")
panel_letter(axB, "B", x=-0.30, y=1.05)

# ---------------- C: Cao dynamic-score distributions ----------------
gsC = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], width_ratios=[1, 0.55],
                              wspace=0.45)
axC = fig.add_subplot(gsC[0])
order = ["Glandular", "Secretory", "Stromal"]
data, counts = [], []
for s in order:
    d = bg[bg.compartment == s]
    data.append(d.cao_MI.dropna().values)
    counts.append(len(d))
bp = axC.boxplot(data, widths=0.48, patch_artist=True, showfliers=False,
                 medianprops=dict(color="#333333", lw=0.8),
                 whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                 boxprops=dict(lw=0.5))
for patch, c in zip(bp["boxes"], ["#9ECAE1", "#F4A582", "#A1D99B"]):
    patch.set_facecolor(c); patch.set_edgecolor("#666666")
for i, s in enumerate(order):
    d = br[br.cao_sheet == s].drop_duplicates("gene")
    xj = np.random.default_rng(20261117 + i).normal(i + 1, 0.055, len(d))
    axC.scatter(xj, d.cao_MI, s=14, color=ACCENT_RED, edgecolor="white", lw=0.3,
                zorder=4)
axC.set_xticks([1, 2, 3])
axC.set_xticklabels([f"{s}\n(n={c})" for s, c in zip(order, counts)], fontsize=5.4)
axC.set_ylabel("Cao dynamic-gene MI score", fontsize=5.6, labelpad=1)
axC.tick_params(axis="y", labelsize=5.4, length=2, pad=1)
axC.tick_params(axis="x", length=0, pad=2)
axC.set_title("Dynamic score of bridge genes", fontsize=5.6, pad=3)
axC.text(0.02, 0.97, "red: bridge genes", transform=axC.transAxes, fontsize=5.4,
         color=ACCENT_RED, va="top")
panel_letter(axC, "C", x=-0.26, y=1.08)

# cluster composition
axC2 = fig.add_subplot(gsC[1])
cl_counts = br.drop_duplicates(["cao_sheet", "gene"]).groupby(
    ["cao_sheet", "cao_cluster"]).size().unstack(fill_value=0).reindex(order)
bottom = np.zeros(len(order))
for cl in sorted(br.cao_cluster.unique()):
    if cl not in cl_counts.columns:
        continue
    vals = cl_counts[cl].values
    axC2.bar(np.arange(len(order)), vals, bottom=bottom, width=0.55,
             color=CATEGORICAL[(int(cl) - 1) % 6], edgecolor="white", lw=0.4,
             label=f"C{int(cl)}")
    bottom += vals
axC2.set_xticks(range(len(order)))
axC2.set_xticklabels(["Gland.", "Secret.", "Stromal"], fontsize=5.4)
axC2.set_ylabel("Bridge genes", fontsize=5.4, labelpad=1)
axC2.set_ylim(0, 15.5)
axC2.tick_params(labelsize=5.4, length=2, pad=1)
axC2.legend(fontsize=5.4, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.02),
            handlelength=0.7, columnspacing=0.5, borderpad=0.05,
            labelspacing=0.2, title="Cao cluster", title_fontsize=5.4,
            frameon=False, handletextpad=0.4)

# ---------------- D: decision flow ----------------
axD = fig.add_subplot(gs[1, 1])
axD.set_axis_off()
axD.set_xlim(0, 1); axD.set_ylim(0, 1)
boxes = [
    ("H1: divergence ≠ DEG\nnetwork-only top-50, 11/11 types (16–50)  →  supported",
     "#DCE6F2", "#2166AC"),
    ("Bridge: network-only ∩ Cao dynamic genes\nEpithelial 10/18 · Glandular 7/16 · Stromal 3/38\n→ count criterion (≥3) met in all three",
     "#DFF0D8", "#1B7837"),
    ("Enrichment (descriptive)\nhypergeometric p = 0.023 · 0.025 · 0.367\nnominal in 2/3 compartments",
     "#EFEFEF", "#666666"),
    ("H2 phase test: NOT TESTABLE\nCao LH+3/+11 samples are fertile-only;\nLH+7 is the only RIF/fertile time point",
     "#FBE3D6", "#B2182B"),
]
ypos = [0.77, 0.52, 0.28, 0.05]
hgt = 0.185
for (txt, fc, ec), yb in zip(boxes, ypos):
    axD.add_patch(FancyBboxPatch((0.02, yb), 0.96, hgt,
                                 boxstyle="round,pad=0.008,rounding_size=0.015",
                                 fc=fc, ec=ec, lw=0.6))
    axD.text(0.5, yb + hgt / 2, txt, ha="center", va="center", fontsize=5.4,
             linespacing=1.35)
    if yb > 0.06:
        axD.add_patch(FancyArrowPatch((0.5, yb), (0.5, yb - 0.045),
                                      arrowstyle="-|>", mutation_scale=5.5,
                                      lw=0.6, color="#555555"))
panel_letter(axD, "D", x=-0.02, y=1.02)

save_cns_figure(fig, f"{FIG}/FigS11_H2_bridge_Cao")
print("FIG_S8_DONE", stats[["cell_type", "n_bridge", "hypergeom_p"]].to_dict("records"))
