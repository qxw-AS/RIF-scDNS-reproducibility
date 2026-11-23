#!/usr/bin/env python3
"""Fig. S7 (v2 numbering) — donor-level robustness of module/HLA/IFN signals."""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from fig_style import ACCENT_RED, ACCENT_BLUE, GREY, BLACK, panel_letter, save_cns_figure

# Paths resolve relative to this package; override with RIF_DATA / RIF_DATA2 / RIF_DATA3 / RIF_FIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
D = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "figS"))
D3 = os.environ.get("RIF_DATA3", os.path.join(_ROOT, "data", "fig3"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
CT = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
      "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]

# ---------------- A: split-half reproducibility ----------------
rows = []
m = pd.read_csv(f"{D3}/split_half_donor_metrics.csv").iloc[0]
rows.append(("Donor split\n(original)", m.gene_level_r, 0.100, "#999999"))
for f in sorted(glob.glob(f"{D3}/split_half_donor_equal_metrics_rep*.csv")):
    m = pd.read_csv(f).iloc[0]
    rep = f.split("rep")[-1].split(".")[0]
    j = pd.read_csv(f"{D3}/split_half_donor_equal_jaccard.csv")
    jv = float(j[(j.rep == int(rep)) & (j.N == 50) & (j.kind == "network_only")].observed.iloc[0])
    rows.append((f"Donor-equal\nrep {int(rep) + 1}", m.gene_level_r, jv, "#B2182B"))

fig = plt.figure(figsize=(7.05, 5.35))
gs = GridSpec(2, 2, figure=fig, height_ratios=[0.82, 1.0], width_ratios=[0.42, 0.58],
              hspace=0.46, wspace=0.30, left=0.085, right=0.965, top=0.955, bottom=0.10)
axA = fig.add_subplot(gs[0, 0])
x = np.arange(len(rows))
axA.bar(x, [r[1] for r in rows], width=0.58, color=[r[3] for r in rows],
        edgecolor="#666666", lw=0.4, zorder=2)
for xi, r in zip(x, rows):
    axA.text(xi, r[1] + 0.012, f"{r[1]:.2f}", ha="center", va="bottom", fontsize=5.4)
axA.set_ylim(0, 0.42)
axA.set_ylabel("Gene-level z correlation (r)", fontsize=5.8, labelpad=1)
axA.set_xticks(x)
axA.set_xticklabels([f"{r[0]}\nJ50 = {r[2]:.3f}" for r in rows], fontsize=5.4)
axA.tick_params(axis="y", labelsize=5.4, length=2, pad=1)
axA.tick_params(axis="x", length=0, pad=2)
axA.text(0.02, 0.96, "bars: gene-level r\nJ50 (network-only) given per split",
         transform=axA.transAxes, fontsize=5.4, va="top", ha="left",
         color="#555555", linespacing=1.3)
axA.set_title("Donor-level reproducibility is not rescued\nby cell-number equalization",
              fontsize=5.8, pad=3)
panel_letter(axA, "A", x=-0.20, y=1.10)

# ---------------- B: donor-level effect sizes ----------------
t = pd.read_csv(f"{D}/s9_donor_tests.csv")
feat_order = ["M2", "HLA_score", "IFN_score"]
feat_lab = ["M2 module", "HLA score", "IFN score"]
M = np.full((len(feat_order), len(CT)), np.nan)
Q = np.full_like(M, np.nan)
P = np.full_like(M, np.nan)
for i, f in enumerate(feat_order):
    for j, ct in enumerate(CT):
        r = t[(t.feature == f) & (t.cell_type == ct)]
        if len(r):
            M[i, j] = r.delta.iloc[0]
            Q[i, j] = r.q.iloc[0]
            P[i, j] = r.p.iloc[0]

axB = fig.add_subplot(gs[0, 1])
cmap = LinearSegmentedColormap.from_list("bwr", ["#2166AC", "#F7F7F7", "#B2182B"])
vmax = np.nanmax(np.abs(M))
im = axB.imshow(M, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax))
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isnan(M[i, j]):
            axB.text(j, i, "—", ha="center", va="center", fontsize=5.4, color="#999999")
        else:
            star = "*" if Q[i, j] < 0.05 else ""
            axB.text(j, i, f"{M[i, j]:+.2f}{star}", ha="center", va="center", fontsize=5.4,
                     fontweight="bold" if Q[i, j] < 0.05 else "normal",
                     color="white" if abs(M[i, j]) > 0.62 * vmax else BLACK)
axB.set_xticks(range(len(CT))); axB.set_xticklabels(CT, rotation=45, ha="right", fontsize=5.4)
axB.set_yticks(range(len(feat_order))); axB.set_yticklabels(feat_lab, fontsize=5.4)
axB.set_xticks(np.arange(-0.5, len(CT), 1), minor=True)
axB.set_yticks(np.arange(-0.5, len(feat_order), 1), minor=True)
axB.grid(which="minor", color="white", lw=0.6)
axB.tick_params(which="both", length=0)
axB.set_title("Donor-level effect size (RIF − Fertile);\n0/130 tests BH q < 0.05",
              fontsize=5.8, pad=3)
panel_letter(axB, "B", x=-0.12, y=1.10)
cb = fig.colorbar(im, ax=axB, fraction=0.030, pad=0.02)
cb.ax.tick_params(labelsize=5.4, length=1.5)
cb.outline.set_linewidth(0.3)
cb.set_label("Δ score", fontsize=5.4, labelpad=1)

# ---------------- C: donor-level correlations ----------------
c = pd.read_csv(f"{D}/s9_donor_correlations.csv")
pairs = [("M2_score", "IFN_score", "M2 – IFN"),
         ("M2_score", "HLA_score", "M2 – HLA"),
         ("HLA_score", "IFN_score", "HLA – IFN")]
R = np.full((len(pairs), len(CT)), np.nan)
Qc = np.full_like(R, np.nan)
for i, (xcol, ycol, _) in enumerate(pairs):
    for j, ct in enumerate(CT):
        r = c[(c.x == xcol) & (c.y == ycol) & (c.cell_type == ct)]
        if len(r):
            R[i, j] = r.rho.iloc[0]
            Qc[i, j] = r.q.iloc[0]

axC = fig.add_subplot(gs[1, :])
im = axC.imshow(R, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -1, 1))
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if np.isnan(R[i, j]):
            axC.text(j, i, "—", ha="center", va="center", fontsize=5.4, color="#999999")
        else:
            star = "*" if Qc[i, j] < 0.05 else ""
            axC.text(j, i, f"{R[i, j]:+.2f}{star}", ha="center", va="center", fontsize=5.4,
                     fontweight="bold" if Qc[i, j] < 0.05 else "normal",
                     color="white" if abs(R[i, j]) > 0.62 else BLACK)
axC.set_xticks(range(len(CT))); axC.set_xticklabels(CT, rotation=45, ha="right", fontsize=5.4)
axC.set_yticks(range(len(pairs))); axC.set_yticklabels([p[2] for p in pairs], fontsize=5.6)
axC.set_xticks(np.arange(-0.5, len(CT), 1), minor=True)
axC.set_yticks(np.arange(-0.5, len(pairs), 1), minor=True)
axC.grid(which="minor", color="white", lw=0.6)
axC.tick_params(which="both", length=0)
axC.set_title("Donor-level Spearman correlations (* BH q < 0.05)", fontsize=5.8, pad=3)
panel_letter(axC, "C", x=-0.055, y=1.10)
cb = fig.colorbar(im, ax=axC, fraction=0.020, pad=0.015)
cb.ax.tick_params(labelsize=5.4, length=1.5)
cb.outline.set_linewidth(0.3)
cb.set_label("ρ", fontsize=5.4, labelpad=1)

save_cns_figure(fig, f"{FIG}/FigS7_donor_robustness")
print("FIG_S9_DONE")
