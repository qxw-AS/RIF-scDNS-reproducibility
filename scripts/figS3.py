#!/usr/bin/env python3
"""Fig. S3 — module stability: gene subsampling, LODO, preservation, split-half."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, pearsonr
from fig_style import (ACCENT_RED, ACCENT_BLUE, GREY, BLACK,
                       DIVERGING, panel_letter, save_cns_figure)

# Paths resolve relative to this package; override with RIF_DATA / RIF_DATA2 / RIF_DATA3 / RIF_FIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
D = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "figS"))
D2 = os.environ.get("RIF_DATA2", os.path.join(_ROOT, "data", "fig2"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
sub = pd.read_csv(f"{D}/s3_gene_subsample.csv")
half = pd.read_csv(f"{D}/s3_half_module_z.csv")
lodo = pd.read_csv(f"{D}/s3_lodo.csv")
def _donor_key(d):
    grp = 0 if d.startswith("LH7") else 1
    tail = d.split("_")[-1]
    num = int(tail) if tail.isdigit() else 0
    pgt = 0 if "PGT" not in d else 1
    return (grp, pgt, num)
lodo = lodo.iloc[sorted(range(len(lodo)), key=lambda i: _donor_key(lodo.left_out.iloc[i]))].reset_index(drop=True)
pres = pd.read_csv(f"{D}/s3_preservation.csv")
mods = pd.read_csv(f"{D2}/gene_modules.csv")
mcolors = plt.cm.tab10(np.arange(8))
sizes = mods.groupby("module").size()

fig = plt.figure(figsize=(7.2, 5.35))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.08], hspace=0.52, wspace=0.42,
              left=0.075, right=0.955, top=0.95, bottom=0.09)

# ---------------- A: gene-subsampling Jaccard ----------------
axA = fig.add_subplot(gs[0, 0])
data = [sub.loc[sub.module == m, "jaccard"].to_numpy() for m in range(1, 9)]
bp = axA.boxplot(data, widths=0.58, patch_artist=True, showfliers=False,
                 medianprops=dict(color=BLACK, lw=0.7),
                 whiskerprops=dict(color="#666666", lw=0.5),
                 capprops=dict(color="#666666", lw=0.5),
                 boxprops=dict(lw=0.4))
for patch, c in zip(bp["boxes"], mcolors):
    patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
for i, d in enumerate(data, 1):
    xj = np.random.default_rng(SEED := 20261117 + i).normal(i, 0.075, len(d))
    axA.scatter(xj, d, s=1.2, color="#555555", alpha=0.35, linewidths=0, zorder=2,
                rasterized=True)
    axA.scatter(i, np.median(d), s=8, color=ACCENT_RED, zorder=3, linewidths=0)
axA.set_xticks(range(1, 9))
axA.set_xticklabels([f"M{m}\n({sizes[m]})" for m in range(1, 9)], fontsize=5.4)
axA.set_xlabel("Module (genes)", fontsize=6, labelpad=2)
axA.set_ylabel("Jaccard vs full-data module", fontsize=6, labelpad=1)
axA.set_ylim(-0.03, 0.95)
axA.tick_params(labelsize=5.5, length=2, pad=1)
axA.axhline(0, color="#BBBBBB", lw=0.4, ls=":")
axA.set_title("80% gene subsampling (100 iterations)", fontsize=6, pad=3)
axA.text(0.02, 0.97, "box: IQR; red: median\nbest match to any full module",
         transform=axA.transAxes, fontsize=5.4, color="#555555", va="top")
panel_letter(axA, "A", x=-0.15, y=1.04)

# ---------------- B: LODO module z + gene-level r ----------------
gsB = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], width_ratios=[4.4, 1.0],
                              wspace=0.30)
axB = fig.add_subplot(gsB[0, 0])
M = lodo[[f"M{m}" for m in range(1, 9)]].to_numpy()
cmap = LinearSegmentedColormap.from_list("div", ["#2166AC", "#F7F7F7", "#B2182B"])
norm = TwoSlopeNorm(vmin=-1.7, vcenter=0, vmax=1.7)
im = axB.imshow(M, aspect="auto", cmap=cmap, norm=norm)
axB.set_xticks(range(8)); axB.set_xticklabels([f"M{m}" for m in range(1, 9)], fontsize=5.4)
axB.set_yticks(range(len(lodo)))
axB.set_yticklabels([f"{d} ({g[0]})" for d, g in zip(lodo.left_out, lodo.group)],
                    fontsize=5.4)
axB.tick_params(length=0, pad=2)
for y in np.arange(-0.5, len(lodo), 1):
    axB.axhline(y, color="white", lw=0.4)
for x in np.arange(-0.5, 8, 1):
    axB.axvline(x, color="white", lw=0.4)
axB.set_title("Leave-one-donor-out (Stromal)", fontsize=6, pad=3)
axB.set_ylabel("Donor", fontsize=5.4, labelpad=2)
cax = axB.inset_axes([1.06, 0.02, 0.030, 0.96])
cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
cb.ax.tick_params(labelsize=5.4, length=1.0, pad=1)
cb.outline.set_linewidth(0.3)

axB2 = fig.add_subplot(gsB[0, 1], sharey=axB)
axB2.barh(np.arange(len(lodo)), lodo.gene_r, height=0.62, color="#9E9E9E",
          edgecolor="white", lw=0.3)
axB2.set_xlim(0.65, 1.03)
axB2.set_xticks([0.7, 0.85, 1.0])
axB2.set_xticklabels(["0.7", "0.85", "1.0"], fontsize=5.4)
axB2.tick_params(axis="x", length=1.5, pad=1)
axB2.tick_params(axis="y", left=False, labelleft=False, length=0)
axB2.set_xlabel("gene ρ", fontsize=5.4, labelpad=1)
axB2.set_title("vs full", fontsize=5.4, pad=3)
axB2.spines["left"].set_visible(False)
panel_letter(axB, "B", x=-0.20, y=1.06)

# ---------------- C: preservation permutation ----------------
axC = fig.add_subplot(gs[1, 0])
pr = pres.sort_values("z_cohesion", ascending=False)
x = np.arange(len(pr))
bars = axC.bar(x, pr.z_cohesion, width=0.62, color="#4393C3", edgecolor="#2166AC",
               lw=0.4)
axC.set_xticks(x)
axC.set_xticklabels([f"M{m}" for m in pr.module], fontsize=5.6)
axC.set_ylabel("Cohesion z vs 999 random gene sets", fontsize=5.8, labelpad=1)
axC.set_ylim(0, 11.5)
axC.tick_params(labelsize=5.4, length=2, pad=1)
axC.set_title("Module preservation", fontsize=6, pad=3)
for xi, (z, p) in enumerate(zip(pr.z_cohesion, pr.p_lower)):
    axC.text(xi, z + 0.18, f"{z:.1f}", ha="center", va="bottom", fontsize=5.4)
axC.text(0.98, 0.92, "all permutation p = 0.001\n(observed variance < null)",
         transform=axC.transAxes, fontsize=5.4, color="#555555", va="top", ha="right")
panel_letter(axC, "C", x=-0.15, y=1.05)

# ---------------- D: split-half module z concordance ----------------
axD = fig.add_subplot(gs[1, 1])
styles = {"cell_random": ("o", ACCENT_BLUE, "cell-random split"),
          "donor_balanced": ("^", ACCENT_RED, "donor-balanced split")}
for scheme, (mk, col, lab) in styles.items():
    g = half[half.scheme == scheme].sort_values("module")
    r = pearsonr(g.z_A, g.z_B).statistic
    axD.scatter(g.z_A, g.z_B, s=16, marker=mk, color=col, edgecolors="white",
                linewidths=0.4, zorder=3, label=f"{lab} (r = {r:.3f})")
lim = (-1.9, 1.9)
axD.plot(lim, lim, color="#888888", lw=0.6, ls="--", zorder=1)
axD.set_xlim(lim); axD.set_ylim(lim)
axD.set_xlabel("Module mean z, half A", fontsize=6, labelpad=1)
axD.set_ylabel("Module mean z, half B", fontsize=6, labelpad=1)
axD.tick_params(labelsize=5.5, length=2, pad=1)
axD.set_aspect("equal", adjustable="box")
axD.legend(fontsize=5.4, loc="upper left", handlelength=1.1, labelspacing=0.3,
           borderpad=0.15, handletextpad=0.3)
axD.set_title("Stromal split-half module z (n = 8 modules)", fontsize=6, pad=3)
panel_letter(axD, "D", x=-0.18, y=1.05)

save_cns_figure(fig, f"{FIG}/FigS3_module_stability")
print("FIG_S3_DONE")
