#!/usr/bin/env python3
"""Fig. S10 (v2 numbering) — direct network-level replication test in GSE248784 (not replicated)."""
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from fig_style import ACCENT_RED, GREY, BLACK, panel_letter, save_cns_figure

# Paths resolve relative to this package; override with RIF_DATA / RIF_DATA2 / RIF_DATA3 / RIF_FIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
BASE = _ROOT
D = f"{BASE}/data/external/gse248784"
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
rep = pd.read_csv(f"{D}/external_network_replication.csv")
ez = pd.read_csv(f"{D}/external_gene_z.csv", index_col=0)
dz = pd.read_csv(f"{BASE}/data/fig2/gene_z_by_type.csv", index_col=0)
gm = pd.read_csv(f"{BASE}/data/fig2/gene_modules.csv")
mods = {int(k): v for k, v in gm.groupby("module").gene.apply(list).items()}

CT_COL = {"Epithelial": "#B2182B", "Endothelial": "#2166AC", "Myeloid": "#E08214",
          "NK cell": "#5AAE61", "Stromal": "#8073AC", "T cell": "#999999"}

fig = plt.figure(figsize=(7.2, 3.05))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.05], wspace=0.32,
              left=0.13, right=0.975, top=0.86, bottom=0.17)

# ---- A: gene-level correlation ----
axA = fig.add_subplot(gs[0, 0])
r = rep.sort_values("spearman_gene_z")
y = np.arange(len(r))
cols = [ACCENT_RED if p < 0.05 else "#9E9E9E" for p in r.p_gene_z]
axA.barh(y, r.spearman_gene_z, color=cols, edgecolor="white", lw=0.4, height=0.62, zorder=2)
axA.axvline(0, color=BLACK, lw=0.5, zorder=3)
for yi, (_, row) in zip(y, r.iterrows()):
    off = 0.004 if row.spearman_gene_z >= 0 else -0.004
    axA.text(row.spearman_gene_z + off, yi, f"{row.spearman_gene_z:+.3f}",
             va="center", ha="left" if row.spearman_gene_z >= 0 else "right", fontsize=5.4)
axA.set_yticks(y); axA.set_yticklabels(r.cell_type, fontsize=5.4)
axA.set_xlim(-0.15, 0.06)
axA.set_xlabel("Spearman ρ: discovery gene z vs external gene z", fontsize=5.4, labelpad=2)
axA.tick_params(axis="x", labelsize=5.4, length=2, pad=1)
axA.tick_params(axis="y", length=0, pad=2)
axA.set_title("Gene-level divergence does not replicate\n(0/6 types positive; median ρ = −0.037; NK ρ = −0.106, p = 0.006)",
              fontsize=5.5, pad=3)
panel_letter(axA, "A", x=-0.24, y=1.16)

# ---- B: module-level scatter ----
axB = fig.add_subplot(gs[0, 1])
xs, ys, cs = [], [], []
for ct in rep.cell_type:
    for m, gs_ in mods.items():
        gg = [g for g in gs_ if g in ez.index and g in dz.index]
        if len(gg) < 5:
            continue
        xs.append(float(dz.loc[gg, ct].mean()))
        ys.append(float(ez.loc[gg, ct].mean()))
        cs.append(CT_COL.get(ct, "#999999"))
xs, ys = np.array(xs), np.array(ys)
rho, p = spearmanr(xs, ys)
lim = (min(xs.min(), ys.min()) - 0.25, max(xs.max(), ys.max()) + 0.25)
axB.plot(lim, lim, color="#777777", lw=0.6, ls=(0, (3, 2)), zorder=1)
axB.axhline(0, color="#cccccc", lw=0.4, zorder=0)
axB.axvline(0, color="#cccccc", lw=0.4, zorder=0)
axB.scatter(xs, ys, s=13, c=cs, edgecolor="white", lw=0.3, zorder=3)
axB.set_xlim(lim); axB.set_ylim(lim)
axB.set_xlabel("Discovery module mean z", fontsize=5.4, labelpad=2)
axB.set_ylabel("External module mean z", fontsize=5.4, labelpad=2)
axB.tick_params(labelsize=5.4, length=2, pad=1)
axB.text(0.03, 0.96, f"ρ = {rho:+.2f}, p = {p:.3f}\n(48 module–type pairs)",
         transform=axB.transAxes, fontsize=5.4, va="top", ha="left")
axB.text(0.03, 0.855, "modules selected on discovery z:\nregression to the mean is expected",
         transform=axB.transAxes, fontsize=5.4, va="top", ha="left",
         color="#666666", linespacing=1.25)
handles = [plt.Line2D([], [], marker="o", ls="none", ms=3.4, mfc=c, mec="white", label=ct)
           for ct, c in CT_COL.items()]
axB.legend(handles=handles, fontsize=5.4, frameon=False, loc="lower right",
           handletextpad=0.3, borderpad=0.1, labelspacing=0.3)
axB.set_title("Module-level concordance is absent", fontsize=5.7, pad=3)
panel_letter(axB, "B", x=-0.16, y=1.16)

save_cns_figure(fig, f"{FIG}/FigS10_external_network_replication")
print("FIG_S11_DONE")
