#!/usr/bin/env python3
"""Fig. S4 — M2/MHC-I composition, HLA expression, module GO, IFN score."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from fig_style import (ACCENT_RED, ACCENT_BLUE, GREY, BLACK,
                       panel_letter, save_cns_figure)

# Paths resolve relative to this package; override with RIF_DATA / RIF_DATA2 / RIF_DATA3 / RIF_FIG.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
D = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "figS"))
D2 = os.environ.get("RIF_DATA2", os.path.join(_ROOT, "data", "fig2"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
CT = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
      "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]
HLA = ["HLA-A", "HLA-B", "HLA-C"]

exp = pd.read_csv(f"{D}/s4_expression.csv")
ifn = pd.read_csv(f"{D}/s4_ifn_score.csv")
m2c = pd.read_csv(f"{D}/s4_m2_contribution.csv")
mods = pd.read_csv(f"{D2}/gene_modules.csv")
go_all = pd.read_csv(f"{D2}/go_enrichment_full.csv")

fig = plt.figure(figsize=(7.1, 7.55))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.14, 0.62],
              width_ratios=[0.42, 0.58], hspace=0.42, wspace=0.34,
              left=0.165, right=0.95, top=0.955, bottom=0.075)

# ---------------- A: HLA-A/B/C expression dot plot ----------------
gsA = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 0], wspace=0.10)
cmap = LinearSegmentedColormap.from_list("bl", ["#F7FBFF", "#9ECAE1", "#2171B5", "#08306B"])
hla = exp[exp.gene.isin(HLA)]
vmin, vmax = hla.mean_expr.min(), hla.mean_expr.max()
size0, size_k = 5, 0.62
for j, g in enumerate(HLA):
    ax = fig.add_subplot(gsA[0, j])
    d = hla[hla.gene == g]
    for xi, grp in enumerate(["RIF", "Fertile"]):
        dd = d[d.group == grp].set_index("cell_type").reindex(CT)
        ax.scatter(np.full(len(CT), xi), np.arange(len(CT)),
                   s=size0 + size_k * dd.pct_expressed, c=dd.mean_expr,
                   cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="#4D4D4D",
                   linewidth=0.3, zorder=3)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(len(CT) - 0.5, -0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["RIF", "Fertile"], fontsize=5.4)
    ax.set_yticks(range(len(CT)))
    if j == 0:
        ax.set_yticklabels(CT, fontsize=5.4)
    else:
        ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.set_title(g, fontsize=6.0, style="italic", pad=2)
    for y in np.arange(-0.5, len(CT), 1):
        ax.axhline(y, color="#EDEDED", lw=0.35, zorder=0)
    if j == 2:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        cb = fig.colorbar(sm, ax=ax, fraction=0.055, pad=0.05)
        cb.ax.tick_params(labelsize=5.4, length=1.5)
        cb.outline.set_linewidth(0.3)
        cb.set_label("Mean expression (log1p)", fontsize=5.4, labelpad=2)
    if j == 0:
        panel_letter(ax, "A", x=-0.36, y=1.06)
        ax.text(0.0, -1.55, "dot size: % cells expressing", fontsize=5.4,
                ha="left", va="bottom", color="#555555", clip_on=False)

# ---------------- B: HLA vs non-HLA contribution to M2 mean z ----------------
axB = fig.add_subplot(gs[0, 1])
m2 = mods[mods.module == 2].copy()
Z = pd.read_csv(os.path.join(_ROOT, "data", "fig2", "gene_z_by_type.csv"), index_col=0)
CT_ORDER = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
            "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]
hla = [g for g in m2.gene if g in HLA]
other = [g for g in m2.gene if g not in HLA]
zh = Z.reindex(m2.gene)[CT_ORDER].loc[hla].sum() / len(m2)
zo = Z.reindex(m2.gene)[CT_ORDER].loc[other].sum() / len(m2)
contrib = pd.DataFrame({"HLA-A/B/C": zh, "other 43 genes": zo}).sort_values("HLA-A/B/C")
xb = np.arange(len(contrib))
axB.bar(xb, contrib["HLA-A/B/C"], color=ACCENT_RED, edgecolor="white", lw=0.4,
        label="HLA-A/B/C", zorder=2)
axB.bar(xb, contrib["other 43 genes"], bottom=contrib["HLA-A/B/C"], color="#BFBFBF",
        edgecolor="white", lw=0.4, label="other 43", zorder=2)
axB.axhline(0, color="#555555", lw=0.5)
axB.set_xticks(xb)
axB.set_xticklabels(contrib.index, rotation=90, fontsize=5.4)
axB.set_ylabel("Contribution to M2 mean z", fontsize=5.6, labelpad=1)
axB.tick_params(axis="y", labelsize=5.4, length=2, pad=1)
axB.tick_params(axis="x", length=1.5, pad=1)
axB.set_ylim(-0.45, 1.45)
share = (zh / (zh + zo) * 100)
axB.set_title("HLA-A/B/C account for ≤24% of M2 mean z", fontsize=6.0, pad=3)
axB.text(0.02, 0.965, "top-ranked genes, but not the whole module\n"
                      f"HLA share: {share.min():.0f}–{share.max():.0f}% across types",
         transform=axB.transAxes, fontsize=5.4, color="#555555", va="top",
         linespacing=1.3)
axB.legend(fontsize=5.4, ncol=2, loc="lower right", frameon=False,
           handlelength=0.8, columnspacing=0.7, borderpad=0.1, handletextpad=0.3)
panel_letter(axB, "B", x=-0.075, y=1.05)

# ---------------- C: module GO top-5 dot map ----------------
gsC = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :], wspace=0.65,
                              width_ratios=[1, 1, 0.045])
blocks = [range(1, 5), range(5, 9)]
nsize = []
for m in range(1, 9):
    d = go_all[go_all.module == m].copy()
    nsize += d.sort_values("Adjusted P-value").head(5).Genes.fillna("").apply(
        lambda s: len([x for x in str(s).split(";") if x])).tolist()
smin, smax = min(nsize), max(nsize)


CUSTOM = {
    "Nucleotide Metabolic Process": "Nucleotide metabolism",
    "Cytoplasmic Sequestering Of Protein": "Cytoplasmic sequest.",
    "Steroid Biosynthetic Process": "Steroid biosynthesis",
    "Type I Interferon-Mediated Signaling Pathway": "Type I IFN signaling",
    "Modulation By Host Of Viral Genome Replication": "Viral genome replication",
    "Antigen Processing And Presentation Of Endogenous Peptide Antigen Via MHC Class I Via ER Pathway": "MHC-I ER pathway",
    "Antigen Processing And Presentation Of Endogenous Peptide Antigen Via MHC Class I Via ER Pathway, TAP-independent": "MHC-I ER, TAP-indep.",
    "Antigen Processing And Presentation Of Endogenous Peptide Antigen Via MHC Class Ib": "MHC-Ib endogenous",
    "Antigen Processing And Presentation Of Peptide Antigen Via MHC Class Ib": "MHC-Ib peptide Ag",
    "Antigen Processing And Presentation Of Endogenous Peptide Antigen": "Endogenous peptide Ag",
    "Phosphatidylinositol Phosphate Biosynthetic Process": "PI phosphate biosyn.",
    "Positive Regulation Of Organelle Assembly": "Organelle assembly (up)",
    "Negative Regulation Of B Cell Activation": "B cell activation (neg.)",
    "Neutrophil Extravasation": "Neutrophil extravas.",
    "Nucleotide-Binding Domain, Leucine Rich Repeat Containing Receptor Signaling Pathway": "NOD-like receptor path.",
    "Insulin Secretion": "Insulin secretion",
    "Peptide Hormone Secretion": "Peptide hormone secr.",
    "Cardiac Muscle Cell-Cardiac Muscle Cell Adhesion": "Cardiac myocyte adhesion",
    "Regulation Of Cell Communication": "Cell communication (r.)",
    "Blood Circulation": "Blood circulation",
    "Basement Membrane Organization": "Basement membrane org.",
    "Positive Regulation Of Multicellular Organismal Process": "Multicellular (pos.)",
    "Positive Regulation Of Cellular Process": "Cellular process (pos.)",
    "Actin Filament Capping": "Actin filament capping",
    "Barbed-End Actin Filament Capping": "Barbed-end capping",
    "Regulation Of Autophagy Of Mitochondrion": "Mitochondrial autophagy",
    "Smooth Muscle Contraction": "Smooth muscle contract.",
    "Regulation Of Mitochondrion Organization": "Mitochondrial org.",
    "Cilium Assembly": "Cilium assembly",
    "Cilium Organization": "Cilium organization",
    "Regulation Of Double-Strand Break Repair Via Nonhomologous End Joining": "DSB repair via NHEJ",
    "Positive Regulation Of Double-Strand Break Repair": "DSB repair (positive)",
    "Positive Regulation Of Double-Strand Break Repair Via Nonhomologous End Joining": "DSB-NHEJ (positive)",
    "Regulation Of Amide Metabolic Process": "Amide metabolism (reg.)",
    "Positive Regulation Of Erythrocyte Differentiation": "Erythrocyte (pos.)",
    "Establishment Of Protein Localization To Organelle": "Organelle targeting",
    "Regulation Of Neural Precursor Cell Proliferation": "Neural precursor (pos.)",
    "Regulation Of Monocyte Differentiation": "Monocyte differentiation",
    "Cellular Response To cAMP": "cAMP response",
    "Negative Regulation Of Protein Kinase Activity": "Protein kinase (neg.)",
}


def short(s, n=23):
    s = str(s)
    if s in CUSTOM:
        return CUSTOM[s]
    for a, b in [("Antigen Processing And Presentation", "Antigen processing & presentation"),
                 ("Positive Regulation", "Pos. regulation"),
                 ("Negative Regulation", "Neg. regulation"),
                 ("Metabolic Process", "metabolic proc."),
                 ("Biosynthetic Process", "biosynthetic proc."),
                 ("Organization", "org."),
                 ("Differentiation", "diff."),
                 ("Communication", "comm.")]:
        s = s.replace(a, b)
    return s if len(s) <= n else s[:n - 1] + "…"


for bi, mods_b in enumerate(blocks):
    ax = fig.add_subplot(gsC[0, bi])
    ylabels, ys, y = [], [], 0
    for m in mods_b:
        d = go_all[go_all.module == m].sort_values("Adjusted P-value").head(5)
        for k, (_, r) in enumerate(d.iterrows()):
            ng = len([x for x in str(r.Genes).split(";") if x and x != "nan"])
            ylog = -np.log10(max(r["Adjusted P-value"], 1e-300))
            ax.scatter(1 + k, y, s=8 + (ng - smin) / max(smax - smin, 1) * 38,
                       c=[ylog], cmap="Reds", vmin=0, vmax=2.5,
                       edgecolor="#7A7A7A", linewidth=0.25, zorder=3)
            ylabels.append(f"M{m} · {short(r.Term_clean)}" if k == 0 else short(r.Term_clean))
            ys.append(y); y += 1
        ax.axhline(y - 0.5, color="#B0B0B0", lw=0.5)
    ax.set_xlim(0.5, 5.5); ax.set_ylim(y - 0.5, -0.5)
    ax.set_yticks(ys); ax.set_yticklabels(ylabels, fontsize=5.4)
    for t, lab in zip(ax.get_yticklabels(), ylabels):
        if lab.startswith("M2"):
            t.set_fontweight("bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels([f"#{i}" for i in range(1, 6)], fontsize=5.4)
    ax.set_xlabel("GO term rank within module", fontsize=5.4, labelpad=1)
    ax.tick_params(axis="both", length=1.5, pad=1.5)
    ax.set_title(("Modules 1–4" if bi == 0 else "Modules 5–8"), fontsize=5.6, pad=3)
    if bi == 0:
        ax.text(0.5, 1.10, "only M2 reaches FDR < 0.05\n(Enrichr background: all detected genes, n = 36,601)",
                transform=ax.transAxes,
                fontsize=5.4, ha="center", va="bottom", color=ACCENT_RED,
                linespacing=1.25)
        panel_letter(ax, "C", x=-0.36, y=1.03)
cax = fig.add_subplot(gsC[0, 2])
sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, 2.5))
cb = fig.colorbar(sm, cax=cax)
cb.set_label("−log$_{10}$ FDR", fontsize=5.4, labelpad=2)
cb.ax.tick_params(labelsize=5.4, length=1.5)
cb.outline.set_linewidth(0.3)
handles = [plt.scatter([], [], s=8 + (v - smin) / max(smax - smin, 1) * 38,
                       facecolor="#F4A582", edgecolor="#7A7A7A", lw=0.3)
           for v in [smin, int((smin + smax) / 2), smax]]
cax.legend(handles, [str(smin), str(int((smin + smax) / 2)), str(smax)],
           title="genes", fontsize=5.4, title_fontsize=5.4, loc="lower center",
           bbox_to_anchor=(0.5, -0.30), frameon=False, handletextpad=0.5,
           labelspacing=0.6, borderpad=0.1)

# ---------------- D: IFN-I/II score by type and group ----------------
gsD = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], width_ratios=[0.60, 0.40],
                              wspace=0.30)
axD = fig.add_subplot(gsD[0])
p = ifn.pivot(index="cell_type", columns="group", values="ifn_score").reindex(CT)
delta = (p["RIF"] - p["Fertile"]).sort_values()
p = p.loc[delta.index]
y = np.arange(len(p))
axD.hlines(y, p.Fertile, p.RIF, color="#C8C8C8", lw=0.9, zorder=1)
axD.scatter(p.Fertile, y, s=24, color=ACCENT_BLUE, edgecolor="white", lw=0.4,
            zorder=3, label="Fertile")
axD.scatter(p.RIF, y, s=24, color=ACCENT_RED, edgecolor="white", lw=0.4,
            zorder=3, label="RIF")
axD.axvline(0, color="#888888", lw=0.5, ls=(0, (3, 2)))
axD.set_yticks(y); axD.set_yticklabels(p.index, fontsize=5.4)
axD.set_xlabel("IFN-I/II score (custom 15-gene panel; not Hallmark)", fontsize=5.6, labelpad=2)
axD.tick_params(axis="x", labelsize=5.4, length=2, pad=1)
axD.tick_params(axis="y", length=0, pad=2)
axD.set_xlim(-0.155, 0.115)
axD.set_title("RIF > Fertile in 9/11 types (B/Plasma, Lymphatic EC exceptions)",
              fontsize=5.6, pad=3)
axD.legend(fontsize=5.4, ncol=2, loc="upper right", handletextpad=0.3,
           columnspacing=0.8, borderpad=0.1)
panel_letter(axD, "D", x=-0.19, y=1.06)

axI = fig.add_subplot(gsD[1])
dm = m2c.set_index("cell_type").reindex(p.index)
rho, pv = spearmanr(dm.M2_mean_z, p.RIF - p.Fertile)
axI.scatter(dm.M2_mean_z, p.RIF - p.Fertile, s=18, color="#555555",
            edgecolor="white", lw=0.3, zorder=3)
for ct in ["B/Plasma", "Lymphatic EC"]:
    if ct in dm.index:
        axI.annotate(ct, (dm.loc[ct, "M2_mean_z"], p.loc[ct, "RIF"] - p.loc[ct, "Fertile"]),
                     xytext=(3, 2), textcoords="offset points", fontsize=5.4,
                     color=ACCENT_RED)
axI.axhline(0, color="#AAAAAA", lw=0.4, ls=(0, (2, 2)))
axI.set_xlabel("M2 mean z", fontsize=5.4, labelpad=1)
axI.set_ylabel("Δ IFN score", fontsize=5.4, labelpad=1)
axI.tick_params(labelsize=5.4, length=2, pad=1)
axI.set_title(f"ρ = {rho:.2f}, p = {pv:.2f}", fontsize=5.4, pad=3)

save_cns_figure(fig, f"{FIG}/FigS4_M2_MHC_GO_IFN")
print("FIG_S4_DONE", f"rho={rho:.3f} p={pv:.3f}")
