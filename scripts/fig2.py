#!/usr/bin/env python3
"""Fig.2 — condition-specific gene-coordination rewiring modules (v2 layout).

# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) gene dendrogram + module strip      -> assets/figures/Manifold (scatter system)     -> param inherit
# (b) M2 zoom (ranked z + type z heatmap) -> assets/figures/MarkerGeneDotPlot              -> visual adapt
# (c) ranked divergence score + marginal  -> assets/figures/Manifold (scatter system)     -> param inherit
# (d) combined z vs pseudobulk log2FC     -> assets/figures/Scatter                        -> param inherit
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, dendrogram

# ---------- Academic Figure Skill Typography Baseline — COPY VERBATIM ----------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# ---------- Academic Figure Skill Color Palette — COPY VERBATIM ----------
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# ---------- Academic Figure Skill Export Baseline — COPY VERBATIM ----------
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "fig2"))
FIG1 = os.environ.get("RIF_DATA1", os.path.join(_ROOT, "data", "fig1"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
os.makedirs(FIG, exist_ok=True)

mods = pd.read_csv(f"{DATA}/gene_modules.csv")
prof = pd.read_csv(f"{DATA}/module_profiles.csv", index_col=0)
go = pd.read_csv(f"{DATA}/go_enrichment_full.csv")
cz = pd.read_csv(f"{DATA}/combined_z.csv")
sel = pd.read_csv(f"{DATA}/panelD_selected.csv")

MOD_COLORS = {
    1: "#4E79A7", 2: "#F28E2B", 3: "#59A14F", 4: "#B07AA1",
    5: "#76B7B2", 6: "#EDC948", 7: "#FF9DA7", 8: "#9C755F",
}
MODS = sorted(MOD_COLORS)
CT_ORDER = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
            "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]
POS, NEG = "#B2182B", "#2166AC"


def panel_letter(ax, letter, x=-0.10, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def short_go(term):
    m = {
        "Nucleotide Metabolic Process": "Nucleotide metabolism",
        "Antigen Processing And Presentation Of Endogenous Peptide Antigen Via MHC Class I Via ER Pathway":
            "MHC-I presentation (ER)",
        "Zymogen Activation": "Zymogen activation",
        "Phosphatidylinositol Phosphate Biosynthetic Process": "PI phosphate biosynthesis",
        "Insulin Secretion": "Insulin secretion",
        "Basement Membrane Organization": "Basement membrane org.",
        "Regulation Of Autophagy Of Mitochondrion": "Mitochondrial autophagy",
    }
    if term in m:
        return m[term]
    t = term.title()
    return t if len(t) <= 28 else t[:25] + "…"


# ============================ figure ============================
fig = plt.figure(figsize=(7.2, 7.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.06, 0.94],
                      wspace=0.50, hspace=0.38, left=0.065, right=0.945,
                      top=0.935, bottom=0.070)

# ---------------------------- Panel A ----------------------------
gsA = gs[0, 0].subgridspec(3, 1, height_ratios=[0.60, 0.055, 0.345], hspace=0.10)
axA1 = fig.add_subplot(gsA[0])
axAs = fig.add_subplot(gsA[1])
axA2 = fig.add_subplot(gsA[2])

genes = mods.gene.tolist()
Zg = pd.read_csv(f"{DATA}/gene_z_by_type.csv", index_col=0).loc[genes].fillna(0.0).to_numpy()
Ln = linkage(Zg, method="ward", optimal_ordering=True)
dn = dendrogram(Ln, no_plot=True)
leaves = np.asarray(dn["leaves"])
leaf_mod = mods.module.to_numpy()[leaves]

segs, cols = [], []
for xs, ys in zip(dn["icoord"], dn["dcoord"]):
    lo = int(round((min(xs[0], xs[2]) - 5) / 10))
    hi = int(round((max(xs[0], xs[2]) - 5) / 10))
    u = set(leaf_mod[lo:hi + 1])
    c = MOD_COLORS[next(iter(u))] if len(u) == 1 else "#CFCFCF"
    for k in range(3):
        segs.append([(xs[k], ys[k]), (xs[k + 1], ys[k + 1])])
        cols.append(c)
axA1.add_collection(LineCollection(segs, colors=cols, linewidths=0.45, alpha=0.95))
axA1.set_xlim(0, 10 * len(genes))
axA1.set_ylim(0, max(dn["dcoord"][-1]) * 1.03)
axA1.set_xticks([])
axA1.set_ylabel("Ward distance", fontsize=6, labelpad=1)
axA1.tick_params(axis="y", labelsize=5.5, length=2, pad=1)
axA1.spines["bottom"].set_visible(False)

mod_idx = np.array([[list(MODS).index(v) for v in leaf_mod]])
axAs.imshow(mod_idx, aspect="auto", interpolation="nearest",
            cmap=ListedColormap([MOD_COLORS[m] for m in MODS]))
axAs.set_xticks([]); axAs.set_yticks([])
for sp in axAs.spines.values():
    sp.set_visible(False)

handles = [Line2D([0], [0], marker="s", linestyle="", markersize=3.4,
                  markerfacecolor=MOD_COLORS[m], markeredgecolor="none",
                  label=f"M{m}") for m in MODS]
axA1.legend(handles=handles, ncol=8, loc="lower center", bbox_to_anchor=(0.5, 1.005),
            fontsize=5.4, handletextpad=0.15, columnspacing=0.55, borderpad=0.1)

P = prof.loc[MODS, CT_ORDER].to_numpy()
imA = axA2.imshow(P, aspect="auto", cmap=LinearSegmentedColormap.from_list("z", DIVERGING),
                  norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5), interpolation="nearest")
axA2.set_yticks(range(len(MODS)))
axA2.set_yticklabels([f"M{m}*" if m == 2 else f"M{m}" for m in MODS], fontsize=5.4)
for tick, m in zip(axA2.get_yticklabels(), MODS):
    tick.set_color(MOD_COLORS[m])
axA2.set_xticks(range(len(CT_ORDER)))
axA2.set_xticklabels(CT_ORDER, rotation=90, ha="center", va="top", fontsize=5.4)
axA2.tick_params(axis="x", length=1.5, pad=1.5)
axA2.tick_params(axis="y", length=0, pad=2)
for x in np.arange(-0.5, len(CT_ORDER), 1):
    axA2.axvline(x, color="white", lw=0.5)
for y in np.arange(-0.5, len(MODS), 1):
    axA2.axhline(y, color="white", lw=0.5)
caxA = axA2.inset_axes([1.03, 0.0, 0.028, 1.0])
cbA = fig.colorbar(imA, cax=caxA)
cbA.set_label("z", fontsize=5.4, labelpad=0.5)
cbA.ax.tick_params(labelsize=5.4, length=1.5, pad=1)
cbA.outline.set_linewidth(0.3)
axA2.set_xlabel("Cell type", fontsize=5.5, labelpad=1)
axA2.text(0.0, -0.42, "*M2 = MHC-I-enriched", transform=axA2.transAxes,
          fontsize=5.4, color=MOD_COLORS[2], va="top", ha="left")
panel_letter(axA1, "A", x=-0.14, y=1.06)

# ---------------------------- Panel B: M2 zoom ----------------------------
gsB = gs[0, 1].subgridspec(1, 2, width_ratios=[0.60, 0.40], wspace=0.06)
axB = fig.add_subplot(gsB[0])
axBh = fig.add_subplot(gsB[1], sharey=axB)

m2 = mods[mods.module == 2].sort_values("combined_z", ascending=True).reset_index(drop=True)
Zm = pd.read_csv(f"{DATA}/gene_z_by_type.csv", index_col=0).reindex(m2.gene)[CT_ORDER]
hla = {"HLA-A", "HLA-B", "HLA-C"}
bcols = [ACCENT_RED if g in hla else "#9E9E9E" for g in m2.gene]
yb = np.arange(len(m2))
axB.barh(yb, m2.combined_z, color=bcols, edgecolor="white", lw=0.35, height=0.72)
axB.axvline(1.5, color="#666666", lw=0.5, ls=(0, (2, 2)))
axB.set_yticks(yb)
axB.set_yticklabels([""] * len(m2), fontsize=5.4, fontstyle="italic")
axB.set_ylim(-0.6, len(m2) - 0.4)
axB.set_xlim(0, 6.1)
axB.set_xlabel("Combined rewiring z (cutoff 1.5)", fontsize=5.6, labelpad=1.5)
axB.tick_params(axis="x", labelsize=5.4, length=2, pad=1)
axB.tick_params(axis="y", length=0, pad=1.5)
axB.set_title("M2 (MHC-I) genes, ranked by combined z", fontsize=5.9, pad=3)
axB.text(0.0, -0.135, "GO BP: MHC-I presentation (ER), FDR = 0.005 (testable-gene background, n = 1,353)\n"
                      "red = HLA-A/B/C; 46 M2 genes; matched-background DEG-union: 0/6 (Fig. 3B)",
         transform=axB.transAxes, fontsize=5.4, color="#555555", va="top")
panel_letter(axB, "B", x=-0.30, y=1.05)

vmaxB = 2.5
cmapB = LinearSegmentedColormap.from_list("z", DIVERGING)
imB = axBh.imshow(Zm.to_numpy(), aspect="auto", cmap=cmapB,
                  norm=TwoSlopeNorm(vmin=-vmaxB, vcenter=0, vmax=vmaxB),
                  interpolation="nearest")
axBh.tick_params(axis="y", left=False, labelleft=False, length=0)
axBh.set_xticks(range(len(CT_ORDER)))
axBh.set_xticklabels(["Epith.", "Gland.", "Cil.", "Strom.", "Endo.", "Lymph.",
                      "T", "NK", "B/Pl.", "Myel.", "Mast"], rotation=90,
                     ha="center", va="top", fontsize=5.4)
axBh.tick_params(axis="x", length=1.5, pad=1)
for x in np.arange(-0.5, len(CT_ORDER), 1):
    axBh.axvline(x, color="white", lw=0.45)
caxB = axBh.inset_axes([1.04, 0.0, 0.05, 1.0])
cbB = fig.colorbar(imB, cax=caxB)
cbB.set_label("type z", fontsize=5.4, labelpad=0.5)
cbB.ax.tick_params(labelsize=5.4, length=1.2, pad=0.5)
cbB.outline.set_linewidth(0.3)

# ---------------------------- Panel C ----------------------------
gsC = gs[1, 0].subgridspec(1, 2, width_ratios=[0.84, 0.16], wspace=0.04)
axC = fig.add_subplot(gsC[0])
axCm = fig.add_subplot(gsC[1], sharey=axC)

cz2 = cz.dropna(subset=["combined_z"]).sort_values("combined_z", ascending=False).reset_index(drop=True)
y = cz2.combined_z.to_numpy()
x = np.arange(len(cz2))
# DEG status in the cell type where this gene has its strongest rewiring z
deg = pd.read_csv(f"{FIG1}/panelD_deg_q.csv")
qmap = deg.set_index(["names", "cell_type"]).pvals_adj
Zt = pd.read_csv(f"{DATA}/gene_z_by_type.csv", index_col=0)
ct_cols = Zt.columns.tolist()
qstat = []
for g in cz2.gene:
    if g not in Zt.index:
        qstat.append(np.nan); continue
    v = Zt.loc[g, ct_cols].astype(float)
    ct = v.abs().idxmax()
    qstat.append(qmap.get((g, ct), np.nan))
cz2["qmax"] = qstat
status = np.where(cz2.qmax.isna(), 0, np.where(cz2.qmax < 0.05, 1, 2))
scol = {0: "#E0E0E0", 1: "#BDBDBD", 2: "#6A3D9A"}   # untested / DEG / network-only
axC.scatter(x, y, s=2.4, c=[scol[s] for s in status], alpha=0.85,
            linewidths=0, rasterized=True)
axC.axhline(0, color="#666666", lw=0.45, ls=(0, (3, 2)))
axC.set_xlim(-20, len(cz2) + 20)
axC.set_xlabel("Gene rank (combined z)", fontsize=6, labelpad=1.5)
axC.set_ylabel("Combined rewiring z (RIF − fertile)", fontsize=6, labelpad=2)
axC.tick_params(labelsize=5.4, length=2, pad=1.5)
axC.set_ylim(-7.2, 7.2)

pos_lab = cz2.head(3)
neg_lab = cz2.tail(3).iloc[::-1]
for k, r in enumerate(pos_lab.itertuples()):
    axC.annotate(r.gene, xy=(k, r.combined_z),
                 xytext=(k + 105 + 105 * k, r.combined_z - 0.45 - 0.55 * k),
                 fontsize=5.4, fontstyle="italic", color=POS, va="center", ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.3, color=POS, shrinkA=0, shrinkB=1))
for k, (idx, r) in enumerate(neg_lab.iterrows()):
    xx = len(cz2) - 1 - k
    axC.annotate(r.gene, xy=(xx, r.combined_z),
                 xytext=(xx - 105 - 105 * k, r.combined_z + 0.45 + 0.55 * k),
                 fontsize=5.4, fontstyle="italic", color=NEG, va="center", ha="right",
                 arrowprops=dict(arrowstyle="-", lw=0.3, color=NEG, shrinkA=0, shrinkB=1))
axC.text(0.985, 0.965, "gained in RIF", transform=axC.transAxes, ha="right", va="top",
         fontsize=5.4, color=POS, fontweight="bold")
axC.text(0.015, 0.035, "lost in RIF", transform=axC.transAxes, ha="left", va="bottom",
         fontsize=5.4, color=NEG, fontweight="bold")
axC.text(0.015, 0.965, f"n = {len(cz2):,} testable genes", transform=axC.transAxes,
         fontsize=5.4, color="#555555", va="top")

n_nat = int((cz2.qmax >= 0.05).sum())
handlesC = [Line2D([0], [0], marker="o", ls="none", ms=3.2, mfc=scol[2],
                   mec="none", label=f"network-only in max-|z| type (q>0.05; {n_nat})"),
            Line2D([0], [0], marker="o", ls="none", ms=3.2, mfc=scol[1],
                   mec="none", label="DEG q < 0.05 in max-|z| type")]
axC.legend(handles=handlesC, loc="upper center", bbox_to_anchor=(0.55, 0.90),
           ncol=1, fontsize=5.4, handletextpad=0.2, labelspacing=0.25,
           borderpad=0.15, frameon=True, framealpha=0.9, edgecolor="none")
panel_letter(axC, "C", x=-0.14, y=1.04)

bins = np.linspace(-7.2, 7.2, 56)
cnt, edges = np.histogram(y, bins=bins)
cent = (edges[:-1] + edges[1:]) / 2
axCm.barh(cent, cnt, height=np.diff(edges) * 0.92,
          color=np.where(cent < 0, NEG, POS), alpha=0.85, linewidth=0)
axCm.set_xlabel("Genes", fontsize=5.4, labelpad=1)
axCm.tick_params(axis="x", labelsize=5.4, length=1.5, pad=1)
axCm.tick_params(axis="y", left=False, labelleft=False, length=0)
axCm.spines["left"].set_visible(False)
axCm.set_ylim(-7.2, 7.2)

# ---------------------------- Panel D ----------------------------
axD = fig.add_subplot(gs[1, 1])
top = (mods.reindex(mods.combined_z.abs().sort_values(ascending=False).index)
       .groupby("module").head(3).sort_values(["module", "combined_z"],
                                              ascending=[True, False]))
dg = sel[sel.gene.isin(top.gene)].copy()
dg["module"] = dg.gene.map(mods.set_index("gene").module)
dg = dg.dropna(subset=["log2FC", "combined_z"])
dg["mlogp"] = -np.log10(np.clip(dg.pval, 1e-300, 1))
axD.axhline(0, color="#666666", lw=0.5, ls=(0, (3, 2)))
scD = axD.scatter(dg.combined_z, dg.log2FC,
                  s=4.0 + 26.0 * np.sqrt(np.clip(dg.mlogp / 3.0, 0, 1)),
                  c=[MOD_COLORS[m] for m in dg.module],
                  edgecolors=np.where(dg.padj < 0.05, BLACK, "#FFFFFF"),
                  linewidths=np.where(dg.padj < 0.05, 0.8, 0.25), alpha=0.9)
axD.set_xlim(1.2, 5.9)
axD.set_ylim(-1.05, 1.05)
axD.set_xticks([2, 3, 4, 5])
axD.set_yticks([-1, 0, 1])
axD.set_xlabel("Combined rewiring z (network)", fontsize=6, labelpad=1.5)
axD.set_ylabel("Pseudobulk log$_2$FC (RIF/Fertile)", fontsize=6, labelpad=1.5)
axD.tick_params(labelsize=5.4, length=2, pad=1.5)
n_sig = int((dg.padj < 0.05).sum())
n_pair = len(dg)
axD.text(0.03, 0.965, f"{n_sig}/{n_pair} gene–type pairs FDR < 0.05\n"
                      f"{dg.gene.nunique()} module-top genes × 11 types",
         transform=axD.transAxes, fontsize=5.4, color=BLACK, va="top", linespacing=1.35,
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
axD.text(0.97, 0.035, "dot size = −log10 P", transform=axD.transAxes,
         fontsize=5.4, color="#666666", ha="right", va="bottom")
handlesD = [Line2D([0], [0], marker="o", ls="none", ms=3.2, mfc=MOD_COLORS[m],
                   mec="none", label=f"M{m}") for m in MODS]
axD.legend(handles=handlesD, loc="upper center", bbox_to_anchor=(0.5, -0.20),
           ncol=8, fontsize=5.4, handletextpad=0.1, columnspacing=0.45,
           borderpad=0.1, labelspacing=0.2, frameon=False)
axD.set_title("Rewiring strength vs\nexpression change", fontsize=5.9, pad=3,
              loc="left")
panel_letter(axD, "D", x=-0.20, y=1.12)

save_cns_figure(fig, f"{FIG}/Fig2_network_modules")
print("saved", f"{FIG}/Fig2_network_modules.pdf/.png")
