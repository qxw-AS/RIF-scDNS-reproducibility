#!/usr/bin/env python3
"""Fig.3 — reproducibility and method comparison (Fig.3A–C).

# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) split-half divergence scatter        -> assets/figures/MarginalDensity (scatter system) -> param inherit
# (b) module vs DEG GO overlap heatmap     -> assets/figures/heatmap                             -> param inherit
# (c) top-N Jaccard curve + null band      -> assets/figures/LineTrend                           -> param inherit
# RULE: all three panels are "param inherit" (Class A/B/C values copied from the
#       named assets); no pre-rendered production PNG is loaded.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

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
ORANGE      = "#F1A340"

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
DATA = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "fig3"))
FIG2 = os.environ.get("RIF_DATA2", os.path.join(_ROOT, "data", "fig2"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
os.makedirs(FIG, exist_ok=True)


def panel_letter(ax, letter, x=-0.10, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def short_go(term):
    m = {
        "Nucleotide Metabolic Process": "Nucleotide metabolism",
        "Antigen Processing And Presentation Of Endogenous Peptide Antigen Via MHC Class I Via ER Pathway":
            "MHC-I presentation (ER)",
        "Antigen Processing And Presentation Of Endogenous Peptide Antigen Via MHC Class I Via ER Pathway, TAP-independent":
            "MHC-I presentation (TAP-ind.)",
        "Zymogen Activation": "Zymogen activation",
        "Phosphatidylinositol Phosphate Biosynthetic Process": "PI phosphate biosynthesis",
        "Insulin Secretion": "Insulin secretion",
        "Basement Membrane Organization": "Basement membrane org.",
        "Regulation Of Autophagy Of Mitochondrion": "Mitochondrial autophagy",
        "Regulation Of Double-Strand Break Repair Via Nonhomologous End Joining":
            "DSB repair via NHEJ (reg.)",
        "Establishment Of Protein Localization To Organelle": "Protein localization (organelle)",
        "Cytoplasmic Translation": "Cytoplasmic translation",
        "Peptide Biosynthetic Process": "Peptide biosynthesis",
        "Macromolecule Biosynthetic Process": "Macromolecule biosynthesis",
        "Negative Regulation Of Gene Expression": "Negative regulation of expr.",
        "Gene Expression": "Gene expression",
        "Regulation Of Transcription By RNA Polymerase II": "RNA Pol II transcription (reg.)",
        "Regulation Of Gene Expression": "Gene expression (reg.)",
        "Regulation Of DNA-templated Transcription": "DNA-templated transcription (reg.)",
        "Positive Regulation Of DNA-templated Transcription": "DNA-templated transcription (pos.)",
        "Negative Regulation Of DNA-templated Transcription": "DNA-templated transcription (neg.)",
        "Positive Regulation Of Transcription By RNA Polymerase II": "RNA Pol II transcription (pos.)",
        "Positive Regulation Of Leukocyte Mediated Cytotoxicity": "Leukocyte cytotoxicity (pos.)",
        "Antigen Processing And Presentation Of Endogenous Peptide Antigen": "MHC-I antigen presentation",
        "Regulation Of Apoptotic Process": "Regulation of apoptosis",
        "Translation": "Translation",
        "Regulation Of Cell Population Proliferation": "Cell population proliferation",
    }
    if term in m:
        return m[term]
    t = term.title()
    return t if len(t) <= 27 else t[:24] + "…"


MOD_COLORS = {
    1: "#4E79A7", 2: "#F28E2B", 3: "#59A14F", 4: "#B07AA1",
    5: "#76B7B2", 6: "#EDC948", 7: "#FF9DA7", 8: "#9C755F",
}
MODS = sorted(MOD_COLORS)

# ============================ figure ============================
fig = plt.figure(figsize=(7.2, 4.5))
if os.environ.get("RIF_CAPTION", "0") == "1":
    fig.text(0.5, 0.994, "Figure 3. Reproducibility and independence of network rewiring",
             ha="center", va="top", fontsize=8.5)
gs = fig.add_gridspec(2, 2, width_ratios=[1.06, 1.0], height_ratios=[1.42, 1.0],
                      wspace=0.42, hspace=0.68, left=0.075, right=0.975,
                      top=0.865, bottom=0.105)

# ---------------------------- Panel A ----------------------------
gsA = gs[0, 0].subgridspec(1, 2, wspace=0.30)


def split_scatter(ax, tag, title):
    z = pd.read_csv(f"{DATA}/split_half{tag}_gene_z.csv")
    m = z.testable_common.to_numpy()
    zA, zB = z.z_A.to_numpy(), z.z_B.to_numpy()
    nonsig = z.deg_nonsig.to_numpy() & m
    topA = np.argsort(-np.abs(np.where(m, zA, np.nan)))[:50]
    topB = np.argsort(-np.abs(np.where(m, zB, np.nan)))[:50]
    inA, inB = np.zeros(len(z), bool), np.zeros(len(z), bool)
    inA[topA], inB[topB] = True, True
    both = nonsig & inA & inB
    one = nonsig & (inA ^ inB)
    other = m & ~(both | one)
    ax.scatter(zA[other], zB[other], s=1.1, c="#D9D9D9", alpha=0.55, lw=0,
               zorder=1)
    ax.scatter(zA[one], zB[one], s=4.0, c=ORANGE, alpha=0.95, lw=0, zorder=2)
    ax.scatter(zA[both], zB[both], s=6.5, c=ACCENT_RED, alpha=1.0, lw=0, zorder=3)
    lim = 4.3
    ax.plot([-lim, lim], [-lim, lim], color="#888888", lw=0.45, ls="--", zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=6.2, pad=3, linespacing=1.25)
    ax.set_xlabel("Divergence z, half A", fontsize=6.2, labelpad=1.5)
    ax.tick_params(labelsize=5.6, length=2, pad=1.5)


j = pd.read_csv(f"{DATA}/split_half_jaccard.csv")
jd = pd.read_csv(f"{DATA}/split_half_donor_jaccard.csv")
r_cell = pd.read_csv(f"{DATA}/split_half_metrics.csv").gene_level_r.iloc[0]
r_don = pd.read_csv(f"{DATA}/split_half_donor_metrics.csv").gene_level_r.iloc[0]
j50c = j[(j.N == 50) & (j.kind == "network_only")].iloc[0]
j50d = jd[(jd.N == 50) & (jd.kind == "network_only")].iloc[0]

axA1 = fig.add_subplot(gsA[0])
split_scatter(axA1, "", f"Cell-level split (same donors)\nr = {r_cell:.3f} · J$_{{50}}$ = {j50c.observed:.3f}")
axA1.set_ylabel("Divergence z, half B", fontsize=6.2, labelpad=1.5)
axA2 = fig.add_subplot(gsA[1])
split_scatter(axA2, "_donor", f"Donor-level split (disjoint)\nr = {r_don:.3f} · J$_{{50}}$ = {j50d.observed:.3f}")
axA2.tick_params(labelleft=False)

handles = [Line2D([0], [0], marker="o", linestyle="", markersize=3.6,
                  markerfacecolor=ACCENT_RED, markeredgecolor="none",
                  label="Network-only in both halves"),
           Line2D([0], [0], marker="o", linestyle="", markersize=3.2,
                  markerfacecolor=ORANGE, markeredgecolor="none",
                  label="Network-only in one half"),
           Line2D([0], [0], marker="o", linestyle="", markersize=2.8,
                  markerfacecolor="#D9D9D9", markeredgecolor="none",
                  label="Other testable genes")]
axA2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.38, -0.24),
            ncol=3, fontsize=5.4, handletextpad=0.25, columnspacing=0.9,
            borderpad=0.1)
panel_letter(axA1, "A", x=-0.26, y=1.10)

# ---------------------------- Panel B ----------------------------
axB = fig.add_subplot(gs[0, 1])
# Module GO on the SAME 36,601-gene background as the DEG-union column
# (go_enrichment_full_allgene.csv; generated by scripts/stage7e_go_allgene.py).
mods = pd.read_csv(f"{FIG2}/go_enrichment_full_allgene.csv")
mods["Term_clean"] = mods.Term.str.replace(r"\s*\(GO:\d+\)$", "", regex=True)
du = pd.read_csv(f"{DATA}/deg_go_union.csv")
du["Term_clean"] = du.Term.str.replace(r"\s*\(GO:\d+\)$", "", regex=True)
# Row set = representative terms per module (selected once from the module
# analysis); values are all recomputed on the 36,601 background above.
M = pd.read_csv(f"{FIG2}/go_module_matrix.csv", index_col=0)
rows = [t for t in M.index if pd.notna(M.loc[t]).any()]
deg_top = du.sort_values("Adjusted P-value").head(7)
for t in deg_top.Term_clean:
    if t not in rows:
        rows.append(t)
mat = pd.DataFrame(0.0, index=rows, columns=[f"M{m}" for m in MODS] + ["DEG"])
for _, r in mods[mods.Term_clean.isin(rows)].iterrows():
    v = -np.log10(max(r["Adjusted P-value"], 1e-300))
    mat.loc[r.Term_clean, f"M{r['module']}"] = max(mat.loc[r.Term_clean, f"M{r['module']}"], v)
for _, r in du[du.Term_clean.isin(rows)].iterrows():
    mat.loc[r.Term_clean, "DEG"] = -np.log10(max(r["Adjusted P-value"], 1e-300))
# order: shared (sig in module and DEG) -> module terms -> DEG-only terms
mod_terms = set(rows) & set(M.index)
deg_sig = set(du.loc[du["Adjusted P-value"] < 0.05, "Term_clean"])
shared = [t for t in rows if t in mod_terms and t in deg_sig]
mt = [t for t in rows if t in mod_terms and t not in deg_sig]
dt = [t for t in rows if t not in mod_terms]
order = shared + mt + dt
mat = mat.loc[order]
vmax = 6.0
cmapB = LinearSegmentedColormap.from_list("go", ["#FFFFFF", "#DEEBF7", "#6BAED6", "#08306B"])
imB = axB.imshow(np.clip(mat.to_numpy(), 0, vmax), aspect="auto", cmap=cmapB,
                 vmin=0, vmax=vmax, interpolation="nearest")
axB.set_yticks(range(len(mat)))
axB.set_yticklabels([short_go(t) for t in mat.index], fontsize=5.4)
axB.set_xticks(range(mat.shape[1]))
axB.set_xticklabels([f"M{m}" for m in MODS] + ["DEG"], fontsize=5.4)
for s in axB.spines.values():
    s.set_visible(False)
for tick, m in zip(axB.get_xticklabels()[:8], MODS):
    tick.set_color(MOD_COLORS[m])
    tick.set_fontweight("bold")
axB.get_xticklabels()[-1].set_fontweight("bold")
axB.tick_params(axis="y", length=0, pad=2)
axB.tick_params(axis="x", length=1.5, pad=2)
for x in np.arange(-0.5, mat.shape[1], 1):
    axB.axvline(x, color="white", lw=0.6)
for y in np.arange(-0.5, mat.shape[0], 1):
    axB.axhline(y, color="white", lw=0.4)
axB.axvline(len(MODS) - 0.5, color="#444444", lw=0.7)
for i in range(mat.shape[0]):
    for jx in range(mat.shape[1]):
        v = mat.iloc[i, jx]
        if v > -np.log10(0.05):
            axB.text(jx, i, "*", ha="center", va="center", fontsize=6.2,
                     color="white" if v > 3.2 else BLACK, fontweight="bold")
caxB = axB.inset_axes([0.0, -0.125, 0.38, 0.026])
cbB = fig.colorbar(imB, cax=caxB, orientation="horizontal")
cbB.set_label("−log10 FDR", fontsize=5.4, labelpad=1)
cbB.ax.tick_params(labelsize=5.4, length=1.5, pad=1)
cbB.outline.set_linewidth(0.3)
axB.text(0.42, -0.105, "All-gene background (36,601): 6/6 M2 terms",
         transform=axB.transAxes, fontsize=5.4, color=BLACK, va="top", ha="left")
axB.text(0.42, -0.163, "Matched (1,353 genes): 0/6; MHC-I FDR = 0.356",
         transform=axB.transAxes, fontsize=5.4, color="#555555", va="top", ha="left")
panel_letter(axB, "B", x=-0.20, y=1.06)

# ---------------------------- Panel C (broken y axis) ----------------------------
gsC = gs[1, :].subgridspec(2, 1, height_ratios=[0.17, 0.83], hspace=0.10)
axCt = fig.add_subplot(gsC[0])
axC = fig.add_subplot(gsC[1], sharex=axCt)
cur = pd.read_csv(f"{DATA}/jaccard_curves.csv")
NS = [50, 100, 200, 400, 800]
rng = np.random.default_rng(7)
for N in NS:
    v = cur[(cur.N == N)].jaccard.dropna().to_numpy()
    x = np.full(len(v), N) * np.exp(rng.uniform(-0.035, 0.035, len(v)))
    axC.scatter(x, v, s=9, c="#2166AC", alpha=0.55, lw=0, zorder=3)
obs = cur.groupby("N").jaccard.agg(["median",
                                    lambda s: s.quantile(.25),
                                    lambda s: s.quantile(.75)]).dropna()
obs.columns = ["med", "q1", "q3"]
obs = obs.loc[NS]
axC.fill_between(obs.index, obs.q1, obs.q3, color="#2166AC", alpha=0.13, lw=0)
axC.plot(obs.index, obs.med, color=BLACK, lw=1.2, marker="o", ms=3.2, zorder=4)
# No null band is drawn: the divergence list is restricted to the per-type
# testable pool (642-1,258 genes) while the DEG list spans 36,601 genes, so a
# hypergeometric null computed on a single universe is not exchangeable with
# the observed overlap.  J50 is reported descriptively (see legend, Methods).
axCt.axhline(0.5, color=ACCENT_RED, lw=0.9, ls="--")
axCt.text(880, 0.5035, "pre-specified Jaccard = 0.5", fontsize=5.6, color=ACCENT_RED,
          va="bottom", ha="right")
axCt.set_ylim(0.48, 0.52)
axCt.set_yticks([0.5])
axCt.set_yticklabels(["0.5"], fontsize=5.6)
axCt.tick_params(axis="y", length=2, pad=1.5)
axCt.tick_params(axis="x", bottom=False, labelbottom=False)
axCt.spines["bottom"].set_visible(False)

axC.set_xscale("log")
axC.set_xticks(NS)
axC.set_xticklabels([str(n) for n in NS])
axC.minorticks_off()
axC.set_xlim(44, 900)
axC.set_ylim(0, 0.18)
axC.set_yticks([0, 0.05, 0.10, 0.15])
axC.set_yticklabels(["0", "0.05", "0.10", "0.15"])
axC.set_xlabel("Top-N genes per method", fontsize=6.8, labelpad=2)
axC.set_ylabel("Jaccard (divergence ∩ DEG)", fontsize=6.8, labelpad=2)
axC.tick_params(labelsize=6)
med50 = obs.loc[50, "med"]
axC.text(0.015, 0.965, f"median J$_{{50}}$ = {med50:.2f}; all 11 types ≤ "
         f"{cur[cur.N==50].jaccard.max():.3f}\n"
         "network-only at N=50: 16–50 per type\n"
         "n = 11 cell types (9 at N=800); J$_{50}$ descriptive, no chance benchmark",
         transform=axC.transAxes, fontsize=5.4, color=BLACK, va="top", ha="left",
         linespacing=1.35)
panel_letter(axCt, "C", x=-0.055, y=1.12)

# axis-break marks
d = 0.014
kw = dict(transform=axCt.transAxes, color="#333333", clip_on=False, lw=0.6)
axCt.plot((-d, +d), (-d, +d), **kw)
axCt.plot((1 - d, 1 + d), (-d, +d), **kw)
kw.update(transform=axC.transAxes)
axC.plot((-d, +d), (1 - d, 1 + d), **kw)
axC.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

save_cns_figure(fig, f"{FIG}/Fig3_reproducibility_method_comparison")
print("FIG3_DONE", flush=True)
