#!/usr/bin/env python3
"""Fig.1 — LH+7 endometrial cell atlas (GSE250130, 10 RIF vs 6 fertile).

# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) UMAP coloured by cell type            -> assets/figures/Manifold (scatter system)      -> param inherit
# (b) UMAP coloured by group                -> assets/figures/Manifold (scatter system)      -> param inherit
# (c) known endometrial marker dot plot     -> assets/figures/MarkerGeneDotPlot              -> visual adapt
# (d) top DEG per cell type dot plot        -> assets/figures/MarkerGeneDotPlot              -> visual adapt
# RULE: "native run" = pre-rendered PNG; "param inherit" = drawing fn copying Class A/B/C values.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

# ---------- Academic Figure Skill Typography Baseline — COPY VERBATIM ----------
import matplotlib as mpl
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

# ------------------------------- data ------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "fig1"))
FIG = os.environ.get("RIF_FIG", os.path.join(_ROOT, "output"))
meta = json.load(open(f"{DATA}/meta.json"))
CELL_TYPES = [c for c in meta["cell_types"]]

CT_COLORS = {
    "Epithelial": "#2166AC", "Glandular": "#4393C3", "Ciliated": "#9970AB",
    "Stromal": "#1B7837", "Smooth muscle": "#5AAE61", "Endothelial": "#F1A340",
    "Lymphatic EC": "#B35806", "T cell": "#B2182B", "NK cell": "#D6604D",
    "B/Plasma": "#762A83", "Myeloid": "#666666", "DC": "#999999", "Mast": "#222222",
}
for i, ct in enumerate(CELL_TYPES):
    CT_COLORS.setdefault(ct, CATEGORICAL_EXTENDED[i % len(CATEGORICAL_EXTENDED)])

GROUP_COLORS = {"Fertile": "#2166AC", "RIF": "#B2182B"}

# Class A parameters inherited from MarkerGeneDotPlot production asset
DOT_CMAP = LinearSegmentedColormap.from_list(
    "nature_bwr", ["#2c5aa0", "#6f8fc8", "#f5f5f5", "#e89b8e", "#c23a2b"])
DOT_EDGE = "#6e6e6e"
DOT_EDGE_LW = 0.45
S_MIN, S_MAX = 3.0, 42.0


def panel_letter(ax, letter, x=-0.10, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def draw_umap(ax, um, color_col, colors, order, s=0.6, alpha=0.85):
    for cat in order:
        d = um[um[color_col] == cat]
        ax.scatter(d["UMAP_1"], d["UMAP_2"], s=s, c=colors[cat], alpha=alpha,
                   linewidths=0, rasterized=True, label=cat)
    ax.set_xticks([]); ax.set_yticklabels([])
    ax.set_xlabel("UMAP-1", fontsize=6, labelpad=2)
    ax.set_ylabel("UMAP-2", fontsize=6, labelpad=2)
    ax.set_aspect("equal", adjustable="datalim")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def dot_panel(ax, df, cell_types, color_vals, cbar_label, cmap, norm,
              size_pct, xlabels, group_of_col, group_order, label_every=1):
    """Dot plot in MarkerGeneDotPlot visual language (adapted to composite layout)."""
    x_map = {g: i for i, g in enumerate(df["gene_column"].unique())}
    y_map = {c: i for i, c in enumerate(cell_types)}
    x = df["gene_column"].map(x_map).to_numpy(float)
    y = df["cell_type"].map(y_map).to_numpy(float)
    sizes = S_MIN + (S_MAX - S_MIN) * np.sqrt(np.clip(size_pct, 0, 100) / 100.0)
    sc = ax.scatter(x, y, s=sizes, c=color_vals, cmap=cmap, norm=norm,
                    edgecolors=DOT_EDGE, linewidths=DOT_EDGE_LW)
    ax.set_xlim(-0.7, len(x_map) - 0.3)
    ax.set_ylim(-0.7, len(cell_types) - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90, ha="center", va="top", fontsize=5.4,
                       fontstyle="italic")
    ax.set_yticks(range(len(cell_types)))
    ax.set_yticklabels(cell_types, fontsize=6)
    ax.tick_params(axis="x", length=2.5, width=0.5, color="#666666", pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#8d8d8d")
        ax.spines[side].set_linewidth(0.6)
    # block separators between cell-type gene blocks
    starts = df.groupby(group_of_col, sort=False)["gene_column"].apply(
        lambda s: list(s).index(s.iloc[0]))
    bounds = []
    for ct in group_order:
        cols = df.loc[df[group_of_col] == ct, "gene_column"].unique()
        if len(cols) == 0:
            continue
        idx = [x_map[c] for c in cols]
        bounds.append((min(idx), max(idx)))
    for _, end in bounds[:-1]:
        ax.axvline(end + 0.5, color="#DDDDDD", lw=0.4, zorder=0)
    return sc


def main():
    um = pd.read_csv(f"{DATA}/umap.csv")
    dot_c = pd.read_csv(f"{DATA}/panelC_marker_dot.csv")
    dot_d = pd.read_csv(f"{DATA}/panelD_selected.csv")

    present = [c for c in CELL_TYPES if c in set(um.cell_type)]

    fig = plt.figure(figsize=(183 / 25.4, 176 / 25.4), dpi=300)
    fig.patch.set_facecolor("white")

    if os.environ.get("RIF_CAPTION", "0") == "1":
        fig.text(0.5, 0.994, "Figure 1. Single-cell atlas of LH+7 endometrium in RIF and fertile donors",
                 ha="center", va="top", fontsize=8.5)

    # ---------------- Row 1: A / B UMAP ----------------
    axA = fig.add_axes([0.045, 0.690, 0.235, 0.250])
    axA_leg = fig.add_axes([0.285, 0.690, 0.115, 0.250]); axA_leg.set_axis_off()
    axB = fig.add_axes([0.470, 0.690, 0.235, 0.250])
    axB_leg = fig.add_axes([0.710, 0.690, 0.075, 0.250]); axB_leg.set_axis_off()

    draw_umap(axA, um, "cell_type", CT_COLORS, present)
    axA.set_title(f"Cell types (n = {len(present)})", fontsize=7, pad=3)
    panel_letter(axA, "A", x=-0.14, y=1.04)

    for i, ct in enumerate(present):
        axA_leg.scatter(0.05, 0.95 - i * (0.90 / max(len(present) - 1, 1)), s=14,
                        c=CT_COLORS[ct], linewidths=0, clip_on=False)
        axA_leg.text(0.15, 0.95 - i * (0.90 / max(len(present) - 1, 1)), ct,
                     fontsize=6, va="center")
    axA_leg.set_xlim(0, 1); axA_leg.set_ylim(0, 1)

    groups = ["Fertile", "RIF"]
    # Equal-size downsampling for visualisation only (12,000 cells/group,
    # stratified by donor).  Never used for statistics.
    rng = np.random.default_rng(20261117)
    N_VIS = 12000
    keep = []
    for g in groups:
        d = um[um.group == g]
        alloc = (d["sample"].value_counts(normalize=True) * N_VIS).round().astype(int)
        diff = N_VIS - int(alloc.sum())
        if diff != 0:
            alloc.iloc[0] += diff
        for s, n in alloc.items():
            idx = d.index[d["sample"] == s]
            n = min(int(n), len(idx))
            keep += list(rng.choice(idx, n, replace=False))
    um_b = um.loc[sorted(keep)]
    draw_umap(axB, um_b, "group", GROUP_COLORS, groups, s=0.6, alpha=0.75)
    n_raw = um.groupby("group").size()
    axB.set_title(f"Group ({N_VIS:,} cells/group shown)", fontsize=7, pad=3)
    panel_letter(axB, "B", x=-0.30, y=1.04)
    n_samp = um.groupby("group")["sample"].nunique()
    for i, g in enumerate(groups):
        axB_leg.scatter(0.08, 0.72 - i * 0.22, s=16, c=GROUP_COLORS[g],
                        linewidths=0, clip_on=False)
        axB_leg.text(0.24, 0.72 - i * 0.22, f"{g} ({n_samp[g]} donors)",
                     fontsize=6, va="center")
    axB_leg.text(0.02, 0.24, f"all cells\nFertile / RIF: {n_raw['Fertile']:,} / {n_raw['RIF']:,}",
                 fontsize=5.4, va="top", linespacing=1.25, color="#555555")
    axB_leg.set_xlim(0, 1); axB_leg.set_ylim(0, 1)

    # ---------------- Row 2: C marker dot plot ----------------
    dot_c = dot_c[dot_c.cell_type.isin(present)].copy()
    order_c = []
    for ct in present:
        genes = dot_c.loc[dot_c.marker_group == ct, "gene"].drop_duplicates().tolist()
        order_c += [g for g in genes if g not in order_c]
    dot_c = dot_c[dot_c["gene"].isin(order_c)].copy()
    dot_c["gene_column"] = pd.Categorical(dot_c["gene"], categories=order_c, ordered=True)
    dot_c = dot_c.sort_values(["cell_type", "gene_column"]).reset_index(drop=True)
    cols_c = order_c

    axC = fig.add_axes([0.112, 0.385, 0.745, 0.220])
    axC_leg = fig.add_axes([0.858, 0.400, 0.055, 0.170]); axC_leg.set_axis_off()
    caxC = fig.add_axes([0.925, 0.450, 0.010, 0.110])

    pctC = dot_c["pct_exp"].to_numpy()
    valC = dot_c["avg_exp_scaled"].to_numpy()
    sizeC = S_MIN + (S_MAX - S_MIN) * np.sqrt(np.clip(pctC, 0, 100) / 100.0)
    scC = axC.scatter(
        [cols_c.index(g) for g in dot_c["gene"]],
        [present.index(c) for c in dot_c["cell_type"]],
        s=sizeC, c=np.clip(valC, 0, 1), cmap=DOT_CMAP,
        edgecolors=DOT_EDGE, linewidths=DOT_EDGE_LW)
    axC.set_xlim(-0.7, len(cols_c) - 0.3); axC.set_ylim(len(present) - 0.3, -0.7)
    axC.set_xticks(range(len(cols_c)))
    axC.set_xticklabels(cols_c, rotation=90, ha="center", va="top", fontsize=5.4,
                        fontstyle="italic")
    axC.set_yticks(range(len(present))); axC.set_yticklabels(present, fontsize=6)
    axC.tick_params(axis="x", length=2.5, width=0.5, color="#666666", pad=1.5)
    axC.tick_params(axis="y", length=0, pad=2)
    for side in ["left", "bottom"]:
        axC.spines[side].set_color("#8d8d8d"); axC.spines[side].set_linewidth(0.6)
    # separators between cell-type gene blocks
    prev = None
    for i, g in enumerate(cols_c):
        ct_g = dot_c.loc[dot_c.gene == g, "marker_group"].iloc[0]
        if prev is not None and ct_g != prev:
            axC.axvline(i - 0.5, color="#DDDDDD", lw=0.4, zorder=0)
        prev = ct_g
    panel_letter(axC, "C", x=-0.134, y=1.05)
    axC.set_title("Known endometrial cell-type markers", fontsize=7, pad=3)

    # C size legend + colorbar
    levels = [25, 50, 75, 100]
    ys = np.linspace(0.86, 0.10, len(levels))
    for yv, lv in zip(ys, levels):
        s = S_MIN + (S_MAX - S_MIN) * np.sqrt(lv / 100.0)
        axC_leg.scatter(0.30, yv, s=s, c="#8f8f8f", edgecolors="#5e5e5e", linewidths=0.4)
        axC_leg.text(0.55, yv, f"{lv}", va="center", fontsize=5.5)
    axC_leg.text(0.02, 0.48, "% cells", rotation=90, va="center",
                 ha="center", fontsize=5.5)
    axC_leg.set_xlim(0, 1); axC_leg.set_ylim(0, 1)
    cbC = plt.colorbar(scC, cax=caxC)
    cbC.set_ticks([0, 0.5, 1])
    cbC.ax.tick_params(labelsize=5.4, length=2, width=0.5)
    cbC.outline.set_edgecolor("#5e5e5e"); cbC.outline.set_linewidth(0.5)
    caxC.set_title("Scaled\nmean", fontsize=5.5, pad=2)

    # ---------------- Row 3: D top DEG per cell type ----------------
    axD = fig.add_axes([0.112, 0.075, 0.745, 0.220])
    axD_leg = fig.add_axes([0.858, 0.090, 0.055, 0.170]); axD_leg.set_axis_off()
    caxD = fig.add_axes([0.925, 0.140, 0.010, 0.110])

    if len(dot_d) > 0:
        dot_d = dot_d[dot_d.cell_type.isin(present) & dot_d["gene"].notna()].copy()
        deg_full = pd.read_csv(f"{DATA}/panelD_deg_q.csv")
        BLOCK, GAP = 4, 0.9
        STEP = BLOCK + GAP
        blocks = {}
        for src in present:
            d = dot_d[dot_d.cell_type == src].copy()
            if d.empty:
                blocks[src] = []
                continue
            up = d[d.log2FC > 0].nlargest(2, "log2FC")
            dn = d[d.log2FC < 0].nsmallest(2, "log2FC")
            blocks[src] = pd.concat([up, dn])["gene"].tolist()[:BLOCK]
        cols_d, xlabels_d = [], []
        for k, src in enumerate(present):
            for j in range(BLOCK):
                cols_d.append(k * STEP + j)
                genes = blocks[src]
                g = genes[j] if j < len(genes) else ""
                xlabels_d.append(("↑ " if j < 2 else "↓ ") + g if g else "")
        # full matrix: every cell type (rows) x every selected gene (columns)
        rec = []
        for yi, ct in enumerate(present):
            sub = deg_full[deg_full.cell_type == ct]
            for k, src in enumerate(present):
                for j, g in enumerate(blocks[src]):
                    r = sub[sub.names == g]
                    if r.empty:
                        continue
                    rec.append({"y": yi, "x": k * STEP + j,
                                "gene": g, "log2FC": r.logfoldchanges.iloc[0],
                                "pct_rif": r.pct_rif.iloc[0],
                                "padj": r.pvals_adj.iloc[0]})
        dd = pd.DataFrame(rec).dropna(subset=["log2FC"])
        missing = [ct for ct in present if not blocks[ct]]
        pctD = dd["pct_rif"].to_numpy()
        sizeD = S_MIN + (S_MAX - S_MIN) * np.sqrt(np.clip(pctD, 0, 100) / 100.0)
        normD = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
        face = np.clip(dd["log2FC"], -2, 2)
        scD = axD.scatter(dd["x"], dd["y"], s=sizeD, c=face,
                          cmap=DOT_CMAP, norm=normD,
                          edgecolors=DOT_EDGE, linewidths=DOT_EDGE_LW)
        axD.set_xlim(-0.7, len(present) * STEP - GAP - 0.3)
        axD.set_ylim(len(present) - 0.3, -0.7)
        axD.set_xticks(cols_d)
        axD.set_xticklabels(xlabels_d, rotation=90, ha="center", va="top",
                            fontsize=5.4, fontstyle="italic")
        axD.set_yticks(range(len(present))); axD.set_yticklabels(present, fontsize=6)
        axD.tick_params(axis="x", length=2.5, width=0.5, color="#666666", pad=1.5)
        axD.tick_params(axis="y", length=0, pad=2)
        for side in ["left", "bottom"]:
            axD.spines[side].set_color("#8d8d8d"); axD.spines[side].set_linewidth(0.6)
        # block separators: one column-block per cell type
        for k in range(1, len(present)):
            axD.axvline(k * STEP - GAP / 2 - 0.5, color="#CFCFCF", lw=0.5, zorder=0)
        if missing:
            for ct in missing:
                k = present.index(ct)
                axD.text(k * STEP + (BLOCK - 1) / 2, k,
                         "no DEG", fontsize=5.4, color="#777777", va="center", ha="center")
        for yv, lv in zip(ys, levels):
            s = S_MIN + (S_MAX - S_MIN) * np.sqrt(lv / 100.0)
            axD_leg.scatter(0.30, yv, s=s, c="#8f8f8f", edgecolors="#5e5e5e",
                            linewidths=0.4)
            axD_leg.text(0.55, yv, f"{lv}", va="center", fontsize=5.5)
        axD_leg.text(0.02, 0.48, "RIF cells (%)", rotation=90, va="center",
                     ha="center", fontsize=5.5)
        axD_leg.set_xlim(0, 1); axD_leg.set_ylim(0, 1)
        cbD = plt.colorbar(scD, cax=caxD)
        cbD.set_ticks([-2, -1, 0, 1, 2])
        cbD.ax.tick_params(labelsize=5.4, length=2, width=0.5)
        cbD.outline.set_edgecolor("#5e5e5e"); cbD.outline.set_linewidth(0.5)
        caxD.set_title("log$_2$FC\n(RIF/Fertile)", fontsize=5.5, pad=2)
        axD.set_title("Top DEGs per cell type", fontsize=7, pad=3)
    else:
        axD.text(0.5, 0.5, "No significant DEGs", ha="center", va="center",
                 transform=axD.transAxes, fontsize=7)
    panel_letter(axD, "D", x=-0.134, y=1.05)

    os.makedirs(FIG, exist_ok=True)
    save_cns_figure(fig, f"{FIG}/Fig1_LH7_atlas")
    print("SAVED", f"{FIG}/Fig1_LH7_atlas.pdf", flush=True)


if __name__ == "__main__":
    main()
