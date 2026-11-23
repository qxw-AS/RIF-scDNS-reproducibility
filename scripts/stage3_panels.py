
#!/usr/bin/env python3
"""Stage 3: build Fig.1 panel data from the annotated LH+7 object.
Outputs CSVs: umap coords, marker dotplot (panel C), per-cell-type DEGs (panel D).
"""
import os, sys, json
import numpy as np
import pandas as pd
import scanpy as sc

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

IN = os.path.join(_ROOT, "data", "processed", "lh7_annotated.h5ad")
OUT = os.path.join(_ROOT, "data", "fig1")
os.makedirs(OUT, exist_ok=True)
sc.settings.seed = 20261117

# canonical endometrial markers, one compact set per cell type (panel C)
PANEL_C = {
    "Epithelial":        ["KRT8", "KRT18", "KRT19", "ELF3"],
    "Glandular":         ["PAEP", "SPP1", "GPX3", "DKK1"],
    "Ciliated":          ["FOXJ1", "TPPP3", "PIFO", "RSPH1"],
    "Stromal":           ["DCN", "LUM", "COL1A1", "HAND2"],
    "Endothelial":       ["PECAM1", "VWF", "PLVAP", "KDR"],
    "Lymphatic EC":      ["PROX1", "CCL21", "MMRN2", "CLDN5"],
    "T cell":            ["CD3D", "CD3E", "IL7R", "CD2"],
    "NK cell":           ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "B/Plasma":          ["CD79A", "MS4A1", "JCHAIN", "MZB1"],
    "Myeloid":           ["CD68", "CD14", "C1QA", "MRC1"],
    "Mast":              ["TPSAB1", "CPA3", "KIT", "MS4A2"],
}

def main():
    adata = sc.read_h5ad(IN)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    adata = adata[adata.obs["cell_type"] != "Unassigned"].copy()
    order = [c for c in PANEL_C if c in set(adata.obs.cell_type)]
    adata.obs["cell_type"] = pd.Categorical(adata.obs["cell_type"], categories=order, ordered=True)
    print("cells analysed", adata.n_obs, "types", order, flush=True)
    masks = {ct: (adata.obs["cell_type"] == ct).to_numpy() for ct in order}

    # --- panel A/B: UMAP coordinates ---
    um = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP_1", "UMAP_2"])
    um["cell_type"] = adata.obs["cell_type"].astype(str).values
    um["group"] = adata.obs["group"].astype(str).values
    um["sample"] = adata.obs["sample"].astype(str).values
    um.to_csv(f"{OUT}/umap.csv", index=False)

    # --- panel C: marker dotplot ---
    rows = []
    for ct in order:
        m = masks[ct]
        for ctk, genes in PANEL_C.items():
            for g in genes:
                if g not in adata.var_names:
                    continue
                v = adata[m, g].X
                v = np.asarray(v.todense()).ravel() if hasattr(v, "todense") else np.asarray(v).ravel()
                rows.append({"cell_type": ct, "marker_group": ctk, "gene": g,
                             "pct_exp": float((v > 0).mean() * 100),
                             "avg_exp": float(v.mean())})
    dot = pd.DataFrame(rows)
    dot["avg_exp_scaled"] = dot.groupby("gene")["avg_exp"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0)
    dot.to_csv(f"{OUT}/panelC_marker_dot.csv", index=False)

    # --- panel D: per cell type DEG (RIF vs Fertile), Wilcoxon + BH ---
    deg_path = f"{OUT}/panelD_deg_full.csv"
    notes_path = f"{OUT}/panelD_notes.csv"
    if "--reuse-deg" in sys.argv and os.path.exists(deg_path):
        deg = pd.read_csv(deg_path)
        notes = (pd.read_csv(notes_path).to_dict("records")
                 if os.path.exists(notes_path) else [])
        print("reused DEG table", deg.shape, flush=True)
    else:
        deg_rows, notes = [], []
        for ct in order:
            sub = adata[masks[ct]].copy()
            n_rif = int((sub.obs.group == "RIF").sum())
            n_fer = int((sub.obs.group == "Fertile").sum())
            if n_rif < 20 or n_fer < 20:
                notes.append({"cell_type": ct, "n_rif": n_rif, "n_fer": n_fer,
                              "reason": "low cell number: no DEG test"})
                print(ct, n_rif, n_fer, "low cell number", flush=True)
                del sub
                continue
            sc.tl.rank_genes_groups(sub, "group", groups=["RIF"], reference="Fertile",
                                    method="wilcoxon", tie_correct=True, pts=True)
            df = sc.get.rank_genes_groups_df(sub, "RIF")
            df = df.dropna(subset=["logfoldchanges"])
            fer = (sub.obs.group == "Fertile").to_numpy()
            Xf = sub.X[fer]
            pct_f = np.asarray((Xf > 0).mean(axis=0)).ravel() * 100
            df["pct_fer"] = pd.Series(pct_f, index=sub.var_names).reindex(df["names"]).to_numpy()
            df["pct_rif"] = df["pct_nz_group"] * 100
            df["cell_type"] = ct
            df["n_rif"] = n_rif
            df["n_fer"] = n_fer
            deg_rows.append(df)
            print(ct, n_rif, n_fer, "sig", int((df.pvals_adj < 0.05).sum()), flush=True)
            del sub
        deg = pd.concat(deg_rows, ignore_index=True) if deg_rows else pd.DataFrame()
        deg.to_csv(deg_path, index=False)
        pd.DataFrame(notes, columns=["cell_type", "n_rif", "n_fer", "reason"]).to_csv(
            notes_path, index=False)

    # select top 2 up + top 2 down per cell type, padj<0.05
    sel = []
    for ct in order:
        d = deg[deg.cell_type == ct] if len(deg) else deg
        if d.empty:
            if not any(n["cell_type"] == ct for n in notes) and not deg.empty:
                notes.append({"cell_type": ct, "n_rif": np.nan, "n_fer": np.nan,
                              "reason": "no significant DEG"})
            continue
        d = d[d.pvals_adj < 0.05]
        up = d[d.logfoldchanges > 0].sort_values("scores", ascending=False).head(2)
        dn = d[d.logfoldchanges < 0].sort_values("scores").head(2)
        sel.append(pd.concat([up, dn]))
    sel = pd.concat(sel, ignore_index=True) if sel else pd.DataFrame(
        columns=["names", "scores", "logfoldchanges", "pvals", "pvals_adj",
                 "pct_nz_group", "pct_fer", "pct_rif", "cell_type", "n_rif", "n_fer"])
    sel = sel.rename(columns={"names": "gene"})
    sel.to_csv(f"{OUT}/panelD_selected.csv", index=False)
    pd.DataFrame(notes, columns=["cell_type", "n_rif", "n_fer", "reason"]).to_csv(
        f"{OUT}/panelD_notes.csv", index=False)

    counts = pd.crosstab(adata.obs["cell_type"], adata.obs["group"])
    counts.to_csv(f"{OUT}/celltype_group_counts.csv")
    meta = {"n_cells": int(adata.n_obs), "n_samples": int(adata.obs["sample"].nunique()),
            "n_cell_types": len(order), "cell_types": order,
            "n_rif": int((adata.obs.group == "RIF").sum()),
            "n_fer": int((adata.obs.group == "Fertile").sum()),
            "cells_per_type": {k: int(v) for k, v in
                               adata.obs["cell_type"].value_counts().items()}}
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print("STAGE3_DONE", meta, flush=True)

if __name__ == "__main__":
    main()
