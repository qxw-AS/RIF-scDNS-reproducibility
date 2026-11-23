
#!/usr/bin/env python3
"""Fig. S4 data: M2 composition, HLA driver, module GO, IFN score."""
import os
import glob
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

D2 = os.path.join(_ROOT, "data", "fig2")
SC = os.path.join(_ROOT, "data", "scdeg")
OUT = os.path.join(_ROOT, "data", "figS")

CT = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
      "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]
HLA = ["HLA-A", "HLA-B", "HLA-C"]
IFN = ["STAT1", "IRF1", "CXCL9", "CXCL10", "IDO1", "CD74", "GBP1", "GBP2",
       "GBP5", "IFIT1", "IFIT3", "ISG15", "MX1", "OAS1", "HLA-DRA"]

mods = pd.read_csv(f"{D2}/gene_modules.csv")
m2 = mods[mods.module == 2].gene.tolist()
cz = pd.read_csv(f"{D2}/combined_z.csv").set_index("gene")
prof = pd.read_csv(f"{D2}/module_profiles.csv", index_col=0)
Z = pd.read_csv(f"{D2}/gene_z_by_type.csv", index_col=0)

# ---- M2 weighted contribution: HLA vs rest ----
rows = []
for ct in CT:
    h = Z.loc[[g for g in HLA if g in Z.index], ct]
    o = Z.loc[[g for g in m2 if g not in HLA], ct]
    hm = h.mean(skipna=True)
    om = o.mean(skipna=True)
    rows.append({"cell_type": ct, "M2_mean_z": prof.loc[2, ct],
                 "HLA_mean_z": hm, "other_mean_z": om,
                 "HLA_weighted": hm * len(h) / len(m2) if pd.notna(hm) else 0.0,
                 "other_weighted": om * len(o) / len(m2),
                 "n_genes": len(m2), "n_HLA": len(h)})
m2c = pd.DataFrame(rows)
m2c.to_csv(f"{OUT}/s4_m2_contribution.csv", index=False)

# leave-one-gene-out M2 mean z
loo = []
for g in m2:
    vals = Z.loc[[x for x in m2 if x != g]].mean(axis=0)
    loo.append({"removed_gene": g, "spearman_with_full": spearmanr(vals, prof.loc[2]).statistic,
                "max_abs_delta": float((vals - prof.loc[2]).abs().max())})
pd.DataFrame(loo).to_csv(f"{OUT}/s4_m2_loo.csv", index=False)

# ---- expression aggregation from per-sample raw counts ----
files = sorted(glob.glob(f"{SC}/*.npz"))
genes = None
acc = {}   # (ct, group, gene) -> [sum_counts, n_cells, n_expr]
ifn_cells = {ct: {"RIF": [], "Fertile": []} for ct in CT}
for f in files:
    base = f.split("/")[-1][:-4]
    sample, ct = base.split("__", 1)
    ct = ct.replace("B-Plasma", "B/Plasma")
    if ct not in CT:
        continue
    grp = "RIF" if sample.startswith("RIF") else "Fertile"
    d = np.load(f, allow_pickle=True)
    if genes is None:
        genes = d["genes"].astype(str)
        gidx = {g: i for i, g in enumerate(genes)}
    mat = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    n = mat.shape[0]
    targets = [g for g in HLA + IFN if g in gidx]
    for g in targets:
        col = np.asarray(mat[:, gidx[g]].todense()).ravel()
        key = (ct, grp, g)
        if key not in acc:
            acc[key] = [0.0, 0, 0.0]
        acc[key][0] += float(col.sum()); acc[key][1] += n; acc[key][2] += float((col > 0).sum())
    # IFN score: store raw per-cell values; z-score across all cells of the type later
    ok = [g for g in IFN if g in gidx]
    if ok:
        vals = np.column_stack([np.asarray(mat[:, gidx[g]].todense()).ravel() for g in ok])
        ifn_cells[ct][grp].append(vals)

exp_rows = []
for (ct, grp, g), (sc, nc, ne) in acc.items():
    exp_rows.append({"cell_type": ct, "group": grp, "gene": g,
                     "mean_expr": sc / max(nc, 1), "pct_expressed": 100 * ne / max(nc, 1),
                     "n_cells": nc})
pd.DataFrame(exp_rows).to_csv(f"{OUT}/s4_expression.csv", index=False)

ifn_rows = []
for ct in CT:
    vr = np.vstack(ifn_cells[ct]["RIF"]) if ifn_cells[ct]["RIF"] else np.empty((0, len(IFN)))
    vf = np.vstack(ifn_cells[ct]["Fertile"]) if ifn_cells[ct]["Fertile"] else np.empty((0, len(IFN)))
    v = np.vstack([vr, vf])
    if len(v):
        z = (v - v.mean(0)) / np.where(v.std(0) < 1e-9, 1, v.std(0))
        score = z.mean(1)
        ifn_rows.append({"cell_type": ct, "group": "RIF", "ifn_score": float(score[:len(vr)].mean()),
                         "n_cells": int(len(vr))})
        ifn_rows.append({"cell_type": ct, "group": "Fertile", "ifn_score": float(score[len(vr):].mean()),
                         "n_cells": int(len(vf))})
pd.DataFrame(ifn_rows).to_csv(f"{OUT}/s4_ifn_score.csv", index=False)

# ---- M2 GO terms ----
go = pd.read_csv(f"{D2}/go_enrichment_full.csv")
g2 = go[go.module == 2].sort_values("Adjusted P-value").head(8).copy()
g2["n_genes"] = g2.Genes.fillna("").apply(lambda s: len([x for x in str(s).split(";") if x]))
g2[["Term_clean", "Adjusted P-value", "n_genes", "Genes"]].to_csv(f"{OUT}/s4_m2_go.csv", index=False)

print(m2c.to_string(index=False))
print(pd.DataFrame(ifn_rows).pivot(index="cell_type", columns="group", values="ifn_score").to_string())
print("S4_DONE")
