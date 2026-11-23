
#!/usr/bin/env python3
"""Stage 6e: pseudobulk differential expression for Fig.2D.

Per cell type: summed counts per donor -> CPM -> log1p -> Welch t-test
(RIF vs fertile) across donors; BH across genes within the cell type.
Reported alongside the network divergence result to test whether rewired
genes are also differentially expressed.
"""
import os
import glob, os
import numpy as np
import pandas as pd
from scipy import stats

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

PB = os.path.join(_ROOT, "data", "pseudobulk")
DATA = os.path.join(_ROOT, "data", "fig2")
MIN_CELLS = 10
MIN_RIF, MIN_FER = 5, 3


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


records, genes_ref = [], None
for f in sorted(glob.glob(f"{PB}/*.npz")):
    s = os.path.basename(f)[:-4]
    z = np.load(f, allow_pickle=True)
    g = z["genes"].astype(str)
    if genes_ref is None:
        genes_ref = g
    for i, t in enumerate(z["types"].astype(str)):
        records.append({"sample": s, "cell_type": t, "n_cells": int(z["n_cells"][i]),
                        "sums": z["sums"][i], "detected": z["detected"][i]})

group = {r["sample"]: ("RIF" if r["sample"].startswith("RIF") else "Fertile") for r in records}
types = sorted({r["cell_type"] for r in records})
allrows, tested = [], []
for ct in types:
    rs = [r for r in records if r["cell_type"] == ct and r["n_cells"] >= MIN_CELLS]
    rif = [r for r in rs if group[r["sample"]] == "RIF"]
    fer = [r for r in rs if group[r["sample"]] == "Fertile"]
    if len(rif) < MIN_RIF or len(fer) < MIN_FER:
        print(f"{ct}: insufficient samples {len(rif)}/{len(fer)}", flush=True)
        continue

    def logcpm(r):
        return np.log1p(r["sums"] / max(r["sums"].sum(), 1) * 1e6)

    A = np.vstack([logcpm(r) for r in rif])
    B = np.vstack([logcpm(r) for r in fer])
    cpm_a = np.vstack([r["sums"] / max(r["sums"].sum(), 1) * 1e6 for r in rif]).mean(0)
    cpm_b = np.vstack([r["sums"] / max(r["sums"].sum(), 1) * 1e6 for r in fer]).mean(0)
    pct_a = sum(r["detected"] for r in rif) / max(sum(r["n_cells"] for r in rif), 1) * 100
    pct_b = sum(r["detected"] for r in fer) / max(sum(r["n_cells"] for r in fer), 1) * 100
    lfc = np.log2((cpm_a + 1) / (cpm_b + 1))
    keep = ((cpm_a >= 1) | (cpm_b >= 1)) & ((pct_a >= 10) | (pct_b >= 10))
    idx = np.where(keep)[0]
    t, p = stats.ttest_ind(A[:, idx], B[:, idx], axis=0, equal_var=False)
    df = pd.DataFrame({"cell_type": ct, "gene": genes_ref[idx], "log2FC": lfc[idx],
                       "pval": p, "mean_cpm_rif": cpm_a[idx], "mean_cpm_fer": cpm_b[idx],
                       "pct_rif": pct_a[idx], "pct_fer": pct_b[idx],
                       "n_rif_samples": len(rif), "n_fer_samples": len(fer)})
    df["padj"] = bh(df["pval"].to_numpy())
    allrows.append(df)
    tested.append({"cell_type": ct, "n_rif_samples": len(rif), "n_fer_samples": len(fer),
                   "n_genes_tested": len(df),
                   "n_padj_lt_0.05": int((df.padj < 0.05).sum())})
    print(f"{ct}: {len(rif)}/{len(fer)} samples, tested {len(df)}, "
          f"FDR<0.05 {int((df.padj < 0.05).sum())}", flush=True)

deg = pd.concat(allrows, ignore_index=True)
deg.to_csv(f"{DATA}/pseudobulk_deg_full.csv", index=False)
pd.DataFrame(tested).to_csv(f"{DATA}/pseudobulk_deg_summary.csv", index=False)

mods = pd.read_csv(f"{DATA}/gene_modules.csv")
top = (mods.reindex(mods.combined_z.abs().sort_values(ascending=False).index)
       .groupby("module").head(3))
sel = deg[deg.gene.isin(top.gene)].merge(
    top[["gene", "module", "combined_z"]], on="gene", how="right")
sel.to_csv(f"{DATA}/panelD_selected.csv", index=False)
print("selected genes:", top.gene.nunique(), "rows:", len(sel), flush=True)
print(sel.groupby("module").gene.nunique().to_string(), flush=True)
print("STAGE6E_DONE", flush=True)
