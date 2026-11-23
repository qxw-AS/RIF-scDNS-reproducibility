
#!/usr/bin/env python3
"""Stage 7c: divergence-vs-DEG top-N overlap curves (Fig.3C).

For every cell type:
  * divergence ranking = |gene_z_by_type[cell_type]| among genes testable for
    divergence in that type (>=5% detection in both groups, Fig.2 pipeline);
  * DEG ranking = scanpy Wilcoxon pvals_adj (Fig.1D), genome-wide (36,601
    genes), ties broken by p-value then |log2FC|;
  * observed Jaccard of the two top-N sets, N = 50..800;
  * permutation null (10,000 draws): intersection ~ Hypergeometric(36,601,
    N_DEG, N_div), i.e. random re-ranking of one method with respect to the
    other; this is exact because N_div genes are a fixed-size sample of the
    transcriptome and the DEG top-N is a fixed set;
  * N is capped at the number of testable genes for a cell type (Epithelial
    642, Mast 662), so those types stop at N=600;
  * "network-only" count at N=50 = top-50 divergence genes with DEG
    pvals_adj > 0.05 (pre-registered H1 criterion).

A global sensitivity analysis (combined_z vs min-padj across cell types) is
written to jaccard_global.csv but is not used in Fig.3 (the analysis unit of
scDNS is the cell type; see the statistics report).
"""
import os
import numpy as np
import pandas as pd

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

D = os.path.join(_ROOT, "data", "fig2")
OUT = os.path.join(_ROOT, "data", "fig3")
NS = [50, 100, 200, 400, 800]
N_PERM = 10000
rng = np.random.default_rng(20261117)

Z = pd.read_csv(f"{D}/gene_z_by_type.csv", index_col=0)
deg = pd.read_csv(os.path.join(_ROOT, "data", "fig1", "panelD_deg_full.csv"))
M = deg.names.nunique()
print("DEG universe:", M, flush=True)


def deg_rank(ct, universe=None):
    d = deg[deg.cell_type == ct].copy()
    d["absfc"] = d.logfoldchanges.abs()
    d = d.sort_values(["pvals_adj", "pvals", "absfc"], ascending=[True, True, False])
    if universe is not None:
        d = d[d.names.isin(universe)]
    return d.names.tolist()


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


def null_jac(na, nb, n_perm=N_PERM):
    inter = rng.hypergeometric(nb, M - nb, na, size=n_perm)
    return inter / (na + nb - inter)


rows, summ = [], []
for ct in Z.columns:
    z = Z[ct].dropna()
    div = z.abs().sort_values(ascending=False).index.tolist()
    degl = deg_rank(ct)
    for N in NS:
        if N > len(div):
            rows.append({"cell_type": ct, "N": N, "n_div": len(div), "n_deg": N,
                         "shared": np.nan, "jaccard": np.nan, "null_mean": np.nan,
                         "null_lo": np.nan, "null_hi": np.nan, "p_upper": np.nan,
                         "note": "N>testable"})
            continue
        a, b = div[:N], degl[:N]
        inter = len(set(a) & set(b))
        obs = inter / (2 * N - inter)
        nj = null_jac(N, N)
        rows.append({"cell_type": ct, "N": N, "n_div": len(div), "n_deg": N,
                     "shared": inter, "jaccard": obs,
                     "null_mean": float(nj.mean()),
                     "null_lo": float(np.quantile(nj, .025)),
                     "null_hi": float(np.quantile(nj, .975)),
                     "p_upper": float((1 + (nj >= obs).sum()) / (1 + len(nj))),
                     "note": ""})
    # pre-registered N=50 decision inputs
    a, b = div[:50], degl[:50]
    d50 = deg[deg.cell_type == ct].set_index("names").reindex(a)
    nonsig = int(((d50.pvals_adj > 0.05) | d50.pvals_adj.isna()).sum())
    j50 = len(set(a) & set(b)) / len(set(a) | set(b))
    rule = "H1 supported" if (j50 <= 0.5 and nonsig >= 5) else "review"
    summ.append({"cell_type": ct, "n_testable": len(div),
                 "jaccard_N50": j50, "shared_N50": len(set(a) & set(b)),
                 "network_only_N50": nonsig, "decision_rule": rule,
                 "n_rif_cells": int(deg[deg.cell_type == ct].n_rif_cells.iloc[0]),
                 "n_fer_cells": int(deg[deg.cell_type == ct].n_fer_cells.iloc[0])})
    print(f"{ct}: testable {len(div)}, J50 {j50:.3f}, network-only {nonsig}", flush=True)

cur = pd.DataFrame(rows)
cur.to_csv(f"{OUT}/jaccard_curves.csv", index=False)
pd.DataFrame(summ).to_csv(f"{OUT}/jaccard_summary.csv", index=False)

# ---------------- global sensitivity (not used in Fig.3) ----------------
cz = pd.read_csv(f"{D}/combined_z.csv")
cz = cz.dropna(subset=["combined_z"])
divg = cz.reindex(cz.combined_z.abs().sort_values(ascending=False).index).gene.tolist()
gmin = deg.groupby("names").pvals_adj.min()
gl = gmin.sort_values().index.tolist()
grows = []
for N in NS:
    a, b = divg[:N], gl[:N]
    inter = len(set(a) & set(b))
    obs = inter / (2 * N - inter)
    nj = null_jac(N, N)
    grows.append({"N": N, "shared": inter, "jaccard": obs,
                  "null_mean": float(nj.mean()),
                  "p_upper": float((1 + (nj >= obs).sum()) / (1 + len(nj)))})
pd.DataFrame(grows).to_csv(f"{OUT}/jaccard_global.csv", index=False)
print(pd.DataFrame(grows).to_string(index=False), flush=True)
print("STAGE7C_DONE", flush=True)
