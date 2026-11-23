
#!/usr/bin/env python3
"""Stage 6c: differential rewiring modules (hierarchical clustering + cut).

dynamicTreeCut (R) is unavailable in this Python-only runtime, so the
pre-registered fallback is used: Ward hierarchical clustering of gene-level
z-profiles across the 11 cell types, a distance cut chosen to yield 8-12
modules with >=30 genes, and iterative merging of small modules into the
nearest module centroid. Declared in Fig2_statistics_report.md.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, to_tree
from scipy.spatial.distance import cdist

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

DATA = os.path.join(_ROOT, "data", "fig2")
SEED = 20261117
TARGET = (8, 12)
MIN_SIZE = 30
CZ_MIN = 1.5   # effect-size filter: strong rewiring genes only

Z = pd.read_csv(f"{DATA}/gene_z_by_type.csv", index_col=0)
cz = pd.read_csv(f"{DATA}/combined_z.csv")
keep = cz.loc[(cz.combined_z.notna()) & (cz.combined_z.abs() >= CZ_MIN), "gene"].tolist()
Z = Z.loc[[g for g in Z.index if g in set(keep)]]
Zf = Z.fillna(0.0).to_numpy()
print("genes clustered:", Z.shape, flush=True)

Ln = linkage(Zf, method="ward", optimal_ordering=True)
print("linkage done", flush=True)


def merge_small(labels, min_size):
    labels = labels.copy()
    uniq = np.unique(labels)
    big = [u for u in uniq if (labels == u).sum() >= min_size]
    if len(big) <= 1:
        return labels
    cents = {u: Zf[labels == u].mean(0) for u in uniq}
    for u in uniq:
        if u in big:
            continue
        d = {b: np.linalg.norm(cents[u] - cents[b]) for b in big}
        labels[labels == u] = min(d, key=d.get)
    return labels


best = None
for h in np.linspace(2, 40, 380):
    lab = fcluster(Ln, t=h, criterion="distance")
    n0 = len(np.unique(lab))
    if n0 < 2:
        continue
    lab2 = merge_small(lab, MIN_SIZE)
    n = len(np.unique(lab2))
    if TARGET[0] <= n <= TARGET[1]:
        best = (h, lab2, n)
        break
if best is None:
    # fall back: the cut whose merged cluster count is closest to 10
    cands = []
    for h in np.linspace(1, 50, 500):
        lab = fcluster(Ln, t=h, criterion="distance")
        lab2 = merge_small(lab, MIN_SIZE)
        cands.append((abs(len(np.unique(lab2)) - 10), h, lab2))
    cands.sort(key=lambda x: x[0])
    h, lab2 = cands[0][1], cands[0][2]
    best = (h, lab2, len(np.unique(lab2)))
h, labels, nmod = best
print(f"cut height {h:.3f}, modules {nmod}, sizes {np.bincount(labels)[1:]}", flush=True)

# order modules by mean combined_z (M1 = strongest coordination gain in RIF)
czmap = dict(zip(cz.gene, cz.combined_z))
df = pd.DataFrame({"gene": Z.index, "cluster": labels})
df["combined_z"] = df.gene.map(czmap)
order = df.groupby("cluster").combined_z.mean().sort_values(ascending=False).index.tolist()
rename = {c: i + 1 for i, c in enumerate(order)}
df["module"] = df.cluster.map(rename)
df = df.drop(columns="cluster")

prof = Z.assign(module=df.set_index("gene").module).groupby("module").mean()
prof.to_csv(f"{DATA}/module_profiles.csv")
df.to_csv(f"{DATA}/gene_modules.csv", index=False)

summ = df.groupby("module").agg(n_genes=("gene", "size"),
                                mean_combined_z=("combined_z", "mean"),
                                median_combined_z=("combined_z", "median")).reset_index()
summ["n_positive"] = df[df.combined_z > 0].groupby("module").size().reindex(summ.module).fillna(0).astype(int).to_numpy()
summ.to_csv(f"{DATA}/module_summary.csv", index=False)
print(summ.to_string(index=False), flush=True)

# dendrogram leaves reordered once, reused by the figure
dn = dendrogram(Ln, no_plot=True)
np.save(f"{DATA}/dendro_leaves.npy", np.asarray(dn["leaves"]))
np.save(f"{DATA}/dendro_icoord.npy", np.asarray([np.asarray(x) for x in dn["icoord"]], dtype=object), allow_pickle=True)
np.save(f"{DATA}/dendro_dcoord.npy", np.asarray([np.asarray(x) for x in dn["dcoord"]], dtype=object), allow_pickle=True)
json.dump({"cut_height": float(h), "n_modules": int(nmod), "min_size": MIN_SIZE,
           "combined_z_cutoff": CZ_MIN, "n_genes_clustered": int(Z.shape[0])},
          open(f"{DATA}/module_params.json", "w"), indent=1)
print("STAGE6C_DONE", flush=True)
