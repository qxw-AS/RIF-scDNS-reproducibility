#!/usr/bin/env python3
"""Stage 6c-sens: cut-height sensitivity of the module partition.

Re-derives the Ward partition of stage6c_modules.py over a dense grid of cut
heights and records, at each height, the number of modules and how much the
module that carries the MHC-I signal (M2) changes.  Written to answer the
obvious reviewer question: is M2 an artefact of the cut height chosen by the
8-12 module rule?

Input : data/fig2/gene_z_by_type.csv, data/fig2/combined_z.csv,
        data/fig2/gene_modules.csv (published partition, used as reference)
Output: data/fig2/module_cutheight_sensitivity.csv

Run from the package root:  python scripts/stage6c_cutheight_sensitivity.py
"""
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.environ.get("RIF_DATA", os.path.join(_ROOT, "data", "fig2"))

MIN_SIZE = 30
CZ_MIN = 1.5
GRID = np.round(np.arange(5.0, 16.001, 0.05), 2)
HLA = ["HLA-A", "HLA-B", "HLA-C"]

Z = pd.read_csv(os.path.join(DATA, "gene_z_by_type.csv"), index_col=0)
cz = pd.read_csv(os.path.join(DATA, "combined_z.csv"))
keep = cz.loc[(cz.combined_z.notna()) & (cz.combined_z.abs() >= CZ_MIN), "gene"].tolist()
Z = Z.loc[[g for g in Z.index if g in set(keep)]]
Zf = Z.fillna(0.0).to_numpy()

ref = pd.read_csv(os.path.join(DATA, "gene_modules.csv")).set_index("gene").module.reindex(Z.index)
m2_ref = set(ref[ref == 2].index)
genes = list(Z.index)
hla_idx = [genes.index(g) for g in HLA]


def merge_small(labels, min_size):
    """Merge clusters smaller than `min_size` into the nearest large centroid."""
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


Ln = linkage(Zf, method="ward", optimal_ordering=True)

rows = []
for h in GRID:
    lab = merge_small(fcluster(Ln, t=h, criterion="distance"), MIN_SIZE)
    best_j, best_k = 0.0, -1
    for k in np.unique(lab):
        s = set(np.asarray(genes)[lab == k])
        j = len(m2_ref & s) / len(m2_ref | s)
        if j > best_j:
            best_j, best_k = j, k
    rows.append({
        "cut_height": float(h),
        "n_modules": int(len(np.unique(lab))),
        "best_matching_module_size": int((lab == best_k).sum()),
        "m2_jaccard_vs_published": round(best_j, 4),
        "hla_abc_co_clustered": bool(len({lab[i] for i in hla_idx}) == 1),
    })

out = pd.DataFrame(rows)
out.to_csv(os.path.join(DATA, "module_cutheight_sensitivity.csv"), index=False)

stable = out[(out.m2_jaccard_vs_published == 1.0) & out.hla_abc_co_clustered]
print(f"grid {len(out)} heights, {GRID.min():.2f}-{GRID.max():.2f}")
print(f"M2 recovered intact (Jaccard = 1.00, HLA co-clustered) from "
      f"h = {stable.cut_height.min():.2f} to {stable.cut_height.max():.2f} "
      f"({len(stable)} grid points)")
print(f"modules over that window: {stable.n_modules.min()}-{stable.n_modules.max()}")
print("STAGE6C_SENS_DONE")
