
#!/usr/bin/env python3
"""Stage 10b: Harmony + Leiden + marker-based annotation for GSE248784."""
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

BASE = os.path.join(_ROOT, "data", "external", "gse248784")
PROC = os.path.join(BASE, "processed")
SEED = 20261117

adata = sc.read_h5ad(os.path.join(PROC, "external_qc.h5ad"))
print("loaded", adata.shape, flush=True)

sc.pp.highly_variable_genes(adata, n_top_genes=1500, batch_key="sample")
hv = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(hv, max_value=10)
sc.tl.pca(hv, n_comps=50, svd_solver="arpack", random_state=SEED)
sce.pp.harmony_integrate(hv, "sample", max_iter_harmony=30)
adata.obsm["X_pca_harmony"] = hv.obsm["X_pca_harmony"]
sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca_harmony", random_state=SEED)
sc.tl.leiden(adata, resolution=0.8, key_added="leiden", random_state=SEED,
             flavor="igraph", n_iterations=2, directed=False)
sc.tl.umap(adata, min_dist=0.3, random_state=SEED)
print("clusters:", adata.obs.leiden.nunique(), flush=True)
print(adata.obs.leiden.value_counts().sort_index().to_string(), flush=True)

# ---------------- marker-based cluster annotation ----------------
MARKERS = json.load(open(os.path.join(BASE, "marker_sets.json")))
marker_genes = sorted({g for v in MARKERS.values() for g in v})
present = [g for g in marker_genes if g in adata.var_names]
clusters = list(adata.obs.leiden.cat.categories)

M = np.zeros((len(clusters), len(present)), dtype=float)
frac = np.zeros_like(M)
for i, cl in enumerate(clusters):
    sub = adata[adata.obs.leiden == cl]
    M[i] = np.asarray(sub[:, present].X.mean(0)).ravel()
    frac[i] = np.asarray((sub[:, present].X > 0).mean(0)).ravel()
Z = (M - M.mean(0)) / (M.std(0) + 1e-8)

score = {}
for ct, genes in MARKERS.items():
    idx = [j for j, g in enumerate(present) if g in set(genes)]
    score[ct] = Z[:, idx].mean(1) if idx else np.full(len(clusters), -np.inf)
S = pd.DataFrame(score, index=clusters)
assign = S.idxmax(1)

# rank genes per cluster for inspection
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", n_genes=15)
top = {}
for cl in clusters:
    df = sc.get.rank_genes_groups_df(adata, group=cl).head(15)
    top[cl] = ",".join(df.names.astype(str))

ann = pd.DataFrame({"cluster": clusters,
                    "n_cells": [int((adata.obs.leiden == c).sum()) for c in clusters],
                    "assigned": assign.values,
                    "top_markers": [top[c] for c in clusters]})
for ct in MARKERS:
    ann[f"score_{ct}"] = S[ct].values
ann.to_csv(os.path.join(PROC, "external_cluster_annotation.csv"), index=False)
print(ann[["cluster", "n_cells", "assigned", "top_markers"]].to_string(index=False), flush=True)

# majority group per cluster (QC: check RIF/Normal mixing)
ct = pd.crosstab(adata.obs.leiden, adata.obs.group)
ct.to_csv(os.path.join(PROC, "external_cluster_group_counts.csv"))
print(ct.to_string(), flush=True)

# map smooth muscle to stromal for discovery comparability
mapped = assign.replace({"Smooth muscle": "Stromal"}).to_dict()
adata.obs["cell_type"] = adata.obs.leiden.map(mapped).astype("category")
adata.write_h5ad(os.path.join(PROC, "external_annotated.h5ad"), compression="gzip")

pd.DataFrame(M, index=clusters, columns=present).to_csv(
    os.path.join(PROC, "external_marker_mean_by_cluster.csv"))
pd.DataFrame(frac, index=clusters, columns=present).to_csv(
    os.path.join(PROC, "external_marker_frac_by_cluster.csv"))
print("cell types:", adata.obs.cell_type.value_counts().to_dict(), flush=True)
print("STAGE10B_DONE", flush=True)
