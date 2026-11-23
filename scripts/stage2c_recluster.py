
#!/usr/bin/env python3
"""Stage 2c: re-cluster at higher resolution to resolve rare endometrial types,
then annotate clusters by z-scored canonical marker-set expression.

Reuses the Harmony embedding from stage 2 (no re-integration needed).
"""
import os, gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

OUT = os.path.join(_ROOT, "data", "processed")
sc.settings.seed = 20261117
sc.settings.n_jobs = 3

MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "ELF3"],
    "Glandular": ["PAEP", "SPP1", "GPX3", "DKK1"],
    "Ciliated": ["FOXJ1", "TPPP3", "PIFO", "CAPS", "RSPH1"],
    "Stromal": ["DCN", "LUM", "COL1A1", "COL1A2", "PGR", "HAND2"],
    "Smooth muscle": ["ACTA2", "MYH11", "DES", "CNN1"],
    "Endothelial": ["PECAM1", "VWF", "PLVAP", "KDR"],
    "Lymphatic EC": ["PROX1", "LYVE1", "CCL21", "MMRN2"],
    "T cell": ["CD3D", "CD3E", "IL7R", "CD2", "CD247"],
    "NK cell": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "B/Plasma": ["CD79A", "MS4A1", "JCHAIN", "MZB1"],
    "Myeloid": ["CD68", "CD14", "C1QA", "MRC1", "LYZ"],
    "DC": ["CD1C", "LAMP3", "HLA-DRA", "CLEC9A"],
    "Mast": ["TPSAB1", "CPA3", "KIT", "MS4A2"],
}

obs = pd.read_pickle(f"{OUT}/obs.pkl")
genes = pd.read_csv(f"{OUT}/genes.csv", header=None)[0].astype(str).tolist()
adata = ad.AnnData(obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
adata.obsm["X_pca_harmony"] = np.load(f"{OUT}/X_pca_harmony.npy")
print("loaded", adata.shape, flush=True)

sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony")
print("neighbors done", flush=True)
sc.tl.leiden(adata, resolution=2.0, key_added="leiden", flavor="igraph",
             n_iterations=2, directed=False)
print("leiden clusters", adata.obs.leiden.nunique(), flush=True)
sc.tl.umap(adata, min_dist=0.3)
np.save(f"{OUT}/umap.npy", adata.obsm["X_umap"])
adata.obs[["leiden"]].to_pickle(f"{OUT}/leiden.pkl")

# ---- per-cluster marker-set z-scores from the parked expression matrix ----
z = np.load(f"{OUT}/X_compact.npz")
X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
adata2 = ad.AnnData(X=X, obs=adata.obs.copy(),
                    var=pd.DataFrame(index=pd.Index(genes, name="gene")))
del z
gc.collect()

score = pd.DataFrame(index=adata2.obs["leiden"].cat.categories, dtype=float)
codes = adata2.obs["leiden"].cat.codes.to_numpy()
cats = adata2.obs["leiden"].cat.categories
ind = sp.csr_matrix((np.ones(len(codes), np.float32),
                     (codes, np.arange(len(codes)))), shape=(len(cats), len(codes)))
agg = np.asarray((ind @ X).todense())            # cluster x gene summed expression
counts = np.bincount(codes, minlength=len(cats)).astype(np.float32)
mean_expr = pd.DataFrame(agg / counts[:, None], index=cats, columns=genes)
detail = {}
for ct, gs in MARKERS.items():
    gs = [g for g in gs if g in mean_expr.columns]
    detail[ct] = mean_expr[gs].mean(axis=1)
    score[ct] = detail[ct]
del agg
gc.collect()
gene_z = (mean_expr - mean_expr.mean()) / (mean_expr.std() + 1e-9)
score_z = pd.DataFrame(index=score.index)
for ct, gs in MARKERS.items():
    gs = [g for g in gs if g in gene_z.columns]
    score_z[ct] = gene_z[gs].mean(axis=1)

assign = score_z.idxmax(axis=1)
top2 = score_z.apply(lambda r: ", ".join(f"{c}:{v:.2f}" for c, v in r.nlargest(2).items()), axis=1)
sizes = adata2.obs["leiden"].value_counts().reindex(score_z.index)
tab = pd.DataFrame({"n_cells": sizes, "assign": assign, "top2_z": top2,
                    "max_z": score_z.max(axis=1)})
tab.to_csv(f"{OUT}/cluster_assignment.csv")
print(tab.to_string(), flush=True)
score_z.round(2).to_csv(f"{OUT}/cluster_marker_zscores.csv")

adata.obs["cell_type"] = adata.obs["leiden"].map(assign).astype("category")
adata2.obs["cell_type"] = adata.obs["cell_type"].values
adata2.obsm["X_umap"] = adata.obsm["X_umap"]
print(pd.crosstab(adata2.obs["cell_type"], adata2.obs["group"]).to_string(), flush=True)
adata2.write_h5ad(f"{OUT}/lh7_annotated.h5ad", compression="lzf")
print("STAGE2C_DONE", adata2.shape, flush=True)
