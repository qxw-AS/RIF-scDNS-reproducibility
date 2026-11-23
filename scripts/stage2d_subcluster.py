
#!/usr/bin/env python3
"""Stage 2d: hierarchical annotation.

Broad lineages from the resolution-2.0 Leiden clustering are re-clustered
within lineage and annotated with lineage-specific canonical marker sets,
so that rare populations (DC, mast, plasma, lymphatic endothelium, smooth
muscle, glandular epithelium) are resolved instead of being absorbed by
abundant neighbours.
"""
import os, gc, json
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

LINEAGE_MARKERS = {
    "Epithelial": {
        "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "ELF3"],
        "Glandular": ["PAEP", "SPP1", "GPX3", "DKK1"],
        "Ciliated": ["FOXJ1", "TPPP3", "PIFO", "CAPS", "RSPH1"],
    },
    "Stromal": {
        "Stromal": ["DCN", "LUM", "COL1A1", "COL1A2", "PGR", "HAND2"],
        "Smooth muscle": ["ACTA2", "MYH11", "DES", "CNN1"],
    },
    "Endothelial": {
        "Endothelial": ["PECAM1", "VWF", "PLVAP", "KDR", "ADGRL4"],
        "Lymphatic EC": ["PROX1", "LYVE1", "CCL21", "MMRN2"],
    },
    "Immune": {
        "T cell": ["CD3D", "CD3E", "IL7R", "CD2"],
        "NK cell": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
        "Myeloid": ["CD68", "CD14", "C1QA", "MRC1", "LYZ", "CD163"],
        "DC": ["CD1C", "LAMP3", "CLEC9A", "HLA-DRA"],
        "B/Plasma": ["CD79A", "MS4A1", "JCHAIN", "MZB1"],
        "Mast": ["TPSAB1", "CPA3", "KIT", "MS4A2"],
    },
}
LINEAGE_OF = {"Epithelial": "Epithelial", "Glandular": "Epithelial", "Ciliated": "Epithelial",
              "Stromal": "Stromal", "Smooth muscle": "Stromal",
              "Endothelial": "Endothelial", "Lymphatic EC": "Endothelial",
              "T cell": "Immune", "NK cell": "Immune", "Myeloid": "Immune",
              "DC": "Immune", "B/Plasma": "Immune", "Mast": "Immune"}
RES = {"Epithelial": 1.2, "Stromal": 1.0, "Endothelial": 0.8, "Immune": 1.5}

obs = pd.read_pickle(f"{OUT}/obs.pkl")
genes = pd.read_csv(f"{OUT}/genes.csv", header=None)[0].astype(str).tolist()
ca = pd.read_csv(f"{OUT}/cluster_assignment.csv", index_col=0)
ca.index = ca.index.astype(str)
obs["leiden"] = pd.read_pickle(f"{OUT}/leiden.pkl")["leiden"].values
obs["broad"] = obs["leiden"].astype(str).map(ca["assign"])
obs["lineage"] = obs["broad"].map(LINEAGE_OF)
print(obs["lineage"].value_counts().to_string(), flush=True)

z = np.load(f"{OUT}/X_compact.npz")
X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
del z
gc.collect()
gidx = {g: i for i, g in enumerate(genes)}
harmony = np.load(f"{OUT}/X_pca_harmony.npy")

cell_type = obs["broad"].astype(object).values
obs["sub"] = pd.NA
rows = []

for lin, markers in LINEAGE_MARKERS.items():
    mask = (obs["lineage"] == lin).to_numpy()
    idx = np.where(mask)[0]
    sub = ad.AnnData(obs=obs.iloc[idx].copy(),
                     var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    sub.obsm["X_pca_harmony"] = harmony[idx]
    sc.pp.neighbors(sub, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony")
    sc.tl.leiden(sub, resolution=RES[lin], key_added="sub", flavor="igraph",
                 n_iterations=2, directed=False)
    print(f"[{lin}] n={sub.n_obs} subclusters={sub.obs['sub'].nunique()}", flush=True)

    codes = sub.obs["sub"].cat.codes.to_numpy()
    cats = sub.obs["sub"].cat.categories
    ind = sp.csr_matrix((np.ones(len(codes), np.float32),
                         (codes, np.arange(len(codes)))), shape=(len(cats), len(codes)))
    agg = np.asarray((ind @ X[idx]).todense())
    counts = np.bincount(codes, minlength=len(cats)).astype(np.float32)
    mean_expr = pd.DataFrame(agg / counts[:, None], index=cats, columns=genes)
    del agg
    mean_expr.to_csv(f"{OUT}/means_{lin}.csv")
    pd.DataFrame({"cell": obs.index[idx],
                  "sub": sub.obs["sub"].astype(str).to_numpy()}).to_csv(
        f"{OUT}/sub_{lin}.csv.gz", index=False)
    gz = (mean_expr - mean_expr.mean()) / (mean_expr.std() + 1e-9)
    score_z = pd.DataFrame(index=cats)
    for ct, gs in markers.items():
        gs = [g for g in gs if g in gz.columns]
        score_z[ct] = gz[gs].mean(axis=1)
    assign = score_z.idxmax(axis=1)
    top2 = score_z.apply(lambda r: ", ".join(f"{c}:{v:.2f}" for c, v in r.nlargest(2).items()), axis=1)
    tab = pd.DataFrame({"lineage": lin, "n_cells": counts.astype(int),
                        "assign": assign, "top2_z": top2, "max_z": score_z.max(axis=1)})
    tab.index.name = "sub"
    rows.append(tab)
    print(tab.to_string(), flush=True)

    cell_type[idx] = sub.obs["sub"].map(assign).to_numpy()
    obs.iloc[idx, obs.columns.get_loc("sub")] = sub.obs["sub"].astype(str).to_numpy()
    del sub, ind, mean_expr, gz, score_z
    gc.collect()

obs["cell_type"] = pd.Categorical(cell_type)
pd.concat(rows).to_csv(f"{OUT}/subcluster_assignment.csv")
print(pd.crosstab(obs["cell_type"], obs["group"]).to_string(), flush=True)

# write final annotated object (expression restored)
adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
adata.obsm["X_umap"] = np.load(f"{OUT}/umap.npy")
adata.write_h5ad(f"{OUT}/lh7_annotated.h5ad", compression="lzf")
print("STAGE2D_DONE", adata.shape, adata.obs["cell_type"].nunique(), flush=True)
