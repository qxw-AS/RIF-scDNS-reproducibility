
#!/usr/bin/env python3
"""Stage 2 (memory-safe, staged): merge compact matrices -> PCA -> Harmony ->
neighbours -> UMAP -> Leiden -> marker annotation.  The expression matrix is
parked on disk while the graph/clustering runs, keeping peak RSS < 3 GiB.
"""
import os, json, gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import anndata as ad
import scanpy as sc
import harmonypy

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

QC = os.path.join(_ROOT, "data", "qc")
OUT = os.path.join(_ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)
sc.settings.seed = 20261117
sc.settings.n_jobs = 3

SAMPLES = ["RIF_1", "RIF_2", "RIF_3", "RIF_4", "RIF_5", "RIF_6", "RIF_7", "RIF_8",
           "RIF_9", "RIF_10", "LH7_PGT_1", "LH7_PGT_2", "LH7_PGT_3",
           "LH7_1", "LH7_2", "LH7_3"]

MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "ELF3"],
    "Glandular": ["PAEP", "SPP1", "GPX3", "DKK1"],
    "Ciliated": ["FOXJ1", "TPPP3", "PIFO", "CAPS", "RSPH1"],
    "Stromal": ["DCN", "LUM", "COL1A1", "COL1A2", "PGR", "HAND2"],
    "Smooth muscle": ["ACTA2", "MYH11", "DES", "CNN1"],
    "Endothelial": ["PECAM1", "VWF", "PLVAP", "KDR"],
    "Lymphatic EC": ["PROX1", "LYVE1", "CCL21", "MMRN2"],
    "T cell": ["CD3D", "CD3E", "IL7R", "CD2"],
    "NK cell": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "B/Plasma": ["CD79A", "MS4A1", "JCHAIN", "MZB1"],
    "Myeloid": ["CD68", "CD14", "C1QA", "MRC1", "LYZ"],
    "DC": ["CD1C", "LAMP3", "HLA-DRA", "CLEC9A"],
    "Mast": ["TPSAB1", "CPA3", "KIT", "MS4A2"],
}


def rss(tag):
    import resource
    print(f"[mem] {tag} maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024:.2f} GB",
          flush=True)


def merge():
    var_sets, n_obs, nnz = {}, {}, {}
    for s in SAMPLES:
        with h5py.File(f"{QC}/{s}.h5ad", "r") as h:
            vk = h["var"].attrs.get("_index", "gene")
            if isinstance(vk, bytes):
                vk = vk.decode()
            var_sets[s] = np.array([x.decode() if isinstance(x, bytes) else x
                                    for x in h["var"][vk][:]])
            nnz[s] = int(h["X"]["data"].shape[0])
            n_obs[s] = int(h["X"]["indptr"].shape[0] - 1)
    union = sorted(set().union(*var_sets.values()))
    gidx = {g: i for i, g in enumerate(union)}
    total_cells, total_nnz = sum(n_obs.values()), sum(nnz.values())
    print(f"union genes={len(union)} cells={total_cells} nnz={total_nnz/1e6:.1f}M", flush=True)

    data = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int32)
    indptr = np.zeros(total_cells + 1, dtype=np.int64)
    obs_frames, row0, pos = [], 0, 0
    for s in SAMPLES:
        a = sc.read_h5ad(f"{QC}/{s}.h5ad")
        X = a.X.tocsr()
        cols = np.array([gidx[g] for g in a.var_names])
        counts = np.diff(X.indptr)
        indptr[row0 + 1:row0 + 1 + a.n_obs] = pos + np.cumsum(counts)
        data[pos:pos + X.nnz] = X.data
        indices[pos:pos + X.nnz] = cols[X.indices]
        pos += X.nnz
        o = a.obs.copy()
        o.index = [f"{s}-{bc}" for bc in o.index]
        obs_frames.append(o)
        row0 += a.n_obs
        print(s, a.shape, X.nnz, flush=True)
        del a, X
        gc.collect()
    assert pos == total_nnz
    obs = pd.concat(obs_frames)
    obs["sample"] = obs["sample"].astype("category")
    obs["group"] = obs["group"].astype("category")
    adata = ad.AnnData(X=sp.csr_matrix((data, indices, indptr),
                                       shape=(total_cells, len(union))),
                       obs=obs, var=pd.DataFrame(index=pd.Index(union, name="gene")))
    rss("after merge")
    return adata


def load_parked():
    obs = pd.read_pickle(f"{OUT}/obs.pkl")
    genes = pd.read_csv(f"{OUT}/genes.csv", header=None)[0].astype(str).tolist()
    return ad.AnnData(obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))


def main():
    if os.path.exists(f"{OUT}/X_pca_harmony.npy") and os.path.exists(f"{OUT}/obs.pkl"):
        adata = load_parked()
        adata.obsm["X_pca_harmony"] = np.load(f"{OUT}/X_pca_harmony.npy")
        print("resume: harmony embedding found", flush=True)
    elif os.path.exists(f"{OUT}/X_pca.npy") and os.path.exists(f"{OUT}/obs.pkl"):
        adata = load_parked()
        adata.obsm["X_pca"] = np.load(f"{OUT}/X_pca.npy")
        print("resume: pca embedding found", flush=True)
        ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, ["sample"],
                                   max_iter_harmony=8, random_state=20261117)
        adata.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr).T.astype(np.float32)
        del ho
        gc.collect()
        rss("after harmony")
        np.save(f"{OUT}/X_pca_harmony.npy", adata.obsm["X_pca_harmony"])
    else:
        adata = merge()

        sc.pp.pca(adata, n_comps=40, zero_center=False, svd_solver="arpack",
                  random_state=20261117)
        rss("after pca")

        # park the expression matrix on disk; clustering needs only the embedding
        np.savez(f"{OUT}/X_compact.npz", data=adata.X.data, indices=adata.X.indices,
                 indptr=adata.X.indptr, shape=np.array(adata.X.shape))
        adata.obs.to_pickle(f"{OUT}/obs.pkl")
        pd.Series(adata.var_names).to_csv(f"{OUT}/genes.csv", index=False, header=False)
        np.save(f"{OUT}/X_pca.npy", adata.obsm["X_pca"])
        print("matrix parked", flush=True)
        adata.X = None
        gc.collect()
        rss("after park")

        ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, ["sample"],
                                   max_iter_harmony=8, random_state=20261117)
        adata.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr).T.astype(np.float32)
        del ho
        gc.collect()
        rss("after harmony")
        np.save(f"{OUT}/X_pca_harmony.npy", adata.obsm["X_pca_harmony"])

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony")
    rss("after neighbors")
    sc.tl.umap(adata, min_dist=0.3)
    rss("after umap")
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden", flavor="igraph",
                 n_iterations=2, directed=False)
    print("leiden clusters", adata.obs.leiden.nunique(), flush=True)

    adata.obs[["leiden"]].to_pickle(f"{OUT}/leiden.pkl")
    np.save(f"{OUT}/umap.npy", adata.obsm["X_umap"])

    # ---------------- reload expression for annotation ----------------
    z = np.load(f"{OUT}/X_compact.npz")
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    adata2 = ad.AnnData(X=X, obs=adata.obs.copy(),
                        var=pd.DataFrame(index=pd.Index(pd.read_csv(
                            f"{OUT}/genes.csv", header=None)[0].astype(str), name="gene")))
    adata2.obsm["X_umap"] = adata.obsm["X_umap"]
    del adata, z
    gc.collect()
    rss("after reload")

    for ct, genes in MARKERS.items():
        g = [x for x in genes if x in adata2.var_names]
        sc.tl.score_genes(adata2, g, score_name=f"score_{ct}", use_raw=False)
    score_cols = [f"score_{ct}" for ct in MARKERS]
    cl_mean = adata2.obs.groupby("leiden", observed=True)[score_cols].mean()
    assign = cl_mean.idxmax(axis=1).str.replace("score_", "", regex=False)
    adata2.obs["cell_type"] = adata2.obs["leiden"].map(assign).astype("category")
    print(assign.to_string(), flush=True)

    sc.tl.rank_genes_groups(adata2, "leiden", method="wilcoxon", n_genes=8)
    tab = sc.get.rank_genes_groups_df(adata2, None)
    tab.groupby("group", observed=True).head(8).to_csv(f"{OUT}/leiden_top_markers.csv",
                                                       index=False)
    ct_counts = pd.crosstab(adata2.obs["cell_type"], adata2.obs["group"])
    ct_counts.to_csv(f"{OUT}/celltype_group_counts.csv")
    print(ct_counts.to_string(), flush=True)

    adata2.write_h5ad(f"{OUT}/lh7_annotated.h5ad", compression="lzf")
    print("STAGE2_DONE", adata2.shape, flush=True)


if __name__ == "__main__":
    main()
