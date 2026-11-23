
#!/usr/bin/env python3
"""Stage 1 (memory-safe): read GSE250130 LH+7 10x matrices with scipy, QC,
doublet removal (scrublet, Python equivalent of DoubletFinder), then store a
compact log-normalised matrix restricted to HVG + marker genes.

Container memory limit is 4 GiB, so the full 36,601-gene matrix (46M nnz/sample)
is never concatenated across samples.
"""
import os, sys, gzip, glob, tarfile, tempfile, shutil, json, time, gc
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import anndata as ad
import scanpy as sc

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

RAW = os.path.join(_ROOT, "data", "raw")
OUT = os.path.join(_ROOT, "data", "qc")
os.makedirs(OUT, exist_ok=True)
sc.settings.seed = 20261117

SAMPLE_GROUP = {
    "RIF_1": "RIF", "RIF_2": "RIF", "RIF_3": "RIF", "RIF_4": "RIF", "RIF_5": "RIF",
    "RIF_6": "RIF", "RIF_7": "RIF", "RIF_8": "RIF", "RIF_9": "RIF", "RIF_10": "RIF",
    "LH7_PGT_1": "Fertile", "LH7_PGT_2": "Fertile", "LH7_PGT_3": "Fertile",
    "LH7_1": "Fertile", "LH7_2": "Fertile", "LH7_3": "Fertile",
}

MARKER_GENES = [
    "EPCAM", "KRT8", "KRT18", "KRT19", "ELF3", "PAEP", "SPP1", "GPX3", "DKK1",
    "FOXJ1", "TPPP3", "PIFO", "CAPS", "RSPH1", "DCN", "LUM", "COL1A1", "COL1A2",
    "PGR", "HAND2", "ACTA2", "MYH11", "DES", "CNN1", "PECAM1", "VWF", "PLVAP",
    "KDR", "PROX1", "LYVE1", "CCL21", "MMRN2", "CD3D", "CD3E", "IL7R", "CD2",
    "NKG7", "GNLY", "KLRD1", "NCAM1", "CD79A", "MS4A1", "JCHAIN", "MZB1", "CD68",
    "CD14", "C1QA", "MRC1", "LYZ", "CD1C", "LAMP3", "HLA-DRA", "CLEC9A",
    "TPSAB1", "CPA3", "KIT", "MS4A2", "PTPRC", "MKI67", "TOP2A", "INSR", "SOX9",
]


def extract(tar_path):
    tmp = tempfile.mkdtemp(dir="/tmp")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(tmp)
    d = None
    for root, _, files in os.walk(tmp):
        if ("matrix.mtx.gz" in files) or ("matrix.mtx" in files):
            d = root
            break
    if d is None:
        raise RuntimeError(f"no 10x matrix in {tar_path}")
    return tmp, d


def read_10x(d):
    mtx = os.path.join(d, "matrix.mtx.gz")
    if not os.path.exists(mtx):
        mtx = os.path.join(d, "matrix.mtx")
    M = scipy.io.mmread(mtx).T.tocsr().astype(np.float32)  # cells x genes
    feat = pd.read_csv(os.path.join(d, "features.tsv.gz" if
                       os.path.exists(os.path.join(d, "features.tsv.gz")) else "genes.tsv.gz"),
                       sep="\t", header=None)
    bc = pd.read_csv(os.path.join(d, "barcodes.tsv.gz"), sep="\t", header=None)[0].values
    genes = feat[1].astype(str).values
    # gene symbols may repeat; keep first occurrence
    _, idx = np.unique(genes, return_index=True)
    idx = np.sort(idx)
    M = M[:, idx]
    genes = genes[idx]
    return M, bc, genes


def main():
    summary = {}
    only = set(sys.argv[1:])
    for name, grp in SAMPLE_GROUP.items():
        if only and name not in only:
            continue
        out = f"{OUT}/{name}.h5ad"
        if os.path.exists(out):
            a = sc.read_h5ad(out)
            summary[name] = {"group": grp, "cells_qc": int(a.n_obs), "genes": int(a.n_vars),
                             "cells_raw": int(a.uns.get("cells_raw", 0))}
            print("skip", name, summary[name], flush=True)
            continue
        t0 = time.time()
        tmp, d = extract(f"{RAW}/{name}.tar.gz")
        try:
            M, bc, genes = read_10x(d)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        raw_n = M.shape[0]
        print(f"[{name}] raw {M.shape} nnz={M.nnz}", flush=True)

        # QC: nFeature 200-6000, MT < 20%
        n_genes = np.asarray((M > 0).sum(axis=1)).ravel()
        total = np.asarray(M.sum(axis=1)).ravel()
        mt_idx = np.where(np.char.startswith(genes.astype(str), "MT-"))[0]
        mt_pct = np.asarray(M[:, mt_idx].sum(axis=1)).ravel() / np.maximum(total, 1) * 100
        keep = (n_genes >= 200) & (n_genes <= 6000) & (mt_pct < 20)
        M = M[keep].tocsr()
        bc = bc[keep]
        print(f"[{name}] qc {M.shape}", flush=True)

        # normalise (single copy, in place)
        total = np.asarray(M.sum(axis=1)).ravel()
        M = sp.diags((1e4 / np.maximum(total, 1)).astype(np.float32)) @ M
        M = M.tocsr().astype(np.float32)
        M.data = np.log1p(M.data)

        adata = ad.AnnData(X=M, obs=pd.DataFrame(index=bc),
                           var=pd.DataFrame(index=pd.Index(genes, name="gene")))
        # restrict HVG candidates to genes detected in >=1% of cells so that the
        # compact matrix stays informative (dispersion-only HVGs are too sparse)
        det = np.asarray((M > 0).sum(axis=0)).ravel()
        cand = det >= max(20, int(0.01 * M.shape[0]))
        adata_c = adata[:, cand].copy()
        sc.pp.highly_variable_genes(adata_c, n_top_genes=3000, flavor="seurat")
        hvg = adata_c.var_names[adata_c.var["highly_variable"]].tolist()
        del adata_c
        gc.collect()
        print(f"[{name}] hvg {len(hvg)} (from {int(cand.sum())} candidates)", flush=True)

        # slice what we need BEFORE releasing the full-gene matrix (4 GiB container)
        gi = {g: i for i, g in enumerate(genes)}
        keep_genes = sorted(set(hvg) | (set(MARKER_GENES) & set(genes.tolist())))
        sel = np.array([gi[g] for g in keep_genes])
        Xs = M[:, sel].tocsr()
        raw_counts = M[:, [gi[g] for g in hvg]].tocsr()
        n_cells = M.shape[0]
        del M, adata
        gc.collect()

        # doublet detection on raw counts restricted to HVG
        tmp_ad = ad.AnnData(X=raw_counts, obs=pd.DataFrame(index=bc))
        try:
            sc.pp.scrublet(tmp_ad, expected_doublet_rate=0.06, random_state=20261117)
            dbl = tmp_ad.obs["predicted_doublet"].to_numpy().astype(bool)
            scores = tmp_ad.obs["doublet_score"].to_numpy()
            print(f"[{name}] scrublet removed {int(dbl.sum())}", flush=True)
        except Exception as e:
            print("scrublet skipped", name, e, flush=True)
            dbl = np.zeros(n_cells, bool)
            scores = np.zeros(n_cells, np.float32)
        del tmp_ad, raw_counts
        gc.collect()

        adata = ad.AnnData(X=Xs, obs=pd.DataFrame(index=bc),
                           var=pd.DataFrame(index=pd.Index(keep_genes, name="gene")))
        adata.obs["sample"] = name
        adata.obs["group"] = grp
        adata.obs["doublet_score"] = scores
        adata.obs["n_genes"] = n_genes[keep]
        adata.obs["pct_mt"] = mt_pct[keep]
        adata.uns["cells_raw"] = raw_n
        adata.uns["hvg"] = hvg
        adata = adata[~dbl].copy()
        adata.write_h5ad(out, compression="lzf")

        summary[name] = {"group": grp, "cells_raw": int(raw_n),
                         "cells_qc": int(adata.n_obs), "genes": int(adata.n_vars),
                         "sec": round(time.time() - t0, 1)}
        print(name, summary[name], flush=True)

    json.dump(summary, open(f"{OUT}/qc_summary.json", "w"), indent=1)
    print("STAGE1_DONE", flush=True)


if __name__ == "__main__":
    main()
