
#!/usr/bin/env python3
"""Stage 4: per-sample x per-cell-type pseudobulk from the raw 10x matrices.

The compact clustering matrix only carries per-sample HVGs, so genes that are
not HVGs in a given sample would appear as structural zeros. Pseudobulk is
therefore recomputed from the raw counts over all 36,601 genes, using the
final cell-type labels. Output: one npz per sample with summed counts,
detected-cell counts and cell numbers per cell type.
"""
import os, sys, gzip, tarfile, tempfile, shutil, gc, time
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import anndata as ad

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

RAW = os.path.join(_ROOT, "data", "raw")
OUT = os.path.join(_ROOT, "data", "pseudobulk")
H5 = os.path.join(_ROOT, "data", "processed", "lh7_annotated.h5ad")
os.makedirs(OUT, exist_ok=True)

obs = ad.read_h5ad(H5, backed="r").obs[["sample", "cell_type"]]
print("obs", obs.shape, flush=True)


def extract(tar_path):
    tmp = tempfile.mkdtemp(dir="/tmp")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(tmp)
    for root, _, files in os.walk(tmp):
        if ("matrix.mtx.gz" in files) or ("matrix.mtx" in files):
            return tmp, root
    raise RuntimeError(f"no matrix in {tar_path}")


def read_10x(d):
    mtx = os.path.join(d, "matrix.mtx.gz")
    if not os.path.exists(mtx):
        mtx = os.path.join(d, "matrix.mtx")
    M = scipy.io.mmread(mtx).T.tocsr().astype(np.float32)
    feat = pd.read_csv(os.path.join(d, "features.tsv.gz" if
                       os.path.exists(os.path.join(d, "features.tsv.gz")) else "genes.tsv.gz"),
                       sep="\t", header=None)
    bc = pd.read_csv(os.path.join(d, "barcodes.tsv.gz"), sep="\t", header=None)[0].astype(str).values
    genes = feat[1].astype(str).values
    _, idx = np.unique(genes, return_index=True)
    idx = np.sort(idx)
    return M[:, idx], bc, genes[idx]


for name in sorted(obs["sample"].unique()):
    out = f"{OUT}/{name}.npz"
    if os.path.exists(out):
        print("skip", name, flush=True)
        continue
    t0 = time.time()
    tmp, d = extract(f"{RAW}/{name}.tar.gz")
    try:
        M, bc, genes = read_10x(d)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    o = obs[obs["sample"] == name]
    bcodes = pd.Index([i.split("-", 1)[1] if i.startswith(name + "-") else i for i in o.index],
                      name="bc")
    pos = pd.Series(np.arange(len(bc)), index=bc)
    rows = pos.reindex(bcodes).to_numpy()
    if np.isnan(rows).any():
        raise RuntimeError(f"{name}: barcodes missing")
    rows = rows.astype(np.int64)

    labels = o["cell_type"].astype(str).values
    types = [t for t in pd.unique(labels) if t != "Unassigned"]
    sums, dets, ns = [], [], []
    for t in types:
        sel = rows[labels == t]
        sub = M[sel]
        sums.append(np.asarray(sub.sum(axis=0)).ravel().astype(np.float32))
        dets.append(np.asarray((sub > 0).sum(axis=0)).ravel().astype(np.int32))
        ns.append(int(len(sel)))
        del sub
    np.savez_compressed(out, types=np.array(types), genes=genes,
                        sums=np.vstack(sums), detected=np.vstack(dets),
                        n_cells=np.array(ns))
    print(name, M.shape, len(types), "types", round(time.time() - t0, 1), "s", flush=True)
    del M
    gc.collect()

print("STAGE4_DONE", flush=True)
