
#!/usr/bin/env python3
"""Stage 5: single-cell Wilcoxon DEG (RIF vs fertile) per cell type on REAL
raw-derived expression (no HVG structural zeros).

Pass 1 re-reads each raw 10x matrix, keeps QC-passing barcodes, normalises
(CPM/1e4 + log1p) and stores up to 300 randomly sampled cells per cell type.
Pass 2 stacks the samples per cell type and runs scanpy's Wilcoxon test
(reference = fertile) with BH correction, as specified in the study design.
"""
import os, gc, time
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import tarfile, tempfile, shutil

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

RAW = os.path.join(_ROOT, "data", "raw")
OUT = os.path.join(_ROOT, "data", "scdeg")
H5 = os.path.join(_ROOT, "data", "processed", "lh7_annotated.h5ad")
FIG = os.path.join(_ROOT, "data", "fig1")
os.makedirs(OUT, exist_ok=True)
MAX_CELLS = 300
rng = np.random.default_rng(20261117)
sc.settings.seed = 20261117


def safe(t):
    return t.replace("/", "-")

obs = ad.read_h5ad(H5, backed="r").obs[["sample", "cell_type"]]
samples = sorted(obs["sample"].unique())


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


def pass1():
    for name in samples:
        o = obs[obs["sample"] == name]
        types = [t for t in pd.unique(o["cell_type"].astype(str)) if t != "Unassigned"]
        todo = [t for t in types if not os.path.exists(f"{OUT}/{name}__{safe(t)}.npz")]
        if not todo:
            print("skip", name, flush=True)
            continue
        t0 = time.time()
        tmp, d = extract(f"{RAW}/{name}.tar.gz")
        try:
            M, bc, genes = read_10x(d)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        bcodes = pd.Index([i.split("-", 1)[1] if i.startswith(name + "-") else i
                           for i in o.index])
        pos = pd.Series(np.arange(len(bc)), index=bc)
        rows = pos.reindex(bcodes).to_numpy().astype(np.int64)
        labels = o["cell_type"].astype(str).values
        for t in todo:
            sel = rows[labels == t]
            if len(sel) == 0:
                continue
            if len(sel) > MAX_CELLS:
                sel = rng.choice(sel, MAX_CELLS, replace=False)
            sub = M[sel]
            tot = np.asarray(sub.sum(axis=1)).ravel()
            sub = sp.diags((1e4 / np.maximum(tot, 1)).astype(np.float32)) @ sub
            sub = sub.tocsr()
            sub.data = np.log1p(sub.data)
            np.savez_compressed(f"{OUT}/{name}__{safe(t)}.npz", data=sub.data,
                                indices=sub.indices, indptr=sub.indptr,
                                shape=np.array(sub.shape), genes=genes, n=len(sel))
            del sub
        del M
        gc.collect()
        print(name, "done", round(time.time() - t0, 1), "s", flush=True)
    print("PASS1_DONE", flush=True)


def pass2():
    genes_ref = None
    type_rows, notes = [], []
    types = sorted({t for t in obs["cell_type"].astype(str).unique() if t != "Unassigned"})
    for t in types:
        blocks, cell_samples, nsamp = [], [], []
        for name in samples:
            f = f"{OUT}/{name}__{safe(t)}.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f, allow_pickle=True)
            X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
            g = z["genes"].astype(str)
            if genes_ref is None:
                genes_ref = g
            elif len(g) != len(genes_ref):
                raise RuntimeError("gene length mismatch")
            blocks.append(X)
            cell_samples += [name] * X.shape[0]
            nsamp.append(name)
        if len(blocks) == 0:
            continue
        X = sp.vstack(blocks).tocsr()
        del blocks
        groups = ["RIF" if s.startswith("RIF") else "Fertile" for s in cell_samples]
        sub = ad.AnnData(X=X, obs=pd.DataFrame({"group": pd.Categorical(groups),
                                                "sample": cell_samples},
                                               index=[f"c{i}" for i in range(X.shape[0])]),
                         var=pd.DataFrame(index=pd.Index(genes_ref, name="gene")))
        n_rif = int((sub.obs.group == "RIF").sum())
        n_fer = int((sub.obs.group == "Fertile").sum())
        if n_rif < 50 or n_fer < 50:
            notes.append({"cell_type": t, "n_rif_cells": n_rif, "n_fer_cells": n_fer,
                          "reason": "low cell number: no DEG test"})
            print(t, "low cells", n_rif, n_fer, flush=True)
            continue
        sc.tl.rank_genes_groups(sub, "group", groups=["RIF"], reference="Fertile",
                                method="wilcoxon", tie_correct=True, pts=True)
        df = sc.get.rank_genes_groups_df(sub, "RIF")
        df = df.dropna(subset=["logfoldchanges"])
        fer = (sub.obs.group == "Fertile").to_numpy()
        pct_f = np.asarray((sub.X[fer] > 0).mean(axis=0)).ravel() * 100
        df["pct_fer"] = pd.Series(pct_f, index=sub.var_names).reindex(df["names"]).to_numpy()
        df["pct_rif"] = df["pct_nz_group"] * 100
        df["cell_type"] = t
        df["n_rif_cells"] = n_rif
        df["n_fer_cells"] = n_fer
        type_rows.append(df)
        sig = df.pvals_adj < 0.05
        print(t, "cells", n_rif, n_fer, "sig", int(sig.sum()), flush=True)
        del sub, X
        gc.collect()
    deg = pd.concat(type_rows, ignore_index=True)
    deg.to_csv(f"{FIG}/panelD_deg_full.csv", index=False)

    sel = []
    for t in types:
        d = deg[deg.cell_type == t]
        if d.empty:
            if not any(n["cell_type"] == t for n in notes):
                notes.append({"cell_type": t, "n_rif_cells": 0, "n_fer_cells": 0,
                              "reason": "no significant DEG"})
            continue
        d = d[(d.pvals_adj < 0.05) & (d.logfoldchanges.abs() >= 0.5) &
              (d.logfoldchanges.abs() <= 5.0) &
              ((d.pct_rif >= 15) | (d.pct_fer >= 15))]
        # exclude technical / poorly annotated transcripts that survive BH by
        # zero-inflation rather than biology
        tech = (d.names.str.startswith(("MT-", "RPL", "RPS", "HB", "MTRNR"))
                | d.names.str.match(r"^(AC\d|AL\d|LINC|SNHG|MIR|Z\d{5})")
                | d.names.str.contains(r"-AS\d?$"))
        d = d[~tech]
        if d.empty:
            notes.append({"cell_type": t, "n_rif_cells": int(deg[deg.cell_type == t].n_rif_cells.iloc[0]),
                          "n_fer_cells": int(deg[deg.cell_type == t].n_fer_cells.iloc[0]),
                          "reason": "no significant DEG (BH<0.05, |log2FC|>=0.5)"})
            continue
        up = d[d.logfoldchanges > 0].nlargest(2, "logfoldchanges")
        dn = d[d.logfoldchanges < 0].nsmallest(2, "logfoldchanges")
        sel.append(pd.concat([up, dn]))
    sel = pd.concat(sel, ignore_index=True) if sel else deg.iloc[0:0]
    sel = sel.rename(columns={"names": "gene", "logfoldchanges": "log2FC"})
    sel.to_csv(f"{FIG}/panelD_selected.csv", index=False)
    pd.DataFrame(notes, columns=["cell_type", "n_rif_cells", "n_fer_cells", "reason"]).to_csv(
        f"{FIG}/panelD_notes.csv", index=False)
    print(sel.groupby("cell_type").size().to_string(), flush=True)
    print("notes", notes, flush=True)
    print("PASS2_DONE", flush=True)


if __name__ == "__main__":
    import sys
    if "--pass2-only" in sys.argv:
        pass2()
    else:
        pass1()
        pass2()
