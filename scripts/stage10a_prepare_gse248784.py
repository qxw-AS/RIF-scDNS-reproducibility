
#!/usr/bin/env python3
"""Stage 10a: extract, QC and concatenate GSE248784 (external scRNA-seq).

GSE248784: 21 LH+7? endometrial samples (11 RIF, 10 Normal), 10x/Singleron
matrices. GEO metadata does not report cycle phase; this is recorded as a
limitation. We keep the 2,000-gene discovery panel plus IFN and marker genes
so the concatenated object stays small enough for 5 GB RAM.
"""
import glob
import gzip
import gc
import json
import os
import re
import shutil
import tarfile

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

BASE = os.path.join(_ROOT, "data", "external", "gse248784")
EXT = os.path.join(BASE, "extracted")
PROC = os.path.join(BASE, "processed")
SEED = 20261117
MAX_CELLS = 3500  # cgroup memory limit is 4 GB; cap cells per sample
os.makedirs(EXT, exist_ok=True)
os.makedirs(PROC, exist_ok=True)

# ---------------- gene sets ----------------
GS = json.load(open(os.path.join(BASE, "gene_sets.json")))
MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "CLDN4", "ELF3"],
    "Glandular": ["PAX8", "MUC1", "LGR5", "PROM1", "SOX9", "KRT7"],
    "Ciliated": ["FOXJ1", "PIFO", "CAPS", "RSPH1", "TPPP3", "DNAH5"],
    "Stromal": ["VIM", "DCN", "LUM", "COL1A1", "COL3A1", "PDGFRB", "THY1"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "EGFL7", "AQP1"],
    "Lymphatic EC": ["PROX1", "LYVE1", "PDPN", "CCL21", "MMRN1"],
    "T cell": ["CD3D", "CD3E", "CD2", "TRAC", "IL7R", "CD8A"],
    "NK cell": ["NKG7", "GNLY", "KLRD1", "PRF1", "NCAM1", "KLRF1"],
    "B/Plasma": ["MS4A1", "CD79A", "JCHAIN", "MZB1", "IGHG1", "IGKC"],
    "Myeloid": ["LYZ", "CD68", "CD14", "AIF1", "FCGR3A", "CSF1R"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3", "MS4A2", "KIT"],
    "Smooth muscle": ["ACTA2", "MYH11", "RGS5", "NOTCH3", "TAGLN"],
}
MARKER_GENES = sorted({g for v in MARKERS.values() for g in v})
MODULE_GENES = sorted({g for v in GS["modules"].values() for g in v})
PHASE_GENES = ["PAEP", "IGFBP1", "GPX3", "DKK1",
               "MKI67", "TOP2A", "CCNB1", "BIRC5", "UBE2C"]
KEEP = (set(GS["panel"]) | set(GS["IFN"]) | set(MARKER_GENES) |
        set(MODULE_GENES) | set(PHASE_GENES))
json.dump(MARKERS, open(os.path.join(BASE, "marker_sets.json"), "w"), indent=1)

# ---------------- sample groups ----------------
titles, accs = None, None
with gzip.open(os.path.join(BASE, "series_matrix.txt.gz"), "rt") as fh:
    for line in fh:
        if line.startswith("!Sample_title"):
            titles = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_geo_accession"):
            accs = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
if not titles or not accs:
    raise RuntimeError("could not parse series matrix")
groups = {}
for acc, t in zip(accs, titles):
    donor = t.split(";")[0]
    groups[acc] = {"group": "RIF" if donor.upper().startswith("RIF") else "Normal",
                   "donor": donor}
print("samples:", len(groups), pd.Series([v["group"] for v in groups.values()]).value_counts().to_dict())

# ---------------- extract and organize ----------------
tar = os.path.join(BASE, "GSE248784_RAW.tar")
if not glob.glob(os.path.join(EXT, "GSM*.gz")):
    print("extracting", tar, flush=True)
    with tarfile.open(tar, "r") as tf:
        tf.extractall(EXT)

pat = re.compile(r"^(GSM\d+)_(.+?)_(barcodes|features|matrix)\.(tsv|mtx)\.gz$")
for f in glob.glob(os.path.join(EXT, "**", "*.gz"), recursive=True):
    m = pat.match(os.path.basename(f))
    if not m:
        continue
    gsm, sid, kind, ext = m.groups()
    d = os.path.join(EXT, gsm)
    os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, {"barcodes": "barcodes.tsv.gz",
                            "features": "features.tsv.gz",
                            "matrix": "matrix.mtx.gz"}[kind])
    if os.path.abspath(f) != os.path.abspath(dest):
        shutil.move(f, dest)

# ---------------- load / QC ----------------
adatas, qc = [], []
for gsm in sorted(groups):
    d = os.path.join(EXT, gsm)
    if not os.path.exists(os.path.join(d, "matrix.mtx.gz")):
        print("MISSING", gsm, flush=True)
        continue
    a = sc.read_10x_mtx(d, var_names="gene_symbols", make_unique=True, gex_only=True)
    a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    n0 = a.n_obs
    sc.pp.filter_cells(a, min_genes=300)
    a = a[(a.obs.n_genes_by_counts < 6000) & (a.obs.pct_counts_mt < 20)].copy()
    sc.pp.filter_genes(a, min_cells=3)
    n1 = a.n_obs
    qc.append({"gsm": gsm, "donor": groups[gsm]["donor"], "group": groups[gsm]["group"],
               "cells_raw": n0, "cells_qc": n1,
               "median_genes": float(np.median(a.obs.n_genes_by_counts)),
               "median_umi": float(np.median(a.obs.total_counts)),
               "median_pct_mt": float(np.median(a.obs.pct_counts_mt))})
    a = a[:, [g for g in a.var_names if g in KEEP]].copy()
    if a.n_obs > MAX_CELLS:
        sc.pp.subsample(a, n_obs=MAX_CELLS, random_state=SEED)
    n2 = a.n_obs
    a.obs["gsm"] = gsm
    a.obs["donor"] = groups[gsm]["donor"]
    a.obs["group"] = groups[gsm]["group"]
    a.obs["sample"] = gsm
    a.obs["cells_qc"] = n1
    counts = a.X.astype(np.uint16)
    a.layers["counts"] = counts
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    adatas.append(a)
    print(f"{gsm} {groups[gsm]['donor']:>8} {groups[gsm]['group']:>6} "
          f"{n0}->{n1}->{n2} cells, {a.n_vars} genes", flush=True)
    del a
    gc.collect()

qcdf = pd.DataFrame(qc)
qcdf.to_csv(os.path.join(PROC, "external_sample_QC.csv"), index=False)
print(qcdf.to_string(index=False), flush=True)

adata = ad.concat(adatas, join="inner", label="batch",
                  keys=[str(a.obs["gsm"].iloc[0]) for a in adatas],
                  index_unique="-")
del adatas
gc.collect()
adata.obs["sample"] = adata.obs["sample"].astype("category")
adata.obs["group"] = adata.obs["group"].astype("category")
adata.obs["donor"] = adata.obs["donor"].astype("category")
adata.obs_names_make_unique()
print("concatenated:", adata.shape, "genes:", adata.n_vars, flush=True)
print("cells per group:", adata.obs.groupby("group", observed=True).size().to_dict(), flush=True)
adata.write_h5ad(os.path.join(PROC, "external_qc.h5ad"), compression="gzip")
print("STAGE10A_DONE", flush=True)
