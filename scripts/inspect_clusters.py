
#!/usr/bin/env python3
"""Diagnostic: per-Leiden-cluster mean expression of canonical endometrial markers."""
import os
import numpy as np
import pandas as pd
import scanpy as sc

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

IN = os.path.join(_ROOT, "data", "processed", "lh7_annotated.h5ad")
adata = sc.read_h5ad(IN)
print("loaded", adata.shape, flush=True)

MARK = ["EPCAM", "KRT8", "KRT18", "KRT19", "PAEP", "SPP1", "GPX3", "DKK1",
        "FOXJ1", "TPPP3", "PIFO", "CAPS", "RSPH1",
        "DCN", "LUM", "COL1A1", "COL1A2", "PGR", "HAND2",
        "ACTA2", "MYH11", "DES", "CNN1",
        "PECAM1", "VWF", "PLVAP", "KDR", "ADGRL4", "A2M",
        "PROX1", "LYVE1", "CCL21", "MMRN2",
        "CD3D", "CD3E", "IL7R", "CD2", "CD247",
        "NKG7", "GNLY", "KLRD1", "NCAM1", "KLRB1",
        "CD79A", "MS4A1", "JCHAIN", "MZB1", "IGHG1",
        "CD68", "CD14", "C1QA", "MRC1", "LYZ", "CD163",
        "CD1C", "LAMP3", "HLA-DRA", "CLEC9A", "CD207",
        "TPSAB1", "CPA3", "KIT", "MS4A2",
        "MKI67", "TOP2A", "PTPRC"]
MARK = [g for g in MARK if g in adata.var_names]
print("markers present", len(MARK), flush=True)

rows = []
for cl in adata.obs["leiden"].cat.categories:
    sub = adata[adata.obs["leiden"] == cl, MARK]
    m = np.asarray(sub.X.mean(axis=0)).ravel()
    rows.append(pd.Series(m, index=MARK, name=cl))
tab = pd.DataFrame(rows)
tab["n_cells"] = adata.obs["leiden"].value_counts().reindex(tab.index).values
tab.to_csv(os.path.join(_ROOT, "data", "processed", "cluster_marker_means.csv"))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 100)
print(tab.round(1).iloc[:, :].to_string(), flush=True)
print("INSPECT_DONE", flush=True)
