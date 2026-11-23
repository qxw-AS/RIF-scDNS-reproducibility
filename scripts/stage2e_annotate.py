
#!/usr/bin/env python3
"""Stage 2e: final cell-type labels from the cached subcluster marker means.

Rules use absolute mean expression of canonical markers (log1p CPM/1e4), so
they are auditable and do not depend on z-score tie-breaking. Subclusters
without any lineage marker are left as "Unassigned" (reported, not plotted).
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

OUT = os.path.join(_ROOT, "data", "processed")


def lab_epithelial(m):
    if m["FOXJ1"] >= 0.3 or (m["TPPP3"] >= 1.5 and m["RSPH1"] >= 0.8):
        return "Ciliated"
    if m["PAEP"] >= 0.5 or m["DKK1"] >= 0.5 or m["GPX3"] >= 0.3:
        return "Glandular"
    return "Epithelial"


def lab_stromal(m):
    return "Stromal"


def lab_endothelial(m):
    if m["CCL21"] >= 0.6 and m["PROX1"] >= 0.25:
        return "Lymphatic EC"
    if m["VWF"] >= 0.5:
        return "Endothelial"
    return "Unassigned"


def lab_immune(m):
    if m["TPSAB1"] >= 0.5 or m["CPA3"] >= 0.5:
        return "Mast"
    if m["JCHAIN"] >= 0.5 or m["CD79A"] >= 0.5 or m["MS4A1"] >= 0.5:
        return "B/Plasma"
    if m["CD68"] >= 0.4:
        return "Myeloid"
    if m["CD3D"] >= 0.5:
        return "T cell"
    if m["NKG7"] >= 1.0 or (m["IL7R"] >= 1.0 and m["KIT"] >= 0.5):
        return "NK cell"           # includes KIT+ IL7R+ CD3- ILC
    return "Unassigned"


RULES = {"Epithelial": lab_epithelial, "Stromal": lab_stromal,
         "Endothelial": lab_endothelial, "Immune": lab_immune}

adata = sc.read_h5ad(f"{OUT}/lh7_annotated.h5ad")
obs = adata.obs
obs["cell_type"] = pd.NA
for lin, rule in RULES.items():
    s = pd.read_csv(f"{OUT}/sub_{lin}.csv.gz").set_index("cell")["sub"].astype(str)
    m = pd.read_csv(f"{OUT}/means_{lin}.csv", index_col=0)
    m.index = m.index.astype(str)
    lab = {sub: rule(m.loc[sub]) for sub in m.index}
    mask = obs.index.isin(s.index)
    obs.loc[mask, "cell_type"] = s.reindex(obs.index[mask]).map(lab).to_numpy()
    print(f"[{lin}] labelled {int(mask.sum())} cells", flush=True)

obs["cell_type"] = pd.Categorical(obs["cell_type"])
adata.obs = obs
print(obs["cell_type"].value_counts().to_string(), flush=True)
print(pd.crosstab(obs["cell_type"], obs["group"]).to_string(), flush=True)
adata.write_h5ad(f"{OUT}/lh7_annotated.h5ad", compression="lzf")
print("STAGE2E_DONE", adata.shape, flush=True)
