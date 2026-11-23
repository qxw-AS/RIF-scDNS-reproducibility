
#!/usr/bin/env python3
"""Stage 8a-extra: Leiden cluster x donor composition matrix for Fig. S1D."""
import os
import anndata as ad
import pandas as pd

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

D = os.path.join(_ROOT, "data")
OUT = f"{D}/figS"
a = ad.read_h5ad(f"{D}/processed/lh7_annotated.h5ad", backed="r")
obs = a.obs[["sample", "group", "leiden"]].copy()
obs["leiden"] = obs.leiden.astype(str)
mat = (obs.groupby(["leiden", "sample"], observed=True).size()
       .rename("n").reset_index())
tot = mat.groupby("leiden")["n"].transform("sum")
mat["frac"] = mat.n / tot
wide = mat.pivot(index="leiden", columns="sample", values="frac").fillna(0.0)
wide = wide.reindex(sorted(wide.index, key=lambda x: int(x)))
smeta = obs.groupby("sample")["group"].first()
wide.to_csv(f"{OUT}/s1_cluster_sample_matrix.csv")
smeta.to_csv(f"{OUT}/s1_sample_group.csv")
print(wide.shape, "STAGE8A_EXTRA_DONE", flush=True)
