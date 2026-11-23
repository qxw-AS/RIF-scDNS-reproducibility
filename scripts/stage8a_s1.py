
#!/usr/bin/env python3
"""Fig. S1 data: cohort structure, QC, composition, annotation robustness."""
import json
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.stats import ranksums
from sklearn.metrics import adjusted_rand_score

SEED = 20261117
D = os.path.join(_ROOT, "data")
OUT = f"{D}/figS"
import os

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))
os.makedirs(OUT, exist_ok=True)

a = ad.read_h5ad(f"{D}/processed/lh7_annotated.h5ad", backed="r")
obs_all = a.obs.copy()
a.file.close()
obs = obs_all[obs_all.cell_type != "Unassigned"].copy()

CT_ORDER = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
            "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]
SAMPLE_ORDER = [f"RIF_{i}" for i in range(1, 11)] + ["LH7_1", "LH7_2", "LH7_3",
                                                     "LH7_PGT_1", "LH7_PGT_2", "LH7_PGT_3"]

# ---- donor x cell type counts ----
cnt = (obs.groupby(["sample", "group", "cell_type"], observed=True).size()
       .rename("n").reset_index())
cnt["cell_type"] = pd.Categorical(cnt.cell_type, CT_ORDER, ordered=True)
cnt.to_csv(f"{OUT}/s1_donor_counts.csv", index=False)

# ---- per-sample QC ----
qc = (obs.groupby(["sample", "group"], observed=True)
      .agg(n_cells=("n_genes", "size"), n_genes_median=("n_genes", "median"),
           n_genes_mean=("n_genes", "mean"), n_genes_q25=("n_genes", lambda x: x.quantile(.25)),
           n_genes_q75=("n_genes", lambda x: x.quantile(.75)),
           pct_mt_median=("pct_mt", "median"), pct_mt_mean=("pct_mt", "mean"),
           pct_mt_q25=("pct_mt", lambda x: x.quantile(.25)),
           pct_mt_q75=("pct_mt", lambda x: x.quantile(.75)))
      .reset_index())
qc.to_csv(f"{OUT}/s1_qc_by_sample.csv", index=False)

# ---- donor-level composition + Wilcoxon ----
comp = obs.groupby(["sample", "group", "cell_type"], observed=True).size().rename("n").reset_index()
tot = comp.groupby("sample", observed=True).n.sum().rename("total")
comp = comp.join(tot, on="sample")
comp["frac"] = comp.n / comp.total
comp.to_csv(f"{OUT}/s1_composition_by_donor.csv", index=False)

rows = []
for ct in CT_ORDER:
    x = comp[(comp.cell_type == ct) & (comp.group == "RIF")].frac.to_numpy()
    y = comp[(comp.cell_type == ct) & (comp.group == "Fertile")].frac.to_numpy()
    p = ranksums(x, y).pvalue if len(x) and len(y) else np.nan
    rows.append({"cell_type": ct, "median_frac_RIF": np.median(x), "median_frac_Fertile": np.median(y),
                 "log2_ratio": np.log2((np.median(x) + 1e-4) / (np.median(y) + 1e-4)),
                 "p_wilcoxon": p, "n_RIF_donors": len(x), "n_Fer_donors": len(y)})
stat = pd.DataFrame(rows)
# Benjamini-Hochberg
p = stat.p_wilcoxon.to_numpy()
order = np.argsort(p)
q = np.empty_like(p)
m = len(p)
prev = 1.0
for rank, idx in enumerate(order[::-1], start=1):
    val = p[idx] * m / (m - rank + 1)
    prev = min(prev, val)
    q[idx] = prev
stat["padj_BH"] = q
stat.to_csv(f"{OUT}/s1_composition_stats.csv", index=False)

# ---- Leiden cluster sample fraction and annotation confidence ----
cl = pd.read_csv(f"{D}/processed/cluster_assignment.csv", index_col=0)
cl.index = cl.index.astype(str)
# parse top2 z string
def parse_margin(s):
    try:
        vals = [float(x.split(":")[1]) for x in str(s).split(",")]
        return vals[0] - max(vals[1:]) if len(vals) > 1 else vals[0]
    except Exception:
        return np.nan

cl["margin"] = cl["top2_z"].map(parse_margin)
cl = cl.rename(columns={"assign": "assigned_type"})
obs["leiden"] = obs.leiden.astype(str)
lf = obs.groupby("leiden", observed=True)["sample"].value_counts(normalize=True).rename("frac").reset_index()
mx = lf.sort_values("frac").groupby("leiden", observed=True).tail(1).set_index("leiden")
mx = mx.rename(columns={"sample": "top_sample", "frac": "max_sample_frac"})
cl = cl.join(mx[["top_sample", "max_sample_frac"]], how="left")
cl["n_cells_obs"] = obs.groupby("leiden", observed=True).size()
cl.to_csv(f"{OUT}/s1_cluster_qc.csv")

# ---- subsampling ARI across resolutions ----
X = np.load(f"{D}/processed/X_pca_harmony.npy")
assert X.shape[0] == len(obs_all), (X.shape, len(obs_all))
ad_full = ad.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32), obs=obs_all[[]].copy())
ad_full.obsm["X_pca_harmony"] = X.astype(np.float32)
sc.pp.neighbors(ad_full, use_rep="X_pca_harmony", n_neighbors=15, random_state=SEED)
res_list = [1.0, 1.5, 2.0, 2.5]
full_lab = {}
for res in res_list:
    sc.tl.leiden(ad_full, resolution=res, key_added=f"l{res}", flavor="igraph",
                 n_iterations=2, directed=False, random_state=SEED)
    full_lab[res] = ad_full.obs[f"l{res}"].astype(str).to_numpy()

rng = np.random.default_rng(SEED)
ari_rows, nclust_rows = [], []
for rep in range(3):
    idx = rng.choice(len(obs_all), size=int(0.8 * len(obs_all)), replace=False)
    idx.sort()
    sub = ad_full[idx].copy()
    sc.pp.neighbors(sub, use_rep="X_pca_harmony", n_neighbors=15, random_state=SEED + rep)
    for res in res_list:
        sc.tl.leiden(sub, resolution=res, key_added=f"l{res}", flavor="igraph",
                     n_iterations=2, directed=False, random_state=SEED + rep)
        lab = sub.obs[f"l{res}"].astype(str).to_numpy()
        ari = adjusted_rand_score(full_lab[res][idx], lab)
        ari_rows.append({"repeat": rep + 1, "resolution": res, "ari": ari,
                         "n_cells": len(idx), "n_clusters_sub": len(np.unique(lab)),
                         "n_clusters_full": len(np.unique(full_lab[res]))})
for res in res_list:
    nclust_rows.append({"resolution": res, "n_clusters_full": len(np.unique(full_lab[res]))})
pd.DataFrame(ari_rows).to_csv(f"{OUT}/s1_subsample_ari.csv", index=False)
pd.DataFrame(nclust_rows).to_csv(f"{OUT}/s1_resolution_clusters.csv", index=False)

json.dump({"n_cells": int(len(obs)), "n_samples": int(obs["sample"].nunique()),
           "n_clusters_leiden": int(obs.leiden.nunique()),
           "n_clusters_single_sample_gt50": int((cl.max_sample_frac > 0.5).sum()),
           "ari_median": float(np.median([r["ari"] for r in ari_rows])),
           "ari_min": float(np.min([r["ari"] for r in ari_rows]))},
          open(f"{OUT}/s1_summary.json", "w"), indent=2)
print(pd.DataFrame(ari_rows).to_string(index=False))
print(stat.to_string(index=False))
print("S1_DONE")
