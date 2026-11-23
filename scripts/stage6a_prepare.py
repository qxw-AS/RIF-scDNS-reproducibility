
#!/usr/bin/env python3
"""Stage 6a: stratified cell sampling + shared gene panel for Fig.2.

Design (aligned with the pre-registered plan, R2):
  * within each cell type, sample the same number of RIF and fertile cells
    (<=1500 per group) to avoid precision-driven divergence differences;
  * sampling is stratified by donor sample;
  * a common gene panel (~2000 genes) is required for cross-cell-type
    Stouffer combination of gene-level divergence scores.

Output: data/fig2/{cell_type}.npz  (sampled log-normalised expression)
        data/fig2/gene_panel.csv, sampling_summary.csv, gene_stats.csv
"""
import os
import json, os
import numpy as np
import pandas as pd
import anndata as ad

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

SEED = 20261117
CAP = 1500
H5 = os.path.join(_ROOT, "data", "processed", "lh7_annotated.h5ad")
OUT = os.path.join(_ROOT, "data", "fig2")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

meta = json.load(open(os.path.join(_ROOT, "data", "fig1", "meta.json")))
TYPES = [t for t in meta["cell_types"] if t != "Unassigned"]
GROUPS = ["RIF", "Fertile"]

adata = ad.read_h5ad(H5, backed="r")
obs = adata.obs.copy()
genes_all = adata.var_names.to_numpy().astype(str)


def stratified_sample(sub, n_target):
    """Allocate n_target cells across donors proportionally, >=1 per donor."""
    counts = sub["sample"].value_counts()
    donors = counts.index.to_numpy()
    cnt = counts.to_numpy()
    total = cnt.sum()
    if n_target >= total:
        return np.concatenate([np.where(sub["sample"].to_numpy() == d)[0] for d in donors])
    props = cnt / total * n_target
    alloc = np.floor(props).astype(int)
    alloc = np.maximum(alloc, 1)
    alloc = np.minimum(alloc, cnt)
    while alloc.sum() > n_target:
        j = np.argmax(np.where(alloc > 1, alloc, -1))
        if alloc[j] <= 1:
            break
        alloc[j] -= 1
    frac = props - np.floor(props)
    while alloc.sum() < n_target:
        cand = np.where(alloc < cnt)[0]
        if len(cand) == 0:
            break
        j = cand[np.argmax(frac[cand])]
        alloc[j] += 1
        frac[j] = -1
    picks = []
    for d, k in zip(donors, alloc):
        idx = np.where(sub["sample"].to_numpy() == d)[0]
        picks.append(rng.choice(idx, size=int(k), replace=False))
    return np.concatenate(picks)


# ---------- pass 1: sample cells, per-type gene statistics ----------
samples, stats_rows, summary_rows = {}, [], []
for ct in TYPES:
    sub_all = obs[obs["cell_type"] == ct]
    n_by_group = sub_all["group"].value_counts()
    n_per = int(min(CAP, n_by_group.min()))
    chosen = []
    for g in GROUPS:
        sub = sub_all[sub_all["group"] == g]
        chosen.append(sub.index.to_numpy()[stratified_sample(sub, n_per)])
    chosen = np.concatenate(chosen)
    pos = obs.index.get_indexer(chosen)
    assert (pos >= 0).all()
    X = adata[pos].to_memory().X
    X = np.asarray(X.todense(), dtype=np.float32) if hasattr(X, "todense") else np.asarray(X, dtype=np.float32)
    mean = X.mean(0)
    var = X.var(0)
    det = (X > 0).mean(0)
    stats_rows.append(pd.DataFrame({"cell_type": ct, "gene": genes_all,
                                    "mean": mean, "var": var, "detect": det}))
    samples[ct] = {"X": X,
                   "sample": obs.loc[chosen, "sample"].to_numpy().astype(str),
                   "group": obs.loc[chosen, "group"].to_numpy().astype(str)}
    summary_rows.append({"cell_type": ct, "n_rif": int((samples[ct]["group"] == "RIF").sum()),
                         "n_fertile": int((samples[ct]["group"] == "Fertile").sum()),
                         "n_per_group_target": n_per,
                         "n_donors_rif": int(obs.loc[chosen][samples[ct]["group"] == "RIF"]["sample"].nunique()),
                         "n_donors_fertile": int(obs.loc[chosen][samples[ct]["group"] == "Fertile"]["sample"].nunique())})
    print(f"{ct}: sampled {X.shape}, groups {np.bincount(pd.factorize(samples[ct]['group'])[0])}", flush=True)

st = pd.concat(stats_rows, ignore_index=True)

# ---------- gene panel ----------
# within-type dispersion z-score, then median across cell types. A gene is
# testable in a cell type when detected in >=5% of cells there; the panel
# requires >=2% detection in >=6 of 11 types, so combined_z uses a variable
# number of testable types (Stouffer with k = number of testable types).
st["disp"] = st["var"] / (st["mean"] + 1e-6)
st["zdisp"] = st.groupby("cell_type")["disp"].transform(
    lambda v: (v - v.median()) / (1.4826 * (np.median(np.abs(v - v.median())) + 1e-9)))
score = st.pivot(index="gene", columns="cell_type", values="zdisp").median(axis=1)
det = st.pivot(index="gene", columns="cell_type", values="detect")
meanmax = st.pivot(index="gene", columns="cell_type", values="mean").max(axis=1)

bad_prefix = ("MT-", "RPL", "RPS", "MTRNR")
is_bad = pd.Series(score.index, index=score.index).str.startswith(bad_prefix)
n_detected = (det >= 0.02).sum(axis=1)
ok = (n_detected >= 6) & (meanmax >= 0.10) & (~is_bad)
pool = score[ok].sort_values(ascending=False)
panel = pool.head(2000).index.to_numpy()
print(f"gene panel: {len(panel)} genes (det>=2% in >=6 types: {int(ok.sum())})", flush=True)

pd.DataFrame({"gene": panel,
              "median_disp_z": score.loc[panel].to_numpy(),
              "n_types_detected": n_detected.loc[panel].to_numpy(),
              "min_detect": det.loc[panel].min(axis=1).to_numpy()}).to_csv(f"{OUT}/gene_panel.csv", index=False)
pd.DataFrame(summary_rows).to_csv(f"{OUT}/sampling_summary.csv", index=False)
st.to_csv(f"{OUT}/gene_stats.csv.gz", index=False, compression="gzip")

# ---------- pass 2: save sampled matrices restricted to panel ----------
gidx = pd.Index(genes_all).get_indexer(panel)
for ct in TYPES:
    X = samples[ct]["X"][:, gidx]
    np.savez_compressed(f"{OUT}/{ct.replace(' ', '_').replace('/', '-')}.npz",
                        X=X, genes=panel,
                        sample=samples[ct]["sample"], group=samples[ct]["group"])
    print(f"saved {ct}: {X.shape}", flush=True)

print("STAGE6A_DONE", flush=True)
