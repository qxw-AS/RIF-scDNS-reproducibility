
#!/usr/bin/env python3
"""Direct network-level replication in GSE248784 (MAGIC/Pearson/Fisher-z, same as discovery)."""
import os
import json
import time

import anndata as ad
import magic
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

BASE = _ROOT
PROC = f"{BASE}/data/external/gse248784/processed"
FIG2 = f"{BASE}/data/fig2"
SEED = 20261117
TOP_K = 50
MIN_DET = 0.05
MAX_CELLS = 4000
MODS = {f"M{k}": v for k, v in json.load(open(f"{BASE}/data/external/gse248784/gene_sets.json"))["modules"].items()}

adata = ad.read_h5ad(f"{PROC}/external_annotated.h5ad")
print("loaded", adata.shape, flush=True)

dz_disc = pd.read_csv(f"{FIG2}/gene_z_by_type.csv", index_col=0)
comb = pd.read_csv(f"{FIG2}/combined_z.csv", index_col=0)
rows, ext_z = [], {}

for ct in adata.obs["cell_type"].cat.categories:
    sub = adata[adata.obs["cell_type"] == ct]
    n_by = sub.obs.groupby("group", observed=True).size().to_dict()
    d_by = sub.obs.groupby("group", observed=True)["donor"].nunique().to_dict()
    if ct not in dz_disc.columns or min(n_by.values()) < 200 or min(d_by.values()) < 3:
        print(f"skip {ct}: cells={n_by}, donors={d_by}", flush=True)
        continue
    t0 = time.time()
    mats, genes = {}, None
    testable = None
    for g in ["RIF", "Normal"]:
        s = sub[sub.obs["group"] == g]
        X = s.X
        X = np.asarray(X.todense(), dtype=np.float64) if hasattr(X, "todense") else np.asarray(X, dtype=np.float64)
        if X.shape[0] > MAX_CELLS:
            idx = np.random.default_rng(SEED).choice(X.shape[0], MAX_CELLS, replace=False)
            X = X[idx]
        genes = np.asarray(sub.var_names, dtype=str)
        det = (X > 0).mean(0)
        testable = det >= MIN_DET if testable is None else (testable & (det >= MIN_DET))
        df = pd.DataFrame(X, columns=genes)
        n_pca = int(min(50, max(10, X.shape[0] // 10)))
        imp = np.asarray(magic.MAGIC(knn=5, n_pca=n_pca, t=3, random_state=SEED,
                                     verbose=False).fit_transform(df), dtype=np.float64)
        sd = imp.std(0); sd[sd < 1e-8] = np.inf
        Z = (imp - imp.mean(0)) / sd
        C = np.clip((Z.T @ Z) / max(imp.shape[0] - 1, 1), -0.999, 0.999)
        mats[g] = np.arctanh(C)
    D = mats["RIF"] - mats["Normal"]
    np.fill_diagonal(D, 0)
    A = np.abs(D)
    k = min(TOP_K, A.shape[1] - 1)
    s = np.zeros(len(genes))
    for i in range(len(genes)):
        p = np.argpartition(A[i], -k)[-k:]
        w = np.abs(D[i, p]); sw = w.sum()
        s[i] = np.sum(w * D[i, p]) / sw if sw > 1e-12 else 0.0
    s[~testable] = np.nan
    ok = np.isfinite(s)
    zz = np.full(len(genes), np.nan)
    if ok.sum() > 10:
        zz[ok] = (s[ok] - s[ok].mean()) / (s[ok].std() if s[ok].std() > 1e-12 else 1.0)
    ext_z[ct] = pd.Series(zz, index=genes)
    common = pd.Index(genes)[ok].intersection(dz_disc.index)
    dvals = dz_disc.loc[common, ct].astype(float)
    evals = pd.Series(zz, index=genes).loc[common]
    both = dvals.notna() & evals.notna()
    rho, p = spearmanr(dvals[both], evals[both])
    # module-level direction
    mod_d, mod_e = [], []
    for m, gs in MODS.items():
        gg = [x for x in gs if x in common]
        if len(gg) < 5:
            continue
        mod_d.append(float(dz_disc.loc[gg, ct].mean()))
        mod_e.append(float(pd.Series(zz, index=genes).loc[gg].mean()))
    mrho = spearmanr(mod_d, mod_e) if len(mod_d) >= 4 else (np.nan, np.nan)
    m2 = [x for x in MODS["M2"] if x in common]
    sign_m2 = float((np.sign(dz_disc.loc[m2, ct]) == np.sign(pd.Series(zz, index=genes).loc[m2])).mean())
    rows.append({"cell_type": ct, "n_rif_cells": n_by["RIF"], "n_normal_cells": n_by["Normal"],
                 "n_donors_RIF": d_by["RIF"], "n_donors_Normal": d_by["Normal"],
                 "n_testable": int(ok.sum()), "n_common": len(common),
                 "spearman_gene_z": rho, "p_gene_z": p,
                 "spearman_module_z": mrho[0], "p_module_z": mrho[1],
                 "m2_sign_concordance": sign_m2})
    print(f"{ct}: cells={n_by}, rho_gene={rho:+.3f} p={p:.3g}, rho_module={mrho[0]:+.2f}, "
          f"M2 sign={sign_m2:.2f}, {time.time()-t0:.0f}s", flush=True)

out = pd.DataFrame(rows)
out.to_csv(f"{BASE}/data/external/gse248784/external_network_replication.csv", index=False)
pd.DataFrame(ext_z).to_csv(f"{BASE}/data/external/gse248784/external_gene_z.csv")
tot = int(out.n_common.sum())
pool = pd.concat([pd.Series({c: out.set_index("cell_type").loc[c, "spearman_gene_z"]}) for c in out.cell_type])
print("\n=== summary ===")
print(out.round(3).to_string(index=False))
print("types tested:", len(out), "| median rho:", round(float(out.spearman_gene_z.median()), 3),
      "| positive types:", int((out.spearman_gene_z > 0).sum()), "/", len(out))
