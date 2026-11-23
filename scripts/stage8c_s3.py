
#!/usr/bin/env python3
"""Fig. S3 data: module stability (gene subsampling, donor split, LODO, preservation)."""
import os
import json
import numpy as np
import pandas as pd
import magic
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

SEED = 20261117
D = os.path.join(_ROOT, "data", "fig2")
OUT = os.path.join(_ROOT, "data", "figS")
TOPK = 50
MIN_DET = 0.05
CT = "Stromal"

mods = pd.read_csv(f"{D}/gene_modules.csv")
Z = pd.read_csv(f"{D}/gene_z_by_type.csv", index_col=0)
prof = pd.read_csv(f"{D}/module_profiles.csv", index_col=0)
params = json.load(open(f"{D}/module_params.json"))
cut = params["cut_height"]
mods_list = sorted(mods.module.unique())
refsets = {m: set(g.gene) for m, g in mods.groupby("module")}
Zg = Z.loc[mods.gene].fillna(0.0)

# ---- gene subsampling stability ----
rng = np.random.default_rng(SEED)
rows = []
for it in range(100):
    genes = rng.choice(mods.gene.to_numpy(), size=int(0.8 * len(mods)), replace=False)
    lab = fcluster(linkage(Zg.loc[genes].to_numpy(), method="ward"), t=cut, criterion="distance")
    for m in mods_list:
        s = set(genes[lab == m]) if m in np.unique(lab) else set()
        if not s:
            rows.append({"iter": it, "module": m, "jaccard": 0.0, "n": 0}); continue
        j = max(len(s & rs) / len(s | rs) for rs in refsets.values())
        rows.append({"iter": it, "module": m, "jaccard": float(j), "n": len(s)})
sub = pd.DataFrame(rows)
sub.to_csv(f"{OUT}/s3_gene_subsample.csv", index=False)

# ---- module mean z across split halves (Stromal) ----
def module_means(zser):
    return {m: float(np.nanmean([zser.get(g, np.nan) for g in refsets[m]])) for m in mods_list}

half_rows = []
for scheme, fn in [("cell_random", "split_half_gene_z.csv"), ("donor_balanced", "split_half_donor_gene_z.csv")]:
    h = pd.read_csv(os.path.join(_ROOT, "data", "fig3", fn)).set_index("gene")
    for m in mods_list:
        va = np.nanmean([h.z_A.get(g, np.nan) for g in refsets[m]])
        vb = np.nanmean([h.z_B.get(g, np.nan) for g in refsets[m]])
        half_rows.append({"scheme": scheme, "module": m, "z_A": va, "z_B": vb})
pd.DataFrame(half_rows).to_csv(f"{OUT}/s3_half_module_z.csv", index=False)

# ---- LODO for Stromal ----
d = np.load(f"{D}/{CT}.npz", allow_pickle=True)
X = d["X"].astype(np.float64); genes = d["genes"].astype(str)
group = d["group"].astype(str); sample = d["sample"].astype(str)


def divergence_z(mask):
    dz = None
    testable = np.ones(len(genes), bool)
    for g in ["RIF", "Fertile"]:
        Xi = X[mask & (group == g)]
        testable &= (Xi > 0).mean(0) >= MIN_DET
        n_pca = int(min(50, max(10, Xi.shape[0] // 10)))
        imp = np.asarray(magic.MAGIC(knn=5, n_pca=n_pca, t=3, random_state=SEED,
                                     verbose=False).fit_transform(pd.DataFrame(Xi, columns=genes)), float)
        sd = imp.std(0); sd[sd < 1e-8] = np.inf
        Zs = (imp - imp.mean(0)) / sd
        F = np.arctanh(np.clip((Zs.T @ Zs) / max(imp.shape[0] - 1, 1), -0.999, 0.999))
        if g == "RIF":
            Fr = F
        else:
            dz = Fr - F
    A = np.abs(dz); np.fill_diagonal(A, -np.inf)
    k = min(TOPK, A.shape[1] - 1)
    s = np.zeros(len(genes))
    for i in range(len(genes)):
        p = np.argpartition(A[i], -k)[-k:]
        w = np.abs(dz[i, p]); s[i] = np.sum(w * dz[i, p]) / w.sum() if w.sum() > 1e-12 else 0
    s[~testable] = np.nan
    ok = np.isfinite(s)
    out = np.full(len(genes), np.nan)
    if ok.sum() > 10:
        out[ok] = (s[ok] - s[ok].mean()) / (s[ok].std() if s[ok].std() > 1e-12 else 1)
    return out


full_z = Z[CT].reindex(genes).to_numpy()
full_mod = {m: np.nanmean([full_z[genes.tolist().index(g)] for g in refsets[m] if g in set(genes)]) for m in mods_list}
rows = []
for dn in sorted(np.unique(sample)):
    mask = sample != dn
    if (group[mask] == "RIF").sum() < 50 or (group[mask] == "Fertile").sum() < 50:
        continue
    z = divergence_z(mask)
    corr = spearmanr(z, full_z, nan_policy="omit").statistic
    vals = {m: np.nanmean([z[genes.tolist().index(g)] for g in refsets[m] if g in set(genes)]) for m in mods_list}
    row = {"left_out": dn, "group": group[sample == dn][0], "gene_r": float(corr)}
    row.update({f"M{m}": vals[m] for m in mods_list})
    rows.append(row)
    print(dn, round(float(corr), 3), flush=True)
lodo = pd.DataFrame(rows)
lodo.to_csv(f"{OUT}/s3_lodo.csv", index=False)

# ---- module preservation permutation ----
obs_stats, null_stats = [], []
Xz = Zg.to_numpy()
for m in mods_list:
    idx = [mods.index[mods.gene == g][0] for g in refsets[m]]
    obs = float(np.mean(np.var(Xz[idx], axis=0)))
    null = []
    for _ in range(1000):
        j = rng.choice(len(mods), size=len(idx), replace=False)
        null.append(float(np.mean(np.var(Xz[j], axis=0))))
    null = np.array(null)
    obs_stats.append({"module": m, "observed": obs, "null_mean": null.mean(),
                      "z": (obs - null.mean()) / null.std(), "p": (1 + (null >= obs).sum()) / 1001})
pd.DataFrame(obs_stats).to_csv(f"{OUT}/s3_preservation.csv", index=False)

print("S3_DONE")
