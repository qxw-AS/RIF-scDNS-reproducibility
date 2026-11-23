
#!/usr/bin/env python3
"""Fig. S2 data: network construction flow, parameter sensitivity, threshold stability."""
import os
import json
import numpy as np
import pandas as pd
import magic
from scipy.stats import spearmanr, hypergeom
from scipy.cluster.hierarchy import linkage, fcluster

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

SEED = 20261117
D = os.path.join(_ROOT, "data", "fig2")
OUT = os.path.join(_ROOT, "data", "figS")
TYPES = ["Stromal", "Epithelial", "T cell"]
GRID_K = [3, 5, 8]
GRID_T = [2, 3, 4]
TOPK = 50
MIN_DET = 0.05


def zscore(v):
    ok = np.isfinite(v)
    out = np.full_like(v, np.nan, dtype=float)
    if ok.sum() > 10:
        mu, sd = v[ok].mean(), v[ok].std()
        out[ok] = (v[ok] - mu) / (sd if sd > 1e-12 else 1.0)
    return out


def divergence(X, genes, group, knn=5, t=3, rank=False, impute=True):
    dz = None
    testable = np.ones(len(genes), bool)
    for g in ["RIF", "Fertile"]:
        Xi = X[group == g]
        testable &= (Xi > 0).mean(0) >= MIN_DET
        if impute:
            n_pca = int(min(50, max(10, Xi.shape[0] // 10)))
            imp = np.asarray(magic.MAGIC(knn=knn, n_pca=n_pca, t=t, random_state=SEED,
                                         verbose=False).fit_transform(pd.DataFrame(Xi, columns=genes)), float)
        else:
            imp = Xi
        if rank:
            from scipy.stats import rankdata
            imp = np.column_stack([rankdata(imp[:, j]) for j in range(imp.shape[1])])
        sd = imp.std(0)
        sd[sd < 1e-8] = np.inf
        Z = (imp - imp.mean(0)) / sd
        C = np.clip((Z.T @ Z) / max(imp.shape[0] - 1, 1), -0.999, 0.999)
        F = np.arctanh(C)
        if g == "RIF":
            Fr = F
        else:
            dz = Fr - F
    A = np.abs(dz)
    np.fill_diagonal(A, -np.inf)
    k = min(TOPK, A.shape[1] - 1)
    s = np.zeros(len(genes))
    for i in range(len(genes)):
        p = np.argpartition(A[i], -k)[-k:]
        w = np.abs(dz[i, p])
        s[i] = np.sum(w * dz[i, p]) / w.sum() if w.sum() > 1e-12 else 0.0
    s[~testable] = np.nan
    return zscore(s)


# ---------------- parameter grid ----------------
ref = pd.read_csv(f"{D}/gene_z_by_type.csv", index_col=0)
rows = []
store = {}
for ct in TYPES:
    d = np.load(f"{D}/{ct.replace(' ', '_')}.npz", allow_pickle=True)
    X = d["X"].astype(np.float64); genes = d["genes"].astype(str); group = d["group"].astype(str)
    r0 = ref[ct].reindex(genes).to_numpy()
    for k in GRID_K:
        for t in GRID_T:
            z = divergence(X, genes, group, knn=k, t=t)
            ok = np.isfinite(z) & np.isfinite(r0)
            rho = spearmanr(z[ok], r0[ok]).statistic
            a = np.argsort(-np.abs(np.where(ok, z, np.nan)))[:50]
            b = np.argsort(-np.abs(np.where(ok, r0, np.nan)))[:50]
            j50 = len(set(a) & set(b)) / len(set(a) | set(b))
            rows.append({"cell_type": ct, "knn": k, "t": t, "spearman_vs_ref": float(rho),
                         "top50_jaccard_vs_ref": float(j50), "n_testable": int(ok.sum())})
            store[(ct, k, t)] = (z, genes)
            print(ct, k, t, round(float(rho), 3), round(float(j50), 3), flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/s2_grid.csv", index=False)

# ---------------- correlation metric sensitivity ----------------
mrows = []
for ct in TYPES:
    d = np.load(f"{D}/{ct.replace(' ', '_')}.npz", allow_pickle=True)
    X = d["X"].astype(np.float64); genes = d["genes"].astype(str); group = d["group"].astype(str)
    r0 = ref[ct].reindex(genes).to_numpy()
    zs = {
        "MAGIC + Pearson (reference)": store[(ct, 5, 3)][0],
        "MAGIC + Spearman": divergence(X, genes, group, rank=True, impute=True),
        "No MAGIC + Pearson": divergence(X, genes, group, impute=False),
    }
    for name, z in zs.items():
        ok = np.isfinite(z) & np.isfinite(r0)
        rho = spearmanr(z[ok], r0[ok]).statistic
        a = np.argsort(-np.abs(np.where(ok, z, np.nan)))[:50]
        b = np.argsort(-np.abs(np.where(ok, r0, np.nan)))[:50]
        j50 = len(set(a) & set(b)) / len(set(a) | set(b))
        mrows.append({"cell_type": ct, "metric": name, "spearman_vs_ref": float(rho),
                      "top50_jaccard_vs_ref": float(j50)})
    print(ct, "metric sensitivity done", flush=True)
pd.DataFrame(mrows).to_csv(f"{OUT}/s2_metric.csv", index=False)

# ---------------- effect-size threshold stability ----------------
Z = ref.copy()
cz = pd.read_csv(f"{D}/combined_z.csv")
refmod = pd.read_csv(f"{D}/gene_modules.csv")
go = pd.read_csv(f"{D}/go_enrichment_full.csv")
mhc = go[(go.module == 2) & go.Term.str.contains("MHC Class I", na=False)]
mhc_genes = set(";".join(mhc.Genes.dropna().astype(str)).split(";"))
bg = set(cz.loc[cz.combined_z.notna(), "gene"])


def merge_small(Zf, labels, min_size=30):
    labels = labels.copy()
    uniq = np.unique(labels)
    big = [u for u in uniq if (labels == u).sum() >= min_size]
    if len(big) <= 1:
        return labels
    cents = {u: Zf[labels == u].mean(0) for u in uniq}
    for u in uniq:
        if u in big:
            continue
        dd = {b: np.linalg.norm(cents[u] - cents[b]) for b in big}
        labels[labels == u] = min(dd, key=dd.get)
    return labels


def bh(p):
    p = np.asarray(p, float); q = np.empty_like(p); m = len(p); prev = 1.0
    for rank, idx in enumerate(np.argsort(p)[::-1], 1):
        prev = min(prev, p[idx] * m / (m - rank + 1)); q[idx] = prev
    return q


trows = []
refsets = {m: set(g.gene) for m, g in refmod.groupby("module")}
for cut in [1.0, 1.25, 1.5, 2.0, 2.5]:
    genes = cz.loc[(cz.combined_z.notna()) & (cz.combined_z.abs() >= cut), "gene"].tolist()
    Zs = Z.loc[Z.index.isin(genes)]
    Zf = Zs.fillna(0.0).to_numpy()
    if len(Zs) < 60:
        trows.append({"cz_min": cut, "n_genes": len(Zs), "n_modules": 0, "hla_module": 0,
                      "min_mhc_q": np.nan, "mean_matched_jaccard": np.nan})
        continue
    Ln = linkage(Zf, method="ward", optimal_ordering=True)
    best = None
    for h in np.linspace(2, 40, 380):
        lab = fcluster(Ln, t=h, criterion="distance")
        if len(np.unique(lab)) < 2:
            continue
        lab2 = merge_small(Zf, lab)
        n = len(np.unique(lab2))
        if 8 <= n <= 12:
            best = (h, lab2, n); break
    if best is None:
        cands = []
        for h in np.linspace(1, 50, 500):
            lab2 = merge_small(Zf, fcluster(Ln, t=h, criterion="distance"))
            cands.append((abs(len(np.unique(lab2)) - 10), lab2))
        cands.sort(key=lambda x: x[0]); best = (np.nan, cands[0][1], len(np.unique(cands[0][1])))
    h, lab, nmod = best
    df = pd.DataFrame({"gene": Zs.index, "module": lab})
    hla_genes = {"HLA-A", "HLA-B", "HLA-C"}
    hla_mods = df[df.gene.isin(hla_genes)].module.value_counts()
    hla_mod = int(hla_mods.index[0]) if len(hla_mods) else -1
    # module-level one-sided MHC-I overlap
    ps = []
    for m, g in df.groupby("module"):
        s = set(g.gene); n = len(s); K = len(mhc_genes & bg); N = len(bg)
        k = len(s & mhc_genes)
        ps.append(hypergeom.sf(k - 1, N, K, n) if k > 0 else 1.0)
    q = bh(ps)
    minq = float(np.min(q))
    # matched Jaccard vs reference modules
    js = []
    for m, g in df.groupby("module"):
        s = set(g.gene)
        js.append(max(len(s & rs) / len(s | rs) for rs in refsets.values()))
    trows.append({"cz_min": cut, "n_genes": len(Zs), "n_modules": nmod,
                  "hla_module": int(hla_mod), "hla_module_size": int((df.module == hla_mod).sum()) if hla_mod > 0 else 0,
                  "min_mhc_q": minq, "mean_matched_jaccard": float(np.mean(js)),
                  "n_mhc_sig_modules": int((q < 0.05).sum())})
    print(cut, len(Zs), nmod, "HLA module", hla_mod, "minq", minq, flush=True)
pd.DataFrame(trows).to_csv(f"{OUT}/s2_threshold.csv", index=False)
print("S2_DONE")
