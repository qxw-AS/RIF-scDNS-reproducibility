
#!/usr/bin/env python3
"""Stage 6b: condition-specific gene coordination networks and divergence.

scDNS uses DREMI + information-theoretic divergence between condition-specific
gene-pair relationships. Here (Python-only environment, no R/scDNS) we use the
documented approximation:
  MAGIC imputation per group -> Pearson correlation per gene pair ->
  Fisher z -> Delta z = z(RIF) - z(fertile) -> gene-level signed divergence
  s_i = weighted mean Delta z over the 50 strongest |Delta z| partners ->
  within-cell-type z-score -> Stouffer combination across testable cell types.

Positive combined_z = coordination gained in RIF; negative = lost.
All approximations are declared in Fig2_statistics_report.md.
"""
import os
import glob, json, os, time
import numpy as np
import pandas as pd
import magic

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

SEED = 20261117
DATA = os.path.join(_ROOT, "data", "fig2")
OUT = DATA
TOP_K = 50
MIN_DET = 0.05          # gene must be detected in >=5% of cells in BOTH groups
MIN_TYPES = 4           # minimum testable cell types for combined_z
GROUPS = ["RIF", "Fertile"]

mpl_types = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
             "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]
files = {os.path.basename(f)[:-4]: f for f in glob.glob(f"{DATA}/*.npz")}
files = {k.replace("_", " ").replace("B-Plasma", "B/Plasma"): v for k, v in files.items()}
# restore exact keys (file names replaced spaces and slashes)
key_map = {}
for ct in mpl_types:
    fn = ct.replace(" ", "_").replace("/", "-")
    key_map[ct] = f"{DATA}/{fn}.npz"

z_by_type, stats = {}, []
for ct in mpl_types:
    t0 = time.time()
    d = np.load(key_map[ct], allow_pickle=True)
    X = d["X"].astype(np.float64)
    genes = d["genes"].astype(str)
    group = d["group"].astype(str)
    zs = np.full(len(genes), np.nan)
    testable = np.ones(len(genes), bool)
    info = {"cell_type": ct, "n_rif": int((group == "RIF").sum()),
            "n_fertile": int((group == "Fertile").sum())}

    dz = None
    for g in GROUPS:
        Xi = X[group == g]
        det = (Xi > 0).mean(0)
        testable &= det >= MIN_DET
        df = pd.DataFrame(Xi, columns=genes)
        n_pca = int(min(50, max(10, Xi.shape[0] // 10)))
        imp = magic.MAGIC(knn=5, n_pca=n_pca, t=3, random_state=SEED,
                          verbose=False).fit_transform(df)
        imp = np.asarray(imp, dtype=np.float64)
        sd = imp.std(0)
        sd[sd < 1e-8] = np.inf          # constant gene -> zero correlation
        Z = (imp - imp.mean(0)) / sd
        C = (Z.T @ Z) / max(imp.shape[0] - 1, 1)
        np.clip(C, -0.999, 0.999, out=C)
        F = np.arctanh(C)
        if g == "RIF":
            F_rif = F
        else:
            dz = F_rif - F
        if g == "RIF":
            info["n_pca"] = n_pca
    info["n_testable_genes"] = int(testable.sum())

    # gene-level signed divergence: weighted mean Delta z over top-K partners
    A = np.abs(dz)
    np.fill_diagonal(A, -np.inf)
    k = min(TOP_K, A.shape[1] - 1)
    s = np.zeros(len(genes))
    for i in range(len(genes)):
        p = np.argpartition(A[i], -k)[-k:]
        w = np.abs(dz[i, p])
        sw = w.sum()
        s[i] = np.sum(w * dz[i, p]) / sw if sw > 1e-12 else 0.0
    s[~testable] = np.nan
    ok = np.isfinite(s)
    if ok.sum() > 10:
        mu, sdv = s[ok].mean(), s[ok].std()
        zs[ok] = (s[ok] - mu) / (sdv if sdv > 1e-12 else 1.0)
    z_by_type[ct] = zs
    info.update({"n_divergence_tested": int(ok.sum()),
                 "mean_abs_dz": float(np.abs(dz[ok]).mean()) if ok.any() else np.nan,
                 "mean_s": float(np.nanmean(s)), "sd_s": float(np.nanstd(s)),
                 "seconds": round(time.time() - t0, 1)})
    stats.append(info)
    print(f"{ct}: {info['n_rif']}/{info['n_fertile']} cells, testable {ok.sum()}, "
          f"mean|dz| {info['mean_abs_dz']:.4f}, {info['seconds']}s", flush=True)

Z = pd.DataFrame(z_by_type, index=genes)
Z.index.name = "gene"
Z.to_csv(f"{OUT}/gene_z_by_type.csv")

k_types = Z.notna().sum(axis=1)
sum_z = Z.fillna(0).sum(axis=1)
combined = sum_z / np.sqrt(k_types.replace(0, np.nan))
n_pos = (Z > 0).sum(axis=1)
out = pd.DataFrame({"gene": Z.index, "combined_z": combined.to_numpy(),
                    "k_types": k_types.to_numpy(), "n_positive": n_pos.to_numpy(),
                    "mean_z": Z.mean(axis=1).to_numpy(),
                    "max_abs_z": Z.abs().max(axis=1).to_numpy()})
out.loc[out.k_types < MIN_TYPES, "combined_z"] = np.nan
out = out.sort_values("combined_z", ascending=False)
out.to_csv(f"{OUT}/combined_z.csv", index=False)
pd.DataFrame(stats).to_csv(f"{OUT}/type_divergence_stats.csv", index=False)
print(out["combined_z"].describe().to_string())
print("genes with combined_z:", int(out.combined_z.notna().sum()), flush=True)
print("STAGE6B_DONE", flush=True)
