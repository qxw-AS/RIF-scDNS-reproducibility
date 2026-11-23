
#!/usr/bin/env python3
"""Stage 10c: donor-level module/HLA/IFN scores in GSE248784.

Primary unit = donor x cell type pseudobulk. Scores are mean z-scored
expression of module members across samples within a cell type, matching the
GSE106602 analysis. Mann-Whitney RIF vs Normal; BH within cell type and
globally.
"""
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

BASE = os.path.join(_ROOT, "data", "external", "gse248784")
PROC = os.path.join(BASE, "processed")
FIG2 = os.path.join(_ROOT, "data", "fig2")

GS = json.load(open(os.path.join(BASE, "gene_sets.json")))
MODULES = {f"M{k}": v for k, v in GS["modules"].items()}
HLA = GS["HLA"]
IFN = GS["IFN"]
FEATS = list(MODULES) + ["HLA_score", "IFN_score"] + HLA

adata = ad.read_h5ad(os.path.join(PROC, "external_annotated.h5ad"))
print("loaded", adata.shape, adata.obs.cell_type.nunique(), "cell types", flush=True)

# ---------------- pseudobulk per sample x cell type ----------------
rows = []
counts = {}
for ct in adata.obs.cell_type.cat.categories:
    sub = adata[adata.obs.cell_type == ct]
    genes = [g for g in sub.var_names if g in
             set(HLA) | set(IFN) | {g for v in MODULES.values() for g in v}]
    mat, meta = [], []
    for s in sub.obs["sample"].cat.categories:
        idx = np.where(sub.obs["sample"].values == s)[0]
        if len(idx) < 20:
            continue
        c = np.asarray(sub.layers["counts"][idx].sum(0)).ravel()
        mat.append(c)
        meta.append({"sample": s,
                     "donor": sub.obs["donor"].values[idx][0],
                     "group": sub.obs["group"].values[idx][0],
                     "cell_type": ct, "n_cells": len(idx)})
    if len(mat) < 6:
        print("skip", ct, "samples", len(mat), flush=True)
        continue
    M = np.vstack(mat)
    lib = M.sum(1, keepdims=True)
    lib[lib == 0] = 1
    L = np.log1p(M / lib * 1e6)
    Z = (L - L.mean(0)) / (L.std(0) + 1e-8)
    df = pd.DataFrame(meta)
    df["sample"] = df["sample"].astype(str)
    df["group"] = df["group"].astype(str)
    for mod, gl in MODULES.items():
        use = [j for j, g in enumerate(genes) if g in set(gl)]
        df[mod] = Z[:, use].mean(1) if use else np.nan
    use = [j for j, g in enumerate(genes) if g in set(HLA)]
    df["HLA_score"] = Z[:, use].mean(1) if use else np.nan
    use = [j for j, g in enumerate(genes) if g in set(IFN)]
    df["IFN_score"] = Z[:, use].mean(1) if use else np.nan
    for g in HLA:
        if g in genes:
            df[g] = Z[:, genes.index(g)]
        else:
            df[g] = np.nan
    rows.append(df)
    counts[ct] = (len(mat), int(df.group.value_counts().get("RIF", 0)),
                  int(df.group.value_counts().get("Normal", 0)))
    print(ct, counts[ct], flush=True)

scores = pd.concat(rows, ignore_index=True)
scores.to_csv(os.path.join(PROC, "s10_donor_scores.csv"), index=False)

# ---------------- composition ----------------
comp = pd.crosstab(adata.obs["sample"], adata.obs["cell_type"], normalize="index")
meta = adata.obs.drop_duplicates("sample").set_index("sample")[["group"]]
comp = comp.join(meta)
comp.to_csv(os.path.join(PROC, "s10_composition.csv"))
comp_rows = []
for ct in comp.columns.drop("group"):
    a = comp.loc[comp.group == "RIF", ct].values
    b = comp.loc[comp.group == "Normal", ct].values
    if len(a) < 3 or len(b) < 3:
        continue
    comp_rows.append({"cell_type": ct, "mean_RIF": a.mean(), "mean_Normal": b.mean(),
                      "delta": a.mean() - b.mean(),
                      "p": mannwhitneyu(a, b, alternative="two-sided").pvalue})
C = pd.DataFrame(comp_rows)
if len(C):
    C["q"] = multipletests(C.p, method="fdr_bh")[1]
    C.to_csv(os.path.join(PROC, "s10_composition_tests.csv"), index=False)
    print("\ncomposition differences:")
    print(C.sort_values("p").to_string(index=False), flush=True)

# ---------------- cycle-phase proxy ----------------
SEC = ["PAEP", "IGFBP1", "GPX3", "DKK1"]
PROLIF = ["MKI67", "TOP2A", "CCNB1", "BIRC5", "UBE2C"]
# pooled counts per sample across all cells
mat, meta2 = [], []
for s in adata.obs["sample"].cat.categories:
    idx = np.where(adata.obs["sample"].values == s)[0]
    mat.append(np.asarray(adata.layers["counts"][idx].sum(0)).ravel())
    meta2.append((s, adata.obs["group"].values[idx][0]))
Mall = np.vstack(mat)
lib = Mall.sum(1, keepdims=True); lib[lib == 0] = 1
Lall = np.log1p(Mall / lib * 1e6)
Zall = (Lall - Lall.mean(0)) / (Lall.std(0) + 1e-8)
ph = pd.DataFrame({"sample": [x[0] for x in meta2],
                   "group": [x[1] for x in meta2]})
for name, gl in [("secretory_score", SEC), ("prolif_score", PROLIF)]:
    use = [j for j, g in enumerate(adata.var_names) if g in set(gl)]
    ph[name] = Zall[:, use].mean(1)
ph["phase_score"] = ph.secretory_score - ph.prolif_score
ph.to_csv(os.path.join(PROC, "s10_phase.csv"), index=False)
for c in ["secretory_score", "prolif_score", "phase_score"]:
    a = ph.loc[ph.group == "RIF", c]
    b = ph.loc[ph.group == "Normal", c]
    print(f"phase {c}: RIF {a.mean():.2f} vs Normal {b.mean():.2f}, "
          f"p={mannwhitneyu(a, b, alternative='two-sided').pvalue:.3f}", flush=True)

# ---------------- donor-level tests ----------------
tests = []
for ct, g in scores.groupby("cell_type"):
    a, b = g[g.group == "RIF"], g[g.group == "Normal"]
    if len(a) < 3 or len(b) < 3:
        continue
    for f in FEATS:
        x, y = a[f].dropna().values, b[f].dropna().values
        if len(x) < 3 or len(y) < 3:
            continue
        p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        tests.append({"cell_type": ct, "feature": f,
                      "n_RIF": len(x), "n_Normal": len(y),
                      "mean_RIF": x.mean(), "mean_Normal": y.mean(),
                      "delta": x.mean() - y.mean(),
                      "median_RIF": np.median(x), "median_Normal": np.median(y),
                      "p": p})
T = pd.DataFrame(tests)
T["q_within_type"] = np.nan
for ct, idx in T.groupby("cell_type").groups.items():
    T.loc[idx, "q_within_type"] = multipletests(T.loc[idx, "p"], method="fdr_bh")[1]
T["q_global"] = multipletests(T["p"], method="fdr_bh")[1]
T = T.sort_values(["feature", "p"])
T.to_csv(os.path.join(PROC, "s10_tests.csv"), index=False)
print("\nTop nominal donor-level results:")
print(T.sort_values("p").head(15)[["cell_type", "feature", "n_RIF", "n_Normal",
                                   "delta", "p", "q_within_type", "q_global"]].to_string(index=False), flush=True)

# ---------------- module direction vs discovery ----------------
disc = pd.read_csv(os.path.join(FIG2, "gene_modules.csv"))
disc_mean = disc.groupby("module").combined_z.mean()
disc_mean.index = [f"M{int(i)}" for i in disc_mean.index]
md = T[T.feature.isin(MODULES)].copy()
md["discovery_mean_z"] = md.feature.map(disc_mean)
md["concordant"] = np.sign(md.delta) == np.sign(md.discovery_mean_z)
md.to_csv(os.path.join(PROC, "s10_module_direction.csv"), index=False)
# one row per module: strongest |delta| cell type
best = md.reindex(md.delta.abs().sort_values(ascending=False).index).groupby("feature").head(1)
rho, p = spearmanr(best.discovery_mean_z, best.delta)
print("\nmodule direction (best per module):")
print(best[["feature", "cell_type", "discovery_mean_z", "delta", "p", "q_global"]].to_string(index=False))
print(f"sign concordance {best.concordant.mean():.3f}, Spearman rho={rho:.3f} p={p:.3f}", flush=True)

# ---------------- M2 gene-level concordance ----------------
cz = pd.read_csv(os.path.join(FIG2, "combined_z.csv")).set_index("gene").combined_z
m2 = GS["M2"]
# pooled pseudobulk log2FC across all cells, and epithelial/glandular-specific
def pooled_log2fc(sub):
    genes = [g for g in sub.var_names]
    mat, meta = [], []
    for s in sub.obs["sample"].cat.categories:
        idx = np.where(sub.obs["sample"].values == s)[0]
        if len(idx) == 0:
            continue
        mat.append(np.asarray(sub.layers["counts"][idx].sum(0)).ravel())
        meta.append((s, sub.obs["group"].values[idx][0]))
    M = np.vstack(mat)
    lib = M.sum(1, keepdims=True); lib[lib == 0] = 1
    L = np.log2(M / lib * 1e6 + 1)
    gser = pd.Series([x[1] for x in meta])
    return genes, L, gser

gene_rows = []
for label, mask in [("all", np.ones(adata.n_obs, bool)),
                    ("Epithelial", (adata.obs.cell_type == "Epithelial").values),
                    ("Glandular", (adata.obs.cell_type == "Glandular").values),
                    ("Stromal", (adata.obs.cell_type == "Stromal").values)]:
    sub = adata[mask].copy()
    if sub.n_obs < 500:
        continue
    genes, L, gser = pooled_log2fc(sub)
    for g in m2:
        if g not in genes:
            continue
        j = genes.index(g)
        rif = L[gser.values == "RIF", j].mean()
        nor = L[gser.values == "Normal", j].mean()
        gene_rows.append({"compartment": label, "gene": g,
                          "external_log2FC": rif - nor,
                          "discovery_combined_z": cz.get(g, np.nan)})
G = pd.DataFrame(gene_rows)
G.to_csv(os.path.join(PROC, "s10_m2_genes.csv"), index=False)
for label, g in G.groupby("compartment"):
    r, pv = spearmanr(g.discovery_combined_z, g.external_log2FC)
    sc_ = (np.sign(g.discovery_combined_z) == np.sign(g.external_log2FC)).mean()
    print(f"M2 {label}: n={len(g)} rho={r:.3f} p={pv:.3f} sign={sc_:.3f}", flush=True)

# ---------------- summary ----------------
summary = {
    "n_samples": int(scores["sample"].nunique()),
    "n_cells": int(adata.n_obs),
    "n_cell_types": int(adata.obs.cell_type.nunique()),
    "groups": {k: int(v) for k, v in scores.drop_duplicates("sample").group.value_counts().items()},
    "cell_type_samples": {k: {"samples": v[0], "RIF": v[1], "Normal": v[2]} for k, v in counts.items()},
    "any_q05": {
        "M2": bool((T[(T.feature == "M2")].q_global < 0.05).any()),
        "HLA_score": bool((T[(T.feature == "HLA_score")].q_global < 0.05).any()),
        "IFN_score": bool((T[(T.feature == "IFN_score")].q_global < 0.05).any()),
    },
    "min_q_M2": float(T[T.feature == "M2"].q_global.min()),
    "min_q_HLA": float(T[T.feature == "HLA_score"].q_global.min()),
    "min_q_IFN": float(T[T.feature == "IFN_score"].q_global.min()),
    "module_sign_concordance": float(best.concordant.mean()),
    "module_spearman_rho": float(rho),
    "module_spearman_p": float(p),
}
json.dump(summary, open(os.path.join(PROC, "s10_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
print("STAGE10C_DONE", flush=True)
