
#!/usr/bin/env python3
"""Donor-level (pseudobulk) validation of module/HLA/IFN signals.

Uses per-sample pseudobulk sums already generated from the discovery cohort.
No new data; tests whether Fig. S4 signals survive at the donor level.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

D2 = os.path.join(_ROOT, "data", "fig2")
PB = os.path.join(_ROOT, "data", "pseudobulk")
OUT = os.path.join(_ROOT, "data", "figS")
CT = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
      "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]
HLA = ["HLA-A", "HLA-B", "HLA-C"]
IFN = ["STAT1", "IRF1", "CXCL9", "CXCL10", "IDO1", "CD74", "GBP1", "GBP2",
       "GBP5", "IFIT1", "IFIT3", "ISG15", "MX1", "OAS1", "HLA-DRA"]
MIN_CELLS = 20

mods = pd.read_csv(f"{D2}/gene_modules.csv")
mods = mods[mods.module.isin(range(1, 9))]
mod_genes = {m: g.gene.tolist() for m, g in mods.groupby("module")}
ALL_G = sorted(set(HLA) | set(IFN) | {g for gs in mod_genes.values() for g in gs})

rows = []
for f in sorted(glob.glob(f"{PB}/*.npz")):
    sample = f.split("/")[-1][:-4]
    grp = "RIF" if sample.startswith("RIF") else "Fertile"
    d = np.load(f, allow_pickle=True)
    genes = d["genes"].astype(str)
    gi = {g: i for i, g in enumerate(genes)}
    sums = d["sums"].astype(np.float64)
    types = d["types"].astype(str)
    ncells = d["n_cells"].astype(int)
    lib = sums.sum(1)
    for ti, ct in enumerate(types):
        if ct not in CT or ncells[ti] < MIN_CELLS:
            continue
        cpm = np.log1p(sums[ti] / max(lib[ti], 1) * 1e6)
        rec = {"sample": sample, "group": grp, "cell_type": ct, "n_cells": int(ncells[ti])}
        for g in ALL_G:
            rec[g] = cpm[gi[g]] if g in gi else np.nan
        for m, gs in mod_genes.items():
            ok = [g for g in gs if g in gi]
            rec[f"M{m}_genes"] = len(ok)
        rows.append(rec)

df = pd.DataFrame(rows)
# module scores and IFN score: z-score genes across samples within cell type, then mean
for ct, sub in df.groupby("cell_type"):
    idx = sub.index
    for m, gs in mod_genes.items():
        zs = []
        for g in gs:
            if g not in sub.columns:
                continue
            v = sub[g].astype(float)
            sd = v.std(ddof=0)
            if sd < 1e-9:
                continue
            zs.append((v - v.mean()) / sd)
        df.loc[idx, f"M{m}_score"] = np.mean(zs, axis=0) if zs else np.nan
    zs = []
    for g in IFN:
        v = sub[g].astype(float)
        sd = v.std(ddof=0)
        if sd < 1e-9:
            continue
        zs.append((v - v.mean()) / sd)
    df.loc[idx, "IFN_score"] = np.mean(zs, axis=0)
    for g in HLA:
        if g in sub.columns:
            v = sub[g].astype(float)
            df.loc[idx, f"{g}_z"] = (v - v.mean()) / v.std(ddof=0) if v.std(ddof=0) > 1e-9 else np.nan
    df.loc[idx, "HLA_score"] = df.loc[idx, [f"{g}_z" for g in HLA]].mean(axis=1)

df.to_csv(f"{OUT}/s9_donor_scores.csv", index=False)

# group tests per cell type
tests = []
for ct, sub in df.groupby("cell_type"):
    rif, fer = sub[sub.group == "RIF"], sub[sub.group == "Fertile"]
    for m in range(1, 9):
        a, b = rif[f"M{m}_score"].dropna(), fer[f"M{m}_score"].dropna()
        if len(a) >= 3 and len(b) >= 3:
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
            tests.append({"cell_type": ct, "feature": f"M{m}", "n_RIF": len(a),
                          "n_Fertile": len(b), "mean_RIF": a.mean(),
                          "mean_Fertile": b.mean(), "delta": a.mean() - b.mean(), "p": p})
    for feat in HLA + ["HLA_score", "IFN_score"]:
        a, b = rif[feat].dropna(), fer[feat].dropna()
        if len(a) >= 3 and len(b) >= 3:
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
            tests.append({"cell_type": ct, "feature": feat, "n_RIF": len(a),
                          "n_Fertile": len(b), "mean_RIF": a.mean(),
                          "mean_Fertile": b.mean(), "delta": a.mean() - b.mean(), "p": p})
t = pd.DataFrame(tests)
t["q"] = multipletests(t.p, method="fdr_bh")[1]
t.to_csv(f"{OUT}/s9_donor_tests.csv", index=False)

# donor-level correlations
cors = []
for ct, sub in df.groupby("cell_type"):
    for x, y in [("M2_score", "IFN_score"), ("M2_score", "HLA_score"),
                 ("HLA_score", "IFN_score")]:
        s = sub[[x, y]].dropna()
        if len(s) >= 6:
            r, p = spearmanr(s[x], s[y])
            cors.append({"cell_type": ct, "x": x, "y": y, "n": len(s),
                         "rho": r, "p": p})
c = pd.DataFrame(cors)
c["q"] = multipletests(c.p, method="fdr_bh")[1]
c.to_csv(f"{OUT}/s9_donor_correlations.csv", index=False)

print("DONOR_LEVEL_DONE")
print(t.sort_values(["feature", "p"]).groupby("feature").head(3).to_string(index=False))
print(c.to_string(index=False))
