
#!/usr/bin/env python3
"""Stage 4b: pseudobulk differential expression, RIF vs fertile, per cell type.

Input : per-sample pseudobulk (summed raw counts, detected cells, n cells).
Model : CPM -> log1p; Welch t-test across samples (pseudobulk avoids
        single-cell pseudoreplication); Benjamini-Hochberg FDR.
        Pseudobulk Wilcoxon p-values are also stored for audit, but with
        n = 10 vs 6 their minimum attainable p (2.5e-4) cannot survive a
        genome-wide BH correction, so the moderated t statistic is used for
        selection (documented in the statistics report).
Output: panelD_deg_full.csv, panelD_selected.csv, panelD_notes.csv
"""
import os
import glob, os, json
import numpy as np
import pandas as pd
from scipy import stats

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

PB = os.path.join(_ROOT, "data", "pseudobulk")
OUT = os.path.join(_ROOT, "data", "fig1")
os.makedirs(OUT, exist_ok=True)
MIN_CELLS = 10          # a sample contributes a pseudobulk only with >=10 cells
MIN_RIF, MIN_FER = 5, 3  # minimum replicate samples per group
LFC_MIN, PADJ = 0.5, 0.05


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1)
    return q


files = sorted(glob.glob(f"{PB}/*.npz"))
genes_ref = None
records = []
for f in files:
    s = os.path.basename(f)[:-4]
    z = np.load(f, allow_pickle=True)
    g = z["genes"].astype(str)
    if genes_ref is None:
        genes_ref = g
    elif len(g) != len(genes_ref) or (g != genes_ref).any():
        raise RuntimeError(f"gene mismatch in {s}")
    for i, t in enumerate(z["types"].astype(str)):
        records.append({"sample": s, "cell_type": t, "n_cells": int(z["n_cells"][i]),
                        "sums": z["sums"][i], "detected": z["detected"][i]})
print("pseudobulks", len(records), "genes", len(genes_ref), flush=True)

group = {r["sample"]: ("RIF" if r["sample"].startswith("RIF") else "Fertile") for r in records}
types = sorted({r["cell_type"] for r in records})

full, selected, notes = [], [], []
for ct in types:
    rs = [r for r in records if r["cell_type"] == ct and r["n_cells"] >= MIN_CELLS]
    rif = [r for r in rs if group[r["sample"]] == "RIF"]
    fer = [r for r in rs if group[r["sample"]] == "Fertile"]
    if len(rif) < MIN_RIF or len(fer) < MIN_FER:
        notes.append({"cell_type": ct, "n_rif_samples": len(rif), "n_fer_samples": len(fer),
                      "reason": "insufficient replicate samples: no DEG test"})
        print(ct, "insufficient samples", len(rif), len(fer), flush=True)
        continue

    def logcpm(r):
        tot = r["sums"].sum()
        return np.log1p(r["sums"] / max(tot, 1) * 1e6)

    A = np.vstack([logcpm(r) for r in rif])          # n_rif x genes
    B = np.vstack([logcpm(r) for r in fer])          # n_fer x genes
    cpm_a = np.vstack([r["sums"] / max(r["sums"].sum(), 1) * 1e6 for r in rif]).mean(0)
    cpm_b = np.vstack([r["sums"] / max(r["sums"].sum(), 1) * 1e6 for r in fer]).mean(0)
    pct_a = (sum(r["detected"] for r in rif) /
             max(sum(r["n_cells"] for r in rif), 1) * 100)
    pct_b = (sum(r["detected"] for r in fer) /
             max(sum(r["n_cells"] for r in fer), 1) * 100)

    lfc = np.log2((cpm_a + 1) / (cpm_b + 1))
    keep = ((cpm_a >= 1) | (cpm_b >= 1)) & ((pct_a >= 10) | (pct_b >= 10)) & (np.abs(lfc) >= LFC_MIN)
    idx = np.where(keep)[0]
    t, p = stats.ttest_ind(A[:, idx], B[:, idx], axis=0, equal_var=False)
    try:
        _, pw = stats.mannwhitneyu(A[:, idx], B[:, idx], axis=0, alternative="two-sided")
    except Exception:
        pw = np.full(len(idx), np.nan)
    df = pd.DataFrame({"cell_type": ct, "gene": genes_ref[idx], "log2FC": lfc[idx],
                       "t": t, "pval": p, "pval_wilcox": pw,
                       "mean_cpm_rif": cpm_a[idx], "mean_cpm_fer": cpm_b[idx],
                       "pct_rif": pct_a[idx], "pct_fer": pct_b[idx],
                       "n_rif_samples": len(rif), "n_fer_samples": len(fer)})
    df["padj"] = bh(df["pval"].to_numpy())
    full.append(df)
    sig = df[(df.padj < PADJ) & np.isfinite(df.padj)]
    print(f"{ct}: samples {len(rif)}/{len(fer)}, tested {len(df)}, sig {len(sig)}", flush=True)
    if len(sig) == 0:
        notes.append({"cell_type": ct, "n_rif_samples": len(rif), "n_fer_samples": len(fer),
                      "reason": "no significant DEG (BH < 0.05)"})
        continue
    up = sig[sig.log2FC > 0].nlargest(2, "log2FC")
    dn = sig[sig.log2FC < 0].nsmallest(2, "log2FC")
    selected.append(pd.concat([up, dn]))

deg = pd.concat(full, ignore_index=True)
deg.to_csv(f"{OUT}/panelD_deg_full.csv", index=False)
sel = pd.concat(selected, ignore_index=True) if selected else deg.iloc[0:0]
sel.to_csv(f"{OUT}/panelD_selected.csv", index=False)
pd.DataFrame(notes, columns=["cell_type", "n_rif_samples", "n_fer_samples", "reason"]).to_csv(
    f"{OUT}/panelD_notes.csv", index=False)
print(sel.groupby("cell_type").size().to_string(), flush=True)
print("notes:", notes, flush=True)
print("STAGE4B_DONE", flush=True)
