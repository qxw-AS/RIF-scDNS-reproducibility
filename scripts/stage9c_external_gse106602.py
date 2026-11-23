
#!/usr/bin/env python3
"""Fig. S7 — independent bulk RNA-seq direction consistency (GSE106602).

External cohort: mid-secretory (LH+7) endometrium, 19 repeated-IVF-failure
(IVF_fri) vs 16 healthy controls. Bulk tissue, not cell-type-resolved; this is a
direction-consistency check, not a cell-type-level replication.
"""
import os
import gzip
import json
import numpy as np
import pandas as pd
import requests
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

D2 = os.path.join(_ROOT, "data", "fig2")
EXT = os.path.join(_ROOT, "data", "external", "gse106602")
OUT = os.path.join(_ROOT, "data", "figS")
HLA = ["HLA-A", "HLA-B", "HLA-C"]
IFN = ["STAT1", "IRF1", "CXCL9", "CXCL10", "IDO1", "CD74", "GBP1", "GBP2",
       "GBP5", "IFIT1", "IFIT3", "ISG15", "MX1", "OAS1", "HLA-DRA"]

mods = pd.read_csv(f"{D2}/gene_modules.csv")
mods = mods[mods.module.isin(range(1, 9))]
mod_genes = {m: g.gene.tolist() for m, g in mods.groupby("module")}
all_syms = sorted(set(HLA) | set(IFN) | {g for gs in mod_genes.values() for g in gs})

# ---------- sample metadata ----------
lines = [l.rstrip("\n") for l in gzip.open(f"{EXT}/series_matrix.txt.gz", "rt")]
def get(key):
    for l in lines:
        if l.startswith(key):
            return [x.strip('"') for x in l.split("\t")[1:]]
    return None
titles = get("!Sample_title")
times = None
for l in lines:
    if l.startswith("!Sample_characteristics_ch1") and "LH+" in l:
        if "receptivity time-point" in l:
            times = [x.strip('"').replace("receptivity time-point: ", "") for x in l.split("\t")[1:]]
states = None
for l in lines:
    if l.startswith("!Sample_characteristics_ch1") and "tissue state" in l:
        states = [x.strip('"').replace("tissue state: ", "") for x in l.split("\t")[1:]]
meta = pd.DataFrame({"title": titles, "time": times, "state": states})
meta["group"] = np.where(meta.state == "IVF", "RIF_like", "Healthy")
print(meta.groupby(["group", "time"]).size())

# ---------- counts ----------
counts = pd.read_csv(f"{EXT}/full_counts.tsv.gz", sep="\t", index_col=0)
print("counts", counts.shape)
# primary: LH+7 only, IVF_fri vs Healthy
sel = meta[(meta.time == "LH+7") & meta.title.isin(counts.columns)]
print(sel.groupby("group").size())

# ---------- symbol -> ENSG ----------
cache = f"{EXT}/symbol_to_ensg.json"
try:
    sym2ens = json.load(open(cache))
except Exception:
    sym2ens = {}
missing = [s for s in all_syms if s not in sym2ens]
for i in range(0, len(missing), 150):
    chunk = missing[i:i + 150]
    r = requests.post("https://rest.ensembl.org/lookup/symbol/homo_sapiens",
                      json={"symbols": chunk},
                      headers={"Content-Type": "application/json"}, timeout=60)
    r.raise_for_status()
    for sym, rec in r.json().items():
        sym2ens[sym] = rec.get("id")
    print("mapped", i + len(chunk), "/", len(missing), flush=True)
json.dump(sym2ens, open(cache, "w"), indent=0)
ens2sym = {v: k for k, v in sym2ens.items() if v}
print("mapped symbols", sum(1 for s in all_syms if sym2ens.get(s)))

# ---------- normalize ----------
sub = counts[sel.title.tolist()].copy()
sub = sub[sub.sum(axis=1) > 0]
cpm = sub / sub.sum(axis=0) * 1e6
logcpm = np.log2(cpm + 1)

rows = []
for m, gs in mod_genes.items():
    ens = [sym2ens[g] for g in gs if sym2ens.get(g) in logcpm.index]
    if len(ens) < 3:
        continue
    z = logcpm.loc[ens].apply(lambda v: (v - v.mean()) / v.std(ddof=0) if v.std(ddof=0) > 1e-9 else v * 0, axis=1)
    score = z.mean(axis=0)
    a = score[sel[sel.group == "RIF_like"].title]
    b = score[sel[sel.group == "Healthy"].title]
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    disc = float(mods[mods.module == m].combined_z.mean())
    rows.append({"module": f"M{m}", "n_genes_mapped": len(ens),
                 "discovery_combined_z": disc,
                 "external_delta": a.mean() - b.mean(),
                 "external_mean_RIF": a.mean(), "external_mean_Healthy": b.mean(),
                 "p": p, "direction_concordant": bool(np.sign(a.mean() - b.mean()) == np.sign(disc))})
mod = pd.DataFrame(rows)
mod["q"] = multipletests(mod.p, method="fdr_bh")[1]
mod.to_csv(f"{OUT}/s7_external_modules.csv", index=False)

# ---------- M2 gene-level ----------
m2 = mods[mods.module == 2].copy()
gene_rows = []
for _, r in m2.iterrows():
    ens = sym2ens.get(r.gene)
    if ens not in logcpm.index:
        continue
    a = logcpm.loc[ens, sel[sel.group == "RIF_like"].title]
    b = logcpm.loc[ens, sel[sel.group == "Healthy"].title]
    gene_rows.append({"gene": r.gene, "combined_z": r.combined_z,
                      "external_log2FC": a.mean() - b.mean(),
                      "n_genes": 1})
g = pd.DataFrame(gene_rows)
rho, pv = spearmanr(g.combined_z, g.external_log2FC)
sign_conc = float((np.sign(g.combined_z) == np.sign(g.external_log2FC)).mean())
g.to_csv(f"{OUT}/s7_external_m2_genes.csv", index=False)

# ---------- HLA / IFN ----------
feat_rows = []
for gname in HLA + ["IFN_score", "HLA_score"]:
    if gname == "IFN_score":
        ens = [sym2ens[x] for x in IFN if sym2ens.get(x) in logcpm.index]
        z = logcpm.loc[ens].apply(lambda v: (v - v.mean()) / v.std(ddof=0) if v.std(ddof=0) > 1e-9 else v * 0, axis=1)
        score = z.mean(axis=0)
    elif gname == "HLA_score":
        ens = [sym2ens[x] for x in HLA if sym2ens.get(x) in logcpm.index]
        z = logcpm.loc[ens].apply(lambda v: (v - v.mean()) / v.std(ddof=0) if v.std(ddof=0) > 1e-9 else v * 0, axis=1)
        score = z.mean(axis=0)
    else:
        ens = sym2ens.get(gname)
        if ens not in logcpm.index:
            continue
        score = logcpm.loc[ens]
    a = score[sel[sel.group == "RIF_like"].title]
    b = score[sel[sel.group == "Healthy"].title]
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    feat_rows.append({"feature": gname, "mean_RIF": a.mean(), "mean_Healthy": b.mean(),
                      "delta": a.mean() - b.mean(), "p": p})
feat = pd.DataFrame(feat_rows)
feat["q"] = multipletests(feat.p, method="fdr_bh")[1]
feat.to_csv(f"{OUT}/s7_external_features.csv", index=False)

summary = {"n_samples": int(len(sel)), "n_RIF": int((sel.group == "RIF_like").sum()),
           "n_healthy": int((sel.group == "Healthy").sum()),
           "n_modules_mapped": int(len(mod)),
           "module_direction_concordant": int(mod.direction_concordant.sum()),
           "m2_genes_mapped": int(len(g)), "m2_spearman_rho": float(rho),
           "m2_spearman_p": float(pv), "m2_sign_concordance": sign_conc}
json.dump(summary, open(f"{OUT}/s7_external_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print(mod.to_string(index=False))
print(feat.to_string(index=False))
print("S7_DONE")
