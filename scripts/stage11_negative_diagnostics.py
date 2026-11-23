
#!/usr/bin/env python3
"""Why are both external cohorts negative? Quantify power, phase and QC confounds."""
import gzip
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
import statsmodels.api as sm

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

BASE = _ROOT
EXT = f"{BASE}/data/external"
D784 = f"{EXT}/gse248784/processed"
D602 = f"{EXT}/gse106602"
GS = json.load(open(f"{EXT}/gse248784/gene_sets.json"))
MODS = {f"M{k}": v for k, v in GS["modules"].items()}
FEATS = ["M2", "HLA_score", "IFN_score"]
PHASE_SEC = ["PAEP", "IGFBP1", "GPX3", "DKK1"]
PHASE_PRO = ["MKI67", "TOP2A", "CCNB1", "BIRC5", "UBE2C"]
rows = []


def mde(n1, n2, alpha=0.05, power=0.8):
    from scipy.stats import norm
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.sqrt(1 / n1 + 1 / n2)


# ================= GSE248784 =================
ds = pd.read_csv(f"{D784}/s10_donor_scores.csv")
ph = pd.read_csv(f"{D784}/s10_phase.csv")[["sample", "group", "phase_score"]]
qc = pd.read_csv(f"{D784}/external_sample_QC.csv")
ds = ds.merge(ph, on="sample", how="left", suffixes=("", "_ph"))
ds = ds.merge(qc[["gsm", "median_genes"]], left_on="sample", right_on="gsm", how="left")

# 1) technical QC by group
for col in ["median_genes", "median_umi", "median_pct_mt"]:
    a = qc.loc[qc["group"] == "RIF", col].dropna()
    b = qc.loc[qc["group"] == "Normal", col].dropna()
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    rows.append({"dataset": "GSE248784", "check": f"QC {col}", "value": f"{a.median():.1f} vs {b.median():.1f}", "p": p})

# 2) phase correlation with module score, per cell type
for ct in ds.cell_type.unique():
    s = ds[ds.cell_type == ct].dropna(subset=["phase_score"])
    if s["group"].nunique() < 2 or len(s) < 8:
        continue
    for f in FEATS:
        rho, p = spearmanr(s[f], s.phase_score)
        rows.append({"dataset": "GSE248784", "check": f"phase-corr {f} [{ct}]",
                     "value": f"rho={rho:+.2f} (n={len(s)})", "p": p})

# 3) group effect adjusted for phase (linear model), pooled over types
for f in FEATS:
    s = ds.dropna(subset=[f, "phase_score"]).copy()
    s["rif"] = (s["group"] == "RIF").astype(int)
    X = sm.add_constant(s[["rif", "phase_score"]])
    m = sm.OLS(s[f], X).fit()
    raw = s.loc[s["group"] == "RIF", f].mean() - s.loc[s["group"] == "Normal", f].mean()
    rows.append({"dataset": "GSE248784", "check": f"adjusted group effect {f}",
                 "value": f"b_rif={m.params['rif']:+.3f} (raw delta={raw:+.3f})",
                 "p": m.pvalues["rif"]})
    # phase coefficient
    rows.append({"dataset": "GSE248784", "check": f"adjusted phase effect {f}",
                 "value": f"b_phase={m.params['phase_score']:+.3f}", "p": m.pvalues["phase_score"]})
    # technical covariate
    s2 = s.dropna(subset=["median_genes"])
    rho_t, p_t = spearmanr(s2[f], s2["median_genes"])
    m2 = sm.OLS(s2[f], sm.add_constant(s2[["rif", "phase_score", "median_genes"]])).fit()
    rows.append({"dataset": "GSE248784", "check": f"tech-corr {f}",
                 "value": f"rho_genes={rho_t:+.2f}", "p": p_t})
    rows.append({"dataset": "GSE248784", "check": f"adj group (phase+genes) {f}",
                 "value": f"b_rif={m2.params['rif']:+.3f}", "p": m2.pvalues["rif"]})

# 4) observed effect vs MDE
n1, n2 = 11, 10
for f in FEATS:
    s = ds.dropna(subset=[f])
    sd = s[f].std()
    d = (s.loc[s["group"] == "RIF", f].mean() - s.loc[s["group"] == "Normal", f].mean()) / sd
    rows.append({"dataset": "GSE248784", "check": f"power {f}",
                 "value": f"observed |d|={abs(d):.2f} SD; MDE={mde(n1, n2):.2f} SD", "p": np.nan})

# ================= GSE106602 =================
lines = [l.rstrip("\n") for l in gzip.open(f"{D602}/series_matrix.txt.gz", "rt")]


def get(key, contains=None):
    for l in lines:
        if l.startswith(key) and (contains is None or contains in l):
            return [x.strip('"') for x in l.split("\t")[1:]]
    return None


titles = get("!Sample_title")
times = [x.replace("receptivity time-point: ", "") for x in get("!Sample_characteristics_ch1", "receptivity time-point")]
states = [x.replace("tissue state: ", "") for x in get("!Sample_characteristics_ch1", "tissue state")]
meta = pd.DataFrame({"title": titles, "time": times, "state": states})
counts = pd.read_csv(f"{D602}/full_counts.tsv.gz", sep="\t", index_col=0)
sel = meta[(meta.time == "LH+7") & meta.title.isin(counts.columns)].copy()
sel["group"] = np.where(sel.state == "IVF", "RIF_like", "Healthy")

X = counts[sel.title]
cpm = X.div(X.sum(0), axis=1) * 1e6
logcpm = np.log2(cpm + 1)
lib = X.sum(0)
det = (X > 0).sum(0)
for name, ser in [("library size", lib), ("detected genes", det)]:
    qser = ser.reindex(sel.title)
    a = qser[(sel["group"] == "RIF_like").values]
    b = qser[(sel["group"] == "Healthy").values]
    rows.append({"dataset": "GSE106602", "check": f"QC {name}",
                 "value": f"{a.median():.0f} vs {b.median():.0f}",
                 "p": mannwhitneyu(a, b, alternative="two-sided").pvalue})

sym2ens = json.load(open(f"{D602}/symbol_to_ensg.json"))
sym2ens.update({
    "PAEP": "ENSG00000122133", "IGFBP1": "ENSG00000146678",
    "GPX3": "ENSG00000211445", "DKK1": "ENSG00000107984",
    "MKI67": "ENSG00000148773", "TOP2A": "ENSG00000131747",
    "CCNB1": "ENSG00000134057", "BIRC5": "ENSG00000089685",
    "UBE2C": "ENSG00000175063",
})
ens2sym = {v: k for k, v in sym2ens.items()}


def score_genes(symbols):
    ens = [sym2ens[s] for s in symbols if s in sym2ens and sym2ens[s] in logcpm.index]
    z = logcpm.loc[ens].sub(logcpm.loc[ens].mean(1), axis=0).div(logcpm.loc[ens].std(1).replace(0, np.nan), axis=0)
    return z.mean(0), len(ens)


sec, nsec = score_genes(PHASE_SEC)
pro, npro = score_genes(PHASE_PRO)
phase602 = (sec - pro).reindex(sel.title)
sel["phase"] = phase602.values
a = phase602[(sel["group"] == "RIF_like").values]; b = phase602[(sel["group"] == "Healthy").values]
rows.append({"dataset": "GSE106602", "check": "phase proxy (sec-pro)",
             "value": f"{a.mean():+.2f} vs {b.mean():+.2f} (genes {nsec}/{npro})",
             "p": mannwhitneyu(a, b, alternative="two-sided").pvalue})

# module scores identical to S7: gene z across selected samples, mean members
mod_scores = {}
for m, genes in MODS.items():
    ens = [sym2ens[g] for g in genes if g in sym2ens and sym2ens[g] in logcpm.index]
    z = logcpm.loc[ens].sub(logcpm.loc[ens].mean(1), axis=0).div(logcpm.loc[ens].std(1).replace(0, np.nan), axis=0)
    mod_scores[m] = z.mean(0).reindex(sel.title)
feat602 = {"M2": mod_scores["M2"]}
for gname, gs in [("HLA_score", ["HLA-A", "HLA-B", "HLA-C"]),
                  ("IFN_score", GS["IFN"])]:
    feat602[gname], _ = score_genes(gs)
    feat602[gname] = feat602[gname].reindex(sel.title)

for f in FEATS:
    s = pd.DataFrame({
        "score": feat602[f],
        "group": sel.set_index("title")["group"],
        "phase": phase602,
        "det": det.reindex(sel.title),
    }).dropna()
    a = s.loc[s["group"] == "RIF_like", "score"]; b = s.loc[s["group"] == "Healthy", "score"]
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    rho, prho = spearmanr(s.score, s.phase)
    rho_d, p_d = spearmanr(s.score, s.det)
    s["rif"] = (s["group"] == "RIF_like").astype(int)
    mm = sm.OLS(s.score, sm.add_constant(s[["rif", "phase"]])).fit()
    mm2 = sm.OLS(s.score, sm.add_constant(s[["rif", "phase", "det"]])).fit()
    rows.append({"dataset": "GSE106602", "check": f"tech-corr {f}",
                 "value": f"rho_detected={rho_d:+.2f}", "p": p_d})
    rows.append({"dataset": "GSE106602", "check": f"adj group (phase+det) {f}",
                 "value": f"b_rif={mm2.params['rif']:+.3f}", "p": mm2.pvalues["rif"]})
    rows.append({"dataset": "GSE106602", "check": f"phase-corr {f}",
                 "value": f"rho={rho:+.2f}", "p": prho})
    rows.append({"dataset": "GSE106602", "check": f"adjusted group effect {f}",
                 "value": (f"b_rif={mm.params['rif']:+.3f} (raw delta={a.mean()-b.mean():+.3f}); "
                           f"b_phase={mm.params['phase']:+.3f}"),
                 "p": mm.pvalues["rif"]})
    sd = s.score.std()
    d = (a.mean() - b.mean()) / sd
    rows.append({"dataset": "GSE106602", "check": f"power {f}",
                 "value": f"observed |d|={abs(d):.2f} SD; MDE={mde(len(a), len(b)):.2f} SD", "p": np.nan})

out = pd.DataFrame(rows)
out.to_csv(f"{EXT}/negative_diagnostics.csv", index=False)
pd.set_option("display.width", 220)
print(out.to_string(index=False))
print("\n=== sample balance GSE106602 ===")
print(sel.groupby(["group", "time"]).size())
print("\n=== module score SD by dataset (donor level) ===")
print("GSE248784:", ds.groupby("cell_type")[FEATS].std().mean().round(3).to_dict())
print("GSE106602:", {f: round(float(feat602[f].std()), 3) for f in FEATS})
