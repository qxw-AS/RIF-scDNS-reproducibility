
#!/usr/bin/env python3
"""Stage 7b: GO overlap between rewiring modules and DEG pathways (Fig.3B).

  * module GO: re-used from stage 6d (go_enrichment_full.csv; background =
    the 1,353 divergence-testable genes);
  * DEG GO: union of the top-100 DEGs of every cell type (Fig.1D Wilcoxon,
    pvals_adj < 0.05) -> one pooled Enrichr test on GO_Biological_Process_2023
    with background = all 36,601 tested genes; per-cell-type results are also
    computed for the statistics report (not shown in the panel);
  * overlap = terms with FDR < 0.05 in both a module and the DEG union, plus
    Jaccard of the top-10 term lists.

Outputs (data/fig3/)
  deg_go_union.csv, deg_go_by_type.csv, go_overlap_matrix.csv,
  go_overlap_summary.csv
"""
import os
import time
import numpy as np
import pandas as pd
import gseapy

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

D = os.path.join(_ROOT, "data", "fig2")
OUT = os.path.join(_ROOT, "data", "fig3")
LIB = "GO_Biological_Process_2023"
N_TOP = 100

deg = pd.read_csv(os.path.join(_ROOT, "data", "fig1", "panelD_deg_full.csv"))
background = sorted(deg.names.unique().tolist())
mods = pd.read_csv(f"{D}/go_enrichment_full.csv")
mods["Term_clean"] = mods["Term"].str.replace(r"\s*\(GO:\d+\)$", "", regex=True)


def enrich(genes, label):
    for attempt in range(3):
        try:
            r = gseapy.enrichr(gene_list=genes, gene_sets=LIB, organism="human",
                               background=background, outdir=None)
            res = r.results.copy()
            res["Term_clean"] = res["Term"].str.replace(r"\s*\(GO:\d+\)$", "", regex=True)
            print(f"{label}: {len(genes)} genes, {int((res['Adjusted P-value']<0.05).sum())} "
                  f"terms FDR<0.05", flush=True)
            return res
        except Exception as e:
            print(f"{label} attempt {attempt+1}: {e}", flush=True)
            time.sleep(5)
    return pd.DataFrame()


# ---------------- pooled DEG union ----------------
sets = {}
for ct, g in deg.groupby("cell_type"):
    d = g.sort_values("pvals_adj")
    d = d[d.pvals_adj < 0.05]
    sets[ct] = d.names.head(N_TOP).tolist()
union = sorted({x for v in sets.values() for x in v})
print("DEG union size:", len(union), flush=True)
du = enrich(union, "DEG_union")
du["set"] = "DEG_union"
du.to_csv(f"{OUT}/deg_go_union.csv", index=False)

by = []
for ct, gl in sets.items():
    r = enrich(gl, ct)
    if len(r):
        r["cell_type"] = ct
        by.append(r)
pd.concat(by, ignore_index=True).to_csv(f"{OUT}/deg_go_by_type.csv", index=False)

# ---------------- overlap matrix ----------------
mod_sig = mods[mods["Adjusted P-value"] < 0.05].copy()
# top-2 terms per module (non-duplicated) as displayed in Fig.2B
top_mod = []
for m, g in mods.groupby("module"):
    chosen = []
    for _, r in g.sort_values("Adjusted P-value").iterrows():
        gs = set(str(r["Genes"]).split(";")) if pd.notna(r["Genes"]) else set()
        if any(len(gs & c) / max(len(gs | c), 1) > 0.8 for c in chosen):
            continue
        chosen.append(gs)
        top_mod.append(r)
        if len(chosen) >= 2:
            break
top_mod = pd.DataFrame(top_mod)
deg_top = du.sort_values("Adjusted P-value").head(10)
shared = sorted(set(mod_sig.Term_clean) & set(du.loc[du["Adjusted P-value"] < 0.05, "Term_clean"]))
rows = list(dict.fromkeys(top_mod.Term_clean.tolist() + deg_top.Term_clean.tolist() + shared))
print("shared significant terms:", shared, flush=True)

mod_order = sorted(mods.module.unique())
mat = pd.DataFrame(np.nan, index=rows, columns=[f"M{m}" for m in mod_order] + ["DEG"])
for _, r in mods[mods.Term_clean.isin(rows)].iterrows():
    v = -np.log10(max(r["Adjusted P-value"], 1e-300))
    c = f"M{r['module']}"
    if np.isnan(mat.loc[r.Term_clean, c]) or v > mat.loc[r.Term_clean, c]:
        mat.loc[r.Term_clean, c] = v
for _, r in du[du.Term_clean.isin(rows)].iterrows():
    mat.loc[r.Term_clean, "DEG"] = -np.log10(max(r["Adjusted P-value"], 1e-300))
mat.index.name = "GO_term"
mat.to_csv(f"{OUT}/go_overlap_matrix.csv")

# row order: shared/significant first, then module-specific, then DEG-specific
def prio(t):
    ins = t in shared
    mm = mat.loc[t, [f"M{m}" for m in mod_order]].max()
    dd = mat.loc[t, "DEG"]
    return (0 if ins else 1 if (pd.notna(mm) and mm > -np.log10(.05)) else 2 if (pd.notna(dd) and dd > -np.log10(.05)) else 3,
            -(np.nanmax([mm, dd]) if np.isfinite(np.nanmax([mm, dd])) else 0))
mat = mat.loc[sorted(rows, key=prio)]
mat.to_csv(f"{OUT}/go_overlap_matrix.csv")

top10_mod = {m: set(mods[(mods.module == m) & (mods["Adjusted P-value"] < 0.05)]
                    .sort_values("Adjusted P-value").Term_clean.head(10))
             for m in mod_order}
top10_deg = set(du.sort_values("Adjusted P-value").head(10).Term_clean)
summ = []
for m in mod_order:
    inter = top10_mod[m] & top10_deg
    j = len(inter) / len(top10_mod[m] | top10_deg) if (top10_mod[m] | top10_deg) else np.nan
    summ.append({"module": m, "n_sig_terms": int((mods[(mods.module == m)]["Adjusted P-value"] < 0.05).sum()),
                 "n_shared_with_DEG": len(set(mod_sig[mod_sig.module == m].Term_clean) & set(du.loc[du["Adjusted P-value"] < 0.05, "Term_clean"])),
                 "top10_jaccard_with_DEG": j})
pd.DataFrame(summ).to_csv(f"{OUT}/go_overlap_summary.csv", index=False)
print(pd.DataFrame(summ).to_string(index=False), flush=True)
print("shared:", shared, flush=True)
print("STAGE7B_DONE", flush=True)
