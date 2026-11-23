
#!/usr/bin/env python3
"""Stage 7d: DEG-union GO with the matched 1,353-gene background (Fig.3B).

The original stage7b test used all 36,601 expressed genes as background while
the module GO used the 1,353 divergence-testable genes.  FDRs from the two
tests are therefore not directly comparable.  This script repeats the DEG-union
test with the same 1,353-gene background and reports whether the M2 terms stay
significant (v2 design, P1 item).

Outputs (data/fig3/)
  deg_go_union_matched.csv, go_matched_compare.csv
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
cz = pd.read_csv(f"{D}/combined_z.csv")
background = sorted(cz.loc[cz.combined_z.notna(), "gene"].tolist())
print("matched background:", len(background), flush=True)

sets = {}
for ct, g in deg.groupby("cell_type"):
    d = g.sort_values("pvals_adj")
    d = d[d.pvals_adj < 0.05]
    sets[ct] = d.names.head(N_TOP).tolist()
union = sorted({x for v in sets.values() for x in v})
print("DEG union size:", len(union), flush=True)

res = None
for attempt in range(3):
    try:
        r = gseapy.enrichr(gene_list=union, gene_sets=LIB, organism="human",
                           background=background, outdir=None)
        res = r.results.copy()
        break
    except Exception as e:
        print(f"attempt {attempt + 1}: {e}", flush=True)
        time.sleep(5)
if res is None:
    raise SystemExit("Enrichr failed")

res["Term_clean"] = res["Term"].str.replace(r"\s*\(GO:\d+\)$", "", regex=True)
res.to_csv(f"{OUT}/deg_go_union_matched.csv", index=False)
print(f"DEG union: {int((res['Adjusted P-value'] < 0.05).sum())} terms FDR<0.05", flush=True)

# compare M2's 8 selected terms
mods = pd.read_csv(f"{D}/go_enrichment_full.csv")
mods["Term_clean"] = mods["Term"].str.replace(r"\s*\(GO:\d+\)$", "", regex=True)
M = pd.read_csv(f"{D}/go_module_matrix.csv", index_col=0)
m2_terms = [t for t in M.index if pd.notna(M.loc[t, "2"])]
cmp = []
for t in m2_terms:
    mm = mods[(mods.module == 2) & (mods.Term_clean == t)]
    dd = res[res.Term_clean == t]
    cmp.append({
        "term": t,
        "module_fdr": float(mm["Adjusted P-value"].min()) if len(mm) else np.nan,
        "deg_fdr_36601": np.nan,
        "deg_fdr_1353": float(dd["Adjusted P-value"].min()) if len(dd) else np.nan,
        "deg_in_top100_union": bool(len(dd)),
    })
old = pd.read_csv(f"{OUT}/deg_go_union.csv")
old["Term_clean"] = old.Term.str.replace(r"\s*\(GO:\d+\)$", "", regex=True)
for c in cmp:
    o = old[old.Term_clean == c["term"]]
    if len(o):
        c["deg_fdr_36601"] = float(o["Adjusted P-value"].min())
cmp = pd.DataFrame(cmp)
cmp.to_csv(f"{OUT}/go_matched_compare.csv", index=False)
print(cmp.to_string(index=False), flush=True)
print("STAGE7D_DONE", flush=True)
