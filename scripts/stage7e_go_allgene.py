
#!/usr/bin/env python3
"""Stage 7e: module GO on the 36,601-gene (all tested genes) background.

Fig.3B compares module GO with DEG-union GO. To keep the two sides on the same
background, every module gene set is re-tested with Enrichr
(GO_Biological_Process_2023; BH) using the same 36,601 genes that entered the
DEG analysis as background.  The 1,353-gene matched-background results
(stage 7d) are kept as a sensitivity analysis.

Outputs (data/fig2/)
  go_enrichment_full_allgene.csv  modules x terms, FDR on 36,601 background
  go_module_matrix_allgene.csv    -log10(FDR) matrix (representative rows)
"""
import os
import numpy as np
import pandas as pd
from gseapy import enrich

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

DEG = os.path.join(_ROOT, "data", "fig1", "panelD_deg_full.csv")
FIG2 = os.path.join(_ROOT, "data", "fig2")
LIB = "GO_Biological_Process_2023"

deg = pd.read_csv(DEG)
background = sorted(deg.names.unique().tolist())
gm = pd.read_csv(f"{FIG2}/gene_modules.csv")

rows = []
for m in sorted(gm.module.unique()):
    genes = gm.loc[gm.module == m, "gene"].tolist()
    res = enrich(gene_list=genes, gene_sets=LIB, background=background,
                 outdir=None).results
    res["Term_clean"] = res.Term.str.replace(r"\s*\(GO:\d+\)$", "", regex=True)
    res["module"] = m
    rows.append(res)
    print(f"M{m}: {len(genes)} genes, "
          f"{int((res['Adjusted P-value'] < 0.05).sum())} terms FDR<0.05",
          flush=True)

full = pd.concat(rows, ignore_index=True)
full.to_csv(f"{FIG2}/go_enrichment_full_allgene.csv", index=False)

# representative -log10(FDR) matrix, two non-redundant terms per module
chosen, mat_rows = [], []
for m in sorted(gm.module.unique()):
    picked = 0
    for _, r in full[full.module == m].sort_values("Adjusted P-value").iterrows():
        gs = set(str(r["Genes"]).split(";"))
        if any(len(gs & c) / max(len(gs | c), 1) > 0.8 for c in chosen):
            continue
        chosen.append(gs)
        mat_rows.append(r)
        picked += 1
        if picked >= 2:
            break
mat = pd.DataFrame(np.nan, index=[r["Term_clean"] for r in mat_rows],
                   columns=[f"M{m}" for m in sorted(gm.module.unique())])
for r in mat_rows:
    mat.loc[r["Term_clean"], f"M{r['module']}"] = -np.log10(
        max(r["Adjusted P-value"], 1e-300))
mat.index.name = "GO_term"
mat.to_csv(f"{FIG2}/go_module_matrix_allgene.csv")
print("STAGE7E_DONE", flush=True)
