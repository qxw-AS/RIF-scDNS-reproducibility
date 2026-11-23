
#!/usr/bin/env python3
"""Stage 6d: GO Biological Process enrichment per rewiring module.

gseapy Enrichr, library GO_Biological_Process_2023, background = the 1,353
clustered genes (not the whole genome) to avoid panel-selection bias.
Outputs full results + the module x term matrix used in Fig.2B.
"""
import os, re, time
import numpy as np
import pandas as pd
import gseapy

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

DATA = os.path.join(_ROOT, "data", "fig2")
mods = pd.read_csv(f"{DATA}/gene_modules.csv")
summ = pd.read_csv(f"{DATA}/module_summary.csv")
# background = all genes that were testable for divergence (not only the
# strong-rewiring subset used for clustering), to avoid conditioning on
# the outcome when testing GO over-representation.
cz = pd.read_csv(f"{DATA}/combined_z.csv")
background = cz.loc[cz.combined_z.notna(), "gene"].tolist()
allres = []
for m in sorted(mods.module.unique()):
    genes = mods.loc[mods.module == m, "gene"].tolist()
    for attempt in range(3):
        try:
            r = gseapy.enrichr(gene_list=genes, gene_sets="GO_Biological_Process_2023",
                               organism="human", background=background,
                               outdir=None)
            res = r.results.copy()
            break
        except Exception as e:
            print(f"module {m} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(5)
            res = pd.DataFrame()
    if res is None or len(res) == 0:
        print(f"module {m}: no enrichment result", flush=True)
        continue
    res["module"] = m
    allres.append(res)
    print(f"module {m}: {len(genes)} genes, {int((res['Adjusted P-value']<0.05).sum())} terms FDR<0.05", flush=True)
    time.sleep(1)

full = pd.concat(allres, ignore_index=True)
full["Term_clean"] = full["Term"].str.replace(r"\s*\(GO:\d+\)$", "", regex=True)
full.to_csv(f"{DATA}/go_enrichment_full.csv", index=False)

# top 1 term per module; for modules with any FDR<0.05, keep up to 2 terms
# that are not near-duplicates (Jaccard of overlapping genes <= 0.8).
# Terms are kept even when FDR>=0.05; the star marks FDR<0.05.
sel_rows = []
for m, g in full.groupby("module"):
    sig = (g["Adjusted P-value"] < 0.05).any()
    n_keep = 2 if sig else 1
    chosen = []
    for _, r in g.sort_values("Adjusted P-value").iterrows():
        genes_r = set(str(r["Genes"]).split(";")) if pd.notna(r["Genes"]) else set()
        dup = False
        for c in chosen:
            inter = len(genes_r & c[1])
            union = len(genes_r | c[1]) or 1
            if inter / union > 0.8:
                dup = True
                break
        if not dup:
            chosen.append((r, genes_r))
        if len(chosen) >= n_keep:
            break
    sel_rows.extend([c[0] for c in chosen])
sel = pd.DataFrame(sel_rows)
terms = sel.Term_clean.drop_duplicates().tolist()
print("selected terms:", len(terms), flush=True)

M = pd.DataFrame(np.nan, index=terms, columns=sorted(mods.module.unique()))
for _, r in full[full.Term_clean.isin(terms)].iterrows():
    v = -np.log10(max(r["Adjusted P-value"], 1e-300))
    if np.isnan(M.loc[r.Term_clean, r.module]) or v > M.loc[r.Term_clean, r.module]:
        M.loc[r.Term_clean, r.module] = v
# order rows by the module in which each term is strongest
first_mod = {t: sel[sel.Term_clean == t].module.iloc[0] for t in terms}
M = M.loc[sorted(terms, key=lambda t: (first_mod[t], -np.nanmax(M.loc[t])))]
M.index.name = "GO_term"
M.to_csv(f"{DATA}/go_module_matrix.csv")
pd.DataFrame([{"module": m, "n_genes": int((mods.module == m).sum())}
              for m in sorted(mods.module.unique())]).to_csv(f"{DATA}/go_module_sizes.csv", index=False)
print(M.notna().sum(axis=1).to_string(), flush=True)
print("STAGE6D_DONE", flush=True)
