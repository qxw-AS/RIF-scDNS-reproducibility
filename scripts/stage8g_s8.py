
#!/usr/bin/env python3
"""Fig. S8 data: H2 bridge between network-only genes and Cao 2025 dynamic sets."""
import os
import numpy as np
import pandas as pd
import openpyxl
from scipy.stats import hypergeom

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

D2 = os.path.join(_ROOT, "data", "fig2")
D1 = os.path.join(_ROOT, "data", "fig1")
XLS = os.path.join(_ROOT, "data", "external", "cao_supp_data2.xlsx")
OUT = os.path.join(_ROOT, "data", "figS")

# Cao compartments that have a matched compartment in the LH7 atlas
MAP = {"Epithelial": ["Glandular", "Secretory"],
       "Glandular": ["Glandular", "Secretory"],
       "Stromal": ["Stromal"]}

Z = pd.read_csv(f"{D2}/gene_z_by_type.csv", index_col=0)
deg = pd.read_csv(f"{D1}/panelD_deg_full.csv")
cz = pd.read_csv(f"{D2}/combined_z.csv").set_index("gene")["combined_z"]

wb = openpyxl.load_workbook(XLS, read_only=True)
cao = {}
for s in ["Stromal", "Luminal", "Glandular", "Secretory"]:
    rows = list(wb[s].iter_rows(values_only=True))
    d = pd.DataFrame(rows[1:], columns=rows[0])
    d = d[pd.to_numeric(d.geneCluster, errors="coerce").notna()].copy()
    d["geneCluster"] = d.geneCluster.astype(int)
    d["MIscore"] = pd.to_numeric(d.MIscore, errors="coerce")
    cao[s] = d[["Gene", "geneCluster", "MIscore"]].rename(
        columns={"Gene": "gene", "geneCluster": "cao_cluster", "MIscore": "cao_MI"})
    cao[s]["cao_sheet"] = s

stats, genes = [], []
for ct, sheets in MAP.items():
    z = Z[ct].dropna()
    testable = set(z.index)
    div = z.abs().sort_values(ascending=False).index.tolist()[:50]
    dp = deg[deg.cell_type == ct].set_index("names").pvals_adj
    netonly = [g for g in div if g not in dp.index or dp[g] > 0.05]
    dyn = pd.concat([cao[s] for s in sheets], ignore_index=True)
    dyn = dyn[dyn.gene.isin(testable)]
    cg = set(dyn.gene)
    br = [g for g in netonly if g in cg]
    M, K, n, k = len(testable), len(cg), len(netonly), len(br)
    p = hypergeom.sf(k - 1, M, K, n) if n else np.nan
    stats.append({"cell_type": ct, "cao_compartments": "+".join(sheets),
                  "n_testable": M, "n_cao_dynamic_testable": K,
                  "n_network_only": n, "n_bridge": k,
                  "expected_bridge": n * K / M if M else np.nan,
                  "hypergeom_p": p})
    for g in br:
        sub = dyn[dyn.gene == g]
        for _, r in sub.iterrows():
            genes.append({"cell_type": ct, "gene": g,
                          "combined_z": float(cz.get(g, np.nan)),
                          "deg_padj": float(dp.get(g, np.nan)),
                          "cao_sheet": r.cao_sheet, "cao_cluster": int(r.cao_cluster),
                          "cao_MI": float(r.cao_MI)})

pd.DataFrame(stats).to_csv(f"{OUT}/s8_bridge_stats.csv", index=False)
pd.DataFrame(genes).to_csv(f"{OUT}/s8_bridge_genes.csv", index=False)

# background MI distribution for the three matched compartments
bg = pd.concat([cao[s].assign(compartment=s) for s in ["Glandular", "Secretory", "Stromal"]],
               ignore_index=True)
bg.to_csv(f"{OUT}/s8_cao_dynamic.csv", index=False)

print(pd.DataFrame(stats).to_string(index=False))
print(pd.DataFrame(genes).to_string(index=False))
print("S8_DONE")
