
#!/usr/bin/env python3
"""Fig. S6 data: H1 decision robustness across cell types and scope."""
import os
import numpy as np
import pandas as pd

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

D2 = os.path.join(_ROOT, "data", "fig2")
D3 = os.path.join(_ROOT, "data", "fig3")
OUT = os.path.join(_ROOT, "data", "figS")

cur = pd.read_csv(f"{D3}/jaccard_curves.csv")
summ = pd.read_csv(f"{D3}/jaccard_summary.csv")
glob = pd.read_csv(f"{D3}/jaccard_global.csv")
Z = pd.read_csv(f"{D2}/gene_z_by_type.csv", index_col=0)

# cross-type Jaccard of top-50 divergence genes
sets = {ct: set(Z[ct].dropna().abs().sort_values(ascending=False).index[:50])
        for ct in Z.columns}
cts = list(Z.columns)
J = pd.DataFrame(index=cts, columns=cts, dtype=float)
for a in cts:
    for b in cts:
        J.loc[a, b] = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
J.to_csv(f"{OUT}/s6_cross_type_jaccard.csv")

# per-type p matrix
P = cur.pivot_table(index="cell_type", columns="N", values="p_upper")
P.to_csv(f"{OUT}/s6_pvalue_matrix.csv")

# global vs cell-type envelope
med = cur.groupby("N").jaccard.agg(["median", "min", "max", "count"]).reset_index()
med = med.merge(glob[["N", "jaccard"]].rename(columns={"jaccard": "global"}), on="N", how="left")
med.to_csv(f"{OUT}/s6_scope_envelope.csv", index=False)

print("cross-type J range:", float(J.to_numpy()[np.triu_indices(len(cts), 1)].min()),
      float(J.to_numpy()[np.triu_indices(len(cts), 1)].max()))
print(med.to_string(index=False))
print("S6_DONE")
