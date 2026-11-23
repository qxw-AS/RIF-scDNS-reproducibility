
#!/usr/bin/env python3
"""Fig. S5 data: DEG robustness, pseudobulk concordance, donor consistency, power."""
import os
import glob
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.stats import spearmanr

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

D1 = os.path.join(_ROOT, "data", "fig1")
D2 = os.path.join(_ROOT, "data", "fig2")
SC = os.path.join(_ROOT, "data", "scdeg")
OUT = os.path.join(_ROOT, "data", "figS")
SEED = 20261117

CT = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
      "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]

cell = pd.read_csv(f"{D1}/panelD_deg_full.csv")
pb = pd.read_csv(f"{D2}/pseudobulk_deg_full.csv").rename(columns={"gene": "names"})
merged = cell.merge(pb[["cell_type", "names", "log2FC", "padj"]],
                    on=["cell_type", "names"], suffixes=("_cell", "_pb"), how="inner")
merged["sig_cell"] = merged.pvals_adj < 0.05
merged["sig_pb"] = merged.padj < 0.05
merged.to_csv(f"{OUT}/s5_deg_merged.csv.gz", index=False, compression="gzip")

rows = []
for ct, g in merged.groupby("cell_type"):
    rows.append({"cell_type": ct, "n_pairs": len(g),
                 "spearman_log2FC": spearmanr(g.logfoldchanges, g.log2FC).statistic,
                 "n_sig_cell": int(g.sig_cell.sum()), "n_sig_pb": int(g.sig_pb.sum()),
                 "n_sig_both": int((g.sig_cell & g.sig_pb).sum())})
pd.DataFrame(rows).to_csv(f"{OUT}/s5_concordance.csv", index=False)

# ---- donor-level consistency for Fig.1D selected pairs ----
sel = pd.read_csv(f"{D1}/panelD_selected.csv")
sel = sel[sel.cell_type.isin(CT)]
expr = {}
for f in glob.glob(f"{SC}/*.npz"):
    base = f.split("/")[-1][:-4]
    sample, ct = base.split("__", 1)
    ct = ct.replace("B-Plasma", "B/Plasma")
    if ct not in CT:
        continue
    d = np.load(f, allow_pickle=True)
    genes = d["genes"].astype(str)
    from scipy.sparse import csr_matrix
    mat = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    gi = {g: i for i, g in enumerate(genes)}
    for gene in sel.loc[sel.cell_type == ct, "gene"].unique():
        if gene not in gi:
            continue
        col = np.asarray(mat[:, gi[gene]].todense()).ravel()
        expr[(ct, sample, gene)] = float(col.mean())

rows = []
for _, r in sel.iterrows():
    sam = [s for (ct, s, g) in expr if ct == r.cell_type and g == r.gene]
    if not sam:
        continue
    rif = [expr[(r.cell_type, s, r.gene)] for s in sam if s.startswith("RIF")]
    fer = [expr[(r.cell_type, s, r.gene)] for s in sam if s.startswith("LH7")]
    if not rif or not fer:
        continue
    mf = np.mean(fer)
    diffs = np.array(rif) - mf
    sign = np.sign(np.mean(rif) - mf)
    rows.append({"cell_type": r.cell_type, "gene": r.gene,
                 "overall_diff": float(np.mean(rif) - mf),
                 "frac_RIF_same_dir": float((np.sign(diffs) == sign).mean()),
                 "n_RIF_donors": len(rif), "n_fer_donors": len(fer)})
don = pd.DataFrame(rows)
don.to_csv(f"{OUT}/s5_donor_consistency.csv", index=False)

# ---- downsampling power curve (2,000-gene network panel) ----
key = {}
for ct in CT:
    fn = ct.replace(" ", "_").replace("/", "-")
    key[ct] = f"{D2}/{fn}.npz"
rows = []
for ct in ["Stromal", "Epithelial", "T cell"]:
    d = np.load(key[ct], allow_pickle=True)
    X = d["X"].astype(np.float32); genes = d["genes"].astype(str); group = d["group"].astype(str)
    rng = np.random.default_rng(SEED)
    for N in [100, 300, 500, 1000]:
        r = np.where(group == "RIF")[0]; f = np.where(group == "Fertile")[0]
        n = min(N, len(r), len(f))
        ri = rng.choice(r, n, replace=False); fi = rng.choice(f, n, replace=False)
        idx = np.concatenate([ri, fi])
        a = ad.AnnData(X=np.vstack([X[idx]]), obs=pd.DataFrame({"group": group[idx]}))
        a.var_names = genes
        sc.tl.rank_genes_groups(a, "group", groups=["RIF"], reference="Fertile",
                                method="wilcoxon", use_raw=False)
        res = a.uns["rank_genes_groups"]
        padj = np.asarray(res["pvals_adj"]["RIF"])
        rows.append({"cell_type": ct, "N_per_group": n,
                     "n_genes_tested": int(np.isfinite(padj).sum()),
                     "n_sig_padj05": int((padj < 0.05).sum()),
                     "n_sig_padj01": int((padj < 0.01).sum())})
        print(ct, N, rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/s5_power.csv", index=False)

print("S5_DONE")
