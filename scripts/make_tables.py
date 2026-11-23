
#!/usr/bin/env python3
"""Build Supplementary Tables S1–S5 (xlsx)."""
import os
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

D = os.path.join(_ROOT, "data")
OUT = os.path.join(_ROOT, "tables")
os.makedirs(OUT, exist_ok=True)
CT = ["Epithelial", "Glandular", "Ciliated", "Stromal", "Endothelial",
      "Lymphatic EC", "T cell", "NK cell", "B/Plasma", "Myeloid", "Mast"]
HDR_FILL = PatternFill("solid", fgColor="EDEDED")


def write_table(path, sheets, widths=None):
    wb = openpyxl.Workbook(write_only=True)
    for title, df in sheets:
        ws = wb.create_sheet(title=title[:31])
        cells = []
        for c in df.columns:
            cell = WriteOnlyCell(ws, value=str(c))
            cell.font = Font(bold=True)
            cell.fill = HDR_FILL
            cells.append(cell)
        ws.append(cells)
        for row in df.itertuples(index=False, name=None):
            ws.append(list(row))
        ws.freeze_panes = "A2"
        for j, c in enumerate(df.columns):
            w = 12
            if widths and c in widths:
                w = widths[c]
            ws.column_dimensions[get_column_letter(j + 1)].width = w
    wb.save(path)
    print("WROTE", path, [(t, len(d)) for t, d in sheets], flush=True)


# ---------------- Table S1 ----------------
qc = pd.read_csv(f"{D}/figS/s1_qc_by_sample.csv")
qc = qc.sort_values(["group", "sample"], ascending=[True, True]).reset_index(drop=True)
qc["n_genes_IQR"] = qc.apply(lambda r: f"{r.n_genes_q25:.0f}–{r.n_genes_q75:.0f}", axis=1)
qc["pct_mt_IQR"] = qc.apply(lambda r: f"{r.pct_mt_q25:.2f}–{r.pct_mt_q75:.2f}", axis=1)
t1 = qc.rename(columns={"n_cells": "n_cells_after_QC", "n_genes_median": "n_genes_median",
                        "pct_mt_median": "pct_mito_median"})[
    ["sample", "group", "n_cells_after_QC", "n_genes_median", "n_genes_IQR",
     "pct_mito_median", "pct_mt_IQR"]].copy()
t1.columns = ["sample", "group", "n_cells_after_QC", "n_genes_median",
              "n_genes_IQR", "pct_mito_median", "pct_mito_IQR"]

comp = pd.read_csv(f"{D}/figS/s1_composition_by_donor.csv")
comp["frac"] = comp.frac.round(5)
comp = comp.sort_values(["group", "sample", "cell_type"])
t2 = comp[["sample", "group", "cell_type", "n", "total", "frac"]].rename(
    columns={"n": "n_cells", "total": "n_cells_sample", "frac": "fraction_of_sample"})

piv = comp.pivot_table(index="cell_type", columns="group", values="n", aggfunc="sum").reindex(CT)
piv["total"] = piv.sum(axis=1)
piv["fraction_total"] = (piv.total / piv.total.sum()).round(5)
piv = piv.reset_index().rename_axis(None, axis=1)
piv.columns = [str(c) for c in piv.columns]

clu = pd.read_csv(f"{D}/figS/s1_cluster_qc.csv", index_col=0).reset_index()
clu = clu.rename(columns={clu.columns[0]: "leiden_cluster"})[
    ["leiden_cluster", "n_cells", "assigned_type", "max_z", "margin",
     "top_sample", "max_sample_frac"]].round(4)
clu = clu.sort_values("leiden_cluster")
notes = pd.DataFrame({"note": [
    "Cohort: 16 donors (10 RIF, 6 fertile); 149,590 annotated cells in 11 cell types.",
    "Unassigned cells (n = 2,784) were excluded before annotation and are not included in the 149,590 cells or in the 53-cluster QC table; per-cell QC for these cells was not re-extracted in this supplementary pass.",
    "Cell-type composition was compared at donor level (Wilcoxon rank-sum, BH correction); all q > 0.8.",
    "Annotation stability: 80% per-sample cell subsampling, 3 repeats, Leiden resolutions 1.0/1.5/2.0/2.5; ARI versus full-data labels.",
    "Leiden cluster QC: 53 clusters; 14 clusters have >50% of cells from one sample (9.6% of all cells).",
]})

write_table(f"{OUT}/TableS1_sample_QC.xlsx",
            [("S1_samples", t1), ("S1_celltype_counts", t2), ("S1_celltype_totals", piv),
             ("S1_cluster_QC", clu), ("S1_notes", notes)],
            widths={"sample": 14, "group": 9, "cell_type": 15, "n_cells": 10,
                    "note": 110})

# ---------------- Table S2 ----------------
deg = pd.read_csv(f"{D}/fig1/panelD_deg_full.csv")
t2 = pd.DataFrame({
    "cell_type": deg.cell_type, "gene": deg.names,
    "log2FC_RIF_vs_Fertile": deg.logfoldchanges.round(5),
    "p_value": deg.pvals.apply(lambda v: f"{v:.3e}"),
    "FDR": deg.pvals_adj.apply(lambda v: f"{v:.3e}"),
    "pct_fertile": deg.pct_fer.round(2), "pct_RIF": deg.pct_rif.round(2),
    "n_RIF_cells": deg.n_rif_cells, "n_fertile_cells": deg.n_fer_cells,
    "significant_FDR05": deg.pvals_adj < 0.05,
    "FDR_sort": deg.pvals_adj,
}).sort_values(["cell_type", "FDR_sort"], kind="mergesort").drop(columns="FDR_sort")
write_table(f"{OUT}/TableS2_DEG_full.xlsx", [("S2_DEG_all", t2)],
            widths={"cell_type": 14, "gene": 16, "p_value": 12, "FDR": 12})

# ---------------- Table S3 ----------------
mods = pd.read_csv(f"{D}/fig2/gene_modules.csv")
cz = pd.read_csv(f"{D}/fig2/combined_z.csv").set_index("gene")["combined_z"]
Z = pd.read_csv(f"{D}/fig2/gene_z_by_type.csv", index_col=0)
gs = pd.read_csv(f"{D}/figS/s3_gene_subsample.csv")
lodo = pd.read_csv(f"{D}/figS/s3_lodo.csv")
pres = pd.read_csv(f"{D}/figS/s3_preservation.csv")
half = pd.read_csv(f"{D}/figS/s3_half_module_z.csv")
prof = pd.read_csv(f"{D}/fig2/module_profiles.csv", index_col=0)

t3 = mods.copy()
t3["combined_z"] = t3.gene.map(cz).round(4)
t3 = t3.merge(Z.round(4), left_on="gene", right_index=True, how="left")
t3["n_types_nonNA"] = Z.notna().sum(axis=1).reindex(t3.gene).values
t3 = t3[["gene", "module", "combined_z"] + CT + ["n_types_nonNA"]]
t3 = t3.sort_values(["module", "combined_z"], ascending=[True, False])

gs_sum = gs.groupby("module").jaccard.agg(
    n_iter="size", mean="mean", median="median", q25=lambda s: s.quantile(.25),
    q75=lambda s: s.quantile(.75), min="min", max="max").round(4).reset_index()
gs_sum.columns = ["module", "n_iter", "jaccard_mean", "jaccard_median",
                  "jaccard_q25", "jaccard_q75", "jaccard_min", "jaccard_max"]
lodo_sum = lodo[[f"M{i}" for i in range(1, 9)]].agg(
    ["mean", "min", "max"]).T.reset_index().rename(columns={"index": "module_col"})
lodo_sum["module"] = lodo_sum.module_col.str.replace("M", "").astype(int)
lodo_sum = lodo_sum[["module", "mean", "min", "max"]]
lodo_sum.columns = ["module", "lodo_mean_z", "lodo_min_z", "lodo_max_z"]
lodo_sum[["lodo_mean_z", "lodo_min_z", "lodo_max_z"]] = lodo_sum[
    ["lodo_mean_z", "lodo_min_z", "lodo_max_z"]].round(3)
half_r = pd.DataFrame([
    {"module": int(m), "split_half_z_r": round(float(np.corrcoef(d.z_A, d.z_B)[0, 1]), 4)}
    for m, d in half.groupby("module")])
stab = gs_sum.merge(lodo_sum, on="module").merge(
    pres[["module", "observed", "null_mean", "z_cohesion", "p_lower"]], on="module").merge(
    half_r, on="module")
stab = stab[["module", "n_iter", "jaccard_mean", "jaccard_median", "jaccard_q25",
             "jaccard_q75", "jaccard_min", "jaccard_max", "lodo_mean_z", "lodo_min_z",
             "lodo_max_z", "observed", "null_mean", "z_cohesion", "p_lower", "split_half_z_r"]]

proft = prof.copy()
proft.insert(0, "module", proft.index)
proft = proft.reset_index(drop=True).round(4)

write_table(f"{OUT}/TableS3_module_genes_stability.xlsx",
            [("S3_module_genes", t3), ("S3_module_profiles", proft),
             ("S3_stability_summary", stab)],
            widths={"gene": 16, "module": 8})

# ---------------- Table S4 ----------------
go = pd.read_csv(f"{D}/fig2/go_enrichment_full.csv")
t4 = pd.DataFrame({
    "module": go.module, "GO_term": go.Term_clean, "GO_term_full": go.Term,
    "p_value": go["P-value"].apply(lambda v: f"{v:.3e}"),
    "FDR": go["Adjusted P-value"].apply(lambda v: f"{v:.3e}"),
    "odds_ratio": go["Odds Ratio"], "combined_score": go["Combined Score"],
    "n_genes": go.Genes.fillna("").apply(lambda s: len([x for x in str(s).split(";") if x])),
    "genes": go.Genes,
}).sort_values(["module", "FDR"])

dgo = pd.read_csv(f"{D}/fig3/deg_go_by_type.csv")
t4b = pd.DataFrame({
    "cell_type": dgo.cell_type, "GO_term": dgo.Term_clean, "GO_term_full": dgo.Term,
    "p_value": dgo["P-value"].apply(lambda v: f"{v:.3e}"),
    "FDR": dgo["Adjusted P-value"].apply(lambda v: f"{v:.3e}"),
    "odds_ratio": dgo["Odds Ratio"], "combined_score": dgo["Combined Score"],
    "n_genes": dgo.Genes.fillna("").apply(lambda s: len([x for x in str(s).split(";") if x])),
    "genes": dgo.Genes,
}).sort_values(["cell_type", "FDR"])

# matched-background sensitivity sheet (Fig.3B inset note)
m2 = go[(go.module == 2) & (go["Adjusted P-value"] < 0.05)].copy()
du = pd.read_csv(f"{D}/fig3/deg_go_union.csv")
dum = pd.read_csv(f"{D}/fig3/deg_go_union_matched.csv")
du = du.drop_duplicates("Term_clean").set_index("Term_clean")
dum = dum.drop_duplicates("Term_clean").set_index("Term_clean")
t4c = pd.DataFrame({
    "module": m2.module,
    "GO_term": m2.Term_clean,
    "GO_term_full": m2.Term,
    "module_FDR_bg1353": m2["Adjusted P-value"].round(6),
    "deg_FDR_bg36601": m2.Term_clean.map(du["Adjusted P-value"]).round(6),
    "deg_FDR_bg1353": m2.Term_clean.map(dum["Adjusted P-value"]).round(6),
    "module_sig_bg1353": m2["Adjusted P-value"] < 0.05,
    "deg_sig_bg36601": m2.Term_clean.map(du["Adjusted P-value"]) < 0.05,
    "deg_sig_bg1353": m2.Term_clean.map(dum["Adjusted P-value"]) < 0.05,
    "background_module_n": 1353,
    "background_deg_n": 36601,
    "background_deg_sensitivity_n": 1353,
    "genes": m2.Genes,
    "note": ("Fig.3B heatmap uses the 36,601-gene background for both module "
             "and DEG columns; the 1,353-gene matched background is shown here "
             "as a sensitivity analysis (0/6 DEG terms significant)."),
}).sort_values("module_FDR_bg1353")

write_table(f"{OUT}/TableS4_GO_enrichment.xlsx",
            [("S4_module_GO", t4), ("S4_DEG_GO_by_type", t4b),
             ("S4_matched_background", t4c)],
            widths={"GO_term": 40, "GO_term_full": 55, "genes": 50, "note": 70})

# ---------------- Table S5 ----------------
czs = pd.read_csv(f"{D}/fig2/combined_z.csv").set_index("gene")["combined_z"]
modmap = mods.set_index("gene").module
bridge = pd.read_csv(f"{D}/figS/s8_bridge_genes.csv")
bridge_set = set(zip(bridge.cell_type, bridge.gene))
deg_idx = deg.set_index(["cell_type", "names"])
rows = []
for ct in CT:
    z = Z[ct].dropna()
    div = z.abs().sort_values(ascending=False).index.tolist()[:50]
    for rank, g in enumerate(div, 1):
        dp = deg_idx.pvals_adj.get((ct, g), np.nan)
        lfc = deg_idx.logfoldchanges.get((ct, g), np.nan)
        rows.append({
            "cell_type": ct, "rank": rank, "gene": g,
            "divergence_z": round(float(z.loc[g]), 4),
            "abs_z": round(abs(float(z.loc[g])), 4),
            "network_only": bool(pd.isna(dp) or dp > 0.05),
            "DEG_tested": bool(not pd.isna(dp)),
            "DEG_padj": "" if pd.isna(dp) else f"{dp:.3e}",
            "DEG_log2FC": "" if pd.isna(lfc) else round(float(lfc), 4),
            "combined_z": "" if g not in czs.index else round(float(czs[g]), 4),
            "module": "" if g not in modmap.index else int(modmap[g]),
            "cao_bridge": (ct, g) in bridge_set,
        })
t5 = pd.DataFrame(rows)
t5n = t5[t5.network_only].reset_index(drop=True)
write_table(f"{OUT}/TableS5_network_only_candidates.xlsx",
            [("S5_top50_divergence", t5), ("S5_network_only", t5n)],
            widths={"cell_type": 14, "gene": 16})

print("TABLES_DONE", flush=True)
