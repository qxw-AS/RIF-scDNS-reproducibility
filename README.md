# Reproducibility package — Figures 1–3, Figure S1 and the M2 sensitivity checks

**Version 1.0.0** · Companion to: Li T, Qiu X. *Cell-type-specific gene-coordination divergence in the LH+7 endometrium of women with recurrent implantation failure* (submitted, 2026).

This folder contains the source data, plotting scripts and environment needed to regenerate **Fig. 1, Fig. 2, Fig. 3 and Fig. S1** from the delivered tables. Every table these four figure scripts read is included, and the package has been run end to end against the submitted figures (see *Verification*).

The package certifies those four figures and the module/cut-height re-derivation only. Scripts that need inputs we do not ship are labelled in *What runs out of the box* and are included for traceability, not for standalone execution.

## Layout

```
data/fig1/   umap.csv, panelC_marker_dot.csv, panelD_selected.csv, panelD_deg_q.csv,
             pseudobulk_sensitivity.csv, celltype_group_counts.csv, meta.json
data/fig2/   combined_z.csv, gene_z_by_type.csv, gene_modules.csv, module_profiles.csv,
             go_enrichment_full.csv, go_enrichment_full_allgene.csv, go_module_matrix.csv,
             module_cutheight_sensitivity.csv, ...
data/fig3/   split_half*.csv, jaccard_curves.csv, jaccard_global.csv, deg_go_union*.csv,
             deg_go_by_type.csv, go_matched_compare.csv, ...
data/figS/   s1_*.csv (cohort counts, QC, composition, cluster × donor matrix, resolution)
scripts/     56 .py files in three tiers (see *What runs out of the box*):
             [runs here]   fig1.py  fig2.py  fig3.py  figS1.py  fig_style.py
                           stage6c_modules.py  stage6c_cutheight_sensitivity.py
             [plot only]   figS2.py … figS11.py (10 files) — source data not shipped
             [archival]    stage1_qc.py … stage12_external_network.py (39 files),
                           make_tables.py, export_production.py,
                           download_gse248784.py, inspect_clusters.py
output/      created at run time by the figure scripts; not tracked, because the
             published versions live in the manuscript's figure files
environment.txt   Python 3.11.17 and the package versions used for the analysis
```

## What runs out of the box

| Tier | Files | What it produces | Inputs shipped |
|---|---|---|---|
| Runs | `fig1.py` `fig2.py` `fig3.py` `figS1.py` (+ `fig_style.py`) | the four certified figures | yes, in `data/` |
| Runs | `stage6c_modules.py` `stage6c_cutheight_sensitivity.py` | the M1–M8 partition and the cut-height grid (Table S5, sheet `S5_cutheight`) | yes, in `data/fig2/` |
| Plot only | `figS2.py` … `figS11.py` | Fig. S2–S11 | **no** — the source tables for these panels are not part of this package, so they stop at the first read |
| Archival | `stage1_qc.py` … `stage12_external_network.py`, `make_tables.py`, `export_production.py`, `download_gse248784.py`, `inspect_clusters.py` | the analysis code behind every derived table | **no** — needs the raw GEO matrices and the intermediate objects built from them |

Paths resolve from the package root in all 56 scripts, so none is tied to a particular machine. `export_production.py` additionally needs Ghostscript (`gs`) and `pdftoppm`.

## Running the scripts

```bash
python scripts/fig1.py     # Fig. 1 panels A–D
python scripts/fig2.py     # Fig. 2 panels A–D
python scripts/fig3.py     # Fig. 3 panels A–C
python scripts/figS1.py    # Fig. S1 panels A–E
python scripts/stage6c_cutheight_sensitivity.py   # Table S5 sheet S5_cutheight
```

Optional overrides: `RIF_ROOT` (package root), `RIF_DATA` (figure-specific data directory), `RIF_DATA1` / `RIF_DATA2` / `RIF_DATA3` (fig1 / fig2 / fig3 data where a script crosses directories) and `RIF_FIG` (output directory). `fig_style.py` must stay alongside the plotting scripts.

The journal requires the legends in a separate file, so no in-figure caption is drawn by default; set `RIF_CAPTION=1` to add the caption line that was used while drafting.

## Verification

Run on 2026-11-19 from this directory. Text extracted from each generated PDF was compared line by line with the corresponding production PDF in `02_Figures/` and `04_Supplementary/`; the regenerated figures are not committed here, so compare them against the published Figures 1–3 and Fig. S1.

| Figure | Result |
|---|---|
| Fig. 1 | 155 / 155 text lines identical |
| Fig. 2 | 95 / 95 text lines identical, including the panel-B annotation |
| Fig. 3 | identical except the removed chance band in panel C (item 5 below) |
| Fig. S1 | identical except one axis-label line-break artefact in the text layer |
| Module partition | `stage6c_modules.py` reproduced the published partition exactly: h = 9.821, 8 modules, 100% label agreement, 46/46 genes in M2 |
| Cut-height sensitivity | `stage6c_cutheight_sensitivity.py` reproduced the reported values: M2 intact (Jaccard = 1.00, HLA-A/B/C co-clustered) at all 122 grid points from h = 7.95 to 14.00 |

Page widths of the regenerated PDFs (178.5–183.2 mm) are the analysis-export widths; `export_production.py` scales them to the submitted 170 mm and rasterises the 300 dpi TIFF.

Re-verified on 2026-11-23 from a clean copy of this repository, under a deliberately different environment (Python 3.10.4, numpy 1.26.4, pandas 2.3.3, matplotlib 3.10.9, scipy 1.15.3, Windows): all four PNGs reproduced **byte for byte** (MD5 identical), the four PDFs kept identical file sizes with only the embedded creation timestamps differing, `stage6c_modules.py` returned h = 9.821 with 8 modules and 46 genes in M2, and the cut-height grid returned M2 intact (Jaccard = 1.00, HLA-A/B/C co-clustered) at all 122 points from h = 7.95 to 14.00.

## Changes relative to the v2.1 bundle

1. **Input paths parametrised in all 56 scripts.** The scripts previously read `/workspace/rif_fig1/...`; every path is now resolved from the package location, with `RIF_ROOT` / `RIF_DATA*` / `RIF_FIG` as overrides. No machine-specific path remains anywhere in `scripts/`.
2. **`scripts/fig2.py` added.** The v2.1 bundle contained no Fig. 2 script. Its panel-B annotation was corrected to the value plotted, `FDR = 0.005 (testable-gene background, n = 1,353)`; the analysis-workspace copy still carried the superseded `(all-gene background)` string.
3. **`data/fig1/panelD_deg_q.csv` replaces the 43 MB `panelD_deg_full.csv` for plotting.** It holds the five columns the scripts use (`names`, `cell_type`, `logfoldchanges`, `pct_rif`, `pvals_adj`) for the 1,371 genes drawn in Fig. 1D and Fig. 2C, extracted from the full table (402,611 rows). The full table is available from the corresponding author.
4. **`data/fig3/go_overlap_matrix.csv` removed.** Its module column carried FDRs computed on the 1,353-gene background while its DEG column carried values computed on the 36,601-gene background, so the two columns were not comparable. Fig. 3B is drawn from `data/fig2/go_enrichment_full_allgene.csv`, which is included here.
5. **Cut-height sensitivity added (Table S5, sheet `S5_cutheight`).** `stage6c_cutheight_sensitivity.py` re-derives the Ward partition over h = 5.00–16.00 in 0.05 steps (221 grid points) and records, at each height, the number of modules, the size of the cluster best matching M2, its Jaccard overlap with the published 46-gene M2, and whether HLA-A/B/C stay co-clustered. M2 is recovered intact at every height from 7.95 to 14.00 (the module count ranges from 5 to 9 over that window), fragments below 7.95, and absorbs neighbouring clusters at 15 and above.
6. **Fig. 3C no longer draws the hypergeometric chance band.** The divergence list is restricted to the per-type testable pool (642–1,258 genes) whereas the DEA list spans 36,601 genes, so a null computed on a single universe is not exchangeable with the observed overlap and the band invited a calibration the design does not support. The null is still present in `data/fig3/jaccard_curves.csv` (`null_mean`, `null_lo`, `null_hi`) and is not used for inference.

## Notes

- Random seed 20261117 throughout; the horizontal jitter in Fig. 3C uses seed 7.
- `export_production.py` regenerates the 170 mm vector PDFs and 300 dpi RGB TIFFs and requires Ghostscript (`gs`) and `pdftoppm`.
- `gseapy` is needed only to re-run `stage7b/7d/7e`, not the four figure scripts. Enrichr libraries are read from a local cache; the GMT files shipped in the full delivery (`09_环境/gseapy_cache/`) must be copied to `~/.cache/gseapy/`.
- The analysis stages (`stage1_qc.py` … `stage12_external_network.py`) are the code that produced every derived table, included so that each can be traced to its source. They resolve their paths from the package root, but still expect the raw GEO matrices and the intermediate objects built from them, which are not shipped; they are therefore not runnable from this bundle alone. The scripts listed under *What runs out of the box* are. Entry points for the analyses referred to in the text: `stage6c_modules.py` (Ward clustering, cut height and small-module merging that define M1–M8), `stage7a_split_half.py` (split-half correlations), `stage9a_donor_levels.py` (donor-level pseudobulk DEA), `stage9b_donor_equal_split.py` (donor-balanced split), `stage7e_go_allgene.py` (36,601-gene background).
- **Table S7 inputs are intentionally outside this bundle.** The leave-one-donor-out GO results (`lodo_summary.csv`, `lodo_module_go.csv`, and the 32,000-row per-donor combined-z long table) and the iLISI/cLISI/kBET outputs were recomputed outside this package and are reported in `04_Supplementary/TableS7_LODO_integration.xlsx`. The LODO procedure: omit one donor per fold, recompute within-type divergence z on the remaining 15 donors, combine across types by Stouffer's method, recluster the fixed 456 full-sample genes under both the fixed cut (h = 9.82) and the adaptive 8–12 module rule, then run GO on the local GMT with the background fixed at 1,353 genes (which reproduces the full-sample M2 FDR of 0.004957). The integration diagnostics used the Harmony embedding of 149,590 annotated cells with iLISI/cLISI at k = 90 (pynndescent) and kBET at k = 50 (chi-square, alpha = 0.05). These outputs are reported only in `04_Supplementary/TableS7_LODO_integration.xlsx` and, consistently with the manuscript's data-availability statement, are available from the corresponding author on reasonable request. They are not required to regenerate Figures 1, 2, 3 or S1, which are the four figures this package certifies.

- **Table S8 (no-imputation control) inputs are also outside this bundle.** The no-imputation control reported in Table S8 was computed on the same 1,353-gene panel and the frozen 456-gene set, with the MAGIC step omitted and all downstream settings fixed (Ward cut h = 9.82, local GMT, 1,353-gene background, seed 20261117). Only summary values were supplied: top-50 Jaccard median 0.250 (range 0.124–0.429), combined-z Spearman rho = 0.772, M2 not recovered under the fixed 456-gene rule (best-matching 114-gene module MHC-I FDR = 0.467), the re-selected 376-gene sensitivity module (82 genes, MHC-I FDR = 0.0479, Jaccard 0.267 with M2), and MAGIC/no-imputation inflation of mean |Fisher z| (3.85–5.64-fold) and mean |Delta z| (3.73–6.08-fold). Per-type Jaccard/rho tables and per-donor long tables were not provided and, consistently with the manuscript data-availability statement, are available from the corresponding author on reasonable request. Table S8 is not required to regenerate Figures 1, 2, 3 or S1. See also refs 19 and 20 in the manuscript (Andrews & Hemberg 2018; Ly & Vingron 2022).
