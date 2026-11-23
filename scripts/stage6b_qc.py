
#!/usr/bin/env python3
"""Stage 6b QC: split-half reproducibility of the divergence signal (Stromal).

Independent donor halves are processed through the identical MAGIC/correlation
pipeline; gene-level divergence vectors and gene-pair Delta z matrices are then
correlated across halves. Within-group split-half vectors serve as a negative
control (they should not correlate with the cross-condition signal).
"""
import os
import numpy as np
import pandas as pd
import magic

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

SEED = 20261117
D = os.path.join(_ROOT, "data", "fig2")
d = np.load(f"{D}/Stromal.npz", allow_pickle=True)
X = d["X"].astype(float)
genes = d["genes"].astype(str)
group = d["group"].astype(str)
rng = np.random.default_rng(SEED)
r = np.where(group == "RIF")[0]
f = np.where(group == "Fertile")[0]
rng.shuffle(r); rng.shuffle(f)


def fz(Z):
    sd = Z.std(0)
    sd[sd < 1e-8] = np.inf
    Z = (Z - Z.mean(0)) / sd
    return np.arctanh(np.clip(Z.T @ Z / (Z.shape[0] - 1), -0.999, 0.999))


def impute(part):
    df = pd.DataFrame(X[part], columns=genes)
    return np.asarray(magic.MAGIC(knn=5, n_pca=50, t=3, random_state=SEED,
                                  verbose=False).fit_transform(df), float)


def svec(dz):
    A = np.abs(dz)
    np.fill_diagonal(A, -np.inf)
    s = np.zeros(len(genes))
    for i in range(len(genes)):
        p = np.argpartition(A[i], -50)[-50:]
        w = np.abs(dz[i, p])
        sw = w.sum()
        s[i] = np.sum(w * dz[i, p]) / sw if sw > 1e-12 else 0.0
    return s


Fr = [fz(impute(r[:750])), fz(impute(r[750:]))]
Ff = [fz(impute(f[:750])), fz(impute(f[750:]))]
dzA, dzB = Fr[0] - Ff[0], Fr[1] - Ff[1]
sA, sB = svec(dzA), svec(dzB)
iu = np.triu_indices(len(genes), k=1)
sel = rng.choice(len(iu[0]), 500000, replace=False)
Ffull = [fz(impute(r)), fz(impute(f))]
dzF = Ffull[0] - Ffull[1]

out = pd.DataFrame([
    {"metric": "gene_level_s_cross_half_r", "value": float(np.corrcoef(sA, sB)[0, 1])},
    {"metric": "gene_pair_dz_cross_half_r", "value": float(np.corrcoef(dzA[iu][sel], dzB[iu][sel])[0, 1])},
    {"metric": "full_vs_half_pair_dz_r", "value": float(np.corrcoef(dzF[iu][sel], dzA[iu][sel])[0, 1])},
    {"metric": "full_vs_half_gene_s_r", "value": float(np.corrcoef(svec(dzF), sA)[0, 1])},
    {"metric": "within_RIF_split_s_vs_cross_sB_r", "value": float(np.corrcoef(svec(Fr[0] - Fr[1]), sB)[0, 1])},
    {"metric": "within_FER_split_s_vs_cross_sB_r", "value": float(np.corrcoef(svec(Ff[0] - Ff[1]), sB)[0, 1])},
    {"metric": "mean_abs_dz_cross_half_A", "value": float(np.abs(dzA).mean())},
    {"metric": "mean_abs_dz_full_RIF_vs_FER", "value": float(np.abs(dzF).mean())},
])
out.to_csv(f"{D}/qc_split_half.csv", index=False)
print(out.to_string(index=False))
print("QC_DONE")
