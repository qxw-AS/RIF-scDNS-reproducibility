
#!/usr/bin/env python3
"""Donor-equalized split-half sensitivity for the Stromal network.

The published donor-level split (Fig. 3A sensitivity) subsamples each half to a
fixed total cell number but keeps donor proportions, so a few large donors can
dominate. Here every donor contributes the same number of cells to each half,
which tests whether the low donor-level reproducibility is explained by cell
number imbalance.
"""
import os
import numpy as np
import pandas as pd
import magic

# Paths resolve relative to this package; override with RIF_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get('RIF_ROOT', os.path.dirname(_HERE))

SEED = 20261117
DATA = os.path.join(_ROOT, "data", "fig2")
OUT = os.path.join(_ROOT, "data", "fig3")
CT = "Stromal"
NP_DONOR = 150
NS = [50, 100, 200]
N_PERM = 1000
TOP_K = 50
MIN_DET = 0.05

d = np.load(f"{DATA}/{CT}.npz", allow_pickle=True)
X = d["X"].astype(np.float64)
genes = d["genes"].astype(str)
sample = d["sample"].astype(str)
group = d["group"].astype(str)

deg = pd.read_csv(os.path.join(_ROOT, "data", "fig1", "panelD_deg_full.csv"))
deg_sig = (deg[deg.cell_type == CT].set_index("names").reindex(genes).pvals_adj <= 0.05)
deg_sig = deg_sig.fillna(False).to_numpy()


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
    k = min(TOP_K, A.shape[1] - 1)
    s = np.zeros(len(genes))
    for i in range(len(genes)):
        p = np.argpartition(A[i], -k)[-k:]
        w = np.abs(dz[i, p])
        sw = w.sum()
        s[i] = np.sum(w * dz[i, p]) / sw if sw > 1e-12 else 0.0
    return s


def half_state(idx_rif, idx_fer):
    imp = {"RIF": impute(idx_rif), "Fertile": impute(idx_fer)}
    det = {"RIF": (X[idx_rif] > 0).mean(0), "Fertile": (X[idx_fer] > 0).mean(0)}
    testable = (det["RIF"] >= MIN_DET) & (det["Fertile"] >= MIN_DET)
    dz = fz(imp["RIF"]) - fz(imp["Fertile"])
    return {"dz": dz, "s": svec(dz), "testable": testable}


def zscore(v, mask):
    out = np.full(len(v), np.nan)
    x = v[mask]
    out[mask] = (x - x.mean()) / (x.std() if x.std() > 1e-12 else 1.0)
    return out


def equal_donor(idx, donors, n_per):
    picks = []
    for dn in donors:
        cand = idx[sample[idx] == dn]
        if len(cand) == 0:
            continue
        picks.append(np.random.default_rng(SEED + sum(ord(c) for c in dn)).choice(
            cand, size=min(n_per, len(cand)), replace=False))
    return np.concatenate(picks)


rng = np.random.default_rng(SEED)
rows = []
for rep in range(2):
    halves = {}
    assigned = {}
    for g in ("RIF", "Fertile"):
        idx = np.where(group == g)[0]
        donors = np.array(sorted(set(sample[idx])))
        rng.shuffle(donors)
        A, B = donors[::2], donors[1::2]
        halves[(g, "A")] = equal_donor(idx, A, NP_DONOR)
        halves[(g, "B")] = equal_donor(idx, B, NP_DONOR)
        assigned[g] = {"A": list(A), "B": list(B)}
    st = {k: half_state(halves[("RIF", k)], halves[("Fertile", k)]) for k in ("A", "B")}
    pool = st["A"]["testable"] & st["B"]["testable"]
    zA, zB = zscore(st["A"]["s"], pool), zscore(st["B"]["s"], pool)
    nonsig = (~deg_sig) & pool
    gene_r = float(np.corrcoef(zA[pool], zB[pool])[0, 1])
    iu = np.triu_indices(len(genes), k=1)
    sel = rng.choice(len(iu[0]), 300000, replace=False)
    pair_r = float(np.corrcoef(st["A"]["dz"][iu][sel], st["B"]["dz"][iu][sel])[0, 1])
    print(f"rep{rep}: cells {len(halves[('RIF','A')])}/{len(halves[('RIF','B')])} "
          f"{len(halves[('Fertile','A')])}/{len(halves[('Fertile','B')])}, pool {int(pool.sum())}, "
          f"gene_r {gene_r:.3f}, pair_r {pair_r:.3f}", flush=True)
    for N in NS:
        topA = np.argsort(-np.abs(np.where(pool, zA, np.nan)))[:N]
        topB = np.argsort(-np.abs(np.where(pool, zB, np.nan)))[:N]
        netA = set(topA[nonsig[topA]]) if np.isfinite(zA[topA]).all() else set()
        netB = set(topB[nonsig[topB]]) if np.isfinite(zB[topB]).all() else set()
        obs_net = len(netA & netB) / len(netA | netB) if netA | netB else np.nan
        null = []
        for _ in range(N_PERM):
            a = set(rng.choice(np.where(nonsig)[0], len(netA), replace=False))
            b = set(rng.choice(np.where(nonsig)[0], len(netB), replace=False))
            null.append(len(a & b) / len(a | b))
        rows.append({"rep": rep, "N": N, "kind": "network_only",
                     "setA": len(netA), "setB": len(netB), "shared": len(netA & netB),
                     "observed": obs_net, "null_mean": float(np.mean(null)),
                     "p_upper": float((1 + (np.asarray(null) >= obs_net).sum()) / (1 + len(null))),
                     "n_rif_A": len(halves[("RIF", "A")]), "n_rif_B": len(halves[("RIF", "B")]),
                     "n_fer_A": len(halves[("Fertile", "A")]), "n_fer_B": len(halves[("Fertile", "B")]),
                     "pool": int(pool.sum()), "gene_r": gene_r, "pair_r": pair_r})
    pd.DataFrame([{"rep": rep, "scheme": "donor_equal_150", "gene_level_r": gene_r,
                   "gene_pair_dz_r": pair_r, "common_testable_genes": int(pool.sum()),
                   "network_only_pool": int(nonsig.sum()),
                   "donors_RIF_A": ";".join(map(str, assigned["RIF"]["A"])),
                   "donors_RIF_B": ";".join(map(str, assigned["RIF"]["B"])),
                   "donors_FER_A": ";".join(map(str, assigned["Fertile"]["A"])),
                   "donors_FER_B": ";".join(map(str, assigned["Fertile"]["B"]))}]).to_csv(
        f"{OUT}/split_half_donor_equal_metrics_rep{rep}.csv", index=False)

pd.DataFrame(rows).to_csv(f"{OUT}/split_half_donor_equal_jaccard.csv", index=False)
print("STAGE9B_DONE")
