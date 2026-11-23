
#!/usr/bin/env python3
"""Stage 7a: split-half reproducibility of network-only genes (Fig.3A).

Two split schemes are run with the identical Fig.2 approximation pipeline
(MAGIC knn=5 n_pca=50 t=3 -> Pearson -> Fisher z -> Delta z -> gene-level
signed divergence s -> within-half z-score):

  PRIMARY  cell-level random split (the pre-specified "sample-internal
           split-half"): each group's cells are randomly divided 750/750, so
           both halves share all donors. This measures within-donor /
           technical reproducibility.
  SENSITIVITY donor-level balanced split: donors are allocated to halves and
           cells subsampled to equal counts. The halves share no donors, so
           this measures cross-donor reproducibility.

"network-only" = top-N |z| gene AND Stromal DEG pvals_adj > 0.05 in the fixed
full-data Fig.1D Wilcoxon reference (the DEG list is a fixed annotation; only
the network is split). Jaccard is reported for N = 50/100/200 (the common
testable pool of a half is ~600 genes, so larger N saturates).

Outputs (data/fig3/)
  split_half_gene_z.csv          primary per-gene z + flags
  split_half_jaccard.csv         primary observed / null Jaccard vs N
  split_half_metrics.csv         primary scalar metrics
  split_half_donor_*             donor-level sensitivity equivalents
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
NS = [50, 100, 200]
N_PERM = 2000
TOP_K = 50
MIN_DET = 0.05
rng = np.random.default_rng(SEED)

d = np.load(f"{DATA}/{CT}.npz", allow_pickle=True)
X = d["X"].astype(np.float64)
genes = d["genes"].astype(str)
sample = d["sample"].astype(str)
group = d["group"].astype(str)

# ------------------------- DEG reference -------------------------
deg = pd.read_csv(os.path.join(_ROOT, "data", "fig1", "panelD_deg_full.csv"))
dct = deg[deg.cell_type == CT].set_index("names").reindex(genes)
deg_sig = (dct.pvals_adj <= 0.05).fillna(False).to_numpy()


def balanced_donor_split(sub_idx):
    s = pd.Series(sample[sub_idx])
    cnt = s.value_counts().sort_values(ascending=False)
    halves = {"A": [], "B": []}
    for i, dn in enumerate(cnt.index):
        halves["A" if i % 2 == 0 else "B"].append(dn)
    return {k: sub_idx[np.isin(sample[sub_idx], v)] for k, v in halves.items()}, halves


def subsample_stratified(idx, n_target):
    s = pd.Series(sample[idx])
    cnt = s.value_counts()
    donors, c = cnt.index.to_numpy(), cnt.to_numpy()
    if n_target >= len(idx):
        return idx
    props = c / c.sum() * n_target
    alloc = np.minimum(np.maximum(np.floor(props).astype(int), 1), c)
    while alloc.sum() > n_target:
        j = np.argmax(np.where(alloc > 1, alloc, -1))
        if alloc[j] <= 1:
            break
        alloc[j] -= 1
    frac = props - np.floor(props)
    while alloc.sum() < n_target:
        cand = np.where(alloc < c)[0]
        if len(cand) == 0:
            break
        j = cand[np.argmax(frac[cand])]
        alloc[j] += 1
        frac[j] = -1.0
    picks = []
    for dn, k in zip(donors, alloc):
        picks.append(rng.choice(idx[sample[idx] == dn], size=int(k), replace=False))
    return np.concatenate(picks)


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


def run_scheme(name, halves, donors, tag):
    st = {k: half_state(halves[("RIF", k)], halves[("Fertile", k)]) for k in ("A", "B")}
    pool = st["A"]["testable"] & st["B"]["testable"]
    zA, zB = zscore(st["A"]["s"], pool), zscore(st["B"]["s"], pool)
    nonsig = (~deg_sig) & pool
    iu = np.triu_indices(len(genes), k=1)
    sel = rng.choice(len(iu[0]), 500000, replace=False)
    gene_r = float(np.corrcoef(zA[pool], zB[pool])[0, 1])
    pair_r = float(np.corrcoef(st["A"]["dz"][iu][sel], st["B"]["dz"][iu][sel])[0, 1])
    print(f"[{name}] cells {len(halves[('RIF','A')])}/{len(halves[('RIF','B')])} "
          f"{len(halves[('Fertile','A')])}/{len(halves[('Fertile','B')])}, "
          f"pool {int(pool.sum())}, gene_r {gene_r:.3f}, pair_r {pair_r:.3f}", flush=True)

    rows_j = []
    for N in NS:
        topA = np.argsort(-np.abs(np.where(pool, zA, np.nan)))[:N]
        topB = np.argsort(-np.abs(np.where(pool, zB, np.nan)))[:N]
        netA, netB = set(topA[nonsig[topA]]), set(topB[nonsig[topB]])
        obs_net = len(netA & netB) / len(netA | netB)
        obs_all = len(set(topA) & set(topB)) / len(set(topA) | set(topB))
        null_net, null_all = [], []
        for _ in range(N_PERM):
            a = set(rng.choice(np.where(nonsig)[0], len(netA), replace=False))
            b = set(rng.choice(np.where(nonsig)[0], len(netB), replace=False))
            null_net.append(len(a & b) / len(a | b))
            a = set(rng.choice(np.where(pool)[0], N, replace=False))
            b = set(rng.choice(np.where(pool)[0], N, replace=False))
            null_all.append(len(a & b) / len(a | b))
        nn, na = np.asarray(null_net), np.asarray(null_all)
        for kind, obs, null, sA, sB, sh in (
                ("network_only", obs_net, nn, len(netA), len(netB), len(netA & netB)),
                ("all", obs_all, na, len(topA), len(topB), len(set(topA) & set(topB)))):
            rows_j.append({"N": N, "kind": kind, "setA": sA, "setB": sB, "shared": sh,
                           "observed": obs, "null_mean": float(null.mean()),
                           "null_lo": float(np.quantile(null, .025)),
                           "null_hi": float(np.quantile(null, .975)),
                           "p_upper": float((1 + (null >= obs).sum()) / (1 + len(null)))})
        print(f"[{name}] N={N}: network-only J={obs_net:.3f} "
              f"({len(netA)}/{len(netB)}, shared {len(netA & netB)}, null {nn.mean():.3f}); "
              f"all J={obs_all:.3f} (null {na.mean():.3f})", flush=True)

    pd.DataFrame({"gene": genes, "z_A": zA, "z_B": zB, "testable_common": pool,
                  "deg_nonsig": nonsig}).to_csv(f"{OUT}/split_half{tag}_gene_z.csv", index=False)
    pd.DataFrame(rows_j).to_csv(f"{OUT}/split_half{tag}_jaccard.csv", index=False)
    pd.DataFrame([{"cell_type": CT, "scheme": name,
                   "n_rif_half_A": len(halves[("RIF", "A")]),
                   "n_rif_half_B": len(halves[("RIF", "B")]),
                   "n_fer_half_A": len(halves[("Fertile", "A")]),
                   "n_fer_half_B": len(halves[("Fertile", "B")]),
                   "donors_RIF_A": ";".join(donors["RIF"]["A"]),
                   "donors_RIF_B": ";".join(donors["RIF"]["B"]),
                   "donors_FER_A": ";".join(donors["Fertile"]["A"]),
                   "donors_FER_B": ";".join(donors["Fertile"]["B"]),
                   "common_testable_genes": int(pool.sum()),
                   "network_only_pool": int(nonsig.sum()),
                   "gene_level_r": gene_r, "gene_pair_dz_r": pair_r,
                   "n_perm": N_PERM, "seed": SEED}]).to_csv(
        f"{OUT}/split_half{tag}_metrics.csv", index=False)


# ---------------- PRIMARY: cell-level random split ----------------
cell_halves = {}
for g in ("RIF", "Fertile"):
    idx = np.where(group == g)[0].copy()
    rng.shuffle(idx)
    n = min(750, len(idx) // 2)
    cell_halves[(g, "A")], cell_halves[(g, "B")] = idx[:n], idx[n:2 * n]
run_scheme("cell_random", cell_halves,
           {"RIF": {"A": ["shared"], "B": ["shared"]},
            "Fertile": {"A": ["shared"], "B": ["shared"]}}, "")

# ---------------- SENSITIVITY: donor-level split ----------------
don_halves, don_assigned = {}, {}
for g in ("RIF", "Fertile"):
    idx = np.where(group == g)[0]
    h, donors = balanced_donor_split(idx)
    nper = min(len(h["A"]), len(h["B"]))
    for k in ("A", "B"):
        don_halves[(g, k)] = subsample_stratified(h[k], nper)
    don_assigned[g] = donors
run_scheme("donor_balanced", don_halves, don_assigned, "_donor")
print("STAGE7A_DONE", flush=True)
