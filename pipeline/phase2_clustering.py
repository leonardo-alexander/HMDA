"""Phase 2 clustering logic extracted from HMDA.ipynb.

Narrative markdown, dendrogram/PCA/silhouette plots, and validation cells stay
in the notebook; only the data-transformation and algorithm code lives here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from . import config as cfg


def add_engineered_flags(clean):
    """Cell 55: engineered behavioural flags + Phase 2-4 feature-role lists.

    Mutates `clean` in place (adds the `_is_*` flag columns) and returns the
    role lists the rest of Phase 2 (and the leakage guard) depend on.
    """
    ANOMALY_FEATS = [c for c in [
        "income", "loan_amount", "property_value", "combined_loan_to_value_ratio",
        "loan_term", "tract_minority_population_percent", "tract_to_msa_income_percentage",
        "ffiec_msa_md_median_family_income",
    ] if c in clean.columns]

    clean["_is_investment"] = (clean.get("occupancy_type").astype(str) == "Investment").astype(float) \
        if "occupancy_type" in clean.columns else 0.0
    clean["_is_refinance"] = clean.get("loan_purpose").astype(str).isin(["Refinance", "CashOut_Refinance"]).astype(float) \
        if "loan_purpose" in clean.columns else 0.0
    clean["_is_manufactured"] = (clean.get("construction_method").astype(str) == "Manufactured").astype(float) \
        if "construction_method" in clean.columns else 0.0
    clean["_is_subordinate"] = (clean.get("lien_status").astype(str) == "Subordinate_Lien").astype(float) \
        if "lien_status" in clean.columns else 0.0
    clean["_is_high_dti"] = clean.get("debt_to_income_ratio").astype(str).isin(["50%-60%", ">60%"]).astype(float) \
        if "debt_to_income_ratio" in clean.columns else 0.0

    CLUSTER_CONT_PART = [c for c in ["income", "combined_loan_to_value_ratio",
                                     "tract_minority_population_percent", "tract_to_msa_income_percentage"]
                         if c in clean.columns]
    CLUSTER_FLAGS = ["_is_investment", "_is_refinance", "_is_manufactured", "_is_subordinate", "_is_high_dti"]
    CLUSTER_FEATS = CLUSTER_CONT_PART + CLUSTER_FLAGS

    return clean, ANOMALY_FEATS, CLUSTER_CONT_PART, CLUSTER_FLAGS, CLUSTER_FEATS


def item_features_for_rules(appdeny):
    """Cell 55: association-rule item features (available at application time)."""
    return [c for c in [
        "derived_race", "derived_ethnicity", "derived_sex", "applicant_age",
        "loan_type", "loan_purpose", "lien_status", "occupancy_type", "construction_method",
        "preapproval", "conforming_loan_limit", "total_units",
        "income_band", "loan_amount_band", "property_value_band", "cltv_band",
        "debt_to_income_ratio", "tract_income_cat", "tract_minority_cat",
    ] if c in appdeny.columns]


def winsorize(s, lo=0.01, hi=0.99):
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(ql, qh)


def prepare_matrices(clean, cluster_cont_part, cluster_flags, cluster_feats, anomaly_feats,
                     random_state=cfg.RANDOM_STATE):
    Xc_cont = clean[cluster_cont_part].astype(float).apply(winsorize)
    Xc_in = pd.concat([Xc_cont, clean[cluster_flags].astype(float)], axis=1)
    Xc = pd.DataFrame(StandardScaler().fit_transform(Xc_in),
                      columns=cluster_feats, index=clean.index).astype("float32")

    Xa = pd.DataFrame(RobustScaler().fit_transform(clean[anomaly_feats].astype(float)),
                      columns=anomaly_feats, index=clean.index).astype("float32")

    rng = np.random.default_rng(random_state)
    N = len(clean)
    idx_med = rng.choice(N, size=min(20000, N), replace=False)
    idx_small = rng.choice(idx_med, size=min(4000, len(idx_med)), replace=False)

    return Xc, Xa, idx_med, idx_small


def knee_point(x, y):
    """Normalize both axes, then pick the point farthest from the chord
    connecting the two endpoints. Shared by K selection and DBSCAN eps selection."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xn = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
    yn = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else np.zeros_like(y)
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    pts = np.c_[xn, yn]
    distances = np.abs(np.cross(p2 - p1, pts - p1)) / np.linalg.norm(p2 - p1)
    return int(x[np.argmax(distances)])


def select_k(Xc, k_range=range(2, 11), random_state=cfg.RANDOM_STATE):
    from sklearn.metrics import silhouette_score

    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(Xc)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(
            Xc, labels, sample_size=min(10000, len(Xc)), random_state=random_state
        ))

    elbow_k = knee_point(list(k_range), inertias)
    sil_k = list(k_range)[int(np.argmax(sils))]

    k_selection = pd.DataFrame({"K": list(k_range), "inertia": inertias, "silhouette": sils})
    k_selection["inertia_reduction_pct"] = (-k_selection["inertia"].pct_change() * 100).round(2)

    best_k = sil_k
    return elbow_k, sil_k, best_k, k_selection


def run_kmeans(Xc, k, random_state=cfg.RANDOM_STATE):
    return KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(Xc.values).astype(int)


def run_dbscan(Xc, idx_med, cluster_feats):
    min_samples = 2 * len(cluster_feats)
    Xc_med = Xc.iloc[idx_med].values

    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(Xc_med)
    kdist = np.sort(nbrs.kneighbors(Xc_med)[0][:, -1])
    eps = float(kdist[knee_point(np.arange(len(kdist)), kdist)])

    db_labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(Xc_med).astype(int)
    dbscan_noise_ids = set(np.array(idx_med)[db_labels == -1].tolist())

    return db_labels, dbscan_noise_ids, eps, min_samples, kdist


def run_hierarchical(Xc, idx_small, best_k):
    Xs = Xc.iloc[idx_small].values
    Zward = linkage(Xs, method="ward")
    hier_labels = fcluster(Zward, t=best_k, criterion="maxclust")
    return hier_labels, Zward


def name_cluster(row):
    if row["pct_investment"] >= 80:
        return "Property investors"
    if row["pct_high_dti"] >= 80:
        return "DTI-stressed borrowers"
    if row["med_income"] >= 300:
        return "Jumbo / high-net-worth buyers"
    if row["pct_manufactured"] >= 50:
        return "Manufactured-housing applicants"
    if row["med_loan"] < 100_000:
        return "Small-loan borrowers"
    if row["pct_refinance"] >= 70:
        return "Refinancers (rate & cash-out)"
    if row["approval_rate"] >= 85:
        return "Mainstream prime purchasers"
    return "Moderate-profile purchasers"


def profile_clusters(clean, cluster_col="kmeans_cluster"):
    """Cell 74: cluster profiling + business naming.

    Mutates `clean` in place (adds `_approved`), matching the original cell so
    later cells that expect `clean["_approved"]` to already exist keep working.
    """
    clean["_approved"] = np.nan
    if "action_taken" in clean.columns:
        clean.loc[clean["action_taken"] == "Originated", "_approved"] = 1
        clean.loc[clean["action_taken"] == "Denied", "_approved"] = 0

    profile = clean.groupby(cluster_col).agg(
        n=(cluster_col, "size"),
        approval_rate=("_approved", "mean"),
        med_income=("income", "median"),
        med_loan=("loan_amount", "median"),
        med_cltv=("combined_loan_to_value_ratio", "median"),
        pct_investment=("_is_investment", "mean"),
        pct_refinance=("_is_refinance", "mean"),
        pct_high_dti=("_is_high_dti", "mean"),
        pct_manufactured=("_is_manufactured", "mean"),
        med_tract_minority=("tract_minority_population_percent", "median"),
    )
    for col in ["approval_rate", "pct_investment", "pct_refinance", "pct_high_dti", "pct_manufactured"]:
        profile[col] = (profile[col] * 100).round(1)
    profile["share_of_data"] = (profile["n"] / len(clean) * 100).round(1)
    profile["segment_name"] = profile.apply(name_cluster, axis=1)

    return clean, profile


def clarans(X, k, numlocal=2, maxneighbor=None, random_state=cfg.RANDOM_STATE):
    """Vectorized CLARANS (Ng & Han, 1994): k-medoids via randomized search,
    with the full pairwise distance matrix computed once up front."""
    rs = np.random.default_rng(random_state)
    n = len(X)
    if maxneighbor is None:
        maxneighbor = min(200, max(50, int(0.0125 * k * (n - k))))
    D = squareform(pdist(X))
    cost = lambda m: D[:, m].min(axis=1).sum()
    best_m, best_c, cap, evals = None, np.inf, numlocal * maxneighbor * 30, 0
    for _ in range(numlocal):
        cur = rs.choice(n, size=k, replace=False)
        cc = cost(cur)
        j = 0
        while j < maxneighbor and evals < cap:
            evals += 1
            pos = int(rs.integers(k))
            inm = set(cur.tolist())
            cand = int(rs.integers(n))
            while cand in inm:
                cand = int(rs.integers(n))
            nb = cur.copy()
            nb[pos] = cand
            nc = cost(nb)
            if nc < cc:
                cur, cc, j = nb, nc, 0
            else:
                j += 1
        if cc < best_c:
            best_c, best_m = cc, cur.copy()
    return best_m, D[:, best_m].argmin(axis=1), best_c


def run_clarans(Xc, idx_small, k, random_state=cfg.RANDOM_STATE):
    Xs_cl = Xc.iloc[idx_small].values
    medoid_ids, clarans_labels, clarans_cost = clarans(Xs_cl, k=k, random_state=random_state)
    return clarans_labels, medoid_ids, clarans_cost


def export_phase2(clean, profile, idx_small, clarans_labels, out_dir):
    out_dir = Path(out_dir)
    clean[["kmeans_cluster"]].to_csv(out_dir / "p2_cluster_assignments.csv", index=True)
    profile.to_csv(out_dir / "p2_cluster_profiles.csv")
    pd.DataFrame({"row": np.array(idx_small), "clarans_cluster": clarans_labels}) \
        .to_csv(out_dir / "p2_clarans_assignments.csv", index=False)
