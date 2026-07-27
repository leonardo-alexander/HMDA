"""Rebuilds every dash_*.csv aggregate (plus the core Phase 1-5 pipeline
outputs) that dashboard_app.py reads, using the pipeline/ modules as the
single source of truth.

Replaces the 9 ad hoc scratchpad scripts used earlier in this project
(replicate.py + build_dash_*.py) that re-derived this data by hand outside
version control. Run this whenever the notebook's pipeline logic changes and
the dashboard's data needs rebuilding:

    python dashboard/build_data.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from pipeline import config as cfg
from pipeline import phase1_preprocessing as p1
from pipeline import phase2_clustering as p2
from pipeline import phase3_association_rules as p3
from pipeline import phase4_anomaly_detection as p4
from pipeline import phase5_reporting as p5

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"
for _d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, TABLE_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = cfg.RANDOM_STATE

MANUAL_RESOLUTION = {
    7727: ("RARE BUT VALID",
           "subordinate lien of $1,005,000 on a $9,205,000 CA principal residence; "
           "CLTV 40.9% x $9.2M = $3.77M total liens >= this loan, so a senior first lien "
           "exists and the figures reconcile; income $758k/yr supports the debt -> "
           "high-net-worth equity draw, not a data error"),
    95313: ("RARE BUT VALID",
            "$35,000 home-improvement second lien on a $2,495,000 CA home; CLTV 19.9% x "
            "$2.5M = $498k total liens >= this loan; every magnitude is plausible -> flagged "
            "only for the tiny-loan-on-expensive-home contrast (and it was denied anyway)"),
    8102: ("RARE BUT VALID",
           "$2,505,000 cash-out first-lien refinance of a $9,305,000 NY second residence; "
           "CLTV 26.9% x $9.3M = $2.50M = the loan itself, i.e. arithmetically "
           "self-consistent; income $1.08M at DTI 36%-<43% fits a jumbo borrower"),
}


def load_raw_with_cache():
    """Uses the cached raw CSV under data/raw/ when present (fast, offline-friendly
    repeat runs); otherwise fetches from HF_URL and caches it for next time."""
    cache_path = DATA_RAW / "hmda_sample.csv"
    if cache_path.exists():
        return p1.load_raw(str(cache_path))
    df = p1.load_raw(cfg.HF_URL)
    df.to_csv(cache_path, index=False)
    return df


def run_phase1():
    df = load_raw_with_cache()
    df = p1.coerce_and_label(df)
    df, df_denials, to_drop = p1.resolve_overlap_and_denials(df)
    df, n_dupes, n_before = p1.drop_duplicate_rows(df)
    df, miss, high_missing, signal_cont, residual, missing_cols = p1.handle_missing(df)
    df, approve_deny = p1.frame_target_and_bin(df)
    corr_target, mi_series, comb, num_feats = p1.score_features(approve_deny, RANDOM_STATE)
    feature_selection_audit, eligible_feature_ranking, process_stage_diagnostics = \
        p1.audit_feature_selection(comb)

    p1.export_phase1(df, approve_deny, df_denials, comb, DATA_INTERIM, TABLE_DIR)
    feature_selection_audit.to_csv(TABLE_DIR / "p1_feature_selection_audit.csv", index=False)
    p5.export_phase1_aggregates(miss, high_missing, feature_selection_audit,
                                n_before, n_dupes, residual, DATA_PROCESSED)

    print(f"Phase 1: clean={df.shape}  approve_deny={approve_deny.shape}  denials={df_denials.shape}")
    return df, approve_deny, df_denials


def run_phase2(clean, appdeny):
    clean, ANOMALY_FEATS, CLUSTER_CONT_PART, CLUSTER_FLAGS, CLUSTER_FEATS = p2.add_engineered_flags(clean)
    ITEM_FEATURES = p2.item_features_for_rules(appdeny)

    for name, feats in [("CLUSTER_FEATS", CLUSTER_FEATS), ("ANOMALY_FEATS", ANOMALY_FEATS),
                        ("ITEM_FEATURES", ITEM_FEATURES)]:
        leak = sorted(set(feats) & set(cfg.POST_DECISION))
        assert not leak, f"post-decision fields leaked into {name}: {leak}"

    Xc, Xa, idx_med, idx_small = p2.prepare_matrices(
        clean, CLUSTER_CONT_PART, CLUSTER_FLAGS, CLUSTER_FEATS, ANOMALY_FEATS, RANDOM_STATE)

    elbow_k, sil_k, BEST_K, k_selection = p2.select_k(Xc, range(2, 11), RANDOM_STATE)
    k_selection.assign(elbow_k=elbow_k, sil_k=sil_k, best_k=BEST_K).to_csv(
        DATA_PROCESSED / "dash_k_selection.csv", index=False)
    clean["kmeans_cluster"] = p2.run_kmeans(Xc, BEST_K, RANDOM_STATE)
    db_labels, dbscan_noise_ids, eps, min_samples, kdist = p2.run_dbscan(Xc, idx_med, CLUSTER_FEATS)
    hier_labels, Zward = p2.run_hierarchical(Xc, idx_small, BEST_K)
    clean, profile = p2.profile_clusters(clean, "kmeans_cluster")
    clarans_labels, medoid_ids, clarans_cost = p2.run_clarans(Xc, idx_small, BEST_K, RANDOM_STATE)

    p2.export_phase2(clean, profile, idx_small, clarans_labels, DATA_PROCESSED)
    p5.export_clustering_comparison(
        Xc, clean["kmeans_cluster"].values, idx_med, db_labels, idx_small,
        hier_labels, clarans_labels, DATA_PROCESSED, RANDOM_STATE)

    print(f"Phase 2: BEST_K={BEST_K}  clusters={clean['kmeans_cluster'].nunique()}  "
          f"dbscan_noise={len(dbscan_noise_ids)}")
    return (clean, ANOMALY_FEATS, CLUSTER_FEATS, ITEM_FEATURES, Xc, Xa, idx_med, idx_small,
            profile, db_labels, dbscan_noise_ids, hier_labels, clarans_labels, medoid_ids, BEST_K)


def run_phase3(appdeny, denials, ITEM_FEATURES):
    onehot, decision_items, n_before = p3.build_transactions(appdeny, ITEM_FEATURES, 0.02)
    frequent = p3.mine_frequent_itemsets(onehot, 0.02, 3)
    all_rules, decision_rules, dec_supp = p3.extract_decision_rules(frequent, onehot, decision_items)

    sig_df = p3.test_significance(decision_rules, onehot, top_n=10)
    sig_df.to_csv(TABLE_DIR / "p3_rule_significance.csv", index=False)

    final_rules, pruned_away, final_sig = p3.prune_redundant(
        decision_rules, all_rules, dec_supp, onehot, 0.02)
    final_sig.to_csv(DATA_PROCESSED / "p3_decision_rules_final.csv", index=False)

    # Written after pruning so the candidate export carries the verdict and the reason.
    # prune_redundant adds improvement/best_subrule to decision_rules in place.
    decision_rules["kept"] = (decision_rules["improvement"] >= 0.02).map({True: "Ya", False: "Tidak"})
    decision_rules["decision_reason"] = [
        (f"Confidence unggul {r['improvement']*100:.1f} poin di atas sub-rule terbaiknya "
         f"({r['best_subrule']}), jadi menambah informasi baru."
         if r["improvement"] >= 0.02 else
         f"Cuma unggul {r['improvement']*100:.1f} poin dari {r['best_subrule']}, di bawah "
         f"ambang 2 poin, jadi dianggap variasi trivial.")
        for _, r in decision_rules.iterrows()
    ]
    decision_rules.to_csv(DATA_PROCESSED / "p3_decision_rules.csv", index=False)

    missing_rule_metrics, interpreted_rules = p3.validate_rule_count(final_rules, 10)
    assert not missing_rule_metrics and len(final_rules) >= 10
    interpreted_rules.to_csv(TABLE_DIR / "p3_interpreted_rules.csv", index=False)

    tmp, marg, piv_rate, piv_n = p3.geography_dti_crosstab(appdeny, cfg.DTI_GRP)
    if len(denials):
        p3.denial_reason_itemsets(denials)

    print(f"Phase 3: {len(decision_rules)} decision rules -> {len(final_rules)} final")
    return final_rules, piv_rate, piv_n


def run_phase4(clean, ANOMALY_FEATS, Xa, idx_med, dbscan_noise_ids):
    clean = p4.flag_statistical(clean, ANOMALY_FEATS)
    clean["iso_pred"], clean["iso_score"] = p4.run_isolation_forest(Xa, RANDOM_STATE)
    lof_flag, lof_ids = p4.run_lof(Xa, idx_med)
    clean["lof_flag"] = lof_flag
    clean, vote = p4.ensemble_vote(clean, dbscan_noise_ids)

    top_anom, show_cols = p4.triage_top(clean, n=15)
    top_anom = p4.resolve_manual_review(top_anom, MANUAL_RESOLUTION)

    # Group-level collective anomalies: the five record-level detectors score rows one at a
    # time, so a pattern that only exists across a whole group is invisible to all of them.
    clean, _collective_groups, _flagged = p4.detect_collective_groups(
        clean, ANOMALY_FEATS, DATA_PROCESSED, RANDOM_STATE)

    p4.export_phase4(top_anom, clean, show_cols, DATA_PROCESSED)

    print(f"Phase 4: anomaly_votes>=3 -> {(clean['anomaly_votes']>=3).sum():,} rows; "
          f"top_anom verdicts={top_anom['verdict'].value_counts().to_dict()}")
    return clean, top_anom


def run_phase5(clean, appdeny, denials, profile, final_rules, top_anom, piv_rate):
    p5.export_dashboard_aggregates(clean, appdeny, denials, DATA_PROCESSED, RANDOM_STATE)
    fair_pivot = piv_rate  # same crosstab as Phase 3's piv_rate, reused rather than recomputed
    p5.build_standalone_dashboard(
        clean, appdeny, profile, final_rules, top_anom, fair_pivot,
        REPORTS_DIR / "HMDA_Interactive_Dashboard.html", RANDOM_STATE)
    print("Phase 5: dashboard aggregates + standalone HTML dashboard exported")


# ---------------------------------------------------------------------------
# Dashboard-only aggregates (previously replicate.py + the 9 build_dash_*.py scripts)
# ---------------------------------------------------------------------------

FEAT_COLS = ["income", "loan_amount", "property_value", "combined_loan_to_value_ratio",
            "debt_to_income_ratio", "derived_race", "loan_purpose", "occupancy_type",
            "action_taken", "tract_minority_population_percent"]


def build_cluster_scatter_extras(clean, idx_med, idx_small, db_labels, hier_labels, clarans_labels):
    have_feat = [c for c in FEAT_COLS if c in clean.columns]

    dbscan_df = clean.iloc[idx_med][have_feat + ["kmeans_cluster"]].copy()
    dbscan_df.insert(0, "row", np.array(idx_med))
    dbscan_df["dbscan_cluster"] = db_labels
    dbscan_df.to_csv(DATA_PROCESSED / "dash_dbscan_scatter.csv", index=False)

    hier_df = clean.iloc[idx_small][have_feat + ["kmeans_cluster"]].copy()
    hier_df.insert(0, "row", np.array(idx_small))
    hier_df["hier_cluster"] = hier_labels
    clarans_map = dict(zip(np.array(idx_small), clarans_labels))
    hier_df["clarans_cluster"] = hier_df["row"].map(clarans_map)
    hier_df.to_csv(DATA_PROCESSED / "dash_hierarchical_scatter.csv", index=False)

    clarans_df = clean.iloc[idx_small][["kmeans_cluster"] + have_feat].copy()
    clarans_df.insert(0, "row", np.array(idx_small))
    clarans_df["clarans_cluster"] = clarans_labels
    clarans_df.to_csv(DATA_PROCESSED / "dash_clarans_scatter.csv", index=False)

    print(f"Cluster-scatter extras: dbscan={len(dbscan_df)} hier={len(hier_df)} clarans={len(clarans_df)} rows")


def build_outlier_taxonomy(clean, random_state=RANDOM_STATE):
    flag_cols = ["flag_iqr", "flag_z", "flag_iso", "lof_flag", "dbscan_noise", "iso_score", "anomaly_votes"]
    tax_cols = ["income", "loan_amount", "property_value", "combined_loan_to_value_ratio",
               "loan_term", "occupancy_type", "total_units", "action_taken"] + flag_cols
    tax = clean[tax_cols].copy()
    tax["global_flag"] = ((tax["flag_iqr"] == 1) | (tax["flag_z"] == 1) | (tax["flag_iso"] == 1)).astype(int)
    tax["local_flag"] = ((tax["lof_flag"] == 1) | (tax["dbscan_noise"] == 1)).astype(int)

    def _category(row):
        if row["global_flag"] and row["local_flag"]:
            return "Both (global + contextual)"
        if row["global_flag"]:
            return "Global outlier"
        if row["local_flag"]:
            return "Contextual/local outlier"
        return "Normal"
    tax["category"] = tax.apply(_category, axis=1)

    flagged = tax[tax["category"] != "Normal"]
    normal_sample = tax[tax["category"] == "Normal"].sample(
        n=min(4000, int((tax["category"] == "Normal").sum())), random_state=random_state)
    tax_out = pd.concat([flagged, normal_sample]).reset_index(drop=True)
    tax_out.to_csv(DATA_PROCESSED / "dash_outlier_taxonomy.csv", index=False)

    tax_summary = tax["category"].value_counts().rename_axis("category").reset_index(name="n")
    tax_summary["pct"] = (tax_summary["n"] / len(tax) * 100).round(2)
    tax_summary.to_csv(DATA_PROCESSED / "dash_outlier_taxonomy_summary.csv", index=False)

    # Collective-pattern evidence: HMDA's mandatory rounding-to-nearest-$10k-midpoint rule
    # means every loan >= $1M reports an amount ending in the same "...5000" suffix -- a
    # genuine group-wide signature of the data-generating process, not of any one record.
    la = clean["loan_amount"].astype(int).astype(str)
    big = clean["loan_amount"] >= 1_000_000
    end5000 = la.str.endswith("5000") & big
    collective_pct = float(end5000.sum() / big.sum() * 100) if big.sum() else 0.0
    collective_summary = pd.DataFrame([{
        "loans_ge_1m": int(big.sum()),
        "loans_ge_1m_ending_5000": int(end5000.sum()),
        "pct": round(collective_pct, 1),
    }])
    collective_summary.to_csv(DATA_PROCESSED / "dash_collective_pattern.csv", index=False)

    print(f"Outlier taxonomy: {tax['category'].value_counts().to_dict()}")


def build_geography_gap(piv_rate, piv_n):
    order = ["Low(<36%)", "Mid(36-50%)", "High(>50%)", "Unknown/Exempt"]
    rows = []
    for grp in order:
        if grp not in piv_rate.columns:
            continue
        low = piv_rate.loc["Low_Minority", grp]
        maj = piv_rate.loc["Majority_Minority", grp]
        rows.append({"dti_group": grp, "low_minority_approval_pct": round(low, 1),
                    "majority_minority_approval_pct": round(maj, 1),
                    "gap_pp": round(low - maj, 1),
                    "n_low_minority": int(piv_n.loc["Low_Minority", grp]),
                    "n_majority_minority": int(piv_n.loc["Majority_Minority", grp])})
    out = pd.DataFrame(rows)
    out.to_csv(DATA_PROCESSED / "dash_dti_geography_gap.csv", index=False)
    print(f"Geography gap: {len(out)} DTI groups")


def build_gender_gap(appdeny):
    """Approval by applicant sex, restricted to single-applicant files.

    HMDA's `derived_sex` is not purely a gender field: the value "Joint" means the applicant
    and co-applicant are of different sexes, so it marks a two-applicant household rather
    than a person's sex. Those files carry 99.9% co-applicant presence and a much higher
    median income, so comparing Joint against Female measures household structure and
    earning power, not gender. Restricting to single-applicant files makes Male vs Female an
    apples-to-apples comparison; the two-applicant rate is exported separately as context.
    """
    if not {"derived_sex", "target_approved"}.issubset(appdeny.columns):
        print("Gender gap: derived_sex missing, skipped")
        return
    d = appdeny.copy()
    d["dti_group"] = d["debt_to_income_ratio"].astype(str).map(cfg.DTI_GRP).fillna("Unknown/Exempt")
    has_co = d["co_applicant_age"].astype(str).ne("No_CoApplicant")
    d["structure"] = np.where(has_co, "Dua pemohon", "Pemohon tunggal")

    solo = d[(~has_co) & d["derived_sex"].isin(["Male", "Female"])]

    overall = (solo.groupby("derived_sex")["target_approved"]
               .agg(n="size", approval_rate="mean").reset_index())
    overall["approval_rate"] = (overall["approval_rate"] * 100).round(1)
    overall["dti_group"] = "Semua"

    by_dti = (solo.groupby(["derived_sex", "dti_group"])["target_approved"]
              .agg(n="size", approval_rate="mean").reset_index())
    by_dti["approval_rate"] = (by_dti["approval_rate"] * 100).round(1)

    out = pd.concat([overall, by_dti], ignore_index=True)
    out = out[out["n"] >= 30]
    out.to_csv(DATA_PROCESSED / "dash_gender_gap.csv", index=False)

    # Context table: what the raw derived_sex categories look like before the restriction,
    # so the "Joint is not a gender" caveat can be shown with its own numbers.
    ctx = (d.groupby("derived_sex")
           .agg(n=("target_approved", "size"),
                approval_rate=("target_approved", "mean"),
                pct_with_coapplicant=("structure", lambda s: (s == "Dua pemohon").mean()),
                median_income=("income", "median"))
           .reset_index())
    ctx["approval_rate"] = (ctx["approval_rate"] * 100).round(1)
    ctx["pct_with_coapplicant"] = (ctx["pct_with_coapplicant"] * 100).round(1)
    ctx.to_csv(DATA_PROCESSED / "dash_gender_context.csv", index=False)

    print(f"Gender gap: {len(solo):,} single-applicant files, "
          f"{out['dti_group'].nunique()} DTI bands (Joint excluded as not-a-gender)")


def build_state_aggregates(clean, appdeny, denials):
    REASON = {"1": "Debt-to-income", "2": "Employment history", "3": "Credit history",
             "4": "Collateral", "5": "Insufficient cash", "6": "Unverifiable information",
             "7": "Incomplete application", "8": "Mortgage insurance denied", "9": "Other"}

    g = appdeny.groupby("state_code", observed=True)
    summary = g.agg(n=("target_approved", "size"),
                    approval_rate=("target_approved", "mean"),
                    median_income=("income", "median"),
                    median_loan=("loan_amount", "median"),
                    median_cltv=("combined_loan_to_value_ratio", "median")).reset_index()
    summary["approval_rate"] = (summary["approval_rate"] * 100).round(1)
    summary["pct_of_national"] = (summary["n"] / summary["n"].sum() * 100).round(2)

    if len(denials):
        dstate = appdeny.loc[appdeny.index.intersection(denials.index), ["state_code"]].join(
            denials[[c for c in denials.columns if c.startswith("denial_reason")]], how="inner")

        def top_reason(sub):
            vc = sub["denial_reason_1"].map(lambda x: REASON.get(str(x).split(".")[0], "Other")).value_counts()
            return vc.index[0] if len(vc) else "Unknown"
        top_by_state = dstate.groupby("state_code", observed=True).apply(top_reason, include_groups=False)
        summary["top_denial_reason"] = summary["state_code"].map(top_by_state).fillna("Unknown")

    summary = summary[summary["n"] >= 30].sort_values("n", ascending=False).reset_index(drop=True)
    summary.to_csv(DATA_PROCESSED / "dash_state_summary.csv", index=False)

    DTI_ORDER = ["<20%", "20%-<30%", "30%-<36%", "36%-<43%", "43%-<50%", "50%-60%", ">60%"]
    sd = appdeny[appdeny["debt_to_income_ratio"].astype(str).isin(DTI_ORDER)].copy()
    sd["debt_to_income_ratio"] = sd["debt_to_income_ratio"].astype(str)
    state_dti = sd.groupby(["state_code", "debt_to_income_ratio"], observed=True)["target_approved"] \
        .agg(["mean", "size"]).reset_index()
    state_dti.columns = ["state_code", "dti_band", "approval_rate", "n"]
    state_dti["approval_rate"] = (state_dti["approval_rate"] * 100).round(1)
    state_dti = state_dti[state_dti["n"] >= 10]
    state_dti.to_csv(DATA_PROCESSED / "dash_state_dti.csv", index=False)

    sc = clean[["state_code", "kmeans_cluster"]].copy()
    sc = sc[sc["state_code"].isin(summary["state_code"])]
    state_seg = sc.groupby(["state_code", "kmeans_cluster"], observed=True).size().reset_index(name="n")
    tot = state_seg.groupby("state_code")["n"].transform("sum")
    state_seg["pct"] = (state_seg["n"] / tot * 100).round(1)
    state_seg.to_csv(DATA_PROCESSED / "dash_state_segment.csv", index=False)

    print(f"State aggregates: {len(summary)} states, {len(state_dti)} state x DTI rows, "
          f"{len(state_seg)} state x segment rows")


def build_term_aggregates(appdeny):
    ORDER = ["<=10yr", "15yr", "20yr", "25yr", "30yr", ">30yr"]

    def band(t):
        if pd.isna(t):
            return np.nan
        t = float(t)
        if t <= 120:
            return "<=10yr"
        if t <= 180:
            return "15yr"
        if t <= 240:
            return "20yr"
        if t <= 300:
            return "25yr"
        if t <= 360:
            return "30yr"
        return ">30yr"

    appdeny = appdeny.copy()
    appdeny["_term_band"] = appdeny["loan_term"].map(band)
    appdeny["_is_high_dti50"] = appdeny["debt_to_income_ratio"].astype(str).isin(["50%-60%", ">60%"]).astype(int)
    appdeny["_is_investment"] = (appdeny["occupancy_type"].astype(str) == "Investment").astype(int)

    g = appdeny.groupby("_term_band", observed=True).agg(
        n=("target_approved", "size"),
        approval_rate=("target_approved", "mean"),
        median_income=("income", "median"),
        median_loan=("loan_amount", "median"),
        median_cltv=("combined_loan_to_value_ratio", "median"),
        pct_high_dti=("_is_high_dti50", "mean"),
        pct_investment=("_is_investment", "mean"),
    ).reindex(ORDER).reset_index().rename(columns={"_term_band": "term_band"})
    g["approval_rate"] = (g["approval_rate"] * 100).round(1)
    g["pct_high_dti"] = (g["pct_high_dti"] * 100).round(1)
    g["pct_investment"] = (g["pct_investment"] * 100).round(1)
    g["pct_of_total"] = (g["n"] / g["n"].sum() * 100).round(1)
    g.to_csv(DATA_PROCESSED / "dash_term_summary.csv", index=False)

    ct = pd.crosstab(appdeny["_term_band"], appdeny["loan_purpose"], normalize="index").mul(100).round(1)
    ct = ct.reindex(ORDER).reset_index().rename(columns={"_term_band": "term_band"})
    purpose_long = ct.melt(id_vars="term_band", var_name="loan_purpose", value_name="pct")
    purpose_long.to_csv(DATA_PROCESSED / "dash_term_purpose.csv", index=False)

    print(f"Term aggregates: {len(g)} term bands, {len(purpose_long)} purpose-composition rows")


def build_phase1_distributions(clean):
    """Binned distributions for the Phase 1 continuous features.

    The dashboard's Fase 1 explains why median imputation and winsorizing were chosen, but
    that argument rests on the data being heavily right-skewed. Exporting the histogram makes
    the reader able to see the skew instead of taking the claim on trust. Bins are computed
    once here so the dashboard never has to load the 100k-row table.
    """
    feats = ["income", "loan_amount", "property_value",
             "combined_loan_to_value_ratio", "loan_term"]
    rows, stats = [], []
    for f in feats:
        if f not in clean.columns:
            continue
        s_num = pd.to_numeric(clean[f], errors="coerce").dropna()
        if not len(s_num):
            continue
        # Clip BOTH tails for display. The upper tail matters most (a single $800M property
        # value would put every real observation into one bar), but income also has valid
        # negative values from business losses, which stretched the axis to -2000 and squeezed
        # the actual distribution into a sliver.
        hi = float(s_num.quantile(0.995))
        lo = float(s_num.quantile(0.005))
        clipped = s_num.clip(lo, hi)
        counts, edges = np.histogram(clipped, bins=40)
        for c, left, right in zip(counts, edges[:-1], edges[1:]):
            rows.append({"feature": f, "bin_left": round(float(left), 2),
                         "bin_right": round(float(right), 2), "count": int(c)})
        stats.append({
            "feature": f,
            "n": int(len(s_num)),
            "mean": round(float(s_num.mean()), 2),
            "median": round(float(s_num.median()), 2),
            "p99": round(float(s_num.quantile(0.99)), 2),
            "max": round(float(s_num.max()), 2),
            "skew": round(float(s_num.skew()), 2),
            "mean_over_median": round(float(s_num.mean() / s_num.median()), 2)
            if s_num.median() else None,
        })
    pd.DataFrame(rows).to_csv(DATA_PROCESSED / "dash_phase1_distributions.csv", index=False)
    pd.DataFrame(stats).to_csv(DATA_PROCESSED / "dash_phase1_distribution_stats.csv", index=False)
    print(f"Phase 1 distributions: {len(stats)} features x 40 bins")


def build_detector_comparison(clean):
    """Systematic side-by-side of the five detectors, plus how anomalies map onto clusters.

    The rubric asks for IQR / Z-score / Isolation Forest to be compared systematically and for
    the anomaly results to be tied back to the clustering. Both need the flag columns, so they
    are derived here once Phase 4 has populated them.
    """
    detectors = [
        ("IQR", "flag_iqr", "Global", ">=3 fitur di luar 1,5xIQR"),
        ("Z-score", "flag_z", "Global", ">=1 fitur |z| > 3"),
        ("Isolation Forest", "flag_iso", "Global", "contamination 1%, 200 pohon"),
        ("Local Outlier Factor", "lof_flag", "Kontekstual", "20 tetangga, contamination 1%"),
        ("DBSCAN-noise", "dbscan_noise", "Kontekstual", "titik noise dari Fase 2"),
    ]
    n = len(clean)
    rows = []
    for name, col, kind, param in detectors:
        if col not in clean.columns:
            continue
        flagged = (clean[col] == 1)
        rows.append({
            "detector": name, "philosophy": kind, "parameter": param,
            "flagged": int(flagged.sum()),
            "pct": round(flagged.mean() * 100, 2),
            "median_votes": float(clean.loc[flagged, "anomaly_votes"].median())
            if flagged.any() else np.nan,
        })
    pd.DataFrame(rows).to_csv(DATA_PROCESSED / "dash_detector_comparison.csv", index=False)

    # Pairwise Jaccard overlap: how much do the detectors actually agree?
    cols = [c for _, c, _, _ in detectors if c in clean.columns]
    names = [nm for nm, c, _, _ in detectors if c in clean.columns]
    ov = []
    for i, a_col in enumerate(cols):
        for j, b_col in enumerate(cols):
            if j <= i:
                continue
            A, B = clean[a_col] == 1, clean[b_col] == 1
            union = int((A | B).sum())
            ov.append({"a": names[i], "b": names[j],
                       "jaccard": round(int((A & B).sum()) / union, 3) if union else 0.0,
                       "both": int((A & B).sum())})
    pd.DataFrame(ov).to_csv(DATA_PROCESSED / "dash_detector_overlap.csv", index=False)

    # Anomaly rate per cluster: the link between Phase 2 and Phase 4.
    if "kmeans_cluster" in clean.columns:
        g = clean.groupby("kmeans_cluster").apply(
            lambda x: pd.Series({
                "n": len(x),
                "high_conf": int((x["anomaly_votes"] >= 3).sum()),
                "pct_high_conf": round((x["anomaly_votes"] >= 3).mean() * 100, 2),
                "mean_votes": round(float(x["anomaly_votes"].mean()), 3),
                "mean_iso_score": round(float(x["iso_score"].mean()), 4),
            }), include_groups=False).reset_index()
        g.to_csv(DATA_PROCESSED / "dash_anomaly_by_cluster.csv", index=False)
        print(f"Detector comparison: {len(rows)} detectors, {len(ov)} pairs, "
              f"{len(g)} clusters")


def build_anomaly_drivers(clean):
    """What actually makes a record anomalous, per segment.

    A list of extreme rows shows the numbers but not the reason. This compares flagged rows
    against their own segment's normal population, feature by feature, so each segment gets a
    plain statement like "anomalies here are driven by loan_amount at 8x the segment median".
    Comparing within-segment matters: a $2M loan is unremarkable in the jumbo segment and very
    unusual in the small-loan one.
    """
    feats = ["income", "loan_amount", "property_value",
             "combined_loan_to_value_ratio", "loan_term"]
    feats = [f for f in feats if f in clean.columns]
    if "anomaly_votes" not in clean.columns or "kmeans_cluster" not in clean.columns:
        return
    rows = []
    for cid, g in clean.groupby("kmeans_cluster"):
        flagged = g[g["anomaly_votes"] >= 3]
        normal = g[g["anomaly_votes"] == 0]
        if len(flagged) < 5 or len(normal) < 50:
            continue
        for f in feats:
            fv = pd.to_numeric(flagged[f], errors="coerce").median()
            nv = pd.to_numeric(normal[f], errors="coerce").median()
            if pd.isna(fv) or pd.isna(nv) or nv == 0:
                continue
            rows.append({
                "kmeans_cluster": int(cid),
                "feature": f,
                "median_normal": round(float(nv), 2),
                "median_flagged": round(float(fv), 2),
                "ratio": round(float(fv) / float(nv), 2),
                "n_flagged": int(len(flagged)),
            })
    out = pd.DataFrame(rows)
    if len(out):
        # Keep the single strongest driver per segment: the feature whose flagged median
        # departs furthest from the segment's own normal median, in either direction.
        out["deviation"] = (out["ratio"] - 1).abs()
        top = out.sort_values("deviation", ascending=False).groupby("kmeans_cluster").head(2)
        top.to_csv(DATA_PROCESSED / "dash_anomaly_drivers.csv", index=False)
        print(f"Anomaly drivers: {out['kmeans_cluster'].nunique()} segments profiled")


def build_anomaly_reasons(clean, random_state=RANDOM_STATE):
    """Per-row anomaly reason for the scatter hover, computed vectorised.

    For every sampled row, find which feature departs furthest from that row's OWN segment
    median and phrase it in plain language. Comparing within-segment is the point: a $2M loan
    is ordinary in the jumbo segment and very unusual in the small-loan one, so a global
    threshold would mislabel both.
    """
    labels = {
        "income": "Income",
        "loan_amount": "Loan amount",
        "property_value": "Property value",
        "combined_loan_to_value_ratio": "CLTV",
        "loan_term": "Loan term",
    }
    feats = [f for f in labels if f in clean.columns]
    if not feats or "kmeans_cluster" not in clean.columns:
        return

    sample = clean.sample(min(8000, len(clean)), random_state=random_state).copy()
    med = clean.groupby("kmeans_cluster")[feats].median()

    # Ratio of each feature to its own segment median, as one aligned frame.
    seg_med = med.reindex(sample["kmeans_cluster"]).to_numpy(dtype=float)
    vals = sample[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(seg_med != 0, vals / seg_med, np.nan)
    dev = np.abs(ratio - 1.0)
    dev = np.where(np.isfinite(dev), dev, -1.0)

    best_i = dev.argmax(axis=1)
    rows = np.arange(len(sample))
    best_ratio = ratio[rows, best_i]
    best_name = np.array([labels[feats[i]] for i in best_i])

    votes = pd.to_numeric(sample.get("anomaly_votes", 0), errors="coerce").fillna(0).to_numpy()
    factor = np.where(best_ratio >= 1, best_ratio, np.where(best_ratio > 0, 1 / best_ratio, np.nan))
    arah = np.where(best_ratio >= 1, "lebih besar", "lebih kecil")

    reason = np.where(
        votes < 1,
        "Tidak ditandai detektor mana pun",
        np.where(
            np.isfinite(factor),
            np.char.add(
                np.char.add(np.char.add(best_name, " "),
                            np.array([f"{v:.1f}x " if np.isfinite(v) else "" for v in factor])),
                np.char.add(arah, " dari median segmennya"),
            ),
            "Ditandai, pendorong utama tidak teridentifikasi",
        ),
    )
    sample["anomaly_reason"] = reason

    cols = [c for c in ["kmeans_cluster", "income", "loan_amount", "property_value",
                        "combined_loan_to_value_ratio", "debt_to_income_ratio",
                        "derived_race", "loan_purpose", "occupancy_type", "action_taken",
                        "_approved", "anomaly_votes", "iso_score",
                        "tract_minority_population_percent", "anomaly_reason"]
            if c in sample.columns]
    sample[cols].to_csv(DATA_PROCESSED / "dash_scatter.csv", index=False)
    n_named = int((votes >= 1).sum())
    print(f"Anomaly reasons: {n_named:,} baris ditandai diberi alasan")


def build_context_fields(appdeny):
    # Contextual (non-demographic) fields for the What-If "more context" section. These
    # deliberately exclude derived_race/derived_ethnicity/derived_sex/tract_minority_cat --
    # rule-mining already showed those add no predictive lift over DTI, and the Fairness
    # tab is the properly-caveated home for that residual gap (association, not causal).
    CONTEXT_FIELDS = ["applicant_age", "occupancy_type", "total_units", "conforming_loan_limit",
                      "property_value_band", "cltv_band", "tract_income_cat"]
    rows = []
    for field in CONTEXT_FIELDS:
        if field not in appdeny.columns:
            continue
        g = appdeny.groupby(field, observed=True)["target_approved"].agg(["mean", "size"])
        for val, (mean, size) in g.iterrows():
            if pd.isna(val) or size < 20:
                continue
            rows.append({"field": field, "value": str(val), "n": int(size),
                        "approval_rate": round(mean * 100, 1)})
    out = pd.DataFrame(rows)
    out.to_csv(DATA_PROCESSED / "dash_context_fields.csv", index=False)
    print(f"Context fields: {len(out)} rows")


def main():
    clean, appdeny, denials = run_phase1()
    (clean, ANOMALY_FEATS, CLUSTER_FEATS, ITEM_FEATURES, Xc, Xa, idx_med, idx_small,
     profile, db_labels, dbscan_noise_ids, hier_labels, clarans_labels, medoid_ids, BEST_K) = \
        run_phase2(clean, appdeny)
    final_rules, piv_rate, piv_n = run_phase3(appdeny, denials, ITEM_FEATURES)
    clean, top_anom = run_phase4(clean, ANOMALY_FEATS, Xa, idx_med, dbscan_noise_ids)
    run_phase5(clean, appdeny, denials, profile, final_rules, top_anom, piv_rate)

    build_cluster_scatter_extras(clean, idx_med, idx_small, db_labels, hier_labels, clarans_labels)
    build_outlier_taxonomy(clean)
    build_geography_gap(piv_rate, piv_n)
    build_gender_gap(appdeny)
    build_phase1_distributions(clean)
    build_detector_comparison(clean)
    build_anomaly_drivers(clean)
    build_anomaly_reasons(clean)
    build_state_aggregates(clean, appdeny, denials)
    build_term_aggregates(appdeny)
    build_context_fields(appdeny)

    print("\nbuild_data.py complete: all dash_*.csv aggregates + core pipeline outputs written.")


if __name__ == "__main__":
    main()
