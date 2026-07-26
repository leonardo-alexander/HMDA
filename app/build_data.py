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
    clean["kmeans_cluster"] = p2.run_kmeans(Xc, BEST_K, RANDOM_STATE)
    db_labels, dbscan_noise_ids, eps, min_samples, kdist = p2.run_dbscan(Xc, idx_med, CLUSTER_FEATS)
    hier_labels, Zward = p2.run_hierarchical(Xc, idx_small, BEST_K)
    clean, profile = p2.profile_clusters(clean, "kmeans_cluster")
    clarans_labels, medoid_ids, clarans_cost = p2.run_clarans(Xc, idx_small, BEST_K, RANDOM_STATE)

    p2.export_phase2(clean, profile, idx_small, clarans_labels, DATA_PROCESSED)

    print(f"Phase 2: BEST_K={BEST_K}  clusters={clean['kmeans_cluster'].nunique()}  "
          f"dbscan_noise={len(dbscan_noise_ids)}")
    return (clean, ANOMALY_FEATS, CLUSTER_FEATS, ITEM_FEATURES, Xc, Xa, idx_med, idx_small,
            profile, db_labels, dbscan_noise_ids, hier_labels, clarans_labels, medoid_ids, BEST_K)


def run_phase3(appdeny, denials, ITEM_FEATURES):
    onehot, decision_items, n_before = p3.build_transactions(appdeny, ITEM_FEATURES, 0.02)
    frequent = p3.mine_frequent_itemsets(onehot, 0.02, 3)
    all_rules, decision_rules, dec_supp = p3.extract_decision_rules(frequent, onehot, decision_items)
    decision_rules.to_csv(DATA_PROCESSED / "p3_decision_rules.csv", index=False)

    sig_df = p3.test_significance(decision_rules, onehot, top_n=10)
    sig_df.to_csv(TABLE_DIR / "p3_rule_significance.csv", index=False)

    final_rules, pruned_away, final_sig = p3.prune_redundant(
        decision_rules, all_rules, dec_supp, onehot, 0.02)
    final_sig.to_csv(DATA_PROCESSED / "p3_decision_rules_final.csv", index=False)

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
    build_state_aggregates(clean, appdeny, denials)
    build_term_aggregates(appdeny)
    build_context_fields(appdeny)

    print("\nbuild_data.py complete: all dash_*.csv aggregates + core pipeline outputs written.")


if __name__ == "__main__":
    main()
