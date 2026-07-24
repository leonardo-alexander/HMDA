"""Phase 1 data-transformation logic extracted from HMDA.ipynb.

Narrative markdown, plots, and validation/assert cells stay in the notebook;
only the data-transformation code lives here so the dashboard's build_data.py
can reuse it instead of re-deriving Phase 1 from scratch.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from . import config as cfg


def load_raw(url=cfg.HF_URL):
    df = pd.read_csv(url, dtype=str, keep_default_na=True)
    df.columns = df.columns.str.replace("-", "_", regex=False)
    df = df.replace({"": np.nan, " ": np.nan, "NA": np.nan, "na": np.nan, "NaN": np.nan,
                     "nan": np.nan, "None": np.nan, "null": np.nan})
    return df


def validate_group_partition(columns, groups):
    """Pure computation behind the Phase 1 partition-validation cell.

    Returns (membership, group_total, uncovered, overlapping, phantom); the
    notebook keeps the prints and asserts against these.
    """
    membership = {}
    for grp, cols in groups.items():
        for c in cols:
            membership.setdefault(c, []).append(grp)

    group_total = sum(len(cols) for cols in groups.values())
    uncovered = sorted(set(columns) - set(membership))
    overlapping = {c: g for c, g in membership.items() if len(g) > 1}
    phantom = sorted(set(membership) - set(columns))
    return membership, group_total, uncovered, overlapping, phantom


def scan_rogue_sentinels(df, whitelist, continuous_cols):
    """Pure computation behind the rogue-sentinel-audit cell.

    CONTINUOUS columns are matched numerically (decimals like "1111.0" still
    count); coded/ID columns are matched by exact string so zero-padded codes
    (e.g. county FIPS "01111") aren't mistaken for the sentinel 1111.
    """
    codes_num = [1111, 8888, 9999]
    codes_str = ["1111", "8888", "9999"]
    rogue = {}
    whitelisted_total = 0
    for col in df.columns:
        if col in continuous_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            hit_map = {code: int((vals == code).sum()) for code in codes_num}
        else:
            s = df[col].astype(str)
            hit_map = {int(code): int((s == code).sum()) for code in codes_str}
        for code, hits in hit_map.items():
            if hits == 0:
                continue
            if col in whitelist.get(str(code), []):
                whitelisted_total += hits
            else:
                rogue.setdefault(col, {})[code] = hits
    return rogue, whitelisted_total


def coerce_and_label(df):
    df = df.copy()

    # 3a - continuous coercion + exempt-field flag
    exempt_block = [c for c in cfg.CATEG_CODE if c in df.columns]
    exempt_hits = df[exempt_block].astype(str).eq("1111").any(axis=1)
    for col in cfg.CONTINUOUS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["any_exempt_field"] = exempt_hits.astype(int)

    # 3b - age bands
    for col in ["applicant_age", "co_applicant_age"]:
        if col in df.columns:
            s = df[col].astype("string").str.strip()
            s = s.replace({"8888": "Age_NA", "9999": "No_CoApplicant"})
            cats = [c for c in cfg.AGE_ORDER if c in set(s.dropna().unique())]
            df[col] = pd.Categorical(s, categories=cats, ordered=True)

    # 3c - DTI harmonization
    def map_dti(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        if v in {"Exempt", "1111"}:
            return "Exempt"
        try:
            iv = int(float(v))
            if 36 <= iv < 43:
                return "36%-<43%"
            if 43 <= iv < 50:
                return "43%-<50%"
        except (ValueError, TypeError):
            pass
        return v

    if "debt_to_income_ratio" in df.columns:
        s = df["debt_to_income_ratio"].map(map_dti)
        cats = [c for c in cfg.DTI_ORDER if c in set(pd.Series(s).dropna().unique())]
        df["debt_to_income_ratio"] = pd.Categorical(s, categories=cats, ordered=True)

    # 3d - total_units ordered categorical
    if "total_units" in df.columns:
        s = df["total_units"].astype("string").str.strip()
        cats = [c for c in cfg.UNITS_ORDER if c in set(s.dropna().unique())]
        df["total_units"] = pd.Categorical(s, categories=cats, ordered=True)

    # 3e - categorical code labels
    for col in cfg.CATEG_CODE:
        if col in df.columns:
            s = df[col].astype("string").str.strip()
            if col in cfg.LABELS:
                s = s.map(lambda x, _c=col: cfg.LABELS[_c].get(x, x) if pd.notna(x) else x)
            else:
                s = s.replace({"1111": "Exempt"})
            df[col] = s.astype("category")

    return df


def resolve_overlap_and_denials(df):
    denial_cols_present = [c for c in cfg.DENIAL if c in df.columns]
    df_denials = df.loc[df.get("action_taken").eq("Denied"), ["action_taken"] + denial_cols_present].copy() \
        if "action_taken" in df.columns else pd.DataFrame()

    drop_overlap = (
        cfg.DEMOGRAPHIC_RAW
        + ["derived_loan_product_type", "derived_dwelling_category"]
        + cfg.IDS
        + cfg.AUS
        + cfg.DENIAL
    )
    to_drop = [c for c in drop_overlap if c in df.columns]
    df = df.drop(columns=to_drop)
    return df, df_denials, to_drop


def drop_duplicate_rows(df):
    n_before = len(df)
    n_dupes = int(df.duplicated(keep="first").sum())
    df = df.drop_duplicates().reset_index(drop=True)
    return df, n_dupes, n_before


def handle_missing(df, missing_drop_threshold=cfg.MISSING_DROP_THRESHOLD):
    miss = df.isna().mean()
    high_missing = miss[miss > missing_drop_threshold].index.tolist()
    protect = {"action_taken"}
    high_missing = [c for c in high_missing if c not in protect]
    df = df.drop(columns=high_missing)

    signal_cont = [c for c in ["property_value", "income", "combined_loan_to_value_ratio",
                               "interest_rate", "loan_term"] if c in df.columns]
    for col in signal_cont:
        if df[col].isna().any():
            df[f"{col}_was_missing"] = df[col].isna().astype(int)

    cont_left = [c for c in cfg.CONTINUOUS if c in df.columns]
    for col in cont_left:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    cat_left = df.select_dtypes(include=["category", "object"]).columns.tolist()
    for col in cat_left:
        if df[col].isna().any():
            if str(df[col].dtype) == "category":
                df[col] = df[col].cat.add_categories(["Unknown"]).fillna("Unknown")
            else:
                df[col] = df[col].fillna("Unknown")

    residual_missing_after_cleaning = int(df.isna().sum().sum())
    missing_columns_after_cleaning = df.columns[df.isna().any()].tolist()

    return df, miss, high_missing, signal_cont, residual_missing_after_cleaning, missing_columns_after_cleaning


def frame_target_and_bin(df):
    if "action_taken" in df.columns:
        approve_deny = df[df["action_taken"].isin(["Originated", "Denied"])].copy()
        approve_deny["target_approved"] = (approve_deny["action_taken"] == "Originated").astype(int)
    else:
        approve_deny = df.copy()

    for new_col, (src, edges, labels) in cfg.bin_specs.items():
        if src in df.columns:
            df[new_col] = pd.cut(df[src], bins=edges, labels=labels)
            approve_deny[new_col] = pd.cut(approve_deny[src], bins=edges, labels=labels) \
                if src in approve_deny.columns else np.nan

    return df, approve_deny


def score_features(approve_deny, random_state=cfg.RANDOM_STATE):
    # 9a - correlation with target
    num_feats = approve_deny.select_dtypes(include=[np.number]).columns.tolist()
    num_feats = [c for c in num_feats if c not in {"target_approved"}
                 and not c.endswith("_rs") and c not in cfg.LEAKAGE]

    corr_target = (approve_deny[num_feats + ["target_approved"]].corr()["target_approved"]
                   .drop("target_approved").sort_values(key=np.abs, ascending=False))

    # 9c - mutual information (entropy-based)
    mi_num = [c for c in num_feats if c in approve_deny.columns]
    mi_cat = [c for c in ["loan_type", "loan_purpose", "lien_status", "occupancy_type",
                          "derived_race", "derived_ethnicity", "derived_sex",
                          "debt_to_income_ratio", "applicant_age",
                          "conforming_loan_limit", "income_band", "cltv_band",
                          "tract_income_cat", "tract_minority_cat"]
              if c in approve_deny.columns]

    X_parts, names, is_disc = [], [], []
    for c in mi_num:
        X_parts.append(approve_deny[c].fillna(approve_deny[c].median()).values.reshape(-1, 1))
        names.append(c)
        is_disc.append(False)
    for c in mi_cat:
        codes = approve_deny[c].astype("category").cat.codes.values.reshape(-1, 1)
        X_parts.append(codes)
        names.append(c)
        is_disc.append(True)

    X_mi = np.hstack(X_parts)
    y_mi = approve_deny["target_approved"].values
    mi = mutual_info_classif(X_mi, y_mi, discrete_features=is_disc, random_state=random_state)
    mi_series = pd.Series(mi, index=names).sort_values(ascending=False)

    # 9d - combined importance
    comb = pd.DataFrame({"feature": corr_target.index,
                         "abs_corr": corr_target.abs().values,
                         "corr": corr_target.values})
    comb = comb.merge(mi_series.rename("mi").reset_index().rename(columns={"index": "feature"}),
                      on="feature", how="outer")
    comb[["abs_corr", "mi"]] = comb[["abs_corr", "mi"]].fillna(0)
    comb["corr_norm"] = comb["abs_corr"] / comb["abs_corr"].max() if comb["abs_corr"].max() else 0
    comb["mi_norm"] = comb["mi"] / comb["mi"].max() if comb["mi"].max() else 0
    comb["score"] = (comb["corr_norm"] + comb["mi_norm"]) / 2
    comb = comb.sort_values("score", ascending=False).reset_index(drop=True)

    return corr_target, mi_series, comb, num_feats


def audit_feature_selection(comb):
    process_stage_diagnostics = [
        c for c in comb["feature"].astype(str)
        if c.endswith("_was_missing")
    ]

    feature_selection_audit = comb.copy()
    feature_selection_audit["role"] = np.where(
        feature_selection_audit["feature"].isin(process_stage_diagnostics),
        "Process diagnostic only",
        np.where(feature_selection_audit["score"] > 0.15, "Strong discovery candidate",
                 np.where(feature_selection_audit["score"] > 0.05,
                          "Moderate discovery candidate", "Weak / supporting"))
    )

    eligible_feature_ranking = feature_selection_audit[
        feature_selection_audit["role"] != "Process diagnostic only"
    ].copy()

    return feature_selection_audit, eligible_feature_ranking, process_stage_diagnostics


def export_phase1(df, approve_deny, df_denials, comb, out_dir):
    out_dir = Path(out_dir)
    df.to_csv(out_dir / "hmda_clean.csv", index=False)
    approve_deny.to_csv(out_dir / "hmda_approve_deny.csv", index=False)
    comb.to_csv(out_dir / "feature_ranking_combined.csv", index=False)
    if len(df_denials):
        df_denials.to_csv(out_dir / "hmda_denials.csv", index=False)
