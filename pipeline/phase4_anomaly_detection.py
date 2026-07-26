"""Phase 4 anomaly-detection logic extracted from HMDA.ipynb.

Narrative markdown, the score-distribution/agreement plots, and the
close-the-loop validation cells (unanimous-vote check) stay in the notebook;
only the detector logic and triage classification live here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from . import config as cfg

SHOW_COLS = ["income", "loan_amount", "property_value", "combined_loan_to_value_ratio",
            "loan_term", "loan_purpose", "occupancy_type", "total_units", "action_taken",
            "anomaly_votes", "iso_score"]


def flag_statistical(clean, anomaly_feats):
    """Cell 105: per-feature IQR and Z-score flags, aggregated into iqr_n/z_n.

    Mutates `clean` in place, matching the original cell.
    """
    iqr_flag = pd.DataFrame(index=clean.index)
    z_flag = pd.DataFrame(index=clean.index)
    for c in anomaly_feats:
        s = clean[c].astype(float)
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        iqr_flag[c] = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        z = (s - s.mean()) / s.std(ddof=0)
        z_flag[c] = z.abs() > 3

    clean["iqr_n"] = iqr_flag.sum(axis=1)
    clean["z_n"] = z_flag.sum(axis=1)
    return clean


def run_isolation_forest(Xa, random_state=cfg.RANDOM_STATE, contamination=0.01, n_estimators=200):
    iso = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=n_estimators)
    iso_pred = iso.fit_predict(Xa)
    iso_score = -iso.score_samples(Xa)
    return iso_pred, iso_score


def run_lof(Xa, idx_med, n_neighbors=20, contamination=0.01):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof_pred = lof.fit_predict(Xa.iloc[idx_med])
    lof_ids = set(np.array(idx_med)[lof_pred == -1].tolist())
    lof_flag = Xa.index.isin(lof_ids).astype(int)
    return lof_flag, lof_ids


def ensemble_vote(clean, dbscan_noise_ids):
    """Cell 111: cross-references DBSCAN noise and builds the 5-detector vote.

    Mutates `clean` in place, matching the original cell.
    """
    clean["dbscan_noise"] = clean.index.isin(dbscan_noise_ids).astype(int)
    clean["flag_iqr"] = (clean["iqr_n"] >= 3).astype(int)
    clean["flag_z"] = (clean["z_n"] >= 1).astype(int)
    clean["flag_iso"] = (clean["iso_pred"] == -1).astype(int)

    vote = clean[["flag_iqr", "flag_z", "flag_iso", "lof_flag", "dbscan_noise"]].sum(axis=1)
    clean["anomaly_votes"] = vote
    return clean, vote


def triage(row):
    """Evidence-based verdict for one row: DATA ERROR / RARE BUT VALID /
    RISK SIGNAL / MANUAL REVIEW."""
    ev = []
    cltv = row.get("combined_loan_to_value_ratio", 0)
    term = row.get("loan_term", 0)
    inc = row.get("income", 0)
    pv = row.get("property_value", 0)
    la = row.get("loan_amount", 0)
    if cltv and cltv > 500:
        ev.append(f"CLTV={cltv:,.0f}% (impossible)")
    if term and term > 600:
        ev.append(f"term={term:,.0f} mo (>50 yrs)")
    if inc and inc > 50_000:
        ev.append(f"income=${inc*1000:,.0f} (implausible)")
    if pv and pv > 300_000_000:
        ev.append(f"property=${pv:,.0f}")
    if ev:
        return "DATA ERROR", "; ".join(ev)
    units = str(row.get("total_units", "1"))
    if str(row.get("occupancy_type")) == "Investment" or units not in {"1", "2", "3", "4"}:
        return "RARE BUT VALID", f"multi-unit ({units}) / investment profile"
    if cltv and cltv > 100 and inc and inc < 50:
        return "RISK SIGNAL", f"CLTV {cltv:.0f}% (>100) with income ${inc*1000:,.0f}"
    if la and la > 5_000_000:
        return "RARE BUT VALID", f"jumbo loan ${la:,.0f}"
    return "MANUAL REVIEW", "extreme magnitude without a clear error signature"


def triage_top(clean, n=15):
    show_cols = [c for c in SHOW_COLS if c in clean.columns]
    top_anom = clean.sort_values(["anomaly_votes", "iso_score"], ascending=False).head(n).copy()
    verdicts = top_anom.apply(triage, axis=1)
    top_anom["verdict"] = [v for v, e in verdicts]
    top_anom["evidence"] = [e for v, e in verdicts]
    return top_anom, show_cols


def resolve_manual_review(top_anom, resolutions):
    """Cell 119: applies hand-reviewed verdicts. Mutates `top_anom` in place."""
    for idx, (verdict, why) in resolutions.items():
        if idx in top_anom.index:
            top_anom.loc[idx, "verdict"] = verdict
            top_anom.loc[idx, "evidence"] = "manual review: " + why
    return top_anom


def export_phase4(top_anom, clean, show_cols, out_dir):
    out_dir = Path(out_dir)
    top_anom[show_cols + ["verdict", "evidence"]].to_csv(out_dir / "p4_anomaly_triage.csv")
    clean.to_csv(out_dir / "p4_anomaly_flags.csv", index=True)


COLLECTIVE_GROUP_SPECS = {
    "state_x_purpose": ["state_code", "loan_purpose"],
    "state_x_product": ["state_code", "loan_type"],
    "product_x_construction": ["loan_type", "construction_method"],
    "purpose_x_construction": ["loan_purpose", "construction_method"],
    "occupancy_x_lien": ["occupancy_type", "lien_status"],
}


def _collective_key(frame, cols):
    return frame[cols].astype("string").fillna("<NA>").agg(" | ".join, axis=1)


def detect_collective_groups(clean, anomaly_feats, out_dir=None,
                             random_state=cfg.RANDOM_STATE, specs=None):
    """Group-level collective anomaly detection.

    The five record-level detectors all score rows individually, so by construction they
    cannot see a pattern that only exists across a whole group. This aggregates rows into
    business-meaningful groups (state x purpose, product x construction, ...), profiles each
    group with medians and IQRs, then runs Isolation Forest over those group profiles.

    A "pure collective" candidate is a group flagged as anomalous while fewer than 25% of
    its members were individually flagged: the group signature is unusual even though its
    individual loans are not. Outcome/denial fields are deliberately excluded so this stays
    structural discovery rather than an approval model.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    specs = specs or COLLECTIVE_GROUP_SPECS
    numeric = [c for c in anomaly_feats
               if c in clean.columns and pd.api.types.is_numeric_dtype(clean[c])]
    if not numeric:
        raise ValueError("No numeric anomaly features available for group profiles.")

    # The notebook sets global_outlier/contextual_outlier in its taxonomy cell. When this
    # runs earlier in the pipeline (build_data.py), derive them from the same detector
    # families so both paths use one definition.
    if "global_outlier" in clean.columns and "contextual_outlier" in clean.columns:
        individual_any = clean["global_outlier"].astype(bool) | clean["contextual_outlier"].astype(bool)
    else:
        glob = ((clean["flag_iqr"] == 1) | (clean["flag_z"] == 1) | (clean["flag_iso"] == 1))
        loc = ((clean["lof_flag"] == 1) | (clean["dbscan_noise"] == 1))
        individual_any = glob | loc
    hits = pd.Series(0, index=clean.index, dtype=int)
    min_group_n = max(50, int(len(clean) * 0.0005))
    profiles = []

    for spec_name, requested in specs.items():
        cols = [c for c in requested if c in clean.columns]
        if len(cols) != len(requested):
            continue

        work = clean[cols + numeric].copy()
        work["_individual_any"] = individual_any
        work["_group_key"] = _collective_key(work, cols)
        g = work.groupby("_group_key", dropna=False, sort=False)

        gp = g[numeric].median().add_suffix("__median")
        q25 = g[numeric].quantile(0.25)
        q75 = g[numeric].quantile(0.75)
        for c in numeric:
            gp[f"{c}__iqr"] = q75[c] - q25[c]
        gp["n"] = g.size()
        gp["member_individual_flag_rate"] = g["_individual_any"].mean()
        gp = gp[gp["n"] >= min_group_n].copy()
        if len(gp) < 10:
            continue

        model_cols = [c for c in gp.columns if c.endswith("__median") or c.endswith("__iqr")]
        Xg = gp[model_cols].replace([np.inf, -np.inf], np.nan)
        Xg = Xg.fillna(Xg.median(numeric_only=True)).fillna(0.0)
        nonconstant = Xg.columns[Xg.nunique(dropna=False) > 1].tolist()
        if len(nonconstant) < 2:
            continue

        Xs = RobustScaler().fit_transform(Xg[nonconstant])
        model = IsolationForest(n_estimators=300, contamination="auto",
                                random_state=random_state, n_jobs=-1)
        pred = model.fit_predict(Xs)
        gp["collective_iso_score"] = -model.decision_function(Xs)
        gp["collective_flag"] = pred == -1
        gp["group_spec"] = spec_name
        gp["group_fields"] = " x ".join(cols)
        gp["group_values"] = gp.index.astype(str)
        gp["pure_collective_candidate"] = (
            gp["collective_flag"] & (gp["member_individual_flag_rate"] < 0.25))

        flagged_keys = set(gp.index[gp["collective_flag"]])
        if flagged_keys:
            row_keys = _collective_key(clean, cols)
            hits += row_keys.isin(flagged_keys).astype(int)

        profiles.append(gp.reset_index(drop=True))

    clean["collective_group_hits"] = hits
    clean["collective_outlier_member"] = hits > 0

    if not profiles:
        return clean, pd.DataFrame(), pd.DataFrame()

    groups = pd.concat(profiles, ignore_index=True).sort_values(
        ["collective_flag", "collective_iso_score"], ascending=[False, False])
    flagged = groups[groups["collective_flag"]].copy()
    pure = flagged[flagged["pure_collective_candidate"]]
    n_members = int(clean["collective_outlier_member"].sum())

    if out_dir is not None:
        out_dir = Path(out_dir)
        flagged.to_csv(out_dir / "p4_collective_group_anomalies.csv", index=False)
        flagged.head(100).to_csv(out_dir / "dash_collective_groups.csv", index=False)
        pd.DataFrame([{
            "grouping_schemes": int(groups["group_spec"].nunique()),
            "flagged_groups": int(len(flagged)),
            "pure_collective_groups": int(len(pure)),
            "unique_member_rows": n_members,
            "pct_member_rows": n_members / len(clean) * 100,
            "median_member_individual_flag_rate": (
                float(flagged["member_individual_flag_rate"].median()) if len(flagged) else np.nan),
            "min_group_n": min_group_n,
            "method": "IsolationForest on aggregated group profiles",
        }]).to_csv(out_dir / "dash_collective_summary.csv", index=False)

    print(f"Collective groups: {len(flagged):,} flagged across "
          f"{groups['group_spec'].nunique()} schemes; {len(pure):,} pure candidates; "
          f"{n_members:,} member rows ({n_members/len(clean)*100:.2f}%).")
    return clean, groups, flagged
