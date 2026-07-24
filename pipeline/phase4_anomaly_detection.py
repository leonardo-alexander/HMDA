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
