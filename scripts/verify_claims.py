"""Verify every hardcoded number in the dashboard against the exported data.

Any claim shown to a reader must be reproducible from data/processed or data/interim.
Run after regenerating data:

    python scripts/verify_claims.py

Exits non-zero if any claim fails, so a stale number cannot silently survive.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
I = ROOT / "data" / "interim"

results = []


def check(label, actual, expected, tol=0.05):
    """Numeric comparison with a small tolerance; strings compared exactly."""
    if isinstance(expected, str) or isinstance(actual, str):
        ok = str(actual) == str(expected)
    else:
        ok = actual is not None and abs(float(actual) - float(expected)) <= tol
    results.append((ok, label, actual, expected))


def load(name, folder=P, **kw):
    p = folder / name
    return pd.read_csv(p, **kw) if p.exists() else None


# ---------------------------------------------------------------- Phase 1
s1 = load("dash_phase1_cleaning_summary.csv")
check("Fase 1 baris mentah", int(s1["raw_rows"].iloc[0]), 100000, 0)
check("Fase 1 baris bersih", int(s1["clean_rows"].iloc[0]), 99995, 0)
check("Fase 1 duplikat", int(s1["duplicates_removed"].iloc[0]), 5, 0)
check("Fase 1 kolom dibuang", int(s1["fields_dropped"].iloc[0]), 6, 0)
check("Fase 1 sisa sel kosong", int(s1["residual_missing_cells"].iloc[0]), 0, 0)

sys.path.insert(0, str(ROOT))
from pipeline import config as cfg  # noqa: E402

check("Ambang buang kolom", cfg.MISSING_DROP_THRESHOLD * 100, 60, 0)
raw_cols = sum(len(v) for v in cfg.GROUPS.values())
check("Total kolom mentah", raw_cols, 99, 0)
check("Kolom IDS", len(cfg.IDS), 5, 0)
check("Kolom demografi mentah", len(cfg.DEMOGRAPHIC_RAW), 30, 0)
check("Kolom AUS", len(cfg.AUS), 5, 0)
check("Kolom alasan penolakan", len(cfg.DENIAL), 4, 0)
check("Kolom leakage", len(cfg.LEAKAGE), 13, 0)

audit = load("p1_feature_selection_audit.csv", ROOT / "results" / "tables")
check("Fitur diskor", len(audit), 31, 0)

# ---------------------------------------------------------------- Phase 2
cmp_df = load("dash_clustering_comparison.csv")
km_s = cmp_df[cmp_df["method"] == "K-Means (pada sampel 4.000)"].iloc[0]
check("K-Means sampel silhouette", km_s["silhouette"], 0.303, 0.002)
check("K-Means sampel Davies-Bouldin", km_s["davies_bouldin"], 1.137, 0.002)
check("K-Means sampel Calinski-Harabasz", km_s["calinski_harabasz"], 869, 2)
ward = cmp_df[cmp_df["method"].str.startswith("Hierarchical")].iloc[0]
check("ARI Ward vs K-Means", ward["ari_vs_kmeans"], 0.906, 0.002)
clar = cmp_df[cmp_df["method"].str.startswith("CLARANS")].iloc[0]
check("ARI CLARANS vs K-Means", clar["ari_vs_kmeans"], 0.710, 0.002)
db = cmp_df[cmp_df["method"] == "DBSCAN"].iloc[0]
check("DBSCAN noise", db["noise"], 895, 0)
check("DBSCAN cluster", db["n_clusters"], 17, 0)

prof = load("p2_cluster_profiles.csv")
check("Jumlah segmen", len(prof), 7, 0)
man = prof[prof["segment_name"] == "Manufactured-housing applicants"].iloc[0]
check("Manufactured approval rate", man["approval_rate"], 43.1, 0.1)
check("Manufactured n", int(man["n"]), 4529, 0)

stab = load("p2_cluster_stability_audit.csv")
check("Cluster mencurigakan", int(stab["spurious_cluster_candidate"].sum()), 0, 0)
man_s = stab[stab["segment_name"] == "Manufactured-housing applicants"].iloc[0]
check("Manufactured silhouette", man_s["mean_silhouette_4k"], 0.495, 0.002)
check("Manufactured Ward purity", man_s["ward_purity_4k"], 1.0, 0.001)
check("Manufactured CLARANS purity", man_s["clarans_purity_4k"], 1.0, 0.001)

# ---------------------------------------------------------------- Phase 3
cand = load("p3_decision_rules.csv")
fin = load("p3_decision_rules_final.csv")
check("Aturan kandidat", len(cand), 28, 0)
check("Aturan final", len(fin), 11, 0)
check("Kandidat diambil", int((cand["kept"] == "Ya").sum()), 11, 0)

dti = fin[fin["antecedent"] == "debt_to_income_ratio=>60%"].iloc[0]
check("DTI>60% confidence", dti["confidence"] * 100, 91.5, 0.2)
check("DTI>60% lift", dti["lift"], 3.96, 0.02)
check("DTI>60% n", int(dti["n"]), 4121, 0)

dti_sub = fin[fin["antecedent"].str.contains("Subordinate_Lien")].iloc[0]
check("DTI+Subordinate confidence", dti_sub["confidence"] * 100, 94.0, 0.2)
check("DTI+Subordinate lift", dti_sub["lift"], 4.06, 0.02)

man_conv = fin[fin["antecedent"] == "construction_method=Manufactured, loan_type=Conventional"].iloc[0]
check("Manufactured+Conventional confidence", man_conv["confidence"] * 100, 63.5, 0.2)
check("Manufactured+Conventional lift", man_conv["lift"], 2.75, 0.02)

den = load("dash_denial_reasons.csv")
check("Alasan Other", den[den["reason"] == "Other"]["pct_of_denials"].iloc[0], 72.9, 0.1)
check("Alasan DTI", den[den["reason"] == "Debt-to-income"]["pct_of_denials"].iloc[0], 8.7, 0.1)

# ---------------------------------------------------------------- Phase 4
tax = load("dash_outlier_taxonomy_summary.csv")
tx = dict(zip(tax["category"], tax["n"]))
check("Outlier global", tx.get("Global outlier"), 9959, 0)
check("Outlier kontekstual", tx.get("Contextual/local outlier"), 589, 0)
check("Outlier keduanya", tx.get("Both (global + contextual)"), 476, 0)
check("Normal", tx.get("Normal"), 88971, 0)

triage = load("p4_anomaly_triage.csv")
check("Rekaman ditriase", len(triage), 15, 0)
check("Verdict RARE BUT VALID", int((triage["verdict"] == "RARE BUT VALID").sum()), 15, 0)

cs = load("dash_collective_summary.csv").iloc[0]
check("Grup kolektif ditandai", int(cs["flagged_groups"]), 33, 0)
check("Pure collective", int(cs["pure_collective_groups"]), 21, 0)

cg = load("p4_collective_group_anomalies.csv")
non_state = cg[~cg["group_spec"].str.startswith("state_")]
man_non_state = non_state[non_state["group_values"].str.contains("Manufactured")]
check("Grup non-geografis", len(non_state), 4, 0)
check("Di antaranya manufactured", len(man_non_state), 3, 0)

state_g = cg[cg["group_spec"].str.startswith("state_")]
SMALL = {"HI", "DC", "PR", "VI", "GU", "AS", "MP", "AK", "WY", "VT", "ND", "SD", "MT", "DE", "RI", "Unknown"}
small_n = state_g["group_values"].map(lambda v: str(v).split(" | ")[0].strip()).isin(SMALL).sum()
check("Grup berbasis state", len(state_g), 29, 0)
check("Yurisdiksi kecil", int(small_n), 15, 0)
check("Porsi yurisdiksi kecil (%)", small_n / len(state_g) * 100, 52, 1.0)

kaudit = load("p4_kmeans_collective_audit.csv")
mrow = kaudit[kaudit["segment_name"] == "Manufactured-housing applicants"].iloc[0]
check("Manufactured skor bukti", int(mrow["evidence_score_0_4"]), 4, 0)
check("Manufactured flag rate anggota (%)", mrow["member_individual_flag_rate"] * 100, 8.7, 0.2)

# ---------------------------------------------------------------- Phase 5
geo = load("dash_dti_geography_gap.csv")
check("Selisih tract DTI rendah", geo[geo["dti_group"] == "Low(<36%)"]["gap_pp"].iloc[0], 12.1, 0.1)
check("Selisih tract DTI tinggi", geo[geo["dti_group"] == "High(>50%)"]["gap_pp"].iloc[0], 1.8, 0.1)

gg = load("dash_gender_gap.csv")
if gg is not None:
    allrow = gg[gg["dti_group"] == "Semua"].set_index("derived_sex")["approval_rate"]
    check("Gender Joint keseluruhan", allrow.get("Joint"), 83.0, 0.1)
    check("Gender Male keseluruhan", allrow.get("Male"), 73.9, 0.1)
    check("Gender Female keseluruhan", allrow.get("Female"), 72.3, 0.1)
    low = gg[gg["dti_group"] == "Low(<36%)"].set_index("derived_sex")["approval_rate"]
    check("Gender Joint DTI rendah", low.get("Joint"), 88.6, 0.1)
    check("Gender Female DTI rendah", low.get("Female"), 80.9, 0.1)

ad = pd.read_csv(I / "hmda_approve_deny.csv", usecols=["target_approved"])
check("Aplikasi berkeputusan", len(ad), 67827, 0)
check("Base approval (%)", ad["target_approved"].mean() * 100, 76.9, 0.1)
check("Base denial (%)", (1 - ad["target_approved"].mean()) * 100, 23.1, 0.1)

# ---------------------------------------------------------------- report
fails = [r for r in results if not r[0]]
for ok, label, actual, expected in results:
    if not ok:
        print(f"  GAGAL  {label}: dashboard={expected}  data={actual}")
print(f"\n{len(results) - len(fails)}/{len(results)} klaim terverifikasi.")
if fails:
    print(f"{len(fails)} klaim TIDAK cocok dengan data.")
    sys.exit(1)
print("Semua angka di dashboard cocok dengan data hasil pipeline.")
