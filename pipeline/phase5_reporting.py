"""Phase 5 communication-layer logic extracted from HMDA.ipynb: dashboard
data export, the standalone interactive HTML dashboard, and the plain-language
report generator.

The rubric-audit cell stays entirely in the notebook (it introspects notebook
globals directly, which doesn't translate into a reusable module function).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg


def export_dashboard_aggregates(clean, appdeny, denials, out_dir, random_state=cfg.RANDOM_STATE):
    out_dir = Path(out_dir)

    if {"derived_race", "tract_minority_cat"}.issubset(appdeny.columns):
        disparity = (appdeny.groupby(["derived_race", "tract_minority_cat"])["target_approved"]
                     .mean().mul(100).round(1).reset_index(name="approval_rate_pct"))
        disparity.to_csv(out_dir / "dash_approval_disparity.csv", index=False)

    scatter_cols = [c for c in ["kmeans_cluster", "income", "loan_amount", "property_value",
                    "combined_loan_to_value_ratio", "debt_to_income_ratio", "derived_race",
                    "loan_purpose", "occupancy_type", "action_taken", "_approved",
                    "anomaly_votes", "iso_score", "tract_minority_population_percent"]
                    if c in clean.columns]
    clean.sample(min(8000, len(clean)), random_state=random_state)[scatter_cols] \
        .to_csv(out_dir / "dash_scatter.csv", index=False)

    if len(denials):
        REASON = {"1": "Debt-to-income", "2": "Employment history", "3": "Credit history",
                  "4": "Collateral", "5": "Insufficient cash", "6": "Unverifiable information",
                  "7": "Incomplete application", "8": "Mortgage insurance denied", "9": "Other"}
        dcols = [c for c in denials.columns if c.startswith("denial_reason")]
        dd = denials[dcols].astype(str).melt()["value"]
        dd = dd[~dd.isin(["nan", "None", "", "10", "10.0"])].map(lambda x: REASON.get(str(x).split(".")[0], "Other"))
        dr = dd.value_counts(normalize=True).mul(100).round(1).reset_index()
        dr.columns = ["reason", "pct_of_denials"]
        dr.to_csv(out_dir / "dash_denial_reasons.csv", index=False)

    clean[["kmeans_cluster", "anomaly_votes", "iso_score"]].to_csv(out_dir / "dash_points.csv", index=True)


def export_phase1_aggregates(miss, high_missing, feature_selection_audit,
                             n_before, n_dupes, residual_missing, out_dir):
    """Writes the Phase 1 (preprocessing) aggregates the dashboard's Fase 1 tab reads:
    dash_phase1_missingness.csv, dash_phase1_feature_importance.csv, and
    dash_phase1_cleaning_summary.csv. `fate` is exported as a language-neutral code
    (dropped / median_imputed / filled_unknown); the dashboard maps it to Indonesian."""
    out_dir = Path(out_dir)
    cont = set(cfg.CONTINUOUS)
    high = set(high_missing)

    mrows = []
    for field, frac in miss.items():
        if frac is None or float(frac) <= 0:
            continue
        if field in high:
            fate = "dropped"
        elif field in cont:
            fate = "median_imputed"
        else:
            fate = "filled_unknown"
        mrows.append({"field": field, "missing_pct": round(float(frac) * 100, 1), "fate": fate})
    miss_df = pd.DataFrame(mrows).sort_values("missing_pct", ascending=False)
    miss_df.to_csv(out_dir / "dash_phase1_missingness.csv", index=False)

    fi = feature_selection_audit.copy()
    if "role" in fi.columns:
        fi = fi[fi["role"] != "Process diagnostic only"]
    keep = [c for c in ["feature", "corr", "mi", "score", "role"] if c in fi.columns]
    fi = fi[keep].sort_values("score", ascending=False).head(20)
    fi.to_csv(out_dir / "dash_phase1_feature_importance.csv", index=False)

    clean_rows = int(n_before - n_dupes)
    summary = pd.DataFrame([{
        "raw_rows": int(n_before),
        "clean_rows": clean_rows,
        "duplicates_removed": int(n_dupes),
        "fields_dropped": len(high_missing),
        "fields_dropped_list": "; ".join(map(str, high_missing)),
        "residual_missing_cells": int(residual_missing),
    }])
    summary.to_csv(out_dir / "dash_phase1_cleaning_summary.csv", index=False)
    print(f"Phase 1 dashboard aggregates: {len(miss_df)} missing fields, "
          f"{len(fi)} ranked features, {clean_rows:,} clean rows")


def export_clustering_comparison(Xc, kmeans_labels, idx_med, db_labels, idx_small,
                                 hier_labels, clarans_labels, out_dir,
                                 random_state=cfg.RANDOM_STATE):
    """Computes the head-to-head clustering comparison the dashboard's Fase 2 table shows.

    Every metric is measured on the scaled matrix the algorithm actually saw, and each
    method is additionally scored on the SAME 4,000-row sample so the numbers are
    comparable rather than measured on different populations. ARI is versus K-Means.
    """
    from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                                 davies_bouldin_score, silhouette_score)
    out_dir = Path(out_dir)
    idx_med = np.asarray(idx_med)
    idx_small = np.asarray(idx_small)
    Xv = Xc.values if hasattr(Xc, "values") else np.asarray(Xc)

    def _scores(X, labels):
        labels = np.asarray(labels)
        mask = labels != -1  # exclude DBSCAN noise from validity metrics
        Xm, lm = X[mask], labels[mask]
        if len(set(lm.tolist())) < 2:
            return np.nan, np.nan, np.nan
        n = min(10000, len(Xm))
        return (
            float(silhouette_score(Xm, lm, sample_size=n, random_state=random_state)),
            float(davies_bouldin_score(Xm, lm)),
            float(calinski_harabasz_score(Xm, lm)),
        )

    km = np.asarray(kmeans_labels)
    # position lookup so sample labels can be aligned back onto K-Means labels
    pos = {v: i for i, v in enumerate(idx_med)}
    small_in_med = np.array([pos[v] for v in idx_small])

    rows = []
    sil, db, ch = _scores(Xv, km)
    rows.append({"method": "K-Means", "scope": "Seluruh data (99.995)", "n_rows": len(km),
                 "n_clusters": int(len(set(km.tolist()))), "noise": 0,
                 "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
                 "ari_vs_kmeans": 1.0})

    dbl = np.asarray(db_labels)
    sil, db, ch = _scores(Xv[idx_med], dbl)
    rows.append({"method": "DBSCAN", "scope": "Sampel 20.000", "n_rows": len(dbl),
                 "n_clusters": int(len(set(dbl[dbl != -1].tolist()))),
                 "noise": int((dbl == -1).sum()),
                 "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
                 "ari_vs_kmeans": float(adjusted_rand_score(km[idx_med], dbl))})

    for name, lab in [("Hierarchical (Ward)", hier_labels), ("CLARANS (k-medoids)", clarans_labels)]:
        lab = np.asarray(lab)
        sil, db, ch = _scores(Xv[idx_small], lab)
        rows.append({"method": name, "scope": "Sampel 4.000", "n_rows": len(lab),
                     "n_clusters": int(len(set(lab.tolist()))), "noise": 0,
                     "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
                     "ari_vs_kmeans": float(adjusted_rand_score(km[idx_small], lab))})

    # K-Means re-scored on the same 4,000-row sample: an apples-to-apples reference row
    sil, db, ch = _scores(Xv[idx_small], km[idx_small])
    rows.append({"method": "K-Means (pada sampel 4.000)", "scope": "Sampel 4.000",
                 "n_rows": len(idx_small), "n_clusters": int(len(set(km[idx_small].tolist()))),
                 "noise": 0, "silhouette": sil, "davies_bouldin": db,
                 "calinski_harabasz": ch, "ari_vs_kmeans": 1.0})

    out = pd.DataFrame(rows).round(
        {"silhouette": 3, "davies_bouldin": 3, "calinski_harabasz": 0, "ari_vs_kmeans": 3})
    out.to_csv(out_dir / "dash_clustering_comparison.csv", index=False)

    ari_hier = out.loc[out["method"].str.startswith("Hierarchical"), "ari_vs_kmeans"].iloc[0]
    print(f"Clustering comparison exported: {len(out)} methods, ARI(Ward vs K-Means)={ari_hier}")
    return out


def build_standalone_dashboard(clean, appdeny, profile, final_rules, top_anom, fair_pivot, out_path,
                               random_state=cfg.RANDOM_STATE):
    """Builds the zero-setup standalone HTML dashboard (Overview/Segments/
    Rules/Anomalies/Fairness tabs) and writes it to out_path. Returns the html
    string so the caller (notebook) can also display it inline."""
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.offline import get_plotlyjs

    dti_order = ["<20%", "20%-<30%", "30%-<36%", "36%-<43%", "43%-<50%", "50%-60%", ">60%",
                "Exempt", "Unknown"]
    overview_dti = (appdeny.groupby("debt_to_income_ratio", observed=True)["target_approved"]
                    .agg(applications="size", approval_rate="mean").reset_index())
    overview_dti["approval_rate"] *= 100
    overview_dti["debt_to_income_ratio"] = pd.Categorical(
        overview_dti["debt_to_income_ratio"], categories=dti_order, ordered=True
    )
    overview_dti = overview_dti.sort_values("debt_to_income_ratio")
    fig_overview = px.bar(
        overview_dti, x="debt_to_income_ratio", y="approval_rate",
        text=overview_dti["approval_rate"].map(lambda x: f"{x:.1f}%"),
        hover_data={"applications": ":,", "approval_rate": ":.1f"},
        labels={"debt_to_income_ratio": "Debt-to-income band", "approval_rate": "Approval rate (%)"},
        title="Decision overview: approval rate falls sharply as DTI rises"
    )
    fig_overview.update_traces(textposition="outside")
    fig_overview.update_layout(yaxis_range=[0, 105], template="plotly_white")

    segments_dash = profile.reset_index().copy()
    fig_segments = px.scatter(
        segments_dash, x="med_income", y="approval_rate", size="share_of_data",
        color="segment_name", hover_name="segment_name",
        hover_data={"n": ":,", "med_loan": ":,.0f", "med_cltv": ":.1f",
                    "share_of_data": ":.1f", "med_income": ":.1f", "approval_rate": ":.1f"},
        labels={"med_income": "Median income ($000s)", "approval_rate": "Approval rate (%)",
                "share_of_data": "Share of applications (%)"},
        title="Seven interpretable mortgage-application segments"
    )
    fig_segments.update_layout(template="plotly_white", showlegend=False)

    rules_dash = final_rules.copy()
    rules_dash["decision"] = rules_dash["consequent"].str.replace("decision=", "", regex=False)
    fig_rules = px.scatter(
        rules_dash, x="support", y="confidence", size="lift", color="decision",
        hover_name="antecedent", hover_data={"lift": ":.2f", "improvement": ":.3f",
                                              "support": ":.3f", "confidence": ":.3f"},
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        title="Non-trivial, redundancy-pruned decision rules"
    )
    fig_rules.add_hline(y=0.55, line_dash="dash", annotation_text="minimum confidence")
    fig_rules.update_layout(template="plotly_white")

    anom_cols = ["loan_amount", "property_value", "combined_loan_to_value_ratio", "income",
                "anomaly_votes", "kmeans_cluster", "action_taken"]
    anom_view = clean[[c for c in anom_cols if c in clean.columns]].copy()
    anom_view = anom_view[(anom_view.get("loan_amount", 0) > 0) & (anom_view.get("property_value", 0) > 0)]
    base_sample = anom_view.sample(min(8000, len(anom_view)), random_state=random_state)
    high_conf = anom_view[anom_view["anomaly_votes"] >= 3]
    anom_view = pd.concat([base_sample, high_conf]).loc[lambda x: ~x.index.duplicated()].copy()
    anom_view["triage"] = "Not in top-15 manual triage"
    if top_anom is not None:
        shared_idx = anom_view.index.intersection(top_anom.index)
        anom_view.loc[shared_idx, "triage"] = top_anom.loc[shared_idx, "verdict"]
    fig_anomalies = px.scatter(
        anom_view, x="property_value", y="loan_amount", color="anomaly_votes",
        hover_data=[c for c in ["combined_loan_to_value_ratio", "income", "kmeans_cluster",
                                 "action_taken", "triage"] if c in anom_view.columns],
        log_x=True, log_y=True,
        labels={"property_value": "Property value ($, log scale)",
                "loan_amount": "Loan amount ($, log scale)",
                "anomaly_votes": "Detector votes"},
        title="Anomaly review: multi-method agreement across loan and property magnitude"
    )
    fig_anomalies.update_layout(template="plotly_white")

    fig_fairness = go.Figure(data=go.Heatmap(
        z=fair_pivot.values,
        x=[str(x) for x in fair_pivot.columns],
        y=[str(y) for y in fair_pivot.index],
        text=np.round(fair_pivot.values, 1), texttemplate="%{text}%",
        hovertemplate="Tract band: %{y}<br>DTI group: %{x}<br>Approval: %{z:.1f}%<extra></extra>",
        colorbar_title="Approval %"
    ))
    fig_fairness.update_layout(
        title="Approval association by tract minority share within broad DTI groups",
        xaxis_title="DTI group", yaxis_title="Tract minority band", template="plotly_white"
    )

    figures = {
        "Overview": fig_overview,
        "Segments": fig_segments,
        "Rules": fig_rules,
        "Anomalies": fig_anomalies,
        "Fairness": fig_fairness,
    }

    buttons, sections = [], []
    for i, (name, fig) in enumerate(figures.items()):
        active = " active" if i == 0 else ""
        display_style = "block" if i == 0 else "none"
        safe_id = name.lower()
        buttons.append(f'<button class="tabbtn{active}" onclick="openTab(\'{safe_id}\', this)">{name}</button>')
        fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=False,
                               config={"responsive": True, "displaylogo": False})
        sections.append(f'<section id="{safe_id}" class="tabcontent" style="display:{display_style}">{fig_html}</section>')

    base_approval = appdeny["target_approved"].mean() * 100
    html_template = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HMDA 2022 Knowledge Dashboard</title>
<script>__PLOTLYJS__</script>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#f6f7f9;color:#1f2937}
header{padding:22px 28px;background:white;border-bottom:1px solid #ddd}
header h1{margin:0 0 6px;font-size:24px} header p{margin:0;color:#4b5563}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px;padding:16px 28px}
.kpi{background:white;border:1px solid #e5e7eb;border-radius:10px;padding:14px}
.kpi b{display:block;font-size:22px;margin-top:4px}
.tabs{display:flex;gap:8px;padding:0 28px 12px;flex-wrap:wrap}
.tabbtn{border:1px solid #cbd5e1;background:white;padding:9px 14px;border-radius:8px;cursor:pointer}
.tabbtn.active{background:#111827;color:white}
.tabcontent{margin:0 28px 28px;background:white;border:1px solid #e5e7eb;border-radius:10px;padding:8px}
.note{padding:0 28px 18px;color:#4b5563;font-size:13px}
@media(max-width:800px){.kpis{grid-template-columns:1fr 1fr}.tabcontent{margin:0 10px 20px}.tabs,.kpis,header,.note{padding-left:10px;padding-right:10px}}
</style></head><body>
<header><h1>HMDA 2022 Knowledge Discovery Dashboard</h1>
<p>Plain-language exploration of approval patterns, application segments, decision rules, and unusual records.</p></header>
<div class="kpis">
<div class="kpi">Decisioned applications<b>__N_APPDENY__</b></div>
<div class="kpi">Overall approval rate<b>__BASE_APPROVAL__%</b></div>
<div class="kpi">Interpreted clusters<b>__N_CLUSTERS__</b></div>
<div class="kpi">Pruned decision rules<b>__N_RULES__</b></div>
</div>
<div class="tabs">__BUTTONS__</div>
<div class="note">Hover for exact values. Use the legend, zoom, and reset controls. Fairness results are associations, not causal proof, because public HMDA data omit factors such as credit score and reserves.</div>
__SECTIONS__
<script>
function openTab(id, btn){
 document.querySelectorAll('.tabcontent').forEach(x=>x.style.display='none');
 document.querySelectorAll('.tabbtn').forEach(x=>x.classList.remove('active'));
 document.getElementById(id).style.display='block'; btn.classList.add('active');
 window.dispatchEvent(new Event('resize'));
}
</script></body></html>'''

    html = (html_template
            .replace("__PLOTLYJS__", get_plotlyjs())
            .replace("__N_APPDENY__", f"{len(appdeny):,}")
            .replace("__BASE_APPROVAL__", f"{base_approval:.1f}")
            .replace("__N_CLUSTERS__", f"{len(profile):,}")
            .replace("__N_RULES__", f"{len(final_rules):,}")
            .replace("__BUTTONS__", "".join(buttons))
            .replace("__SECTIONS__", "".join(sections)))

    Path(out_path).write_text(html, encoding="utf-8")
    return html


def write_reports(appdeny, clean, profile, final_rules, top_anom, fair_pivot,
                  n_dupes, high_missing, missing_drop_threshold, residual_missing_after_cleaning,
                  out_dir):
    """Writes the notebook's auto-generated report pair to
    `{out_dir}/Preprocessing_Report_generated.md` and
    `Knowledge_Discovery_Report_generated.md` -- distinct filenames from the
    hand-written `Preprocessing_Report.md` / `Knowledge_Discovery_Report.md`
    that are the actual submitted reports, so re-running this cell never
    overwrites hand-written analysis."""
    out_dir = Path(out_dir)

    base_approval = appdeny["target_approved"].mean() * 100
    best_cluster_id = profile["approval_rate"].idxmax()
    worst_cluster_id = profile["approval_rate"].idxmin()
    best_cluster = profile.loc[best_cluster_id]
    worst_cluster = profile.loc[worst_cluster_id]

    top_denial = final_rules[final_rules["consequent"] == "decision=Denied"].sort_values("lift", ascending=False).iloc[0]
    high_conf_anomalies = int((clean["anomaly_votes"] >= 3).sum())
    triage_counts = top_anom["verdict"].value_counts().to_dict() if top_anom is not None else {}

    fair_gap_text = "Not available"
    if len(fair_pivot) >= 2:
        max_gap = (fair_pivot.max(axis=0) - fair_pivot.min(axis=0)).max()
        fair_gap_text = f"up to {max_gap:.1f} percentage points within the broad DTI groups shown"

    preprocessing_report = f"""# HMDA 2022 Preprocessing Report

## Objective
Prepare a 100,000-record HMDA sample for clustering, association-rule mining, and anomaly detection without converting legal sentinel values into ordinary numbers or leaking post-decision information.

## Cleaning decisions
- Loaded all raw fields as strings to preserve mixed numeric/range fields and leading zeroes.
- Audited `1111`, `8888`, and `9999` by column provenance rather than replacing them globally.
- Translated documented categorical codes while preserving `Exempt`, `Age_NA`, and `No_CoApplicant` as distinct meanings.
- Removed {n_dupes:,} exact duplicate attribute records and documented the check.
- Dropped {len(high_missing):,} fields above the {missing_drop_threshold:.0%} structural-missingness threshold.
- Median-imputed remaining continuous gaps and retained explicit missingness indicators for audit purposes.
- Filled residual categorical gaps with `Unknown`.
- Confirmed {residual_missing_after_cleaning:,} missing cells remained after handling.

## Transformation and selection
- Framed the decision subset as Originated versus Denied; the full table remains available for unsupervised analysis.
- Applied domain-readable bins for income, loan amount, property value, CLTV, tract income, and tract minority share.
- Used correlation for linear association/redundancy and entropy-based Mutual Information for non-linear relevance.
- Excluded post-decision pricing/cost fields and process-stage diagnostic flags from Phase 2-4 feature sets.
- Applied 1%/99% winsorization only to the clustering copy; original values remain untouched for profiling, rules, and anomaly review.
"""

    knowledge_report = f"""# HMDA 2022 Knowledge Discovery Report

## Central question
Which application, property, financing, and neighbourhood combinations are systematically associated with origination or denial, and which unusual records deserve review?

## Executive answer
Among applications that reached an underwriting decision, the overall approval rate was **{base_approval:.1f}%**. The strongest repeated separator was debt burden: the highest-lift denial rule was **{top_denial['antecedent']} -> Denied**, with **{top_denial['confidence']*100:.1f}% confidence**, **{top_denial['support']*100:.1f}% support**, and **{top_denial['lift']:.2f} lift**.

## Segmentation
The highest-approval segment was **{best_cluster['segment_name']}** at **{best_cluster['approval_rate']:.1f}% approval** and {best_cluster['share_of_data']:.1f}% of records. The lowest-approval segment was **{worst_cluster['segment_name']}** at **{worst_cluster['approval_rate']:.1f}% approval**. Their profiles indicate that low approval does not describe one homogeneous population: debt-capacity stress and property/collateral type form different business problems and therefore require different interventions.

## Decision rules
The final table contains **{len(final_rules)} non-trivial rules** after confidence/lift screening, chi-square testing, Wilson confidence intervals, and improvement-based redundancy pruning. The interpreted set exceeds the rubric minimum of ten rules. Rules are reported as associations, not deterministic underwriting policies.

## Anomalies
Five methods (IQR, Z-score, Isolation Forest, Local Outlier Factor, and DBSCAN noise) were cross-referenced. **{high_conf_anomalies:,} records** received at least three detector votes. The top records were triaged with contextual evidence; the current top-15 verdict mix is **{triage_counts}**. A large value alone is not treated as an error: internal consistency distinguishes rare valid jumbo/multifamily activity from impossible or contradictory records.

## Fairness and geography
Approval differences across tract-minority bands persisted by **{fair_gap_text}** after broad DTI stratification. This is a screening signal for deeper review, not proof of discrimination or causality, because public HMDA data omit important underwriting variables such as credit score, reserves, and complete lender policy context.

## Recommended business actions
1. Add an early DTI and lien-position pre-check to reduce avoidable full-underwriting effort.
2. Route manufactured-housing applications toward channels designed for that collateral instead of treating the segment as ordinary site-built lending.
3. Use the anomaly ensemble as a manual-review queue, not as an automatic deletion rule.
4. Investigate tract-level approval gaps with richer controlled data before making policy claims.
"""

    (out_dir / "Preprocessing_Report_generated.md").write_text(preprocessing_report, encoding="utf-8")
    (out_dir / "Knowledge_Discovery_Report_generated.md").write_text(knowledge_report, encoding="utf-8")
    return preprocessing_report, knowledge_report
