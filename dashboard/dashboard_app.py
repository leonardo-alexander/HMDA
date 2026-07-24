#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import networkx as nx
from dash import Dash, dcc, html, Input, Output, dash_table, no_update

# Data files live one level up, split into interim (Phase 1 exports) and processed
# (Phase 2-4 exports + dashboard aggregates). Checked in this order since most reads
# hit processed/; interim/ only holds the 3 raw hmda_*.csv files.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = [_PROJECT_ROOT / "data" / "processed", _PROJECT_ROOT / "data" / "interim"]

# ============================================================ DESIGN TOKENS
# Colours are the validated data-viz reference palette (categorical hues cleared the
# CVD + normal-vision gates; run scripts/validate_palette.js to reproduce). Chrome is a
# cool, professional finance theme built on a deep navy + steel-blue accent.
NAVY, STEEL, TEAL = "#14294a", "#2a78d6", "#0f9d78"
GREEN, AMBER, RED = "#15803d", "#d97706", "#c02b2b"
BG, CARD, INK, MUTE = "#eef2f6", "#ffffff", "#1a202c", "#64748b"
GRID, AXIS, BORDER = "#e8edf3", "#cbd5e1", "rgba(16,42,74,0.06)"
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"
# Validated 8-hue categorical order (fixed, never cycled) for discrete series.
QUAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = [[0.0, "#e3eefb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]
VERDICT_COLOR = {"DATA ERROR": RED, "RARE BUT VALID": GREEN, "RISK SIGNAL": AMBER, "MANUAL REVIEW": STEEL}

# One registered Plotly template so every figure inherits the same font, gridlines,
# transparent surface (blends into its white card), colourway and hover style.
pio.templates["hmda"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=12, color=INK),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=QUAL,
    dragmode=False,  # no drag-to-zoom / box-select on any figure (hover still works)
    title=dict(font=dict(family=FONT, size=14.5, color=NAVY), x=0.01, xanchor="left", pad=dict(b=6)),
    # automargin: Plotly reserves separate space for tick labels vs. the axis title and
    # grows the figure's margin to fit them, instead of letting long category labels
    # (segment names, race categories, DTI bands) clip against the card edge or collide
    # with the title. This is the general fix; specific charts still tune angle/margin.
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=GRID, tickcolor=AXIS, fixedrange=True,
               automargin=True, tickfont=dict(color=MUTE, size=11),
               title=dict(font=dict(color=MUTE, size=12), standoff=10)),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=GRID, tickcolor=AXIS, fixedrange=True,
               automargin=True, tickfont=dict(color=MUTE, size=11),
               title=dict(font=dict(color=MUTE, size=12), standoff=10)),
    legend=dict(font=dict(color=MUTE, size=11), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=NAVY, bordercolor="rgba(0,0,0,0)",
                    font=dict(family=FONT, color="white", size=12)),
    margin=dict(l=12, r=12, t=46, b=12),
))
TEMPLATE = "hmda"

def blank(msg="Run the notebook first: data file not found."):
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=MUTE))
    f.update_layout(template=TEMPLATE, height=360, xaxis_visible=False, yaxis_visible=False)
    return f

def read(path, **kw):
    for d in DATA_DIRS:
        candidate = d / path
        if candidate.exists():
            try:
                return pd.read_csv(candidate, **kw)
            except Exception:
                return None
    return None

# ============================================================ SCHEMA NORMALIZATION
# The notebook's own Phase 3/4 export cells (HMDA.ipynb) write a minimal schema
# (antecedent, consequent, n, confidence, lift, ...). A separate enrichment pass can
# add reader-friendly columns (if_readable, then, recommendation, outlier_type, ...),
# but the dashboard must not *require* that pass to have run. Whichever CSV shows up
# in this folder (fresh from Colab, a local notebook run, or the enriched build
# scripts), the normalizers below fill in anything missing so nothing downstream KeyErrors.
TOTAL_DECISIONED = 67827  # size of the approve_deny mining table (Phase 1 - decisioned applications)

ITEM_LABELS = {
    "debt_to_income_ratio=>60%": "debt-to-income ratio is above 60%",
    "debt_to_income_ratio=50%-60%": "debt-to-income ratio is 50-60%",
    "debt_to_income_ratio=30%-<36%": "debt-to-income ratio is 30-36%",
    "lien_status=Subordinate_Lien": "the loan is a subordinate (second) lien",
    "loan_type=Conventional": "financed through a conventional loan",
    "construction_method=Manufactured": "the property is manufactured housing",
    "loan_purpose=Home_Purchase": "the purpose is a home purchase",
    "income_band=<30k": "applicant income is under $30k",
    "preapproval=Requested": "the applicant requested preapproval",
    "loan_amount_band=300-500k": "the loan amount is $300k-500k",
}

RECS = {
    frozenset(["debt_to_income_ratio=>60%", "lien_status=Subordinate_Lien"]):
        "Intercept before full underwriting; route to a debt-consolidation alternative instead of a second lien.",
    frozenset(["debt_to_income_ratio=>60%"]):
        "Flag at intake: lowering DTI below 60% (co-borrower, longer term, smaller loan) is the single highest-leverage fix available.",
    frozenset(["construction_method=Manufactured", "loan_type=Conventional"]):
        "Steer manufactured-home buyers to FHA/VA or a chattel-specific program instead of the conventional channel.",
    frozenset(["debt_to_income_ratio=50%-60%", "loan_type=Conventional"]):
        "Route DTI 50-60% applicants to FHA/VA underwriting, which tolerates this band far better than conventional.",
    frozenset(["construction_method=Manufactured", "loan_purpose=Home_Purchase"]):
        "For manufactured-home purchases, verify land ownership/permanent foundation - titling as real property unlocks standard mortgage programs.",
    frozenset(["income_band=<30k", "loan_type=Conventional"]):
        "Pair sub-$30k-income applicants with down-payment assistance before a conventional application.",
    frozenset(["income_band=<30k"]):
        "Route to manual-underwrite / assistance programs - every other income band approves at majority rates.",
    frozenset(["construction_method=Manufactured"]):
        "Treat manufactured housing as a distinct underwriting track; the penalty is structural, not explained by income or location.",
    frozenset(["preapproval=Requested"]):
        "Promote the preapproval track - it moves rejection to the cheap early stage; files that reach final decision essentially never fail.",
    frozenset(["debt_to_income_ratio=30%-<36%", "loan_purpose=Home_Purchase"]):
        "Fast-track these applications - textbook affordability profile with the collateral validated by an arm's-length sale price.",
    frozenset(["debt_to_income_ratio=30%-<36%", "loan_amount_band=300-500k"]):
        "Fast-track - the conforming-loan sweet spot: large enough to clear fixed costs, small enough to stay inside GSE limits.",
}

OUTLIER_TYPE = {
    "DATA ERROR": "Impossible value (data-entry/processing error)",
    "RARE BUT VALID": "Genuine extreme record (jumbo loan, multi-unit, or high-value investment property)",
    "RISK SIGNAL": "Unusual but plausible risk combination (e.g. high leverage + low income)",
    "MANUAL REVIEW": "Extreme magnitude, reviewed by hand",
}

def _rule_phrase(antecedent):
    items = str(antecedent).split(", ")
    return " and ".join(ITEM_LABELS.get(it, it.replace("_", " ").replace("=", ": ")) for it in items)

def _rule_recommendation(antecedent, then, confidence, n):
    key = frozenset(str(antecedent).split(", "))
    if key in RECS:
        return RECS[key]
    verb = "denied" if then == "Denied" else "originated"
    tail = "flag for review or an alternative program before full underwriting" if then == "Denied" \
        else "a strong candidate for fast-tracking"
    return f"Applications matching this profile are {verb} {confidence*100:.0f}% of the time ({n:,} similar cases); {tail}."

def _normalize_rules(df):
    if df is None or not len(df):
        return df
    d = df.copy()
    if "then" not in d.columns:
        d["then"] = d["consequent"].apply(lambda c: "Denied" if "Denied" in str(c) else "Originated")
    if "if_readable" not in d.columns:
        d["if_readable"] = d["antecedent"].apply(_rule_phrase)
    if "n_matched" not in d.columns:
        d["n_matched"] = d["n"] if "n" in d.columns else 0
    if "support" not in d.columns:
        d["support"] = d["n_matched"] / TOTAL_DECISIONED
    if "recommendation" not in d.columns:
        d["recommendation"] = [
            _rule_recommendation(a, t, c, n) for a, t, c, n in
            zip(d["antecedent"], d["then"], d["confidence"], d["n_matched"])
        ]
    return d

def _normalize_triage(df):
    if df is None or not len(df):
        return df
    d = df.copy()
    if "outlier_type" not in d.columns and "verdict" in d.columns:
        d["outlier_type"] = d["verdict"].map(OUTLIER_TYPE).fillna("Unclassified")
    if "why_flagged" not in d.columns and "anomaly_votes" in d.columns:
        d["why_flagged"] = d["anomaly_votes"].apply(
            lambda v: "Flagged by all 5 anomaly detectors - highest confidence." if v == 5
            else f"Flagged by {int(v)} of 5 anomaly detectors.")
    return d

def _ensure_clarans():
    """Prefer the pre-joined dash_clarans_scatter.csv; otherwise build it on the fly
    from the notebook's raw p2_clarans_assignments.csv (row, clarans_cluster) joined
    against p4_anomaly_flags.csv (same row index, full feature columns)."""
    d = read("dash_clarans_scatter.csv")
    if d is not None:
        return d
    ca = read("p2_clarans_assignments.csv")
    if ca is None:
        return None
    cols_needed = ["kmeans_cluster", "income", "loan_amount", "property_value",
                   "combined_loan_to_value_ratio", "debt_to_income_ratio", "derived_race",
                   "loan_purpose", "occupancy_type", "action_taken", "tract_minority_population_percent"]
    flags = read("p4_anomaly_flags.csv", index_col=0)
    if flags is None:
        return None
    have = [c for c in cols_needed if c in flags.columns]
    if not have:
        return None
    merged = ca.set_index("row").join(flags[have], how="left").dropna(subset=[have[0]])
    return merged.reset_index().rename(columns={"index": "row"})

def _ensure_geo_gap():
    """Prefer the pre-built dash_dti_geography_gap.csv; otherwise recompute the
    within-DTI-band approval gap directly from hmda_approve_deny.csv."""
    d = read("dash_dti_geography_gap.csv")
    if d is not None:
        return d
    ad = read("hmda_approve_deny.csv")
    if ad is None or "tract_minority_cat" not in ad.columns or "debt_to_income_ratio" not in ad.columns:
        return None
    dti_grp_map = {"<20%": "Low(<36%)", "20%-<30%": "Low(<36%)", "30%-<36%": "Low(<36%)",
                   "36%-<43%": "Mid(36-50%)", "43%-<50%": "Mid(36-50%)",
                   "50%-60%": "High(>50%)", ">60%": "High(>50%)"}
    tmp = ad.copy()
    tmp["dti_grp"] = tmp["debt_to_income_ratio"].astype(str).map(dti_grp_map).fillna("Unknown/Exempt")
    if "target_approved" not in tmp.columns and "action_taken" in tmp.columns:
        tmp["target_approved"] = (tmp["action_taken"] == "Originated").astype(int)
    if "target_approved" not in tmp.columns:
        return None
    piv_rate = tmp.pivot_table(index="tract_minority_cat", columns="dti_grp",
                               values="target_approved", aggfunc="mean", observed=True) * 100
    piv_n = tmp.pivot_table(index="tract_minority_cat", columns="dti_grp",
                            values="target_approved", aggfunc="size", observed=True)
    if "Low_Minority" not in piv_rate.index or "Majority_Minority" not in piv_rate.index:
        return None
    rows = []
    for grp in ["Low(<36%)", "Mid(36-50%)", "High(>50%)", "Unknown/Exempt"]:
        if grp not in piv_rate.columns:
            continue
        low, maj = piv_rate.loc["Low_Minority", grp], piv_rate.loc["Majority_Minority", grp]
        rows.append({"dti_group": grp, "low_minority_approval_pct": round(low, 1),
                    "majority_minority_approval_pct": round(maj, 1), "gap_pp": round(low - maj, 1),
                    "n_low_minority": int(piv_n.loc["Low_Minority", grp]),
                    "n_majority_minority": int(piv_n.loc["Majority_Minority", grp])})
    return pd.DataFrame(rows) if rows else None

# LOAD DATA
profiles   = read("p2_cluster_profiles.csv")
rules      = _normalize_rules(read("p3_decision_rules_final.csv"))   # pruned, business-relevant set (primary)
rules_all  = _normalize_rules(read("p3_decision_rules.csv"))         # all 28 raw candidates, same schema
clarans    = _ensure_clarans()                                       # CLARANS medoid-sample comparison
dbscan_scatter = read("dash_dbscan_scatter.csv")                     # DBSCAN cluster ids (-1 = noise), 20k sample
hier_scatter   = read("dash_hierarchical_scatter.csv")                # Ward hierarchical cluster ids, 4k sample
outlier_tax    = read("dash_outlier_taxonomy.csv")                    # per-record global/contextual outlier labels
outlier_tax_summary = read("dash_outlier_taxonomy_summary.csv")       # full-population category counts (not sampled)
collective_pattern  = read("dash_collective_pattern.csv")             # loan-amount-rounding collective-outlier evidence
triage     = _normalize_triage(read("p4_anomaly_triage.csv", index_col=0))  # top-15 anomalies, verdict + evidence
geo_gap    = _ensure_geo_gap()                                       # approval gap by tract minority x DTI band
disparity  = read("dash_approval_disparity.csv")
scatter    = read("dash_scatter.csv")
denial     = read("dash_denial_reasons.csv")

# Which of the 28 raw candidates also survived the improvement filter into the 11-rule
# business set - shown as a column on the "all candidates" table so the two views are
# directly comparable rather than living in unrelated tables.
if rules_all is not None and rules is not None:
    _kept_keys = set(zip(rules["antecedent"], rules["consequent"]))
    rules_all["kept"] = [("Yes" if k in _kept_keys else "No")
                         for k in zip(rules_all["antecedent"], rules_all["consequent"])]
elif rules_all is not None:
    rules_all["kept"] = "No"
state_summary = read("dash_state_summary.csv")   # state_code, n, approval_rate, median_income, median_loan, ...
state_dti     = read("dash_state_dti.csv")        # state_code x DTI band -> approval_rate, n
state_segment = read("dash_state_segment.csv")    # state_code x kmeans_cluster -> n
term_summary  = read("dash_term_summary.csv")     # loan_term_band -> n, approval_rate, ... (duration proxy for "time")
context_fields = read("dash_context_fields.csv")  # field, value, n, approval_rate (non-demographic What-If context)
TERM_ORDER = ["<=10yr", "15yr", "20yr", "25yr", "30yr", ">30yr"]
if scatter is None:
    full = read("p4_anomaly_flags.csv")
    if full is not None:
        scatter = full.sample(min(8000, len(full)), random_state=42)

# SELF-LABELLING
def name_clusters(p):
    """Use the notebook's own business naming (segment_name in p2_cluster_profiles.csv)
    instead of a second, separate heuristic. A simpler local rule that only checks
    manufactured/DTI/investment/income/loan-size/tract-minority falls through to the
    same generic bucket for any two "normal" clusters that differ mainly in refinance
    vs. purchase mix (e.g. C0 and C2), producing misleadingly identical labels even
    though the underlying segments are very different (94.5% refinance vs. 0%, 79.7%
    vs. 91.5% approval). The notebook's naming already checks that distinction."""
    if p is None:
        return {}
    return {int(r["kmeans_cluster"]): f"C{int(r['kmeans_cluster'])} - {r['segment_name']}"
            for _, r in p.iterrows()}

CLUSTER_NAMES = name_clusters(profiles)
def clabel(cid):
    try:
        return CLUSTER_NAMES.get(int(cid), f"C{int(cid)}")
    except Exception:
        return str(cid)

# Business-framed one-line description per segment archetype (keyed off the notebook's
# own auto-generated segment_name, so it tracks whatever names this run produced).
SEGMENT_BLURB = {
    "Refinancers (rate & cash-out)":
        "Existing homeowners refinancing for rate or pulling cash out. Already proven mortgage payers, "
        "so approval tracks close to the portfolio norm: the underwriting question is equity, not creditworthiness.",
    "Property investors":
        "100% investment-occupancy applications. Approval is solid but slightly below prime purchasers, since "
        "lenders price in that investment properties are the first payment skipped in a downturn.",
    "Mainstream prime purchasers":
        "The largest segment and the approval benchmark for the whole portfolio. Near-zero risk flags: "
        "this is what a \"clean\" application looks like.",
    "Manufactured-housing applicants":
        "Defined by property type, not income or leverage. Approval collapses to well under half. "
        "See the Rules tab: this penalty is structural (financing channel + transaction type), not explained by income.",
    "DTI-stressed borrowers":
        "Every application here carries debt-to-income above 50%. The lowest approval rate in the portfolio, "
        "driven almost entirely by one number: see the DTI>60% rule on the Rules tab.",
    "Jumbo / high-net-worth buyers":
        "Median income roughly 5x the portfolio median. Approval is strong despite large loan sizes: "
        "capacity to repay is not in question for this group.",
    "Small-loan borrowers":
        "The smallest median loan size in the portfolio (home-improvement-scale). Approval sits close to "
        "the norm: small dollar amounts carry proportionally small risk.",
}
def segment_blurb(row):
    return SEGMENT_BLURB.get(row["segment_name"],
        f"{row['share_of_data']:.0f}% of applications, {row['approval_rate']:.0f}% approval rate.")

# How to approach or handle each segment: one concrete, actionable recommendation per
# cluster, in the same spirit as the "Business recommendation" column on the Rules tab.
CLUSTER_RECS = {
    "Refinancers (rate & cash-out)":
        "Give this segment a dedicated refinance underwriting track (rate-and-term vs. cash-out) instead of "
        "routing it through purchase underwriting. Approval already tracks the portfolio norm, so the lever "
        "here is retention and cross-sell (home-equity products), with extra scrutiny on cash-out equity extraction.",
    "Property investors":
        "Offer a DSCR or portfolio-lender program built for landlords instead of standard owner-occupied "
        "underwriting. Approval is already solid, so the main risk to manage is concentration if this book grows fast.",
    "Mainstream prime purchasers":
        "Protect this segment's speed above all else: it is the largest, lowest-friction, highest-approval "
        "population and the natural baseline for testing any new underwriting rule before rolling it out elsewhere.",
    "Manufactured-housing applicants":
        "Build or partner into a chattel-lending or FHA Title I product for manufactured homes rather than "
        "forcing this segment through conventional underwriting, where the penalty is structural, not risk-based "
        "(see Rules tab). Track fair-lending exposure, since property type here correlates with income and geography.",
    "DTI-stressed borrowers":
        "Screen debt-to-income at intake, before full underwriting, and redirect near-certain denials to a "
        "debt-consolidation or credit-counseling referral. A \"declined but here is a path\" offer (secured card, "
        "budgeting tool) keeps the relationship alive for a stronger future application.",
    "Jumbo / high-net-worth buyers":
        "Treat as a relationship-banking segment: approval is already strong, so the opportunity is service and "
        "cross-sell (private banking, portfolio lending) rather than underwriting changes. Watch investor appetite, "
        "since jumbo loans are harder to sell to the GSEs than conforming ones.",
    "Small-loan borrowers":
        "Build a streamlined, lower-cost underwriting track (e.g. AVM instead of full appraisal) for small-dollar "
        "loans; approval is already near the portfolio average, so the real constraint is origination cost relative "
        "to loan size, not credit risk.",
}
def cluster_recommendation(row):
    return CLUSTER_RECS.get(row["segment_name"],
        "No specific playbook yet for this segment; monitor approval rate and volume before treating it "
        "differently from the portfolio norm.")

# KPI COMPUTATIONS
def kpis():
    n_apps = int(profiles["n"].sum()) if profiles is not None else (len(scatter) if scatter is not None else 0)
    if scatter is not None and "_approved" in scatter and scatter["_approved"].notna().any():
        # decisioned-only rate (Originated / (Originated+Denied)); profiles["n"] includes
        # withdrawn/incomplete rows too, so weighting by it would overstate the denominator.
        appr = float(scatter["_approved"].mean() * 100)
    elif profiles is not None and "approval_rate" in profiles:
        appr = float(np.average(profiles["approval_rate"], weights=profiles["n"]))
    else:
        appr = float("nan")
    n_rules = len(rules) if rules is not None else 0
    max_lift = float(rules["lift"].max()) if rules is not None and len(rules) else float("nan")
    n_clusters = profiles.shape[0] if profiles is not None else 0
    if scatter is not None and "anomaly_votes" in scatter:
        n_anom = int((scatter["anomaly_votes"] >= 3).sum())
    else:
        n_anom = 0
    return [
        ("Applications analysed", f"{n_apps:,}", STEEL),
        ("Approval rate", f"{appr:.1f}%" if appr == appr else "N/A", GREEN),
        ("Denial rate", f"{100-appr:.1f}%" if appr == appr else "N/A", RED),
        ("Segments found", f"{n_clusters}", TEAL),
        ("Business-relevant rules", f"{n_rules}", NAVY),
        ("Top rule lift", f"{max_lift:.1f}x" if max_lift == max_lift else "N/A", AMBER),
        ("High-conf. anomalies", f"{n_anom:,}", "#7d3c98"),
    ]

# ============================================================ FIGURE BUILDERS
def fig_approval_by_cluster():
    if profiles is None:
        return blank()
    d = profiles.copy()
    d["name"] = d["kmeans_cluster"].map(clabel)
    d = d.sort_values("approval_rate")
    f = px.bar(d, x="approval_rate", y="name", orientation="h",
               color="approval_rate", color_continuous_scale=["#c0392b", "#e0a82e", "#2e8b57"],
               text=d["approval_rate"].map(lambda v: f"{v:.0f}%"),
               labels={"approval_rate": "Approval rate (%)", "name": ""})
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(template=TEMPLATE, height=380, coloraxis_showscale=False,
                    margin=dict(l=10, r=60, t=30, b=10),
                    title="Approval rate differs sharply by segment")
    f.update_xaxes(range=[0, 112])
    f.update_yaxes(automargin=True, tickfont=dict(size=11.5))
    return f

def fig_cluster_sizes():
    if profiles is None:
        return blank()
    d = profiles.copy()
    d["name"] = d["kmeans_cluster"].map(clabel)
    f = px.pie(d, values="n", names="name", hole=0.45, color_discrete_sequence=QUAL)
    f.update_traces(textposition="inside", textinfo="percent")
    f.update_layout(template=TEMPLATE, height=380, title="Share of applications by segment",
                    margin=dict(l=10, r=10, t=40, b=10), legend=dict(font=dict(size=9)))
    return f

NUMERIC_AXES = {
    "income": "Income ($k)", "loan_amount": "Loan amount ($)",
    "property_value": "Property value ($)", "combined_loan_to_value_ratio": "CLTV (%)",
    "tract_minority_population_percent": "Tract minority population (%)",
}
def fig_cluster_scatter(xcol, ycol):
    if scatter is None or "kmeans_cluster" not in scatter:
        return blank()
    d = scatter.dropna(subset=[xcol, ycol]).copy()
    d = d[(d[xcol] > 0) & (d[ycol] > 0)] if xcol in ("loan_amount", "property_value", "income") else d
    d["Segment"] = d["kmeans_cluster"].map(clabel)
    logx = xcol in ("loan_amount", "property_value", "income")
    logy = ycol in ("loan_amount", "property_value", "income")
    f = px.scatter(d.sample(min(6000, len(d)), random_state=1), x=xcol, y=ycol, color="Segment",
                   color_discrete_sequence=QUAL, opacity=0.55, log_x=logx, log_y=logy,
                   labels={xcol: NUMERIC_AXES.get(xcol, xcol), ycol: NUMERIC_AXES.get(ycol, ycol)})
    f.update_traces(marker=dict(size=5))
    f.update_layout(template=TEMPLATE, height=460, title="K-Means: segments in feature space",
                    legend=dict(font=dict(size=9)), margin=dict(l=10, r=10, t=40, b=10))
    return f

def fig_clarans_scatter(xcol, ycol):
    if clarans is None:
        return blank("CLARANS comparison data not found.")
    d = clarans.dropna(subset=[xcol, ycol]).copy()
    d = d[(d[xcol] > 0) & (d[ycol] > 0)] if xcol in ("loan_amount", "property_value", "income") else d
    d["CLARANS cluster"] = "M" + d["clarans_cluster"].astype(str)
    logx = xcol in ("loan_amount", "property_value", "income")
    logy = ycol in ("loan_amount", "property_value", "income")
    f = px.scatter(d, x=xcol, y=ycol, color="CLARANS cluster", color_discrete_sequence=QUAL,
                   opacity=0.6, log_x=logx, log_y=logy,
                   labels={xcol: NUMERIC_AXES.get(xcol, xcol), ycol: NUMERIC_AXES.get(ycol, ycol)})
    f.update_traces(marker=dict(size=6))
    f.update_layout(template=TEMPLATE, height=460,
                    title="CLARANS (medoid-based): same 4,000-application sample",
                    legend=dict(font=dict(size=9)), margin=dict(l=10, r=10, t=40, b=10))
    return f

def fig_method_comparison():
    """K-Means vs CLARANS vs Hierarchical, all evaluated on the same 4,000-row sample
    (DBSCAN runs on a different, 20k-row sample and produces a variable cluster count,
    so it gets its own dedicated view instead of forcing it into this comparison)."""
    if clarans is None:
        return blank("CLARANS comparison data not found.")
    km = clarans["kmeans_cluster"].value_counts().sort_index()
    cl = clarans["clarans_cluster"].value_counts().sort_index()
    idx = sorted(set(km.index) | set(cl.index))
    methods = [("K-Means", km, STEEL), ("CLARANS", cl, TEAL)]
    if hier_scatter is not None and "hier_cluster" in hier_scatter.columns:
        hi = hier_scatter["hier_cluster"].value_counts().sort_index()
        idx = sorted(set(idx) | set(hi.index))
        methods.append(("Hierarchical (Ward)", hi, QUAL[6]))
    d = pd.DataFrame({
        "cluster": [f"#{i}" for i in idx] * len(methods),
        "count": [c.get(i, 0) for _, c, _ in methods for i in idx],
        "method": [name for name, _, _ in methods for _ in idx],
    })
    f = px.bar(d, x="cluster", y="count", color="method", barmode="group",
               color_discrete_map={name: color for name, _, color in methods},
               labels={"cluster": "Cluster #", "count": "Applications (4k sample)"})
    f.update_layout(template=TEMPLATE, height=380, margin=dict(l=10, r=10, t=40, b=10),
                    title="Do K-Means, CLARANS and Hierarchical agree on cluster sizes? (same 4,000-row sample)")
    return f

def fig_hierarchical_scatter(xcol, ycol):
    if hier_scatter is None:
        return blank("Hierarchical comparison data not found.")
    d = hier_scatter.dropna(subset=[xcol, ycol]).copy()
    d = d[(d[xcol] > 0) & (d[ycol] > 0)] if xcol in ("loan_amount", "property_value", "income") else d
    d["Hierarchical cluster"] = "H" + d["hier_cluster"].astype(str)
    logx = xcol in ("loan_amount", "property_value", "income")
    logy = ycol in ("loan_amount", "property_value", "income")
    f = px.scatter(d, x=xcol, y=ycol, color="Hierarchical cluster", color_discrete_sequence=QUAL,
                   opacity=0.6, log_x=logx, log_y=logy,
                   labels={xcol: NUMERIC_AXES.get(xcol, xcol), ycol: NUMERIC_AXES.get(ycol, ycol)})
    f.update_traces(marker=dict(size=6))
    f.update_layout(template=TEMPLATE, height=460,
                    title="Hierarchical (Ward linkage): same 4,000-application sample",
                    legend=dict(font=dict(size=9)), margin=dict(l=10, r=10, t=40, b=10))
    return f

def fig_dbscan_sizes():
    """Single-hue bars for the 17 density clusters (no categorical hue battle across 17
    values - see the palette's own all-pairs cap); noise gets its own colour because it
    is a qualitatively different bucket, not the 18th cluster."""
    if dbscan_scatter is None:
        return blank("DBSCAN comparison data not found.")
    vc = dbscan_scatter["dbscan_cluster"].value_counts().sort_index()
    d = pd.DataFrame({"cluster": [("Noise" if i == -1 else f"#{i}") for i in vc.index], "count": vc.values,
                      "is_noise": [i == -1 for i in vc.index]})
    f = px.bar(d, x="cluster", y="count", color="is_noise",
               color_discrete_map={True: RED, False: STEEL},
               labels={"cluster": "DBSCAN cluster", "count": "Applications (20k sample)"})
    f.update_layout(template=TEMPLATE, height=340, showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
                    title="DBSCAN: cluster sizes (red = noise, unclustered)")
    return f

def fig_dbscan_scatter(xcol, ycol):
    """Coloured by Noise vs. Clustered (2 categories) rather than by individual cluster
    id: with 17 density clusters, a per-cluster hue scatter would blow the categorical
    palette's all-pairs cap - the noise/not-noise split is also the distinction that
    actually matters for the Phase 4 anomaly story this chart supports."""
    if dbscan_scatter is None:
        return blank("DBSCAN comparison data not found.")
    d = dbscan_scatter.dropna(subset=[xcol, ycol]).copy()
    d = d[(d[xcol] > 0) & (d[ycol] > 0)] if xcol in ("loan_amount", "property_value", "income") else d
    d["Status"] = np.where(d["dbscan_cluster"] == -1, "Noise (unclustered)", "Clustered")
    logx = xcol in ("loan_amount", "property_value", "income")
    logy = ycol in ("loan_amount", "property_value", "income")
    f = px.scatter(d, x=xcol, y=ycol, color="Status",
                   color_discrete_map={"Clustered": STEEL, "Noise (unclustered)": RED},
                   opacity=0.6, log_x=logx, log_y=logy,
                   labels={xcol: NUMERIC_AXES.get(xcol, xcol), ycol: NUMERIC_AXES.get(ycol, ycol)})
    f.update_traces(marker=dict(size=6))
    f.update_layout(template=TEMPLATE, height=460,
                    title="DBSCAN (density-based): same 20,000-application sample",
                    legend=dict(font=dict(size=9)), margin=dict(l=10, r=10, t=40, b=10))
    return f

def _filter_rules(df, outcome, min_lift):
    if df is None or not len(df):
        return None
    d = df[df["lift"] >= min_lift]
    if outcome != "All":
        d = d[d["then"] == outcome]
    return d.sort_values("lift", ascending=False)

def fig_rules_table_df(df, outcome, min_lift=1.0):
    d = _filter_rules(df, outcome, min_lift)
    if d is None:
        return None
    cols = {"if_readable": "If (applicant/loan profile)", "then": "Then",
            "recommendation": "Business recommendation"}
    keep = ["If (applicant/loan profile)", "Then", "Support", "Confidence", "Lift", "n",
            "Business recommendation"]
    if "kept" in d.columns:
        cols["kept"] = "Business-relevant?"
        keep.insert(1, "Business-relevant?")
    if not len(d):
        return pd.DataFrame(columns=keep)
    d = d.copy()
    d["Support"] = (d["support"] * 100).map(lambda v: f"{v:.1f}%")
    d["Confidence"] = (d["confidence"] * 100).map(lambda v: f"{v:.1f}%")
    d["Lift"] = d["lift"].map(lambda v: f"{v:.2f}×")
    d["n"] = d["n_matched"].map(lambda v: f"{v:,}")
    d = d.rename(columns=cols)
    return d[keep]

def fig_rules_scatter(df, outcome, min_lift=1.0):
    d = _filter_rules(df, outcome, min_lift)
    if d is None or not len(d):
        return blank("No rules match this filter.")
    if outcome == "All":
        f = px.scatter(d, x="support", y="confidence", size="lift", color="then",
                       color_discrete_map={"Denied": RED, "Originated": GREEN},
                       hover_name="if_readable", size_max=30,
                       labels={"support": "Support (how common)", "confidence": "Confidence (how reliable)",
                               "then": ""})
    else:
        f = px.scatter(d, x="support", y="confidence", size="lift", color="lift",
                       color_continuous_scale=["#e0a82e", RED] if outcome == "Denied" else ["#e0a82e", GREEN],
                       hover_name="if_readable", size_max=30,
                       labels={"support": "Support (how common)", "confidence": "Confidence (how reliable)"})
    f.update_layout(template=TEMPLATE, height=380, coloraxis_showscale=False,
                    title=f"{'All' if outcome == 'All' else outcome!r} rules: support vs confidence (bubble = lift)",
                    margin=dict(l=10, r=10, t=40, b=10))
    return f

def fig_rule_network(df, outcome, min_lift=1.0):
    d = _filter_rules(df, outcome, min_lift)
    if d is None or not len(d):
        return blank("No rules match this filter.")
    G = nx.DiGraph()
    for _, r in d.iterrows():
        sink = str(r["then"]).upper()
        for a in str(r["antecedent"]).split(", "):
            G.add_edge(a.strip(), sink, weight=float(r["lift"]))
    pos = nx.spring_layout(G, seed=42, k=0.9)
    ex, ey = [], []
    for u, v in G.edges():
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    edges = go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="#b9c4d0"), hoverinfo="none")
    nx_, ny_, txt, col, siz = [], [], [], [], []
    for n in G.nodes():
        nx_.append(pos[n][0]); ny_.append(pos[n][1]); txt.append(n)
        if n == "DENIED":
            col.append(RED); siz.append(40)
        elif n == "ORIGINATED":
            col.append(GREEN); siz.append(40)
        else:
            col.append(STEEL); siz.append(20 + 6 * G.degree(n))
    nodes = go.Scatter(x=nx_, y=ny_, mode="markers+text", text=txt, textposition="top center",
                       textfont=dict(size=8), marker=dict(size=siz, color=col, line=dict(width=1, color="white")),
                       hoverinfo="text")
    f = go.Figure([edges, nodes])
    title = "What drives DENIED vs. ORIGINATED" if outcome == "All" else f"What drives {outcome.upper()}"
    f.update_layout(template=TEMPLATE, height=380, showlegend=False,
                    title=title, xaxis_visible=False, yaxis_visible=False,
                    margin=dict(l=10, r=10, t=40, b=10))
    return f

def fig_anomaly_scatter():
    if scatter is None or "loan_amount" not in scatter:
        return blank()
    d = scatter.dropna(subset=["income", "loan_amount"]).copy()
    d = d[(d["income"] > 0) & (d["loan_amount"] > 0)]
    if "anomaly_votes" not in d:
        d["anomaly_votes"] = 0
    f = px.scatter(d, x="income", y="loan_amount", color="anomaly_votes",
                   color_continuous_scale="OrRd", log_x=True, log_y=True, opacity=0.6,
                   labels={"income": "Income ($k, log)", "loan_amount": "Loan Amount ($, log)",
                           "anomaly_votes": "Methods Flagging"})
    f.update_layout(template=TEMPLATE, height=420, margin=dict(l=10, r=10, t=40, b=10),
                    title="Outliers: extreme loan/income combinations rise to the top-right")
    return f

def fig_iso_hist():
    if scatter is None or "iso_score" not in scatter:
        return blank()
    f = px.histogram(scatter.dropna(subset=["iso_score"]), x="iso_score", nbins=60,
                     color_discrete_sequence=[STEEL],
                     labels={"iso_score": "Isolation-Forest Anomaly Score"})
    f.update_layout(template=TEMPLATE, height=300, title="Anomaly-score distribution (tail = outliers)",
                    margin=dict(l=10, r=10, t=40, b=10), bargap=0.02)
    return f

def fig_vote_breakdown():
    if scatter is None or "anomaly_votes" not in scatter:
        return blank()
    vc = scatter["anomaly_votes"].astype(int).value_counts().sort_index().reset_index()
    vc.columns = ["votes", "rows"]
    f = px.bar(vc, x="votes", y="rows", text="rows", color="votes", color_continuous_scale="OrRd",
               labels={"votes": "# methods agreeing", "rows": "Records"})
    f.update_layout(template=TEMPLATE, height=300, coloraxis_showscale=False,
                    title="Detector agreement (3+ = high-confidence anomaly)",
                    margin=dict(l=10, r=10, t=40, b=10))
    return f

# ---- Outlier taxonomy: global vs. contextual/local vs. collective ----
# IQR, Z-score and Isolation Forest all score a record against the WHOLE dataset's
# distribution, unconditionally - the textbook definition of a global outlier. LOF and
# DBSCAN-noise instead score a record against its own local neighbourhood's density -
# the textbook definition of a contextual/local outlier (the neighbourhood IS the
# context). "Normal" is drawn first (bottom layer, muted) so the three outlier colours
# read clearly on top rather than getting lost in 89,000 background points.
TAXONOMY_ORDER = ["Normal", "Global outlier", "Contextual/local outlier", "Both (global + contextual)"]
TAXONOMY_COLOR = {"Normal": "#c7d0da", "Global outlier": QUAL[0],
                  "Contextual/local outlier": QUAL[1], "Both (global + contextual)": QUAL[2]}

def fig_outlier_taxonomy(xcol, ycol):
    if outlier_tax is None or not len(outlier_tax):
        return blank("Outlier taxonomy data not found.")
    d = outlier_tax.dropna(subset=[xcol, ycol]).copy()
    d = d[(d[xcol] > 0) & (d[ycol] > 0)] if xcol in ("loan_amount", "property_value", "income") else d
    logx = xcol in ("loan_amount", "property_value", "income")
    logy = ycol in ("loan_amount", "property_value", "income")
    f = px.scatter(d, x=xcol, y=ycol, color="category", category_orders={"category": TAXONOMY_ORDER},
                   color_discrete_map=TAXONOMY_COLOR, opacity=0.65, log_x=logx, log_y=logy,
                   labels={xcol: NUMERIC_AXES.get(xcol, xcol), ycol: NUMERIC_AXES.get(ycol, ycol), "category": ""})
    f.update_traces(marker=dict(size=6))
    f.update_layout(template=TEMPLATE, height=460,
                    title="Where each outlier type sits in feature space",
                    legend=dict(font=dict(size=10)), margin=dict(l=10, r=10, t=40, b=10))
    return f

def _outlier_taxonomy_examples():
    if outlier_tax is None or not len(outlier_tax):
        return html.Div("Data not found.", style={"color": MUTE})
    cols = ["income", "loan_amount", "property_value", "combined_loan_to_value_ratio",
            "occupancy_type", "category"]
    rows = []
    for cat in ["Global outlier", "Contextual/local outlier", "Both (global + contextual)"]:
        sub = outlier_tax[outlier_tax["category"] == cat].sort_values("iso_score", ascending=False).head(3)
        rows.append(sub[cols])
    d = pd.concat(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=cols)
    d = d.rename(columns={"income": "Income ($k)", "loan_amount": "Loan amount", "property_value": "Property value",
                          "combined_loan_to_value_ratio": "CLTV (%)", "occupancy_type": "Occupancy", "category": "Type"})
    sdc = [{"if": {"filter_query": f'{{Type}} = "{cat}"'}, "backgroundColor": color + "22"}
           for cat, color in TAXONOMY_COLOR.items() if cat != "Normal"]
    return _table(d, style_data_conditional=sdc)

def fig_disparity():
    if disparity is None or not len(disparity):
        return blank()
    f = px.bar(disparity, x="derived_race", y="approval_rate_pct", color="tract_minority_cat",
               barmode="group", color_discrete_sequence=px.colors.sequential.Blues_r,
               labels={"derived_race": "", "approval_rate_pct": "Approval rate (%)",
                       "tract_minority_cat": "Tract Minority Level"})
    f.update_layout(template=TEMPLATE, height=460, title="Approval rate by race × neighbourhood minority level",
                    margin=dict(l=10, r=10, t=40, b=10))
    f.update_xaxes(tickangle=-35, automargin=True, tickfont=dict(size=10.5))
    return f

def fig_dti_geo_gap():
    if geo_gap is None or not len(geo_gap):
        return blank()
    d = geo_gap.melt(id_vars=["dti_group", "gap_pp"],
                     value_vars=["low_minority_approval_pct", "majority_minority_approval_pct"],
                     var_name="tract", value_name="approval_pct")
    d["tract"] = d["tract"].map({"low_minority_approval_pct": "Low-minority tract",
                                 "majority_minority_approval_pct": "Majority-minority tract"})
    f = px.bar(d, x="dti_group", y="approval_pct", color="tract", barmode="group",
               color_discrete_map={"Low-minority tract": STEEL, "Majority-minority tract": RED},
               text=d["approval_pct"].map(lambda v: f"{v:.0f}%"),
               labels={"dti_group": "Debt-to-income band", "approval_pct": "Approval rate (%)", "tract": ""})
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(template=TEMPLATE, height=400, margin=dict(l=10, r=10, t=40, b=10),
                    title="The gap survives controlling for DTI: it does not close within DTI bands")
    f.update_xaxes(automargin=True, tickfont=dict(size=11))
    f.update_yaxes(range=[0, 100])
    return f

def fig_denial_reasons():
    if denial is None or not len(denial):
        return blank()
    d = denial.sort_values("pct_of_denials", ascending=True)
    f = px.bar(d, x="pct_of_denials", y="reason", orientation="h", text="pct_of_denials",
               color="pct_of_denials", color_continuous_scale="OrRd",
               labels={"pct_of_denials": "% of Denials", "reason": ""})
    f.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    f.update_layout(template=TEMPLATE, height=360, coloraxis_showscale=False,
                    title="Lender-stated denial reasons (validates the DTI finding)",
                    margin=dict(l=10, r=60, t=30, b=10))
    f.update_xaxes(range=[0, 42])
    f.update_yaxes(automargin=True, tickfont=dict(size=11.5))
    return f

def fig_approval_by_dti():
    if scatter is None or "debt_to_income_ratio" not in scatter or "_approved" not in scatter:
        return blank()
    order = ["<20%", "20%-<30%", "30%-<36%", "36%-<50%", "50%-60%", ">60%"]
    d = scatter.dropna(subset=["debt_to_income_ratio", "_approved"])
    g = d.groupby("debt_to_income_ratio")["_approved"].mean().mul(100).reset_index()
    g = g[g["debt_to_income_ratio"].isin(order)]
    g["debt_to_income_ratio"] = pd.Categorical(g["debt_to_income_ratio"], categories=order, ordered=True)
    g = g.sort_values("debt_to_income_ratio")
    f = px.bar(g, x="debt_to_income_ratio", y="_approved", text=g["_approved"].map(lambda v: f"{v:.0f}%"),
               color="_approved", color_continuous_scale=["#c0392b", "#e0a82e", "#2e8b57"],
               labels={"debt_to_income_ratio": "Debt-to-income band", "_approved": "Approval rate (%)"})
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(template=TEMPLATE, height=360, coloraxis_showscale=False,
                    title="Approval collapses as debt-to-income rises", margin=dict(l=10, r=10, t=40, b=10))
    f.update_xaxes(automargin=True, tickfont=dict(size=11))
    f.update_yaxes(range=[0, 100])
    return f

def fig_gauge(value, title, good_high=True):
    lo_c, mid_c, hi_c = (RED, AMBER, GREEN) if good_high else (GREEN, AMBER, RED)
    f = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": "%"},
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": NAVY},
               "steps": [{"range": [0, 40], "color": lo_c if good_high else hi_c},
                        {"range": [40, 70], "color": mid_c},
                        {"range": [70, 100], "color": hi_c if good_high else lo_c}]}))
    f.update_layout(template=TEMPLATE, height=280, margin=dict(l=20, r=20, t=50, b=10))
    return f

STATE_METRICS = {
    "approval_rate": ("Approval rate (%)", ["#c0392b", "#e0a82e", "#2e8b57"]),
    "n": ("Applications (volume)", "Blues"),
    "median_income": ("Median income ($k)", "Greens"),
    "median_loan": ("Median loan amount ($)", "Purples"),
}

def fig_state_map(metric):
    if state_summary is None or not len(state_summary):
        return blank("State summary data not found. Run the notebook first.")
    d = state_summary.copy()
    label, scale = STATE_METRICS.get(metric, ("Value", "Blues"))
    f = px.choropleth(d, locations="state_code", locationmode="USA-states", color=metric,
                      scope="usa", color_continuous_scale=scale,
                      custom_data=["n", "approval_rate", "median_income", "median_loan", "top_denial_reason"],
                      labels={metric: label})
    # "%{hover_name}" is not a real Plotly token (that bug showed a bare "-" instead of
    # the state code); "%{location}" is the correct token for a choropleth's `locations` column.
    f.update_traces(hovertemplate="<b>%{location}</b><br>Applications: %{customdata[0]:,}<br>"
                                  "Approval rate: %{customdata[1]:.1f}%<br>"
                                  "Median income: $%{customdata[2]:.0f}k<br>"
                                  "Median loan: $%{customdata[3]:,.0f}<br>"
                                  "Top denial reason: %{customdata[4]}<extra></extra>")
    f.update_layout(template=TEMPLATE, height=460, margin=dict(l=10, r=10, t=30, b=10),
                    title=f"{label} by state (click a state to drill in below)",
                    geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor=BG))
    return f

def fig_state_dti(state):
    if state_dti is None or not len(state_dti):
        return blank()
    d = state_dti[state_dti["state_code"] == state].copy()
    if not len(d):
        return blank(f"No DTI-band detail for {state} (too few decisioned applications).")
    order = ["<20%", "20%-<30%", "30%-<36%", "36%-<43%", "43%-<50%", "50%-60%", ">60%"]
    d["dti_band"] = pd.Categorical(d["dti_band"], categories=[o for o in order if o in d["dti_band"].values], ordered=True)
    d = d.sort_values("dti_band")
    f = px.bar(d, x="dti_band", y="approval_rate", text=d["approval_rate"].map(lambda v: f"{v:.0f}%"),
              color="approval_rate", color_continuous_scale=["#c0392b", "#e0a82e", "#2e8b57"],
              labels={"dti_band": "Debt-to-income band", "approval_rate": "Approval rate (%)"})
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(template=TEMPLATE, height=340, coloraxis_showscale=False,
                    title=f"{state}: approval by DTI band", margin=dict(l=10, r=10, t=40, b=10))
    f.update_xaxes(tickangle=-30, automargin=True, tickfont=dict(size=10.5))
    f.update_yaxes(range=[0, 100])
    return f

def fig_state_segment(state):
    if state_segment is None or not len(state_segment):
        return blank()
    d = state_segment[state_segment["state_code"] == state].copy()
    if not len(d):
        return blank(f"No segment detail for {state}.")
    d["Segment"] = d["kmeans_cluster"].map(clabel)
    f = px.pie(d, values="n", names="Segment", hole=0.45, color_discrete_sequence=QUAL)
    f.update_traces(textposition="inside", textinfo="percent")
    f.update_layout(template=TEMPLATE, height=320, title=f"{state}: segment mix",
                    margin=dict(l=10, r=10, t=40, b=10), legend=dict(font=dict(size=8)))
    return f

def _geo_state_detail_children(state):
    """Drill-down content for one state: KPIs, DTI breakdown, segment mix.
    Called both for the tab's initial default state and from the map-click callback,
    so clicking the map is the only interaction needed (no separate location dropdown)."""
    if not state or state_summary is None:
        return html.Div()
    row = state_summary[state_summary["state_code"] == state]
    if not len(row):
        return html.Div()
    row = row.iloc[0]
    appr = float(row["approval_rate"])
    color = GREEN if appr >= 70 else (AMBER if appr >= 50 else RED)
    state_kpis = [
        ("Applications", f"{int(row['n']):,}", STEEL),
        ("Approval rate", f"{appr:.1f}%", color),
        ("Median income", f"${row['median_income']:.0f}k", TEAL),
        ("Median loan", f"${row['median_loan']:,.0f}", NAVY),
        ("Top denial reason", str(row["top_denial_reason"]), "#7d3c98"),
    ]
    return html.Div([
        html.Div(f"Showing: {state}", style={"fontSize": "12px", "color": MUTE, "marginBottom": "6px"}),
        html.Div([kpi_card(l, v, c) for l, v, c in state_kpis],
                 style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "margin": "4px 0 12px"}),
        html.Div([
            html.Div(panel(f"{state}: approval by DTI band", [graph(fig_state_dti(state))]),
                     style={"flex": "1", "minWidth": "340px"}),
            html.Div(panel(f"{state}: segment mix", [graph(fig_state_segment(state))]),
                     style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])

# ---- Duration (loan-term) "pseudo-time" ----
# HMDA 2022 is a single-year snapshot (activity_year is constant, dropped in Phase 1).
# There is no calendar date to filter on. loan_term (the mortgage's tenor: 15/20/30yr, ...)
# is the one genuinely ordered, temporal-feeling axis the data actually supports, so it
# stands in as one more slider on the What-If tab rather than a whole separate timeline tab.
def fig_term_range_detail(lo_idx, hi_idx):
    if term_summary is None or not len(term_summary):
        return html.Div("Data not found.", style={"color": MUTE})
    bands = TERM_ORDER[lo_idx:hi_idx + 1]
    d = term_summary[term_summary["term_band"].isin(bands)]
    if not len(d):
        return html.Div("No data for this range.", style={"color": MUTE})
    n_tot = int(d["n"].sum())
    appr = float(np.average(d["approval_rate"], weights=d["n"]))
    hidti = float(np.average(d["pct_high_dti"], weights=d["n"]))
    color = GREEN if appr >= 70 else (AMBER if appr >= 50 else RED)
    label = bands[0] if len(bands) == 1 else f"{bands[0]} to {bands[-1]}"
    return html.Div([
        html.Div([
            kpi_card("Duration range", label, NAVY),
            kpi_card("Applications", f"{n_tot:,}", STEEL),
            kpi_card("Approval rate", f"{appr:.1f}%", color),
            kpi_card("High-DTI share (>=50%)", f"{hidti:.1f}%", AMBER),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
    ])

# ============================================================ LAYOUT HELPERS
SOFT_SHADOW = "0 1px 2px rgba(16,42,74,0.05), 0 4px 20px rgba(16,42,74,0.05)"

def kpi_card(label, value, color):
    return html.Div([
        html.Div(value, style={"fontSize": "27px", "fontWeight": "800", "color": color,
                               "letterSpacing": "-0.6px", "lineHeight": "1.05"}),
        html.Div(label, style={"fontSize": "10.5px", "color": MUTE, "textTransform": "uppercase",
                               "letterSpacing": "0.6px", "marginTop": "7px", "fontWeight": "600"}),
    ], className="hmda-card", style={"background": CARD, "borderRadius": "14px", "padding": "16px 18px",
        "flex": "1", "boxShadow": SOFT_SHADOW, "minWidth": "132px", "textAlign": "center",
        "border": f"1px solid {BORDER}", "borderTop": f"3px solid {color}"})

def panel(title, children, sub=None):
    head = [html.H3(title, style={"margin": "0 0 2px", "color": NAVY, "fontSize": "15.5px",
                                  "fontWeight": "700", "letterSpacing": "-0.2px"})]
    if sub:
        head.append(html.Div(sub, style={"fontSize": "12px", "color": MUTE, "marginBottom": "10px",
                                         "lineHeight": "1.5"}))
    return html.Div(head + children, className="hmda-card", style={"background": CARD,
            "borderRadius": "16px", "padding": "18px 20px", "boxShadow": SOFT_SHADOW,
            "marginBottom": "16px", "border": f"1px solid {BORDER}"})

def graph(id_or_fig):
    # No zoom of any kind: scroll-zoom off (covers the geo map too), double-click
    # auto-zoom off, modebar hidden. Hover, legend clicks and map clicks still work.
    cfg = {"displayModeBar": False, "responsive": True, "scrollZoom": False, "doubleClick": False}
    if isinstance(id_or_fig, str):
        return dcc.Graph(id=id_or_fig, config=cfg)
    return dcc.Graph(figure=id_or_fig, config=cfg)

def finding_card(title, body, color):
    return html.Div([
        html.Div(title, style={"fontWeight": "700", "color": color, "fontSize": "13px", "marginBottom": "6px"}),
        html.Div(body, style={"fontSize": "12.5px", "color": INK, "lineHeight": "1.55"}),
    ], className="hmda-card", style={"background": CARD, "borderRadius": "16px", "padding": "16px 18px",
        "flex": "1", "minWidth": "260px", "boxShadow": SOFT_SHADOW,
        "border": f"1px solid {BORDER}", "borderLeft": f"4px solid {color}"})

# ============================================================ TABLES
def _table(df, cols=None, style_data_conditional=None):
    if df is None:
        return html.Div("Data not found. Run the notebook first.", style={"color": MUTE})
    d = df[cols] if cols else df
    numeric_cols = d.select_dtypes(include="number").columns
    d = d.copy()
    d[numeric_cols] = d[numeric_cols].round(3)
    sdc = [{"if": {"row_index": "odd"}, "backgroundColor": "#f4f7fb"}]
    if style_data_conditional:
        sdc += style_data_conditional
    return dash_table.DataTable(
        data=d.to_dict("records"),
        columns=[{"name": c, "id": c} for c in d.columns],
        page_size=15, style_as_list_view=True,
        style_table={"borderRadius": "12px", "overflow": "hidden", "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": NAVY, "color": "white", "fontWeight": "700", "fontSize": "11.5px",
                      "textTransform": "uppercase", "letterSpacing": "0.4px", "border": "none",
                      "padding": "10px 10px"},
        style_cell={"fontSize": "11.5px", "padding": "9px 10px", "fontFamily": FONT, "color": INK,
                    "maxWidth": "340px", "whiteSpace": "normal", "textAlign": "left",
                    "border": "none", "borderBottom": f"1px solid {GRID}"},
        style_data_conditional=sdc)

def _profiles_cards():
    if profiles is None:
        return html.Div("Data not found.", style={"color": MUTE})
    cards = []
    for _, r in profiles.sort_values("approval_rate", ascending=False).iterrows():
        color = GREEN if r["approval_rate"] >= 70 else (AMBER if r["approval_rate"] >= 50 else RED)
        cards.append(html.Div([
            html.Div([
                html.Span(clabel(r["kmeans_cluster"]), style={"fontWeight": "700", "fontSize": "13px", "color": NAVY}),
                html.Span(f"  {r['share_of_data']:.1f}% of applications", style={"fontSize": "11px", "color": MUTE}),
            ]),
            html.Div(f"{r['approval_rate']:.0f}% approved", style={"fontSize": "20px", "fontWeight": "800", "color": color, "margin": "4px 0"}),
            html.Div(segment_blurb(r), style={"fontSize": "12px", "color": INK, "lineHeight": "1.45"}),
            html.Div([
                html.Span("Recommended approach: ", style={"fontWeight": "700", "color": NAVY}),
                cluster_recommendation(r),
            ], style={"fontSize": "12px", "color": INK, "lineHeight": "1.45", "marginTop": "8px",
                     "padding": "8px 10px", "background": BG, "borderRadius": "8px"}),
            html.Div(f"Median income ${r['med_income']:.0f}k · median loan ${r['med_loan']:,.0f} · median CLTV {r['med_cltv']:.0f}%",
                     style={"fontSize": "10.5px", "color": MUTE, "marginTop": "6px"}),
        ], style={"background": CARD, "borderRadius": "12px", "padding": "14px 16px",
                  "boxShadow": "0 1px 3px rgba(0,0,0,0.08)", "marginBottom": "10px",
                  "borderLeft": f"4px solid {color}"}))
    return html.Div(cards)

def _anomaly_table():
    if triage is None:
        return _table(None)
    d = triage.reset_index().rename(columns={"index": "row_id"})
    cols = [c for c in ["row_id", "income", "loan_amount", "property_value",
                        "combined_loan_to_value_ratio", "occupancy_type", "action_taken",
                        "anomaly_votes", "verdict", "outlier_type", "evidence"] if c in d.columns]
    sdc = [{"if": {"filter_query": f'{{verdict}} = "{v}"'}, "backgroundColor": c + "22"}
           for v, c in VERDICT_COLOR.items()]
    return _table(d, cols, style_data_conditional=sdc)

# ============================================================ WHAT-IF PREDICTOR
# Ordinal fields get a slider over their full known category order (not just the values
# that happen to appear in a mined rule), so moving it explores the whole spectrum, the
# same way the loan-duration slider already works. A slider always has some value: it
# defaults to the modal (most common) band. Binary fields are a 3-way radio (not
# specified / A / B). Nominal fields with more than 2 options stay dropdowns.
DTI_ORDER = ["<20%", "20%-<30%", "30%-<36%", "36%-<43%", "43%-<50%", "50%-60%", ">60%"]
INCOME_BAND_ORDER = ["<30k", "30-50k", "50-75k", "75-100k", "100-150k", "150-200k", ">200k"]
LOAN_AMOUNT_BAND_ORDER = ["<100k", "100-200k", "200-300k", "300-500k", "500-750k", ">750k"]
PROPERTY_VALUE_BAND_ORDER = ["<100k", "100-200k", "200-350k", "350-500k", "500-750k", ">750k"]
CLTV_BAND_ORDER = ["<60%", "60-80%", "80-90%", "90-95%", "95-100%", ">100%"]
AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
UNITS_ORDER = ["1", "2", "3", "4", "5-24", "25-49", "50-99", "100-149", ">149"]
TRACT_INCOME_ORDER = ["Low_Income", "Moderate_Income", "Middle_Income", "Upper_Income"]

# field -> (label, kind, order_or_values, default)
WHATIF_FIELDS = [
    ("debt_to_income_ratio", "Debt-to-income", "slider", DTI_ORDER, "36%-<43%"),
    ("lien_status", "Lien status", "radio", ["First_Lien", "Subordinate_Lien"], None),
    ("loan_type", "Loan type", "dropdown", ["Conventional", "FHA", "VA", "RHS_FSA"], None),
    ("construction_method", "Construction method", "radio", ["Site_Built", "Manufactured"], None),
    ("loan_purpose", "Loan purpose", "dropdown",
     ["Home_Purchase", "Home_Improvement", "Refinance", "CashOut_Refinance", "Other", "NotApplicable"], None),
    ("income_band", "Income band", "slider", INCOME_BAND_ORDER, "75-100k"),
    ("preapproval", "Preapproval", "radio", ["Not_Requested", "Requested"], None),
    ("loan_amount_band", "Loan amount", "slider", LOAN_AMOUNT_BAND_ORDER, "100-200k"),
]

BASE_APPROVAL = float(np.average(profiles["approval_rate"], weights=profiles["n"])) if profiles is not None else 76.9

# Extra applicant/property/tract fields for context only: none of them appear in any
# mined rule antecedent, so they cannot join the match lookup above. Deliberately
# excludes derived_race / derived_ethnicity / derived_sex / tract_minority_cat: those
# already showed no predictive lift beyond DTI on the Rules tab, and a per-individual
# "pick your race, see your approval odds" widget risks reading as normalizing
# demographic scoring even when labeled "context only". That gap is handled properly,
# with the right caveats, on the Fairness tab instead.
CONTEXT_FIELDS = [
    ("applicant_age", "Applicant age", "slider", AGE_ORDER, "35-44"),
    ("occupancy_type", "Occupancy type", "dropdown", None, None),
    ("total_units", "Total units", "slider", UNITS_ORDER, "1"),
    ("conforming_loan_limit", "Conforming loan limit", "dropdown", None, None),
    ("property_value_band", "Property value", "slider", PROPERTY_VALUE_BAND_ORDER, "200-350k"),
    ("cltv_band", "CLTV", "slider", CLTV_BAND_ORDER, "60-80%"),
    ("tract_income_cat", "Tract income level", "slider", TRACT_INCOME_ORDER, "Middle_Income"),
]

def _dropdown_options(field):
    """Real observed values for a nominal dropdown whose category set isn't hardcoded."""
    if context_fields is None:
        return []
    return context_fields.loc[context_fields["field"] == field, "value"].tolist()

def render_control(field, label, kind, order_or_values, default, id_prefix):
    if kind == "slider":
        default_idx = order_or_values.index(default) if default in order_or_values else 0
        return html.Div([
            html.Label(label, style={"fontSize": "11px", "color": MUTE, "fontWeight": "600"}),
            dcc.Slider(id=f"{id_prefix}-{field}", min=0, max=len(order_or_values) - 1, step=1,
                      value=default_idx, marks={i: v for i, v in enumerate(order_or_values)}),
        ], style={"minWidth": "220px", "flex": "1 1 220px"})
    if kind == "radio":
        return html.Div([
            html.Label(label, style={"fontSize": "11px", "color": MUTE, "fontWeight": "600"}),
            dcc.RadioItems(id=f"{id_prefix}-{field}",
                          options=[{"label": "(not specified)", "value": ""}] +
                                  [{"label": v, "value": v} for v in order_or_values],
                          value="", inline=True, style={"fontSize": "11px", "marginTop": "4px"}),
        ], style={"minWidth": "200px", "flex": "1 1 200px"})
    opts = order_or_values if order_or_values else _dropdown_options(field)
    return html.Div([
        html.Label(label, style={"fontSize": "11px", "color": MUTE, "fontWeight": "600"}),
        dcc.Dropdown(id=f"{id_prefix}-{field}",
                     options=[{"label": "(not specified)", "value": ""}] +
                             [{"label": v, "value": v} for v in opts],
                     value="", clearable=False, style={"fontSize": "12px"}),
    ], style={"minWidth": "180px", "flex": "1 1 180px"})

def decode_control(kind, order_or_values, default, raw):
    if kind == "slider":
        try:
            return order_or_values[int(raw)]
        except (TypeError, ValueError, IndexError):
            return default
    return raw or ""

def context_lookup(field, value):
    if not value or context_fields is None:
        return None
    row = context_fields[(context_fields["field"] == field) & (context_fields["value"] == value)]
    if not len(row):
        return None
    r = row.iloc[0]
    return float(r["approval_rate"]), int(r["n"])

def whatif_match(selected):
    """selected: dict field -> value ('' = not specified). Returns (matched_row_or_None, approval_pct, note)."""
    chosen = {f"{f}={v}" for f, v in selected.items() if v}
    if rules is None or not len(rules) or not chosen:
        return None, BASE_APPROVAL, "No profile selected yet. Showing the portfolio base rate."
    best = None
    for _, r in rules.iterrows():
        items = set(str(r["antecedent"]).split(", "))
        if items.issubset(chosen):
            if best is None or r["lift"] > best["lift"]:
                best = r
    if best is None:
        return None, BASE_APPROVAL, "No strong historical pattern matches this exact combination. It falls into the broad majority, close to the portfolio base rate."
    appr = best["confidence"] * 100 if best["then"] == "Originated" else (1 - best["confidence"]) * 100
    note = f"Closest matching historical pattern: {best['if_readable']} → {best['then']} ({best['confidence']*100:.0f}% of the time, {best['n_matched']:,} similar applications)."
    return best, appr, note

# ============================================================ DASH APP
app = Dash(__name__, title="HMDA Knowledge Discovery Dashboard", suppress_callback_exceptions=True)
server = app.server

# Global stylesheet: web font, refined tabs, controls, scrollbar and card hover. Dash
# requires all {%...%} placeholders to be present in a custom index_string.
app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
      :root { --navy:#14294a; --steel:#2a78d6; --bg:#eef2f6; --mute:#64748b; }
      * { box-sizing: border-box; }
      html, body { margin: 0; padding: 0; }
      body {
        background: linear-gradient(180deg, #eef2f6 0%, #f4f7fa 320px, #f4f7fa 100%);
        font-family: Inter, 'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif;
        -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility; color: #1a202c;
      }
      ::-webkit-scrollbar { width: 11px; height: 11px; }
      ::-webkit-scrollbar-track { background: transparent; }
      ::-webkit-scrollbar-thumb { background: #c4d0dd; border-radius: 7px; border: 3px solid transparent; background-clip: padding-box; }
      ::-webkit-scrollbar-thumb:hover { background: #a6b6c8; border: 3px solid transparent; background-clip: padding-box; }

      /* No transform here on purpose: `transform` on an ancestor creates a new containing
         block for any absolutely-positioned descendant (react-select's dropdown menu,
         among others), which detaches/misplaces the open menu from its trigger the
         moment the pointer sits over a hovered card. Shadow-only hover avoids that. */
      .hmda-card { transition: box-shadow .2s ease; }
      .hmda-card:hover { box-shadow: 0 8px 30px rgba(16,42,74,0.11); }

      /* Tabs: clean underline bar, no default chrome */
      .hmda-tabs, .hmda-tabs > div { border: none !important; box-shadow: none !important; }
      .hmda-tabs .tab {
        border: none !important; background: transparent !important; color: var(--mute) !important;
        font-weight: 600 !important; font-size: 13px !important; padding: 11px 17px !important;
        border-radius: 10px 10px 0 0 !important;
        transition: color .15s ease, background .15s ease, box-shadow .15s ease;
      }
      .hmda-tabs .tab:hover { color: var(--navy) !important; background: rgba(42,120,214,0.07) !important; }
      .hmda-tabs .tab--selected {
        color: var(--navy) !important; background: transparent !important;
        box-shadow: inset 0 -3px 0 var(--steel) !important;
      }

      /* Dropdowns */
      .Select-control { border-radius: 10px !important; border-color: #d4dde7 !important; transition: border-color .15s ease, box-shadow .15s ease; }
      .Select-control:hover { border-color: var(--steel) !important; }
      .is-focused:not(.is-open) > .Select-control { border-color: var(--steel) !important; box-shadow: 0 0 0 3px rgba(42,120,214,0.14) !important; }
      .Select-menu-outer { border-radius: 10px !important; overflow: hidden; margin-top: 4px; border: 1px solid #e2e8f0 !important; box-shadow: 0 10px 30px rgba(16,42,74,0.14) !important; }
      .VirtualizedSelectFocusedOption, .Select-option.is-focused { background: rgba(42,120,214,0.08) !important; }

      /* Sliders */
      .rc-slider-track { background-color: var(--steel) !important; height: 5px !important; }
      .rc-slider-rail { background-color: #dbe3ec !important; height: 5px !important; }
      .rc-slider-handle { border-color: var(--steel) !important; background: #fff !important; box-shadow: 0 1px 4px rgba(16,42,74,0.25) !important; opacity: 1 !important; }
      .rc-slider-handle:hover, .rc-slider-handle-click-focused:focus { border-color: var(--steel) !important; box-shadow: 0 0 0 5px rgba(42,120,214,0.16) !important; }
      .rc-slider-dot { border-color: #dbe3ec !important; }
      .rc-slider-dot-active { border-color: var(--steel) !important; }
      .rc-slider-mark-text { color: var(--mute) !important; font-size: 10.5px !important; }

      /* Radios */
      input[type=radio] { accent-color: var(--steel); }
      .dash-cell, .dash-header { font-family: Inter, sans-serif !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

DISCOVERY = ("What we found that the raw data did not show: denial is governed overwhelmingly by "
             "debt-to-income. Above 60% DTI, about 92% of applications are denied (4x the base rate), and "
             "the lenders' own reasons confirm it. Adding race to that rule barely changes its lift, so the "
             "rule-level decision tracks ability-to-repay, not demography, yet a residual approval gap of "
             "up to 12 points persists between low- and majority-minority tracts even within the same DTI "
             "band. Both facts are true at once; see Executive Summary.")

app.layout = html.Div([
    html.Div([
        html.Div("KNOWLEDGE DISCOVERY DASHBOARD", style={
            "fontSize": "11px", "fontWeight": "700", "letterSpacing": "2.5px",
            "color": "rgba(255,255,255,0.62)", "marginBottom": "8px"}),
        html.H1("HMDA 2022 · Mortgage Approval & Denial", style={
            "margin": "0", "fontSize": "27px", "fontWeight": "800", "letterSpacing": "-0.7px",
            "color": "#ffffff"}),
        html.Div("Group 4 · Home Mortgage Disclosure Act · 100,000-record sample (CFPB open data)",
                 style={"fontSize": "13px", "color": "rgba(255,255,255,0.78)", "marginTop": "6px"}),
    ], style={
        "background": "linear-gradient(120deg, #0e2340 0%, #16345c 46%, #245ea3 100%)",
        "color": "white", "padding": "30px 30px 32px",
        "boxShadow": "0 6px 24px rgba(16,42,74,0.20)",
        "borderBottom": "3px solid #2a78d6"}),

    html.Div([
        html.Div("KEY DISCOVERY", style={"fontWeight": "800", "color": STEEL, "fontSize": "11px",
                                         "letterSpacing": "1.4px", "marginBottom": "5px"}),
        html.Span(DISCOVERY, style={"color": INK, "fontSize": "13px", "lineHeight": "1.6"}),
    ], style={"background": "linear-gradient(135deg, #eef4fc 0%, #f8fbff 100%)",
              "borderLeft": f"4px solid {STEEL}", "borderRadius": "12px",
              "padding": "15px 20px", "margin": "20px 24px 6px",
              "boxShadow": "0 2px 14px rgba(16,42,74,0.05)"}),

    html.Div([kpi_card(l, v, c) for l, v, c in kpis()],
             style={"display": "flex", "gap": "13px", "padding": "12px 24px 4px", "flexWrap": "wrap"}),

    html.Div(dcc.Tabs(id="tabs", value="summary", className="hmda-tabs", children=[
        dcc.Tab(label="Executive Summary", value="summary"),
        dcc.Tab(label="Geography", value="geography"),
        dcc.Tab(label="Segments (Clustering)", value="segments"),
        dcc.Tab(label="Rules (Approve vs Deny)", value="rules"),
        dcc.Tab(label="Anomalies", value="anomalies"),
        dcc.Tab(label="What-If", value="whatif"),
        dcc.Tab(label="Fairness", value="fairness"),
    ]), style={"padding": "18px 24px 0"}),

    html.Div(id="tab-content", style={"padding": "6px 24px 34px"}),
], style={"fontFamily": FONT, "minHeight": "100vh", "paddingBottom": "20px"})

# ============================================================ tab routing
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render(tab):
    if tab == "summary":
        return html.Div([
            html.Div([
                finding_card("1. DTI dominates, but it isn't the whole story",
                             "Every top denial rule is a variant of debt-to-income > 60% (lift 3.96, "
                             "91.5% denied). Adding race or ethnicity to that rule changes its confidence "
                             "by less than 2 points, but a 12.1-point approval gap between low- and "
                             "majority-minority tracts survives even within the same low-DTI band (Fairness tab).",
                             STEEL),
                finding_card("2. The anomaly sweep found nothing new",
                             "Five independent detectors flag 739 high-confidence outliers. Hand-reviewing "
                             "the most extreme 15 finds zero data errors beyond what upstream cleaning "
                             "already caught: every one is a legitimate jumbo, multi-unit, or investment "
                             "record (Anomalies tab).", TEAL),
                finding_card("3. Manufactured housing is penalised on its own",
                             "construction_method=Manufactured alone denies 56.9% of applications (vs. "
                             "23.1% baseline). A property attribute, not income or leverage, is enough to "
                             "flip the majority outcome (Rules tab).", AMBER),
            ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "4px"}),
            html.Div([
                html.Div(panel("Approval rate by segment", [graph(fig_approval_by_cluster())]),
                         style={"flex": "1", "minWidth": "380px"}),
                html.Div(panel("Why applications are denied", [graph(fig_denial_reasons())]),
                         style={"flex": "1", "minWidth": "380px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
            panel("The single clearest pattern", [graph(fig_approval_by_dti())],
                  sub="Approval rate plotted against debt-to-income band, the dominant driver of denial."),
        ])

    if tab == "geography":
        if state_summary is None or not len(state_summary):
            return panel("Geography", [html.Div(
                "State summary data not found. Run the notebook's Phase 5 export (or the "
                "dash_state_summary.csv builder) first.", style={"color": MUTE})])
        states = state_summary.sort_values("n", ascending=False)["state_code"].tolist()
        default_state = states[0] if states else None
        return html.Div([
            panel("Filters", [
                html.Div([html.Label("Map metric", style={"fontSize": "12px", "color": MUTE}),
                          dcc.Dropdown(id="geo-metric",
                                       options=[{"label": v[0], "value": k} for k, v in STATE_METRICS.items()],
                                       value="approval_rate", clearable=False, style={"fontSize": "12px"})],
                         style={"width": "260px"}),
            ]),
            panel("Approval, volume and pricing across the country", [graph("geo-map")],
                  sub="Hover a state for a quick summary. Click it to load the fuller breakdown below."),
            html.Div(_geo_state_detail_children(default_state), id="geo-state-detail"),
        ])

    if tab == "segments":
        return html.Div([
            panel("Clustering method", [
                dcc.RadioItems(id="cluster-method", value="kmeans",
                               options=[
                                   {"label": " K-Means (primary, full 99,995 applications)", "value": "kmeans"},
                                   {"label": " DBSCAN (density-based, 20,000-row sample)", "value": "dbscan"},
                                   {"label": " Hierarchical (Ward linkage, 4,000-row sample)", "value": "hierarchical"},
                                   {"label": " CLARANS (k-medoids, 4,000-row sample)", "value": "clarans"},
                               ],
                               inline=True, style={"fontSize": "12px"}),
            ], sub="All four methods required by the brief: K-Means is the primary segmentation "
                   "(fitted on the full data); DBSCAN, Hierarchical and CLARANS run on samples "
                   "for validation and cross-checking, as documented in the notebook's Phase 2."),
            html.Div(id="segments-method-panel"),
        ])

    if tab == "rules":
        n_all = len(rules_all) if rules_all is not None else 0
        n_biz = len(rules) if rules is not None else 0
        return html.Div([
            panel("Why only 11 are business-relevant?", [html.P(
                f"{n_all} candidate rules cleared the support/confidence/lift thresholds, but {n_all - n_biz} "
                "were trivial restatements, e.g. adding \"White\" or \"First_Lien\" to the DTI>60% rule changed "
                "its confidence by less than 2 points. Those are pruned by an improvement filter: a rule only "
                "counts if it beats its own best sub-rule by at least 2 percentage points. The business section "
                "below shows what survives that filter; the full candidate list, improvement filter and all, "
                "is further down for anyone who wants to see everything the miner actually found.",
                style={"fontSize": "12px", "color": INK, "margin": "0"})]),
            panel("Filters", [
                dcc.RadioItems(id="net-outcome", value="All",
                               options=[{"label": " All (Denied + Originated)", "value": "All"},
                                        {"label": " Why applications are DENIED", "value": "Denied"},
                                        {"label": " Why applications are ORIGINATED", "value": "Originated"}],
                               inline=True, style={"fontSize": "12px", "marginBottom": "12px"}),
                html.Label("Minimum lift", style={"fontSize": "12px", "color": MUTE, "fontWeight": "600"}),
                dcc.Slider(id="rules-min-lift", min=1.0, max=4.5, step=0.1, value=1.0,
                          marks={i: str(i) for i in [1, 2, 3, 4]},
                          tooltip={"placement": "bottom", "always_visible": False}),
            ], sub="Applies to both sections below - both respond to the same view and lift floor."),
            panel("Business-relevant rules (survived the improvement filter)",
                  [html.Div(id="rules-table-container")]),
            html.Div([
                html.Div(panel("Rule landscape", [graph("rules-scatter")]), style={"flex": "1", "minWidth": "360px"}),
                html.Div(panel("Rule network", [graph("rule-network")]), style={"flex": "1", "minWidth": "360px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
            panel("All candidate rules (raw, before pruning)", [html.Div(id="rules-all-table-container")],
                  sub="Every rule that cleared the mining thresholds (lift > 1.2, confidence >= 55%), including "
                      "the ones the improvement filter later dropped. \"Business-relevant?\" marks which ones "
                      "survived into the curated set above."),
        ])

    if tab == "anomalies":
        n_hc = int((scatter["anomaly_votes"] >= 3).sum()) if scatter is not None and "anomaly_votes" in scatter else 0
        tax_opts = [{"label": v, "value": k} for k, v in NUMERIC_AXES.items()]
        n_global = n_local = n_both = n_normal = 0
        if outlier_tax_summary is not None:
            vc_full = dict(zip(outlier_tax_summary["category"], outlier_tax_summary["n"]))
            n_global = int(vc_full.get("Global outlier", 0))
            n_local = int(vc_full.get("Contextual/local outlier", 0))
            n_both = int(vc_full.get("Both (global + contextual)", 0))
            n_normal = int(vc_full.get("Normal", 0))
        n_total = max(n_global + n_local + n_both + n_normal, 1)
        coll_pct = float(collective_pattern["pct"].iloc[0]) if collective_pattern is not None and len(collective_pattern) else None
        return html.Div([
            panel("How to read this", [html.P(
                f"Four detectors (IQR, Z-score, Isolation Forest, LOF) plus DBSCAN-noise cross-reference vote "
                f"on every application; {n_hc:,} are high-confidence anomalies (3+ methods agree). The 15 most "
                f"extreme were hand-triaged into a verdict with evidence, not just flagged as \"weird\".",
                style={"fontSize": "12px", "color": INK, "margin": "0"})]),

            panel("Outlier taxonomy: global, contextual and collective", [
                html.P("Data-mining theory splits outliers into three kinds. This pipeline's five detectors "
                       "split cleanly across two of them, which is itself a finding worth presenting: the two "
                       "detection philosophies catch substantially different records.",
                       style={"fontSize": "12px", "color": INK, "marginBottom": "14px"}),
                html.Div([
                    html.Div([
                        html.Div("GLOBAL OUTLIERS", style={"fontWeight": "800", "color": TAXONOMY_COLOR["Global outlier"], "fontSize": "12px", "letterSpacing": "0.5px"}),
                        html.Div(f"{n_global:,}", style={"fontSize": "24px", "fontWeight": "800", "color": NAVY, "margin": "4px 0"}),
                        html.Div(f"{n_global/n_total*100:.1f}% of applications", style={"fontSize": "11px", "color": MUTE, "marginBottom": "8px"}),
                        html.Div("Detected by IQR, Z-score and Isolation Forest: records that deviate from "
                                 "the WHOLE dataset's distribution, unconditionally (e.g. a $130M property "
                                 "value - extreme no matter what else is true about the record).",
                                 style={"fontSize": "11.5px", "color": INK, "lineHeight": "1.5"}),
                    ], className="hmda-card", style={"flex": "1", "minWidth": "230px", "background": CARD,
                        "borderRadius": "14px", "padding": "14px 16px", "boxShadow": SOFT_SHADOW,
                        "border": f"1px solid {BORDER}", "borderTop": f"3px solid {TAXONOMY_COLOR['Global outlier']}"}),
                    html.Div([
                        html.Div("CONTEXTUAL / LOCAL OUTLIERS", style={"fontWeight": "800", "color": TAXONOMY_COLOR["Contextual/local outlier"], "fontSize": "12px", "letterSpacing": "0.5px"}),
                        html.Div(f"{n_local:,}", style={"fontSize": "24px", "fontWeight": "800", "color": NAVY, "margin": "4px 0"}),
                        html.Div(f"{n_local/n_total*100:.1f}% of applications", style={"fontSize": "11px", "color": MUTE, "marginBottom": "8px"}),
                        html.Div("Detected by LOF and DBSCAN-noise: records that look ordinary by every "
                                 "single-feature threshold, but sit in a sparse neighbourhood relative to "
                                 "similar applications - unusual only given that local context.",
                                 style={"fontSize": "11.5px", "color": INK, "lineHeight": "1.5"}),
                    ], className="hmda-card", style={"flex": "1", "minWidth": "230px", "background": CARD,
                        "borderRadius": "14px", "padding": "14px 16px", "boxShadow": SOFT_SHADOW,
                        "border": f"1px solid {BORDER}", "borderTop": f"3px solid {TAXONOMY_COLOR['Contextual/local outlier']}"}),
                    html.Div([
                        html.Div("BOTH", style={"fontWeight": "800", "color": TAXONOMY_COLOR["Both (global + contextual)"], "fontSize": "12px", "letterSpacing": "0.5px"}),
                        html.Div(f"{n_both:,}", style={"fontSize": "24px", "fontWeight": "800", "color": NAVY, "margin": "4px 0"}),
                        html.Div(f"{n_both/n_total*100:.1f}% of applications", style={"fontSize": "11px", "color": MUTE, "marginBottom": "8px"}),
                        html.Div("Flagged by at least one global AND one contextual/local method - the "
                                 "highest-confidence anomalies, and exactly the pool the top-15 hand-triage "
                                 "(below) was drawn from.",
                                 style={"fontSize": "11.5px", "color": INK, "lineHeight": "1.5"}),
                    ], className="hmda-card", style={"flex": "1", "minWidth": "230px", "background": CARD,
                        "borderRadius": "14px", "padding": "14px 16px", "boxShadow": SOFT_SHADOW,
                        "border": f"1px solid {BORDER}", "borderTop": f"3px solid {TAXONOMY_COLOR['Both (global + contextual)']}"}),
                    html.Div([
                        html.Div("COLLECTIVE OUTLIERS", style={"fontWeight": "800", "color": AMBER, "fontSize": "12px", "letterSpacing": "0.5px"}),
                        html.Div(f"{coll_pct:.0f}%" if coll_pct is not None else "N/A",
                                 style={"fontSize": "24px", "fontWeight": "800", "color": NAVY, "margin": "4px 0"}),
                        html.Div("of loans >= $1M share one trait", style={"fontSize": "11px", "color": MUTE, "marginBottom": "8px"}),
                        html.Div("None of the 5 detectors targets this class (they all score individual "
                                 "records). Every loan >= $1M reports an amount ending in \"...5,000\" - HMDA's "
                                 "mandatory rounding-to-nearest-$10k-midpoint rule. No single loan is odd; "
                                 "the group-wide pattern is a data-generating-process signature, not a risk signal.",
                                 style={"fontSize": "11.5px", "color": INK, "lineHeight": "1.5"}),
                    ], className="hmda-card", style={"flex": "1", "minWidth": "230px", "background": CARD,
                        "borderRadius": "14px", "padding": "14px 16px", "boxShadow": SOFT_SHADOW,
                        "border": f"1px solid {BORDER}", "borderTop": f"3px solid {AMBER}"}),
                ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "6px"}),
            ]),

            panel("Explore the taxonomy", [
                _axis_picker("tax-xcol", "tax-ycol", tax_opts),
                graph("tax-scatter"),
            ], sub="Grey points are the ordinary background population (a 4,000-row reference sample). "
                   "Coloured points are every flagged record, positioned by two attributes of your choosing - "
                   "notice global outliers cluster at the magnitude extremes while contextual/local outliers "
                   "sit inside the ordinary-looking range."),

            panel("Named examples", [_outlier_taxonomy_examples()],
                  sub="The 3 most extreme records in each non-normal category, by Isolation Forest score."),

            panel("Where the outliers are (income vs. loan amount)", [graph(fig_anomaly_scatter())],
                  sub="Top-right = large loans relative to stated income."),
            html.Div([
                html.Div(panel("Anomaly scores", [graph(fig_iso_hist())]), style={"flex": "1", "minWidth": "340px"}),
                html.Div(panel("Method agreement", [graph(fig_vote_breakdown())]), style={"flex": "1", "minWidth": "340px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
            panel("Most extreme records: triaged with evidence", [_anomaly_table()],
                  sub="Verdict types: RARE BUT VALID (legitimate extreme record) · RISK SIGNAL (unusual but "
                      "possible) · DATA ERROR (impossible value) · MANUAL REVIEW (hand-checked)."),
        ])

    if tab == "whatif":
        controls = [render_control(field, label, kind, order, default, "wi")
                   for field, label, kind, order, default in WHATIF_FIELDS]
        context_controls = [render_control(field, label, kind, order, default, "ctx")
                            for field, label, kind, order, default in CONTEXT_FIELDS]
        term_default = TERM_ORDER.index("30yr") if "30yr" in TERM_ORDER else 0
        return html.Div([
            panel("Build an applicant profile", [
                html.Div(controls, style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
                html.Div("Sliders cover the field's whole range and always have a value (defaulting to the "
                         "most common band); dropdowns and toggles default to \"not specified\". This looks "
                         "up the closest matching historical pattern from the Rules tab. It is a lookup "
                         "against real historical patterns, not a trained predictive model, and it is not a "
                         "guarantee for any individual applicant.",
                         style={"fontSize": "11px", "color": MUTE, "marginTop": "10px"}),
            ]),
            panel("Loan duration", [
                dcc.Slider(id="wi-term-band", min=0, max=len(TERM_ORDER) - 1, step=1, value=term_default,
                          marks={i: b for i, b in enumerate(TERM_ORDER)}),
                html.Div(id="wi-term-detail", style={"marginTop": "10px"}),
            ], sub="HMDA 2022 has no calendar date to filter on (activity_year is constant), so loan "
                   "duration stands in as the closest thing to a time axis. Shown for context only: no "
                   "mined rule uses it, so it does not change the match above. 25-year loans are the one "
                   "exception worth knowing: they dip to 56.2% approval, the highest high-DTI share of any band."),
            panel("More context", [
                html.Div(context_controls, style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
                html.Div(id="wi-context-result", style={"marginTop": "10px"}),
            ], sub="Applicant/property/tract attributes that were mined but never survived the improvement "
                   "filter (Rules tab): they don't add information beyond the profile above, so, like loan "
                   "duration, they're shown as historical context only and never change the match result."),
            html.Div(id="whatif-result"),
        ])

    if tab == "fairness":
        return html.Div([
            panel("Approval by race × neighbourhood", [graph(fig_disparity())],
                  sub="Compare approval rates across demographic groups and tract minority levels."),
            panel("Does the gap survive controlling for DTI?", [graph(fig_dti_geo_gap())],
                  sub="Same comparison, split by the strongest predictor of denial in the dataset."),
            panel("How to read this", [html.P(
                "The Rules tab shows denials track debt-to-income, not race: adding race to the core DTI "
                "rule barely moves its lift. This is the cross-check: equal financial profiles should show "
                "equal approval rates. The gap does not close within DTI bands (it is largest, 12.1 points, "
                "in the lowest-risk, low-DTI group), a residual, unexplained pattern that merits further "
                "fair-lending investigation, association only, not a causal finding.",
                style={"fontSize": "13px", "color": INK})]),
        ])
    return html.Div()

# ============================================================ segments sub-callback
def _axis_picker(xcol_id, ycol_id, opts, default_x="income", default_y="loan_amount"):
    return html.Div([
        html.Div([html.Label("X axis", style={"fontSize": "12px", "color": MUTE}),
                  dcc.Dropdown(id=xcol_id, options=opts, value=default_x, clearable=False)],
                 style={"width": "240px"}),
        html.Div([html.Label("Y axis", style={"fontSize": "12px", "color": MUTE}),
                  dcc.Dropdown(id=ycol_id, options=opts, value=default_y, clearable=False)],
                 style={"width": "240px"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "8px"})

@app.callback(Output("segments-method-panel", "children"), Input("cluster-method", "value"))
def _cb_segments(method):
    opts = [{"label": v, "value": k} for k, v in NUMERIC_AXES.items()]

    if method == "dbscan":
        n_noise = int((dbscan_scatter["dbscan_cluster"] == -1).sum()) if dbscan_scatter is not None else 0
        n_total = len(dbscan_scatter) if dbscan_scatter is not None else 0
        pct = (n_noise / n_total * 100) if n_total else 0
        return html.Div([
            panel("DBSCAN cluster sizes", [graph(fig_dbscan_sizes())],
                  sub=f"eps chosen from the k-distance knee (min_samples = 2 x feature count): 17 density "
                      f"clusters plus noise. {n_noise:,} of {n_total:,} applications ({pct:.1f}%) don't "
                      "belong to any dense region - these noise points are exactly the input Phase 4's "
                      "anomaly ensemble cross-references (see Anomalies tab)."),
            panel("Explore the DBSCAN result", [
                _axis_picker("dbscan-xcol", "dbscan-ycol", opts),
                graph("dbscan-scatter"),
            ], sub="Coloured by noise vs. clustered only (not by individual cluster id) - with 17 clusters, "
                   "a per-cluster hue scatter would blow past the validated categorical colour limit."),
        ])

    if method == "hierarchical":
        return html.Div([
            panel("K-Means vs. CLARANS vs. Hierarchical agreement", [graph(fig_method_comparison())],
                  sub="Ward linkage cut at K=7 (the silhouette-selected K from section 2.1). Adjusted Rand "
                      "Index vs. K-Means = 0.907 on the same 4,000-row sample: strong agreement, confirming "
                      "the segment structure isn't a K-Means artifact."),
            panel("Explore the Hierarchical segments", [
                _axis_picker("hier-xcol", "hier-ycol", opts),
                graph("hier-scatter"),
            ]),
        ])

    if method == "clarans":
        return html.Div([
            panel("K-Means vs. CLARANS agreement", [graph(fig_method_comparison())],
                  sub="CLARANS (k-medoids) picks real applications as cluster centres instead of synthetic "
                      "averages, robust to the extreme outliers on the Anomalies tab. Adjusted Rand Index "
                      "vs. K-Means = 0.711, vs. Ward hierarchical = 0.701: moderate-to-strong agreement, as "
                      "expected from two different optimisation objectives on the same data."),
            panel("Explore the CLARANS segments", [
                _axis_picker("clarans-xcol", "clarans-ycol", opts),
                graph("clarans-scatter"),
            ]),
        ])
    return html.Div([
        html.Div([
            html.Div(panel("Segment sizes", [graph(fig_cluster_sizes())]), style={"flex": "1", "minWidth": "340px"}),
            html.Div(panel("Approval rate by segment", [graph(fig_approval_by_cluster())]), style={"flex": "1", "minWidth": "340px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        panel("Explore the segments", [
            _axis_picker("xcol", "ycol", opts),
            graph("cluster-scatter"),
        ], sub="Each point is a loan application, coloured by segment. Pick any two attributes."),
        panel("Segment characteristics: what each cluster means for the business", [_profiles_cards()]),
    ])

@app.callback(Output("cluster-scatter", "figure"), Input("xcol", "value"), Input("ycol", "value"))
def _cb_scatter(x, y):
    return fig_cluster_scatter(x, y)

@app.callback(Output("clarans-scatter", "figure"), Input("clarans-xcol", "value"), Input("clarans-ycol", "value"))
def _cb_clarans_scatter(x, y):
    return fig_clarans_scatter(x, y)

@app.callback(Output("dbscan-scatter", "figure"), Input("dbscan-xcol", "value"), Input("dbscan-ycol", "value"))
def _cb_dbscan_scatter(x, y):
    return fig_dbscan_scatter(x, y)

@app.callback(Output("hier-scatter", "figure"), Input("hier-xcol", "value"), Input("hier-ycol", "value"))
def _cb_hier_scatter(x, y):
    return fig_hierarchical_scatter(x, y)

@app.callback(Output("tax-scatter", "figure"), Input("tax-xcol", "value"), Input("tax-ycol", "value"))
def _cb_tax_scatter(x, y):
    return fig_outlier_taxonomy(x, y)

# ============================================================ geography callbacks
@app.callback(Output("geo-map", "figure"), Input("geo-metric", "value"))
def _cb_geo_map(metric):
    return fig_state_map(metric)

@app.callback(Output("geo-state-detail", "children"), Input("geo-map", "clickData"),
             prevent_initial_call=True)
def _cb_geo_map_click(click_data):
    if click_data and click_data.get("points"):
        loc = click_data["points"][0].get("location")
        if loc:
            return _geo_state_detail_children(loc)
    return no_update

# ============================================================ rules sub-callbacks
@app.callback(Output("rules-table-container", "children"),
             Input("net-outcome", "value"), Input("rules-min-lift", "value"))
def _cb_rules_table(outcome, min_lift):
    return _table(fig_rules_table_df(rules, outcome, min_lift))

@app.callback(Output("rules-scatter", "figure"),
             Input("net-outcome", "value"), Input("rules-min-lift", "value"))
def _cb_rules_scatter(outcome, min_lift):
    return fig_rules_scatter(rules, outcome, min_lift)

@app.callback(Output("rule-network", "figure"),
             Input("net-outcome", "value"), Input("rules-min-lift", "value"))
def _cb_net(outcome, min_lift):
    return fig_rule_network(rules, outcome, min_lift)

@app.callback(Output("rules-all-table-container", "children"),
             Input("net-outcome", "value"), Input("rules-min-lift", "value"))
def _cb_rules_all_table(outcome, min_lift):
    return _table(fig_rules_table_df(rules_all, outcome, min_lift))

# ============================================================ what-if callback
@app.callback(Output("wi-term-detail", "children"), Input("wi-term-band", "value"))
def _cb_wi_term(idx):
    idx = int(idx) if idx is not None else TERM_ORDER.index("30yr")
    return fig_term_range_detail(idx, idx)

@app.callback(
    Output("wi-context-result", "children"),
    [Input(f"ctx-{field}", "value") for field, _, _, _, _ in CONTEXT_FIELDS],
)
def _cb_wi_context(*values):
    chips = []
    for (field, label, kind, order, default), raw in zip(CONTEXT_FIELDS, values):
        value = decode_control(kind, order, default, raw)
        hit = context_lookup(field, value)
        if hit is None:
            continue
        appr, n = hit
        color = GREEN if appr >= 70 else (AMBER if appr >= 50 else RED)
        chips.append(html.Div([
            html.Span(f"{label}: {value}", style={"fontSize": "11px", "color": MUTE}),
            html.Div(f"{appr:.1f}% approved", style={"fontSize": "16px", "fontWeight": "800", "color": color}),
            html.Div(f"n={n:,}", style={"fontSize": "10px", "color": MUTE}),
        ], style={"background": BG, "borderRadius": "8px", "padding": "8px 12px", "minWidth": "140px"}))
    if not chips:
        return html.Div("Pick any attribute above to see its historical approval rate.",
                        style={"fontSize": "11px", "color": MUTE})
    return html.Div(chips, style={"display": "flex", "gap": "10px", "flexWrap": "wrap"})

@app.callback(
    Output("whatif-result", "children"),
    [Input(f"wi-{field}", "value") for field, _, _, _, _ in WHATIF_FIELDS],
)
def _cb_whatif(*values):
    selected = {}
    for (field, label, kind, order, default), raw in zip(WHATIF_FIELDS, values):
        selected[field] = decode_control(kind, order, default, raw)
    best, appr, note = whatif_match(selected)
    outcome_label = "Estimated approval likelihood"
    return panel("Result", [
        html.Div([
            html.Div(graph(fig_gauge(appr, outcome_label)), style={"flex": "1", "minWidth": "280px"}),
            html.Div([
                html.Div(note, style={"fontSize": "13px", "color": INK, "lineHeight": "1.6"}),
            ], style={"flex": "2", "minWidth": "280px", "display": "flex", "alignItems": "center"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "alignItems": "center"}),
    ])

# ============================================================ run
if __name__ == "__main__":
    # Local:  python dashboard_app.py   ->  http://127.0.0.1:8050
    # Colab:  app.run(jupyter_mode="external")   (clickable link)  or  jupyter_mode="inline"
    app.run(debug=False, port=8050)
