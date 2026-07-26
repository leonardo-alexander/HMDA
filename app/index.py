import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import networkx as nx
from dash import Dash, dcc, html, Input, Output, dash_table, no_update

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
QUAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
SEQ_BLUE = [[0.0, "#e3eefb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]
VERDICT_COLOR = {
    "DATA ERROR": RED,
    "RARE BUT VALID": GREEN,
    "RISK SIGNAL": AMBER,
    "MANUAL REVIEW": STEEL,
}

# One registered Plotly template so every figure inherits the same font, gridlines,
# transparent surface (blends into its white card), colourway and hover style.
pio.templates["hmda"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=12, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=QUAL,
        dragmode=False,  # no drag-to-zoom / box-select on any figure (hover still works)
        title=dict(
            font=dict(family=FONT, size=14.5, color=NAVY),
            x=0.01,
            xanchor="left",
            pad=dict(b=6),
        ),
        # automargin: Plotly reserves separate space for tick labels vs. the axis title and
        # grows the figure's margin to fit them, instead of letting long category labels
        # (segment names, race categories, DTI bands) clip against the card edge or collide
        # with the title. This is the general fix; specific charts still tune angle/margin.
        xaxis=dict(
            gridcolor=GRID,
            linecolor=AXIS,
            zerolinecolor=GRID,
            tickcolor=AXIS,
            fixedrange=True,
            automargin=True,
            tickfont=dict(color=MUTE, size=11),
            title=dict(font=dict(color=MUTE, size=12), standoff=10),
        ),
        yaxis=dict(
            gridcolor=GRID,
            linecolor=AXIS,
            zerolinecolor=GRID,
            tickcolor=AXIS,
            fixedrange=True,
            automargin=True,
            tickfont=dict(color=MUTE, size=11),
            title=dict(font=dict(color=MUTE, size=12), standoff=10),
        ),
        legend=dict(font=dict(color=MUTE, size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(
            bgcolor=NAVY,
            bordercolor="rgba(0,0,0,0)",
            font=dict(family=FONT, color="white", size=12),
        ),
        margin=dict(l=12, r=12, t=46, b=12),
    )
)
TEMPLATE = "hmda"


# Resolve CSVs relative to this script rather than the process working directory.
# The pipeline (build_data.py / the notebook) writes exports to <project>/data/processed,
# so default there; fall back to the script directory for bundled deployments where the
# CSVs sit next to index.py. Override with HMDA_DATA_DIR=/path/to/exports when the
# dashboard and data are stored separately (deployment container, Colab-mounted Drive).
def _default_data_dir() -> Path:
    here = Path(__file__).resolve().parent
    processed = here.parent / "data" / "processed"
    return processed if processed.exists() else here


DATA_DIR = Path(os.getenv("HMDA_DATA_DIR", _default_data_dir())).expanduser().resolve()
LOGGER = logging.getLogger("hmda_dashboard")
if not LOGGER.handlers:
    logging.basicConfig(
        level=os.getenv("HMDA_LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )
DATA_LOAD_ISSUES: dict[str, str] = {}


def blank(msg="Run the notebook first: data file not found."):
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=MUTE))
    f.update_layout(
        template=TEMPLATE, height=360, xaxis_visible=False, yaxis_visible=False
    )
    return f


def read(path: str | Path, *, required: bool = False, **kw: Any) -> pd.DataFrame | None:
    """Read a dashboard export without hiding malformed-file errors.

    Missing optional exports degrade the relevant chart to an explanatory blank state.
    Malformed files are also recorded so the dashboard can surface the exact problem in
    its data-health banner instead of silently behaving as though the file never existed.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = DATA_DIR / resolved
        # Interim exports (e.g. hmda_approve_deny.csv) live in data/interim, a sibling
        # of the processed-export DATA_DIR; fall back there when not found in DATA_DIR.
        if not resolved.exists():
            interim = DATA_DIR.parent / "interim" / Path(path).name
            if interim.exists():
                resolved = interim
    key = resolved.name
    try:
        frame = pd.read_csv(resolved, **kw)
        DATA_LOAD_ISSUES.pop(key, None)
        return frame
    except FileNotFoundError:
        message = f"missing from {resolved.parent}"
        DATA_LOAD_ISSUES[key] = message
        if required:
            LOGGER.error("Required data file %s is %s", key, message)
        else:
            LOGGER.info("Optional data file %s is %s", key, message)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        message = f"could not be parsed: {exc}"
        DATA_LOAD_ISSUES[key] = message
        LOGGER.exception("Unable to parse %s", resolved)
    except OSError as exc:
        message = f"could not be read: {exc}"
        DATA_LOAD_ISSUES[key] = message
        LOGGER.exception("Unable to read %s", resolved)
    return None


def has_columns(df: pd.DataFrame | None, columns: Iterable[str]) -> bool:
    """Return True only when a frame exists and contains every requested column."""
    return df is not None and set(columns).issubset(df.columns)


def numeric_series(df: pd.DataFrame | None, column: str) -> pd.Series:
    """Safely coerce a dashboard column to numeric values for calculations."""
    if df is None or column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


# ============================================================ SCHEMA NORMALIZATION
# The notebook's own Phase 3/4 export cells (HMDA.ipynb) write a minimal schema
# (antecedent, consequent, n, confidence, lift, ...). A separate enrichment pass can
# add reader-friendly columns (if_readable, then, recommendation, outlier_type, ...),
# but the dashboard must not *require* that pass to have run. Whichever CSV shows up
# in this folder (fresh from Colab, a local notebook run, or the enriched build
# scripts), the normalizers below fill in anything missing so nothing downstream KeyErrors.
TOTAL_DECISIONED = int(
    os.getenv("HMDA_TOTAL_DECISIONED", "67827")
)  # fallback when support is absent

ITEM_LABELS = {
    "debt_to_income_ratio=>60%": "debt-to-income ratio di atas 60%",
    "debt_to_income_ratio=50%-60%": "debt-to-income ratio 50-60%",
    "debt_to_income_ratio=30%-<36%": "debt-to-income ratio 30-36%",
    "lien_status=Subordinate_Lien": "loan adalah subordinate (second) lien",
    "loan_type=Conventional": "dibiayai lewat conventional loan",
    "construction_method=Manufactured": "properti adalah manufactured housing",
    "loan_purpose=Home_Purchase": "tujuannya pembelian rumah",
    "income_band=<30k": "income pemohon di bawah $30k",
    "preapproval=Requested": "pemohon meminta preapproval",
    "loan_amount_band=300-500k": "loan amount $300k-500k",
}

RECS = {
    frozenset(
        ["debt_to_income_ratio=>60%", "lien_status=Subordinate_Lien"]
    ): "Cegat sebelum underwriting penuh; arahkan ke alternatif debt-consolidation alih-alih second lien.",
    frozenset(
        ["debt_to_income_ratio=>60%"]
    ): "Tandai di intake: menurunkan DTI di bawah 60% (co-borrower, tenor lebih panjang, loan lebih kecil) adalah perbaikan dengan leverage tertinggi.",
    frozenset(
        ["construction_method=Manufactured", "loan_type=Conventional"]
    ): "Arahkan pembeli manufactured-home ke FHA/VA atau program khusus chattel alih-alih jalur conventional.",
    frozenset(
        ["debt_to_income_ratio=50%-60%", "loan_type=Conventional"]
    ): "Arahkan pemohon DTI 50-60% ke underwriting FHA/VA, yang jauh lebih toleran pada band ini dibanding conventional.",
    frozenset(
        ["construction_method=Manufactured", "loan_purpose=Home_Purchase"]
    ): "Untuk pembelian manufactured-home, verifikasi kepemilikan tanah/fondasi permanen - status sebagai real property membuka program mortgage standar.",
    frozenset(
        ["income_band=<30k", "loan_type=Conventional"]
    ): "Pasangkan pemohon berincome di bawah $30k dengan bantuan down-payment sebelum aplikasi conventional.",
    frozenset(
        ["income_band=<30k"]
    ): "Arahkan ke manual-underwrite / program bantuan - semua band income lain disetujui pada tingkat mayoritas.",
    frozenset(
        ["construction_method=Manufactured"]
    ): "Perlakukan manufactured housing sebagai jalur underwriting tersendiri; penaltinya struktural, tidak dijelaskan oleh income atau lokasi.",
    frozenset(
        ["preapproval=Requested"]
    ): "Dorong jalur preapproval - memindahkan penolakan ke tahap awal yang murah; berkas yang sampai keputusan akhir nyaris tidak pernah gagal.",
    frozenset(
        ["debt_to_income_ratio=30%-<36%", "loan_purpose=Home_Purchase"]
    ): "Fast-track aplikasi ini - profil keterjangkauan ideal dengan collateral tervalidasi oleh harga jual arm's-length.",
    frozenset(
        ["debt_to_income_ratio=30%-<36%", "loan_amount_band=300-500k"]
    ): "Fast-track - sweet spot conforming-loan: cukup besar menutup biaya tetap, cukup kecil tetap dalam batas GSE.",
}

OUTLIER_TYPE = {
    "DATA ERROR": "Nilai mustahil (kesalahan input/proses data)",
    "RARE BUT VALID": "Rekaman ekstrem yang sah (jumbo loan, multi-unit, atau properti investasi bernilai tinggi)",
    "RISK SIGNAL": "Kombinasi risiko tidak biasa tapi masuk akal (mis. leverage tinggi + income rendah)",
    "MANUAL REVIEW": "Magnitudo ekstrem, diperiksa manual",
}


def _rule_phrase(antecedent):
    items = str(antecedent).split(", ")
    return " dan ".join(
        ITEM_LABELS.get(it, it.replace("_", " ").replace("=", ": ")) for it in items
    )


def _rule_recommendation(antecedent, then, confidence, n):
    key = frozenset(str(antecedent).split(", "))
    if key in RECS:
        return RECS[key]
    verb = "ditolak" if then == "Denied" else "disetujui"
    tail = (
        "tandai untuk review atau program alternatif sebelum underwriting penuh"
        if then == "Denied"
        else "kandidat kuat untuk fast-track"
    )
    return f"Aplikasi yang cocok dengan profil ini {verb} {confidence*100:.0f}% dari waktu ({n:,} kasus serupa); {tail}."


def _normalize_rules(df):
    if df is None or not len(df):
        return df
    d = df.copy()
    if "then" not in d.columns:
        d["then"] = d["consequent"].apply(
            lambda c: "Denied" if "Denied" in str(c) else "Originated"
        )
    if "if_readable" not in d.columns:
        d["if_readable"] = d["antecedent"].apply(_rule_phrase)
    if "n_matched" not in d.columns:
        d["n_matched"] = d["n"] if "n" in d.columns else 0
    if "support" not in d.columns:
        d["support"] = d["n_matched"] / TOTAL_DECISIONED
    if "recommendation" not in d.columns:
        d["recommendation"] = [
            _rule_recommendation(a, t, c, n)
            for a, t, c, n in zip(
                d["antecedent"], d["then"], d["confidence"], d["n_matched"]
            )
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
            lambda v: (
                "Flagged by all 5 anomaly detectors - highest confidence."
                if v == 5
                else f"Flagged by {int(v)} of 5 anomaly detectors."
            )
        )
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
    cols_needed = [
        "kmeans_cluster",
        "income",
        "loan_amount",
        "property_value",
        "combined_loan_to_value_ratio",
        "debt_to_income_ratio",
        "derived_race",
        "loan_purpose",
        "occupancy_type",
        "action_taken",
        "tract_minority_population_percent",
    ]
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
    if (
        ad is None
        or "tract_minority_cat" not in ad.columns
        or "debt_to_income_ratio" not in ad.columns
    ):
        return None
    dti_grp_map = {
        "<20%": "Low(<36%)",
        "20%-<30%": "Low(<36%)",
        "30%-<36%": "Low(<36%)",
        "36%-<43%": "Mid(36-50%)",
        "43%-<50%": "Mid(36-50%)",
        "50%-60%": "High(>50%)",
        ">60%": "High(>50%)",
    }
    tmp = ad.copy()
    tmp["dti_grp"] = (
        tmp["debt_to_income_ratio"]
        .astype(str)
        .map(dti_grp_map)
        .fillna("Unknown/Exempt")
    )
    if "target_approved" not in tmp.columns and "action_taken" in tmp.columns:
        tmp["target_approved"] = (tmp["action_taken"] == "Originated").astype(int)
    if "target_approved" not in tmp.columns:
        return None
    piv_rate = (
        tmp.pivot_table(
            index="tract_minority_cat",
            columns="dti_grp",
            values="target_approved",
            aggfunc="mean",
            observed=True,
        )
        * 100
    )
    piv_n = tmp.pivot_table(
        index="tract_minority_cat",
        columns="dti_grp",
        values="target_approved",
        aggfunc="size",
        observed=True,
    )
    if (
        "Low_Minority" not in piv_rate.index
        or "Majority_Minority" not in piv_rate.index
    ):
        return None
    rows = []
    for grp in ["Low(<36%)", "Mid(36-50%)", "High(>50%)", "Unknown/Exempt"]:
        if grp not in piv_rate.columns:
            continue
        low, maj = (
            piv_rate.loc["Low_Minority", grp],
            piv_rate.loc["Majority_Minority", grp],
        )
        rows.append(
            {
                "dti_group": grp,
                "low_minority_approval_pct": round(low, 1),
                "majority_minority_approval_pct": round(maj, 1),
                "gap_pp": round(low - maj, 1),
                "n_low_minority": int(piv_n.loc["Low_Minority", grp]),
                "n_majority_minority": int(piv_n.loc["Majority_Minority", grp]),
            }
        )
    return pd.DataFrame(rows) if rows else None


# LOAD DATA
profiles = read("p2_cluster_profiles.csv", required=True)
rules = _normalize_rules(
    read("p3_decision_rules_final.csv", required=True)
)  # pruned, business-relevant set (primary)
rules_all = _normalize_rules(
    read("p3_decision_rules.csv")
)  # all 28 raw candidates, same schema
clarans = _ensure_clarans()  # CLARANS medoid-sample comparison
dbscan_scatter = read(
    "dash_dbscan_scatter.csv"
)  # DBSCAN cluster ids (-1 = noise), 20k sample
hier_scatter = read(
    "dash_hierarchical_scatter.csv"
)  # Ward hierarchical cluster ids, 4k sample
outlier_tax = read(
    "dash_outlier_taxonomy.csv"
)  # per-record global/contextual outlier labels
outlier_tax_summary = read(
    "dash_outlier_taxonomy_summary.csv"
)  # full-population category counts (not sampled)
collective_pattern = read(
    "dash_collective_pattern.csv"
)  # loan-amount-rounding collective-outlier evidence
collective_groups = read(
    "dash_collective_groups.csv"
)  # group-level Isolation Forest hits
collective_summary = read(
    "dash_collective_summary.csv"
)  # counts behind the group panel
gender_gap = read("dash_gender_gap.csv")  # approval by applicant sex within DTI groups
triage = _normalize_triage(
    read("p4_anomaly_triage.csv", index_col=0)
)  # top-15 anomalies, verdict + evidence
geo_gap = _ensure_geo_gap()  # approval gap by tract minority x DTI band
disparity = read("dash_approval_disparity.csv")
scatter = read("dash_scatter.csv", required=True)
denial = read("dash_denial_reasons.csv")
appdeny_full = read(
    "hmda_approve_deny.csv"
)  # every decisioned application with raw/binned fields, for the What-If combined filter
if appdeny_full is not None and "loan_term" in appdeny_full.columns:
    # Loan duration used to sit in its own panel and did not filter anything. Banding it
    # here lets it join the combined profile lookup like every other attribute.
    _t = pd.to_numeric(appdeny_full["loan_term"], errors="coerce")
    appdeny_full["term_band"] = pd.cut(
        _t,
        bins=[-np.inf, 120, 180, 240, 300, 360, np.inf],
        labels=["<=10yr", "15yr", "20yr", "25yr", "30yr", ">30yr"],
    ).astype("string")
if appdeny_full is not None and "target_approved" not in appdeny_full.columns:
    if "action_taken" in appdeny_full.columns:
        appdeny_full["target_approved"] = (
            appdeny_full["action_taken"] == "Originated"
        ).astype(int)
    else:
        appdeny_full = None

# Which of the 28 raw candidates also survived the improvement filter into the 11-rule
# business set - shown as a column on the "all candidates" table so the two views are
# directly comparable rather than living in unrelated tables.
if rules_all is not None and rules is not None:
    _kept_keys = set(zip(rules["antecedent"], rules["consequent"]))
    rules_all["kept"] = [
        ("Yes" if k in _kept_keys else "No")
        for k in zip(rules_all["antecedent"], rules_all["consequent"])
    ]
elif rules_all is not None:
    rules_all["kept"] = "No"
# Fase 1 (preprocessing) aggregates - written by the notebook / build_data.py.
phase1_missing = read("dash_phase1_missingness.csv")  # field, missing_pct, fate
phase1_features = read(
    "dash_phase1_feature_importance.csv"
)  # feature, corr, mi, score, role
phase1_summary = read(
    "dash_phase1_cleaning_summary.csv"
)  # raw_rows, clean_rows, duplicates_removed, ...
# Fase 2: head-to-head clustering validity metrics (silhouette, DB, CH, ARI vs K-Means)
clustering_cmp = read("dash_clustering_comparison.csv")
state_summary = read(
    "dash_state_summary.csv"
)  # state_code, n, approval_rate, median_income, median_loan, ...
state_dti = read("dash_state_dti.csv")  # state_code x DTI band -> approval_rate, n
state_segment = read("dash_state_segment.csv")  # state_code x kmeans_cluster -> n
term_summary = read(
    "dash_term_summary.csv"
)  # loan_term_band -> n, approval_rate, ... (duration proxy for "time")
context_fields = read(
    "dash_context_fields.csv"
)  # field, value, n, approval_rate (non-demographic What-If context)
TERM_ORDER = ["<=10yr", "15yr", "20yr", "25yr", "30yr", ">30yr"]
if scatter is None:
    full = read("p4_anomaly_flags.csv")
    if full is not None:
        scatter = full.sample(min(8000, len(full)), random_state=42)


# SELF-LABELLING
# Indonesian display labels for the notebook's English segment_name values. Only the
# shown label is translated; SEGMENT_BLURB / CLUSTER_RECS stay keyed on the raw English
# segment_name (which is what the data carries), so their lookups are unaffected.
SEGMENT_LABEL_ID = {
    "Refinancers (rate & cash-out)": "Refinancer (rate & cash-out)",
    "Property investors": "Investor properti",
    "Mainstream prime purchasers": "Konsumen primer",
    "Manufactured-housing applicants": "Pemohon manufactured-housing",
    "DTI-stressed borrowers": "Peminjam DTI tinggi",
    "Jumbo / high-net-worth buyers": "Pembeli jumbo / high-net-worth",
    "Small-loan borrowers": "Peminjam loan kecil",
    "Unlabelled segment": "Segmen tanpa label",
}


def name_clusters(p):
    """Use the notebook's own business naming (segment_name in p2_cluster_profiles.csv)
    instead of a second, separate heuristic. A simpler local rule that only checks
    manufactured/DTI/investment/income/loan-size/tract-minority falls through to the
    same generic bucket for any two "normal" clusters that differ mainly in refinance
    vs. purchase mix (e.g. C0 and C2), producing misleadingly identical labels even
    though the underlying segments are very different (94.5% refinance vs. 0%, 79.7%
    vs. 91.5% approval). The notebook's naming already checks that distinction."""
    if not has_columns(p, ["kmeans_cluster", "segment_name"]):
        return {}
    labels: dict[int, str] = {}
    for _, row in p.dropna(subset=["kmeans_cluster"]).iterrows():
        cluster_id = int(row["kmeans_cluster"])
        segment_name = str(row.get("segment_name", "Unlabelled segment"))
        segment_name = SEGMENT_LABEL_ID.get(segment_name, segment_name)
        labels[cluster_id] = f"C{cluster_id} - {segment_name}"
    return labels


CLUSTER_NAMES = name_clusters(profiles)


def clabel(cid):
    try:
        return CLUSTER_NAMES.get(int(cid), f"C{int(cid)}")
    except Exception:
        return str(cid)


# Business-framed one-line description per segment archetype (keyed off the notebook's
# own auto-generated segment_name, so it tracks whatever names this run produced).
SEGMENT_BLURB = {
    "Refinancers (rate & cash-out)": "Pemilik rumah yang melakukan refinance untuk bunga atau menarik cash-out. Sudah terbukti membayar "
    "mortgage, sehingga tingkat persetujuan mendekati norma portfolio: pertanyaan underwriting-nya adalah equity, bukan kelayakan kredit.",
    "Property investors": "100% aplikasi dengan investment-occupancy. Persetujuan cukup kuat tetapi sedikit di bawah prime purchasers, karena "
    "lender memperhitungkan bahwa properti investasi adalah pembayaran pertama yang dilewati saat resesi.",
    "Mainstream prime purchasers": "Segmen terbesar dan tolok ukur persetujuan untuk seluruh portfolio. Nyaris tanpa risk flag: "
    'inilah wujud aplikasi yang "bersih".',
    "Manufactured-housing applicants": "Ditentukan oleh tipe properti, bukan income atau leverage. Persetujuan anjlok hingga jauh di bawah "
    "setengah. Lihat tab Aturan: penalti ini bersifat struktural (jalur pembiayaan + jenis transaksi), tidak dijelaskan oleh income.",
    "DTI-stressed borrowers": "Setiap aplikasi di sini memiliki debt-to-income di atas 50%. Tingkat persetujuan terendah dalam portfolio, "
    "hampir seluruhnya didorong oleh satu angka: lihat aturan DTI>60% pada tab Aturan.",
    "Jumbo / high-net-worth buyers": "Median income sekitar 5x median portfolio. Persetujuan kuat meski ukuran loan besar: "
    "kemampuan membayar tidak dipertanyakan untuk kelompok ini.",
    "Small-loan borrowers": "Ukuran median loan terkecil dalam portfolio (skala home-improvement). Persetujuan mendekati "
    "norma: nominal kecil membawa risiko yang proporsional kecil.",
}


def segment_blurb(row):
    return SEGMENT_BLURB.get(
        row["segment_name"],
        f"{row['share_of_data']:.0f}% of applications, {row['approval_rate']:.0f}% approval rate.",
    )


# How to approach or handle each segment: one concrete, actionable recommendation per
# cluster, in the same spirit as the "Rekomendasi bisnis" column on the Rules tab.
CLUSTER_RECS = {
    "Refinancers (rate & cash-out)": "Beri segmen ini jalur underwriting refinance khusus (rate-and-term vs. cash-out), bukan "
    "dialihkan lewat underwriting pembelian. Persetujuan sudah mengikuti norma portfolio, jadi tuas di sini "
    "adalah retensi dan cross-sell (produk home-equity), dengan perhatian ekstra pada penarikan equity cash-out.",
    "Property investors": "Tawarkan program DSCR atau portfolio-lender yang dirancang untuk landlord, bukan underwriting standar "
    "owner-occupied. Persetujuan sudah kuat, jadi risiko utama yang perlu dikelola adalah konsentrasi bila buku ini tumbuh cepat.",
    "Mainstream prime purchasers": "Jaga kecepatan segmen ini di atas segalanya: ini populasi terbesar, paling minim friksi, dan tertinggi "
    "persetujuannya, sekaligus baseline alami untuk menguji setiap aturan underwriting baru sebelum diterapkan ke segmen lain.",
    "Manufactured-housing applicants": "Bangun atau bermitra pada produk chattel-lending atau FHA Title I untuk manufactured home, alih-alih "
    "memaksa segmen ini melalui underwriting konvensional, di mana penaltinya struktural, bukan berbasis risiko "
    "(lihat tab Aturan). Pantau eksposur fair-lending, karena tipe properti di sini berkorelasi dengan income dan geografi.",
    "DTI-stressed borrowers": "Saring debt-to-income di tahap intake, sebelum underwriting penuh, dan arahkan penolakan yang hampir pasti ke "
    'rujukan debt-consolidation atau credit-counseling. Tawaran "ditolak tetapi ini jalannya" (secured card, '
    "alat budgeting) menjaga hubungan tetap hidup untuk aplikasi masa depan yang lebih kuat.",
    "Jumbo / high-net-worth buyers": "Perlakukan sebagai segmen relationship-banking: persetujuan sudah kuat, jadi peluangnya ada di layanan dan "
    "cross-sell (private banking, portfolio lending), bukan perubahan underwriting. Perhatikan selera investor, "
    "karena jumbo loan lebih sulit dijual ke GSE dibanding yang conforming.",
    "Small-loan borrowers": "Bangun jalur underwriting yang ramping dan berbiaya lebih rendah (mis. AVM alih-alih appraisal penuh) untuk "
    "loan bernominal kecil; persetujuan sudah mendekati rata-rata portfolio, jadi kendala sebenarnya adalah biaya origination "
    "relatif terhadap ukuran loan, bukan risiko kredit.",
}


def cluster_recommendation(row):
    return CLUSTER_RECS.get(
        row["segment_name"],
        "No specific playbook yet for this segment; monitor approval rate and volume before treating it "
        "differently from the portfolio norm.",
    )


# KPI COMPUTATIONS
def kpis():
    n_apps = (
        int(profiles["n"].sum())
        if profiles is not None
        else (len(scatter) if scatter is not None else 0)
    )
    if (
        scatter is not None
        and "_approved" in scatter
        and scatter["_approved"].notna().any()
    ):
        # decisioned-only rate (Originated / (Originated+Denied)); profiles["n"] includes
        # withdrawn/incomplete rows too, so weighting by it would overstate the denominator.
        appr = float(scatter["_approved"].mean() * 100)
    elif profiles is not None and "approval_rate" in profiles:
        appr = float(np.average(profiles["approval_rate"], weights=profiles["n"]))
    else:
        appr = float("nan")
    n_rules = len(rules) if rules is not None else 0
    max_lift = (
        float(rules["lift"].max()) if rules is not None and len(rules) else float("nan")
    )
    n_clusters = profiles.shape[0] if profiles is not None else 0
    if scatter is not None and "anomaly_votes" in scatter:
        n_anom = int((scatter["anomaly_votes"] >= 3).sum())
    else:
        n_anom = 0
    return [
        ("Aplikasi dianalisis", f"{n_apps:,}", STEEL),
        ("Tingkat persetujuan", f"{appr:.1f}%" if appr == appr else "N/A", GREEN),
        ("Tingkat penolakan", f"{100-appr:.1f}%" if appr == appr else "N/A", RED),
        ("Segmen ditemukan", f"{n_clusters}", TEAL),
        ("Aturan relevan bisnis", f"{n_rules}", NAVY),
        (
            "Lift aturan tertinggi",
            f"{max_lift:.1f}x" if max_lift == max_lift else "N/A",
            AMBER,
        ),
        ("Anomali confidence tinggi", f"{n_anom:,}", "#7d3c98"),
    ]


# ============================================================ FIGURE BUILDERS
@lru_cache(maxsize=1)
def fig_approval_by_cluster():
    if profiles is None:
        return blank()
    d = profiles.copy()
    d["name"] = d["kmeans_cluster"].map(clabel)
    d = d.sort_values("approval_rate")
    f = px.bar(
        d,
        x="approval_rate",
        y="name",
        orientation="h",
        color="approval_rate",
        color_continuous_scale=["#c0392b", "#e0a82e", "#2e8b57"],
        text=d["approval_rate"].map(lambda v: f"{v:.0f}%"),
        labels={"approval_rate": "Tingkat persetujuan (%)", "name": ""},
    )
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(
        template=TEMPLATE,
        height=380,
        coloraxis_showscale=False,
        margin=dict(l=10, r=60, t=30, b=10),
        title="Tingkat persetujuan berbeda tajam antar segmen",
    )
    f.update_xaxes(range=[0, 112])
    f.update_yaxes(automargin=True, tickfont=dict(size=11.5))
    return f


@lru_cache(maxsize=1)
def fig_cluster_sizes():
    if profiles is None:
        return blank()
    d = profiles.copy()
    d["name"] = d["kmeans_cluster"].map(clabel)
    f = px.pie(d, values="n", names="name", hole=0.45, color_discrete_sequence=QUAL)
    f.update_traces(textposition="inside", textinfo="percent")
    f.update_layout(
        template=TEMPLATE,
        height=380,
        title="Porsi aplikasi per segmen",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(font=dict(size=9)),
    )
    return f


NUMERIC_AXES = {
    "income": "Income ($k)",
    "loan_amount": "Loan amount ($)",
    "property_value": "Property value ($)",
    "combined_loan_to_value_ratio": "CLTV (%)",
    "tract_minority_population_percent": "Tract minority population (%)",
}


@lru_cache(maxsize=32)
def fig_cluster_scatter(xcol, ycol):
    if scatter is None or "kmeans_cluster" not in scatter:
        return blank()
    d = scatter.dropna(subset=[xcol, ycol]).copy()
    d = (
        d[(d[xcol] > 0) & (d[ycol] > 0)]
        if xcol in ("loan_amount", "property_value", "income")
        else d
    )
    d["Segment"] = d["kmeans_cluster"].map(clabel)
    logx = xcol in ("loan_amount", "property_value", "income")
    logy = ycol in ("loan_amount", "property_value", "income")
    f = px.scatter(
        d.sample(min(6000, len(d)), random_state=1),
        x=xcol,
        y=ycol,
        color="Segment",
        color_discrete_sequence=QUAL,
        opacity=0.55,
        log_x=logx,
        log_y=logy,
        labels={xcol: NUMERIC_AXES.get(xcol, xcol), ycol: NUMERIC_AXES.get(ycol, ycol)},
    )
    f.update_traces(marker=dict(size=5))
    f.update_layout(
        template=TEMPLATE,
        height=460,
        title="K-Means: segmen dalam ruang fitur",
        legend=dict(font=dict(size=9)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return f


@lru_cache(maxsize=32)
def fig_clarans_scatter(xcol, ycol):
    if clarans is None:
        return blank("Data perbandingan CLARANS tidak ditemukan.")
    d = clarans.dropna(subset=[xcol, ycol]).copy()
    d = (
        d[(d[xcol] > 0) & (d[ycol] > 0)]
        if xcol in ("loan_amount", "property_value", "income")
        else d
    )
    d["CLARANS cluster"] = "M" + d["clarans_cluster"].astype(str)
    logx = xcol in ("loan_amount", "property_value", "income")
    logy = ycol in ("loan_amount", "property_value", "income")
    f = px.scatter(
        d,
        x=xcol,
        y=ycol,
        color="CLARANS cluster",
        color_discrete_sequence=QUAL,
        opacity=0.6,
        log_x=logx,
        log_y=logy,
        labels={xcol: NUMERIC_AXES.get(xcol, xcol), ycol: NUMERIC_AXES.get(ycol, ycol)},
    )
    f.update_traces(marker=dict(size=6))
    f.update_layout(
        template=TEMPLATE,
        height=460,
        title="CLARANS (berbasis medoid): sampel 4.000 aplikasi yang sama",
        legend=dict(font=dict(size=9)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return f


@lru_cache(maxsize=1)
def fig_method_comparison():
    """K-Means vs CLARANS vs Hierarchical, all evaluated on the same 4,000-row sample
    (DBSCAN runs on a different, 20k-row sample and produces a variable cluster count,
    so it gets its own dedicated view instead of forcing it into this comparison)."""
    if clarans is None:
        return blank("Data perbandingan CLARANS tidak ditemukan.")
    km = clarans["kmeans_cluster"].value_counts().sort_index()
    cl = clarans["clarans_cluster"].value_counts().sort_index()
    idx = sorted(set(km.index) | set(cl.index))
    methods = [("K-Means", km, STEEL), ("CLARANS", cl, TEAL)]
    if hier_scatter is not None and "hier_cluster" in hier_scatter.columns:
        hi = hier_scatter["hier_cluster"].value_counts().sort_index()
        idx = sorted(set(idx) | set(hi.index))
        methods.append(("Hierarchical (Ward)", hi, QUAL[6]))
    d = pd.DataFrame(
        {
            "cluster": [f"#{i}" for i in idx] * len(methods),
            "count": [c.get(i, 0) for _, c, _ in methods for i in idx],
            "method": [name for name, _, _ in methods for _ in idx],
        }
    )
    f = px.bar(
        d,
        x="cluster",
        y="count",
        color="method",
        barmode="group",
        color_discrete_map={name: color for name, _, color in methods},
        labels={"cluster": "Cluster #", "count": "Aplikasi (sampel 4rb)"},
    )
    f.update_layout(
        template=TEMPLATE,
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Apakah K-Means, CLARANS, dan Hierarchical sepakat soal ukuran cluster? (sampel 4.000 baris yang sama)",
    )
    return f


@lru_cache(maxsize=32)
def fig_hierarchical_scatter(xcol, ycol):
    if hier_scatter is None:
        return blank("Data perbandingan Hierarchical tidak ditemukan.")
    d = hier_scatter.dropna(subset=[xcol, ycol]).copy()
    d = (
        d[(d[xcol] > 0) & (d[ycol] > 0)]
        if xcol in ("loan_amount", "property_value", "income")
        else d
    )
    d["Hierarchical cluster"] = "H" + d["hier_cluster"].astype(str)
    logx = xcol in ("loan_amount", "property_value", "income")
    logy = ycol in ("loan_amount", "property_value", "income")
    f = px.scatter(
        d,
        x=xcol,
        y=ycol,
        color="Hierarchical cluster",
        color_discrete_sequence=QUAL,
        opacity=0.6,
        log_x=logx,
        log_y=logy,
        labels={xcol: NUMERIC_AXES.get(xcol, xcol), ycol: NUMERIC_AXES.get(ycol, ycol)},
    )
    f.update_traces(marker=dict(size=6))
    f.update_layout(
        template=TEMPLATE,
        height=460,
        title="Hierarchical (Ward linkage): sampel 4.000 aplikasi yang sama",
        legend=dict(font=dict(size=9)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return f


@lru_cache(maxsize=1)
def fig_dbscan_sizes():
    """Single-hue bars for the 17 density clusters (no categorical hue battle across 17
    values - see the palette's own all-pairs cap); noise gets its own colour because it
    is a qualitatively different bucket, not the 18th cluster."""
    if dbscan_scatter is None:
        return blank("DBSCAN comparison data not found.")
    vc = dbscan_scatter["dbscan_cluster"].value_counts().sort_index()
    d = pd.DataFrame(
        {
            "cluster": [("Noise" if i == -1 else f"#{i}") for i in vc.index],
            "count": vc.values,
            "is_noise": [i == -1 for i in vc.index],
        }
    )
    f = px.bar(
        d,
        x="cluster",
        y="count",
        color="is_noise",
        color_discrete_map={True: RED, False: STEEL},
        labels={"cluster": "DBSCAN cluster", "count": "Aplikasi (sampel 20rb)"},
    )
    f.update_layout(
        template=TEMPLATE,
        height=340,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        title="DBSCAN: cluster sizes (red = noise, unclustered)",
    )
    return f


@lru_cache(maxsize=32)
def fig_dbscan_scatter(xcol, ycol):
    """Coloured by Noise vs. Clustered (2 categories) rather than by individual cluster
    id: with 17 density clusters, a per-cluster hue scatter would blow the categorical
    palette's all-pairs cap - the noise/not-noise split is also the distinction that
    actually matters for the Phase 4 anomaly story this chart supports."""
    if dbscan_scatter is None:
        return blank("DBSCAN comparison data not found.")
    d = dbscan_scatter.dropna(subset=[xcol, ycol]).copy()
    d = (
        d[(d[xcol] > 0) & (d[ycol] > 0)]
        if xcol in ("loan_amount", "property_value", "income")
        else d
    )
    d["Status"] = np.where(
        d["dbscan_cluster"] == -1, "Noise (tak ter-cluster)", "Ter-cluster"
    )
    logx = xcol in ("loan_amount", "property_value", "income")
    logy = ycol in ("loan_amount", "property_value", "income")
    f = px.scatter(
        d,
        x=xcol,
        y=ycol,
        color="Status",
        color_discrete_map={"Ter-cluster": STEEL, "Noise (tak ter-cluster)": RED},
        opacity=0.6,
        log_x=logx,
        log_y=logy,
        labels={xcol: NUMERIC_AXES.get(xcol, xcol), ycol: NUMERIC_AXES.get(ycol, ycol)},
    )
    f.update_traces(marker=dict(size=6))
    f.update_layout(
        template=TEMPLATE,
        height=460,
        title="DBSCAN (berbasis densitas): sampel 20.000 aplikasi yang sama",
        legend=dict(font=dict(size=9)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
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
    cols = {
        "if_readable": "Jika (profil pemohon/loan)",
        "then": "Maka",
        "recommendation": "Rekomendasi bisnis",
    }
    keep = [
        "Jika (profil pemohon/loan)",
        "Maka",
        "Support",
        "Confidence",
        "Lift",
        "n",
        "Rekomendasi bisnis",
    ]
    if "kept" in d.columns:
        cols["kept"] = "Diambil?"
        keep.insert(1, "Diambil?")
    if "decision_reason" in d.columns:
        cols["decision_reason"] = "Alasan diambil / tidak"
        keep.append("Alasan diambil / tidak")
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
        return blank("Tidak ada aturan yang cocok dengan filter ini.")
    if outcome == "All":
        f = px.scatter(
            d,
            x="support",
            y="confidence",
            size="lift",
            color="then",
            color_discrete_map={"Denied": RED, "Originated": GREEN},
            hover_name="if_readable",
            size_max=30,
            labels={
                "support": "Support (seberapa umum)",
                "confidence": "Confidence (seberapa andal)",
                "then": "",
            },
        )
    else:
        f = px.scatter(
            d,
            x="support",
            y="confidence",
            size="lift",
            color="lift",
            color_continuous_scale=(
                ["#e0a82e", RED] if outcome == "Denied" else ["#e0a82e", GREEN]
            ),
            hover_name="if_readable",
            size_max=30,
            labels={
                "support": "Support (seberapa umum)",
                "confidence": "Confidence (seberapa andal)",
            },
        )
    f.update_layout(
        template=TEMPLATE,
        height=380,
        coloraxis_showscale=False,
        title=f"Aturan {'Semua' if outcome == 'All' else outcome!r}: support vs confidence (bubble = lift)",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return f


@lru_cache(maxsize=1)
def fig_gender_gap():
    """Approval by applicant sex within DTI groups, mirroring the tract-minority chart."""
    if gender_gap is None or not len(gender_gap):
        return blank("Data gender belum tersedia. Jalankan build_data.py dulu.")
    order = ["Semua", "Low(<36%)", "Mid(36-50%)", "High(>50%)"]
    d = gender_gap[gender_gap["dti_group"].isin(order)].copy()
    d["dti_group"] = pd.Categorical(d["dti_group"], categories=order, ordered=True)
    d = d.sort_values("dti_group")
    f = px.bar(
        d, x="dti_group", y="approval_rate", color="derived_sex", barmode="group",
        text=d["approval_rate"].map(lambda v: f"{v:.0f}%"),
        labels={"dti_group": "Kelompok DTI", "approval_rate": "Tingkat persetujuan (%)",
                "derived_sex": ""},
        color_discrete_sequence=QUAL,
        hover_data={"n": ":,"},
    )
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(
        template=TEMPLATE, height=400,
        title="Persetujuan menurut gender di dalam tiap kelompok DTI",
        legend={"orientation": "h", "y": -0.18, "title": ""},
        margin=dict(l=10, r=10, t=44, b=10),
    )
    f.update_yaxes(range=[0, 100])
    return f


FIELD_ABBREV = {
    "debt_to_income_ratio": "DTI",
    "combined_loan_to_value_ratio": "CLTV",
    "loan_amount_band": "besar loan",
    "income_band": "income",
    "property_value_band": "nilai properti",
    "construction_method": "konstruksi",
    "loan_purpose": "tujuan",
    "loan_type": "jenis loan",
    "lien_status": "lien",
    "occupancy_type": "hunian",
    "preapproval": "preapproval",
    "tract_income_cat": "income tract",
    "tract_minority_cat": "tract",
    "applicant_age": "usia",
    "conforming_loan_limit": "conforming",
    "total_units": "unit",
}


def _short_item(item):
    """Shorten `field=value` so network labels fit without overlapping."""
    item = str(item).strip()
    if "=" not in item:
        return item
    field, value = item.split("=", 1)
    return f"{FIELD_ABBREV.get(field, field)}={value}"


def fig_rule_network(df, outcome, min_lift=1.0):
    d = _filter_rules(df, outcome, min_lift)
    if d is None or not len(d):
        return blank("Tidak ada aturan yang cocok dengan filter ini.")

    # Deterministic bipartite layout: antecedents in one evenly spaced column on the left,
    # decision hubs on the right. A spring layout placed nodes randomly, so long labels
    # collided with each other and ran off the plot edges.
    links = {}
    for _, r in d.iterrows():
        sink = str(r["then"]).upper()
        for a in str(r["antecedent"]).split(", "):
            key = (_short_item(a), sink)
            links[key] = max(links.get(key, 0.0), float(r["lift"]))

    ants = sorted({a for a, _ in links})
    hubs = [h for h in ("DENIED", "ORIGINATED") if any(s == h for _, s in links)]
    if not ants or not hubs:
        return blank("Tidak ada aturan yang cocok dengan filter ini.")

    n = len(ants)
    ay = {a: 1.0 - (i / max(n - 1, 1)) for i, a in enumerate(ants)}
    if len(hubs) == 2:
        hy = {"DENIED": 0.72, "ORIGINATED": 0.28}
    else:
        hy = {hubs[0]: 0.5}

    lifts = list(links.values())
    lo, hi = min(lifts), max(lifts)

    traces = []
    for (a, s), lift in links.items():
        width = 1.2 if hi <= lo else 1.2 + (lift - lo) / (hi - lo) * 3.4
        traces.append(go.Scatter(
            x=[0.0, 1.0], y=[ay[a], hy[s]], mode="lines",
            line=dict(width=width, color=RED if s == "DENIED" else GREEN),
            opacity=0.35, hoverinfo="text",
            hovertext=f"{a} → {s}<br>lift {lift:.2f}", showlegend=False,
        ))

    deg = {a: sum(1 for (x, _) in links if x == a) for a in ants}
    traces.append(go.Scatter(
        x=[0.0] * n, y=[ay[a] for a in ants], mode="markers+text",
        text=ants, textposition="middle left", textfont=dict(size=10, color=INK),
        marker=dict(size=[13 + 4 * deg[a] for a in ants], color=STEEL,
                    line=dict(width=1.5, color="white")),
        hoverinfo="text", showlegend=False,
    ))
    traces.append(go.Scatter(
        x=[1.0] * len(hubs), y=[hy[h] for h in hubs], mode="markers+text",
        text=hubs, textposition="middle right",
        textfont=dict(size=11, color=INK, family=FONT),
        marker=dict(size=34, color=[RED if h == "DENIED" else GREEN for h in hubs],
                    line=dict(width=2, color="white")),
        hoverinfo="text", showlegend=False,
    ))

    title = ("Apa pendorong DENIED vs. ORIGINATED" if outcome == "All"
             else f"Apa pendorong {outcome.upper()}")
    f = go.Figure(traces)
    f.update_layout(
        template=TEMPLATE,
        height=max(380, 34 * n + 90),
        showlegend=False,
        title=title,
        margin=dict(l=10, r=10, t=44, b=10),
    )
    # Generous x-padding: labels extend outward from the nodes on both sides.
    f.update_xaxes(visible=False, range=[-1.05, 1.55])
    f.update_yaxes(visible=False, range=[-0.12, 1.12])
    return f


@lru_cache(maxsize=1)
def fig_anomaly_scatter():
    if scatter is None or "loan_amount" not in scatter:
        return blank()
    d = scatter.dropna(subset=["income", "loan_amount"]).copy()
    d = d[(d["income"] > 0) & (d["loan_amount"] > 0)]
    if "anomaly_votes" not in d:
        d["anomaly_votes"] = 0
    f = px.scatter(
        d,
        x="income",
        y="loan_amount",
        color="anomaly_votes",
        color_continuous_scale="OrRd",
        log_x=True,
        log_y=True,
        opacity=0.6,
        labels={
            "income": "Income ($k, log)",
            "loan_amount": "Loan Amount ($, log)",
            "anomaly_votes": "Jumlah metode menandai",
        },
    )
    f.update_layout(
        template=TEMPLATE,
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Outlier: kombinasi loan/income ekstrem naik ke kanan-atas",
    )
    return f


@lru_cache(maxsize=1)
def fig_iso_hist():
    if scatter is None or "iso_score" not in scatter:
        return blank()
    f = px.histogram(
        scatter.dropna(subset=["iso_score"]),
        x="iso_score",
        nbins=60,
        color_discrete_sequence=[STEEL],
        labels={"iso_score": "Skor Anomali Isolation-Forest"},
    )
    f.update_layout(
        template=TEMPLATE,
        height=300,
        title="Distribusi skor anomali (ekor = outlier)",
        margin=dict(l=10, r=10, t=40, b=10),
        bargap=0.02,
    )
    return f


@lru_cache(maxsize=1)
def fig_vote_breakdown():
    if scatter is None or "anomaly_votes" not in scatter:
        return blank()
    vc = scatter["anomaly_votes"].astype(int).value_counts().sort_index().reset_index()
    vc.columns = ["votes", "rows"]
    f = px.bar(
        vc,
        x="votes",
        y="rows",
        text="rows",
        color="votes",
        color_continuous_scale="OrRd",
        labels={"votes": "# metode sepakat", "rows": "Rekaman"},
    )
    f.update_layout(
        template=TEMPLATE,
        height=300,
        coloraxis_showscale=False,
        title="Kesepakatan detektor (3+ = anomali confidence tinggi)",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return f


# ---- Outlier taxonomy: global vs. contextual/local vs. collective ----
# IQR, Z-score and Isolation Forest all score a record against the WHOLE dataset's
# distribution, unconditionally - the textbook definition of a global outlier. LOF and
# DBSCAN-noise instead score a record against its own local neighbourhood's density -
# the textbook definition of a contextual/local outlier (the neighbourhood IS the
# context). "Normal" is drawn first (bottom layer, muted) so the three outlier colours
# read clearly on top rather than getting lost in 89,000 background points.
TAXONOMY_ORDER = [
    "Normal",
    "Global outlier",
    "Contextual/local outlier",
    "Both (global + contextual)",
]
TAXONOMY_COLOR = {
    "Normal": "#c7d0da",
    "Global outlier": QUAL[0],
    "Contextual/local outlier": QUAL[1],
    "Both (global + contextual)": QUAL[2],
}


@lru_cache(maxsize=32)
def fig_outlier_taxonomy(xcol, ycol):
    if outlier_tax is None or not len(outlier_tax):
        return blank("Data taksonomi outlier tidak ditemukan.")
    d = outlier_tax.dropna(subset=[xcol, ycol]).copy()
    d = (
        d[(d[xcol] > 0) & (d[ycol] > 0)]
        if xcol in ("loan_amount", "property_value", "income")
        else d
    )
    logx = xcol in ("loan_amount", "property_value", "income")
    logy = ycol in ("loan_amount", "property_value", "income")
    f = px.scatter(
        d,
        x=xcol,
        y=ycol,
        color="category",
        category_orders={"category": TAXONOMY_ORDER},
        color_discrete_map=TAXONOMY_COLOR,
        opacity=0.65,
        log_x=logx,
        log_y=logy,
        labels={
            xcol: NUMERIC_AXES.get(xcol, xcol),
            ycol: NUMERIC_AXES.get(ycol, ycol),
            "category": "",
        },
    )
    f.update_traces(marker=dict(size=6))
    f.update_layout(
        template=TEMPLATE,
        height=460,
        title="Posisi tiap tipe outlier dalam ruang fitur",
        legend=dict(font=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return f


def _outlier_taxonomy_examples():
    if outlier_tax is None or not len(outlier_tax):
        return html.Div("Data not found.", style={"color": MUTE})
    cols = [
        "income",
        "loan_amount",
        "property_value",
        "combined_loan_to_value_ratio",
        "occupancy_type",
        "category",
    ]
    rows = []
    for cat in [
        "Global outlier",
        "Contextual/local outlier",
        "Both (global + contextual)",
    ]:
        sub = (
            outlier_tax[outlier_tax["category"] == cat]
            .sort_values("iso_score", ascending=False)
            .head(3)
        )
        rows.append(sub[cols])
    d = pd.concat(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=cols)
    d = d.rename(
        columns={
            "income": "Income ($k)",
            "loan_amount": "Loan amount",
            "property_value": "Property value",
            "combined_loan_to_value_ratio": "CLTV (%)",
            "occupancy_type": "Occupancy",
            "category": "Type",
        }
    )
    sdc = [
        {"if": {"filter_query": f'{{Type}} = "{cat}"'}, "backgroundColor": color + "22"}
        for cat, color in TAXONOMY_COLOR.items()
        if cat != "Normal"
    ]
    return _table(d, style_data_conditional=sdc)


@lru_cache(maxsize=1)
def fig_disparity():
    if disparity is None or not len(disparity):
        return blank()
    f = px.bar(
        disparity,
        x="derived_race",
        y="approval_rate_pct",
        color="tract_minority_cat",
        barmode="group",
        color_discrete_sequence=px.colors.sequential.Blues_r,
        labels={
            "derived_race": "",
            "approval_rate_pct": "Tingkat persetujuan (%)",
            "tract_minority_cat": "Tingkat Minoritas Tract",
        },
    )
    f.update_layout(
        template=TEMPLATE,
        height=460,
        title="Tingkat persetujuan menurut ras × tingkat minoritas lingkungan",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    f.update_xaxes(tickangle=-35, automargin=True, tickfont=dict(size=10.5))
    return f


@lru_cache(maxsize=1)
def fig_dti_geo_gap():
    if geo_gap is None or not len(geo_gap):
        return blank()
    d = geo_gap.melt(
        id_vars=["dti_group", "gap_pp"],
        value_vars=["low_minority_approval_pct", "majority_minority_approval_pct"],
        var_name="tract",
        value_name="approval_pct",
    )
    d["tract"] = d["tract"].map(
        {
            "low_minority_approval_pct": "Tract minoritas-rendah",
            "majority_minority_approval_pct": "Tract mayoritas-minoritas",
        }
    )
    f = px.bar(
        d,
        x="dti_group",
        y="approval_pct",
        color="tract",
        barmode="group",
        color_discrete_map={
            "Tract minoritas-rendah": STEEL,
            "Tract mayoritas-minoritas": RED,
        },
        text=d["approval_pct"].map(lambda v: f"{v:.0f}%"),
        labels={
            "dti_group": "Band debt-to-income",
            "approval_pct": "Tingkat persetujuan (%)",
            "tract": "",
        },
    )
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(
        template=TEMPLATE,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Selisih bertahan setelah mengontrol DTI: tidak menutup dalam band DTI",
    )
    f.update_xaxes(automargin=True, tickfont=dict(size=11))
    f.update_yaxes(range=[0, 100])
    return f


@lru_cache(maxsize=1)
def fig_denial_reasons():
    if denial is None or not len(denial):
        return blank()
    d = denial.sort_values("pct_of_denials", ascending=True)
    f = px.bar(
        d,
        x="pct_of_denials",
        y="reason",
        orientation="h",
        text="pct_of_denials",
        color="pct_of_denials",
        color_continuous_scale="OrRd",
        labels={"pct_of_denials": "% of Denials", "reason": ""},
    )
    f.update_traces(
        texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False
    )
    f.update_layout(
        template=TEMPLATE,
        height=360,
        coloraxis_showscale=False,
        title="Alasan penolakan menurut lender (memvalidasi temuan DTI)",
        margin=dict(l=10, r=60, t=30, b=10),
    )
    f.update_xaxes(range=[0, 42])
    f.update_yaxes(automargin=True, tickfont=dict(size=11.5))
    return f


@lru_cache(maxsize=1)
def fig_approval_by_dti():
    if (
        scatter is None
        or "debt_to_income_ratio" not in scatter
        or "_approved" not in scatter
    ):
        return blank()
    order = ["<20%", "20%-<30%", "30%-<36%", "36%-<43%", "43%-<50%", "50%-60%", ">60%"]
    d = scatter.dropna(subset=["debt_to_income_ratio", "_approved"])
    g = d.groupby("debt_to_income_ratio")["_approved"].mean().mul(100).reset_index()
    g = g[g["debt_to_income_ratio"].isin(order)]
    g["debt_to_income_ratio"] = pd.Categorical(
        g["debt_to_income_ratio"], categories=order, ordered=True
    )
    g = g.sort_values("debt_to_income_ratio")
    f = px.bar(
        g,
        x="debt_to_income_ratio",
        y="_approved",
        text=g["_approved"].map(lambda v: f"{v:.0f}%"),
        color="_approved",
        color_continuous_scale=["#c0392b", "#e0a82e", "#2e8b57"],
        labels={
            "debt_to_income_ratio": "Band debt-to-income",
            "_approved": "Tingkat persetujuan (%)",
        },
    )
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(
        template=TEMPLATE,
        height=360,
        coloraxis_showscale=False,
        title="Persetujuan anjlok saat debt-to-income naik",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    f.update_xaxes(automargin=True, tickfont=dict(size=11))
    f.update_yaxes(range=[0, 100])
    return f


def fig_gauge(value, title, good_high=True):
    lo_c, mid_c, hi_c = (RED, AMBER, GREEN) if good_high else (GREEN, AMBER, RED)
    f = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": NAVY},
                "steps": [
                    {"range": [0, 40], "color": lo_c if good_high else hi_c},
                    {"range": [40, 70], "color": mid_c},
                    {"range": [70, 100], "color": hi_c if good_high else lo_c},
                ],
            },
        )
    )
    f.update_layout(template=TEMPLATE, height=280, margin=dict(l=20, r=20, t=50, b=10))
    return f


STATE_METRICS = {
    "approval_rate": ("Tingkat persetujuan (%)", ["#c0392b", "#e0a82e", "#2e8b57"]),
    "n": ("Aplikasi (volume)", "Blues"),
    "median_income": ("Median income ($k)", "Greens"),
    "median_loan": ("Median loan amount ($)", "Purples"),
}


STATE_NAMES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "PR": "Puerto Rico",
    "GU": "Guam",
    "VI": "U.S. Virgin Islands",
    "AS": "American Samoa",
    "MP": "Northern Mariana Islands",
}


def sname(code):
    """Full state name for display. The 2-letter code stays the map's location value."""
    return STATE_NAMES.get(str(code).upper(), str(code))


@lru_cache(maxsize=8)
def fig_state_map(metric):
    if state_summary is None or not len(state_summary):
        return blank(
            "Data ringkasan negara bagian tidak ditemukan. Jalankan notebook dulu."
        )
    d = state_summary.copy()
    d["state_name"] = d["state_code"].map(sname)
    label, scale = STATE_METRICS.get(metric, ("Nilai", "Blues"))
    f = px.choropleth(
        d,
        locations="state_code",
        locationmode="USA-states",
        color=metric,
        scope="usa",
        color_continuous_scale=scale,
        custom_data=[
            "n",
            "approval_rate",
            "median_income",
            "median_loan",
            "top_denial_reason",
            "state_name",
        ],
        labels={metric: label},
    )
    # "%{hover_name}" is not a real Plotly token (that bug showed a bare "-" instead of
    # the state code); "%{location}" is the correct token for a choropleth's `locations` column.
    f.update_traces(
        hovertemplate="<b>%{customdata[5]}</b><br>Aplikasi: %{customdata[0]:,}<br>"
        "Tingkat persetujuan: %{customdata[1]:.1f}%<br>"
        "Median income: $%{customdata[2]:.0f}k<br>"
        "Median loan: $%{customdata[3]:,.0f}<br>"
        "Alasan penolakan teratas: %{customdata[4]}<extra></extra>"
    )
    f.update_layout(
        template=TEMPLATE,
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"{label} per negara bagian, klik untuk rincian di bawah",
        geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor=BG),
    )
    return f


@lru_cache(maxsize=64)
def fig_state_dti(state):
    if state_dti is None or not len(state_dti):
        return blank()
    d = state_dti[state_dti["state_code"] == state].copy()
    if not len(d):
        return blank(
            f"Tidak ada rincian band DTI untuk {sname(state)}, aplikasi berkeputusannya terlalu sedikit."
        )
    order = ["<20%", "20%-<30%", "30%-<36%", "36%-<43%", "43%-<50%", "50%-60%", ">60%"]
    d["dti_band"] = pd.Categorical(
        d["dti_band"],
        categories=[o for o in order if o in d["dti_band"].values],
        ordered=True,
    )
    d = d.sort_values("dti_band")
    f = px.bar(
        d,
        x="dti_band",
        y="approval_rate",
        text=d["approval_rate"].map(lambda v: f"{v:.0f}%"),
        color="approval_rate",
        color_continuous_scale=["#c0392b", "#e0a82e", "#2e8b57"],
        labels={
            "dti_band": "Band debt-to-income",
            "approval_rate": "Tingkat persetujuan (%)",
        },
    )
    f.update_traces(textposition="outside", cliponaxis=False)
    f.update_layout(
        template=TEMPLATE,
        height=340,
        coloraxis_showscale=False,
        title=f"{sname(state)}: persetujuan per band DTI",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    f.update_xaxes(tickangle=-30, automargin=True, tickfont=dict(size=10.5))
    f.update_yaxes(range=[0, 100])
    return f


@lru_cache(maxsize=64)
def fig_state_segment(state):
    if state_segment is None or not len(state_segment):
        return blank()
    d = state_segment[state_segment["state_code"] == state].copy()
    if not len(d):
        return blank(f"Tidak ada rincian segmen untuk {sname(state)}.")
    d["Segment"] = d["kmeans_cluster"].map(clabel)
    f = px.pie(d, values="n", names="Segment", hole=0.45, color_discrete_sequence=QUAL)
    f.update_traces(textposition="inside", textinfo="percent")
    f.update_layout(
        template=TEMPLATE,
        height=320,
        title=f"{sname(state)}: komposisi segmen",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(font=dict(size=8)),
    )
    return f


@lru_cache(maxsize=64)
def _geo_state_detail_children(state):
    """Drill-down content for one state: KPIs, DTI breakdown, segment mix.
    Called both for the tab's initial default state and from the map-click callback,
    so clicking the map is the only interaction needed (no separate location dropdown).
    """
    if not state or state_summary is None:
        return html.Div()
    row = state_summary[state_summary["state_code"] == state]
    if not len(row):
        return html.Div()
    row = row.iloc[0]
    appr = float(row["approval_rate"])
    color = GREEN if appr >= 70 else (AMBER if appr >= 50 else RED)
    state_kpis = [
        ("Aplikasi", f"{int(row['n']):,}", STEEL),
        ("Tingkat persetujuan", f"{appr:.1f}%", color),
        ("Median income", f"${row['median_income']:.0f}k", TEAL),
        ("Median loan", f"${row['median_loan']:,.0f}", NAVY),
        ("Alasan penolakan teratas", str(row["top_denial_reason"]), "#7d3c98"),
    ]
    return html.Div(
        [
            html.Div(
                f"Menampilkan: {sname(state)}",
                style={"fontSize": "12px", "color": MUTE, "marginBottom": "6px"},
            ),
            html.Div(
                [kpi_card(l, v, c) for l, v, c in state_kpis],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "flexWrap": "wrap",
                    "margin": "4px 0 12px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        panel(
                            f"{sname(state)}: persetujuan per band DTI",
                            [graph(fig_state_dti(state))],
                        ),
                        style={"flex": "1", "minWidth": "340px"},
                    ),
                    html.Div(
                        panel(
                            f"{sname(state)}: komposisi segmen",
                            [graph(fig_state_segment(state))],
                        ),
                        style={"flex": "1", "minWidth": "300px"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
        ]
    )


# ---- Duration (loan-term) "pseudo-time" ----
# HMDA 2022 is a single-year snapshot (activity_year is constant, dropped in Phase 1).
# There is no calendar date to filter on. loan_term (the mortgage's tenor: 15/20/30yr, ...)
# is the one genuinely ordered, temporal-feeling axis the data actually supports, so it
# stands in as one more slider on the What-If tab rather than a whole separate timeline tab.
@lru_cache(maxsize=64)
def fig_term_range_detail(lo_idx, hi_idx):
    if term_summary is None or not len(term_summary):
        return html.Div("Data not found.", style={"color": MUTE})
    bands = TERM_ORDER[lo_idx : hi_idx + 1]
    d = term_summary[term_summary["term_band"].isin(bands)]
    if not len(d):
        return html.Div("No data for this range.", style={"color": MUTE})
    n_tot = int(d["n"].sum())
    appr = float(np.average(d["approval_rate"], weights=d["n"]))
    hidti = float(np.average(d["pct_high_dti"], weights=d["n"]))
    color = GREEN if appr >= 70 else (AMBER if appr >= 50 else RED)
    label = bands[0] if len(bands) == 1 else f"{bands[0]} to {bands[-1]}"
    return html.Div(
        [
            html.Div(
                [
                    kpi_card("Rentang durasi", label, NAVY),
                    kpi_card("Aplikasi", f"{n_tot:,}", STEEL),
                    kpi_card("Tingkat persetujuan", f"{appr:.1f}%", color),
                    kpi_card("Porsi high-DTI (>=50%)", f"{hidti:.1f}%", AMBER),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
        ]
    )


# ============================================================ LAYOUT HELPERS
SOFT_SHADOW = "0 1px 2px rgba(16,42,74,0.05), 0 4px 20px rgba(16,42,74,0.05)"


def kpi_card(label, value, color):
    return html.Div(
        [
            html.Div(
                value,
                style={
                    "fontSize": "27px",
                    "fontWeight": "800",
                    "color": color,
                    "letterSpacing": "-0.6px",
                    "lineHeight": "1.05",
                },
            ),
            html.Div(
                label,
                style={
                    "fontSize": "10.5px",
                    "color": MUTE,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.6px",
                    "marginTop": "7px",
                    "fontWeight": "600",
                },
            ),
        ],
        className="hmda-card",
        style={
            "background": CARD,
            "borderRadius": "14px",
            "padding": "16px 18px",
            "flex": "1",
            "boxShadow": SOFT_SHADOW,
            "minWidth": "132px",
            "textAlign": "center",
            "border": f"1px solid {BORDER}",
            "borderTop": f"3px solid {color}",
        },
    )


# Header link buttons (GitHub / dataset source). Ghost-button styling so they read as
# secondary chrome against the navy gradient rather than competing with the title.
HEADER_LINK_STYLE = {
    "display": "inline-block",
    "padding": "7px 14px",
    "borderRadius": "8px",
    "border": "1px solid rgba(255,255,255,0.35)",
    "background": "rgba(255,255,255,0.10)",
    "color": "#ffffff",
    "fontSize": "12px",
    "fontWeight": "600",
    "textDecoration": "none",
    "letterSpacing": "0.2px",
}


def why(items, title="Justifikasi"):
    """Collapsible methodology justification: a list of (question, answer) pairs.

    Rendered as a native <details> block so every analytical choice carries its
    rationale inline, without crowding the charts.
    """
    rows = []
    for q, a in items:
        rows.append(
            html.Div(
                [
                    html.Div(
                        q,
                        style={
                            "fontWeight": "700",
                            "color": NAVY,
                            "fontSize": "12.5px",
                            "marginBottom": "3px",
                        },
                    ),
                    html.Div(
                        a,
                        style={
                            "fontSize": "12px",
                            "color": INK,
                            "lineHeight": "1.6",
                        },
                    ),
                ],
                style={
                    "marginBottom": "12px",
                    "paddingLeft": "10px",
                    "borderLeft": f"2px solid {GRID}",
                },
            )
        )
    return html.Details(
        [
            html.Summary(
                title,
                style={
                    "cursor": "pointer",
                    "fontWeight": "700",
                    "fontSize": "12.5px",
                    "color": STEEL,
                    "padding": "4px 0",
                    "userSelect": "none",
                },
            ),
            html.Div(rows, style={"marginTop": "12px"}),
        ],
        style={
            "background": "#f8fafd",
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px",
            "padding": "12px 16px",
            "marginBottom": "16px",
        },
    )


def panel(title, children, sub=None):
    head = [
        html.H3(
            title,
            style={
                "margin": "0 0 2px",
                "color": NAVY,
                "fontSize": "15.5px",
                "fontWeight": "700",
                "letterSpacing": "-0.2px",
            },
        )
    ]
    if sub:
        head.append(
            html.Div(
                sub,
                style={
                    "fontSize": "12px",
                    "color": MUTE,
                    "marginBottom": "10px",
                    "lineHeight": "1.5",
                },
            )
        )
    return html.Div(
        head + children,
        className="hmda-card",
        style={
            "background": CARD,
            "borderRadius": "16px",
            "padding": "18px 20px",
            "boxShadow": SOFT_SHADOW,
            "marginBottom": "16px",
            "border": f"1px solid {BORDER}",
        },
    )


def graph(id_or_fig):
    # No zoom of any kind: scroll-zoom off (covers the geo map too), double-click
    # auto-zoom off, modebar hidden. Hover, legend clicks and map clicks still work.
    cfg = {
        "displayModeBar": False,
        "responsive": True,
        "scrollZoom": False,
        "doubleClick": False,
    }
    if isinstance(id_or_fig, str):
        return dcc.Graph(id=id_or_fig, config=cfg)
    return dcc.Graph(figure=id_or_fig, config=cfg)


def approval_meter(value, title):
    """Lightweight HTML approval meter for the hot What-If callback.

    Avoids constructing/serializing a Plotly gauge on every profile change.
    """
    value = float(value) if value is not None and np.isfinite(value) else 0.0
    value = max(0.0, min(100.0, value))
    color = GREEN if value >= 70 else (AMBER if value >= 50 else RED)
    return html.Div(
        [
            html.Div(
                f"{value:.1f}%",
                style={
                    "fontSize": "42px",
                    "fontWeight": "800",
                    "color": color,
                    "lineHeight": "1",
                    "letterSpacing": "-1px",
                },
            ),
            html.Div(
                title,
                style={"fontSize": "11px", "color": MUTE, "marginTop": "7px"},
            ),
            html.Div(
                html.Div(
                    style={
                        "width": f"{value:.1f}%",
                        "height": "100%",
                        "background": color,
                        "borderRadius": "999px",
                    }
                ),
                style={
                    "height": "9px",
                    "background": GRID,
                    "borderRadius": "999px",
                    "overflow": "hidden",
                    "marginTop": "14px",
                },
            ),
        ],
        style={
            "padding": "18px 20px",
            "background": BG,
            "borderRadius": "14px",
            "border": f"1px solid {BORDER}",
        },
    )


def finding_card(title, body, color):
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "fontWeight": "700",
                    "color": color,
                    "fontSize": "13px",
                    "marginBottom": "6px",
                },
            ),
            html.Div(
                body, style={"fontSize": "12.5px", "color": INK, "lineHeight": "1.55"}
            ),
        ],
        className="hmda-card",
        style={
            "background": CARD,
            "borderRadius": "16px",
            "padding": "16px 18px",
            "flex": "1",
            "minWidth": "260px",
            "boxShadow": SOFT_SHADOW,
            "border": f"1px solid {BORDER}",
            "borderLeft": f"4px solid {color}",
        },
    )


# ============================================================ TABLES
def _table(df, cols=None, style_data_conditional=None):
    if df is None:
        return html.Div(
            "Data not found. Run the notebook first.", style={"color": MUTE}
        )
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
        page_size=15,
        page_action="native",
        sort_action="native",
        sort_mode="multi",
        style_as_list_view=True,
        style_table={
            "borderRadius": "12px",
            "overflowX": "auto",
            "border": f"1px solid {BORDER}",
        },
        style_header={
            "backgroundColor": NAVY,
            "color": "white",
            "fontWeight": "700",
            "fontSize": "11.5px",
            "textTransform": "uppercase",
            "letterSpacing": "0.4px",
            "border": "none",
            "padding": "10px 10px",
        },
        style_cell={
            "fontSize": "11.5px",
            "padding": "9px 10px",
            "fontFamily": FONT,
            "color": INK,
            "maxWidth": "340px",
            "whiteSpace": "normal",
            "textAlign": "left",
            "border": "none",
            "borderBottom": f"1px solid {GRID}",
        },
        style_data_conditional=sdc,
    )


def _profiles_cards():
    if profiles is None:
        return html.Div("Data not found.", style={"color": MUTE})
    cards = []
    for _, r in profiles.sort_values("approval_rate", ascending=False).iterrows():
        color = (
            GREEN
            if r["approval_rate"] >= 70
            else (AMBER if r["approval_rate"] >= 50 else RED)
        )
        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                clabel(r["kmeans_cluster"]),
                                style={
                                    "fontWeight": "700",
                                    "fontSize": "13px",
                                    "color": NAVY,
                                },
                            ),
                            html.Span(
                                f"  {r['share_of_data']:.1f}% dari aplikasi",
                                style={"fontSize": "11px", "color": MUTE},
                            ),
                        ]
                    ),
                    html.Div(
                        f"{r['approval_rate']:.0f}% disetujui",
                        style={
                            "fontSize": "20px",
                            "fontWeight": "800",
                            "color": color,
                            "margin": "4px 0",
                        },
                    ),
                    html.Div(
                        segment_blurb(r),
                        style={"fontSize": "12px", "color": INK, "lineHeight": "1.45"},
                    ),
                    html.Div(
                        [
                            html.Span(
                                "Pendekatan yang disarankan: ",
                                style={"fontWeight": "700", "color": NAVY},
                            ),
                            cluster_recommendation(r),
                        ],
                        style={
                            "fontSize": "12px",
                            "color": INK,
                            "lineHeight": "1.45",
                            "marginTop": "8px",
                            "padding": "8px 10px",
                            "background": BG,
                            "borderRadius": "8px",
                        },
                    ),
                    html.Div(
                        f"Median income ${r['med_income']:.0f}k · median loan ${r['med_loan']:,.0f} · median CLTV {r['med_cltv']:.0f}%",
                        style={"fontSize": "10.5px", "color": MUTE, "marginTop": "6px"},
                    ),
                ],
                style={
                    "background": CARD,
                    "borderRadius": "12px",
                    "padding": "14px 16px",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                    "marginBottom": "10px",
                    "borderLeft": f"4px solid {color}",
                },
            )
        )
    return html.Div(cards)


def _anomaly_table():
    if triage is None:
        return _table(None)
    d = triage.reset_index().rename(columns={"index": "row_id"})
    cols = [
        c
        for c in [
            "row_id",
            "income",
            "loan_amount",
            "property_value",
            "combined_loan_to_value_ratio",
            "occupancy_type",
            "action_taken",
            "anomaly_votes",
            "verdict",
            "outlier_type",
            "evidence",
        ]
        if c in d.columns
    ]
    sdc = [
        {"if": {"filter_query": f'{{verdict}} = "{v}"'}, "backgroundColor": c + "22"}
        for v, c in VERDICT_COLOR.items()
    ]
    return _table(d, cols, style_data_conditional=sdc)


# ============================================================ WHAT-IF PREDICTOR
# Ordinal fields get a slider over their full known category order (not just the values
# that happen to appear in a mined rule), so moving it explores the whole spectrum, the
# same way the loan-duration slider already works. A slider always has some value: it
# defaults to the modal (most common) band. Binary fields are a 3-way radio (not
# specified / A / B). Nominal fields with more than 2 options stay dropdowns.
DTI_ORDER = ["<20%", "20%-<30%", "30%-<36%", "36%-<43%", "43%-<50%", "50%-60%", ">60%"]
INCOME_BAND_ORDER = [
    "<30k",
    "30-50k",
    "50-75k",
    "75-100k",
    "100-150k",
    "150-200k",
    ">200k",
]
LOAN_AMOUNT_BAND_ORDER = [
    "<100k",
    "100-200k",
    "200-300k",
    "300-500k",
    "500-750k",
    ">750k",
]
PROPERTY_VALUE_BAND_ORDER = [
    "<100k",
    "100-200k",
    "200-350k",
    "350-500k",
    "500-750k",
    ">750k",
]
CLTV_BAND_ORDER = ["<60%", "60-80%", "80-90%", "90-95%", "95-100%", ">100%"]
AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
UNITS_ORDER = ["1", "2", "3", "4", "5-24", "25-49", "50-99", "100-149", ">149"]
TRACT_INCOME_ORDER = ["Low_Income", "Moderate_Income", "Middle_Income", "Upper_Income"]

# field -> (label, kind, order_or_values, default)
WHATIF_FIELDS = [
    ("debt_to_income_ratio", "Debt-to-income", "dropdown", DTI_ORDER, None),
    (
        "lien_status",
        "Status lien",
        "dropdown",
        ["First_Lien", "Subordinate_Lien"],
        None,
    ),
    (
        "loan_type",
        "Jenis loan",
        "dropdown",
        ["Conventional", "FHA", "VA", "RHS_FSA"],
        None,
    ),
    (
        "construction_method",
        "Metode konstruksi",
        "dropdown",
        ["Site_Built", "Manufactured"],
        None,
    ),
    (
        "loan_purpose",
        "Tujuan loan",
        "dropdown",
        [
            "Home_Purchase",
            "Home_Improvement",
            "Refinance",
            "CashOut_Refinance",
            "Other",
            "NotApplicable",
        ],
        None,
    ),
    ("income_band", "Band income", "dropdown", INCOME_BAND_ORDER, None),
    ("preapproval", "Preapproval", "dropdown", ["Not_Requested", "Requested"], None),
    ("loan_amount_band", "Besar loan", "dropdown", LOAN_AMOUNT_BAND_ORDER, None),
]


def compute_base_approval() -> float:
    """Return the best available decisioned-application approval rate.

    Prefers the full decisioned population over the 8,000-row scatter sample: the What-If
    delta compares a filtered rate against this baseline, so a sample-derived baseline
    would shift every comparison by the sampling error (76.98% vs the true 76.88%).
    """
    full = numeric_series(appdeny_full, "target_approved").dropna()
    if len(full):
        return float(full.mean() * 100)
    approved = numeric_series(scatter, "_approved").dropna()
    if len(approved):
        return float(approved.mean() * 100)
    if has_columns(profiles, ["approval_rate", "n"]):
        rates = numeric_series(profiles, "approval_rate")
        weights = numeric_series(profiles, "n")
        valid = rates.notna() & weights.notna() & (weights > 0)
        if valid.any():
            return float(np.average(rates[valid], weights=weights[valid]))
    return float("nan")


BASE_APPROVAL = compute_base_approval()

# Extra applicant/property/tract fields folded into the combined profile match (see
# combined_match() below). None of them appear in any mined rule antecedent (Rules tab),
# so they don't affect the curated rule set -- but they do narrow the direct historical
# lookup that produces the combined approval rate. Deliberately excludes derived_race /
# derived_ethnicity / derived_sex / tract_minority_cat: those already showed no
# predictive lift beyond DTI on the Rules tab, and a per-individual "pick your race, see
# your approval odds" widget risks reading as normalizing demographic scoring even when
# labeled "context only". That gap is handled properly, with the right caveats, on the
# Fairness tab instead.
CONTEXT_FIELDS = [
    ("applicant_age", "Usia pemohon", "dropdown", AGE_ORDER, None),
    ("occupancy_type", "Jenis hunian", "dropdown", None, None),
    ("total_units", "Jumlah unit", "dropdown", UNITS_ORDER, None),
    ("conforming_loan_limit", "Batas conforming loan", "dropdown", None, None),
    ("term_band", "Durasi loan", "dropdown", TERM_ORDER, None),
    (
        "property_value_band",
        "Nilai properti",
        "dropdown",
        PROPERTY_VALUE_BAND_ORDER,
        None,
    ),
    ("cltv_band", "CLTV", "dropdown", CLTV_BAND_ORDER, None),
    (
        "tract_income_cat",
        "Level income tract",
        "dropdown",
        TRACT_INCOME_ORDER,
        None,
    ),
]

# One combined applicant profile: every control across both lists renders together and
# feeds the same combined_match() filter, rather than living in two disconnected panels.
PROFILE_FIELDS = WHATIF_FIELDS + CONTEXT_FIELDS


def _dropdown_options(field):
    """Real observed values for a nominal dropdown whose category set isn't hardcoded."""
    if context_fields is None:
        return []
    return context_fields.loc[context_fields["field"] == field, "value"].tolist()


# Display labels for raw HMDA option values. Only what the user reads changes; the value
# sent back to the callback stays the raw category so the profile lookup still matches.
VALUE_LABELS = {
    "First_Lien": "Lien pertama",
    "Subordinate_Lien": "Lien kedua",
    "Site_Built": "Dibangun di lokasi",
    "Manufactured": "Manufactured / pabrikan",
    "Home_Purchase": "Beli rumah",
    "Home_Improvement": "Renovasi",
    "Refinance": "Refinance",
    "CashOut_Refinance": "Refinance tarik tunai",
    "Other": "Lainnya",
    "NotApplicable": "Tidak berlaku",
    "Not_Requested": "Tidak diminta",
    "Requested": "Diminta",
    "Conventional": "Conventional",
    "Principal_Residence": "Rumah utama",
    "Second_Residence": "Rumah kedua",
    "Investment": "Investasi",
    "Low_Income": "Income rendah",
    "Moderate_Income": "Income menengah-bawah",
    "Middle_Income": "Income menengah",
    "Upper_Income": "Income atas",
    "Age_NA": "Tidak diketahui",
    "No_CoApplicant": "Tanpa co-applicant",
    "Exempt": "Dikecualikan",
    "Unknown": "Tidak diketahui",
    "C": "Conforming",
    "NC": "Non-conforming",
    "U": "Tidak diketahui",
}


def vlabel(v):
    return VALUE_LABELS.get(str(v), str(v))


CONTROL_WRAP_STYLE = {"minWidth": "180px", "flex": "1 1 180px"}

# DTI is monthly debt over monthly income, and the monthly payment itself is a function of
# loan amount and term. So DTI already aggregates income, loan size, and duration: picking
# it alongside those three double-counts the same financial fact and slices the sample down
# to a handful of applications. The What-If tab therefore offers one basis at a time; these
# wrapper ids let the mode switch hide and clear the unused controls.
FIN_WRAP = {
    "debt_to_income_ratio": "wrap-dti",
    "income_band": "wrap-income",
    "loan_amount_band": "wrap-loan",
    "term_band": "wrap-term",
}


def render_control(
    field, label, kind, order_or_values, default, id_prefix, wrap_id=None
):
    # Every control is a dropdown. Sliders were unusable here: with 6-9 bands their tick
    # labels overlapped into an unreadable smear, and they could not express
    # "not specified" the way the dropdowns and toggles already did.
    opts = order_or_values if order_or_values else _dropdown_options(field)
    children = [
        html.Label(
            label, style={"fontSize": "11px", "color": MUTE, "fontWeight": "600"}
        ),
        dcc.Dropdown(
            id=f"{id_prefix}-{field}",
            options=[{"label": "(tidak ditentukan)", "value": ""}]
            + [{"label": vlabel(v), "value": v} for v in opts],
            value="",
            clearable=False,
            style={"fontSize": "12px"},
        ),
    ]
    kwargs = {"id": wrap_id} if wrap_id else {}
    return html.Div(children, style=dict(CONTROL_WRAP_STYLE), **kwargs)


def decode_control(kind, order_or_values, default, raw):
    # All controls are dropdowns now, so the raw value is already the label itself;
    # "" means the user left the field unspecified.
    return raw or ""


# ------------------------------------------------------------------ FAST PROFILE LOOKUP
# The old callback called `.astype(str)` on whole columns every time a slider/dropdown
# changed. On Vercel that means repeated allocations + full Pandas scans in the hot path.
# Encode each profile field once at process start, then callbacks compare compact int16
# NumPy arrays. The result itself is also cached because users often revisit combinations.
_PROFILE_CODES: dict[str, np.ndarray] = {}
_PROFILE_VALUE_CODES: dict[str, dict[str, int]] = {}
_PROFILE_TARGET = np.empty(0, dtype=np.float32)
_PROFILE_N = 0


def _build_profile_lookup() -> None:
    global _PROFILE_TARGET, _PROFILE_N
    if appdeny_full is None or "target_approved" not in appdeny_full.columns:
        return

    _PROFILE_N = len(appdeny_full)
    _PROFILE_TARGET = pd.to_numeric(
        appdeny_full["target_approved"], errors="coerce"
    ).to_numpy(dtype=np.float32, copy=True)

    for field, *_ in PROFILE_FIELDS:
        if field not in appdeny_full.columns:
            continue
        # String conversion/category discovery happens once, never inside a callback.
        cat = pd.Categorical(appdeny_full[field].astype("string").fillna(""))
        # int16 is ample for these low-cardinality HMDA categories and stays cache-friendly.
        _PROFILE_CODES[field] = cat.codes.astype(np.int16, copy=False)
        _PROFILE_VALUE_CODES[field] = {
            str(value): int(code) for code, value in enumerate(cat.categories)
        }


_build_profile_lookup()


@lru_cache(maxsize=4096)
def _combined_match_cached(filters: tuple[tuple[str, str], ...]) -> tuple[float, int]:
    """Fast cached lookup for one normalized profile combination."""
    base = BASE_APPROVAL if np.isfinite(BASE_APPROVAL) else 0.0
    if _PROFILE_N == 0 or not filters:
        return base, 0

    mask = np.ones(_PROFILE_N, dtype=np.bool_)
    usable = 0
    for field, value in filters:
        codes = _PROFILE_CODES.get(field)
        value_codes = _PROFILE_VALUE_CODES.get(field)
        if codes is None or value_codes is None:
            # Preserve the old behavior: unavailable columns simply do not filter.
            continue
        usable += 1
        code = value_codes.get(str(value))
        if code is None:
            return base, 0
        mask &= codes == code
        if not mask.any():
            return base, 0

    if usable == 0:
        return base, 0

    n = int(np.count_nonzero(mask))
    values = _PROFILE_TARGET[mask]
    if not len(values):
        return base, 0
    rate = float(np.nanmean(values) * 100.0)
    if not np.isfinite(rate):
        rate = base
    return rate, n


def combined_match(selected: list[tuple[str, str, str]]):
    """Direct historical approval lookup optimized for sub-100-ms warm callbacks.

    Returns (approval_rate_pct, n_matched, active), preserving the previous API.
    """
    active = [(label, value) for _, label, value in selected if value]
    filters = tuple((field, str(value)) for field, _, value in selected if value)
    rate, n = _combined_match_cached(filters)
    return rate, n, active
    """Compact, visible status so missing exports never fail silently."""
    if not DATA_LOAD_ISSUES:
        return html.Div(
            [
                html.Span("Data siap", style={"fontWeight": "800", "color": GREEN}),
                html.Span(f" · dimuat dari {DATA_DIR}", style={"color": MUTE}),
            ],
            style={"fontSize": "11px", "padding": "0 24px 8px"},
        )

    preview = "; ".join(
        f"{name}: {issue}" for name, issue in list(DATA_LOAD_ISSUES.items())[:4]
    )
    extra = len(DATA_LOAD_ISSUES) - 4
    if extra > 0:
        preview += f"; +{extra} more"
    return html.Div(
        [
            html.Span("Peringatan data", style={"fontWeight": "800", "color": AMBER}),
            html.Span(f" · {preview}", style={"color": MUTE}),
        ],
        style={
            "fontSize": "11px",
            "padding": "8px 12px",
            "margin": "0 24px 8px",
            "background": "#fff8e8",
            "border": "1px solid #f3d7a0",
            "borderRadius": "10px",
        },
    )


def _top_rule(outcome: str, *, exclude_dti: bool = False) -> pd.Series | None:
    if rules is None or not len(rules) or "then" not in rules.columns:
        return None
    candidates = rules[rules["then"] == outcome].copy()
    if exclude_dti and "antecedent" in candidates.columns:
        candidates = candidates[
            ~candidates["antecedent"]
            .astype(str)
            .str.contains("debt_to_income_ratio", na=False)
        ]
    if not len(candidates):
        return None
    order = [
        column
        for column in ["lift", "confidence", "n_matched"]
        if column in candidates.columns
    ]
    return (
        candidates.sort_values(order, ascending=False).iloc[0]
        if order
        else candidates.iloc[0]
    )


def build_discovery() -> str:
    denial_rule = _top_rule("Denied")
    gap = (
        numeric_series(geo_gap, "gap_pp").max() if geo_gap is not None else float("nan")
    )
    if denial_rule is None:
        return (
            "Dashboard sudah siap, tetapi ekspor aturan terkurasi belum tersedia. "
            "Periksa peringatan data sebelum menafsirkan tab analitis."
        )
    statement = (
        f"Pola penolakan terkuat adalah {denial_rule.get('if_readable', denial_rule.get('antecedent', 'aturan teratas'))}: "
        f"confidence {float(denial_rule.get('confidence', 0)) * 100:.1f}%, "
        f"lift {float(denial_rule.get('lift', 0)):.2f}×, mencakup {int(denial_rule.get('n_matched', 0)):,} aplikasi."
    )
    if np.isfinite(gap):
        statement += (
            f" Terpisah, selisih tingkat persetujuan terbesar antara tract minoritas-rendah dan "
            f"mayoritas-minoritas dalam satu kelompok DTI adalah {gap:.1f} poin persentase; ini adalah "
            "asosiasi yang perlu diselidiki, bukan bukti sebab-akibat."
        )
    return statement


def executive_finding_cards():
    cards = []
    top_denied = _top_rule("Denied")
    if top_denied is not None:
        cards.append(
            finding_card(
                "1. Pola penolakan terkuat",
                f"{top_denied.get('if_readable', top_denied.get('antecedent', 'Aturan'))} ditolak "
                f"{float(top_denied.get('confidence', 0)) * 100:.1f}% dari waktu "
                f"(lift {float(top_denied.get('lift', 0)):.2f}×; n={int(top_denied.get('n_matched', 0)):,}).",
                STEEL,
            )
        )

    high_conf = int((numeric_series(scatter, "anomaly_votes") >= 3).sum())
    reviewed = len(triage) if triage is not None else 0
    data_errors = (
        int((triage.get("verdict") == "DATA ERROR").sum())
        if triage is not None and "verdict" in triage
        else 0
    )
    cards.append(
        finding_card(
            "2. Anomali perlu interpretasi",
            f"{high_conf:,} rekaman mendapat setidaknya tiga suara detektor. Triase meninjau {reviewed:,} "
            f"rekaman ekstrem dan melabeli {data_errors:,} sebagai kesalahan data; sisanya tidak otomatis dianggap data buruk.",
            TEAL,
        )
    )

    non_dti = _top_rule("Denied", exclude_dti=True)
    if non_dti is not None:
        cards.append(
            finding_card(
                "3. Pola penolakan non-DTI terkuat",
                f"{non_dti.get('if_readable', non_dti.get('antecedent', 'Aturan'))} ditolak "
                f"{float(non_dti.get('confidence', 0)) * 100:.1f}% dari waktu "
                f"(lift {float(non_dti.get('lift', 0)):.2f}×; n={int(non_dti.get('n_matched', 0)):,}).",
                AMBER,
            )
        )
    return cards


# ============================================================ DASH APP
app = Dash(
    __name__,
    title="HMDA Knowledge Discovery Dashboard",
    suppress_callback_exceptions=True,
)
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
      html { scroll-behavior: smooth; }
      .hmda-navlink { transition: background .15s ease, border-color .15s ease; }
      .hmda-navlink:hover { background: #eef4fc; border-left-color: #2a78d6 !important; }
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
      button:focus-visible, input:focus-visible, [role="tab"]:focus-visible {
        outline: 3px solid rgba(42,120,214,0.32); outline-offset: 2px;
      }
      @media (max-width: 760px) {
        .hmda-tabs .tab { padding: 10px 9px !important; font-size: 11.5px !important; }
        .rc-slider-mark-text { font-size: 9px !important; }
      }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

DISCOVERY = build_discovery()

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    "DASHBOARD PENEMUAN PENGETAHUAN",
                    style={
                        "fontSize": "11px",
                        "fontWeight": "700",
                        "letterSpacing": "2.5px",
                        "color": "rgba(255,255,255,0.62)",
                        "marginBottom": "8px",
                    },
                ),
                html.H1(
                    "HMDA 2022 · Persetujuan & Penolakan Kredit Rumah",
                    style={
                        "margin": "0",
                        "fontSize": "27px",
                        "fontWeight": "800",
                        "letterSpacing": "-0.7px",
                        "color": "#ffffff",
                    },
                ),
                html.Div(
                    "Kelompok 4 · Home Mortgage Disclosure Act · Sampel 100.000 rekaman (data terbuka CFPB)",
                    style={
                        "fontSize": "13px",
                        "color": "rgba(255,255,255,0.78)",
                        "marginTop": "6px",
                    },
                ),
                html.Div(
                    [
                        html.A(
                            "Source Code",
                            href="https://github.com/leonardo-alexander/HMDA",
                            target="_blank",
                            rel="noopener noreferrer",
                            style=HEADER_LINK_STYLE,
                        ),
                        html.A(
                            "Source Dataset",
                            href="https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2022",
                            target="_blank",
                            rel="noopener noreferrer",
                            style=HEADER_LINK_STYLE,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "10px",
                        "flexWrap": "wrap",
                        "marginTop": "16px",
                    },
                ),
            ],
            style={
                "background": "linear-gradient(120deg, #0e2340 0%, #16345c 46%, #245ea3 100%)",
                "color": "white",
                "padding": "30px 30px 32px",
                "boxShadow": "0 6px 24px rgba(16,42,74,0.20)",
                "borderBottom": "3px solid #2a78d6",
            },
        ),
        html.Div(
            [
                html.Div(
                    "TEMUAN UTAMA",
                    style={
                        "fontWeight": "800",
                        "color": STEEL,
                        "fontSize": "11px",
                        "letterSpacing": "1.4px",
                        "marginBottom": "5px",
                    },
                ),
                html.Span(
                    DISCOVERY,
                    style={"color": INK, "fontSize": "13px", "lineHeight": "1.6"},
                ),
            ],
            style={
                "background": "linear-gradient(135deg, #eef4fc 0%, #f8fbff 100%)",
                "borderLeft": f"4px solid {STEEL}",
                "borderRadius": "12px",
                "padding": "15px 20px",
                "margin": "20px 24px 6px",
                "boxShadow": "0 2px 14px rgba(16,42,74,0.05)",
            },
        ),
        html.Div(
            [kpi_card(l, v, c) for l, v, c in kpis()],
            style={
                "display": "flex",
                "gap": "13px",
                "padding": "12px 24px 4px",
                "flexWrap": "wrap",
            },
        ),
        html.Div(
            dcc.Tabs(
                id="tabs",
                value="summary",
                className="hmda-tabs",
                children=[
                    dcc.Tab(label="Executive Summary", value="summary"),
                    dcc.Tab(label="Fase 1 - Preprocessing", value="fase1"),
                    dcc.Tab(label="Fase 2 - Clustering", value="fase2"),
                    dcc.Tab(label="Fase 3 - Association Rules", value="fase3"),
                    dcc.Tab(label="Fase 4 - Anomaly Detection", value="fase4"),
                    dcc.Tab(label="Fase 5 - Reporting & Fairness", value="fase5"),
                ],
            ),
            style={"padding": "18px 24px 0"},
        ),
        dcc.Loading(
            html.Div(id="tab-content", style={"padding": "6px 24px 34px"}),
            type="circle",
            color=STEEL,
        ),
    ],
    style={"fontFamily": FONT, "minHeight": "100vh", "paddingBottom": "20px"},
)


# ============================================================ Methodology rationale
# Every threshold and algorithm choice below is stated with its reason, so the dashboard
# defends its own methodology instead of presenting bare numbers.
WHY_CLEANING = [
    (
        "Kenapa kolom dengan lebih dari 60% nilai kosong dibuang?",
        "Kolom yang lebih dari separuh isinya kosong tidak bisa diimputasi tanpa mengarang "
        "mayoritas nilainya, dan hasil karangan itu akan mendominasi data aslinya. Ambang 60% "
        "sengaja dibuat longgar, bukan 20-30%, karena banyak field HMDA kosong akibat aturan "
        "pelaporan dan bukan karena salah input. Selama masih ada 40% nilai asli, kolomnya tetap "
        "dipakai.",
    ),
    (
        "Kenapa duplikat dihapus lebih dulu?",
        "Baris kembar membuat pola yang sama terhitung dua kali, sehingga support dan confidence "
        "di Fase 3 ikut menggelembung. Dari 100.000 baris cuma 5 yang benar-benar kembar, jadi "
        "dampaknya kecil, tapi pemeriksaannya tetap perlu didokumentasikan.",
    ),
]

WHY_FEATURE_TYPES = [
    (
        "Kenapa fitur harus dipisah per tipe?",
        "Karena tiap tipe butuh perlakuan berbeda, dan salah perlakuan bikin error yang tidak "
        "kelihatan. Kalau loan_purpose bernilai 1, 2, 3 diperlakukan sebagai angka biasa, model "
        "akan mengira tujuan 3 itu tiga kali lebih besar dari tujuan 1, padahal angkanya cuma kode.",
    ),
    (
        "Kenapa partisinya wajib menutup semua kolom?",
        "Supaya tidak ada kolom yang terlewat diam-diam atau malah masuk dua grup sekaligus. "
        "Pemeriksaan ini dijalankan sebagai assert di notebook, jadi kalau ada kolom baru yang "
        "belum dikategorikan, prosesnya langsung berhenti.",
    ),
    (
        "Kenapa kolom ID dibuang dari pemodelan?",
        "LEI dan census_tract itu penanda, bukan karakteristik pemohon. Kalau ikut dipakai, model "
        "bisa menghafal lender atau wilayah tertentu, dan itu bukan pola yang bisa dipakai untuk "
        "aplikasi baru.",
    ),
]

WHY_MISSING = [
    (
        "Kenapa fitur kontinu diisi median, bukan mean?",
        "Income, loan amount, dan property value distribusinya menjulur jauh ke kanan. Ada property "
        "value sampai $130 juta, dan angka sebesar itu menarik mean ke atas sehingga nilai isian "
        "jadi terlalu tinggi untuk pemohon kebanyakan. Median tidak terpengaruh ekor ekstrem.",
    ),
    (
        "Kenapa tidak pakai KNN imputation saja?",
        "Tiga alasan. Mencari tetangga terdekat di 100.000 baris jauh lebih mahal daripada hitung "
        "satu median. Untuk data regulasi, alasan diisi median kolom jauh lebih mudah "
        "dipertanggungjawabkan daripada angka keluaran model. Dan yang paling penting, imputasi "
        "berbasis fitur lain menanamkan korelasi buatan antar fitur, yang nanti bisa muncul lagi "
        "sebagai association rule di Fase 3. Jadi kita menemukan pola yang kita buat sendiri.",
    ),
    (
        "Kenapa kategorikal diisi Unknown, bukan modus?",
        "Karena kosong di sini ada artinya. Sering menandakan Exempt atau memang tidak dilaporkan. "
        "Kalau diisi modus, sinyal itu hilang dan kategori terbanyak jadi menggelembung tanpa alasan. "
        "Unknown mempertahankannya sebagai kategori sendiri yang masih bisa dianalisis.",
    ),
    (
        "Kenapa perlu penanda _was_missing?",
        "Supaya jejak imputasi tetap terekam. Tanpa penanda, nilai hasil isian tidak bisa dibedakan "
        "dari nilai asli. Penanda ini sengaja tidak ikut ke Fase 2 sampai 4 karena sifatnya "
        "diagnostik proses, bukan karakteristik pemohon.",
    ),
]

WHY_FEATURE_SELECTION = [
    (
        "Kenapa pakai korelasi dan mutual information sekaligus?",
        "Korelasi cuma menangkap hubungan linear, jadi fitur yang pengaruhnya melengkung akan "
        "terlihat lemah padahal penting. Mutual information menangkap hubungan non-linear tapi tidak "
        "memberi arah positif atau negatif. Dipakai berdua supaya saling menutup kelemahan.",
    ),
    (
        "Kenapa fitur pasca-keputusan dibuang?",
        "Ini soal kebocoran data. Interest rate, total loan costs, dan origination charges baru ada "
        "setelah aplikasi disetujui. Kalau dipakai untuk menjelaskan persetujuan, hasilnya akan "
        "kelihatan sangat akurat tapi tidak berguna, karena saat aplikasi baru masuk field itu belum "
        "ada isinya.",
    ),
]

WHY_SCALING = [
    (
        "Kenapa winsorize 1% dan 99% sebelum clustering?",
        "K-Means bekerja dengan jarak kuadrat, jadi satu properti $130 juta bisa menarik seluruh "
        "centroid dan bikin clusternya tidak masuk akal. Winsorize memotong ekornya tanpa membuang "
        "barisnya.",
    ),
    (
        "Kenapa winsorize cuma untuk clustering?",
        "Karena Fase 4 justru mencari nilai ekstrem. Kalau ekornya sudah dipotong duluan, deteksi "
        "anomali kehilangan target utamanya. Jadi winsorize cuma dipakai di salinan untuk "
        "clustering, sementara nilai asli tetap utuh untuk profiling, rules, dan anomali.",
    ),
    (
        "Kenapa clustering pakai StandardScaler tapi anomali pakai RobustScaler?",
        "Karena tujuannya bertolak belakang. Clustering pakai jarak Euclidean, jadi semua fitur harus "
        "setara. Tanpa penskalaan, loan amount yang ratusan ribu akan menenggelamkan CLTV yang cuma "
        "persen, dan jaraknya praktis cuma mengukur loan amount. Sementara deteksi anomali justru "
        "mencari outlier, jadi penskalaannya tidak boleh ikut terpengaruh outlier. Mean dan standar "
        "deviasi tertarik nilai ekstrem, akibatnya outlier menggelembungkan std lalu z-score-nya "
        "sendiri malah mengecil. Median dan IQR di RobustScaler tidak punya masalah itu.",
    ),
]

WHY_METHODS = [
    (
        "Kenapa K-Means yang jadi metode utama?",
        "Karena cuma K-Means yang sanggup jalan di seluruh 99.995 aplikasi. Biayanya linear terhadap "
        "jumlah baris, sedangkan hierarchical butuh matriks jarak antar semua pasangan baris. Selain "
        "itu K-Means juga menang di semua metrik saat diuji pada sampel yang sama.",
    ),
    (
        "Kenapa DBSCAN tetap dijalankan?",
        "Untuk dua hal yang tidak bisa dilakukan K-Means. K-Means memaksa semua titik masuk cluster "
        "dan mengasumsikan bentuknya membulat, sedangkan DBSCAN bekerja dari kepadatan sehingga "
        "bentuk clusternya bebas. DBSCAN juga menandai noise, dan 895 titik noise itu dipakai sebagai "
        "salah satu dari lima detektor anomali di Fase 4.",
    ),
    (
        "Kenapa hierarchical Ward?",
        "Untuk mengecek apakah struktur 7 segmen itu memang ada di datanya atau cuma hasil bentukan "
        "K-Means. Ward tidak butuh K ditentukan di awal, dia membangun hierarki penuh yang bisa "
        "dipotong di K berapa pun. ARI 0,906 terhadap K-Means menunjukkan keduanya sangat sepakat.",
    ),
    (
        "Kenapa CLARANS?",
        "Karena pusat clusternya berupa aplikasi nyata, bukan rata-rata buatan. Centroid K-Means bisa "
        "jatuh di titik yang tidak pernah ada di data, sedangkan medoid CLARANS bisa langsung "
        "ditunjuk sebagai contoh konkret dari sebuah segmen.",
    ),
    (
        "Kenapa DBSCAN dan hierarchical pakai sampel?",
        "Ini keputusan teknis, bukan karena algoritmanya menyampel sendiri. Hierarchical memang wajib "
        "disampel karena matriks linkage-nya butuh memori sebanding kuadrat jumlah baris, dan di "
        "100.000 baris itu sekitar 10 miliar pasangan, pasti kehabisan memori. DBSCAN sebenarnya bisa "
        "lebih besar, tapi makin lambat dan penyetelan eps makin sensitif. Sampel 20.000 sudah cukup "
        "untuk melihat struktur kepadatannya.",
    ),
]

WHY_BEST_METHOD = [
    (
        "Kenapa K-Means disebut paling baik?",
        "Karena pada sampel 4.000 baris yang sama, K-Means unggul di ketiga metrik sekaligus. "
        "Silhouette tertinggi 0,303, Davies-Bouldin terendah 1,137, dan Calinski-Harabasz tertinggi "
        "869. Jadi keunggulannya bukan cuma di satu metrik yang kebetulan menguntungkan.",
    ),
    (
        "Kenapa K-nya 7, dasarnya apa?",
        "Silhouette, bukan elbow. Elbow cuma memberi kandidat dan pembacaannya subjektif, dua orang "
        "bisa melihat kurva yang sama lalu memilih K berbeda. Silhouette mengukur langsung seberapa "
        "rapat sebuah titik ke clusternya dibanding cluster tetangga, jadi hasilnya satu angka yang "
        "bisa dibandingkan antar K. Elbow tetap dihitung sebagai pembanding, tapi keputusannya di "
        "silhouette.",
    ),
    (
        "Kenapa angka DBSCAN tidak bisa dibandingkan langsung?",
        "Karena DBSCAN diukur di sampel 20.000 dan titik noise-nya dikeluarkan dari perhitungan. "
        "Davies-Bouldin-nya terlihat paling rendah sebagian justru karena 895 titik tersulit tidak "
        "ikut dihitung. Itu sebabnya baris K-Means pada sampel 4.000 disertakan, supaya ada "
        "pembanding yang setara.",
    ),
    (
        "Kenapa ARI CLARANS cuma 0,710?",
        "Itu wajar, bukan tanda gagal. CLARANS mengoptimalkan medoid sedangkan K-Means mengoptimalkan "
        "centroid, jadi tujuannya memang beda dan hasilnya tidak harus sama persis. Yang penting "
        "keduanya tetap menemukan jumlah segmen yang sama.",
    ),
]

WHY_RULE_THRESHOLD = [
    (
        "Kenapa minimum support 2%?",
        "Di 67.827 aplikasi berkeputusan, 2% itu sekitar 1.357 aplikasi. Cukup besar supaya "
        "confidence-nya stabil dan tidak lahir dari segelintir kasus, tapi masih cukup kecil supaya "
        "segmen minoritas seperti manufactured housing tidak ikut tersapu sebelum sempat dianalisis.",
    ),
    (
        "Kenapa lift harus di atas 1,2?",
        "Lift 1,0 artinya antecedent dan hasilnya saling bebas, jadi aturannya tidak memberi "
        "informasi apa pun. Ambang 1,2 menuntut kejadian bersamanya minimal 20% lebih sering "
        "daripada kebetulan. Tanpa ini, aturan semacam First_Lien maka Originated akan lolos cuma "
        "karena mayoritas aplikasi memang first lien dan mayoritas memang disetujui.",
    ),
    (
        "Kenapa confidence minimal 55%?",
        "Karena aturannya harus mengalahkan tebakan dasar. Base rate-nya 23,1% Denied dan 76,9% "
        "Originated. Untuk aturan penolakan, 55% itu lebih dari dua kali base rate-nya, sekaligus "
        "melewati batas mayoritas. Jadi kalau aturannya bilang Denied, lebih dari separuh kasus yang "
        "cocok memang benar ditolak.",
    ),
    (
        "Kenapa itemset dibatasi maksimal 3?",
        "Supaya masih bisa dipakai dan tidak meledak jumlahnya. Aturan dengan 5 sampai 6 syarat "
        "praktis tidak bisa dijalankan tim underwriting, dan jumlah kombinasinya tumbuh sangat cepat "
        "seiring panjang itemset. Tiga syarat sudah cukup menangkap interaksi seperti DTI ditambah "
        "jenis lien dan jenis loan.",
    ),
]

WHY_RULE_PRUNING = [
    (
        "Kenapa dari 28 aturan cuma 11 yang dipakai?",
        "Karena lolos ambang belum tentu menambah pengetahuan. Menempelkan derived_race=White atau "
        "First_Lien ke aturan DTI di atas 60% menghasilkan aturan yang tetap lolos semua ambang, "
        "padahal confidence-nya nyaris sama dengan aturan induknya. Itu cuma pengulangan dengan kata "
        "tambahan, dan berbahaya karena bisa dibaca seolah ras jadi faktor penolakan padahal tidak "
        "menambah daya pisah sama sekali.",
    ),
    (
        "Bagaimana cara memangkasnya?",
        "Improvement filter. Sebuah aturan cuma dipertahankan kalau confidence-nya mengungguli semua "
        "sub-rule-nya minimal 2 poin persen. Hasilnya 28 kandidat menyusut jadi 11 aturan yang "
        "benar-benar berbeda satu sama lain.",
    ),
    (
        "Kenapa aturannya masih diuji chi-square dan Wilson CI?",
        "Support dan confidence tidak memberi tahu apakah polanya bisa muncul karena kebetulan. "
        "Chi-square menguji apakah asosiasinya beda nyata dari kondisi saling bebas, sedangkan Wilson "
        "confidence interval memberi rentang ketidakpastian yang tetap masuk akal saat n-nya kecil "
        "atau proporsinya mendekati 0 dan 1.",
    ),
]

WHY_DETECTORS = [
    (
        "Kenapa pakai lima detektor sekaligus?",
        "Karena tiap metode punya titik buta, dan ada dua cara pandang yang menangkap hal berbeda. "
        "IQR, Z-score, dan Isolation Forest melihat secara global, mencari nilai yang menyimpang dari "
        "distribusi seluruh data. LOF dan DBSCAN-noise melihat secara kontekstual, mencari baris yang "
        "wajar di tiap fitur tapi aneh dibanding tetangganya. Angka taksonomi membuktikan keduanya "
        "memang beda: cuma 476 baris yang tertangkap dua-duanya, sementara 9.959 cuma global dan 589 "
        "cuma kontekstual.",
    ),
    (
        "Kenapa ambangnya 3 dari 5 suara?",
        "Untuk menekan false positive. Satu metode gampang menandai baris yang sebenarnya wajar, tapi "
        "lima detektor dengan prinsip berbeda jauh lebih sulit sepakat secara kebetulan. Ambang ini "
        "juga memaksa minimal satu metode global dan satu kontekstual ikut setuju di sebagian besar "
        "kasus.",
    ),
    (
        "Kenapa contamination-nya 1%?",
        "Ini asumsi kerja soal berapa banyak baris yang masuk akal ditinjau manual, bukan klaim bahwa "
        "persis 1% datanya salah. Angkanya dipilih supaya antrean tinjauan masih realistis buat tim "
        "manusia. Parameter ini menentukan ambang skor, bukan benar atau salahnya sebuah baris, dan "
        "itu sebabnya keputusan akhirnya tetap lewat triase.",
    ),
]

WHY_TAXONOMY = [
    (
        "Kenapa taksonomi ini cuma memuat dua kelas, bukan tiga?",
        "Karena kelima detektor semuanya menilai baris satu per satu, jadi secara desain hanya bisa "
        "menghasilkan outlier global dan kontekstual. Outlier kolektif menuntut unit analisis yang "
        "berbeda, yaitu kelompok, sehingga ditangani terpisah pada panel anomali tingkat grup di "
        "bawah.",
    ),
    (
        "Kenapa outlier global jauh lebih banyak dari kontekstual?",
        "Karena metode global menandai baris yang ekstrem di fitur mana pun, dan dengan data sebesar "
        "ini selalu ada ekor distribusi yang panjang. Metode kontekstual jauh lebih selektif karena "
        "menuntut baris itu aneh relatif terhadap tetangganya, bukan cuma besar nilainya.",
    ),
]

WHY_TRIAGE = [
    (
        "Kenapa 15 baris teratas ditinjau manual, bukan langsung dibuang?",
        "Karena ekstrem tidak sama dengan salah. Jumbo loan $9 juta itu ekstrem tapi sah, sedangkan "
        "CLTV 900% mustahil secara hitungan. Membedakannya butuh cek konsistensi internal, misalnya "
        "apakah CLTV dikali property value masih cocok dengan loan ini ditambah senior lien, dan "
        "apakah income-nya masuk akal untuk membayar utang segitu.",
    ),
    (
        "Apa hasilnya, dan kenapa itu penting?",
        "Semua 15 baris teratas berakhir dengan verdict RARE BUT VALID, tidak satu pun kesalahan "
        "data. Artinya kalau ensemble ini dipakai sebagai aturan hapus otomatis, 15 aplikasi yang sah "
        "akan ikut terbuang.",
    ),
]

# ============================================================ Fase 1 (preprocessing)
_FATE_LABEL = {
    "dropped": "Dibuang (>60% kosong)",
    "median_imputed": "Imputasi median",
    "filled_unknown": "Diisi 'Unknown'",
}
_FATE_COLOR = {"dropped": RED, "median_imputed": STEEL, "filled_unknown": AMBER}
_ROLE_LABEL = {
    "Strong discovery candidate": "Kandidat kuat",
    "Moderate discovery candidate": "Kandidat sedang",
    "Weak / supporting": "Lemah / pendukung",
}


def _stat_tile(label, value, color=NAVY):
    return html.Div(
        [
            html.Div(
                value, style={"fontSize": "24px", "fontWeight": "800", "color": color}
            ),
            html.Div(
                label, style={"fontSize": "11px", "color": MUTE, "marginTop": "2px"}
            ),
        ],
        className="hmda-card",
        style={
            "flex": "1",
            "minWidth": "150px",
            "background": CARD,
            "borderRadius": "14px",
            "padding": "14px 16px",
            "boxShadow": SOFT_SHADOW,
            "border": f"1px solid {BORDER}",
        },
    )


# Load-test results, measured 26 Juli 2026 against app/index.py on the Flask development
# server (Python 3.14, concurrency via thread pool). These are recorded
# measurements from one run, not values recomputed by the pipeline, so they are stated
# with their environment rather than presented as live metrics.
LOADTEST_COLD = [
    ("Fase 1 - Preprocessing", 229, 51),
    ("Fase 2 - Clustering", 8, 15),
    ("Fase 3 - Association Rules", 25, 13),
    ("Fase 4 - Anomaly Detection", 96, 279),
    ("Fase 5 - Reporting & Fairness", 111, 53),
]
LOADTEST_CONC = [
    (1, 56.0, 18, 17, 32, 36),
    (5, 228.3, 21, 22, 35, 37),
    (10, 277.2, 33, 33, 41, 46),
    (25, 298.0, 72, 80, 87, 90),
]

WHY_LOADTEST = [
    (
        "Mengapa yang diuji callback tab, bukan hanya halaman utama?",
        "Karena halaman utama hanya mengirim kerangka HTML statis. Kerja sebenarnya terjadi di "
        "endpoint /_dash-update-component, tempat callback membangun seluruh grafik dan tabel "
        "sebuah tab. Menguji halaman statis saja akan menghasilkan angka yang terlihat bagus "
        "tetapi tidak mewakili beban nyata.",
    ),
    (
        "Mengapa dibedakan cold dan warm?",
        "Karena fungsi render memakai cache (lru_cache maxsize 8) sementara aplikasi hanya punya "
        "5 tab, sehingga tidak pernah ada eviction: tiap tab hanya mahal sekali per proses. "
        "Melaporkan angka warm saja akan menyembunyikan biaya render pertama, dan itu justru "
        "biaya yang dibayar pengguna pertama setiap kali proses baru dimulai.",
    ),
    (
        "Kenapa menaikkan concurrency tidak sepadan hasilnya?",
        "Karena ada bagian kerja yang terserialisasi: GIL Python ditambah development server. "
        "Dari concurrency 5 ke 25, jadi lima kali lipat, throughput cuma naik 1,3 kali dari "
        "228 ke 298 req/s, sementara latency naik 3,6 kali dari 22 ke 80 ms. Jadi sebagian "
        "besar permintaan tambahan berakhir mengantre, bukan diproses paralel.",
    ),
    (
        "Kenapa angka ini bukan angka produksi?",
        "Tiga sebab. Pertama, pengujian memakai Flask development server yang secara eksplisit "
        "memperingatkan dirinya bukan untuk produksi, sedangkan Vercel memakai runtime WSGI "
        "sendiri. Kedua, pengujian dijalankan tanpa latency jaringan, padahal payload 279 KB "
        "milik Fase 4 akan jauh lebih terasa pada koneksi nyata. Ketiga, cuma satu instance "
        "penguji tanpa variasi geografis. Angka ini valid sebagai profil relatif antar tab, "
        "bukan sebagai kapasitas produksi.",
    ),
    (
        "Apa implikasinya untuk deployment Vercel?",
        "Cold start 2,9 detik dan memori 190 MB dibayar ulang setiap kali instance serverless "
        "baru dimulai, dan penyebab utamanya sama: hmda_approve_deny.csv berukuran 27 MB dibaca "
        "saat import. Merampingkan file itu ke kolom yang benar-benar dipakai What-If akan "
        "memangkas cold start, pemakaian memori, dan ukuran bundle sekaligus.",
    ),
]


WHY_FUNNEL = [
    (
        "Kenapa 30 kolom demografi mentah tidak dipakai?",
        "HMDA memecah ras dan etnis jadi lima kolom terpisah untuk pemohon dan lima lagi untuk "
        "co-applicant, plus kolom penanda cara pengamatannya. Semua itu sudah diringkas HMDA "
        "sendiri jadi derived_race, derived_ethnicity, dan derived_sex, yang justru dipakai di "
        "tab Keadilan. Memakai versi mentahnya cuma menduplikasi informasi yang sama dalam "
        "puluhan kolom.",
    ),
    (
        "Kenapa hasil AUS dan alasan penolakan dibuang?",
        "Keduanya baru ada setelah keputusan dibuat. AUS itu keluaran sistem underwriting "
        "otomatis, dan denial_reason cuma terisi pada aplikasi yang memang ditolak. Kalau "
        "denial_reason ikut dipakai untuk memprediksi penolakan, akurasinya bisa mendekati "
        "sempurna tapi sama sekali tidak berguna, karena kolomnya kosong saat aplikasi baru masuk.",
    ),
    (
        "Kenapa harga dan biaya dianggap leakage?",
        "Interest rate, total loan costs, origination charges, dan rate spread baru ditentukan "
        "ketika loan disetujui. Nilainya justru hasil dari keputusan yang sedang kita pelajari, "
        "bukan penyebabnya. Ikut memasukkannya berarti menjelaskan keputusan dengan keputusan itu "
        "sendiri.",
    ),
    (
        "Kenapa jumlahnya tidak bisa dikurangi berurutan?",
        "Karena ada kolom yang masuk dua kategori sekaligus. total_points_and_fees, "
        "discount_points, dan lender_credits itu leakage sekaligus lebih dari 60% kosong, jadi "
        "kalau angkanya dikurangi berantai, ketiganya terhitung dua kali. Itu sebabnya tabel di "
        "atas menyajikan alasan per kategori, bukan sebagai pengurangan bertahap.",
    ),
    (
        "Setelah disaring, bagaimana sisanya dinilai?",
        "31 fitur yang lolos diberi skor gabungan dari korelasi dan mutual information terhadap "
        "Originated vs Denied, lalu dikelompokkan jadi kandidat kuat, sedang, dan lemah. Yang "
        "lemah tidak langsung dibuang, tetap dilaporkan supaya keputusannya bisa ditelusuri.",
    ),
]


WHY_VIF = [
    (
        "Kenapa perlu VIF kalau sudah ada cek korelasi berpasangan?",
        "Karena korelasi berpasangan cuma melihat dua fitur sekaligus, jadi buta terhadap "
        "redundansi yang baru muncul saat beberapa fitur digabung. Sebuah fitur bisa punya "
        "korelasi rendah terhadap setiap fitur lain satu per satu, tapi tetap bisa diprediksi "
        "hampir sempurna dari kombinasi tiga fitur lainnya. VIF menangkap kasus itu.",
    ),
    (
        "Kenapa ambangnya 10?",
        "Karena VIF = 1 / (1 - R kuadrat), jadi VIF di atas 10 persis sama artinya dengan R "
        "kuadrat di atas 0,90. Itu ambang yang sama dengan cek berpasangan, cuma diterapkan di "
        "dimensi yang benar. Hasilnya konsisten: nol fitur melewati ambang, sejalan dengan nol "
        "pasangan pada cek berpasangan.",
    ),
    (
        "Kenapa DTI diuji terpisah?",
        "Karena DTI sudah diubah jadi band, sehingga tidak ikut masuk ruang fitur numerik dan "
        "tidak muncul di tabel VIF. Padahal justru DTI yang paling dicurigai tumpang tindih "
        "dengan income dan besar loan. Uji terarah ini memetakan band DTI ke titik tengahnya, "
        "lalu mencoba merekonstruksinya dari income, besar loan, nilai properti, dan rasio "
        "loan terhadap income.",
    ),
    (
        "Apa hasilnya, dan apa konsekuensinya?",
        "R kuadrat cuma 0,100, setara VIF 1,11. Artinya 90% variasi DTI tidak bisa dijelaskan "
        "oleh fitur ukuran, jadi DTI membawa informasi yang benar-benar berbeda dan keduanya "
        "layak dipertahankan. Sisa variasi itu masuk akal secara struktural: utang non-hipotek "
        "tidak terekam di HMDA, besar cicilan bergantung pada bunga dan tenor, dan DTI cuma "
        "dilaporkan sebagai tujuh tingkat ordinal.",
    ),
]


def _vif_panel():
    """Multicollinearity audit added in the notebook revision: VIF plus a targeted
    test of whether DTI is just a restatement of the size features."""
    vif_rows = [
        ("tract_owner_occupied_units", 6.00),
        ("tract_one_to_four_family_homes", 4.05),
        ("tract_population", 3.58),
        ("any_exempt_field", 2.95),
        ("property_value_was_missing", 2.61),
        ("loan_term_was_missing", 2.29),
        ("combined_loan_to_value_ratio_was_missing", 2.00),
        ("tract_minority_population_percent", 1.51),
        ("loan_amount", 1.30),
        ("property_value", 1.27),
    ]
    vif_df = pd.DataFrame([{"Fitur": f, "VIF": v} for f, v in vif_rows])
    dti_df = pd.DataFrame(
        [
            {"Prediktor": "loan_to_income", "Spearman terhadap DTI": "+0,373"},
            {"Prediktor": "log_income", "Spearman terhadap DTI": "-0,330"},
            {"Prediktor": "log_loan_amount", "Spearman terhadap DTI": "+0,057"},
            {"Prediktor": "log_property_value", "Spearman terhadap DTI": "+0,007"},
        ]
    )
    return panel(
        "Audit multikolinearitas: VIF dan uji rekonstruksi DTI",
        [
            html.Div(
                [
                    _stat_tile("Fitur numerik diuji", "17"),
                    _stat_tile("VIF tertinggi", "6,00", STEEL),
                    _stat_tile("Di atas ambang 10", "0", GREEN),
                    _stat_tile("R² rekonstruksi DTI", "0,100", GREEN),
                ],
                style={
                    "display": "flex",
                    "gap": "14px",
                    "flexWrap": "wrap",
                    "marginBottom": "16px",
                },
            ),
            html.H4(
                "VIF per fitur (10 tertinggi)",
                style={"fontSize": "13px", "color": NAVY, "margin": "6px 0 8px"},
            ),
            _table(vif_df),
            html.H4(
                "Bisakah DTI direkonstruksi dari fitur ukuran?",
                style={"fontSize": "13px", "color": NAVY, "margin": "16px 0 8px"},
            ),
            _table(dti_df),
            html.Div(
                [
                    html.B(
                        "Kesimpulan: DTI tidak bisa direkonstruksi. ",
                        style={"fontSize": "12px", "color": NAVY},
                    ),
                    html.Span(
                        "R² = 0,100 setara VIF 1,11, jadi 90% variasi DTI tidak dijelaskan "
                        "oleh income, besar loan, maupun nilai properti. DTI dan fitur ukuran "
                        "membawa informasi berbeda dan sama-sama dipertahankan.",
                        style={"fontSize": "12px", "color": INK, "lineHeight": "1.6"},
                    ),
                ],
                style={
                    "background": "#eefaf1",
                    "border": "1px solid #b7e4c7",
                    "borderRadius": "10px",
                    "padding": "12px 14px",
                    "margin": "14px 0",
                },
            ),
            why(WHY_VIF),
        ],
        sub="Dihitung pada 67.827 aplikasi berkeputusan; uji DTI memakai 62.618 baris "
        "setelah 5.209 Exempt/Unknown dibuang.",
    )


WHY_COLLECTIVE_GROUPS = [
    ("Kenapa perlu deteksi tingkat grup?",
     "Karena kelima detektor sebelumnya menilai baris satu per satu, jadi secara desain buta "
     "terhadap pola yang cuma ada di tingkat kelompok. Sebuah kelompok bisa punya tanda-tanda "
     "aneh sebagai kesatuan padahal tiap anggotanya wajar kalau dilihat sendirian."),
    ("Bagaimana cara kerjanya?",
     "Baris dikelompokkan menurut kombinasi bisnis, lalu tiap kelompok diringkas jadi median "
     "dan IQR dari fitur anomali. Isolation Forest dijalankan pada profil kelompok itu, bukan "
     "pada barisnya. Kelompok dengan anggota kurang dari 50 dibuang supaya ringkasannya tidak "
     "lahir dari segelintir baris."),
    ("Kenapa sebagian besar hasilnya tidak dianggap temuan?",
     "Karena dari 29 kelompok berbasis negara bagian yang ditandai, 15 di antaranya atau 52% "
     "adalah yurisdiksi kecil seperti Hawaii, DC, dan Puerto Rico, atau kategori state yang "
     "tidak diketahui. Kelompok kecil memang gampang terlihat menyimpang secara statistik, dan "
     "itu variasi geografis biasa, bukan anomali yang perlu ditindaklanjuti. Menyebutnya temuan "
     "akan mengada-ada."),
    ("Apa itu pure collective candidate?",
     "Kelompok yang ditandai anomali padahal kurang dari 25% anggotanya pernah ditandai detektor "
     "individual. Inilah kasus yang paling berarti, karena keanehannya benar-benar muncul di "
     "tingkat kelompok dan tidak akan tertangkap deteksi per baris."),
    ("Kenapa field hasil keputusan tidak dipakai?",
     "Supaya ini tetap penemuan struktural, bukan model yang memprediksi persetujuan. Kalau "
     "action_taken atau alasan penolakan ikut masuk, yang ditemukan cuma cerminan keputusan yang "
     "sudah diambil."),
]


KEY_TAKEAWAYS = [
    ("Beban utang paling menentukan",
     "Aplikasi dengan DTI di atas 60% ditolak pada 91,5% kasus historis, jauh di atas rata-rata "
     "portfolio 23,1%.", "91,5%", RED),
    ("Penalti manufactured housing",
     "Segmen manufactured housing hanya disetujui 43,1%, dan penaltinya melekat pada jenis "
     "properti, bukan pada income pemohon.", "43,1%", AMBER),
    ("Alasan penolakan belum terurai",
     "Hampir tiga perempat penolakan tercatat sebagai Other, sementara DTI yang terbukti paling "
     "kuat hanya tercatat 8,7%.", "72,9%", STEEL),
    ("Selisih bertahan di kelompok aman",
     "Selisih persetujuan antar lingkungan justru terlebar pada kelompok berisiko rendah, bukan "
     "pada yang berisiko tinggi.", "12,1 poin", NAVY),
]

RECOMMENDED_ACTIONS = [
    ("Saring DTI di tahap intake",
     "Periksa DTI sebelum underwriting penuh dan tawarkan rujukan alternatif untuk pemohon di "
     "atas 60%."),
    ("Sediakan produk manufactured housing",
     "Bangun jalur chattel lending atau FHA Title I agar segmen ini tidak dipaksa lewat "
     "underwriting konvensional."),
    ("Audit fair lending di segmen DTI rendah",
     "Telaah selisih persetujuan memakai data underwriting lengkap yang tidak tersedia di HMDA "
     "publik."),
]


WHY_CLUSTER_FEATS = [
    ("Kenapa keempat metode memakai fitur yang sama persis?",
     "Supaya perbandingannya adil. Kalau tiap algoritma diberi fitur berbeda, selisih hasilnya "
     "bisa jadi karena fiturnya, bukan karena algoritmanya. Dengan matriks yang identik, "
     "perbedaan ARI dan silhouette murni berasal dari cara kerja masing-masing metode."),
    ("Kenapa loan_amount dan property_value tidak ikut clustering?",
     "Karena keduanya sudah terwakili lewat CLTV, yang merupakan rasio pinjaman terhadap nilai "
     "properti. Memasukkan ketiganya membuat dimensi ukuran terhitung berkali-kali dan "
     "mendominasi jarak. Keduanya tetap dipakai penuh di deteksi anomali, karena di sana "
     "magnitudo mentah justru yang dicari."),
    ("Kenapa ada fitur biner di antara fitur kontinu?",
     "Karena karakteristik seperti investasi, refinance, manufactured, subordinate lien, dan "
     "DTI tinggi itu penentu segmen yang kuat tetapi sifatnya ya atau tidak. Setelah "
     "StandardScaler, keduanya berada pada skala yang sebanding sehingga tidak ada yang "
     "mendominasi jarak."),
    ("Kenapa clustering tidak memakai RobustScaler?",
     "Karena tujuannya berbeda. Clustering butuh semua fitur setara, dan ekor ekstremnya sudah "
     "ditangani lebih dulu lewat winsorize 1% dan 99%. Deteksi anomali justru mencari ekor itu, "
     "jadi di sana dipakai RobustScaler yang median dan IQR-nya tidak terseret outlier."),
]


WHY_PCA = [
    ("Berapa komponen PCA yang dipakai?",
     "Dua, yaitu PC1 dan PC2. Ruang fiturnya punya 9 dimensi, jadi tersedia 9 komponen, "
     "tetapi yang digambar cuma dua."),
    ("Kenapa dua, bukan jumlah yang menangkap 90% variansi?",
     "Karena PCA di sini semata alat gambar, bukan bagian dari clustering. Layar cuma punya "
     "dua sumbu, jadi dua komponen adalah batas yang bisa diplot. Kalau tujuannya menangkap "
     "variansi, jumlahnya akan jauh berbeda: butuh 7 komponen untuk mencapai 86,2% dan 8 "
     "komponen untuk 93,2%."),
    ("Berapa variansi yang tertangkap dua komponen itu?",
     "PC1 menjelaskan 16,7% dan PC2 15,2%, jadi totalnya cuma 31,9%. Sisanya 68,1% tidak "
     "terlihat di gambar. Ini karena kesembilan fitur relatif tidak saling berkorelasi, "
     "sejalan dengan audit VIF di Fase 1 yang menemukan nol fitur melewati ambang."),
    ("Apa konsekuensinya untuk membaca scatter plot?",
     "Dua cluster yang tampak bertumpuk di gambar belum tentu benar-benar bertumpuk, karena "
     "bisa saja terpisah pada dimensi yang tidak digambar. Karena itu penilaian kualitas "
     "cluster memakai silhouette, Davies-Bouldin, dan Calinski-Harabasz yang dihitung pada "
     "seluruh 9 dimensi, bukan pada proyeksi 2 dimensi ini."),
    ("Apakah clustering dijalankan pada hasil PCA?",
     "Tidak. K-Means, DBSCAN, Ward, dan CLARANS semuanya bekerja pada matriks 9 fitur yang "
     "penuh. PCA baru diterapkan sesudahnya, hanya untuk menempatkan titik-titik di gambar."),
]


CLUSTER_SCOPE = {
    "shared": ("Keempat metode", "cakupan masing-masing, lihat tabel perbandingan"),
    "kmeans": ("K-Means", "seluruh 99.995 baris"),
    "dbscan": ("DBSCAN", "sampel 20.000 baris"),
    "hierarchical": ("Hierarchical (Ward)", "sampel 4.000 baris"),
    "clarans": ("CLARANS", "sampel 4.000 baris"),
}


def _cluster_feats_details(method):
    """Collapsible feature list for one clustering method.

    All four methods share one matrix on purpose, so any difference in their results
    comes from the algorithm rather than from the inputs.
    """
    name, scope = CLUSTER_SCOPE.get(method, ("Metode ini", "sampel"))
    cont = ["income", "combined_loan_to_value_ratio",
            "tract_minority_population_percent", "tract_to_msa_income_percentage"]
    flags = ["_is_investment", "_is_refinance", "_is_manufactured",
             "_is_subordinate", "_is_high_dti"]
    df = pd.DataFrame(
        [{"Fitur": c, "Tipe": "kontinu",
          "Perlakuan": "winsorize 1/99% lalu StandardScaler"} for c in cont]
        + [{"Fitur": f, "Tipe": "biner", "Perlakuan": "StandardScaler"} for f in flags]
    )
    return html.Details(
        [
            html.Summary(
                "Fitur yang dipakai dan cara penskalaannya (klik)",
                style={"cursor": "pointer", "fontWeight": "700", "fontSize": "12.5px",
                       "color": STEEL, "padding": "4px 0", "userSelect": "none"},
            ),
            html.Div(
                [
                    html.P(
                        "Kesembilan fitur berikut dipakai keempat metode clustering, dengan "
                        "perlakuan penskalaan yang sama. Yang berbeda cuma cakupan barisnya, dan "
                        "itu ada di tabel perbandingan.",
                        style={"fontSize": "12px", "color": INK, "margin": "0 0 10px",
                               "lineHeight": "1.6"},
                    ),
                    _table(df),
                    html.P(
                        "Penskalaan memakai StandardScaler, bukan RobustScaler. Clustering butuh "
                        "semua fitur setara pada jarak Euclidean, dan ekor ekstremnya sudah "
                        "ditangani lebih dulu lewat winsorize. RobustScaler dipakai di Fase 4, "
                        "karena di sana outlier justru yang dicari sehingga penskalaannya tidak "
                        "boleh ikut terseret nilai ekstrem.",
                        style={"fontSize": "11.5px", "color": MUTE, "margin": "10px 0 0",
                               "lineHeight": "1.6"},
                    ),
                    html.P(
                        "loan_amount dan property_value sengaja tidak ikut, karena keduanya sudah "
                        "terwakili lewat CLTV. Memasukkan ketiganya membuat dimensi ukuran "
                        "terhitung berulang dan mendominasi jarak. Keduanya tetap dipakai penuh "
                        "di matriks anomali Fase 4.",
                        style={"fontSize": "11.5px", "color": MUTE, "margin": "8px 0 0",
                               "lineHeight": "1.6"},
                    ),
                ],
                style={"marginTop": "12px"},
            ),
        ],
        style={"background": "#f8fafd", "border": f"1px solid {BORDER}",
               "borderRadius": "10px", "padding": "12px 16px", "marginBottom": "16px"},
    )


WHY_RULE_FEATS = [
    ("Kenapa fiturnya kategorikal semua, tidak ada yang kontinu?",
     "Karena Apriori bekerja pada item yang ada atau tidak ada, bukan pada angka. Fitur "
     "kontinu seperti income dan besar loan lebih dulu diubah jadi band, sehingga "
     "income=<30k bisa diperlakukan sebagai satu item."),
    ("Kenapa fiturnya jauh lebih banyak dari clustering?",
     "Karena tujuannya beda. Clustering butuh sedikit dimensi supaya jaraknya bermakna, "
     "sedangkan association rule justru mencari kombinasi kondisi, jadi makin banyak kandidat "
     "item makin kaya polanya. Ledakan kombinasinya ditahan oleh minimum support 2% dan "
     "batas panjang itemset maksimal 3."),
    ("Kenapa fitur demografis ikut di sini padahal tidak dipakai di What-If?",
     "Supaya bisa diuji, bukan supaya dipakai menilai. Dengan memasukkan ras, etnis, dan "
     "gender sebagai kandidat item, kita bisa melihat apakah menambahkannya ke aturan DTI "
     "meningkatkan daya pisah. Hasilnya tidak, karena semua varian itu gugur di improvement "
     "filter. Justru ketidakhadirannya di aturan final yang jadi temuan."),
    ("Kenapa tidak ada penskalaan di sini?",
     "Karena tidak ada jarak yang dihitung. Apriori cuma menghitung seberapa sering item "
     "muncul bersama, jadi StandardScaler maupun RobustScaler tidak relevan. Yang berperan "
     "sebagai gantinya adalah binning."),
]


def _rule_features_details():
    """Feature list behind the transaction matrix used for Apriori."""
    groups = [
        ("Profil pemohon", ["derived_race", "derived_ethnicity", "derived_sex",
                            "applicant_age", "income_band"]),
        ("Karakteristik loan", ["loan_type", "loan_purpose", "lien_status", "preapproval",
                                "conforming_loan_limit", "loan_amount_band"]),
        ("Properti & agunan", ["occupancy_type", "construction_method", "total_units",
                               "property_value_band", "cltv_band"]),
        ("Beban utang", ["debt_to_income_ratio"]),
        ("Konteks lingkungan", ["tract_income_cat", "tract_minority_cat"]),
    ]
    df = pd.DataFrame(
        [{"Kelompok": g, "Fitur": f,
          "Perlakuan": ("sudah band, dipakai apa adanya"
                        if f.endswith(("_band", "_cat")) or f == "debt_to_income_ratio"
                        else "kategorikal, dipakai apa adanya")}
         for g, feats in groups for f in feats]
    )
    return html.Details(
        [
            html.Summary(
                "Fitur yang dipakai association rules dan perlakuannya (klik)",
                style={"cursor": "pointer", "fontWeight": "700", "fontSize": "12.5px",
                       "color": STEEL, "padding": "4px 0", "userSelect": "none"},
            ),
            html.Div(
                [
                    html.P(
                        "19 fitur dijadikan kandidat item, lalu diubah menjadi matriks "
                        "transaksi one-hot pada 67.827 aplikasi berkeputusan. Item yang muncul "
                        "di bawah 2% atau di atas 95% dibuang lebih dulu, menyisakan 82 item "
                        "yang benar-benar ditambang.",
                        style={"fontSize": "12px", "color": INK, "margin": "0 0 10px",
                               "lineHeight": "1.6"},
                    ),
                    _table(df),
                    html.P(
                        "Tidak ada penskalaan di sini, karena Apriori tidak menghitung jarak. "
                        "Fitur kontinu ditangani lewat binning, bukan StandardScaler. Fitur "
                        "pasca-keputusan seperti interest rate dan biaya tetap dikecualikan, "
                        "sama seperti di clustering dan deteksi anomali.",
                        style={"fontSize": "11.5px", "color": MUTE, "margin": "10px 0 0",
                               "lineHeight": "1.6"},
                    ),
                ],
                style={"marginTop": "12px"},
            ),
        ],
        style={"background": "#f8fafd", "border": f"1px solid {BORDER}",
               "borderRadius": "10px", "padding": "12px 16px", "marginBottom": "16px"},
    )


def _cluster_features_panel():
    """Exactly which features each matrix uses, and how each is scaled."""
    clust = pd.DataFrame([
        {"Fitur": "income", "Tipe": "kontinu", "Perlakuan": "winsorize 1/99% lalu StandardScaler"},
        {"Fitur": "combined_loan_to_value_ratio", "Tipe": "kontinu", "Perlakuan": "winsorize 1/99% lalu StandardScaler"},
        {"Fitur": "tract_minority_population_percent", "Tipe": "kontinu", "Perlakuan": "winsorize 1/99% lalu StandardScaler"},
        {"Fitur": "tract_to_msa_income_percentage", "Tipe": "kontinu", "Perlakuan": "winsorize 1/99% lalu StandardScaler"},
        {"Fitur": "_is_investment", "Tipe": "biner", "Perlakuan": "StandardScaler"},
        {"Fitur": "_is_refinance", "Tipe": "biner", "Perlakuan": "StandardScaler"},
        {"Fitur": "_is_manufactured", "Tipe": "biner", "Perlakuan": "StandardScaler"},
        {"Fitur": "_is_subordinate", "Tipe": "biner", "Perlakuan": "StandardScaler"},
        {"Fitur": "_is_high_dti", "Tipe": "biner", "Perlakuan": "StandardScaler"},
    ])
    anom = pd.DataFrame([
        {"Fitur": f, "Perlakuan": "RobustScaler, nilai asli tanpa winsorize"}
        for f in ["income", "loan_amount", "property_value",
                  "combined_loan_to_value_ratio", "loan_term",
                  "tract_minority_population_percent",
                  "tract_to_msa_income_percentage",
                  "ffiec_msa_md_median_family_income"]
    ])
    scope = pd.DataFrame([
        {"Metode": "K-Means", "Fitur": "9 fitur yang sama", "Penskalaan": "StandardScaler", "Cakupan": "Seluruh 99.995 baris"},
        {"Metode": "DBSCAN", "Fitur": "9 fitur yang sama", "Penskalaan": "StandardScaler", "Cakupan": "Sampel 20.000"},
        {"Metode": "Hierarchical (Ward)", "Fitur": "9 fitur yang sama", "Penskalaan": "StandardScaler", "Cakupan": "Sampel 4.000"},
        {"Metode": "CLARANS", "Fitur": "9 fitur yang sama", "Penskalaan": "StandardScaler", "Cakupan": "Sampel 4.000"},
    ])
    return panel(
        "Fitur yang dipakai tiap metode dan cara penskalaannya",
        [
            html.P(
                "Keempat metode clustering memakai satu matriks yang sama, yaitu 9 fitur dengan "
                "StandardScaler. Deteksi anomali di Fase 4 memakai matriks terpisah dengan "
                "RobustScaler.",
                style={"fontSize": "12px", "color": INK, "margin": "0 0 12px"},
            ),
            _table(scope),
            html.H4("Matriks clustering: 9 fitur",
                    style={"fontSize": "13px", "color": NAVY, "margin": "16px 0 8px"}),
            _table(clust),
            html.H4("Matriks anomali: 8 fitur",
                    style={"fontSize": "13px", "color": NAVY, "margin": "16px 0 8px"}),
            _table(anom),
            why(WHY_CLUSTER_FEATS),
        ],
        sub="Fitur pasca-keputusan seperti interest rate dan biaya dikecualikan dari kedua "
        "matriks untuk mencegah kebocoran data.",
    )


def _key_takeaways_panel():
    """Closing section: the cross-phase findings, in plain business language."""
    cards = [
        html.Div(
            [
                html.Div(num, style={"fontSize": "22px", "fontWeight": "800",
                                     "color": color, "marginBottom": "2px"}),
                html.Div(title, style={"fontSize": "13px", "fontWeight": "700",
                                       "color": NAVY, "marginBottom": "6px"}),
                html.Div(body, style={"fontSize": "11.5px", "color": INK,
                                      "lineHeight": "1.5"}),
            ],
            className="hmda-card",
            style={"flex": "1", "minWidth": "220px", "background": CARD,
                   "borderRadius": "14px", "padding": "14px 16px",
                   "boxShadow": SOFT_SHADOW, "border": f"1px solid {BORDER}",
                   "borderTop": f"3px solid {color}"},
        )
        for title, body, num, color in KEY_TAKEAWAYS
    ]
    actions = [
        html.Div(
            [
                html.Span(f"{i}. ", style={"fontWeight": "800", "color": STEEL}),
                html.Span(title, style={"fontWeight": "700", "color": NAVY}),
                html.Div(body, style={"fontSize": "12px", "color": INK,
                                      "lineHeight": "1.55", "marginTop": "3px"}),
            ],
            style={"padding": "10px 12px", "background": BG, "borderRadius": "10px",
                   "marginBottom": "8px", "fontSize": "12.5px"},
        )
        for i, (title, body) in enumerate(RECOMMENDED_ACTIONS, 1)
    ]
    return html.Div(
        [
            panel(
                "Key Takeaways",
                [html.Div(cards, style={"display": "flex", "gap": "14px",
                                        "flexWrap": "wrap"})],
                sub="Temuan yang didukung lebih dari satu fase. Seluruhnya asosiasi historis, "
                "bukan hubungan sebab-akibat.",
            ),
            panel(
                "Tindakan yang disarankan",
                actions,
                sub="Tiap tindakan mengikuti langsung dari pola yang ditemukan, bukan dari "
                "praktik umum industri.",
            ),
        ]
    )


def _collective_groups_panel():
    """Group-level collective anomalies.

    Reported narrowly on purpose. Half the state-based hits are small jurisdictions, which is
    ordinary geographic variation rather than a finding, so the panel leads with the one
    pattern that is corroborated by other phases instead of presenting all 33 as discoveries.
    """
    if collective_groups is None or not len(collective_groups):
        return html.Div()

    d = collective_groups.copy()
    non_state = d[~d["group_spec"].str.startswith("state_")]
    tbl = pd.DataFrame({
        "Kelompok": non_state["group_values"],
        "Skema": non_state["group_spec"],
        "Anggota": non_state["n"].map(lambda v: f"{int(v):,}"),
        "Skor": non_state["collective_iso_score"].map(lambda v: f"{v:.3f}"),
        "Anggota ditandai individual": non_state["member_individual_flag_rate"].map(
            lambda v: f"{v*100:.0f}%"),
    })

    return panel(
        "Anomali kolektif tingkat grup",
        [
            html.Div(
                [
                    html.B("Temuan: manufactured housing muncul lagi sebagai pola kolektif. ",
                           style={"fontSize": "12.5px", "color": NAVY}),
                    html.Span(
                        "Dari 4 kelompok yang ditandai di luar skema geografis, 3 di antaranya "
                        "adalah manufactured housing. Ini konfirmasi independen atas segmen yang "
                        "sama, yang di Fase 2 juga paling kohesif dengan silhouette 0,495 dan "
                        "purity 1,000, serta memperoleh skor bukti 4 dari 4 pada audit kolektif "
                        "K-Means dengan hanya 8,7% anggotanya pernah ditandai detektor individual. "
                        "Artinya keanehannya benar-benar ada di tingkat kelompok, bukan pada "
                        "loan-nya satu per satu.",
                        style={"fontSize": "12px", "color": INK, "lineHeight": "1.6"},
                    ),
                ],
                style={"background": "#eefaf1", "border": "1px solid #b7e4c7",
                       "borderRadius": "10px", "padding": "12px 14px", "marginBottom": "14px"},
            ),
            _table(tbl),
            html.Div(
                [
                    html.B("Yang sengaja tidak diklaim sebagai temuan. ",
                           style={"fontSize": "12px", "color": NAVY}),
                    html.Span(
                        "29 kelompok lain berbasis negara bagian, dan 15 di antaranya atau 52% "
                        "adalah yurisdiksi kecil seperti Hawaii, DC, dan Puerto Rico, atau state "
                        "yang tidak diketahui. Kelompok berukuran kecil memang mudah terlihat "
                        "menyimpang, jadi itu variasi geografis biasa dan tidak dilaporkan sebagai "
                        "penemuan.",
                        style={"fontSize": "12px", "color": INK, "lineHeight": "1.6"},
                    ),
                ],
                style={"background": "#fff8e8", "border": "1px solid #f3d7a0",
                       "borderRadius": "10px", "padding": "12px 14px", "margin": "14px 0"},
            ),
            html.Div(
                [
                    html.B("Insight bisnis: ", style={"fontSize": "12px", "color": NAVY}),
                    html.Span(
                        "manufactured housing perlu jalur produk sendiri, bukan pengetatan "
                        "underwriting. Polanya konsisten di tingkat segmen maupun kelompok, dan "
                        "mayoritas anggotanya normal secara individual, jadi masalahnya ada pada "
                        "kecocokan produk, bukan pada kualitas masing-masing pemohon.",
                        style={"fontSize": "12px", "color": INK, "lineHeight": "1.6"},
                    ),
                ],
                style={"background": BG, "borderRadius": "10px", "padding": "12px 14px",
                       "marginBottom": "14px"},
            ),
            why(WHY_COLLECTIVE_GROUPS),
        ],
        sub="Isolation Forest pada profil kelompok, bukan pada baris. Ditampilkan hanya kelompok "
        "non-geografis, karena hasil berbasis negara bagian didominasi yurisdiksi kecil.",
    )


def _feature_funnel_panel():
    """Show the Phase-1 column flow and the feature subsets used downstream.

    Important distinction:
    - 60 = columns available in the cleaned/engineered working table after binning.
    - 31 = pre-decision candidate features scored with correlation + mutual information.
    - 9 / 19 / 8 = algorithm-specific subsets; they overlap and must not be summed.
    """
    # Audited directly from the final notebook run.
    RAW_COLS = 99
    REDUNDANT_ID_OUTCOME_DROPPED = 46
    HIGH_MISSING_DROPPED = 6
    CLEAN_BASE_COLS = 53
    ANALYSIS_READY_COLS = 60
    SCORED_CANDIDATES = 31
    CLUSTER_N = 9
    ARM_N = 19
    ANOMALY_N = 8

    cleaning_tiles = html.Div(
        [
            _stat_tile("Kolom mentah", f"{RAW_COLS}", NAVY),
            _stat_tile(
                "Redundan / ID / outcome dibuang",
                f"{REDUNDANT_ID_OUTCOME_DROPPED}",
                RED,
            ),
            _stat_tile("High-missing dibuang", f"{HIGH_MISSING_DROPPED}", AMBER),
            _stat_tile("Kolom tabel bersih", f"{CLEAN_BASE_COLS}", GREEN),
            _stat_tile("Setelah binning", f"{ANALYSIS_READY_COLS}", TEAL),
        ],
        style={
            "display": "flex",
            "gap": "14px",
            "flexWrap": "wrap",
            "marginBottom": "10px",
        },
    )

    downstream_tiles = html.Div(
        [
            _stat_tile("Kandidat fitur yang diskor", f"{SCORED_CANDIDATES}", NAVY),
            _stat_tile("Clustering", f"{CLUSTER_N}", STEEL),
            _stat_tile("Association rules", f"{ARM_N}", TEAL),
            _stat_tile("Anomaly detection", f"{ANOMALY_N}", AMBER),
        ],
        style={
            "display": "flex",
            "gap": "14px",
            "flexWrap": "wrap",
            "marginBottom": "14px",
        },
    )

    rows = [
        (
            "Pengenal / kardinalitas tinggi",
            5,
            "lei, census_tract, county_code, dan sejenisnya bukan karakteristik applicant yang layak menjadi input mining.",
        ),
        (
            "Demografi mentah",
            30,
            "Representasi rinci applicant/co-applicant diringkas oleh derived_race, derived_ethnicity, dan derived_sex.",
        ),
        (
            "Hasil AUS",
            5,
            "Keluaran automated underwriting system; tidak dipakai sebagai karakteristik applicant.",
        ),
        (
            "Alasan penolakan",
            4,
            "Hanya tersedia pada aplikasi ditolak; dipisahkan untuk analisis alasan denial agar tidak membocorkan outcome.",
        ),
        (
            ">60% kosong",
            6,
            "Missingness terlalu tinggi untuk diimputasi secara defensible.",
        ),
        (
            "Pricing / cost pasca-keputusan",
            13,
            "Tidak masuk feature set Fase 2-4 karena tersedia setelah proses keputusan/pricing dan berisiko leakage.",
        ),
    ]
    df = pd.DataFrame(
        [{"Kelompok": k, "Jumlah": n, "Perlakuan / alasan": r} for k, n, r in rows]
    )

    return panel(
        "Alur kolom dan feature set setelah Fase 1",
        [
            html.P(
                "Angka cleaning dan angka feature selection sengaja dipisahkan. "
                "Tabel bersih tetap menyimpan kolom yang berguna untuk audit, profiling, dan visualisasi, "
                "sedangkan setiap metode mining hanya menerima subset yang relevan dengan algoritmanya.",
                style={"fontSize": "12px", "color": INK, "margin": "0 0 12px"},
            ),
            html.H4(
                "1. Alur kolom pada tabel kerja",
                style={"fontSize": "13px", "color": NAVY, "margin": "4px 0 10px"},
            ),
            cleaning_tiles,
            html.P(
                "Kenapa 99 - 46 - 6 tidak langsung menjadi 53? Selama cleaning ditambahkan "
                "1 helper audit (`any_exempt_field`) dan 5 indikator `_was_missing`. Setelah itu "
                "7 band domain ditambahkan, sehingga tabel kerja akhir memiliki 60 kolom. "
                "Kolom helper/diagnostik tidak otomatis menjadi input algoritma.",
                style={
                    "fontSize": "11.5px",
                    "color": MUTE,
                    "margin": "0 0 16px",
                    "lineHeight": "1.55",
                },
            ),
            html.H4(
                "2. Subset yang benar-benar dipakai untuk mining",
                style={"fontSize": "13px", "color": NAVY, "margin": "4px 0 10px"},
            ),
            downstream_tiles,
            html.P(
                "31 kandidat dinilai dengan |korelasi| dan mutual information terhadap Originated vs Denied. "
                "Dari tabel yang sudah bersih, subset akhir kemudian disesuaikan dengan kebutuhan metode: "
                "9 fitur untuk clustering, 19 untuk association rule mining, dan 8 untuk anomaly detection. "
                "Ketiga subset saling overlap, jadi 9 + 19 + 8 bukan jumlah kolom unik.",
                style={
                    "fontSize": "11.5px",
                    "color": MUTE,
                    "margin": "0 0 16px",
                    "lineHeight": "1.55",
                },
            ),
            html.H4(
                "3. Kelompok kolom yang dieliminasi atau dikecualikan",
                style={"fontSize": "13px", "color": NAVY, "margin": "4px 0 10px"},
            ),
            _table(df),
            html.P(
                "Catatan: kategori pada tabel ini tidak semuanya merupakan langkah drop yang berurutan. "
                "Sebagian kolom pricing/cost tetap tersedia di tabel bersih untuk audit, tetapi dilarang masuk "
                "feature set downstream karena leakage. Beberapa kategori juga overlap dengan high-missing.",
                style={
                    "fontSize": "11px",
                    "color": MUTE,
                    "margin": "10px 0 0",
                    "lineHeight": "1.5",
                },
            ),
            why(WHY_FUNNEL),
        ],
        sub="99 kolom mentah -> cleaning/engineering -> 60 kolom tabel kerja; feature set downstream: 9 clustering, 19 ARM, 8 anomaly.",
    )


def _load_test_panel():
    """Fase 5 reporting: measured performance profile of this dashboard."""
    tiles = html.Div(
        [
            _stat_tile("Total request", "6.174"),
            _stat_tile("Error", "0", GREEN),
            _stat_tile("Throughput puncak", "298 req/s", STEEL),
            _stat_tile("Rata-rata warm", "18 ms", TEAL),
            _stat_tile("Rata-rata sustained", "21 ms", TEAL),
            _stat_tile("Cold start", "2,9 s", AMBER),
            _stat_tile("Memori (RSS)", "190 MB", NAVY),
        ],
        style={
            "display": "flex",
            "gap": "14px",
            "flexWrap": "wrap",
            "marginBottom": "16px",
        },
    )

    cold_df = pd.DataFrame(
        [
            {"Tab": t, "Render pertama (ms)": ms, "Ukuran respons (KB)": kb}
            for t, ms, kb in LOADTEST_COLD
        ]
    )
    conc_df = pd.DataFrame(
        [
            {
                "Concurrency": c,
                "Throughput (req/s)": tp,
                "Rata-rata (ms)": mean,
                "p50 (ms)": p50,
                "p95 (ms)": p95,
                "p99 (ms)": p99,
            }
            for c, tp, mean, p50, p95, p99 in LOADTEST_CONC
        ]
    )

    return panel(
        "Laporan uji beban (load test) dashboard",
        [
            html.P(
                "Pengujian menembak endpoint callback yang benar-benar membangun tiap tab, "
                "bukan sekadar halaman statis. Hasilnya: 6.174 request tanpa satu pun error.",
                style={"fontSize": "12px", "color": INK, "margin": "0 0 12px"},
            ),
            tiles,
            html.Div(
                [
                    html.H4(
                        "Render pertama tiap tab (cache kosong)",
                        style={
                            "fontSize": "13px",
                            "color": NAVY,
                            "margin": "6px 0 8px",
                        },
                    ),
                    _table(cold_df),
                    html.Div(style={"height": "16px"}),
                    html.H4(
                        "Penskalaan terhadap concurrency (100 request, cache terisi)",
                        style={
                            "fontSize": "13px",
                            "color": NAVY,
                            "margin": "6px 0 8px",
                        },
                    ),
                    _table(conc_df),
                    html.Div(style={"height": "16px"}),
                    html.H4(
                        "Pemeriksaan dengan Postman (20 VU, 5 menit)",
                        style={
                            "fontSize": "13px",
                            "color": NAVY,
                            "margin": "6px 0 8px",
                        },
                    ),
                    html.Img(
                        src=app.get_asset_url("p5_load_test.png"),
                        alt="Ringkasan uji beban 20 VU selama 5 menit: 5.579 request, "
                        "17,78 request per detik, rata-rata 8 ms, error 0,00%.",
                        style={
                            "width": "100%",
                            "maxWidth": "760px",
                            "height": "auto",
                            "display": "block",
                            "borderRadius": "10px",
                            "border": f"1px solid {BORDER}",
                            "margin": "0 0 10px",
                        },
                    ),
                    html.P(
                        "5.579 request · 0,00% error · rata-rata 8 ms · P90 12 ms · P95 15 ms · P99 26 ms.",
                        style={
                            "fontSize": "12px",
                            "color": INK,
                            "margin": "0 0 14px",
                            "lineHeight": "1.6",
                        },
                    ),
                    html.P(
                        "Dijalankan memakai Postman pada satu endpoint GET dengan laju tetap 17,8 req/s. Ini bukan uji kapasitas: 20 VU yang cuma menghasilkan 17,8 "
                        "req/s berarti lajunya ditahan oleh tool, bukan oleh aplikasi, dan endpoint "
                        "statis memang jauh lebih murah daripada callback yang membangun tab. Nilainya "
                        "ada sebagai cross check dari Postman: nol error dan latency tetap "
                        "rendah selama 5 menit penuh.",
                        style={
                            "fontSize": "12px",
                            "color": INK,
                            "margin": "0 0 14px",
                            "lineHeight": "1.6",
                        },
                    ),
                ]
            ),
            why(WHY_LOADTEST, "Justifikasi"),
        ],
        sub="Diukur 26 Juli 2026 · Flask development server · Python 3.14 · satu instance",
    )


def _clustering_comparison_panel():
    """Fase 2 method comparison: the metrics table plus a plain-language verdict."""
    if clustering_cmp is None or not len(clustering_cmp):
        return panel(
            "Perbandingan metode clustering",
            [
                html.Div(
                    "Data perbandingan belum tersedia. Jalankan `python app/build_data.py` "
                    "untuk membuat dash_clustering_comparison.csv.",
                    style={"color": MUTE, "fontSize": "12px"},
                )
            ],
        )
    d = clustering_cmp.copy()
    d = d.rename(
        columns={
            "method": "Metode",
            "scope": "Cakupan",
            "n_clusters": "Jumlah cluster",
            "noise": "Noise",
            "silhouette": "Silhouette ↑",
            "davies_bouldin": "Davies-Bouldin ↓",
            "calinski_harabasz": "Calinski-Harabasz ↑",
            "ari_vs_kmeans": "ARI vs K-Means",
        }
    )
    show = [
        c
        for c in [
            "Metode",
            "Cakupan",
            "Jumlah cluster",
            "Noise",
            "Silhouette ↑",
            "Davies-Bouldin ↓",
            "Calinski-Harabasz ↑",
            "ARI vs K-Means",
        ]
        if c in d.columns
    ]

    # Verdict computed from the data, not hard-coded, so it stays true after a re-run.
    same = clustering_cmp[clustering_cmp["scope"].astype(str).str.contains("4.000")]
    verdict = ""
    if len(same):
        best = same.loc[same["silhouette"].idxmax()]
        verdict = (
            f"Pada sampel 4.000 baris yang sama, pemenangnya adalah {best['method']} "
            f"(silhouette {best['silhouette']:.3f}, Davies-Bouldin {best['davies_bouldin']:.3f}, "
            f"Calinski-Harabasz {best['calinski_harabasz']:.0f})."
        )
    return panel(
        "Perbandingan metode clustering: mana yang terbaik dan mengapa",
        [
            _table(d[show]),
            html.Div(
                [
                    html.P(
                        [
                            html.B("Cara membaca: "),
                            "Silhouette makin tinggi makin baik (cluster rapat dan terpisah). "
                            "Davies-Bouldin makin rendah makin baik (cluster tidak saling tumpang tindih). "
                            "Calinski-Harabasz makin tinggi makin baik. ARI mengukur kesepakatan "
                            "dengan K-Means: 1,0 identik, 0 acak.",
                        ],
                        style={
                            "fontSize": "12px",
                            "color": INK,
                            "lineHeight": "1.6",
                            "margin": "12px 0 8px",
                        },
                    ),
                    html.P(
                        [
                            html.B("Kesimpulan: "),
                            verdict,
                            " K-Means dipakai sebagai segmentasi "
                            "utama karena unggul pada ketiga metrik validitas sekaligus satu-satunya "
                            "yang skalabel ke seluruh 99.995 aplikasi. Ward hierarchical menyepakatinya "
                            "kuat (ARI 0,906), yang menegaskan struktur 7 segmen memang ada di data dan "
                            "bukan artefak satu algoritma. CLARANS lebih rendah (ARI 0,710) karena "
                            "mengoptimalkan medoid, bukan centroid, sehingga wajar berbeda.",
                        ],
                        style={
                            "fontSize": "12px",
                            "color": INK,
                            "lineHeight": "1.6",
                            "margin": "0",
                        },
                    ),
                    html.P(
                        "Catatan kejujuran metrik: baris DBSCAN diukur pada sampel 20.000 dan "
                        "mengecualikan noise, jadi angkanya tidak sebanding langsung dengan baris "
                        'sampel 4.000. Baris "K-Means (pada sampel 4.000)" sengaja disertakan '
                        "sebagai pembanding setara.",
                        style={
                            "fontSize": "11.5px",
                            "color": MUTE,
                            "lineHeight": "1.6",
                            "margin": "10px 0 0",
                        },
                    ),
                ]
            ),
            why(WHY_BEST_METHOD),
        ],
        sub="Semua metode dinilai pada matriks berskala yang benar-benar dipakai algoritmanya.",
    )


def _fase1_content():
    if phase1_summary is None or not len(phase1_summary):
        return panel(
            "Fase 1 - Prapemrosesan",
            [
                html.Div(
                    "Data prapemrosesan belum tersedia. Jalankan notebook (sel "
                    '"export_phase1_aggregates" pada Fase 5) atau `python app/build_data.py` '
                    "untuk membuat berkas dash_phase1_*.csv, lalu muat ulang dashboard.",
                    style={"color": MUTE, "fontSize": "13px", "lineHeight": "1.6"},
                )
            ],
        )
    s = phase1_summary.iloc[0]
    tiles = html.Div(
        [
            _stat_tile("Baris mentah", f"{int(s['raw_rows']):,}"),
            _stat_tile("Baris bersih", f"{int(s['clean_rows']):,}", GREEN),
            _stat_tile("Duplikat dihapus", f"{int(s['duplicates_removed']):,}", AMBER),
            _stat_tile("High-missing dibuang", f"{int(s['fields_dropped']):,}", RED),
            _stat_tile("Sel kosong tersisa", f"{int(s['residual_missing_cells']):,}"),
        ],
        style={
            "display": "flex",
            "gap": "14px",
            "flexWrap": "wrap",
            "marginBottom": "16px",
        },
    )
    children = [
        panel(
            "Ringkasan pembersihan data",
            [
                html.P(
                    "Alur KDD Fase 1: audit nilai sentinel, analisis missingness struktural, "
                    "penghapusan duplikat, imputasi, lalu seleksi fitur untuk Fase 2-4.",
                    style={"fontSize": "12px", "color": INK, "margin": "0 0 12px"},
                ),
                tiles,
                why(WHY_CLEANING),
            ],
        ),
        panel(
            "Segmentasi tipe fitur",
            [
                html.P(
                    "Sebelum analisis, seluruh kolom dipartisi menurut makna dan perlakuan "
                    "analitisnya. Partisi ini wajib menutup setiap kolom (divalidasi), sehingga tidak "
                    "ada fitur yang terlewat atau diperlakukan ganda:",
                    style={"fontSize": "12px", "color": INK, "margin": "0 0 8px"},
                ),
                html.Ul(
                    [
                        html.Li(
                            [
                                html.B("CONTINUOUS"),
                                " - fitur bilangan riil (income, loan_amount, "
                                "property_value, CLTV). Diimputasi median lalu diskalakan.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("STRING_BAND"),
                                " - fitur rentang angka (debt_to_income_ratio, "
                                "applicant_age). Dipertahankan sebagai band karena HMDA melaporkannya "
                                "sebagai rentang, bukan satu angka.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("CATEG_CODE"),
                                " - kategorikal berkode (action_taken, "
                                "loan_purpose, occupancy_type). Diterjemahkan dari kode angka ke label bermakna.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("TEXT_CATEG"),
                                " - kategorikal teks (state_code, derived_race). "
                                "Dipakai apa adanya.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("IDS"),
                                " - pengenal (lei, census_tract). Dikecualikan dari pemodelan.",
                            ]
                        ),
                    ],
                    style={
                        "fontSize": "12px",
                        "color": INK,
                        "lineHeight": "1.6",
                        "margin": "0",
                        "paddingLeft": "18px",
                    },
                ),
                html.P(
                    "Pemisahan ini penting karena tiap tipe butuh pembersihan, encoding, dan "
                    "penskalaan yang berbeda. Memperlakukan kode kategorikal sebagai angka kontinu, "
                    "misalnya, akan menciptakan urutan (ordinality) palsu.",
                    style={"fontSize": "12px", "color": MUTE, "margin": "10px 0 0"},
                ),
                why(WHY_FEATURE_TYPES),
            ],
        ),
    ]

    if phase1_missing is not None and len(phase1_missing):
        m = phase1_missing.copy()
        m["fate_label"] = m["fate"].map(_FATE_LABEL).fillna(m["fate"])
        fig_m = px.bar(
            m,
            x="missing_pct",
            y="field",
            orientation="h",
            color="fate_label",
            color_discrete_map={_FATE_LABEL[k]: v for k, v in _FATE_COLOR.items()},
            labels={
                "missing_pct": "% nilai kosong",
                "field": "",
                "fate_label": "Penanganan",
            },
        )
        fig_m.update_layout(
            template="hmda",
            height=max(360, 22 * len(m)),
            yaxis={"categoryorder": "total ascending"},
            legend={"orientation": "h", "y": -0.14, "title": ""},
        )
        children.append(
            panel(
                "Missingness per kolom dan penanganannya",
                [graph(fig_m), why(WHY_MISSING)],
                sub="Kolom dengan >60% nilai kosong dibuang; sisanya diimputasi median (fitur "
                "kontinu) atau diisi 'Unknown' (fitur kategorikal).",
            )
        )

    children.append(
        panel(
            "Penanganan nilai hilang: metode dan alasannya",
            [
                html.Ul(
                    [
                        html.Li(
                            [
                                html.B("Buang kolom >60% hilang."),
                                " Missingness struktural: "
                                "mengimputasi kolom yang mayoritasnya kosong sama saja mengarang data.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("Fitur CONTINUOUS -> imputasi median."),
                                " Median dipilih, bukan "
                                "mean, karena distribusi income/loan/property sangat skewed dengan outlier "
                                "ekstrem; median tahan terhadap nilai ekstrem sehingga pusat distribusi tidak "
                                "tertarik oleh segelintir nilai raksasa. Median juga transparan dan mudah "
                                "diaudit - penting untuk data regulasi. Metode seperti KNN-imputation sengaja "
                                "dihindari: mahal pada 100.000 baris dan kurang dapat dijelaskan.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B('Fitur kategorikal -> diisi "Unknown".'),
                                " Nilai tidak ditebak. "
                                '"Hilang" sering bermakna (mis. Exempt / tidak dilaporkan), jadi '
                                "dipertahankan sebagai kategori tersendiri, bukan disamarkan.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("Indikator missingness (_was_missing)."),
                                " Ditambahkan untuk kolom "
                                "sinyal utama, sehingga jejak imputasi tetap terekam dan dapat diaudit.",
                            ]
                        ),
                    ],
                    style={
                        "fontSize": "12px",
                        "color": INK,
                        "lineHeight": "1.6",
                        "margin": "0",
                        "paddingLeft": "18px",
                    },
                ),
            ],
            sub="Nilai hilang tidak ditangani dengan satu metode seragam, melainkan sesuai tipe fitur "
            "dan penyebab hilangnya.",
        )
    )

    if phase1_features is not None and len(phase1_features):
        f = phase1_features.copy().sort_values("score").tail(15)
        if "role" in f.columns:
            f["role_label"] = f["role"].map(_ROLE_LABEL).fillna(f["role"])
            color_arg = {
                "color": "role_label",
                "color_discrete_map": {
                    "Kandidat kuat": GREEN,
                    "Kandidat sedang": STEEL,
                    "Lemah / pendukung": MUTE,
                },
            }
        else:
            color_arg = {}
        fig_f = px.bar(
            f,
            x="score",
            y="feature",
            orientation="h",
            labels={
                "score": "Skor gabungan (korelasi + mutual information)",
                "feature": "",
                "role_label": "Peran",
            },
            **color_arg,
        )
        fig_f.update_layout(
            template="hmda",
            height=max(360, 26 * len(f)),
            legend={"orientation": "h", "y": -0.14, "title": ""},
        )
        children.append(
            panel(
                "Seleksi fitur: pentingnya fitur untuk keputusan persetujuan",
                [graph(fig_f), why(WHY_FEATURE_SELECTION)],
                sub="Skor menggabungkan |korelasi| (hubungan linear) dan mutual information "
                "(relevansi non-linear) terhadap Originated vs Denied. Fitur pasca-keputusan "
                "dan diagnostik proses dikecualikan dari input Fase 2-4.",
            )
        )

    children.append(_feature_funnel_panel())
    children.append(_vif_panel())

    children.append(
        panel(
            "Transformasi & penskalaan",
            [
                html.Ul(
                    [
                        html.Li(
                            [
                                html.B("Clustering (Fase 2)."),
                                " Fitur kontinu di-winsorize pada 1%/99% "
                                "(mengekang ekor ekstrem agar tidak mendominasi jarak), lalu diskalakan "
                                "dengan StandardScaler (z-score) sehingga setiap fitur berkontribusi setara "
                                "pada jarak Euclidean K-Means.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("Deteksi anomali (Fase 4)."),
                                " Memakai RobustScaler (berpusat pada "
                                "median, diskalakan dengan IQR). Robust dipilih karena penskalaan tidak boleh "
                                "terdistorsi oleh outlier yang justru sedang dicari; median dan IQR tidak "
                                "sensitif terhadap nilai ekstrem, sehingga anomali sejati tetap jauh dari "
                                "pusat alih-alih menekan skala.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("Binning domain."),
                                " Diterapkan pada income, loan amount, property "
                                "value, CLTV, tract income, dan tract minority share agar rentang mentah "
                                "menjadi kategori yang mudah dibaca untuk association rule mining (Fase 3).",
                            ]
                        ),
                    ],
                    style={
                        "fontSize": "12px",
                        "color": INK,
                        "lineHeight": "1.6",
                        "margin": "0",
                        "paddingLeft": "18px",
                    },
                ),
                html.P(
                    "Penskalaan sengaja berbeda untuk clustering dan deteksi anomali karena tujuannya "
                    "berbeda: clustering butuh setiap fitur setara, sedangkan deteksi anomali harus "
                    "menjaga outlier tetap menonjol.",
                    style={"fontSize": "12px", "color": MUTE, "margin": "10px 0 0"},
                ),
                why(WHY_SCALING),
            ],
        )
    )

    return html.Div(children)


# ============================================================ tab routing
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
@lru_cache(maxsize=8)
def render(tab):
    # KDD phase routing. Fase 2/3/4 reuse the existing analytical panels; Fase 5 is the
    # reporting layer, composed from the executive-summary, geography, what-if and
    # fairness panels; Fase 1 is the preprocessing view built from dash_phase1_*.csv.
    if tab == "fase1":
        return _fase1_content()
    if tab == "fase2":
        tab = "segments"
    elif tab == "fase3":
        tab = "rules"
    elif tab == "fase4":
        tab = "anomalies"
    elif tab == "fase5":
        sections = [
            ("f5-geo", "Geografi", render("geography")),
            ("f5-whatif", "What-If", render("whatif")),
            ("f5-fair", "Fairness", render("fairness")),
            ("f5-load", "Load Test", _load_test_panel()),
            ("f5-key", "Key Takeaways", _key_takeaways_panel()),
        ]
        nav = html.Div(
            [
                html.Div(
                    "Bagian",
                    style={
                        "fontSize": "10.5px",
                        "fontWeight": "800",
                        "letterSpacing": "1.2px",
                        "color": MUTE,
                        "marginBottom": "10px",
                    },
                ),
            ]
            + [
                html.A(
                    title,
                    href=f"#{sid}",
                    className="hmda-navlink",
                    style={
                        "display": "block",
                        "padding": "8px 10px",
                        "marginBottom": "4px",
                        "borderRadius": "8px",
                        "fontSize": "12.5px",
                        "fontWeight": "600",
                        "color": NAVY,
                        "textDecoration": "none",
                        "borderLeft": f"3px solid {GRID}",
                    },
                )
                for sid, title, _ in sections
            ],
            style={
                "position": "sticky",
                "top": "12px",
                "alignSelf": "flex-start",
                "flex": "0 0 170px",
                "background": CARD,
                "borderRadius": "14px",
                "padding": "14px 12px",
                "border": f"1px solid {BORDER}",
                "boxShadow": SOFT_SHADOW,
            },
        )
        body = html.Div(
            [
                html.Div(content, id=sid, style={"scrollMarginTop": "12px"})
                for sid, _, content in sections
            ],
            style={"flex": "1", "minWidth": "0"},
        )
        return html.Div(
            [nav, body],
            style={"display": "flex", "gap": "18px", "alignItems": "flex-start"},
        )

    if tab == "summary":
        return html.Div(
            [
                html.Div(
                    executive_finding_cards(),
                    style={
                        "display": "flex",
                        "gap": "14px",
                        "flexWrap": "wrap",
                        "marginBottom": "4px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            panel(
                                "Tingkat persetujuan per segmen",
                                [graph(fig_approval_by_cluster())],
                            ),
                            style={"flex": "1", "minWidth": "380px"},
                        ),
                        html.Div(
                            panel(
                                "Kenapa aplikasi ditolak",
                                [graph(fig_denial_reasons())],
                            ),
                            style={"flex": "1", "minWidth": "380px"},
                        ),
                    ],
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                ),
                panel(
                    "Pola paling jelas",
                    [graph(fig_approval_by_dti())],
                    sub="Tingkat persetujuan menurut band debt-to-income, pendorong penolakan paling dominan.",
                ),
            ]
        )

    if tab == "geography":
        if state_summary is None or not len(state_summary):
            return panel(
                "Geografi",
                [
                    html.Div(
                        "Data ringkasan negara bagian tidak ditemukan. Jalankan ekspor Fase 5 notebook (atau "
                        "pembuat dash_state_summary.csv) dulu.",
                        style={"color": MUTE},
                    )
                ],
            )
        states = state_summary.sort_values("n", ascending=False)["state_code"].tolist()
        default_state = states[0] if states else None
        return html.Div(
            [
                panel(
                    "Filters",
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Metrik peta",
                                    style={"fontSize": "12px", "color": MUTE},
                                ),
                                dcc.Dropdown(
                                    id="geo-metric",
                                    options=[
                                        {"label": v[0], "value": k}
                                        for k, v in STATE_METRICS.items()
                                    ],
                                    value="approval_rate",
                                    clearable=False,
                                    style={"fontSize": "12px"},
                                ),
                            ],
                            style={"width": "260px"},
                        ),
                    ],
                ),
                panel(
                    "Persetujuan, volume, dan harga di seluruh negeri",
                    [graph("geo-map")],
                    sub="Arahkan kursor ke sebuah negara bagian untuk ringkasan cepat. Klik untuk memuat rincian lebih lengkap di bawah.",
                ),
                html.Div(
                    _geo_state_detail_children(default_state), id="geo-state-detail"
                ),
            ]
        )

    if tab == "segments":
        # Order: read the method rationale and the shared setup first, then choose a
        # method, then look at its charts. The feature list lives here once because all
        # four methods run on the identical matrix; repeating it per method was noise.
        return html.Div(
            [
                panel(
                    "Metode clustering dan dasar pemilihannya",
                    [
                        html.P(
                            "Keempat metode dijalankan pada matriks yang sama persis, yaitu 9 "
                            "fitur dengan StandardScaler. K-Means adalah segmentasi utama dan "
                            "dilatih pada seluruh 99.995 aplikasi, sedangkan DBSCAN, Hierarchical, "
                            "dan CLARANS dijalankan pada sampel sebagai validasi silang. Karena "
                            "fitur dan perlakuannya identik, perbedaan hasil di bawah murni "
                            "berasal dari algoritmanya.",
                            style={"fontSize": "12px", "color": INK, "margin": "0 0 12px"},
                        ),
                        why(WHY_METHODS, "Kenapa empat metode ini? (klik)"),
                        _cluster_feats_details("shared"),
                        why(WHY_PCA, "Berapa komponen PCA yang dipakai dan kenapa? (klik)"),
                    ],
                ),
                _clustering_comparison_panel(),
                panel(
                    "Pilih metode untuk dilihat grafiknya",
                    [
                        dcc.RadioItems(
                            id="cluster-method",
                            value="kmeans",
                            options=[
                                {
                                    "label": " K-Means (utama, seluruh 99.995 aplikasi)",
                                    "value": "kmeans",
                                },
                                {
                                    "label": " DBSCAN (berbasis densitas, sampel 20.000 baris)",
                                    "value": "dbscan",
                                },
                                {
                                    "label": " Hierarchical (Ward linkage, sampel 4.000 baris)",
                                    "value": "hierarchical",
                                },
                                {
                                    "label": " CLARANS (k-medoids, sampel 4.000 baris)",
                                    "value": "clarans",
                                },
                            ],
                            inline=True,
                            style={"fontSize": "12px"},
                        ),
                    ],
                    sub="Grafik di bawah mengikuti metode yang dipilih di sini.",
                ),
                html.Div(id="segments-method-panel"),
            ]
        )

    if tab == "rules":
        n_all = len(rules_all) if rules_all is not None else 0
        n_biz = len(rules) if rules is not None else 0
        return html.Div(
            [
                panel(
                    "Mengapa hanya 11 yang relevan bisnis?",
                    [
                        html.P(
                            f"{n_all} aturan kandidat lolos ambang support/confidence/lift, tetapi {n_all - n_biz} "
                            'merupakan pengulangan trivial, mis. menambahkan "White" atau "First_Lien" ke aturan DTI>60% '
                            "mengubah confidence-nya kurang dari 2 poin. Aturan seperti itu dipangkas oleh improvement "
                            "filter: sebuah aturan hanya dihitung bila mengungguli sub-rule terbaiknya minimal 2 poin "
                            "persentase. Bagian bisnis di bawah menampilkan yang lolos filter itu; daftar kandidat lengkap, "
                            "berikut improvement filter-nya, ada lebih jauh ke bawah bagi yang ingin melihat semua "
                            "temuan miner.",
                            style={
                                "fontSize": "12px",
                                "color": INK,
                                "margin": "0 0 12px",
                            },
                        ),
                        why(WHY_RULE_PRUNING),
                        _rule_features_details(),
                        why(WHY_RULE_FEATS, "Kenapa fitur-fitur itu yang dipakai? (klik)"),
                    ],
                ),
                panel(
                    "Filter",
                    [
                        dcc.RadioItems(
                            id="net-outcome",
                            value="All",
                            options=[
                                {
                                    "label": " Semua (Denied + Originated)",
                                    "value": "All",
                                },
                                {
                                    "label": " Mengapa aplikasi DITOLAK",
                                    "value": "Denied",
                                },
                                {
                                    "label": " Mengapa aplikasi DISETUJUI",
                                    "value": "Originated",
                                },
                            ],
                            inline=True,
                            style={"fontSize": "12px", "marginBottom": "12px"},
                        ),
                        html.Label(
                            "Lift minimum",
                            style={
                                "fontSize": "12px",
                                "color": MUTE,
                                "fontWeight": "600",
                            },
                        ),
                        dcc.Slider(
                            id="rules-min-lift",
                            min=1.0,
                            max=4.5,
                            step=0.1,
                            value=1.0,
                            marks={i: str(i) for i in [1, 2, 3, 4]},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    sub="Berlaku untuk kedua bagian di bawah - keduanya mengikuti tampilan dan ambang lift yang sama.",
                ),
                panel("Kenapa ambangnya segini?", [why(WHY_RULE_THRESHOLD)]),
                panel(
                    "Aturan relevan bisnis",
                    [html.Div(id="rules-table-container")],
                ),
                html.Div(
                    [
                        html.Div(
                            panel("Rules Landscape", [graph("rules-scatter")]),
                            style={"flex": "1", "minWidth": "360px"},
                        ),
                        html.Div(
                            panel("Rules Network", [graph("rule-network")]),
                            style={"flex": "1", "minWidth": "360px"},
                        ),
                    ],
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                ),
                panel(
                    "Semua aturan kandidat",
                    [html.Div(id="rules-all-table-container")],
                    sub="Setiap aturan yang lolos ambang mining (lift > 1.2, confidence >= 55%)",
                ),
            ]
        )

    if tab == "anomalies":
        n_hc = (
            int((scatter["anomaly_votes"] >= 3).sum())
            if scatter is not None and "anomaly_votes" in scatter
            else 0
        )
        tax_opts = [{"label": v, "value": k} for k, v in NUMERIC_AXES.items()]
        n_global = n_local = n_both = n_normal = 0
        if outlier_tax_summary is not None:
            vc_full = dict(
                zip(outlier_tax_summary["category"], outlier_tax_summary["n"])
            )
            n_global = int(vc_full.get("Global outlier", 0))
            n_local = int(vc_full.get("Contextual/local outlier", 0))
            n_both = int(vc_full.get("Both (global + contextual)", 0))
            n_normal = int(vc_full.get("Normal", 0))
        n_total = max(n_global + n_local + n_both + n_normal, 1)
        return html.Div(
            [
                panel(
                    "Cara membaca ini",
                    [
                        html.P(
                            f"Empat detektor (IQR, Z-score, Isolation Forest, LOF) ditambah DBSCAN-noise saling "
                            f"memberi suara pada tiap aplikasi; {n_hc:,} adalah anomali confidence tinggi (3+ metode "
                            f"sepakat). 15 yang paling ekstrem ditriase manual menjadi verdict beserta bukti, bukan "
                            f'sekadar ditandai "aneh".',
                            style={
                                "fontSize": "12px",
                                "color": INK,
                                "margin": "0 0 12px",
                            },
                        ),
                        why(WHY_DETECTORS),
                    ],
                ),
                panel(
                    "Taksonomi outlier: global, kontekstual, dan kolektif",
                    [
                        html.P(
                            "Teori data-mining membagi outlier menjadi tiga jenis. Kelima detektor pipeline ini "
                            "terbagi rapi ke dalam dua di antaranya, dan itu sendiri temuan yang layak disajikan: "
                            "kedua filosofi deteksi menangkap rekaman yang jauh berbeda.",
                            style={
                                "fontSize": "12px",
                                "color": INK,
                                "marginBottom": "14px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "OUTLIER GLOBAL",
                                            style={
                                                "fontWeight": "800",
                                                "color": TAXONOMY_COLOR[
                                                    "Global outlier"
                                                ],
                                                "fontSize": "12px",
                                                "letterSpacing": "0.5px",
                                            },
                                        ),
                                        html.Div(
                                            f"{n_global:,}",
                                            style={
                                                "fontSize": "24px",
                                                "fontWeight": "800",
                                                "color": NAVY,
                                                "margin": "4px 0",
                                            },
                                        ),
                                        html.Div(
                                            f"{n_global/n_total*100:.1f}% dari aplikasi",
                                            style={
                                                "fontSize": "11px",
                                                "color": MUTE,
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            "Terdeteksi oleh IQR, Z-score, dan Isolation Forest: rekaman yang menyimpang dari "
                                            "distribusi SELURUH dataset, tanpa syarat, misalnya property value $130 juta. "
                                            "Insight bisnis: 10% portfolio terlalu banyak untuk ditinjau manual, jadi "
                                            "pakai ini sebagai lapis penyaring awal, bukan antrean kerja.",
                                            style={
                                                "fontSize": "11.5px",
                                                "color": INK,
                                                "lineHeight": "1.5",
                                            },
                                        ),
                                    ],
                                    className="hmda-card",
                                    style={
                                        "flex": "1",
                                        "minWidth": "230px",
                                        "background": CARD,
                                        "borderRadius": "14px",
                                        "padding": "14px 16px",
                                        "boxShadow": SOFT_SHADOW,
                                        "border": f"1px solid {BORDER}",
                                        "borderTop": f"3px solid {TAXONOMY_COLOR['Global outlier']}",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "OUTLIER KONTEKSTUAL / LOKAL",
                                            style={
                                                "fontWeight": "800",
                                                "color": TAXONOMY_COLOR[
                                                    "Contextual/local outlier"
                                                ],
                                                "fontSize": "12px",
                                                "letterSpacing": "0.5px",
                                            },
                                        ),
                                        html.Div(
                                            f"{n_local:,}",
                                            style={
                                                "fontSize": "24px",
                                                "fontWeight": "800",
                                                "color": NAVY,
                                                "margin": "4px 0",
                                            },
                                        ),
                                        html.Div(
                                            f"{n_local/n_total*100:.1f}% dari aplikasi",
                                            style={
                                                "fontSize": "11px",
                                                "color": MUTE,
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            "Terdeteksi oleh LOF dan DBSCAN-noise: rekaman yang tampak biasa pada setiap ambang "
                                            "fitur tunggal, tetapi berada di lingkungan yang jarang relatif terhadap "
                                            "aplikasi serupa. Insight bisnis: justru kelompok ini yang lolos aturan "
                                            "ambang konvensional, jadi paling berharga untuk tinjauan underwriting.",
                                            style={
                                                "fontSize": "11.5px",
                                                "color": INK,
                                                "lineHeight": "1.5",
                                            },
                                        ),
                                    ],
                                    className="hmda-card",
                                    style={
                                        "flex": "1",
                                        "minWidth": "230px",
                                        "background": CARD,
                                        "borderRadius": "14px",
                                        "padding": "14px 16px",
                                        "boxShadow": SOFT_SHADOW,
                                        "border": f"1px solid {BORDER}",
                                        "borderTop": f"3px solid {TAXONOMY_COLOR['Contextual/local outlier']}",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "KEDUANYA",
                                            style={
                                                "fontWeight": "800",
                                                "color": TAXONOMY_COLOR[
                                                    "Both (global + contextual)"
                                                ],
                                                "fontSize": "12px",
                                                "letterSpacing": "0.5px",
                                            },
                                        ),
                                        html.Div(
                                            f"{n_both:,}",
                                            style={
                                                "fontSize": "24px",
                                                "fontWeight": "800",
                                                "color": NAVY,
                                                "margin": "4px 0",
                                            },
                                        ),
                                        html.Div(
                                            f"{n_both/n_total*100:.1f}% dari aplikasi",
                                            style={
                                                "fontSize": "11px",
                                                "color": MUTE,
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            "Ditandai oleh setidaknya satu metode global DAN satu kontekstual/lokal, "
                                            "jadi anomali dengan keyakinan tertinggi. Insight bisnis: 476 rekaman itu "
                                            "jumlah yang realistis untuk antrean tinjauan manual, dan inilah kumpulan "
                                            "tempat triase 15 teratas diambil.",
                                            style={
                                                "fontSize": "11.5px",
                                                "color": INK,
                                                "lineHeight": "1.5",
                                            },
                                        ),
                                    ],
                                    className="hmda-card",
                                    style={
                                        "flex": "1",
                                        "minWidth": "230px",
                                        "background": CARD,
                                        "borderRadius": "14px",
                                        "padding": "14px 16px",
                                        "boxShadow": SOFT_SHADOW,
                                        "border": f"1px solid {BORDER}",
                                        "borderTop": f"3px solid {TAXONOMY_COLOR['Both (global + contextual)']}",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "gap": "14px",
                                "flexWrap": "wrap",
                                "marginBottom": "6px",
                            },
                        ),
                        why(WHY_TAXONOMY),
                    ],
                ),
                panel(
                    "Jelajahi taksonomi",
                    [
                        _axis_picker("tax-xcol", "tax-ycol", tax_opts),
                        graph("tax-scatter"),
                    ],
                    sub="Titik abu-abu adalah populasi latar biasa (sampel acuan 4.000 baris). "
                    "Titik berwarna adalah setiap rekaman yang ditandai, diposisikan menurut dua atribut pilihan Anda - "
                    "perhatikan outlier global mengumpul di ekstrem magnitudo, sedangkan outlier kontekstual/lokal "
                    "berada di dalam rentang yang tampak biasa.",
                ),
                panel(
                    "Contoh bernama",
                    [_outlier_taxonomy_examples()],
                    sub="3 rekaman paling ekstrem di tiap kategori non-normal, menurut skor Isolation Forest.",
                ),
                panel(
                    "Di mana outlier berada (income vs. loan amount)",
                    [graph(fig_anomaly_scatter())],
                    sub="Kanan-atas = loan besar relatif terhadap income yang dilaporkan.",
                ),
                html.Div(
                    [
                        html.Div(
                            panel("Skor anomali", [graph(fig_iso_hist())]),
                            style={"flex": "1", "minWidth": "340px"},
                        ),
                        html.Div(
                            panel("Kesepakatan metode", [graph(fig_vote_breakdown())]),
                            style={"flex": "1", "minWidth": "340px"},
                        ),
                    ],
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                ),
                _collective_groups_panel(),
                panel(
                    "Rekaman paling ekstrem: ditriase dengan bukti",
                    [_anomaly_table(), why(WHY_TRIAGE)],
                    sub="Jenis verdict: RARE BUT VALID (rekaman ekstrem yang sah) · RISK SIGNAL (tidak biasa tetapi "
                    "mungkin) · DATA ERROR (nilai mustahil) · MANUAL REVIEW (diperiksa manual). "
                    "Insight bisnis: seluruh 15 teratas terbukti sah, jadi ensemble ini tidak boleh "
                    "dipakai sebagai aturan hapus otomatis karena akan membuang aplikasi jumbo yang benar.",
                ),
            ]
        )

    if tab == "whatif":
        controls = [
            render_control(
                field, label, kind, order, default, "wi", wrap_id=FIN_WRAP.get(field)
            )
            for field, label, kind, order, default in WHATIF_FIELDS
        ] + [
            render_control(
                field, label, kind, order, default, "ctx", wrap_id=FIN_WRAP.get(field)
            )
            for field, label, kind, order, default in CONTEXT_FIELDS
        ]
        return html.Div(
            [
                panel(
                    "Basis finansial",
                    [
                        dcc.RadioItems(
                            id="wi-fin-mode",
                            value="dti",
                            options=[
                                {"label": " Pakai DTI", "value": "dti"},
                                {
                                    "label": " Pakai income + besar loan + durasi",
                                    "value": "income_loan",
                                },
                                {
                                    "label": " Semua (sampel bisa jadi kecil)",
                                    "value": "all",
                                },
                            ],
                            inline=True,
                            style={"fontSize": "12px"},
                        )
                    ],
                    sub="Ini penjaga ukuran sampel, bukan klaim bahwa DTI dan fitur ukuran itu "
                    "redundan. Uji rekonstruksi di Fase 1 justru menunjukkan sebaliknya: DTI cuma "
                    "10% dijelaskan oleh income dan besar loan, jadi keduanya membawa informasi "
                    "berbeda. Masalahnya di sini murni praktis, karena tiap filter tambahan "
                    "mempersempit pencarian sampai tersisa segelintir aplikasi dan angkanya jadi "
                    "tidak stabil. Pilih satu basis dulu, lalu pakai mode Semua kalau memang "
                    "butuh menggabungkan keduanya.",
                ),
                panel(
                    "Bangun profil pemohon",
                    [
                        html.Div(
                            controls,
                            style={
                                "display": "flex",
                                "gap": "20px",
                                "flexWrap": "wrap",
                            },
                        ),
                        html.Div(
                            "Semua field default ke tidak ditentukan. Hasil di bawah adalah tingkat "
                            "persetujuan historis nyata di antara aplikasi yang cocok dengan semua "
                            "atribut terpilih sekaligus, bukan model prediktif.",
                            style={
                                "fontSize": "11px",
                                "color": MUTE,
                                "marginTop": "10px",
                            },
                        ),
                    ],
                ),
                html.Div(id="whatif-result"),
            ]
        )

    if tab == "fairness":
        return html.Div(
            [
                panel(
                    "Persetujuan menurut ras × lingkungan",
                    [graph(fig_disparity())],
                    sub="Bandingkan tingkat persetujuan antar kelompok demografis dan tingkat minoritas tract.",
                ),
                panel(
                    "Apakah selisih bertahan setelah mengontrol DTI?",
                    [graph(fig_dti_geo_gap())],
                    sub="Perbandingan yang sama, dipisah menurut prediktor penolakan terkuat dalam dataset.",
                ),
                panel(
                    "Persetujuan menurut gender, dikontrol DTI",
                    [graph(fig_gender_gap())],
                    sub="Perlakuan yang sama seperti analisis tract: selisih mentah baru bermakna "
                    "setelah beban utang yang setara dibandingkan dengan yang setara.",
                ),
            ]
        )
    return html.Div()


# ============================================================ segments sub-callback
def _axis_picker(xcol_id, ycol_id, opts, default_x="income", default_y="loan_amount"):
    return html.Div(
        [
            html.Div(
                [
                    html.Label("X axis", style={"fontSize": "12px", "color": MUTE}),
                    dcc.Dropdown(
                        id=xcol_id, options=opts, value=default_x, clearable=False
                    ),
                ],
                style={"width": "240px"},
            ),
            html.Div(
                [
                    html.Label("Y axis", style={"fontSize": "12px", "color": MUTE}),
                    dcc.Dropdown(
                        id=ycol_id, options=opts, value=default_y, clearable=False
                    ),
                ],
                style={"width": "240px"},
            ),
        ],
        style={"display": "flex", "gap": "16px", "marginBottom": "8px"},
    )


@app.callback(
    Output("segments-method-panel", "children"), Input("cluster-method", "value")
)
@lru_cache(maxsize=4)
def _cb_segments(method):
    opts = [{"label": v, "value": k} for k, v in NUMERIC_AXES.items()]

    if method == "dbscan":
        n_noise = (
            int((dbscan_scatter["dbscan_cluster"] == -1).sum())
            if dbscan_scatter is not None
            else 0
        )
        n_total = len(dbscan_scatter) if dbscan_scatter is not None else 0
        pct = (n_noise / n_total * 100) if n_total else 0
        return html.Div(
            [
                panel(
                    "Ukuran cluster DBSCAN",
                    [graph(fig_dbscan_sizes())],
                    sub=f"eps dipilih dari knee k-distance (min_samples = 2 x jumlah fitur): 17 cluster densitas "
                    f"plus noise. {n_noise:,} dari {n_total:,} aplikasi ({pct:.1f}%) tidak "
                    "termasuk region padat mana pun - titik noise inilah input yang di-cross-reference "
                    "oleh ensemble anomali Fase 4 (lihat tab Anomali).",
                ),
                panel(
                    "Jelajahi hasil DBSCAN",
                    [
                        _axis_picker("dbscan-xcol", "dbscan-ycol", opts),
                        graph("dbscan-scatter"),
                    ],
                    sub="Diwarnai hanya noise vs. clustered (bukan per id cluster) - dengan 17 cluster, "
                    "scatter berwarna per-cluster akan melampaui batas warna kategorikal yang tervalidasi.",
                ),
            ]
        )

    if method == "hierarchical":
        return html.Div(
            [
                panel(
                    "Kesepakatan K-Means vs. CLARANS vs. Hierarchical",
                    [graph(fig_method_comparison())],
                    sub="Ward linkage dipotong pada K=7 (K terpilih dari silhouette di bagian 2.1). Adjusted Rand "
                    "Index vs. K-Means = 0,907 pada sampel 4.000 baris yang sama: kesepakatan kuat, menegaskan "
                    "struktur segmen bukan artefak K-Means.",
                ),
                panel(
                    "Jelajahi segmen Hierarchical",
                    [
                        _axis_picker("hier-xcol", "hier-ycol", opts),
                        graph("hier-scatter"),
                    ],
                ),
            ]
        )

    if method == "clarans":
        return html.Div(
            [
                panel(
                    "Kesepakatan K-Means vs. CLARANS",
                    [graph(fig_method_comparison())],
                    sub="CLARANS (k-medoids) memilih aplikasi nyata sebagai pusat cluster alih-alih rata-rata "
                    "sintetis, robust terhadap outlier ekstrem pada tab Anomali. Adjusted Rand Index "
                    "vs. K-Means = 0,711, vs. Ward hierarchical = 0,701: kesepakatan sedang-ke-kuat, sesuai "
                    "harapan dari dua tujuan optimasi berbeda pada data yang sama.",
                ),
                panel(
                    "Jelajahi segmen CLARANS",
                    [
                        _axis_picker("clarans-xcol", "clarans-ycol", opts),
                        graph("clarans-scatter"),
                    ],
                ),
            ]
        )
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        panel("Ukuran segmen", [graph(fig_cluster_sizes())]),
                        style={"flex": "1", "minWidth": "340px"},
                    ),
                    html.Div(
                        panel(
                            "Tingkat persetujuan per segmen",
                            [graph(fig_approval_by_cluster())],
                        ),
                        style={"flex": "1", "minWidth": "340px"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
            panel(
                "Jelajahi segmen",
                [
                    _axis_picker("xcol", "ycol", opts),
                    graph("cluster-scatter"),
                ],
                sub="Setiap titik adalah satu aplikasi kredit, diwarnai per segmen. Pilih dua atribut apa pun.",
            ),
            panel(
                "Karakteristik segmen: makna tiap cluster bagi bisnis",
                [_profiles_cards()],
            ),
        ]
    )


@app.callback(
    Output("cluster-scatter", "figure"), Input("xcol", "value"), Input("ycol", "value")
)
def _cb_scatter(x, y):
    return fig_cluster_scatter(x, y)


@app.callback(
    Output("clarans-scatter", "figure"),
    Input("clarans-xcol", "value"),
    Input("clarans-ycol", "value"),
)
def _cb_clarans_scatter(x, y):
    return fig_clarans_scatter(x, y)


@app.callback(
    Output("dbscan-scatter", "figure"),
    Input("dbscan-xcol", "value"),
    Input("dbscan-ycol", "value"),
)
def _cb_dbscan_scatter(x, y):
    return fig_dbscan_scatter(x, y)


@app.callback(
    Output("hier-scatter", "figure"),
    Input("hier-xcol", "value"),
    Input("hier-ycol", "value"),
)
def _cb_hier_scatter(x, y):
    return fig_hierarchical_scatter(x, y)


@app.callback(
    Output("tax-scatter", "figure"),
    Input("tax-xcol", "value"),
    Input("tax-ycol", "value"),
)
def _cb_tax_scatter(x, y):
    return fig_outlier_taxonomy(x, y)


# ============================================================ geography callbacks
@app.callback(Output("geo-map", "figure"), Input("geo-metric", "value"))
def _cb_geo_map(metric):
    return fig_state_map(metric)


@app.callback(
    Output("geo-state-detail", "children"),
    Input("geo-map", "clickData"),
    prevent_initial_call=True,
)
def _cb_geo_map_click(click_data):
    if click_data and click_data.get("points"):
        loc = click_data["points"][0].get("location")
        if loc:
            return _geo_state_detail_children(loc)
    return no_update


# ============================================================ rules sub-callbacks
@app.callback(
    Output("rules-table-container", "children"),
    Input("net-outcome", "value"),
    Input("rules-min-lift", "value"),
)
def _cb_rules_table(outcome, min_lift):
    return _table(fig_rules_table_df(rules, outcome, min_lift))


@app.callback(
    Output("rules-scatter", "figure"),
    Input("net-outcome", "value"),
    Input("rules-min-lift", "value"),
)
def _cb_rules_scatter(outcome, min_lift):
    return fig_rules_scatter(rules, outcome, min_lift)


@app.callback(
    Output("rule-network", "figure"),
    Input("net-outcome", "value"),
    Input("rules-min-lift", "value"),
)
def _cb_net(outcome, min_lift):
    return fig_rule_network(rules, outcome, min_lift)


@app.callback(
    Output("rules-all-table-container", "children"),
    Input("net-outcome", "value"),
    Input("rules-min-lift", "value"),
)
def _cb_rules_all_table(outcome, min_lift):
    return _table(fig_rules_table_df(rules_all, outcome, min_lift))


# ============================================================ what-if callback
@app.callback(
    [
        Output("wrap-dti", "style"),
        Output("wrap-income", "style"),
        Output("wrap-loan", "style"),
        Output("wrap-term", "style"),
        Output("wi-debt_to_income_ratio", "value"),
        Output("wi-income_band", "value"),
        Output("wi-loan_amount_band", "value"),
        Output("ctx-term_band", "value"),
    ],
    Input("wi-fin-mode", "value"),
)
def _cb_fin_mode(mode):
    """Show one financial basis at a time and clear the hidden side so a stale
    selection cannot keep filtering invisibly."""
    shown = dict(CONTROL_WRAP_STYLE)
    hidden = {"display": "none"}
    if mode == "all":
        return (shown, shown, shown, shown, no_update, no_update, no_update, no_update)
    if mode == "income_loan":
        return hidden, shown, shown, shown, "", no_update, no_update, no_update
    return shown, hidden, hidden, hidden, no_update, "", "", ""


@app.callback(
    Output("whatif-result", "children"),
    [Input(f"wi-{field}", "value") for field, _, _, _, _ in WHATIF_FIELDS]
    + [Input(f"ctx-{field}", "value") for field, _, _, _, _ in CONTEXT_FIELDS],
)
def _cb_whatif(*values):
    n_wi = len(WHATIF_FIELDS)
    selected = []
    for (field, label, kind, order, default), raw in zip(WHATIF_FIELDS, values[:n_wi]):
        selected.append((field, label, decode_control(kind, order, default, raw)))
    for (field, label, kind, order, default), raw in zip(CONTEXT_FIELDS, values[n_wi:]):
        selected.append((field, label, decode_control(kind, order, default, raw)))

    appr, n, active = combined_match(selected)
    if not active:
        note = "Pilih atribut apa pun di atas untuk melihat tingkat persetujuan historis profil itu."
    elif n == 0:
        note = (
            f"Tidak ada aplikasi historis yang cocok dengan semua {len(active)} atribut terpilih "
            "sekaligus. Kombinasinya terlalu spesifik atau memang tidak ada di sampel ini, jadi "
            "yang ditampilkan tingkat persetujuan portfolio keseluruhan."
        )
    else:
        detail = "; ".join(f"{label}={vlabel(value)}" for label, value in active)
        note = f"{n:,} aplikasi historis cocok dengan semua {len(active)} atribut terpilih. {detail}."
        if n < 30:
            note += f" Hati-hati, cuma {n:,} aplikasi yang cocok, jadi angkanya belum stabil."
    outcome_label = "Tingkat persetujuan historis gabungan"

    def _base_delta(rate, matched, active_filters):
        """Show the gap against the portfolio base rate.

        Without this, a profile whose rate lands near the 77% base reads as if the filter
        never ran, when in fact the attribute simply has little effect on approval.
        """
        if not active_filters or matched == 0 or not np.isfinite(BASE_APPROVAL):
            return html.Div()
        diff = rate - BASE_APPROVAL
        if abs(diff) < 1:
            color, verdict = MUTE, "praktis sama dengan rata-rata portfolio"
        elif diff > 0:
            color, verdict = GREEN, "lebih tinggi dari rata-rata portfolio"
        else:
            color, verdict = RED, "lebih rendah dari rata-rata portfolio"
        return html.Div(
            [
                html.Span(
                    f"{diff:+.1f} poin ",
                    style={"fontWeight": "800", "fontSize": "17px", "color": color},
                ),
                html.Span(
                    f"{verdict} ({BASE_APPROVAL:.1f}%)",
                    style={"fontSize": "12.5px", "color": MUTE},
                ),
            ]
        )

    return panel(
        "Hasil",
        [
            html.Div(
                [
                    html.Div(
                        approval_meter(appr, outcome_label),
                        style={"flex": "1", "minWidth": "280px"},
                    ),
                    html.Div(
                        [
                            _base_delta(appr, n, active),
                            html.Div(
                                note,
                                style={
                                    "fontSize": "13px",
                                    "color": INK,
                                    "lineHeight": "1.6",
                                },
                            ),
                        ],
                        style={
                            "flex": "2",
                            "minWidth": "280px",
                            "display": "flex",
                            "flexDirection": "column",
                            "justifyContent": "center",
                            "gap": "8px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "20px",
                    "flexWrap": "wrap",
                    "alignItems": "center",
                },
            ),
        ],
    )


# ============================================================ run
if __name__ == "__main__":
    app.run(
        host=os.getenv("HMDA_HOST", "127.0.0.1"),
        port=int(os.getenv("HMDA_PORT", "8050")),
        debug=os.getenv("HMDA_DEBUG", "0") == "1",
    )
