# HMDA 2022 — Knowledge Discovery Project

A full KDD pipeline over a 100,000-record HMDA 2022 mortgage-application
sample: preprocessing, feature selection, clustering (K-Means, DBSCAN, Ward
hierarchical, CLARANS), association-rule mining, multi-method anomaly
detection, and two interactive dashboards.

## Directory layout

```
pipeline/            Importable Phase 1-5 pipeline logic (single source of truth)
  config.py             shared constants: column roles, labels, sentinel whitelist,
                         bin specs, leakage guards, DTI grouping
  phase1_preprocessing.py   load, clean, harmonize sentinels, dedupe, impute,
                         bin, score/select features
  phase2_clustering.py      engineered flags, K-Means/DBSCAN/hierarchical/CLARANS,
                         cluster profiling + naming
  phase3_association_rules.py  Apriori, rule extraction, significance testing,
                         Bayardo improvement-filter pruning, geography x DTI crosstab
  phase4_anomaly_detection.py  IQR/Z-score, Isolation Forest, LOF, ensemble vote,
                         evidence-based triage
  phase5_reporting.py       dashboard aggregate export, standalone HTML dashboard,
                         plain-language report generator

notebooks/
  HMDA.ipynb            The primary deliverable. All narrative markdown (Indonesian),
                         plots, and validation/assert cells live here; the actual
                         data-transformation code is imported from pipeline/.

dashboard/
  dashboard_app.py       Full-featured Dash app (Executive Summary, Geography,
                         Segments, Rules, Anomalies, What-If, Fairness tabs).
  build_data.py           Rebuilds every CSV dashboard_app.py reads, by running
                         the Phase 1-5 pipeline via pipeline/. Run this after any
                         change to the pipeline/ modules.

data/
  raw/                  Cached copy of the source CSV (offline reproducibility)
  interim/              Phase 1 output: hmda_clean.csv, hmda_approve_deny.csv,
                         hmda_denials.csv
  processed/            Phase 2-4 outputs (cluster assignments/profiles, decision
                         rules, anomaly flags/triage) + all dash_*.csv aggregates

results/
  figures/              All Phase 1-4 PNG plots
  tables/               feature_ranking_combined.csv, p1_feature_selection_audit.csv,
                         p3_rule_significance.csv, p3_interpreted_rules.csv,
                         HMDA_Rubric_Audit.csv

reports/
  Preprocessing_Report.md          Submitted report (hand-written)
  Knowledge_Discovery_Report.md    Submitted report (hand-written)
  *_generated.md                   Notebook's auto-generated short-form versions of
                                   the same two reports (written by cell 124; never
                                   overwrites the hand-written files above)
  HMDA_Interactive_Dashboard.html   Standalone zero-setup dashboard (no Dash server
                                   needed — open directly in a browser)
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (Linux/Mac)
pip install -r requirements.txt
```

`requirements.txt` covers both the notebook's full scientific-computing stack
(numpy/pandas/scipy/scikit-learn/mlxtend/matplotlib/seaborn + Jupyter) and the
Dash app's runtime (dash/plotly/Flask).

## Running the notebook

Open `notebooks/HMDA.ipynb` in Jupyter/VS Code and run all cells top to
bottom. Cell 1 adds the project root to `sys.path` so `from pipeline import
...` resolves regardless of which directory Jupyter's working directory is
set to, as long as the notebook itself stays in `notebooks/`.

The notebook fetches its source data from a Hugging Face URL on first run
(`pipeline.config.HF_URL`); no manual download is required, though a cached
copy also lives at `data/raw/hmda_sample.csv`.

## Running the dashboards

**Standalone HTML dashboard** — no server needed:
```
Open reports/HMDA_Interactive_Dashboard.html directly in a browser.
```
Regenerate it by re-running the notebook's Phase 5 cells, or via
`dashboard/build_data.py` (see below).

**Full Dash app**:
```
python dashboard/dashboard_app.py
```
Then open the printed local URL in a browser. `dashboard_app.py` reads CSVs
from `data/processed/` first, falling back to `data/interim/`.

**Rebuilding dashboard data** after changing pipeline logic:
```
python dashboard/build_data.py
```
This re-runs the full Phase 1-5 pipeline and rewrites every CSV both
dashboards depend on, in one reviewable script (replacing what used to be 9
ad hoc, unversioned scratch scripts).

## Notes

- All 5 phases' data-transformation logic lives in `pipeline/`; the notebook
  imports and calls it. Narrative markdown, plots, and validation/assert
  cells (partition checks, rogue-sentinel audit, rubric audit, etc.) stay
  inline in the notebook.
- `RANDOM_STATE = 42` everywhere for reproducibility (`pipeline/config.py`).
- The rubric-audit cell (notebook cell 125) intentionally was not moved into
  `pipeline/`: it introspects the notebook's own `globals()` to confirm each
  rubric criterion was actually executed, which is inherently notebook-only.
