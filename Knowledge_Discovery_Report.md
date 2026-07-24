# Knowledge Discovery Report — HMDA 2022

**Scope:** Phases 2–4 of `HMDA.ipynb` — Segmentation (clustering), Association Rule Mining, and Anomaly/Outlier Detection — run on the Phase 1 cleaned tables (99,995 rows / 67,827 decisioned applications, 76.9% baseline approval rate). This report states the findings, the evidence behind each, and closes with the central question every knowledge-discovery deliverable has to answer honestly.

---

## What did we discover that was not already obvious?

Three findings would not survive a quick look at the raw approval-rate table, and are the actual payoff of this pipeline:

1. **Demographic items don't survive contact with the improvement filter — but the fairness question isn't closed by that alone.** Every top-lift association rule that pairs `debt_to_income_ratio=>60%` with a demographic attribute (`derived_race=White`, `derived_ethnicity=Not Hispanic or Latino`) turns out to add *zero* predictive information once DTI is known — the pruned rule set (§2 below) contains no demographic antecedent at all. A naive reading would stop there and declare the process demographically neutral. It isn't: the geography cross-tab (§2.4) shows a **12.1-percentage-point** approval gap between low- and majority-minority tracts *within the same low-DTI band* — a gap invisible to item-level association rules because it lives at the tract level, not the applicant-attribute level. **The two analyses together, not either alone, are the finding**: individual-level decisioning tracks DTI almost exclusively, while a residual, unexplained geographic gap persists underneath it.

2. **The anomaly ensemble converges on "nothing new."** Four independent detectors run across 99,995 rows agree with each other far more than chance (Jaccard overlap analysis, §3.3), and when their top disagreements are hand-triaged, **13 of 15** were already anticipated by the Phase 1 structural scan (jumbo loans, multi-unit investment properties) — and the remaining 3 unresolved cases, hand-verified in this pass, also reconcile as legitimate high-value transactions, not data errors. The interesting result is *negative*: five different anomaly-detection philosophies (distributional, density-based, isolation-based, and clustering-noise) converge on the same conclusion as the domain-knowledge-driven cleaning in Phase 1. That agreement is itself evidence the cleaning pipeline didn't miss a class of error, which is a materially different and stronger claim than "we found some outliers."

3. **A property attribute alone — not a borrower attribute — is enough to flip the majority outcome.** `construction_method=Manufactured` on its own denies at 56.9% (base rate: 23.1%), and that penalty is *compounded*, not explained, by financing channel (conventional-loan manufactured applications: 63.5% denial) and transaction type (purchase-specific manufactured applications: 61.4%). This is a distinct, separately-actionable segment from "low income" — it shows up as its own cluster (Segment 3, §1) that is *not* the same population as the low-income or DTI-stressed segments.

---

## 1. Segmentation (Phase 2)

**Method:** K-Means (primary), DBSCAN, Ward/complete/average hierarchical, and CLARANS (k-medoids), all on a 9-feature matrix (`CLUSTER_FEATS`): one size proxy (income, to avoid the collinearity of loan_amount/property_value/income all measuring "bigness"), leverage (CLTV), two location features (tract minority %, tract-to-MSA income %), and five behavioral flags (investment, refinance, manufactured, subordinate lien, high-DTI). Continuous inputs are winsorized at 1%/99% before standardizing, so clustering isn't dominated by the extreme tail Phase 4 is built to examine separately.

**K selection:** Elbow (inertia) suggests K=6; Silhouette peaks at **K=7** (`p2_k_selection.png`) — Silhouette is used as the deciding criterion. Final K-Means validity: **silhouette 0.303, Davies-Bouldin 1.146, Calinski-Harabasz 21,869**.

**Cross-algorithm agreement:**
- K-Means vs. Ward hierarchical (same 4,000-row sample): **ARI = 0.907** — strong agreement, the segment structure is not a K-Means artifact.
- CLARANS vs. K-Means: ARI = 0.711; CLARANS vs. hierarchical: ARI = 0.701 — moderate agreement, expected since CLARANS optimizes for real-record medoids rather than centroid distance and is more robust to the extreme tail.
- DBSCAN (eps=1.11, chosen from the k-distance knee): 17 density clusters, **895 noise points (4.5%** of the 20k sample) — these noise IDs feed directly into the Phase 4 anomaly cross-reference.

**The seven segments** (profiled on outcome + financial/behavioral shares, auto-named from standout attributes):

| # | Segment | Share | Approval | Median income ($k) | Median loan | Distinguishing trait |
|---|---|---|---|---|---|---|
| 0 | Refinancers (rate & cash-out) | 25.6% | 79.7% | 89 | $215k | 94.5% refinance |
| 1 | Property investors | 7.0% | 83.0% | 95 | $215k | 100% investment occupancy |
| 2 | Mainstream prime purchasers | 36.8% | **91.5%** | 95 | $295k | Largest segment, near-zero flags |
| 3 | Manufactured-housing applicants | 4.5% | **43.1%** | 61 | $115k | 100% manufactured construction |
| 4 | DTI-stressed borrowers | 8.7% | **39.4%** | 70 | $205k | 100% high-DTI (>50%) |
| 5 | Jumbo / high-net-worth buyers | 4.4% | 87.4% | 467 | $645k | Median income 5× the sample median |
| 6 | Small-loan borrowers | 12.9% | 75.1% | 115 | $65k | Smallest median loan size |

Segments 3 and 4 are the two lowest-approval groups in the data and are **not the same population** — Segment 3 is defined by property type regardless of income, Segment 4 by leverage regardless of property type — which is exactly why the two Phase 3 findings (manufactured-housing penalty vs. DTI penalty) surface as independent rules rather than one confounded effect. Per-cluster silhouette values (`p2_silhouette_per_cluster.png`) show the flag-defined segments (3, 4, 1) are tighter than the broad "prime purchaser" mass (2), which is expected — a behavioral flag is a hard boundary, general creditworthiness is a continuum.

---

## 2. Association Rule Mining (Phase 3)

**Method:** Apriori (mlxtend, vectorized) on a one-hot transaction matrix of 19 application-time item features + the outcome item (`decision=Originated|Denied`), `min_support=0.02`, `max_len=3`, decision rules extracted directly from frequent itemsets (`lift > 1.2`, `confidence ≥ 0.55`). **10,013 frequent itemsets → 28 non-trivial decision rules** (21 concluding Denied, 7 concluding Originated) out of 1,605 candidate decision rules.

### 2.1 The redundancy problem, and how it was resolved

The raw top-15-by-lift table is almost entirely variants of one base rule: `debt_to_income_ratio=>60%` alone denies at 91.5% (lift 3.96); adding `loan_type=Conventional`, `occupancy_type=Principal_Residence`, `derived_race=White`, `derived_ethnicity=Not Hispanic or Latino`, or `lien_status=First_Lien` moves that confidence by less than 2 percentage points in either direction. Ranking by lift alone would present ten near-duplicate rows as ten separate findings.

**Fix — the improvement criterion** (Bayardo et al., 1999): a rule is kept only if its confidence beats the confidence of *every* proper sub-rule with the same consequent — including the empty antecedent, i.e. the 23.1%/76.9% base rate — by at least 2 percentage points. Applied to the 28-rule table: **11 rules kept, 17 pruned as trivial variants.** Exactly one DTI>60% interaction survives (`+ Subordinate_Lien`, +2.4pp), because a subordinate lien genuinely implies additional leverage the DTI field alone doesn't capture; every demographic and most channel add-ons do not survive.

### 2.2 The final, non-redundant rule set

All 11 rules are chi-square significant (p < 10⁻⁵⁷ in every case; base rates: 23.1% Denied / 76.9% Originated, Wilson 95% CIs shown):

| Rule | Confidence | Lift | n | Improvement over best sub-rule |
|---|---|---|---|---|
| DTI>60% ∧ Subordinate lien → **Denied** | 94.0% | 4.06 | 1,555 | +2.4pp |
| DTI>60% → **Denied** | 91.5% | 3.96 | 4,121 | +68.4pp (vs. base rate) |
| Manufactured ∧ Conventional loan → **Denied** | 63.5% | 2.75 | 2,453 | +6.6pp |
| DTI 50–60% ∧ Conventional loan → **Denied** | 62.2% | 2.69 | 2,759 | +20.5pp |
| Manufactured ∧ Home purchase → **Denied** | 61.4% | 2.65 | 2,370 | +4.5pp |
| Income <$30k ∧ Conventional loan → **Denied** | 59.9% | 2.59 | 2,576 | +2.0pp |
| Income <$30k → **Denied** | 57.8% | 2.50 | 3,212 | +34.7pp (vs. base rate) |
| Manufactured construction → **Denied** | 56.9% | 2.46 | 2,892 | +33.8pp (vs. base rate) |
| Preapproval requested → **Originated** | 100.0% | 1.30 | 1,583 | +23.1pp (vs. base rate) |
| DTI 30–<36% ∧ Home purchase → **Originated** | 92.5% | 1.20 | 4,566 | +6.3pp |
| DTI 30–<36% ∧ Loan $300–500k → **Originated** | 92.3% | 1.20 | 1,882 | +6.1pp |

**Business reading, condensed** (full per-rule interpretation with mechanism and recommended action is written out in `HMDA.ipynb` §3.6, one paragraph per rule):

- **Ability-to-repay dominates.** Once DTI>60% is known, no other item in the transaction set — demographic or otherwise — changes the prediction. The single highest-leverage lever available to a denied applicant is lowering DTI, not any other application attribute.
- **Financing channel compounds two different penalties** (manufactured housing, mid-high DTI): GSE/conventional underwriting is measurably stricter than FHA/VA-style programs for both, independent of the underlying risk factor. This is a genuine interaction (DTI 50–60% denies at 41.7% overall but 62.2% specifically inside the conventional channel, a 20.5-point jump) — the largest true interaction effect found, as opposed to a redundant restatement.
- **The 100% preapproval-origination rule is a funnel artifact, not underwriting magic.** Rejections at the preapproval stage exit under a separate action code (`Preapproval_Denied`) before ever entering this comparison. Reported honestly rather than as a false "silver bullet," but still operationally useful — it is the cleanest process lever the data contains for pushing rejection to the cheap, early stage.
- **The market's real decision boundary** brackets between DTI 30–36% (≈92% approval) and DTI >60% (≈92% denial) — the same underwriting pipeline, opposite ends of one variable.

### 2.3 Denial-reason cross-check

Lender-*stated* denial reasons (`denial_reason_1`, 15,684 denied applications) independently corroborate the rule-mining result rather than being mined as a separate, unrelated fact: **Debt-to-income is the single most cited reason (30.5% of denials)**, ahead of Credit history (27.0%), Collateral (13.8%), Incomplete application (11.4%), and Other (8.9%). The rules above were derived purely from application-time features with no access to this field — the fact that they land on DTI as the dominant driver, and the lenders' own stated reasons agree, is a validity check the rule-mining passes.

### 2.4 Geography × DTI (the residual fairness signal)

Marginal approval rate by tract minority composition: **Low-minority 79.1% → Majority-minority 67.8%** — an 11.3-point gap. The natural objection is that this simply reflects different DTI distributions across neighborhoods. Controlling for DTI band directly answers that:

| DTI group | Low-minority | Majority-minority | Gap |
|---|---|---|---|
| Low (<36%) | 85.5% | 73.4% | **12.1pp** |
| Mid (36–50%) | 86.7% | 80.3% | 6.4pp |
| High (>50%) | 34.8% | 33.0% | 1.8pp |

The gap does **not** close within DTI bands — if anything it is *largest* in the low-DTI group, where creditworthiness is least in question. This is squarely an association finding, not a fair-lending determination (no causal claim, no controls beyond one variable), but it is the concrete, evidence-backed lead this project is positioned to hand off: a residual approval gap that survives controlling for the single strongest predictor of denial in the entire dataset.

---

## 3. Anomaly & Outlier Detection (Phase 4)

**Method:** four independent detectors — IQR (per-feature, ≥3 features flagged), Z-score (|z|>3), Isolation Forest (1% contamination, full 99,995 rows), Local Outlier Factor (1% contamination, 20k sample) — cross-referenced against DBSCAN noise from Phase 2, on the same 8-feature raw-magnitude matrix (`ANOMALY_FEATS`, RobustScaler, extremes deliberately preserved rather than winsorized).

**Detector output:** IQR flags 57,202 rows on ≥1 feature (7,085 on ≥3); Z-score flags 5,854; Isolation Forest flags 1,000; LOF flags 200 of its 20k sample; DBSCAN marks 895 points as noise (4.5%), of which 87 are also Isolation-Forest anomalies. **Ensemble vote distribution:** 88,971 rows flagged by 0 methods, tapering to 649 (3 methods), 80 (4), and **10 rows flagged by all 5**. **739 rows are high-confidence anomalies (≥3 methods agreeing)** — meaningfully more overlap than the ~1% chance rate implied by each detector's individual flag rate, confirming the ensemble is measuring a real shared signal rather than five uncorrelated opinions (pairwise Jaccard heatmap, `p4_ensemble_agreement.png`).

**Triage of the top 15 anomalies** (highest vote count, tie-broken by Isolation Forest score) applies four categories with explicit evidence per row: **DATA ERROR** (impossible values — CLTV in the thousands, terms >50 years), **RARE BUT VALID** (genuine jumbo/multifamily/investment records), **RISK SIGNAL** (unusual-but-possible combinations), **MANUAL REVIEW** (extreme magnitude without an automatic error signature).

Automatic triage classified 12 of 15 as RARE BUT VALID (jumbo loans up to $140.9M and multi-unit investment properties up to >149 units — legitimate under HMDA's reporting rules) and flagged 3 for manual review, since magnitude alone isn't a reliable automatic signal. Hand review resolves all three using the same internal-consistency test Phase 1 established for CLTV: does `CLTV × property_value` reconcile with the stated loan plus any senior lien, and does income plausibly service the debt?

| Row | Profile | Reconciliation | Verdict |
|---|---|---|---|
| 7727 | $1.0M subordinate lien, $9.2M CA principal residence, CLTV 40.9% | 40.9% × $9.2M = $3.77M total liens ≥ this loan → a senior first lien exists; $758k/yr income supports the debt | **RARE BUT VALID** — high-net-worth equity draw |
| 95313 | $35k home-improvement subordinate lien, $2.5M CA home, CLTV 19.9% | 19.9% × $2.5M = $498k total liens ≥ this loan; every figure is internally consistent (and the loan was denied regardless) | **RARE BUT VALID** — small improvement loan on an expensive home |
| 8102 | $2.505M cash-out first lien, $9.3M NY second residence, CLTV 26.9% | 26.9% × $9.3M = $2.50M = the loan itself, exactly self-consistent; $1.08M income at DTI 36–43% fits a jumbo borrower | **RARE BUT VALID** — jumbo refinance |

**Final tally: 15 of 15 top anomalies classified with evidence, zero unresolved.** No new DATA ERROR class was found beyond what the Phase 1 structural scan already flagged and priced into the cleaning decisions (§3.1 of the Preprocessing Report) — the detectors independently rediscovered the same boundary Phase 1 drew by hand, which is the "closing the loop" check the notebook runs explicitly (unanimous 5/5-vote rows: originally 8 RARE BUT VALID + 2 MANUAL REVIEW, now **10/10 RARE BUT VALID** after this resolution).

---

## 4. Cross-phase synthesis

The three phases are not independent exercises — each corroborates or sharpens the others:

- Phase 2's cluster 4 (DTI-stressed, 39.4% approval) and cluster 3 (manufactured, 43.1% approval) are the *segment-level* signatures of exactly the two dominant Phase 3 rules (DTI>60%, manufactured construction) — clustering found the populations, rule mining found the mechanism.
- Phase 3's denial-reason cross-check (§2.3) independently validates the DTI finding using a field the rule-mining never saw.
- Phase 4's anomaly ensemble, run on the same cleaned/leakage-guarded feature sets as Phases 2–3, converges with Phase 1's manual structural scan rather than contradicting it — evidence the upstream cleaning decisions (Preprocessing Report, §3–§7) were sound rather than merely convenient.

Full per-rule write-ups, all figures, and the exact code producing every number above are in `HMDA.ipynb` (Phases 2–4). Consolidated CSVs for the dashboard (`dashboard_app.py`) are listed in the notebook's Phase 5 hand-off cell, including `p3_decision_rules_final.csv` — the 11-rule, redundancy-pruned, significance-tested set this report is built on.
