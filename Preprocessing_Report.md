# Preprocessing Report — HMDA 2022 Knowledge Discovery Pipeline

**Dataset:** 100,000-record sample of the 2022 Home Mortgage Disclosure Act (HMDA) loan-level public file (CFPB), 99 raw columns. Source: `hmda_sample.csv` (Hugging Face). Companion notebook: `HMDA.ipynb`, Phase 1 (§1–§10).

This report documents every cleaning decision made on the raw HMDA extract, the evidence that motivated it, and the validation that confirms it did what it was supposed to do. Figures and code cited by section number refer to the corresponding cell in `HMDA.ipynb`.

---

## 1. Loading strategy

All 99 columns are loaded as strings (`dtype=str`), not inferred. Two failure modes this avoids:

- **Silent mistyping.** Pandas infers a column's dtype from a scan of the file; several HMDA fields mix small integer codes with wider numeric ranges (e.g. `debt_to_income_ratio` mixes exact percentages with published range labels), which risks partial numeric coercion depending on which rows the scanner sees.
- **Leading-zero loss** in code fields that are semantically categorical (e.g. state/county codes).

Column names are normalized (`-` → `_`) and a broad null-token set (`""`, `"NA"`, `"null"`, `"None"`, …) is mapped to `NaN` up front, so every later `.isna()` check is trustworthy. Result: **100,000 rows × 99 columns**, no shape loss at this step.

## 2. Feature-type segmentation

Columns are grouped by *semantic* role, not just dtype — nominal/ordinal alone is insufficient because several fields mix a numeric code with a text label (e.g. `1111` = "Exempt" inside an otherwise numeric column), and applying a single global rule (e.g. "treat 1111 as missing everywhere") would corrupt unrelated fields:

| Group | Examples | Handling implication |
|---|---|---|
| `CONTINUOUS` | `loan_amount`, `income`, `interest_rate` | coerce to numeric; watch for real vs. sentinel-code collisions |
| `STRING_BAND` | `debt_to_income_ratio`, `applicant_age`, `total_units` | mixed range-label / exact-value fields, need custom banding |
| `CATEG_CODE` | `loan_type`, `occupancy_type`, `hoepa_status` | integer codes with a documented label map, `1111` = Exempt |
| `TEXT_CATEG` | `state_code`, `derived_race`, `conforming_loan_limit` | already human-readable |
| `IDS` | `lei`, `census_tract`, `activity_year` | identifiers / near-constants, not features |
| `DEMOGRAPHIC_RAW` | `applicant_race_1..5`, `*_observed` | redundant with the `derived_*` summary fields |
| `AUS` / `DENIAL` | `aus_1..5`, `denial_reason_1..4` | institutional tooling / outcome-only fields |

Label dictionary source: FFIEC's public LAR data-field documentation.

**Validation — the 8 groups are a partition, checked, not assumed.** A column silently left out of every group would skip every downstream rule (sentinel handling, label mapping, drop lists); a column claimed by two groups could be handled twice with conflicting rules. Cross-checking the 8 group lists directly against `df.columns`: **99 raw columns, sum of group sizes = 99, 0 uncovered, 0 overlapping, 0 phantom entries (group members not actually in `df`)** — confirmed by an executable assertion, not by eyeballing the lists against the column count.

## 3. Pre-cleaning diagnostics (evidence gathered before any cell mutates data)

Four checks run on the raw frame to decide *how* to clean, not just to clean:

**3.1 Distribution scan.** Histograms of all continuous fields (`p1_raw_distribution_scan.png`) show near-universal heavy right skew in dollar fields, and `combined_loan_to_value_ratio` reaching into the thousands of percent — a data-error signature investigated in Phase 4, not silently clipped here.

**3.2 Sentinel-code scan.** Per the FFIEC dictionary, codes `1111` / `8888` / `9999` are only documented in specific columns. Counting hits against that documented whitelist (not a blanket scan) found:

| Code | Columns | Count |
|---|---|---|
| `1111` | 14 categorical/AUS columns (institutional-exemption block) | 1,854–1,890 each |
| `8888` | `applicant_age`, `co_applicant_age` | 10,466 / 8,523 |
| `9999` | `co_applicant_age` | 54,717 (no co-applicant) |

A separate check confirmed no continuous field contains a code-valued collision requiring masking (§4 below).

**3.2b Rogue-sentinel audit (full column sweep).** The whitelist above only proves the documented columns behave as expected — it says nothing about the other 85. Re-running the same scan across **every** raw column (continuous fields matched numerically to tolerate decimals; coded/ID fields matched by exact string so a zero-padded FIPS code like `"01111"` is never mistaken for the sentinel `1111`) finds exactly **8 columns with hits outside the whitelist**:

- **7 `CONTINUOUS` columns, 179 rows total** (`tract_owner_occupied_units`, `tract_one_to_four_family_homes`, `tract_population`, `origination_charges`, `lender_credits`, `total_points_and_fees`, `income`) — this total reconciles exactly with the independent lookalike count computed for the §4 masking decision (same formula, same 179 rows): not a new finding, a cross-check that passes.
- **`denial_reason_1`, 1,554 rows** with value `1111`, spread across *every* `action_taken` value (only 144 of them — 0.9% — are actually Denied). This reads as the same institution-level "Exempt" convention used in the 14 whitelisted columns, just undocumented for this specific field. It changes nothing operationally: `denial_reason_*` is dropped from the main table regardless (§5), and the 144-row sliver sits below the 3% `min_support` floor used in the Phase 3 denial-reason mining, so it was already silently excluded there — this audit demonstrates that exclusion rather than leaving it assumed.
- **Zero** hits in `STRING_BAND`, `CATEG_CODE`, `TEXT_CATEG`, `IDS`, `DEMOGRAPHIC_RAW`, `AUS`. One deliberate near-miss confirms the method: `county_code` has 9 rows reading `"01111"` (a genuine zero-padded Alabama FIPS code) that a naive numeric comparison would misread as the sentinel — caught and correctly excluded by matching coded fields on exact string instead of numeric value.

**3.3 `debt_to_income_ratio` structure.** HMDA publishes this field as **exact integers only for 36 ≤ DTI < 50**, and as coarse range labels elsewhere (`<20%`, `>60%`, …). Two diagnostics decided how to re-bin the exact-integer window:

- The published range labels alone would put **~45%** of decisioned applications into one 36–50 bucket — a bucket dense enough to flood any downstream rule-mining with low-lift, high-support noise.
- Approval rate by *exact* DTI integer, 36–49 (`p1_dti_structure.png`, cell 12 output), is flat inside 36–42 and flat again inside 43–49, but steps down **4.1 percentage points exactly at 42→43** (0.878 → 0.837) — coinciding with the former Qualified-Mortgage 43% threshold. This is the largest single-step change in the window, so one cut at 43% captures all the target-relevant signal; a naive equal-depth 3-way split was rejected because it would not align with where lending behavior actually changes.

**3.4 Structural diagnostics** (`p1_structural_diagnostics.png`): a missingness heatmap by `action_taken` shows pricing/cost fields (`interest_rate`, `total_loan_costs`, `origination_charges`, …) are **≈100% missing for every action other than Originated/Purchased** — these values are only assigned once a loan is priced, so missingness here is *structural*, not random, and must not be globally imputed (this becomes the leakage guard in the Phase 1↔2 bridge). The same figure shows `income ≤ 0` occurs (759 rows) — plausible for business-loss applicants, not treated as an error — and `CLTV > 200%` occurs in a small but real tail, flagged for Phase 4 rather than clipped here. Top-3 states account for ~27% of the sample: concentrated but not so dominant that geography findings just describe one metro.

## 4. Sentinel / exempt / not-applicable handling

| Field type | Rule | Rationale |
|---|---|---|
| Continuous | Coerce to numeric; **no value masking** | No continuous field documents a sentinel code — numeric exemption is encoded as the string `"Exempt"`, which coercion already nulls. `any_exempt_field` is derived separately from the 14-column categorical-exempt block (§3.2), since exemption travels by *institution*, not by continuous value. This keeps rare genuine code-valued amounts (e.g. a real $1,111 lender credit, individually verified against `origination_charges` and `action_taken` in the notebook, and reconfirmed at scale by the §3.2b rogue-sentinel audit) from being wrongly nulled. |
| `applicant_age`, `co_applicant_age` | `8888`→`"Age_NA"`, `9999`→`"No_CoApplicant"`, kept as ordered categories | Preserves the "no co-applicant" signal as information rather than deleting it. |
| `debt_to_income_ratio` | Exact integers split at 43% into `36%-<43%` / `43%-<50%`, joining the existing published bands | Directly implements the §3.3 finding — the only cut point supported by the approval-rate step. |
| `total_units` | Kept as the existing ordered band (`1,2,3,4,5-24,…,>149`) | Already domain-correct; no transformation needed. |
| Categorical codes | `1111` → `"Exempt"`; documented code maps applied (`action_taken`, `loan_type`, `occupancy_type`, …) | Human-readable categories for every downstream table and rule. |

## 5. Resolving derived-vs-inherited redundancy

Where HMDA publishes both a raw multi-column concept and a pre-derived summary field, only one representation is kept, chosen for interpretability and to avoid double-counting a signal in correlation/mutual-information/rule-mining:

| Concept | Kept | Dropped |
|---|---|---|
| Race | `derived_race` | `applicant_race_1..5`, `co_applicant_race_1..5` |
| Ethnicity | `derived_ethnicity` | `applicant_ethnicity_1..5`, `co_applicant_ethnicity_1..5` |
| Sex | `derived_sex` | `applicant_sex`, `co_applicant_sex`, `*_observed` |
| Loan product | `loan_type` + `lien_status` | `derived_loan_product_type` |
| Dwelling | `construction_method` + `total_units` | `derived_dwelling_category` |
| Age > 62 | `applicant_age` band | `applicant_age_above_62`, `co_applicant_age_above_62` |
| Geography | `state_code` + `tract_*` | `census_tract`, `county_code`, `derived_msa_md` (high-cardinality IDs) |
| — | — | `activity_year` (constant), `lei` (institution ID), `aus_1..5` (tooling, not applicant trait) |

**This drop is validated, not assumed:** for both derived-field pairs, grouping by the kept-field combination and counting distinct derived values per group returns a maximum of **1** in every case — i.e., the dropped field is a pure function of the kept fields, so dropping it loses zero information and it is exactly recoverable if ever needed.

`denial_reason_*` is **set aside into a separate frame** (`hmda_denials.csv`, 15,684 rows), not deleted and not merged into the approve/deny mining table — it only exists when `action_taken = Denied`, so including it in an approval-vs-denial analysis would be tautological (it would mine "denied because a denial reason is present"). It feeds a dedicated denial-reason analysis in Phase 3 instead.

**Net effect:** 46 redundant/ID/outcome-adjacent columns dropped; **54 columns remain**.

## 6. Duplicates

Exact full-row duplicates (evaluated after dropping `lei`/`census_tract`, i.e. on the substantive attribute profile) — **5 rows removed**: 100,000 → **99,995**.

## 7. Missing-value handling

Decision order, each with a stated rationale:

1. **Drop fields >60% missing by design** (data-driven threshold, not arbitrary): `total_points_and_fees`, `discount_points`, `lender_credits`, `prepayment_penalty_term`, `intro_rate_period`, `multifamily_affordable_units` — all >70% Exempt/NA nationally, offering little discovery value at that sparsity. **6 fields dropped**, `action_taken` explicitly protected from this rule regardless of its missingness.
2. **Add `_was_missing` indicators** for 5 high-signal continuous fields (`property_value`, `income`, `combined_loan_to_value_ratio`, `interest_rate`, `loan_term`) **before** imputing — so Phase 3's rule mining can still discover patterns like "`property_value` missing ⇒ higher denial" that median-imputation would otherwise erase. (This is exactly what surfaces later: `combined_loan_to_value_ratio_was_missing` and `property_value_was_missing` turn out to be the two strongest correlates of approval in §9.)
3. **Median-impute** remaining continuous fields — robust to the heavy right skew documented in §3.1, and simple enough to stay interpretable for a discovery (not prediction) project.
4. **Categorical residual missing → `"Unknown"`** — structural missingness is already captured by `Exempt` / `No_CoApplicant` / `Age_NA` from §4, so anything still `NaN` here is genuinely unknown, not structurally absent.

Result: **0 missing values remain**; 48 columns after the drop. Every field's missingness and its fate (dropped / imputed / filled Unknown) is plotted against the 60% threshold line in `p1_missingness_fate.png` for auditability.

## 8. Data transformation: target framing and binning

**Target framing is scoped, not global.** `target_approved` (Originated=1 / Denied=0) is defined only on the `approve_deny` subset (67,827 of 99,995 rows; **76.9% approval rate**). The full `df`/`clean` table stays target-free, because Phases 2 and 4 are unsupervised (clustering, anomaly detection) and must not carry an outcome label baked into their input space. This directly matches the assigned mining angle — *which applicant combinations are systematically approved or denied* (association strength), not building a classifier.

**Binning for association-rule mining** turns 7 continuous fields into interpretable domain bands (`income_band`, `loan_amount_band`, `property_value_band`, `interest_rate_band`, `cltv_band`, `tract_income_cat`, `tract_minority_cat`), including two tract-level bands specifically because the project brief asks about geography and demographics. Every band is validated non-empty, and `p1_band_validation.png` plots counts alongside approval rate per band — approval rises monotonically with income and falls monotonically with CLTV, confirming the edges sit where lending behavior actually changes rather than at arbitrary round numbers.

## 9. Feature selection: correlation **and** entropy

Two independent scoring methods, because they catch different things — correlation misses non-linear/categorical dependence, entropy (mutual information) is noisier on its own:

**9a. Correlation with approval**, computed only over an explicitly **leakage-excluded** feature set: `interest_rate`, `rate_spread`, `total_loan_costs`, `total_points_and_fees`, `origination_charges`, `discount_points`, `lender_credits`, `hoepa_status`, `purchaser_type`, and their `_was_missing` flags are all removed before scoring, because every one of them only exists *because* a loan was priced/purchased post-decision (§3.4) — including them would let the target leak into its own predictors. With that guard in place, the top correlates are the missingness flags themselves (`combined_loan_to_value_ratio_was_missing`: −0.156, `property_value_was_missing`: −0.132) and genuine application-time signals (`tract_to_msa_income_percentage`: +0.094, `tract_minority_population_percent`: −0.078).

**9b. Redundancy filter** (|r| > 0.90 among numeric features): **0 pairs found** — no numeric feature needs dropping for collinearity.

**9c. Mutual information** (`mutual_info_classif`, numeric + label-encoded categorical): `debt_to_income_ratio` leads by a wide margin (0.0904), ahead of `combined_loan_to_value_ratio` (0.0390) and `income` (0.0287) — foreshadowing DTI's dominance in the Phase 3 rule mining.

**9d. Combined ranking** (min-max normalized average of |r| and MI) buckets features into **Strong** (score > 0.15, 13 features), **Moderate** (7), **Weak** (11), exported to `feature_ranking_combined.csv` and used to prioritize which fields feed clustering (Phase 2) and rule mining (Phase 3).

## 10. Exported artifacts

| File | Shape | Purpose |
|---|---|---|
| `hmda_clean.csv` | 99,995 × 60 | Full labeled/binned table — clustering, profiling |
| `hmda_approve_deny.csv` | 67,827 × 61 | Primary decisioned-only mining table (`target_approved`) |
| `feature_ranking_combined.csv` | — | Correlation + entropy ranking |
| `hmda_denials.csv` | 15,684 × 5 | Denial-reason-only side frame |

## Summary of casualties and safeguards

- Duplicates removed: 5
- High-missing fields dropped: 6 (of 99 → 54 pre-drop → 48 post-drop → 60 post-binning with `clean`)
- Residual missing values after Phase 1: **0**
- Leakage-relevant fields (`interest_rate`, `rate_spread`, loan-cost block, `hoepa_status`, `purchaser_type`, and their derived `_was_missing`/`_band` fields) are excluded from every feature set used downstream — verified with an executable assertion at the Phase 1→2 bridge (`HMDA.ipynb`, "Leakage guard" cell), not just a documentation comment: `CLUSTER_FEATS`, `ANOMALY_FEATS`, and `ITEM_FEATURES` are each checked against the post-decision list and the pipeline raises if any leak through.

This last point is not required by the assignment rubric, but it is the methodological safeguard most worth highlighting: it converts "we didn't leak the target" from a claim into a machine-checked invariant that would break the notebook run if violated by a future edit.
