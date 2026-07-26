# HMDA 2022 Preprocessing Report

## Objective
Prepare a 100,000-record HMDA sample for clustering, association-rule mining, and anomaly detection without converting legal sentinel values into ordinary numbers or leaking post-decision information.

## Cleaning decisions
- Loaded all raw fields as strings to preserve mixed numeric/range fields and leading zeroes.
- Audited `1111`, `8888`, and `9999` by column provenance rather than replacing them globally.
- Translated documented categorical codes while preserving `Exempt`, `Age_NA`, and `No_CoApplicant` as distinct meanings.
- Removed 5 exact duplicate attribute records and documented the check.
- Dropped 6 fields above the 60% structural-missingness threshold.
- Median-imputed remaining continuous gaps and retained explicit missingness indicators for audit purposes.
- Filled residual categorical gaps with `Unknown`.
- Confirmed 0 missing cells remained after handling.

## Transformation and selection
- Framed the decision subset as Originated versus Denied; the full table remains available for unsupervised analysis.
- Applied domain-readable bins for income, loan amount, property value, CLTV, tract income, and tract minority share.
- Used correlation for linear association/redundancy and entropy-based Mutual Information for non-linear relevance.
- Excluded post-decision pricing/cost fields and process-stage diagnostic flags from Phase 2-4 feature sets.
- Applied 1%/99% winsorization only to the clustering copy; original values remain untouched for profiling, rules, and anomaly review.
