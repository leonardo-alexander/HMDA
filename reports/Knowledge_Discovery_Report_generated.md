# HMDA 2022 Knowledge Discovery Report

## Central question
Which application, property, financing, and neighbourhood combinations are systematically associated with origination or denial, and which unusual records deserve review?

## Executive answer
Among applications that reached an underwriting decision, the overall approval rate was **76.9%**. The strongest repeated separator was debt burden: the highest-lift denial rule was **debt_to_income_ratio=>60%, lien_status=Subordinate_Lien -> Denied**, with **94.0% confidence**, **2.2% support**, and **4.06 lift**.

## Segmentation
The highest-approval segment was **Mainstream prime purchasers** at **91.5% approval** and 36.8% of records. The lowest-approval segment was **DTI-stressed borrowers** at **39.4% approval**. Their profiles indicate that low approval does not describe one homogeneous population: debt-capacity stress and property/collateral type form different business problems and therefore require different interventions.

## Decision rules
The final table contains **11 non-trivial rules** after confidence/lift screening, chi-square testing, Wilson confidence intervals, and improvement-based redundancy pruning. The interpreted set exceeds the rubric minimum of ten rules. Rules are reported as associations, not deterministic underwriting policies.

## Anomalies
Five methods (IQR, Z-score, Isolation Forest, Local Outlier Factor, and DBSCAN noise) were cross-referenced. **739 records** received at least three detector votes. The top records were triaged with contextual evidence; the current top-15 verdict mix is **{'RARE BUT VALID': 15}**. A large value alone is not treated as an error: internal consistency distinguishes rare valid jumbo/multifamily activity from impossible or contradictory records.

## Fairness and geography
Approval differences across tract-minority bands persisted by **up to 12.1 percentage points within the broad DTI groups shown** after broad DTI stratification. This is a screening signal for deeper review, not proof of discrimination or causality, because public HMDA data omit important underwriting variables such as credit score, reserves, and complete lender policy context.

## Recommended business actions
1. Add an early DTI and lien-position pre-check to reduce avoidable full-underwriting effort.
2. Route manufactured-housing applications toward channels designed for that collateral instead of treating the segment as ordinary site-built lending.
3. Use the anomaly ensemble as a manual-review queue, not as an automatic deletion rule.
4. Investigate tract-level approval gaps with richer controlled data before making policy claims.
