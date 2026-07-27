"""Phase 3 association-rule mining logic extracted from HMDA.ipynb.

Narrative markdown, the rule-space/rule-network plots, and the rule-count
validation cell stay in the notebook; only the data-transformation and rule
computations live here.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def build_transactions(appdeny, item_features, min_support=0.02):
    tx_src = appdeny.copy()
    tx_src["decision"] = tx_src["action_taken"]
    item_cols = item_features + ["decision"]
    tx_src = tx_src[item_cols].astype(str)

    onehot = pd.get_dummies(tx_src, prefix=item_cols, prefix_sep="=")
    decision_items = [c for c in onehot.columns if c.startswith("decision=")]

    supp = onehot.mean()
    keep = supp[(supp >= min_support) & (supp <= 0.95)].index.tolist()
    keep = sorted(set(keep) | set(decision_items))
    n_before = onehot.shape[1]
    onehot = onehot[keep].astype(bool)

    return onehot, decision_items, n_before


def mine_frequent_itemsets(onehot, min_support=0.02, max_len=3):
    from mlxtend.frequent_patterns import apriori
    return apriori(onehot, min_support=min_support, use_colnames=True, max_len=max_len, low_memory=True)


def extract_decision_rules(frequent, onehot, decision_items, min_lift=1.2, min_confidence=0.50):
    supp_lookup = {frozenset(s): float(v) for s, v in zip(frequent["itemsets"], frequent["support"])}

    def _item_mean(col):
        s = onehot[col]
        return float(s.sparse.to_dense().mean()) if hasattr(s, "sparse") else float(s.mean())
    dec_supp = {d: _item_mean(d) for d in decision_items}

    rows = []
    for S, supS in supp_lookup.items():
        dec_in = [d for d in S if d.startswith("decision=")]
        if len(dec_in) != 1 or len(S) < 2:
            continue
        d = dec_in[0]
        A = S - {d}
        supA = supp_lookup.get(A)
        if not supA:
            continue
        conf = supS / supA
        lift = conf / dec_supp[d]
        leverage = supS - supA * dec_supp[d]
        conviction = (1 - dec_supp[d]) / (1 - conf) if conf < 1 else np.inf
        rows.append({"antecedent": ", ".join(sorted(A)), "consequent": d,
                     "support": supS, "confidence": conf, "lift": lift,
                     "leverage": leverage, "conviction": conviction, "n_items": len(A)})

    all_rules = pd.DataFrame(rows)
    decision_rules = (all_rules[(all_rules["lift"] > min_lift) & (all_rules["confidence"] >= min_confidence)]
                      .sort_values("lift", ascending=False).reset_index(drop=True))
    return all_rules, decision_rules, dec_supp


def wilson_ci(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def _match_stats(items, consequent, onehot):
    mask = np.logical_and.reduce([onehot[it].values for it in items])
    tgt = onehot[consequent].values
    nA, k = int(mask.sum()), int((mask & tgt).sum())
    lo, hi = wilson_ci(k, nA)
    tot_t = int(tgt.sum())
    tbl = np.array([[k, nA - k],
                    [tot_t - k, len(onehot) - nA - (tot_t - k)]])
    chi2, pval, _, _ = chi2_contingency(tbl)
    return nA, k, lo, hi, chi2, pval


def test_significance(decision_rules, onehot, top_n=10):
    sig_rows = []
    for _, r in decision_rules.head(top_n).iterrows():
        items = [it for it in str(r["antecedent"]).split(", ") if it in onehot.columns]
        if len(items) != r["n_items"]:
            continue
        nA, k, lo, hi, chi2, pval = _match_stats(items, r["consequent"], onehot)
        sig_rows.append({"antecedent": r["antecedent"], "consequent": r["consequent"],
                         "n_matched": nA, "rate": round(k / nA, 3),
                         "ci95": f"[{lo*100:.1f}%, {hi*100:.1f}%]",
                         "lift": round(r["lift"], 2), "chi2": round(chi2, 1),
                         "p_value": f"{pval:.1e}"})
    return pd.DataFrame(sig_rows)


def prune_redundant(decision_rules, all_rules, dec_supp, onehot, min_improvement=0.02):
    """Improvement-criterion pruning (Bayardo et al., 1999): a rule is kept
    only if its confidence beats every proper sub-rule with the same
    consequent (including the empty/base-rate sub-rule) by min_improvement.

    Mutates `decision_rules` in place (adds improvement/best_subrule columns),
    matching the original cell so later references to the same object see them.
    """
    conf_lookup = {(frozenset(str(r["antecedent"]).split(", ")), r["consequent"]): float(r["confidence"])
                  for _, r in all_rules.iterrows()}

    def best_subrule(items, cons):
        best_c, best_a = dec_supp[cons], "(base rate)"
        for k in range(1, len(items)):
            for B in combinations(items, k):
                c = conf_lookup.get((frozenset(B), cons))
                if c is not None and c > best_c:
                    best_c, best_a = c, ", ".join(sorted(B))
        return best_c, best_a

    sub = [best_subrule(tuple(str(r["antecedent"]).split(", ")), r["consequent"])
          for _, r in decision_rules.iterrows()]
    decision_rules["improvement"] = decision_rules["confidence"] - np.array([c for c, a in sub])
    decision_rules["best_subrule"] = [a for c, a in sub]

    final_rules = (decision_rules[decision_rules["improvement"] >= min_improvement]
                  .sort_values("lift", ascending=False).reset_index(drop=True))
    pruned_away = decision_rules[decision_rules["improvement"] < min_improvement]

    final_sig_rows = []
    for _, r in final_rules.iterrows():
        items = str(r["antecedent"]).split(", ")
        nA, k, lo, hi, chi2, pval = _match_stats(items, r["consequent"], onehot)
        final_sig_rows.append({"antecedent": r["antecedent"], "consequent": r["consequent"], "n": nA,
                               "confidence": round(float(r["confidence"]), 3),
                               "lift": round(float(r["lift"]), 2),
                               "improvement": round(float(r["improvement"]), 3),
                               "ci95": f"[{lo*100:.1f}%, {hi*100:.1f}%]",
                               "chi2": round(chi2, 1), "p_value": f"{pval:.1e}"})
    final_sig = pd.DataFrame(final_sig_rows)

    return final_rules, pruned_away, final_sig


def validate_rule_count(final_rules, min_rules=10, required_metrics=("support", "confidence", "lift")):
    missing_rule_metrics = set(required_metrics) - set(final_rules.columns)
    interpreted_rules = final_rules.head(max(min_rules, min(15, len(final_rules)))).copy()
    interpreted_rules.insert(0, "rule_id", [f"R{i}" for i in range(1, len(interpreted_rules) + 1)])
    return missing_rule_metrics, interpreted_rules


def geography_dti_crosstab(appdeny, dti_grp):
    tmp = appdeny.copy()
    tmp["dti_grp"] = tmp["debt_to_income_ratio"].astype(str).map(dti_grp).fillna("Unknown/Exempt")

    marg = tmp.groupby("tract_minority_cat", observed=True)["target_approved"].agg(["mean", "size"])
    marg["mean"] = (marg["mean"] * 100).round(1)

    piv_rate = tmp.pivot_table(index="tract_minority_cat", columns="dti_grp",
                               values="target_approved", aggfunc="mean", observed=True) * 100
    piv_n = tmp.pivot_table(index="tract_minority_cat", columns="dti_grp",
                            values="target_approved", aggfunc="size", observed=True)
    return tmp, marg, piv_rate, piv_n


def denial_reason_itemsets(denials, min_support=0.03, max_len=3):
    from mlxtend.frequent_patterns import apriori
    dr_cols = [c for c in denials.columns if c.startswith("denial_reason")]
    dr = denials[dr_cols].astype(str)
    dr_oh = pd.get_dummies(dr, prefix=dr_cols, prefix_sep="=")
    dr_oh = dr_oh.loc[:, ~dr_oh.columns.str.contains("=nan|=None|=10\\b")].astype(bool)
    return apriori(dr_oh, min_support=min_support, use_colnames=True, max_len=max_len)


def export_phase3(decision_rules, sig_df, final_sig, interpreted_rules, data_dir, table_dir):
    data_dir = Path(data_dir)
    table_dir = Path(table_dir)
    decision_rules.to_csv(data_dir / "p3_decision_rules.csv", index=False)
    final_sig.to_csv(data_dir / "p3_decision_rules_final.csv", index=False)
    sig_df.to_csv(table_dir / "p3_rule_significance.csv", index=False)
    interpreted_rules.to_csv(table_dir / "p3_interpreted_rules.csv", index=False)
