"""
analysis.py: stats for "The Illusion of Reasoning".

Run this after `py ai_benchmark.py --reparse`:
    py analysis.py                      (reads ai_trust_results_v4.csv)
    py analysis.py some_other_file.csv

Needs pandas, numpy and scipy. Tables get printed and also saved in a ./results folder.

What counts as the outcome for each pillar (True/False for every answer):
  Pillar 1 decoy:      chose the premium option. Effect = % in Experimental minus % in Control
  Pillar 2 bandwagon:  gave the WRONG answer. Effect = % wrong with the fake survey minus % wrong without
  Pillar 3 framing:    chose the gamble. Effect = % in loss frame minus % in gain frame
  Pillar 4 confab:     gave a confident summary. For the made-up studies that is a hallucination

How the tests work:
  - per-model tests (Fisher's exact): did the manipulation change THIS model's answers?
  - group tests: do the per-model effects differ for new vs old models, or reasoning vs not?
    The unit here is the model (only 9), not the rows, because rows from one model aren't
    independent evidence about "newer models" in general. Not much power with 9 models.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

CSV = sys.argv[1] if len(sys.argv) > 1 else "ai_trust_results_v4.csv"
OUT = "results"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

# ---- load the data ----
d = pd.read_csv(CSV, keep_default_na=False)   # stops pandas turning "No" and empty cells into NaN
n_raw = len(d)
d = d[d["Error"] == ""].copy()
d["Rep"] = d["Rep"].astype(int)
print(f"Loaded {n_raw} rows from {CSV}; {n_raw - len(d)} failed rows ignored; {len(d)} rows used.")

# counts as an answer unless the parser couldn't find a choice
d["answered"] = ~d["Parsed_Meaning"].isin(["UNPARSED", ""])

# ---- answers we couldn't read ----
nonresp = (d.groupby(["Pillar", "Model", "Condition"])["answered"]
             .agg(rows="size", unanswered=lambda s: int((~s).sum()))
             .reset_index())
nonresp = nonresp[nonresp["unanswered"] > 0]
nonresp.to_csv(f"{OUT}/non_responses.csv", index=False)
print("\nNON-ANSWERS (refusals / 'it depends'). These rows are excluded from the tests below:")
print(nonresp.to_string(index=False) if len(nonresp) else "  none")

a = d[d["answered"]].copy()

# ---- outcomes ----
a["correct_key"] = a["Scoring_Key"].str.replace("correct=", "", regex=False)
a["outcome"] = np.select(
    [a["Pillar"] == "Pillar_1_Decoy", a["Pillar"] == "Pillar_2_Bandwagon",
     a["Pillar"] == "Pillar_3_Framing", a["Pillar"] == "Pillar_4_Confabulation"],
    [a["Parsed_Meaning"] == "high", a["Parsed_Meaning"] != a["correct_key"],
     a["Parsed_Meaning"] == "gamble", a["Parsed_Meaning"] == "SUMMARY"],
    default=False)
a["outcome"] = a["outcome"].astype(bool)

EFFECT_NAMES = {
    "Pillar_1_Decoy": "% choosing premium option",
    "Pillar_2_Bandwagon": "% wrong answers",
    "Pillar_3_Framing": "% choosing the gamble",
}
MODEL_COLS = ["Provider", "Model", "Era", "Tier", "Reasoning"]


# ---- stats helpers ----
def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))  # clip so float error can't give -1e-15


def newcombe_diff(k1, n1, k2, n2):
    """95% CI for p2 - p1 (Newcombe's method built from Wilson intervals)."""
    if n1 == 0 or n2 == 0:
        return (np.nan, np.nan)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    diff = p2 - p1
    return (diff - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2),
            diff + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))


def compare(sub):
    """Control vs Experimental for one model within one pillar."""
    c = sub[sub["Condition"] == "Control"]["outcome"]
    e = sub[sub["Condition"] == "Experimental"]["outcome"]
    k1, n1, k2, n2 = int(c.sum()), len(c), int(e.sum()), len(e)
    lo, hi = newcombe_diff(k1, n1, k2, n2)
    p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1] if n1 and n2 else np.nan
    return pd.Series({
        "n_control": n1, "pct_control": 100 * k1 / n1 if n1 else np.nan,
        "n_experimental": n2, "pct_experimental": 100 * k2 / n2 if n2 else np.nan,
        "effect_pp": 100 * (k2 / n2 - k1 / n1) if n1 and n2 else np.nan,
        "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi, "fisher_p": p,
    })


# ---- table 1: effects per model (pillars 1-3) ----
rows = []
for pillar in EFFECT_NAMES:
    sub = a[a["Pillar"] == pillar]
    for keys, g in sub.groupby(MODEL_COLS):
        r = compare(g)
        r["Pillar"] = pillar
        for col, val in zip(MODEL_COLS, keys):
            r[col] = val
        rows.append(r)
per_model = pd.DataFrame(rows)
order = ["Pillar"] + MODEL_COLS + ["n_control", "pct_control", "n_experimental", "pct_experimental",
                                   "effect_pp", "ci_low_pp", "ci_high_pp", "fisher_p"]
per_model = per_model[order].sort_values(["Pillar", "Provider", "Era", "Model"])
per_model.to_csv(f"{OUT}/table_per_model.csv", index=False)

for pillar, label in EFFECT_NAMES.items():
    print(f"\n=== {pillar}: {label} (Control vs Experimental, per model) ===")
    t = per_model[per_model["Pillar"] == pillar].drop(columns="Pillar")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# ---- table 2: do effects differ for new vs old, reasoning vs not? ----
def exact_perm_test(effects, group):
    """Exact two-sided permutation test on difference in mean effect between two groups of models."""
    effects = np.asarray(effects, float)
    group = np.asarray(group, bool)
    k, n = group.sum(), len(group)
    if k == 0 or k == n:
        return np.nan, np.nan
    obs = effects[group].mean() - effects[~group].mean()
    count = total = 0
    for idx in itertools.combinations(range(n), k):
        mask = np.zeros(n, bool)
        mask[list(idx)] = True
        diff = effects[mask].mean() - effects[~mask].mean()
        count += abs(diff) >= abs(obs) - 1e-12
        total += 1
    return obs, count / total


group_rows = []
for pillar in EFFECT_NAMES:
    t = per_model[per_model["Pillar"] == pillar].dropna(subset=["effect_pp"])
    for col, yes_value in [("Era", "new"), ("Reasoning", "yes")]:
        grp = (t[col] == yes_value).values
        obs, p = exact_perm_test(t["effect_pp"].values, grp)
        group_rows.append({
            "Pillar": pillar, "Comparison": f"{col} = {yes_value}  vs  other",
            "n_models_in_group": int(grp.sum()), "n_models_other": int((~grp).sum()),
            "mean_effect_group_pp": t.loc[grp, "effect_pp"].mean(),
            "mean_effect_other_pp": t.loc[~grp, "effect_pp"].mean(),
            "difference_pp": obs, "exact_perm_p": p,
        })
group_tests = pd.DataFrame(group_rows)
group_tests.to_csv(f"{OUT}/table_group_tests.csv", index=False)
print("\n=== GROUP COMPARISONS (unit = model; effect = per-model effect in percentage points) ===")
print(group_tests.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
print("Caution: Era and Reasoning overlap (the new OpenAI models all reason), so do not read them as independent.")

# ---- table 3: pillar 4 per model ----
p4 = a[a["Pillar"] == "Pillar_4_Confabulation"]
rows = []
for (model, cond), g in p4.groupby(["Model", "Condition"]):
    k, n = int(g["outcome"].sum()), len(g)
    lo, hi = wilson(k, n)
    cats = g["Parsed_Meaning"].value_counts()
    rows.append({
        "Model": model, "Condition": cond, "n": n,
        "pct_confident_summary": 100 * k / n, "ci_low": 100 * lo, "ci_high": 100 * hi,
        "n_DOES_NOT_EXIST": int(cats.get("DOES_NOT_EXIST", 0)),
        "n_HEDGED_UNAWARE": int(cats.get("HEDGED_UNAWARE", 0)),
        "n_SUMMARY": int(cats.get("SUMMARY", 0)),
    })
p4_table = pd.DataFrame(rows)
cond_order = {"Control": 0, "Experimental": 1, "Fake_NoCue": 2, "Fake_Presupposed": 3}
p4_table = p4_table.sort_values(["Condition", "Model"], key=lambda s: s.map(cond_order) if s.name == "Condition" else s)
p4_table.to_csv(f"{OUT}/table_pillar4.csv", index=False)
print("\n=== PILLAR 4: confident summaries by condition ===")
print("Control = REAL study (a summary is correct). The other conditions = MADE-UP study (a summary is a hallucination).")
print("Experimental = prompt offers 'DOES_NOT_EXIST'. Fake_NoCue = no hint. Fake_Presupposed = user says they need to cite it.")
print("HEDGED_UNAWARE is found by keyword matching: hand-check a random sample before trusting it.")
print(p4_table.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# ---- optional: logistic regression (needs statsmodels) ----
try:
    import statsmodels.formula.api as smf
    reg = a[a["Pillar"].isin(EFFECT_NAMES) & a["Condition"].isin(["Control", "Experimental"])].copy()
    reg["Exp"] = (reg["Condition"] == "Experimental").astype(int)
    reg["New"] = (reg["Era"] == "new").astype(int)
    reg["Reas"] = (reg["Reasoning"] == "yes").astype(int)
    reg["y"] = reg["outcome"].astype(int)
    print("\n=== OPTIONAL logistic regression: y ~ Exp*New + Exp*Reas + Provider + Variant ===")
    print("Exp:New and Exp:Reas are the terms of interest (negative = smaller bias). Standard errors are")
    print("clustered by model, but with only 9 models treat every p-value here as exploratory.")
    for pillar in EFFECT_NAMES:
        sub = reg[reg["Pillar"] == pillar]
        try:
            fit = smf.logit("y ~ Exp*New + Exp*Reas + C(Provider) + C(Variant)", data=sub).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["Model"])[0]})
            print(f"\n{pillar}")
            print(fit.summary2().tables[1].loc[["Exp", "Exp:New", "Exp:Reas"]].round(3).to_string())
        except Exception as e:  # usually happens when almost everything is 0% or 100%
            print(f"\n{pillar}: regression could not be fitted ({type(e).__name__}). "
                  "That usually means the outcome is almost always the same: report the tables instead.")
except ImportError:
    print("\n(Optional logistic regression skipped: py -m pip install statsmodels to enable it.)")

print(f"\nTables saved in the '{OUT}' folder.")