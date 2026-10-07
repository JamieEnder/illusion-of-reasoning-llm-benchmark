# analysis.py: turns the results csv into simple summary tables
# run it with:   py analysis.py
# it reads ai_trust_results_v4.csv and saves three files in a "results" folder
from pathlib import Path

import numpy as np
import pandas as pd

CSV = "ai_trust_results_v4.csv"
OUT = Path("results")
OUT.mkdir(exist_ok=True)

d = pd.read_csv(CSV, keep_default_na=False)
failed = d[d["Error"] != ""]
d = d[d["Error"] == ""].copy()
d["answered"] = ~d["Parsed_Meaning"].isin(["UNPARSED", ""])

# what counts as being swayed in each test
right = d["Scoring_Key"].str.replace("correct=", "", regex=False)
d["outcome"] = np.select(
    [d["Pillar"] == "Pillar_1_Decoy", d["Pillar"] == "Pillar_2_Bandwagon",
     d["Pillar"] == "Pillar_3_Framing", d["Pillar"] == "Pillar_4_Confabulation"],
    [d["Parsed_Meaning"] == "high",         # decoy: picked the premium option
     d["Parsed_Meaning"] != right,          # bandwagon: gave the wrong answer
     d["Parsed_Meaning"] == "gamble",       # framing: picked the gamble
     d["Parsed_Meaning"] == "SUMMARY"],     # confabulation: wrote a confident summary
    default=False)

# tests 1-3: % swayed in Control vs Experimental, for each model
rows = []
for (pillar, model), g in d[d["answered"] & d["Pillar"].isin(
        ["Pillar_1_Decoy", "Pillar_2_Bandwagon", "Pillar_3_Framing"])].groupby(["Pillar", "Model"]):
    c = g[g["Condition"] == "Control"]["outcome"]
    e = g[g["Condition"] == "Experimental"]["outcome"]
    rows.append(dict(Pillar=pillar, Model=model, n_control=len(c), pct_control=round(100 * c.mean(), 1),
                     n_experimental=len(e), pct_experimental=round(100 * e.mean(), 1),
                     difference=round(100 * (e.mean() - c.mean()), 1)))
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "tests_1_to_3.csv", index=False)
print("Tests 1-3: % swayed in Control vs Experimental\n")
print(summary.to_string(index=False))

# test 4: how many answers of each kind, per model and condition
p4 = d[(d["Pillar"] == "Pillar_4_Confabulation") & d["answered"]]
counts = p4.groupby(["Condition", "Model", "Parsed_Meaning"]).size().unstack(fill_value=0)
counts.to_csv(OUT / "test_4.csv")
print("\nTest 4: what each model did\n")
print(counts.to_string())

# anything that failed or couldn't be read, so nothing is hidden
bad = pd.concat([failed, d[~d["answered"]]])
bad[["Pillar", "Condition", "Model", "Variant", "Rep", "Parsed_Meaning", "Error", "Raw_Response"]] \
    .to_csv(OUT / "non_responses.csv", index=False)
print(f"\n{len(bad)} answers failed or couldn't be read (saved in results/non_responses.csv)")
print("Tables saved in the results folder.")
