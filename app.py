# app.py: dashboard for "The Illusion of Reasoning"
# run it with:   py -m streamlit run app.py
# it reads ai_trust_results_v4.csv from the same folder
import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import fisher_exact

st.set_page_config(page_title="The Illusion of Reasoning", layout="wide")

DATA_FILE = Path(__file__).parent / "ai_trust_results_v4.csv"
GITHUB_URL = ""  # paste the repo link here once it exists

# tab label, what counts as the outcome, what a bigger effect means
PILLARS = {
    "Pillar_1_Decoy": ("1 Decoy", "% choosing the premium option",
                       "Adding a useless 'decoy' option shouldn't change which plan a model picks. "
                       "A big jump means the decoy worked on it."),
    "Pillar_2_Bandwagon": ("2 Bandwagon", "% wrong answers",
                           "A fake survey says most experts believe the wrong answer. "
                           "A big jump means the model went along with the crowd."),
    "Pillar_3_Framing": ("3 Framing", "% choosing the gamble",
                         "Same maths, worded as lives saved (gain frame) or lost (loss frame). "
                         "A big jump means the wording changed the model's risk-taking."),
}
RESULT_COLS = ["Provider", "Model", "Era", "Tier", "Reasoning", "n_control", "pct_control", "n_experimental",
               "pct_experimental", "effect_pp", "ci_low", "ci_high", "fisher_p"]
COND_LABELS = {"Control": "Control (real study, hint given)",
               "Experimental": "Fake study, hint given",
               "Fake_NoCue": "Fake study, no hint",
               "Fake_Presupposed": "Fake study, user says they need to cite it"}


# ---- stats helpers (same maths as analysis.py) ----
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_diff(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    diff = p2 - p1
    return (diff - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2),
            diff + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))


@st.cache_data
def load_data(raw):
    d = pd.read_csv(io.BytesIO(raw), keep_default_na=False)
    d = d[d["Error"] == ""].copy()
    d["answered"] = ~d["Parsed_Meaning"].isin(["UNPARSED", ""])
    right = d["Scoring_Key"].str.replace("correct=", "", regex=False)
    d["outcome"] = np.select(
        [d["Pillar"] == "Pillar_1_Decoy", d["Pillar"] == "Pillar_2_Bandwagon",
         d["Pillar"] == "Pillar_3_Framing", d["Pillar"] == "Pillar_4_Confabulation"],
        [d["Parsed_Meaning"] == "high", d["Parsed_Meaning"] != right,
         d["Parsed_Meaning"] == "gamble", d["Parsed_Meaning"] == "SUMMARY"],
        default=False)
    return d


def compare_models(sub):
    """Control vs Experimental for each model (pillars 1-3)."""
    rows = []
    for keys, g in sub[sub["answered"]].groupby(["Provider", "Model", "Era", "Tier", "Reasoning"]):
        c = g[g["Condition"] == "Control"]["outcome"]
        e = g[g["Condition"] == "Experimental"]["outcome"]
        if len(c) == 0 or len(e) == 0:
            continue
        k1, n1, k2, n2 = int(c.sum()), len(c), int(e.sum()), len(e)
        lo, hi = newcombe_diff(k1, n1, k2, n2)
        rows.append(dict(zip(["Provider", "Model", "Era", "Tier", "Reasoning"], keys)) | {
            "n_control": n1, "pct_control": 100 * k1 / n1,
            "n_experimental": n2, "pct_experimental": 100 * k2 / n2,
            "effect_pp": 100 * (k2 / n2 - k1 / n1), "ci_low": 100 * lo, "ci_high": 100 * hi,
            "fisher_p": fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]})
    return pd.DataFrame(rows, columns=RESULT_COLS)


# ---- charts ----
def dumbbell(t, xtitle):
    """one row per model: grey dot = Control, purple dot = Experimental"""
    order = list(t["Model"])
    pts = t.melt(id_vars=["Model"], value_vars=["pct_control", "pct_experimental"],
                 var_name="Condition", value_name="pct")
    pts["Condition"] = pts["Condition"].map({"pct_control": "Control", "pct_experimental": "Experimental"})
    lines = alt.Chart(t).mark_rule(color="#9ca3af").encode(
        y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)), x="pct_control:Q", x2=alt.X2("pct_experimental"))
    dots = alt.Chart(pts).mark_circle(size=150).encode(
        y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("pct:Q", title=xtitle, scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Condition:N", scale=alt.Scale(domain=["Control", "Experimental"],
                                                       range=["#6b7280", "#7c3aed"])),
        tooltip=["Model", "Condition", alt.Tooltip("pct:Q", format=".1f")])
    return lines + dots


def heatmap(h, order):
    base = alt.Chart(h).encode(x=alt.X("Pillar:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)))
    rect = base.mark_rect().encode(color=alt.Color(
        "effect_pp:Q", title="effect (pp)",
        scale=alt.Scale(scheme="redblue", domain=[-100, 100], reverse=True)))
    text = base.mark_text().encode(text=alt.Text("effect_pp:Q", format=".0f"))
    return rect + text


# ---- load data ----
if DATA_FILE.exists():
    raw = DATA_FILE.read_bytes()
else:
    up = st.file_uploader("Upload ai_trust_results_v4.csv", type="csv")
    if up is None:
        st.info("Upload the results csv to get started.")
        st.stop()
    raw = up.getvalue()
df = load_data(raw)

# ---- sidebar filters ----
st.sidebar.header("Filters")


def pick(label, col, data):
    options = sorted(data[col].unique())
    chosen = st.sidebar.multiselect(label, options, default=options)
    return data[data[col].isin(chosen)]


df = pick("Model family", "Provider", df)
df = pick("Tier", "Tier", df)
df = pick("Era", "Era", df)
df = pick("Reasoning switched on", "Reasoning", df)
df = pick("Model", "Model", df)
if df.empty:
    st.warning("No models match those filters.")
    st.stop()

# ---- page ----
st.title("The Illusion of Reasoning")
st.caption("Do newer 'reasoning' models resist human-style cognitive biases better than older ones? "
           "OpenAI and Anthropic models, 4 tests. Exploratory results, not a finished paper."
           + (f"  [Code and data]({GITHUB_URL})" if GITHUB_URL else ""))

tab_names = ["Overview"] + [v[0] for v in PILLARS.values()] + ["4 Confabulation", "Data"]
tabs = st.tabs(tab_names)

# overview: one effect number per model and pillar
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Answers shown", f"{len(df):,}")
    c2.metric("Models shown", df["Model"].nunique())
    c3.metric("Scenarios (all 4 tests)", df["Variant"].nunique())
    parts = []
    for pillar, (label, _, _) in PILLARS.items():
        t = compare_models(df[df["Pillar"] == pillar])
        if not t.empty:
            parts.append(t[["Model", "effect_pp"]].assign(Pillar=label))
    if parts:
        h = pd.concat(parts)
        st.subheader("How much did each test change each model's answers?")
        st.caption("Effect = % in the Experimental condition minus % in Control, in percentage points. "
                   "Red = the manipulation shifted the model a lot, white = it made no difference.")
        st.altair_chart(heatmap(h, sorted(h["Model"].unique())))

# pillars 1-3 all work the same way
for tab, (pillar, (label, xtitle, blurb)) in zip(tabs[1:4], PILLARS.items()):
    with tab:
        st.markdown(blurb)
        sub = df[df["Pillar"] == pillar]
        scenarios = sorted(sub["Variant"].unique())
        chosen = st.multiselect("Scenarios included", scenarios, default=scenarios, key=pillar)
        sub = sub[sub["Variant"].isin(chosen)]
        t = compare_models(sub)
        if t.empty:
            st.info("Nothing to show with these filters.")
            continue
        st.altair_chart(dumbbell(t, xtitle))
        shown = t.drop(columns=["Era", "Tier"]).round(1)
        shown["fisher_p"] = t["fisher_p"].map(lambda p: f"{p:.3g}")
        st.dataframe(shown)
        with st.expander("Effect (pp) by scenario"):
            st.caption("Only about 5 answers per condition in each cell, so look for patterns, not exact numbers.")
            by_scen = {s: compare_models(sub[sub["Variant"] == s]).set_index("Model")["effect_pp"] for s in chosen}
            st.dataframe(pd.DataFrame(by_scen).round(0))

# pillar 4 has more than two conditions, so it gets its own layout
with tabs[4]:
    st.markdown("Models are asked to summarise a study. In the Control it is real, so a summary is correct. "
                "In the other three conditions the study is made up, so a confident summary means the model "
                "invented one.")
    p4 = df[(df["Pillar"] == "Pillar_4_Confabulation") & df["answered"]].copy()
    if p4.empty:
        st.info("Nothing to show with these filters.")
    else:
        p4["Condition"] = p4["Condition"].map(COND_LABELS)
        counts = p4.groupby(["Condition", "Model", "Parsed_Meaning"]).size().reset_index(name="n")
        counts["pct"] = 100 * counts["n"] / counts.groupby(["Condition", "Model"])["n"].transform("sum")
        chart = alt.Chart(counts).mark_bar().encode(
            y=alt.Y("Model:N", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("pct:Q", title="% of answers", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Parsed_Meaning:N", title="what the model did",
                            scale=alt.Scale(domain=["SUMMARY", "HEDGED_UNAWARE", "DOES_NOT_EXIST"],
                                            range=["#dc2626", "#f59e0b", "#16a34a"])),
            tooltip=["Model", "Parsed_Meaning", "n", alt.Tooltip("pct:Q", format=".0f")],
        ).properties(height=170).facet(row=alt.Row("Condition:N", sort=list(COND_LABELS.values()), title=None))
        st.altair_chart(chart)
        st.caption("Red = confident summary, orange = said it couldn't verify the study, green = said it doesn't exist.")
        st.dataframe(counts.pivot_table(index=["Condition", "Model"], columns="Parsed_Meaning",
                                        values="n", fill_value=0))

# raw data
with tabs[5]:
    st.caption("Every answer behind the charts, with the exact prompt and the model's raw reply.")
    cols = ["Pillar", "Condition", "Model", "Variant", "Rep", "Parsed_Meaning", "Raw_Response", "Prompt"]
    st.dataframe(df[cols])
    st.download_button("Download these rows as csv", df.to_csv(index=False).encode("utf-8"),
                       file_name="illusion_of_reasoning_filtered.csv")
