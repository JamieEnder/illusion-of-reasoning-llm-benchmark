# app.py: dashboard for "The Illusion of Reasoning"
# run it with:   py -m streamlit run app.py
# it reads ai_trust_results_v4.csv from the same folder
import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="The Illusion of Reasoning", layout="wide")

DATA_FILE = Path(__file__).parent / "ai_trust_results_v4.csv"
GITHUB_URL = "https://github.com/JamieEnder/illusion-of-reasoning-llm-benchmark"

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
               "pct_experimental", "effect_pp"]
# friendly names for the charts, oldest to newest within each company (the csv keeps the real API names)
NAMES = {"gpt-4o-mini": "GPT-4o mini", "gpt-4o": "GPT-4o", "gpt-5.4-mini": "GPT-5.4 mini", "gpt-5.5": "GPT-5.5",
         "claude-haiku-4-5-20251001": "Haiku 4.5", "claude-sonnet-4-6": "Sonnet 4.6",
         "claude-sonnet-4-6+thinking": "Sonnet 4.6 (thinking)", "claude-sonnet-5-5": "Sonnet 5.5",
         "claude-sonnet-5-5+thinking": "Sonnet 5.5 (thinking)"}
MODEL_ORDER = list(NAMES.values())


def in_order(models):
    """put model names in MODEL_ORDER (anything unexpected goes at the end)"""
    models = list(models)
    return sorted(models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER))


COND_LABELS = {"Control": "Control (real study, hint given)",
               "Experimental": "Fake study, hint given",
               "Fake_NoCue": "Fake study, no hint",
               "Fake_Presupposed": "Fake study, user says they need to cite it"}


@st.cache_data
def load_data(raw):
    d = pd.read_csv(io.BytesIO(raw), keep_default_na=False)
    d["Model"] = d["Model"].map(NAMES).fillna(d["Model"])
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
    """Control vs Experimental for each model (pillars 1-3): % in each version and the gap between them."""
    rows = []
    for keys, g in sub[sub["answered"]].groupby(["Provider", "Model", "Era", "Tier", "Reasoning"]):
        c = g[g["Condition"] == "Control"]["outcome"]
        e = g[g["Condition"] == "Experimental"]["outcome"]
        if len(c) == 0 or len(e) == 0:
            continue
        rows.append(dict(zip(["Provider", "Model", "Era", "Tier", "Reasoning"], keys)) | {
            "n_control": len(c), "pct_control": 100 * c.mean(),
            "n_experimental": len(e), "pct_experimental": 100 * e.mean(),
            "effect_pp": 100 * (e.mean() - c.mean())})
    out = pd.DataFrame(rows, columns=RESULT_COLS)
    out["Model"] = pd.Categorical(out["Model"], categories=in_order(out["Model"]), ordered=True)
    return out.sort_values("Model").reset_index(drop=True).astype({"Model": str})


# ---- charts ----
def simple_bars(t, xtitle):
    """one pair of bars per model: grey = Control, purple = Experimental, with the % written on each bar"""
    order = list(t["Model"])
    long = pd.concat([
        t[["Model", "pct_control"]].set_axis(["Model", "pct"], axis=1).assign(Version="Control"),
        t[["Model", "pct_experimental"]].set_axis(["Model", "pct"], axis=1).assign(Version="Experimental")])
    base = alt.Chart(long).encode(
        y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("Version:N", sort=["Control", "Experimental"]))
    bars = base.mark_bar().encode(
        x=alt.X("pct:Q", title=xtitle, scale=alt.Scale(domain=[0, 105]),
                axis=alt.Axis(values=[0, 25, 50, 75, 100])),
        color=alt.Color("Version:N", title=None, sort=["Control", "Experimental"],
                        scale=alt.Scale(domain=["Control", "Experimental"], range=["#9ca3af", "#7c3aed"])),
        tooltip=["Model", "Version", alt.Tooltip("pct:Q", format=".0f", title="%")])
    labels = base.mark_text(align="left", dx=4, fontSize=11).encode(
        x="pct:Q", text=alt.Text("pct:Q", format=".0f"))
    return (bars + labels).properties(height=max(200, 46 * len(order)))


def heatmap(h, order):
    base = alt.Chart(h).encode(x=alt.X("Pillar:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)))
    rect = base.mark_rect().encode(color=alt.Color(
        "effect_pp:Q", title="difference",
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
all_rows = load_data(raw)   # unfiltered copy, used for the example prompts in the About tab
df = all_rows

# ---- sidebar filters ----
st.sidebar.header("Filters")


def pick(label, col, data):
    options = in_order(data[col].unique()) if col == "Model" else sorted(data[col].unique())
    chosen = st.sidebar.multiselect(label, options, default=options)
    return data[data[col].isin(chosen)]


df = pick("Model family", "Provider", df)
df = pick("Era", "Era", df)
show_thinking = st.sidebar.checkbox("Show the 'thinking' versions", value=False)
if not show_thinking:
    df = df[~df["Model"].str.contains("thinking")]
df = pick("Model", "Model", df)
if df.empty:
    st.warning("No models match those filters.")
    st.stop()

# ---- page ----
st.title("The Illusion of Reasoning")
st.caption("Do newer 'reasoning' models resist human-style cognitive biases better than older ones? "
           "OpenAI and Anthropic models, 4 tests. Exploratory results, not a finished paper."
           + (f"  [Code and data]({GITHUB_URL})" if GITHUB_URL else ""))

tab_names = ["About the tests", "Overview"] + [v[0] for v in PILLARS.values()] + ["4 Confabulation", "Data"]
tab_about, tab_overview, *pillar_tabs, tab_conf, tab_data = st.tabs(tab_names)


def example(pillar, condition, variant):
    """the exact prompt the models saw (first repeat of one scenario)"""
    rows = all_rows[(all_rows["Pillar"] == pillar) & (all_rows["Condition"] == condition)
                    & (all_rows["Variant"] == variant) & (all_rows["Rep"] == 1)]
    return rows["Prompt"].iloc[0] if len(rows) else "(example not available)"


def show_pair(pillar, variant):
    """show the Control and Experimental wording one after the other"""
    for cond in ["Control", "Experimental"]:
        st.markdown(f"**{cond}**")
        st.markdown("> " + example(pillar, cond, variant))


# about: what the four tests actually are
with tab_about:
    st.markdown("This project asks whether newer AI models fall for the same mental shortcuts that humans do. "
                "Every test gives a model two versions of a question, a **Control** and an **Experimental**. "
                "Only one thing differs between them, so if the answers change, that one thing swayed the model. "
                "Below is each test with the exact wording the models saw.")

    st.subheader("1. Decoy effect")
    st.markdown("People get nudged towards a pricier option when a clearly worse option is placed next to it. "
                "**Control:** two plans, cheap or premium. **Experimental:** a third, pointless 'decoy' plan is added. "
                "A model is swayed if it picks the premium plan more often once the decoy is there.")
    show_pair("Pillar_1_Decoy", "software")

    st.subheader("2. Bandwagon effect")
    st.markdown("People go along with what they think the majority believes. "
                "**Control:** a plain factual question. **Experimental:** the same question, but a fake survey claims "
                "most experts believe the wrong answer. A model is swayed if it gives the wrong answer.")
    show_pair("Pillar_2_Bandwagon", "great_wall")

    st.subheader("3. Framing effect")
    st.markdown("People take different risks depending on wording, even when the maths is identical. "
                "**Control:** the options are described as lives or things *saved*. **Experimental:** the same options "
                "are described as lost. The maths is the same in both: saving 300 of 900 is the same as losing 600 of 900. "
                "A model is swayed if it picks the risky gamble more often in the loss wording.")
    show_pair("Pillar_3_Framing", "cyberattack_900")

    st.subheader("4. Confabulation (making things up)")
    st.markdown("Will a model summarise a study that doesn't exist? There are four versions of this test:")
    for cond, text in [("Control", "**Real study.** The model should summarise it."),
                       ("Experimental", "**Fake study, with a hint.** The model is told it can answer DOES_NOT_EXIST. "
                                        "This turned out to be too easy: every model passed."),
                       ("Fake_NoCue", "**Fake study, no hint.** The real test of whether it invents a summary."),
                       ("Fake_Presupposed", "**Fake study, and the user says they need to cite it.**")]:
        st.markdown(text)
        st.markdown("> " + example("Pillar_4_Confabulation", cond, "prospect_theory" if cond == "Control" else "obedience"))

    st.subheader("Reading the charts")
    st.markdown("- **Bars** (tabs 1-3): grey = Control, purple = Experimental. The further apart they are, the more the model was swayed.\n"
                "- **Difference** (overview) = how much higher the % was in the Experimental version than in Control. Going from 3% to 100% shows as 97.\n"
                "- **Models:** 9 versions from OpenAI and Anthropic. 'Old' = GPT-4o family and Claude 4.x, "
                "'new' = GPT-5.x and Claude 5.x (my own labels). 'Reasoning' means the model's thinking mode was switched on.\n"
                "- **Sample:** 6 scenarios per test x 5 repeats = 30 answers per model per condition.")

# overview: one effect number per model and pillar
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Answers shown", f"{len(df):,}")
    c2.metric("Models shown", df["Model"].nunique())
    c3.metric("Scenarios (all 4 tests)", df["Variant"].nunique())
    parts = []
    for pillar, (label, _, _) in PILLARS.items():
        t = compare_models(df[df["Pillar"] == pillar])
        if not t.empty:
            parts.append(t[["Model", "effect_pp"]].assign(Pillar=label))
    # test 4 has no single control, so its column = confident summaries of the fake study
    # with no hint, minus the same thing when the hint is given
    p4o = df[(df["Pillar"] == "Pillar_4_Confabulation") & df["answered"]]
    rates = p4o.groupby(["Model", "Condition"])["outcome"].mean().unstack() * 100
    if {"Fake_NoCue", "Experimental"} <= set(rates.columns):
        parts.append((rates["Fake_NoCue"] - rates["Experimental"]).rename("effect_pp")
                     .reset_index().assign(Pillar="4 Confabulation"))
    if parts:
        h = pd.concat(parts)
        st.subheader("How much did each test change each model's answers?")
        st.caption("Each number = how much higher the % was in the Experimental version than in Control (3% to 100% shows as 97). "
                   "Red = the manipulation shifted the model a lot, white = it made no difference. "
                   "For 4 Confabulation the effect is how much more often the model invented a summary of "
                   "a made-up study when it was NOT given the 'DOES_NOT_EXIST' hint.")
        st.altair_chart(heatmap(h, in_order(h["Model"].unique())))

# pillars 1-3 all work the same way
for tab, (pillar, (label, xtitle, blurb)) in zip(pillar_tabs, PILLARS.items()):
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
        st.altair_chart(simple_bars(t, xtitle))
        st.caption("Grey = the normal question, purple = the question with the nudge added. "
                   "The further apart the two bars, the more the model was swayed. "
                   "Each bar is based on about 30 answers, so a small gap could just be luck, "
                   "while a big gap is much more trustworthy.")
        shown = t[["Model", "pct_control", "pct_experimental"]].round(0).rename(columns={
            "pct_control": "Control %", "pct_experimental": "Experimental %"})
        st.dataframe(shown, hide_index=True)

# pillar 4 has more than two conditions, so it gets its own layout
with tab_conf:
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
            y=alt.Y("Model:N", sort=in_order(counts["Model"].unique()), title=None, axis=alt.Axis(labelLimit=260)),
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
with tab_data:
    st.caption("Every answer behind the charts, with the exact prompt and the model's raw reply.")
    cols = ["Pillar", "Condition", "Model", "Variant", "Rep", "Parsed_Meaning", "Raw_Response", "Prompt"]
    st.dataframe(df[cols])
    st.download_button("Download these rows as csv", df.to_csv(index=False).encode("utf-8"),
                       file_name="illusion_of_reasoning_filtered.csv")
