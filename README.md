# Swayed or Steady?

**Testing AI chatbots on classic cognitive psychology experiments**

AI chatbots work by predicting the next most likely word, one piece at a time. Critics say that means they don't really understand anything, so can you trust what they tell you? Would you bet on them picking the right word? I wanted to test that properly instead of just arguing about it.

I used 9 versions of OpenAI and Anthropic models: a mix of older and newer models, full-size and smaller versions, and some with thinking switched on. I presented them with four classic psychological tests: the decoy effect, the bandwagon effect, the framing effect, and confabulation (making things up). Each test checks whether the model gives in to a nudge that shouldn't change a good answer.

**Live dashboard:** https://swayed-or-steady.streamlit.app (start with the "About the tests" tab)
**Paper:** still writing it, I'll put the link here when it's up.

## The four tests

Each test has a Control version and an Experimental version of the same question. Only one thing changes between them, so if the model's answers change, it's because of the one thing I changed.

| Test | What I change in the Experimental version | What counts as being swayed |
|---|---|---|
| 1. Decoy | I add a worse "decoy" option next to the premium one | The model picks the premium option more often |
| 2. Bandwagon | A fake survey says most experts believe the wrong answer | The model gives the wrong answer |
| 3. Framing | Same choice, but worded as lives saved vs lives lost | The model picks the risky gamble more often |
| 4. Confabulation | The model is asked to summarise a study that doesn't exist | The model writes a confident summary instead of saying it can't verify it |

Test 4 has four versions: a real study (the control), a fake study where the prompt says the model can answer "DOES_NOT_EXIST", a fake study with no hint at all, and a fake study where the user says they need to cite it.

## Process

- **Models:** gpt-4o-mini, gpt-4o, gpt-5.4-mini, gpt-5.5, claude-haiku-4-5, claude-sonnet-4-6, claude-sonnet-5-5, plus versions of both Sonnets with thinking switched on. That's 9 in total.
- **Scenarios:** 6 different scenarios for each test, each run 5 times, so 30 answers per model per condition.
- **Fairness:** every model gets exactly the same prompts, and the order of the options is shuffled but kept the same across models.
- **Scoring:** a script reads each answer and labels it. For test 4, it decides whether the model admitted it didn't know the study or confidently made up a summary. To check it was working, I labelled 60 answers by hand and used them to fix the script's rules. Then I labelled 40 new answers it had never seen, and the script matched me on all 40.

**How I compared them:** To measure the impact, I compared how frequently each model gave the swayed answer across both versions. With a sample of 30 answers per condition, only the larger gaps are big enough to trust over random chance.

**Temperature (randomness):** I set 0.7, but several models don't allow it, so only gpt-4o and gpt-4o-mini actually ran at 0.7 and the rest used their default.

## What I found so far

- **Decoy:** every model picked the premium option 90-100% of the time once the decoy was added. Without the decoy they varied a lot, from 3% for Sonnet 4.6 to 100% for Haiku. Sonnet 4.6 moved the most (3% to 100%). Haiku already picked the premium option every time without the decoy, so it couldn't go any higher.
- **Bandwagon:** 8 of the 9 models never gave a wrong answer in either version. Only Haiku slipped, giving a wrong answer 17% of the time with the fake survey.
- **Framing:** the wording changed the older models a lot. In the "lost" wording, GPT-4o picked the gamble 97% of the time (0% in the "saved" wording), GPT-4o mini 90% (3%) and Sonnet 4.6 57% (0%). Haiku 4.5 and Sonnet 4.6 with thinking moved less (30% and 37%). GPT-5.5 and both Sonnet 5.5 versions barely changed (0-3%), and GPT-5.4 mini moved a bit (13% to 33%).
- **Confabulation:** when the prompt gave no hint that the study might be fake, only GPT-4o and GPT-4o mini wrote confident summaries of studies that don't exist. Every other model said it couldn't verify them.

**Putting it together:** the newer models did better on framing and confabulation. On framing the older models were swayed much more, and only the two oldest OpenAI models (GPT-4o and GPT-4o mini) made up summaries of fake studies. Decoy was different: every model ended up picking the premium option almost every time, old and new. In summary, the newer models performed better in two out of the four tests. This doesn't show AI is always trustworthy. It shows newer models resisted some of these nudges much better in my setup.

## Limitations

- It's only two companies and nine models.
- Which models count as "old" and "new" is my own grouping.
- "Thinking" means the setting was switched on, not that I checked the model used it, and the thinking models are also newer, so I can't fully separate the two.
- Temperature (randomness) wasn't the same for every model.
- 6 scenarios x 5 repeats aren't really 30 separate tests, so the gaps look a bit more certain than they are.
- Some tests couldn't show differences for some models. Almost none got the bandwagon questions wrong, and Haiku was already at 100% on the decoy test.
- My first confabulation prompt told the model it could say "DOES_NOT_EXIST", which made it too easy, so I added the no-hint versions.
- The prompts are short and artificial, and they ask for a one-word answer, which isn't how people normally use chatbots.
- The results are a snapshot from October 2026, and some of these model versions will get retired.

## What's in this repo

| File | What it does |
|---|---|
| `ai_benchmark.py` | Sends the prompts to the APIs, saves the answers, and cleans up the results file |
| `ai_trust_results_v4.csv` | All 2,700 answers, with the exact prompt, the model's reply, and the label |
| `analysis.py` | Turns the results into simple tables in a `results` folder |
| `p4_handcheck.py` | Makes the samples I labelled by hand for test 4 and scores the agreement |
| `app.py` | The Streamlit dashboard |
| `requirements.txt` | The packages the dashboard needs |

## Running it yourself

1. Install the packages: `pip install openai anthropic pandas numpy streamlit altair`
2. Add your own API keys as environment variables (`OPENAI_API_KEY` and `ANTHROPIC_API_KEY`). There are no keys in this repo.
3. Check which models you can reach: `py ai_benchmark.py --preflight-only`
4. Run the benchmark: `py ai_benchmark.py --reps 5`. This costs real API credits, and some of these models may have been retired by now.
5. Clean up the results file: `py ai_benchmark.py --reparse`
6. Make the summary tables: `py analysis.py`
7. Open the dashboard: `py -m streamlit run app.py`

## A note on AI help

I built this with help from AI assistants (Claude and Gemini): they assisted with the code, and I used them for feedback on the study design and for checking the analysis. The idea, the choice of tests and models, and the hand-labelling of the test 4 answers were mine. The tables and charts come from code that counts the answers in the results file, so anyone can rerun it and get the same numbers. I also checked a sample of the figures by hand against the raw data in Excel, and they matched.
