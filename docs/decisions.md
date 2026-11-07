# Decisions log

Newest first. One line per decision: date · decision · why.

| Date | Decision | Why |
| --- | --- | --- |
| 2026-11-07 | MLflow with a SQLite file (no server) in `CYCLINGML_MLRUNS` | Works offline, on Colab with Drive, and in CI; MLflow 3 rejects plain-folder stores |
| 2026-11-07 | One package (`src/cyclingml`), app and notebooks only call it | Experiments and demo run identical code |
| 2026-11-05 | Streamlit app, English UI, demo by end of February 2027 | Fastest path to a working demo; Kevin's choice |
| 2026-11-05 | Modules: profiling, training response, pacing + pro placement; no LLM/RAG layer | Rigorous ML results prioritised |
| 2026-11-05 | GoldenCheetah OpenData as training data; no Strava API data for training | Strava API terms forbid AI/ML training use |
