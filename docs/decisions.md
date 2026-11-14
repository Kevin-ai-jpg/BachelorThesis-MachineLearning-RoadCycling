# Decisions log

Newest first. One line per decision: date · decision · why.

| Date | Decision | Why |
| --- | --- | --- |
| 2026-11-08 | D4: Phase 2 cleaning rules: drop activities > 12 h or with impossible speed, remove power spikes before computing power curves, take rider bests from cleaned data only | Audit found absurd rider maxima (best 20-min 2,977 W, a 1.5-million-hour rider) from spikes and broken clocks; thresholds go in the Phase 2 config (task 2.2) |
| 2026-11-08 | D3: recover unlabelled activities as cycling when they have power plus cycling-like cadence and speed; validate the rule on labelled Bike vs Run first; add missed labels (e.g. "Rennrad") | 280 k unlabelled activities are excluded today; Rennrad (4.6 k) is German for road bike. Precision/recall to be reported in a result card before use |
| 2026-11-08 | D2: use W/kg only for riders whose weight looks real; flag riders whose weight is mostly 75 kg | 5.1% of power rides are exactly 75 kg (likely a profile default); 2,649 usable riders have real-looking weights; rounding gives ~±3% W/kg error (audit, `audit_report.md`) |
| 2026-11-08 | D1: proceed to Phase 2 | Gate 1 passed: 2,799 usable riders vs 500 required (task 1.2 audit, `docs/handoff.md` §3) |
| 2026-11-07 | Use GoldenCheetah's own per-activity metrics (TSS, NP, best power 1 s–90 min, weight) for the audit; recompute from samples later (Phase 2) | Fast audit without reading millions of CSVs; our own features stay verifiable |
| 2026-11-07 | Download GoldenCheetah OpenData from OSF with our own client (`cyclingml.io.osf`) | The S3 mirror is closed and the official Python/R tools only support S3 |
| 2026-11-07 | Usable rider (Gate 1) = ≥ 100 cycling power rides over ≥ 180 days; weight usable if ≥ 90% plausible and < 50% at a suspected default | Enough history for profiling and weekly training blocks; thresholds in configs/audit_goldencheetah.yaml |
| 2026-11-07 | MLflow with a SQLite file (no server) in `CYCLINGML_MLRUNS` | Works offline, on Colab with Drive, and in CI; MLflow 3 rejects plain-folder stores |
| 2026-11-07 | One package (`src/cyclingml`), app and notebooks only call it | Experiments and demo run identical code |
| 2026-11-05 | Streamlit app, English UI, demo by end of February 2027 | Fastest path to a working demo; Kevin's choice |
| 2026-11-05 | Modules: profiling, training response, pacing + pro placement; no LLM/RAG layer | Rigorous ML results prioritised |
| 2026-11-05 | GoldenCheetah OpenData as training data; no Strava API data for training | Strava API terms forbid AI/ML training use |
