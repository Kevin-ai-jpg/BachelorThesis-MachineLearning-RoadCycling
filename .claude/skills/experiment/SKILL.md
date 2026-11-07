---
name: experiment
description: Design and run one experiment fast from a plain-language question (e.g. "does GMM beat k-means on the power-curve features?"), log it to MLflow and write a one-page result card with a recommendation. Use whenever Kevin wants to test an idea, compare settings or check a result.
argument-hint: "[question to test]"
---

Kevin wants to answer: **$ARGUMENTS**

Goal: a trustworthy answer as fast as possible, so Kevin can decide. Not polished code.

1. **Frame it** (show Kevin, 3 lines max): the question, the metric that answers it (and whether higher
   or lower is better), the baseline it is compared against, and what result would change a decision.
   If the question is ambiguous or needs data that doesn't exist yet, ask one short question first.
2. **Config**: write `configs/<area>_<short-name>.yaml` (experiment name = the module area, e.g.
   `profiling`, `response`, `pacing`, `placement`). Every parameter that varies goes in the config.
3. **Code**: reuse `src/cyclingml/`. Add only what is missing, with a unit test for any new feature
   or metric function. Keep throwaway exploration in `notebooks/scratch/` (not committed).
4. **Run** through `cyclingml.tracking.run`, one run per setting, fixed seeds. Splits by rider (and by
   time for prediction tasks), never random rows. If the run will take more than ~10 minutes, don't
   start it: give Kevin the exact command to run (laptop overnight, Kaggle or Colab) and stop.
5. **Compare** with `python -m cyclingml.runs <experiment> --metric <metric>`. Report the best run,
   the baseline, the gap, and how stable it is (spread across seeds or folds). Say plainly if the
   difference is within noise.
6. **Result card**: write `docs/experiments/YYYY-MM-DD-<short-name>.md` from the template in
   `docs/experiments/README.md`: question, setup, results table, verdict, recommendation, next step.
7. **Reply to Kevin in 5 lines or fewer**: the answer, the key number, the recommendation (keep / drop /
   test further), and what to decide. Don't record a decision yourself: that is `/decide`.

Rules: never tune on the test set; report negative results as clearly as positive ones; if a number
looks too good, check for leakage (or ask the `leakage-reviewer` agent) before reporting it.
