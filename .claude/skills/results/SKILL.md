---
name: results
description: Summarise logged MLflow runs for an experiment area (profiling, response, pacing, placement, data-audit) into a short comparison and a recommendation. Use when Kevin asks what the results are, which setting won, or pastes run output.
argument-hint: "[experiment name] [metric]"
---

Summarise results for: **$ARGUMENTS**

1. If no experiment is named, list them: `python -m cyclingml.runs`.
2. Get the table: `python -m cyclingml.runs <experiment> --metric <metric>` (add `--ascending` when
   lower is better, like MAE). If Kevin pasted output from Colab or his laptop instead, use that.
3. Read the matching result cards in `docs/experiments/` for context.
4. Answer in this shape, short:
   - **Best so far:** run, settings, metric (vs baseline: +/- and %)
   - **How sure:** spread across seeds/folds; is the gap bigger than the noise?
   - **Pattern:** which parameter matters most (one sentence)
   - **Recommendation:** keep / drop / test X next, and whether this part looks ready to freeze
     (≥ 85% likely to stay in the app; see `docs/module_status.md`)
5. Point out anything suspicious: a metric that jumped, runs from an old git commit, missing baseline.
