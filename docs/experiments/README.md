# Experiment result cards

One file per experiment: `YYYY-MM-DD-<short-name>.md`, written by `/experiment`.
These become the raw material for the thesis results chapter.

## Template

```markdown
# <Question in one line>

**Date:** YYYY-MM-DD · **Area:** profiling | response | pacing | placement · **Config:** configs/<file>.yaml
**Git commit:** <hash> · **MLflow experiment:** <name>

## Setup
Data (riders, rides, split), what was compared, metric (higher/lower is better), baseline.

## Results
| Run | Setting | Metric | vs baseline |
| --- | --- | --- | --- |

Spread across seeds/folds: ...

## Verdict
One or two sentences: what the numbers say, including whether the difference is within noise.

## Recommendation
keep / drop / test further — and what to try next.
```
