# Do our reader, cleaning and power curve reproduce GoldenCheetah's best-power values?

**Date:** 2026-11-14 · **Area:** features (tasks 2.1–2.3) · **Config:** configs/clean_rides.yaml
**Git commit:** see PR for task-2.1-2.3 · **MLflow experiment:** none (validation check, no model)

## Setup
6 GoldenCheetah OpenData athletes: 5 random usable riders (seed 1, zips < 25 MB) plus the usable
rider with the most absurd metadata (best 20-min 2,977 W), as a stress test.
Each was run through `python -m cyclingml.rider <zip> --goldencheetah` (read CSV → clean to 1 Hz
→ best power 1 s–3 h), then `python -m cyclingml.eval.gc_check <dir>` compared, per power ride,
our best power with GoldenCheetah's own stored metric at 5 s, 1, 5, 20 and 60 min.
Metric: ratio ours / GoldenCheetah (1 = identical) and share of rides within ±2%.

## Results
| Rider | Kept / rejected / duplicate | Power rides | Median ratio (all durations) | Within ±2% |
| --- | --- | --- | --- | --- |
| gc1 | 99 / 1 / 0 | 13 | 1.000 | 100% |
| gc2 | 286 / 4 / 5 | 22 | 1.000 | 100% |
| gc3 | 225 / 2 / 0 | 214 | 1.000 | 97–100% |
| gc4 | 120 / 5 / 6 | 111 | 1.000 | 81% (5 s) – 100% |
| gc5 | 331 / 12 / 8 | 278 | 1.000 | 98–100% |
| gc6 (stress test) | 1158 / 104 / 0 | 1134 | 0.50–0.60 | 17% |

Runtime: 47–183 s per normal rider, 690 s for gc6 (1,262 activities) on the laptop, reading
from Google Drive.

Rejections: 105 activities longer than 12 h (104 of them gc6: broken clock), 21 shorter than
5 min, 2 with > 1% of samples faster than 120 km/h.

## What the deviations are
* **gc4, 5 s (17% of rides lower by > 5%):** our raw and cleaned values are identical, but
  GoldenCheetah's stored 5 s best is higher than any 5 s window in the exported samples.
  GoldenCheetah's metadata disagrees with its own CSVs; our cleaning is not the cause.
* **gc6:** GoldenCheetah's stored values are impossible: e.g. a ride whose highest single sample
  is 385 W has a stored 1-min best of 652 W. Our values follow the samples. This confirms D4
  (take rider bests from cleaned samples, not from the metadata).
* **gc6, open problem:** even after cleaning, gc6's top 20-min values (377–439 W, typical 152 W,
  99th percentile 279 W) come from minutes of noisy power (90th percentile ~700 W at HR 141).
  This is a faulty power meter producing sustained noise, which a 7-sample spike filter cannot
  catch by design.

## Verdict
On normal riders our pipeline reproduces GoldenCheetah exactly (median ratio 1.000, ≥ 97% of
rides within ±2% except gc4's 5 s). Every large deviation traced back to broken GoldenCheetah
metadata, not to our code.

## Recommendation
Keep the pipeline (move tasks 2.1–2.3 to *candidate*). Before computing rider bests for all
2,799 riders, decide how to guard against sustained power-meter noise (gc6), e.g. flag an
activity whose best power is far above the rider's own distribution or inconsistent with heart
rate. Proposed as D8 in the PR.
