# Which unlabelled GoldenCheetah activities are rides? (D3)

**Date:** 2026-11-14 · **Area:** data-audit (task 1.2, D3) · **Config:** configs/data_sport_rule.yaml
**Git commit:** 44b95cb + this branch · **MLflow experiment:** data-audit (runs `d3-*`)

## Setup
Data: the full GoldenCheetah index, 2.59 M activities from 6,583 athletes. 279,847 have no sport
label, and many cycling activities carry labels the audit did not know ("Rennrad", "Rolle",
"Morning Ride", "BDC"…).

1. **Label aliases:** the audit's cycling labels plus 30 more (`labels.bike` in the config).
2. **Rule** (`cyclingml.clean.sport.rule_is_ride`): has power **and** average cadence 40–130 rpm
   (or no cadence sensor) **and** average speed either ≥ `speed_min` and ≤ 20 m/s (outdoor), or
   missing / below 0.5 m/s (indoor trainer).
   Running power meters (Stryd) record a cycling-like cadence (~85 strides/min), so average
   speed is what separates runs (median 3.0 m/s) from rides (median 7.5 m/s).

Evaluation pool: 1.31 M *labelled* activities with power (bike labels = positive; run, swim,
row, ski, walk… = negative; 97.3% are bikes). **Split by rider**, 50/50, seed 42 (2,963 riders
each side). `speed_min` and the indoor branch were chosen on TRAIN riders only (grid of 10
settings: highest recall with precision ≥ 0.99). Every number below is on TEST riders, with
a 95% bootstrap interval over riders.

## Results
| Method | Precision | Recall | Non-rides accepted (FPR) |
| --- | --- | --- | --- |
| Baseline: has power | 0.972 [0.965–0.980] | 1.000 | 100% |
| **Rule** (speed ≥ 3.5 m/s or indoor) | **0.995 [0.993–0.997]** | **0.984 [0.981–0.985]** | 15.8% |
| Decision tree (depth 4, same inputs + 4 more) | 0.992 [0.989–0.994] | 0.999 | 28.1% |

On the training riders, a stricter rule (speed ≥ 4.5 m/s) accepts only 4.4% of non-rides at
recall 0.958 (full grid: `grid_train.csv`, MLflow).

The rule's false positives on test riders are mostly runs (2,135) and swims (376, which have no
speed and so pass the indoor branch), then SkiErg and RowErg.

**Applied to the whole dataset:**

| | Power rides | Usable riders (Gate 1) | Usable with real weight |
| --- | --- | --- | --- |
| Before (audit) | 1,265,488 | 2,799 | 2,649 |
| + label aliases | 1,279,618 | 2,821 | – |
| + aliases + rule on unknown labels | **1,420,773 (+12.3%)** | **2,948 (+149)** | **2,793 (+144)** |

The rule accepts 141,155 of the 145,106 unknown-label activities with power (108,809 of them
have an empty label).

**Estimated contamination:** the rule rejects 3,951 unknown activities. From the test rates
(rejects 84.2% of non-rides and 1.65% of rides), the unknown pool holds ~1,900 non-rides, of
which ~300 are accepted, i.e. **~0.2% of the recovered activities are not rides**.

## Verdict
The rule clearly beats "has power" (precision 0.995 vs 0.972, intervals do not overlap) at a
1.6% recall cost. The tree is no better: lower precision, harder to explain. Precision looks high
partly because 97% of the pool are rides. The honest error rate is the 15.8% of non-rides that
still pass, which matters little here because the unknown pool is ~99% rides.

## Recommendation
Keep the rule (speed ≥ 3.5 m/s or indoor) plus the label aliases. Together they add 149 usable
riders and 155 k power rides. Optional: use speed ≥ 4.5 m/s if you prefer fewer false runs over
more rides. Swims passing as indoor rides could be blocked with a cadence > 0 requirement on the
indoor branch: a small follow-up, not needed for the decision.
