# CLAUDE.md

Bachelor thesis (Babeș-Bolyai University): machine learning for road cycling. Kevin directs and runs
experiments; Claude writes the code. The plan with task numbers lives in the claude.ai Project
"ThesisProjectApplication" (doc: "Thesis App Plan: ML for Road Cycling"). Branch names and pull
requests reference the task number, e.g. `task-2.3-power-curve`.

## Goal
A Streamlit demo (end of February 2027) that takes a rider's Garmin/Strava export and shows:
rider type (clustering), a training recommendation (training-response model), a pacing plan, and
placement vs Continental / ProTeam / WorldTour riders (percentiles + Monte Carlo season simulation).
Priority if time is short: rigorous ML evaluation over UI polish.

## Layout
- `src/cyclingml/` — all logic. App and notebooks only call into it.
  - `io/` readers + downloaders · `clean/` · `features/` · `models/` · `eval/`
  - `paths.py` — every path comes from env vars (`CYCLINGML_DATA`, `CYCLINGML_MLRUNS`)
  - `config.py` — YAML configs · `tracking.py` — MLflow runs
- `configs/` one YAML per experiment · `notebooks/` Colab notebooks · `app/` Streamlit
- `reference/` small versioned CSVs (pro power profiles), every row cites its paper
- `data/` never committed (personal data, large files)
- `docs/decisions.md` decisions log — append when a choice is made

## Commands
```
pip install -e ".[dev,track]"        # add ml,app as needed, or .[all]
pytest                                # tests
ruff check . && ruff format .         # lint + format (CI fails otherwise)
python -m cyclingml.smoke configs/smoke.yaml   # setup check, logs one MLflow run
mlflow ui --backend-store-uri sqlite:///data/mlruns/mlflow.db
```

## Conventions
- Python ≥3.10, type hints, numpy/pandas; ride data = 1 Hz Parquet with columns
  `time_s, power_w, hr_bpm, cadence_rpm, speed_mps, altitude_m, distance_m` (missing = NaN).
- Power units: watts; W/kg only where body mass is known. Durations in seconds.
- Every feature function gets a unit test, ideally against synthetic data with a known answer.
- No future data in features: splits are by rider and by time; a leakage test guards each dataset.
- Experiments: config in `configs/`, run via `cyclingml.tracking.run`, never hard-coded parameters.
- Anything Kevin must run (long jobs, Colab) ships as a notebook or CLI command plus the exact
  command in the pull request description.
- Kevin defends this code at the exam: keep it readable, explain non-obvious maths in docstrings
  with the source (paper) cited.

## Data sources
GoldenCheetah OpenData (training) · ~10 amateur riders with consent (case studies) · published pro
profiles: Leo et al. 2021/2022/2024, Pinot & Grappe 2011 (reference) · ProCyclingStats via the
`procyclingstats` package, cached and rate-limited (calendars, stages, climbs). Strava API data must
not be used for model training (Strava API terms).
