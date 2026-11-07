# Machine Learning for Road Cycling

Bachelor thesis project, Faculty of Mathematics and Computer Science, Babeș-Bolyai University, Cluj-Napoca.

From a rider's power-meter data, the app:

1. **Profiles the rider type** (sprinter, puncheur, climber, time-trialist, all-rounder) by clustering power-duration curves.
2. **Predicts training response**: how 5- and 20-minute power will change after a training block, compared against the Banister fitness–fatigue model.
3. **Plans pacing** on a course using a physics model and W′ balance.
4. **Places the rider against pros**: percentiles vs Continental, ProTeam and WorldTour riders, plus a Monte Carlo simulation of a pro season.

## Quick start

```bash
git clone https://github.com/Kevin-ai-jpg/BachelorThesis-MachineLearning-RoadCycling.git
cd BachelorThesis-MachineLearning-RoadCycling
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,track]"
pytest
python -m cyclingml.smoke configs/smoke.yaml
```

On Google Colab, open `notebooks/00_colab_setup.ipynb`.

## Data

Raw and processed data are **not** in this repository. Set `CYCLINGML_DATA` to the data folder
(default `./data`). Sources: [GoldenCheetah OpenData](https://osf.io/6hfpz/), consenting amateur
riders (anonymised), published professional power profiles, and ProCyclingStats.

## Project layout

```
src/cyclingml/   io · clean · features · models · eval
app/             Streamlit app
notebooks/       Colab notebooks
configs/         experiment configs (YAML)
reference/       pro power-profile tables with citations
tests/           pytest
docs/            decisions log
```
