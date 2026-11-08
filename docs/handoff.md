# Handoff: moving to Claude Code + VS Code

*Written 2026-11-08 at the end of the claude.ai phase. Read with `CLAUDE.md`.*

## 1. The project in five lines

- **Thesis:** machine learning for road cycling (BSc, Babeș-Bolyai University). Kevin directs and runs experiments; Claude writes the code.
- **Deliverable:** a Streamlit demo by **end of February 2027**. It takes a rider's Garmin/Strava export and shows rider type (clustering), a training recommendation (training-response model), a pacing plan, and placement vs Continental / ProTeam / WorldTour (percentiles + Monte Carlo season simulation).
- **Priority:** rigorous ML evaluation over UI polish. Kevin learns each module once he is ≥ 85% sure it stays (`/learn`).
- **Plan:** "Thesis App Plan: ML for Road Cycling" in the claude.ai Project *ThesisProjectApplication* (8 phases, task numbers like 2.3).
- **Repo:** https://github.com/Kevin-ai-jpg/BachelorThesis-MachineLearning-RoadCycling (`main` is current; PRs #1–#6 merged).

## 2. Status

| Phase / task | Status | Notes |
| --- | --- | --- |
| 0 · Setup (package, CI, Colab, MLflow) | Done | CI on Python 3.10 + 3.12; MLflow = SQLite file in `CYCLINGML_MLRUNS` |
| 1.1 · GoldenCheetah download | Done | 6,616 athlete zips on Drive, 0 failed |
| 1.2 · Index + data audit | Done | Gate 1 **passed**, see section 3 |
| 1.3 · Pro reference tables | Waiting | Kevin gets the PDFs (Leo 2021/2022/2024, Pinot & Grappe 2011) in the next 2–3 weeks; needed from Phase 3 on |
| 1.4 · ProCyclingStats scraper | Not started | Needed for Phase 5 |
| 1.5 · Amateur intake (consent, export guide, anonymiser) | Not started | ~10 riders' data expected next week or so |
| 1.6 · Team emails | Closed | Teams declined; the plan never depended on team data |
| 2 · Preprocessing and features | **Next** | Start with 2.1–2.3 |

## 3. Audit results (task 1.2)

- **Gate 1 passed: 2,799 usable riders** (≥ 100 power rides over ≥ 180 days), 500 required.
- **Typical usable rider:** 285 power rides over 2.2 years, about 3 per week; median best 20-min power 295 W (trained amateurs). 685 riders have 500+ power rides.
- **Rides by count:** ≥ 20 power rides 4,711 · ≥ 50: 3,755 · ≥ 100: 2,904 · ≥ 200: 1,915 · ≥ 500: 685.
- **Body weight:** present on 100% and plausible (40–130 kg) on 99.9% of power rides. 5.1% are exactly 75 kg (possible profile default); **2,649 usable riders have weights that look real**. Weights are often rounded (72, 80, 70 kg), so W/kg carries about ±3% error.
- **Implausible values:** < 0.03% of rides, but some rider maxima are absurd (best 20-min 2,977 W; one rider with 1.5 million hours), from spikes and broken device clocks. Recompute everything from cleaned samples in Phase 2.
- **Sports:** Bike 1.77 M activities · **unlabelled 280 k** (excluded so far) · Run 256 k · VirtualRide 56 k · MTB 8.5 k · Rennrad 4.6 k (German "road bike", currently missed) · Road 3.5 k.
- **Missing from this handoff:** the report's Overview section (athlete and activity totals, date range, % female). Re-run the last cells of `02_data_audit.ipynb` or read `audit_report.md` on Drive.

## 4. Decisions proposed, waiting for Kevin's OK

Record each with `/decide` once agreed:

1. **D1:** proceed to Phase 2 (Gate 1 passed).
2. **D2:** use W/kg for riders whose weight looks real; flag riders with mostly 75 kg.
3. **D3:** recover unlabelled activities as cycling when they have power plus cycling-like cadence and speed. Validate the rule on labelled Bike vs Run first; add missed labels such as "Rennrad".
4. **D4:** Phase 2 cleaning rules: drop activities over 12 h or with impossible speed, remove power spikes before computing power curves, take rider bests from cleaned data only.

Also proposed, to add to the plan:

- **D5:** Chronos-Bolt (Hugging Face) as an extra baseline in Phase 4.
- **D6:** a personalisation comparison in Phase 4 (global model vs per-rider fine-tuned / mixed-effects).
- **D7:** host the final demo on Hugging Face Spaces with anonymised sample data only.

## 5. Where the data is

| What | Where (Google Drive) |
| --- | --- |
| Root | `My Drive/thesis-cycling/data` = `CYCLINGML_DATA` |
| Athlete zips (6,616) | `raw/goldencheetah/zips/` |
| OSF manifest | `raw/goldencheetah/manifest.csv` |
| Index (metadata only) | `raw/goldencheetah/index/athletes.parquet` + `rides/part-*.parquet` + `progress.tsv` |
| Audit report | `raw/goldencheetah/audit_report.md`, plus `index/riders.parquet` (per-rider table) |
| MLflow | `mlruns/mlflow.db` |

**On the laptop:** install Google Drive for desktop and set `CYCLINGML_DATA` to the local path of that folder. Mark `raw/goldencheetah/index` *Available offline*. Phase 2 reads the per-second CSVs inside the zips, so check the size of `zips/`; if it fits on disk, make it available offline too, otherwise Drive streams each zip on demand (slower).

## 6. Laptop setup (once)

```bash
git clone https://github.com/Kevin-ai-jpg/BachelorThesis-MachineLearning-RoadCycling.git
cd BachelorThesis-MachineLearning-RoadCycling
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[all]"
pytest                            # expect all tests to pass
```

Windows, once (then restart VS Code):

```
setx CYCLINGML_DATA "G:\My Drive\thesis-cycling\data"
```

(Use your actual Drive letter and path.)

Open the folder in VS Code and accept the recommended extensions (`.vscode/extensions.json`):

| Extension | Why |
| --- | --- |
| **Claude Code** (`anthropic.claude-code`) | Claude inside VS Code; can also run notebook cells (asks first) |
| Python + Pylance | Interpreter, navigation, type hints |
| Jupyter | Run notebooks locally |
| Ruff | Lint + format on save (same rules as CI) |
| Data Wrangler | Browse parquet/pandas tables |

## 7. Claude Code: what's already in the repo

Loaded automatically when you open the folder: `CLAUDE.md`, `.claude/settings.json`, skills and agent.

| Command | Use |
| --- | --- |
| `/next` | Where we stand and the single next step |
| `/experiment <question>` | Frame, run and log an experiment; result card in `docs/experiments/` |
| `/results <experiment>` | Compare runs: best setting, gap vs baseline, noise or real |
| `/decide <decision>` | Record a decision (`docs/decisions.md`) and module status (`docs/module_status.md`) |
| `/learn <module>` | Walkthrough on real data plus exam questions, for frozen modules |
| agent `leakage-reviewer` | Independent check before trusting a good result |

A hook runs ruff fix + format after every Python edit. Tests, lint and `python -m cyclingml.*` commands run without permission prompts.

## 8. Plugins

**None are required.** The repo's skills cover the workflow, and every plugin enabled adds to each session's context. Optional, from the official marketplace (`/plugin` → Discover):

| Plugin | Why | Needs |
| --- | --- | --- |
| `pyright-lsp` | Better Python code navigation and type errors for Claude | `pip install pyright` |
| `code-review` | Extra review pass on pull requests | — |

Install them only if you feel the need.

## 9. Lessons learned (avoid repeating)

- **Colab keeps old code in memory.** Setup cells now drop cached `cyclingml` modules; if in doubt, *Runtime → Restart session*.
- **Google Drive disconnects in Colab on long jobs** (`Transport endpoint is not connected`). The index resumes and remounts by itself. The laptop is more reliable for long runs.
- **OSF rate limits (HTTP 429).** The downloader uses a shared throttle. An `OSF_TOKEN` (free OSF account) helps if needed again.
- **Broken device clocks:** some activities are dated year 1, and some durations are absurd. The reader and loader skip dates outside 1990–2030; Phase 2 must also cap durations.
- **GoldenCheetah CSV names use local time; metadata uses UTC.** They are matched on mm:ss within ±15 h (90% matched on the sample athlete; the rest are mostly manual entries).
- **Strava API data must not be used for training** (Strava API terms).

## 10. Next-step prompt (paste into Claude Code)

```text
Read docs/handoff.md and CLAUDE.md first.

1. Decisions: I agree with D1–D4 in docs/handoff.md section 4. Record them with /decide and
   set "Data audit" to frozen in docs/module_status.md. Leave D5–D7 as proposals for now.

2. Phase 2, tasks 2.1–2.3, on branch task-2.1-2.3-readers-cleaning-power-curve:
   - 2.1 Readers: FIT (fitdecode) and GPX (gpxpy) into the standard 1 Hz frame from CLAUDE.md,
     same columns as cyclingml.io.goldencheetah.read_ride_csv.
   - 2.2 Cleaning: resample to 1 Hz, fill short gaps (<= 5 s) and mark longer stops, keep coasting
     zeros, remove power spikes, drop activities > 12 h or with impossible speed (D4). All
     thresholds in a YAML config.
   - 2.3 Power curve: best average power for 1 s to 3 h per activity (fast cumulative-sum method),
     plus each rider's rolling 90-day best. Test against a brute-force implementation on
     synthetic data.
   Unit tests for everything; open a PR when CI passes.

3. Then run /experiment "Which unlabelled GoldenCheetah activities are rides?" using the index
   at $CYCLINGML_DATA/raw/goldencheetah/index (D3). Build the rule on labelled activities first
   (Bike/VirtualRide/MTB/Road/Rennrad vs Run), report its precision and recall, then apply it
   to the 280k unlabelled ones and report how many usable riders and rides it adds.

Keep replies short: results, a recommendation, and anything I must run myself (with the exact
command) if it takes more than 10 minutes.
```

## 11. Open items for Kevin

- [ ] Laptop setup (section 6) and Drive for desktop; check the size of `zips/`
- [ ] Paste or find the audit Overview section (section 3)
- [ ] PDFs for task 1.3 within 2–3 weeks
- [ ] Amateur riders: consent and exports (task 1.5; ask Claude for the consent form and export guide)
- [ ] Decide on D5–D7 and update the plan doc
- [ ] Check the faculty's rules on AI assistance and how to disclose it
