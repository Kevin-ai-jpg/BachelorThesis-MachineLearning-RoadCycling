# Module status

A module is **frozen** when Kevin is ≥ 85% sure it stays in the app; then it is learned with `/learn`.
Status: `exploring` → `candidate` → `frozen`. Updated by `/decide`.

| Module | Plan tasks | Status | Chosen approach | Evidence | Learned |
| --- | --- | --- | --- | --- | --- |
| Project setup (package, CI, MLflow) | 0.1–0.5 | frozen | One package, env-based paths, MLflow SQLite | Smoke run b114d86 on Colab | no |
| OSF downloader | 1.1 | frozen | Own OSF API client, shared throttle, resumable | 6,616 athletes listed; download running | no |
| GoldenCheetah reader + index | 1.1–1.2 | candidate | Metadata index (athletes + rides parquet), resumable | Real sample athlete: 731 activities parsed | no |
| Data audit | 1.2 | candidate | Thresholds in configs/audit_goldencheetah.yaml | Waiting for full index | no |
| Pro reference tables | 1.3 | not started | | Waiting for PDFs | no |
| PCS scraper | 1.4 | not started | | | no |
| Amateur intake + anonymiser | 1.5 | not started | | | no |
| Features (power curve, CP, load, durability) | 2.x | not started | | | no |
| Rider-type profiling | 3.x | not started | | | no |
| Training-response model | 4.x | not started | | | no |
| Pacing | 5.1–5.3 | not started | | | no |
| Pro placement + season simulation | 5.4–5.7 | not started | | | no |
| Streamlit app | 6.x | not started | | | no |
