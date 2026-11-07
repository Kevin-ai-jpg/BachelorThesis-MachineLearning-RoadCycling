---
name: decide
description: Record a decision Kevin has made (e.g. "use GMM with 5 clusters", "drop the LSTM") in the decisions log and update the module status. Use only when Kevin states a decision.
argument-hint: "[the decision]"
disable-model-invocation: true
---

Kevin decided: **$ARGUMENTS**

1. Add one row at the top of the table in `docs/decisions.md`: today's date · the decision · why
   (cite the result card or MLflow run that supports it, e.g. `docs/experiments/2026-11-20-gmm-vs-kmeans.md`).
2. Update `docs/module_status.md` for the affected module: status (`exploring` → `candidate` →
   `frozen`), the chosen approach, and the evidence. A module is `frozen` when Kevin is ≥ 85% sure it
   stays in the app; then set "Learn next" to yes.
3. If the decision changes defaults, update the config in `configs/` so the code matches the decision.
4. If a module just became `frozen`, tell Kevin it is ready for `/learn <module>`.
5. Confirm in one or two lines what was recorded.
