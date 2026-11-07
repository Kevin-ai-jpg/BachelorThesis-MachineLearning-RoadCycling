---
name: learn
description: Teach Kevin one module of the app (how it works, why, what the results mean) so he can explain and defend it at the thesis exam. Use when Kevin asks to learn or understand a module, or after a module is frozen.
argument-hint: "[module, e.g. profiling | goldencheetah-reader | osf-downloader | audit]"
---

Teach Kevin: **$ARGUMENTS**

Kevin learns a module once it is ≥ 85% sure to stay (see `docs/module_status.md`). If the module is
still `exploring`, say so and offer a short overview instead of the full walkthrough.

Build `docs/learning/<module>.md`, then walk Kevin through it in the chat, one part at a time:

1. **In one paragraph**: what problem the module solves in the app and in the thesis.
2. **The idea** in plain words, then the maths if any (formulas with the paper they come from).
3. **Walk through a real example**: run the module on real data (a GoldenCheetah athlete or Kevin's
   own ride) and show input → each step → output with actual numbers. Prefer one concrete example
   over general description.
4. **The code map**: the 3–6 functions that matter, file and line, what each does in one line.
   Skip boilerplate.
5. **Design choices and alternatives**: why this approach, what was tried or rejected and why
   (link `docs/decisions.md` and the result cards). This is what examiners ask.
6. **Results**: the key numbers and what they mean, including limitations.
7. **Exam check**: 5 questions an examiner could ask, with short model answers. Then ask Kevin
   2 of them and give feedback on his answers.

Go at Kevin's pace: after each part, check whether he wants more depth or to move on. When done,
set "Learned" to yes for the module in `docs/module_status.md`.
