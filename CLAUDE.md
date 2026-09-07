# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A **Miuul Data Science Bootcamp** study workspace, not an application. It contains course scripts
for the "Feature Engineering & Data Pre-Processing" module. There is no package, no entry point,
no build, no lint config and no test suite — code is run interactively, block by block, in
PyCharm's Python Console.

The user is a bootcamp student following along with an instructor. Communication is in **Turkish**;
code comments in the working files are Turkish too.

## File roles (this is the important part)

| Path | Role |
|---|---|
| `Downloaded/feature_engineering.py` | **Instructor's reference.** 1112 lines, complete A→Z, correct. Read it, quote it, adapt it — but never treat it as the user's work and never modify it. |
| `Fengineering/FE1.py` | **The user's working file.** ~500 lines. They retype/adapt the reference here at their own pace. All new work goes here. |
| `scriptMıuul.py` | Loose practice snippets from earlier modules (comprehensions, numpy, matplotlib). Unrelated to the FE module. |
| `dataset/titanic.csv` | 60 KB. The main teaching dataset. |
| `dataset/application_train.csv` | **166 MB.** Used only to demo functions on a wide dataset. Loading it costs 5–10 s. |
| `pycharm_troubleshooting_log.md` | Log of environment freezes/errors already diagnosed and fixed. Read before debugging a slow/frozen IDE. |
| `GEMINI.md` | Prior assistant instructions for this same project (superseded by this file). |

### Default workflow for "what's next?" / "help me continue"

Find where `FE1.py` stops, locate the matching section in `Downloaded/feature_engineering.py`,
and supply the next block adapted to this environment. As of the last check, `FE1.py` ends at
*Missing Values → Eksik Değerlerin Yakalanması* (`missing_values_table`); the reference continues
at "Eksik Değer Problemini Çözme" (`feature_engineering.py:407`).

Reference section order: Outliers (catch → IQR thresholds → `grab_col_names` → remove/re-assign →
LOF) → Missing Values → Encoding (Label / One-Hot / Rare) → Scaling → Feature Extraction →
end-to-end Titanic pipeline with a RandomForest score (`feature_engineering.py:915`).

## Running code

Interpreter: `C:\Users\infka\Desktop\miuulV1\.venv\Scripts\python.exe` (PyCharm SDK name
"Python 3.14 (miuulV1)"). It lives **outside** this project, shared with the `miuulV1` project,
and is the only one here with `missingno` installed — the system `C:\Python314\python.exe` lacks it.

```bash
"C:\Users\infka\Desktop\miuulV1\.venv\Scripts\python.exe" Fengineering\FE1.py
```

Always run from the project root: the loader functions use relative paths
(`pd.read_csv("dataset/titanic.csv")`), so any other working directory raises `FileNotFoundError`.

`FE1.py` is divided into `# %%` cells. In PyCharm the user runs a cell with **Ctrl+Enter**
(`Alt+Shift+E` is unreliable in their setup). Keep new code cell-shaped and self-contained —
re-declare `df = load()` at the top of a cell rather than depending on console state.

## Hard constraints

**Never `import` anything from `Downloaded/feature_engineering.py`.** It is a flat script, so an
import executes all 1112 lines: 4 loads of the 166 MB CSV, 7 blocking `plt.show()` windows, and a
RandomForest fit. This froze the IDE once already and the import line was deliberately removed from
`FE1.py`. Copy the function body instead.

**Python 3.14 / pandas 3.0 dtype break.** The reference checks text columns with `dtypes == "O"`
(`feature_engineering.py:160-169`, and in the `fillna` lambdas at 427/437/498). Under this pandas
version string columns load as native `str`/`string`, so `"O"` misses them and they leak into
`num_cols`, producing `TypeError: unsupported operand type(s) for -: 'str' and 'str'` in the
quantile math. Whenever adapting reference code, translate the check to:

```python
dataframe[col].dtype.name in ["object", "category", "string", "str"]
```

`FE1.py`'s `grab_col_names` is already fixed this way — keep new helpers consistent with it.

**Don't load `application_train.csv` unless the block genuinely needs a wide dataset**, and never
load it into a variable that gets overwritten by the Titanic data a few lines later.

**Don't add `dataset/` to the IDE index.** `.idea/miuul2.iml` marks it `excludeFolder` specifically
to stop PyCharm from indexing the 166 MB file. Pandas can still read it.

## Working style for this repo

- Write code **only when asked**. Otherwise explain, point at the reference, and let the user type.
- On an error: fix it and hand back working code first. Skip the theory lecture.
- After any code, add a short plain-language explanation of the logic and the key methods.
- Keep the user's existing naming even where it drifts from the reference
  (`outlier_threshold` vs the reference's `outlier_thresholds`, `remove_outliers` vs
  `remove_outlier`) — matching their file matters more than matching the instructor's.
