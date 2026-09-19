# gbt-numpy port — `agent_code/mayank_gbt_np`, shipped as `mayank_agent`

**The submitted agent. Same decisions as the scikit-learn model, none of the
dependency: 600 boosted trees walked by numpy indexing, 0.787 ms per step.**

## Why this exists

`mayank_gbt` loads six fitted `HistGradientBoostingRegressor` estimators from a
joblib pickle. A pickle is only readable by a library version compatible with
the one that wrote it, and that file records `_sklearn_version 1.8.0`.
**scikit-learn >= 1.9.0 refuses it outright** with
`ModuleNotFoundError: No module named '_loss'`.

The failure lands inside `setup()`, so the agent dies before its first step
rather than playing badly — and because an unpinned `scikit-learn` resolves to
the newest release, **that crash is the default outcome in a freshly built
grading environment**. Pinning `scikit-learn==1.8.0` in `requirements.txt` was
rejected as a fix: the tournament image carries numba, pytorch and tensorflow,
each pinning a narrow numpy range, so forcing a resolve could take the container
down before the agent is ever called.

Exporting the tree structure to plain numpy arrays removes the dependency and
the pickle together. The shipped agent imports `os`, `json`,
`collections.deque`, `numpy` and the framework's own `settings` — **nothing
else** — and reads its model with `allow_pickle=False`, so loading it cannot
execute code either.

## What is in the model file

`analysis/export_gbt_numpy.py` flattens all 600 trees (100 boosting iterations ×
6 actions) into one node table, rewriting each tree's child indices from
tree-local to global so all six actions are walked in a single vectorised pass.

| | |
|---|---|
| Trees | 600 |
| Nodes | 36600 |
| Max depth | 20 |
| Arrays | 17 |
| `feature_version` / `n_features` | 6 / 38 |
| `learning_rate` / `leaf_scale` | 0.1 / **1.0** |
| File size | 401294 bytes |
| `load_model()` | 2.2 ms steady, 4.5 ms cold |

`leaf_scale` is 1.0 because scikit-learn applies the learning rate as shrinkage
when a leaf is finalised **during fitting**, not at prediction time, so the
stored leaf values already carry the factor 0.1 and the prediction is a plain
sum. The loss is `HalfSquaredError`, whose link is the identity, so `predict()`
and the raw prediction coincide and there is no inverse link to apply. Both
constants ride in the archive so the arithmetic stays auditable rather than
remembered:

```
Q(s, a) = baseline[a] + sum over that action's 100 trees of leaf_value
```

The traversal (`q_values()`) keeps one cursor per tree in a single array and
advances all 600 together, freezing finished trees with an `active` mask, so the
work per level is a handful of numpy gathers over 600 elements and the loop runs
at most `max_depth` times. The branch rule matches scikit-learn exactly,
including `node_missing_left` for NaN. The per-action sum is one `bincount` over
`tree_action`.

## Which checkpoint, and why

Exported from **`runs/gbt_v1_r20000`**, not the final 40000-round model:

| Checkpoint | Score | Coins | `killed_self` | Invalid rate |
|---|---|---|---|---|
| gbt_v1_r10000 | 3.445 ± 0.166 | 2.770 | 0.385 | 0.0046 |
| **gbt_v1_r20000** | **3.665 ± 0.194** | 2.715 | **0.265** | **0.0040** |
| gbt_v1_r30000 | 3.675 ± 0.191 | 2.700 | 0.325 | 0.0048 |
| gbt_v1_r40000 | 2.890 ± 0.162 | 2.440 | 0.515 | 0.0099 |

r20000 and r30000 are tied on score within one SEM, but r20000 suicides 6 points
less often and has a lower invalid rate. r40000 is clearly degraded — score down
0.8 and `killed_self` doubled against r20000.

## Verification

### 1. The port reproduces scikit-learn

`analysis/verify_gbt_numpy.py`, 10000 real game states collected by actually
playing `mayank_gbt` (5064 against `rule_based_agent`, 4936 against
`random_agent`), each scored twice — once through the fitted estimators under
scikit-learn 1.8.0, once through the exported arrays:

```
max |Q_sklearn - Q_numpy| : 1.492140e-13   (threshold 1e-5)
mean |difference|         : 1.920925e-14
Q range in reference      : [-56.779, 45.784]
argmax disagreements      : 0 / 10000
VERDICT: PASS
```

Difference is at float64 rounding, and **argmax — the only thing that decides
play — never disagrees once**.

### 2. The archive is bit-exactly reproducible from the pickle

Re-exporting every checkpoint and hashing the result identifies the source
unambiguously:

| Re-exported from | sha256 |
|---|---|
| gbt_v1 | `517e505a…` |
| gbt_v1_r10000 | `a66720eb…` |
| **gbt_v1_r20000** | **`e1e87861…` — matches the shipped npz** |
| gbt_v1_r30000 | `b90e3f8b…` |
| gbt_v1_r40000 | `517e505a…` (identical to gbt_v1) |

### 3. End-to-end: the shipped agent plays the sklearn model's games

200 rounds, `classic`, 3 `rule_based_agent`s, seeds 20240101–20240300:

| Metric | `mayank_gbt_np` (shipped) | `mayank_gbt` (sklearn) |
|---|---|---|
| Score | **3.665 ± 0.194** | 3.665 ± 0.194 |
| Coins | 2.715 | 2.715 |
| `killed_self` | 0.265 | 0.265 |
| Invalid rate | 0.0040 | 0.0040 |
| **Total steps** | **63627** | **63627** |

The step total matching exactly means every step of all 200 games was identical,
not merely the aggregates. For reference on the same seeds, the linear
`v9_urgency` agent scores **3.385**.

Full picture for the shipped agent: kills 0.190 ± 0.031, steps 318.1 ± 9.3,
win rate 0.390 ± 0.035, mean rank 2.095, rank histogram 78 / 55 / 37 / 30.
Action mix is balanced — `UP 20.1% · RIGHT 19.9% · LEFT 19.7% · DOWN 19.1% ·
WAIT 9.2% · BOMB 12.0%`.

### 4. Per-step inference time against the 500 ms budget

40 rounds vs 3 `rule_based_agent`s on `classic`, 13063 timed steps. Measured on
two independent clocks: the framework's own (`AgentRunner.process_event`, the
value `settings.TIMEOUT` is compared against) and a `perf_counter` wrap on the
same `act()` call.

| | direct | framework |
|---|---|---|
| Mean | 0.787 ms | 0.787 ms |
| Median | 0.758 ms | 0.759 ms |
| p95 | 1.329 ms | 1.329 ms |
| p99 | 1.510 ms | 1.510 ms |
| p99.9 | 1.681 ms | 1.681 ms |
| **Max** | **1.989 ms** | 1.990 ms |

Worst case is **0.40% of the 500 ms budget — 251× headroom**. Zero steps over
50 ms; the machine would have to be **25× slower** before one step touched that
threshold. The first `act()` call took 0.868 ms, below the median-plus-noise
band, so there is no cold-start artifact inflating the max; the ten slowest steps
are scattered across the run (positions 2163 … 11271 of 13063) on a few busy
mid-game boards. Per-round maxima are stable: min 0.845, median 1.542, max
1.989 ms, every one of the 40 rounds under 2 ms.

Measured on Apple M1, 8 cores, Python 3.14.2, numpy 2.4.4. The agent pins
`OMP_NUM_THREADS` and friends to 1 before importing numpy, and the traversal is
pure gather/`bincount` with no BLAS on the path, so these figures are
single-threaded — they neither gain from more cores nor degrade under contention
from the other three agents. `setup()`'s 2.2 ms model load is not charged against
the per-step timeout.

### 5. The submission archive

`final-project-agent-code.zip`, 406083 bytes, sha256 `198b9437029b7300…`:

```
mayank_agent/
mayank_agent/callbacks.py   36295   sha256 7e02f923a503c38f…
mayank_agent/train.py        2222   sha256 3f16fb16384c6271…
mayank_agent/model.npz     401294   sha256 e1e8786137d84233…
```

Four entries, nothing else — no `__MACOSX`, no `.DS_Store`, no `__pycache__`, no
`logs/`, no `runs/`. All three files are **byte-identical** to
`agent_code/mayank_gbt_np/`. The folder is renamed to `mayank_agent` so the
tournament name matches earlier submissions; nothing depends on it, because every
path is built from `os.path.dirname(os.path.abspath(__file__))`.

No absolute or machine-specific paths in `callbacks.py` or `train.py`. The npz
was scanned too: its 17 members are bare `*.npy` names with no directory
components, and it embeds no paths and no scikit-learn references.

### 6. Fresh-clone play with scikit-learn absent

A tree containing only the framework (`main.py`, `environment.py`, `agents.py`,
`settings.py`, `items.py`, `events.py`, `fallbacks.py`, `replay.py`, `assets/`),
`agent_code/rule_based_agent`, and the submission **extracted from the zip**.
`BOMBERMAN_RUN` unset; `sklearn`, `joblib`, `scipy` and `threadpoolctl` made
genuinely unimportable by a `sys.meta_path` hook raising `ModuleNotFoundError`.

```
[verify] sklearn / joblib / scipy / threadpoolctl are blocked
[callbacks] run '<none: agent directory>' -> …/agent_code/mayank_agent/model.npz
Loaded 600 trees over 36600 nodes, max depth 20.

10 rounds:  mayank_agent 42  |  rule_based_agent_2 34  ·  _1 24  ·  _0 10
```

Zero errors or exceptions in the agent log, zero "exceeded think time" warnings
in `game.log`. With `BOMBERMAN_RUN` unset, `run_dir()` correctly resolves to the
agent directory itself rather than a `runs/<name>/` subfolder.

`--train 1` was also exercised: `train.py` satisfies `AGENT_API`, prints its
"frozen export cannot be trained" warning to both the log and the console, and
plays on without crashing rather than turning an accidental flag into a dead
agent.

## Open items

- **`evaluate.py` cannot evaluate a run-less agent.** `resolve_model()` does
  `os.environ[api.RUN_ENV_VAR] = run` unconditionally, and `run` is `None` for
  this agent, so it dies with `TypeError: str expected, not NoneType`
  (`analysis/evaluate.py:104`). The 200-round numbers above came from an ad-hoc
  harness instead. One-line guard would fix it.
- **`requirements.txt` still describes the linear agent.** It says the tournament
  loads `agent_code/mayank_agent/callbacks.py` importing "json, os and
  collections … numpy, and settings" — which happens to remain true — but the
  `train.py` sentence ("adds csv, subprocess and datetime") describes the old
  linear `train.py`. The shipped `train.py` imports nothing at all.
- **`train.py` still calls itself `mayank_gbt_np`** in its warning string and
  points at `agent_code/mayank_gbt`, neither of which exists in the grader's
  tree. Cosmetic — the string is only reachable under `--train 1`, which a
  tournament never uses — and it was left alone deliberately to keep the shipped
  files byte-identical to the repository source.
- **A genuinely fresh clone needs `logs/` to exist.** `setup_logging()` opens
  `{log_dir}/game.log` without creating the directory
  (`environment.py:61`), so a clone that does not carry an empty `logs/` dies
  before the first round. This is the framework's own behaviour, not the agent's.
- **The timing harness is scratch-only.** Nothing in `analysis/` measures
  per-step think time; the numbers in section 4 came from an ad-hoc script that
  wraps `runner.callbacks.act` and `agent.wait_for_act`. Worth keeping as
  `analysis/time_inference.py` if the budget is ever questioned again.
