# v2 — `runs/v2_crates`

**Task 2 features and rewards, but exploration still excluded BOMB. Failed: score 0.**

## What changed from v1, and why

Task 1 was solved, so the agent moved to crates and bombs. Two changes:

1. **Feature set 9 → 21** (feature version 2). Added danger, neighbour
   survivability, bomb value, and crate direction, keeping the v1 bias, wall and
   coin slots unchanged.
2. **Reward table rewritten for Task 2** — `BOMB_DROPPED` went from a flat −5
   (correct for Task 1, where bombing is pure waste) to 0, with the value of a
   bomb now expressed by two new symmetric event pairs.

What was **not** changed: `act()` still excluded `BOMB` and `WAIT` from the
ε-greedy branch (`np.random.choice(['UP','RIGHT','DOWN','LEFT'])`), a Task 1
holdover. That omission is what sank the run.

## Config

| | |
|---|---|
| Feature version | **2** (21 features) |
| Features | `[0]` bias · `[1:5]` walls · `[5:9]` coin dir · `[9]` danger · `[10]` urgency · `[11:15]` neighbour safe · `[15]` bomb hits crate · `[16]` bomb has escape · `[17:21]` crate dir |
| ALPHA / GAMMA | 0.01 / 0.95 |
| Epsilon | 1.0 → 0.05, decay 0.999 |
| Exploration | **4 moves only — BOMB and WAIT never sampled** |
| Rounds trained | 5000 |
| Scenario | `classic` (crate density 0.75, 9 coins) |

Reward table:

| Event | Value |
|---|---|
| `COIN_COLLECTED` | +10.0 |
| `CRATE_DESTROYED` | +1.0 |
| `COIN_FOUND` | +1.0 |
| `MOVED_TOWARD_COIN` / `MOVED_AWAY_FROM_COIN` | +1.0 / −1.0 |
| `ESCAPED_BLAST` / `MOVED_INTO_BLAST` | +3.0 / −3.0 |
| `BOMB_NEXT_TO_CRATE` / `USELESS_BOMB` | +2.0 / −2.0 |
| `WAITED` | −1.0 |
| `INVALID_ACTION` | −3.0 |
| `BOMB_DROPPED` | 0.0 |
| `KILLED_SELF` | −50.0 |
| `GOT_KILLED` | 0.0 (fires alongside `KILLED_SELF`; non-zero would double-count) |

## Training curve

| Rounds | score | steps | reward | ε |
|---|---|---|---|---|
| 1970–2726 | 0.00 | 400.0 | −71.5 | 0.098 |
| 2727–3483 | 0.00 | 400.0 | −38.9 | 0.053 |
| 3484–4240 | 0.00 | 400.0 | −36.6 | 0.050 |
| 4241–5000 | 0.00 | 400.0 | −36.6 | 0.050 |

**Score is 0 for every logged round.** Every round runs the full 400 steps.

## Evaluation — `eval_classic.json` (30 rounds, greedy)

| Metric | Mean | SEM |
|---|---|---|
| Score | **0.000** | 0.000 |
| Steps survived | 400.0 | 0.0 |
| Suicide rate | 0.0000 | — |

Per-round scores: 0 in **30/30**.

## Diagnostics that mattered

**Action histogram** (12000 actions, persisted in `eval_classic.json`):

```
UP     4196  35.0%
RIGHT  1200  10.0%
DOWN   4204  35.0%
LEFT   1200  10.0%
WAIT   1200  10.0%
BOMB      0   0.0%   <- never chosen
```

**Position trace**, round 1 (seed 20240101) — a perfect 2-cycle for all 400 steps:

```
    1  (15,1)   DOWN    (15,2)
    2  (15,2)   UP      (15,1)     stepped back
    3  (15,1)   DOWN    (15,2)     stepped back
    ...
distinct tiles visited : 2
steps stepping back    : 399 / 400
VERDICT: the agent is stuck -- it never leaves these tiles.
```

**Weight magnitudes** (`mean |w|` per action row) explain why:

```
UP=0.239  RIGHT=0.273  DOWN=0.239  LEFT=0.240  WAIT=0.010  BOMB=0.144
```

`WAIT` sits at **0.010** — the random-initialisation scale (`rand()*0.01`), two
orders of magnitude below the movement rows. `update_weights` only touches the
row of the action actually taken, so an action that is never explored and never
greedy is **never updated at all**. WAIT was never taken; BOMB (0.144) was taken
a handful of times via the greedy branch very early, driven negative, and then
abandoned.

## Conclusion → v3a

The agent cannot break crates, so on `classic` no coin is ever reachable, so the
coin shaping never fires. The only reward signal it reliably received was
`WAITED: -1.0` — "don't stand still" — and it learned exactly that, oscillating
forever between two tiles.

**Fix for v3a: explore all six actions.** Without `BOMB` in the exploration mix,
features `[15]` and `[16]` and their rewards can never receive a gradient.

## ⚠ Data integrity — read before citing

- **No `meta.json`** — this run has no version stamp. It loads on the shape
  check alone, which is the weak half of the guard.
- **`training_log.csv` has no header row** and covers **rounds 1970–5000 only**
  (3031 rows). Rounds 1–1969 are stranded in `runs/v1_coin_heaven/training_log.csv`.
  `plot_training.py` will fail on this file until the header is restored.
- Scenario (`classic`) is known from the launch command, not from any file.

## Diagnostics not persisted

| Diagnostic | Status | Regenerate with |
|---|---|---|
| Action histogram | ✅ saved in `eval_classic.json` | — |
| Position trace | ❌ **terminal only** — `--json` does not persist traces | `python3 analysis/evaluate.py --agent mayank_agent --run v2_crates --scenario classic -n 1 --trace` |
| Per-action weight magnitudes | ❌ **no saved tool exists** | ad-hoc: unpickle `model.pt`, print `np.abs(W[i]).mean()` per row |
