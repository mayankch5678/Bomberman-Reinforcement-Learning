# v1 — `runs/v1_coin_heaven`

**Task 1: collect coins on an empty board. Solved.**

## What this run was

The baseline agent: linear Q-learning over a 9-slot feature vector, trained on
`coin-heaven` (no crates, no bombs, 50 coins, single agent).

## Config

| | |
|---|---|
| Feature version | **1** (9 features) |
| Features | `[0]` bias · `[1:5]` neighbour blocked · `[5:9]` BFS direction to nearest coin |
| ALPHA / GAMMA | 0.01 / 0.95 |
| Epsilon | 1.0 → 0.05, decay 0.999 per round |
| Rounds trained | **2000** |
| Final epsilon | 0.1352 |
| Scenario | `coin-heaven` |

Reward table:

| Event | Value |
|---|---|
| `COIN_COLLECTED` | +10.0 |
| `MOVED_TOWARD_COIN` / `MOVED_AWAY_FROM_COIN` | +1.0 / −1.0 |
| `WAITED` | −1.0 |
| `INVALID_ACTION` | −3.0 |
| `BOMB_DROPPED` | −5.0 |
| `KILLED_SELF` | −50.0 |
| `SURVIVED_ROUND` | 0.0 |

## Training curve

| Rounds | score | steps | reward | ε |
|---|---|---|---|---|
| 1–500 | 37.77 | 389.7 | +82.3 | 0.786 |
| 501–1000 | 48.92 | 256.3 | +426.6 | 0.477 |
| 1001–1500 | 49.00 | 182.4 | +500.6 | 0.289 |
| 1501–2000 | 48.94 | 157.6 | +523.0 | 0.175 |

Score saturates at 49 by round ~257 and is 49 in 99.7% of rounds after 600.
All remaining learning shows up as **step efficiency**, 400 → 157.

## Evaluation — `eval_coin-heaven.json` (100 rounds, greedy, seeds 20240101+)

| Metric | Mean | SEM |
|---|---|---|
| Score | **50.000** | 0.000 |
| Coins | **50.000** | 0.000 |
| Steps survived | 125.100 | 0.847 |
| Suicide rate | 0.0000 | 0.0000 |
| Invalid-action rate | 0.0017 | 0.0003 |

**The greedy policy is better than the training log suggests**: it collects all
50 coins in 100/100 rounds in 125 steps, versus 49.0 / 147 steps in training.
The gap is the 13.5% random actions still active at round 2000 — ε never
reached its 0.05 floor, which needs ~2994 rounds at decay 0.999.

Score and coins have **zero variance**, so they cannot discriminate between any
two competent agents on this scenario. Steps is the only metric with signal left.

## Conclusion → v2

Task 1 is solved and the feature set is exhausted. Moving to Task 2 requires
crates and bombs, which means new features (danger, neighbour safety, bomb
value, crate direction) and a new reward table. Feature version bumped to 2.

## ⚠ Data integrity — read before citing

- **`model.pt` in this directory is NOT the v1 model.** The real (6, 9) weights
  were overwritten by a later training run before the run was archived and are
  **not recoverable**. The file present is a (6, 21) v2-era model, misfiled.
- **`meta.json` is wrong**: it reports `n_features: 21` and
  `rounds_trained: 3969`. The true values are 9 and 2000.
- **`training_log.csv` contains 3969 rows, not 2000.** Rows 1–2000 are this run;
  rows 2001–3969 are the first 1969 rounds of the v2 `classic` run, appended
  because both wrote to the same legacy path. **Cite rows 1–2000 only.**
- This run **cannot be re-evaluated** — feature-version 1 weights are refused by
  feature-version 2 code, and the weights are gone regardless. The evaluation
  above is archived and is the only surviving measurement.

## Diagnostics not persisted

None outstanding — the evaluation JSON and training log (rows 1–2000) carry
everything cited here.
