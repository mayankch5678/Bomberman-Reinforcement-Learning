# v3c — `runs/v3c_safe_explore`

**Masked exploration. First run to score above zero — and the first to freeze.**

## What changed from v3b, and why

v3b's diagnosis: the agent was bomb-phobic because exploratory bombs were
followed by *random* escapes, so nearly every one ended in `KILLED_SELF` (−50).
One change, in `callbacks.py:act()`:

```python
def explorable_actions(features):
    allowed = [name for i, name in enumerate(DIRECTIONS)
               if features[IDX_SAFE + i] > 0]      # drop unsurvivable moves
    allowed.append('WAIT')                          # always legal -> never empty
    if features[IDX_BOMB_ESCAPE] > 0:               # drop suicidal bombs
        allowed.append('BOMB')
    return allowed
```

The ε-branch samples **uniformly** over that set, replacing v3a/v3b's fixed
`p=[.2,.2,.2,.2,.1,.1]`. **Greedy selection stays unrestricted** — the mask
shapes what the agent *tries* while learning, never what it may conclude.

Features and reward table **unchanged from v3b**.

## Config

| | |
|---|---|
| Feature version | 2 (21 features) |
| ALPHA / GAMMA | 0.01 / 0.95 |
| Epsilon | 1.0 → 0.05, decay 0.999 |
| Exploration | **masked, uniform over survivable actions** |
| Rounds trained | 1200 |
| Final epsilon | 0.301 |
| Scenario | `loot-crate` |
| git commit | `0f55c1d` |

Reward table identical to v3b (crate shaping +1.0/−1.5, `LIVING_COST` −0.1,
`CRATE_DESTROYED` +3.0, `COIN_FOUND` +2.0, `KILLED_SELF` −50.0).

Mask verified: open board → all six; corner (1,1) → `RIGHT, DOWN, WAIT, BOMB`;
sealed dead end → `DOWN, WAIT` (BOMB excluded); every neighbour unsurvivable →
`WAIT`; all-zero features → `WAIT`. Never empty. Uniformity confirmed over
20 000 draws (25.3 / 24.6 / 25.1 / 24.9 %).

## Training curve — vs v3b, identical settings

| Rounds | v3b score | **v3c score** | v3b reward | **v3c reward** |
|---|---|---|---|---|
| 1–300 | 0.06 | **0.45** | −65.0 | **−29.4** |
| 301–600 | 0.15 | **1.28** | −62.7 | **−12.4** |
| 601–900 | 0.41 | **2.50** | −66.6 | **+14.1** |
| 901–1200 | 0.80 | **3.28** | −73.4 | **+41.7** |

v3b's reward never turned positive; **v3c crosses zero around round 700**.

## Evaluation — `eval_loot-crate.json` (30 rounds, greedy)

| Metric | Mean | SEM |
|---|---|---|
| Score | **4.500** | **2.332** |
| Steps survived | 400.0 | 0.0 |
| Suicide rate | 0.0000 | — |

Action histogram (12000 actions):
`WAIT 73.4% · RIGHT 11.0% · LEFT 10.6% · UP 2.1% · DOWN 1.9% · BOMB 1.1%`

**Bomb-phobia is cured** — BOMB is chosen, suicide rate stays 0.

## Diagnostics that mattered

**The score is bimodal**, which is what the ±2.33 SEM is really saying:

```
per-round scores: [43, 43, 40, 6, 2, 1, 0 x24]
```

Three rounds nearly clear the board; 24 score exactly zero.

**Spawn-tile analysis over all 30 seeds** — the split is decided at step 1:

- 19/30 rounds pick `WAIT` at spawn. **All 19 score 0.**
- 11/30 pick something else; these account for every scoring round.

| At frozen spawns | |
|---|---|
| crate-BFS distance = 0 | **0 / 19** (it is 1 in all 19) |
| `bombcrate` [15] set | **19 / 19** |
| `bombesc` [16] set | **1 / 19** |

Crate-distance can never be 0 at spawn **by construction**: `build_arena`
(`environment.py:181-185`) clears crates from each start position *and its four
neighbours*. `bombcrate` is still 1 because the blast reaches 3 tiles.

**Two structurally identical corners, opposite outcomes:**

```
FROZEN  seed 20240101, spawn (15,1), crateBFS 1, bombcrate 1, bombesc 0
  features: bias wallU wallR safeD safeL bombcrate crateD
  UP -10.325  RIGHT -9.900  DOWN +39.636  LEFT +39.657  WAIT +41.830 <- greedy  BOMB +25.286

ACTIVE  seed 20240104, spawn (1,15), crateBFS 1, bombcrate 1, bombesc 0
  features: bias wallD wallL safeU safeR bombcrate crateR
  UP +39.090  RIGHT +40.839 <- greedy  DOWN -11.237  LEFT -8.131  WAIT +39.670  BOMB +24.427
```

Same structure, different orientation. `WAIT` wins by **2.17** in one and loses
by **1.17** in the other — a ~3-point swing on Q-values near +40.

**Why WAIT wins — the weights:**

```
bias per action:  UP +5.06  RIGHT +5.30  DOWN +4.56  LEFT +5.31  WAIT +25.73  BOMB +4.92

crateUP    -> UP    +1.880  | WAIT +3.377   WAIT wins
crateRIGHT -> RIGHT +1.984  | WAIT +2.138   WAIT wins
crateDOWN  -> DOWN  +2.525  | WAIT +5.257   WAIT wins
crateLEFT  -> LEFT  +2.190  | WAIT +2.756   WAIT wins

safeDOWN   -> DOWN +44.900  | WAIT +3.165   (safety features work correctly)
```

`WAIT` carries a bias **~20 points above every other action**, and in all four
directions "a crate lies this way" raises `WAIT` more than the move toward it.
Weight magnitudes show the same thing: `WAIT` mean |w| = **8.640** versus
~4.4–5.0 for every other row.

The mask caused this: it guarantees `WAIT` is always legal and never fatal, so
`WAIT` is the one action that never incurred `KILLED_SELF`. `WAITED` (−1.0) and
`LIVING_COST` (−0.1) are two orders of magnitude too small to erode a +25 bias.

## Conclusion → v3d (not yet run)

The mask fixed bomb-phobia and created wait-preference in its place. This is a
reward-balance problem, not an exploration one. Candidate fixes:

- raise `MOVED_TOWARD_CRATE` well above +1.0 so crate-seeking can outvote the
  `WAIT` bias;
- penalise `WAIT` specifically when `danger` is 0 — waiting while safe is pure
  stalling, while waiting in cover is legitimate;
- both, with the ratio chosen so a stalled round is worse than a failed attempt.

## Diagnostics not persisted

| Diagnostic | Status | Regenerate with |
|---|---|---|
| Action histogram | ✅ saved in `eval_loot-crate.json` | — |
| Per-round score spread | ✅ saved (`rounds[]` in the eval JSON) | — |
| Position trace of a frozen round | ❌ **terminal only** | `python3 analysis/evaluate.py --agent mayank_agent --run v3c_safe_explore --scenario loot-crate -n 30 --trace 2` |
| **30-seed spawn table** (pos, crate-BFS, bombcrate, bombesc, greedy action) | ❌ **terminal only, no saved tool** | ad-hoc: loop seeds, reseed `world.rng`, `new_round()`, compute features + `W @ features` |
| **Q-value dumps at frozen/active spawns** | ❌ **terminal only, no saved tool** | as above, print the full Q row |
| **Per-direction weight comparison** | ❌ **terminal only, no saved tool** | ad-hoc: unpickle `model.pt`, compare `W[dir, 17+i]` against `W[WAIT, 17+i]` |
| Exploration-mask unit checks | ❌ **terminal only** | ad-hoc: call `explorable_actions()` on hand-built states |
