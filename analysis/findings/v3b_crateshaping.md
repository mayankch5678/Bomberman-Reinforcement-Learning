# v3b — `runs/v3b_crateshaping`

**Crate-approach shaping + living cost. Training improved, greedy policy still score 0.**

## What changed from v3a, and why

v3a showed the agent had no gradient pulling it anywhere once coins were
unreachable behind crates. Four changes, all in `train.py`:

1. **Crate-approach shaping** — `MOVED_TOWARD_CRATE` / `MOVED_AWAY_FROM_CRATE`,
   from BFS distance to the nearest *crate-approach tile* (a free tile adjacent
   to a crate; you cannot stand on a crate). Gated behind coin reachability:
   coin shaping wins whenever a coin is reachable, mirroring the gate on the
   `[17:21]` crate-direction feature.
2. **Deliberately asymmetric** — away (−1.5) costs more than toward (+1.0), so a
   two-tile oscillation is **strictly loss-making** rather than break-even. This
   is the one non-symmetric pair in the table, targeting v2's 2-cycle directly.
3. **Flat living cost** `−0.1` per step, via a new `step_reward()` used at both
   call sites so it is charged exactly once per step.
4. **`CRATE_DESTROYED` 1.0 → +3.0**, **`COIN_FOUND` 1.0 → +2.0**.

v3a's exploration fix was kept.

## Config

| | |
|---|---|
| Feature version | 2 (21 features) |
| ALPHA / GAMMA | 0.01 / 0.95 |
| Epsilon | 1.0 → 0.05, decay 0.999 |
| Exploration | all 6 actions, `p=[.2,.2,.2,.2,.1,.1]` |
| Rounds trained | **1200** |
| Final epsilon | 0.301 |
| Scenario | `loot-crate` |
| git commit | `0f55c1d` |

Reward table (changes from v3a in **bold**):

| Event | Value |
|---|---|
| `COIN_COLLECTED` | +10.0 |
| `CRATE_DESTROYED` | **+3.0** |
| `COIN_FOUND` | **+2.0** |
| `MOVED_TOWARD_COIN` / `MOVED_AWAY_FROM_COIN` | +1.0 / −1.0 |
| **`MOVED_TOWARD_CRATE` / `MOVED_AWAY_FROM_CRATE`** | **+1.0 / −1.5** (asymmetric) |
| `ESCAPED_BLAST` / `MOVED_INTO_BLAST` | +3.0 / −3.0 |
| `BOMB_NEXT_TO_CRATE` / `USELESS_BOMB` | +2.0 / −2.0 |
| `WAITED` | −1.0 |
| `INVALID_ACTION` | −3.0 |
| `BOMB_DROPPED` | 0.0 |
| `KILLED_SELF` | −50.0 |
| **`LIVING_COST`** (every step) | **−0.1** |

Verified cycle costs: crate 2-cycle **−0.70**, coin 2-cycle **−0.20**,
equidistant no-event cycle **−0.20** — all strictly negative, as designed.
Safety ordering preserved: away-from-crate to escape a blast **+1.40**;
toward-crate into a blast **−2.10**.

## Training curve

| Rounds | score | steps | reward | ε |
|---|---|---|---|---|
| 1–300 | 0.06 | 18.6 | −65.0 | 0.863 |
| 301–600 | 0.15 | 25.1 | −62.7 | 0.640 |
| 601–900 | 0.41 | 40.3 | −66.6 | 0.474 |
| 901–1200 | 0.80 | 65.4 | −73.4 | 0.351 |

Score and steps both rise, but **reward never turns positive**.

## Evaluation — `eval_loot-crate.json` (30 rounds, greedy)

| Metric | Mean | SEM |
|---|---|---|
| Score | **0.000** | 0.000 |
| Steps survived | 400.0 | 0.0 |
| Suicide rate | 0.0000 | — |

Action histogram (12000 actions): `UP 35.0% · DOWN 35.0% · RIGHT 10.0% ·
LEFT 10.0% · WAIT 10.0% · BOMB 0.0% (never chosen)`.

## Diagnostics that mattered

**The 2-cycle survived.** Trace of round 1 (seed 20240101), `loot-crate`:

```
distinct tiles visited : 2
steps stepping back    : 399 / 400
VERDICT: the agent is stuck -- it never leaves these tiles.
```

**Q-value dump at the two cycle tiles** — this is the decisive diagnostic:

```
(15,1)  features: wallU wallR safeD safeL bombcrate crateD
   UP=-2.08  RIGHT=-1.94  DOWN=+2.95  LEFT=+2.11  WAIT=+2.60  BOMB=-14.12  -> DOWN
(15,2)  features: wallR wallD wallL safeU bombcrate bombesc
   UP=+0.93  RIGHT=-4.07  DOWN=-2.70  LEFT=-4.08  WAIT=+0.79  BOMB= -8.51  -> UP
```

Crate-BFS distance: (15,1) = 1, (15,2) = **0**. So the crate events *do* fire on
this cycle and it *is* strictly negative (−0.7 per cycle) exactly as designed.

**The agent walks to the bombing tile correctly and then cannot pull the
trigger.** At (15,2) — crate-distance 0, both `bombcrate` and `bombesc` set, i.e.
a bomb that is both productive *and* survivable — `BOMB` scores **−8.51**. So it
walks back, pays −0.7, and repeats.

**BOMB row, largest-magnitude weights:**

```
urgency    -11.952
danger      -9.436
bombcrate   -4.172   <- backwards
bias        -3.701
```

The weight on `bombcrate` in the BOMB row is **−4.17**: the agent learned that
"a bomb here would destroy a crate" makes bombing *worse*.

## Conclusion → v3c

The shaping works as specified; it is not the binding constraint. The binding
constraint is bomb-phobia from credit assignment: with `BOMB` explored at 10%
uniformly and the subsequent escape also random, an exploratory bomb almost
always ends in `KILLED_SELF` (−50), so the feature identifying a *good* bomb
became a death predictor. Raising `CRATE_DESTROYED` to +3 cannot close that gap
while one mistake costs −50.

Note also that a flat living cost charged identically to all six actions shifts
every Q-value at a state equally and **cannot by itself break a tie** between
actions — it only biases toward shorter episodes through bootstrapping.

**Fix for v3c: mask the exploration.** Exclude `BOMB` when `bombesc` is 0, and
exclude moves into tiles the safety features mark unsurvivable.

## Diagnostics not persisted

| Diagnostic | Status | Regenerate with |
|---|---|---|
| Action histogram | ✅ saved in `eval_loot-crate.json` | — |
| Position trace | ❌ **terminal only** | `python3 analysis/evaluate.py --agent mayank_agent --run v3b_crateshaping --scenario loot-crate -n 1 --trace` |
| **Q-value dump at (15,1)/(15,2)** | ❌ **terminal only, no saved tool** | ad-hoc: rebuild world at seed 20240101, place agent, print `W @ state_to_features(gs)` |
| **BOMB-row weight breakdown** | ❌ **terminal only, no saved tool** | ad-hoc: unpickle `model.pt`, sort `W[ACTIONS.index('BOMB')]` by `abs` |
| Reward-arithmetic checks (cycle costs) | ❌ **terminal only** | ad-hoc: call `reward_from_events` / `step_reward` on event lists |
