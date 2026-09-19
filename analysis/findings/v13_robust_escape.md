# v13 — `runs/v13_robust_escape`

**Robust-escape feature + bomb-discipline penalties. Broke the mechanism that
caused 83.6% of v9's self-kills — and lost 59% of the score doing it.**

## What changed from v9, and why

The v9 self-kill forensics (200 rounds, `classic`, 3 `rule_based_agent`s) found
that **feature `[16]` was set at 73 of 73 fatal bomb drops**, and an independent
escape search agreed with it every time. The escape was real when the bomb was
dropped; it stopped being real afterwards. 61 of those 73 deaths (83.6%)
contained a refused move inside the four-step window, and 35 of those refusals
were into a tile an opponent stepped onto *within the same step* — the engine
resolves a step's agents in a random permutation (`environment.py:430`), so a
tile can be taken between the agent choosing and the agent moving.

So `[16]` was not wrong, it was asking the wrong question. Two new features,
**feature version 7 (40 slots)**:

```python
IDX_BOMB_ESCAPE_ROBUST = 38  # [16], but assuming nearby opponents close in
IDX_N_ESCAPE_DIRS = 39       # how many of [33:37] are open, 0..4
```

`[38]` re-runs the `[16]` escape search against a pessimistic closure: every
opponent within `ROBUST_OPP_RADIUS` steps is assumed to spend the next
`ROBUST_OPP_STEPS` moves walking at the agent, and every tile it could stand on
is blocked at once (`threat_closure()`). `[39]` is `sum(features[33:37])`, so
"one narrow exit with an opponent nearby" is distinguishable from "three exits".

Two new penalties in `train.py`:

| Event | Condition | Reward |
|---|---|---|
| `POINTLESS_BOMB` | `[15] == 0` **and** `[32] == 0` | −3.0 |
| `UNSAFE_BOMB` | `[38] == 0` | −5.0 |

`POINTLESS_BOMB` **stacks with the pre-existing `USELESS_BOMB` (−2.0)**, which
fires on the no-crate half alone. A bomb that hits neither crate nor opponent
therefore costs −5.0 in total, not −3.0.

## Choosing the closure — measured before training, not after

The closure was first implemented exactly as specified: radius 3, one step of
dilation. Evaluated against **the 6877 bombs v9 actually dropped** over the 200
forensics rounds, scored on how many of its 73 fatal bombs the rule would have
refused:

| radius | steps | bombs flagged | fatal bombs flagged |
|---|---|---|---|
| 3 | 1 | 15 / 6877 (0.2%) | 5 / 73 (6.8%) |
| 3 | 2 | 76 (1.1%) | 8 (11.0%) |
| 5 | 1 | 15 (0.2%) | 5 (6.8%) |
| 5 | 2 | 95 (1.4%) | 13 (17.8%) |
| **5** | **3** | **563 (8.2%)** | **25 (34.2%)** |
| 8 | 3 | 566 (8.2%) | 25 (34.2%) |

At radius 3 / 1 step, `[38]` is a **99.8% copy of `[16]`** and the −5 would have
fired ~15 times per 200 rounds — too rare to train a weight, and the 5 deaths it
caught are exactly the 5 the v9 forensics already classified as "escape sealed by
a body at drop time". The radius saturates (3 ≡ 5 at one step; 5 ≡ 8 at three),
so **the dilation depth is what bites**. Trained at **radius 5, 3 steps**.

The same sweep showed `[39]` is close to uninformative *at drop time*: a rule of
"`[39] < 2`" flags 9.6% of fatal bombs against a 5.8% base rate. It was added
anyway, as specified, because it costs nothing at prediction time. It turned out
to matter enormously — for the wrong reason (see below).

## Config

| | |
|---|---|
| Feature version | **7 (40 features)** |
| ALPHA / GAMMA | 0.01 / 0.95 |
| Epsilon | 1.0 → 0.05, decay 0.9999 (floor at ~29956) |
| Rounds trained | 40000 |
| Final epsilon | 0.05 |
| Total training steps | 4560813 (v9: 4372269) |
| Scenario | `classic`, 3 `rule_based_agent`s |
| `INVALID_IN_BLAST_PENALTY` | −10.0 (`BOMBERMAN_INBLAST_PENALTY=-10`) |
| `USE_OFFENCE` | **False** (`BOMBERMAN_OFFENCE=0`) |
| git commit | `82882cc` |

`train.py` had drifted since v9: it now defaults to the v12 offence rewards
(`KILLED_OPPONENT` +30, `OFFENSIVE_BOMB` +8, opponent nav ±1.0/−1.5) and to
`INVALID_IN_BLAST_PENALTY = -15`. **Both had to be overridden** to keep the rest
of the regime identical to v9, which `runs/v9_urgency/meta.json` records as
`invalid_in_blast: -10.0` with no offence keys at all. Everything else —
`LIVING_COST` −0.1, `WAITED` −1.0, no stall penalty, no bomb-spot shaping —
matches by default.

## Training curve — 2000-round rolling mean, against v9

| Round | v13 score | v9 score | v13 steps | v9 steps |
|---|---|---|---|---|
| 20000 | 1.2820 | 1.3025 | 106.18 | 99.19 |
| 30000 | 2.0505 | 2.1195 | 182.85 | 174.79 |
| 35000 | 2.2045 | 2.3665 | 189.45 | 188.54 |
| 40000 | 2.2520 | 2.3560 | 193.04 | 190.17 |

v13 tracks v9 and ends slightly below it. Peak rolling score **2.2890 at round
33798**; peak rolling steps 201.17 at 37981. Score was **flatter than v9 at the
end**: slope +0.0007 per 1000 rounds over 30000–40000, against v9's +0.0150.

Training-curve score is not comparable to the evaluation numbers below — it
includes the ε = 0.05 exploration tail.

## Evaluation — five seed bases × 200 rounds, greedy

Bases 20240101 / 20241101 / 20242101 / 20243101 / 20244101, so the 200-seed
windows never overlap. v9 was measured on the same ten hundred seeds **before**
the feature-version bump made its weights unloadable.

| Metric | v9 | **v13** | Δ | v9 sd | v13 sd |
|---|---|---|---|---|---|
| Score | 3.5670 | **1.4520** | **−2.1150** | 0.2214 | 0.1194 |
| Coins | 2.9370 | 1.2070 | −1.7300 | 0.1431 | 0.0813 |
| Kills / round | 0.1260 | 0.0490 | −0.0770 | 0.0233 | 0.0129 |
| Steps survived | 272.869 | 282.371 | +9.502 | 7.876 | 11.559 |
| **Suicide rate** | **0.3390** | **0.3320** | **−0.0070** | 0.0397 | 0.0342 |
| Win rate (rank 1) | 0.3870 | 0.1090 | −0.2780 | 0.0182 | 0.0152 |
| Outscored best opp. | 0.2810 | 0.0660 | −0.2150 | 0.0253 | 0.0185 |
| Mean rank (1–4) | 2.0070 | 2.9950 | +0.9880 | 0.0486 | 0.0619 |
| Invalid-action rate | 0.0035 | 0.0055 | +0.0020 | 0.0003 | 0.0004 |

**The suicide rate did not improve.** −0.007 sits well inside the base-to-base
spread. The score regression is far outside it: v13's worst base (1.355) beats
none of v9's five.

## Diagnostics that mattered

### The targeted mechanism did break — the deaths relocated

Re-run of the v9 forensics on base 20240101, n = 200. Self-kills 73 → 57 on this
base, but 0.285 vs 0.365 is inside the five-base spread, so read the
**composition**, not the total.

| Category | v9 | **v13** |
|---|---|---|
| escape blocked by opponent body | 50 (68.5%) | **18 (31.6%)** |
| escape blocked by a bomb it did not see | 7 (9.6%) | 2 (3.5%) |
| **waited in blast** | 8 (11.0%) | **37 (64.9%)** |
| BOMB re-attempt with no bomb left | 3 (4.1%) | 0 |
| escape sealed by a body at drop time | 5 (6.8%) | 0 |
| walked the wrong way | 0 | 0 |
| **any refused move in the window** | **61 (83.6%)** | **20 (35.1%)** |

Both new features are being *used*: `[39]` at the fatal drop was 2 in 44 cases,
3 in 8, 4 in 5 — **never 0 or 1**, so v13 refuses to bomb without at least two
exits. `[32]` (opponent in blast at drop) fell from 16 to 2. And `[38]` was set
at **all 57** fatal drops: it never vetoed a fatal bomb, which is what the
pre-flight predicted — 34.2% recall was measured against *v9's* drop
distribution, and v13 only drops bombs that already pass the check.

### The new failure: WAIT wins the one-exit state

v9 **never once** declined an offered escape direction: 0 of 149 window-steps
where `[33:37]` had something open. v13 declines **30 of 90**, and every single
one is the same state — in the blast, `[39] == 1`, a live route still open:

```
Q(WAIT) − Q(best offered escape) = +2.237 mean, +0.481 min, +3.147 max
recurring exactly:  Q(WAIT) = 21.39  vs  Q(escape) = 19.02   margin +2.37
```

The identical values across many seeds are one canonical feature configuration
recurring under a deterministic linear policy. With **two or more** exits open,
WAIT loses by −8.97 on average and the agent escapes correctly. So the feature
added to separate "one narrow exit" from "three exits" did separate them, and
then WAIT won the narrow one.

Every one of the 74 (v9) / 148 (v13) WAIT steps inside a window had `[39] == 0`
in v9; in v13, 30 of them had `[39] == 1`. That is the entire regression in the
death classification.

### Why the score collapsed: it stopped playing

| | v9 | **v13** |
|---|---|---|
| BOMB share of actions | 12.2% | **4.9%** |
| A→B→A step-backs | 11.2% of moves | **60.7%** |
| Distinct tiles visited / round | 67.7 | **35.6** |
| LEFT+RIGHT share | 41.6% | **76.5%** |

Measured over 50 rounds, base 20240101, both agents. v13 bombs 58% less often
and acquired a severe horizontal 2-cycle — `evaluate.py`'s own trace heuristic
calls anything above 40% step-backs "oscillating". Fewer crates opened means
fewer coins revealed, which is the −1.73 coins. The −5 / −3 bomb penalties plus
the WAIT bias bought survival by declining to act.

### Weights — the linear-model asymmetry, measured

| | v13 | v9 |
|---|---|---|
| `w_BOMB[38]` robust escape | **+8.348** | — |
| `w_BOMB[39]` escape-dir count | +3.003 | — |
| `w_BOMB[16]` bomb escape | +27.283 | +33.202 |
| `w_BOMB[0]` bias | **−28.815** | −21.244 |
| `w_BOMB[37]` trapped | +3.003 | +15.473 |
| `w_WAIT[39]` | **+2.343** | — |
| `w_WAIT[0]` bias | +18.264 | — |

The update is `w_a += alpha * td_error * phi`, so **a penalty paid when
`phi[38] == 0` contributes exactly zero to `w_BOMB[38]`**. The −5 cannot train
the feature it is attached to; it is absorbed by the bias and the other active
slots, i.e. it teaches "bomb less" globally. Both halves are visible above:
`w_BOMB[38] = +8.35` is real discrimination, earned entirely from `[38] == 1`
bombs that went well, while the bias falling 7.57 and `w[16]` falling 5.92 is
the −5 landing where it structurally must. `w_BOMB[37]` collapsing from +15.47
to +3.00 suggests `[39]` took over part of the trapped slot's job.

## Conclusion → v14 (not run)

The intervention hit its target and the deaths moved rather than disappeared.
Two changes, both cheap, before another 40000-round arm is worth spending:

- **Penalise WAIT in a blast conditioned on `[39] >= 1`.** The one-exit state is
  the whole regression, it is 30 of 30 identifiable, and nothing in the current
  reward table distinguishes waiting with an exit open from waiting cornered.
- **Stop `POINTLESS_BOMB` stacking with `USELESS_BOMB`,** or cut it to −1.0. A
  bomb that achieves nothing currently costs −5.0 against `KILLED_SELF`'s −50,
  and the bombing rate more than halved.

A drop-time feature is structurally limited here: v9's forensics established
that the escape is genuine when the bomb is dropped and is destroyed two to four
steps later by opponent movement. The more promising direction is a
*during-escape* feature — e.g. preferring an escape direction whose corridor is
not contested — rather than a stricter gate on the drop itself.

## Diagnostics not persisted

| Diagnostic | Status | Regenerate with |
|---|---|---|
| Learning curve | ✅ `runs/v13_robust_escape/training_log.csv` | — |
| Five-base evaluation | ✅ scratch JSON only, **not in the run dir** | `python3 analysis/evaluate.py --agent mayank_agent --run v13_robust_escape --scenario classic -n 200 --opponents 3 --seed-base <B> --json <out>` for B in 20240101, 20241101, 20242101, 20243101, 20244101 |
| **Self-kill forensics** (feature vector at the fatal drop, 4-step window, blocker identity, Q margins, classification) | ❌ **ad-hoc script, scratch only** | rebuild: wrap `world.perform_agent_action` to name the live blocker at execution time, recompute features from `agent.last_game_state`, find the last `BOMB_DROPPED` before death, classify the 4 following steps |
| **Closure sweep** (radius × steps vs fatal-bomb recall) | ❌ **ad-hoc script, scratch only** | replay v9 with the v6 code, evaluate `threat_closure` variants at every `BOMB_DROPPED` |
| **Oscillation / step-back rate** | ❌ **ad-hoc script, scratch only** | count `A→B→A` over `agent.x/y` per round; `evaluate.py --trace` prints it for one round only |
| Weight breakdown | ❌ terminal only | `np.load('runs/v13_robust_escape/model.npy')`, index `[ACTIONS.index('BOMB'), 38]` etc. |
| v9 baseline on the same five bases | ⚠ measured, scratch only — **and no longer reproducible with this code** | feature version 7 refuses the 38-slot v9 weights by design (`check_compatible`). Needs the v6 `callbacks.py`/`train.py`, e.g. `git show 82882cc:agent_code/mayank_agent/callbacks.py` |
