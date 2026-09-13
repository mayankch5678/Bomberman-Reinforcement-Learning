# v3a — `runs/v3a_explore_all`

**Exploration opened to all six actions. Broke the 2-cycle, still score ~0.**

## What changed from v2, and why

One change, in `callbacks.py:act()`:

```python
# before (v2)
return np.random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT'])
# after (v3a)
return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
```

v2's diagnosis was that `BOMB` and `WAIT` were never explored, so their weight
rows were never updated and the bomb features could never receive a gradient.
v3a samples all six, with WAIT and BOMB at 10% each since most steps should be
moves.

Features and reward table are **unchanged from v2**.

## Config

| | |
|---|---|
| Feature version | 2 (21 features) |
| ALPHA / GAMMA | 0.01 / 0.95 |
| Epsilon | 1.0 → 0.05, decay 0.999 |
| Exploration | **all 6 actions**, `p=[.2,.2,.2,.2,.1,.1]` |
| Rounds trained | **5000** |
| Final epsilon | 0.05 (floor reached) |
| git commit | `0f55c1d` |
| Scenario | **not recorded** — see data integrity below |

Reward table identical to v2 (`CRATE_DESTROYED` +1.0, `COIN_FOUND` +1.0,
`BOMB_DROPPED` 0.0, no crate shaping, no living cost).

## Training curve

| Rounds | score | steps | reward | ε |
|---|---|---|---|---|
| 1–1250 | 0.05 | 35.2 | −72.7 | 0.570 |
| 1251–2500 | 0.14 | 131.7 | −72.2 | 0.163 |
| 2501–3750 | 0.12 | 260.7 | −49.5 | 0.056 |
| 3751–5000 | 0.12 | 283.7 | −44.6 | 0.050 |

Max score over all 5000 rounds: **3.0**. Non-zero in 487/5000 rounds.

The interesting signal is **steps: 35 → 284**. Early rounds end after ~35 steps
because the agent now explores `BOMB` and immediately kills itself; it slowly
learns to survive. Reward improves (−73 → −45) but score does not.

## Evaluation (30 rounds, greedy)

| Scenario | Score | Steps | Suicide | BOMB share | WAIT share |
|---|---|---|---|---|---|
| `classic` (`eval_classic.json`) | 0.000 ± 0.000 | 400.0 | 0.00 | 0.0% | 0.0% |
| `loot-crate` (`eval_loot-crate.json`) | 0.000 ± 0.000 | 400.0 | 0.00 | 0.0% | 0.0% |

Zero on both scenarios, 30/30 rounds each, never bombs when greedy.

## Diagnostics that mattered

**Weight magnitudes** confirm the fix worked at the level it targeted — every
row is now substantially trained, unlike v2 where WAIT sat at initialisation:

```
v2:   UP=0.239  RIGHT=0.273  DOWN=0.239  LEFT=0.240  WAIT=0.010  BOMB=0.144
v3a:  UP=4.060  RIGHT=4.070  DOWN=3.943  LEFT=4.087  WAIT=2.868  BOMB=3.137
```

So `BOMB` did receive a gradient (0.144 → 3.137). It is simply **negative**: the
greedy policy still never picks it.

## Conclusion → v3b

Opening exploration was necessary but not sufficient. The agent survives longer
and its bomb row is trained, but it still scores ~0 because:

- with no coin reachable through crates, coin shaping never fires, so there is
  no gradient pulling the agent anywhere;
- nothing rewards *approaching* a crate, so the crate-direction feature
  `[17:21]` carries no signal;
- nothing makes dithering costly.

**Fix for v3b: crate-approach shaping** (BFS distance to nearest crate, mirroring
the coin logic) with an asymmetric penalty so oscillation is strictly negative,
plus a flat per-step living cost, plus larger `CRATE_DESTROYED` / `COIN_FOUND`.

## ⚠ Data integrity

- **Scenario is not recorded.** `train.py` does not write the scenario into
  `meta.json` (the agent has no access to it), so it is unknown for this run.
  The training-curve shape (max score 3) is consistent with `classic`
  (9 coins) but this is inference, not evidence. Both evaluations are provided
  above so the comparison holds either way.
- Log and meta are otherwise clean: header present, 5000 rows, rounds 1–5000.

## Diagnostics not persisted

| Diagnostic | Status | Regenerate with |
|---|---|---|
| Action histograms | ✅ saved in both eval JSONs | — |
| Position trace | ❌ **never captured for this run** | `python3 analysis/evaluate.py --agent mayank_agent --run v3a_explore_all --scenario classic -n 1 --trace` |
| Q-value dump at spawn | ❌ **never captured, no saved tool** | ad-hoc: rebuild world at a seed, `W @ state_to_features(gs)` |
| Per-action weight magnitudes | ❌ **terminal only, no saved tool** | ad-hoc: unpickle `model.pt`, print `np.abs(W[i]).mean()` |
