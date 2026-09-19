# Findings — run-by-run

One file per training run, in order. Each records what changed and why, the full
config, the evaluation numbers, the diagnostics that drove the next decision,
and which diagnostics were never saved.

## Summary

| Run | One-line change | Scenario | Score (30r greedy) | BOMB | WAIT | Verdict |
|---|---|---|---|---|---|---|
| [v1](v1_coin_heaven.md) | baseline: 9 features, coin direction only | `coin-heaven` | **50.00 ± 0.00** * | n/a | n/a | ✅ **solved** — perfect, but the task has no bombs |
| [v2](v2_crates.md) | 21 features + Task 2 rewards; exploration still excluded BOMB | `classic` | 0.00 ± 0.00 | 0.0% | 0.0% | ❌ **2-cycle** — WAIT/BOMB rows never updated at all |
| [v3a](v3a_explore_all.md) | explore all 6 actions, `p=[.2,.2,.2,.2,.1,.1]` | unrecorded † | 0.00 ± 0.00 | 0.0% | 0.0% | ❌ **no gradient** — BOMB row now trained, but negative |
| [v3b](v3b_crateshaping.md) | crate-approach shaping (+1.0/−1.5), living cost −0.1, `CRATE_DESTROYED` +3 | `loot-crate` | 0.00 ± 0.00 | 0.0% | 10.0% | ❌ **bomb-phobic** — `bombcrate` weight in BOMB row is −4.17 |
| [v3c](v3c_safe_explore.md) | mask exploration: no suicidal BOMB, no unsurvivable moves | `loot-crate` | **4.50 ± 2.33** | 1.1% | 73.4% | ⚠ **first non-zero**, but freezes on 19/30 spawns |
| … | **v3d–v12 have no findings files** — see the gap note below | | | | | |
| [v13](v13_robust_escape.md) | robust-escape feature `[38]` + escape-direction count `[39]`; `POINTLESS_BOMB` −3, `UNSAFE_BOMB` −5 | `classic` | 1.452 ± 0.119 ‡ | 4.9% | 8.6% | ❌ **displaced, not fixed** — refused-move deaths 83.6% → 35.1%, but WAIT wins the one-exit state and score falls 59% |

Not a training run, but kept here because it is what was submitted:

| Artifact | What it is | Score | Verdict |
|---|---|---|---|
| [gbt-numpy port](gbt_numpy_port.md) | `mayank_gbt`'s 600 boosted trees exported to plain numpy, shipped as `mayank_agent` | **3.665 ± 0.194** (200r) | ✅ **submitted** — 0/10000 argmax disagreements vs scikit-learn, 0.787 ms/step |

\* v1 is measured over **100** rounds on `coin-heaven`, not 30 on `loot-crate` —
its number is **not comparable** to the rest of the column. It is also the only
run whose evaluation cannot be reproduced (see below).

‡ v13 is measured over **five seed bases × 200 rounds** on `classic` against 3
`rule_based_agent`s, so its number is not comparable to the 30-round
`loot-crate` column either; the ± is the spread across bases, not a SEM. Its
BOMB/WAIT shares are from a 200-round action histogram on base 20240101. The
`v9_urgency` baseline on the same five bases is **3.567 ± 0.221**.

† v3a's scenario is not recorded anywhere; it evaluates to 0.00 on both
`classic` and `loot-crate`, so the verdict holds either way.

## The arc in one paragraph

v1 solved coin collection outright. v2 added crates and bombs but kept a Task 1
line in `act()` that excluded `BOMB` from exploration — so the BOMB and WAIT
weight rows were literally never updated, and the agent oscillated between two
tiles for 400 steps. v3a opened exploration to all six actions, which trained
the BOMB row but drove it *negative*: random bombs followed by random escapes
are almost always fatal. v3b added crate-approach shaping and a living cost;
the shaping worked exactly as specified (a 2-cycle costs −0.7) but was not the
binding constraint — the agent walked to the bombing tile and refused to bomb,
because it had learned that "a bomb here hits a crate" predicts death. v3c
masked exploration so the agent can never *try* a suicidal bomb or a fatal step.
That cured bomb-phobia and produced the first non-zero score — and revealed the
next problem: `WAIT` is now the only action that never got the agent killed, so
it acquired a +25 bias and the agent freezes on most spawns.

## The gap: v3d–v12

Ten runs exist under `agent_code/mayank_agent/runs/` with **no findings file
between them**: `v3d_bombspot`, `v3e_control`, `v4_long`, `v5_classic`,
`v7_escape`, `v8_inblast6`, `v8_inblast10`, `v9_urgency`, `v9b_urgency13`,
`v12_offence`. Each carries a `meta.json` and a `training_log.csv`, so the config
and the curve are recoverable, but the diagnostics that drove each decision are
not written down anywhere. The two that v13 depends on are recorded in
`v13_robust_escape.md` rather than in files of their own:

- **v5_classic**: 77–89% of self-kills contained a refused move inside the blast.
- **v9_urgency**: the same figure is 83.6%, `[16]` was set at **73 of 73** fatal
  drops, and 35 of 68 body refusals were into a tile taken inside the same step.

Note also that `v9_urgency`, `v9b_urgency13` and `v12_offence` are all feature
version 6, so **the current feature-version-7 code refuses to load them** by
design (`check_compatible`). Re-measuring any of them needs the v6
`callbacks.py`, e.g. `git show 82882cc:agent_code/mayank_agent/callbacks.py`.

## Open items

- **v3d is not yet run.** v3c's conclusion proposes raising
  `MOVED_TOWARD_CRATE` and/or penalising `WAIT` when `danger` is 0.
- **v14 is not yet run.** v13's conclusion proposes penalising WAIT-in-blast
  conditioned on `[39] >= 1`, and stopping `POINTLESS_BOMB` from stacking with
  `USELESS_BOMB`.
- **`evaluate.py` cannot evaluate an agent with no run layout** — it assigns
  `None` to `$BOMBERMAN_RUN` and raises `TypeError` (`evaluate.py:104`). This is
  why the submitted agent's numbers come from an ad-hoc harness.
- **v1's model file is lost and its log is contaminated** (rows 2001–3969 belong
  to the v2 run). See the data-integrity section in `v1_coin_heaven.md`.
- **v2 has no `meta.json` and its log has no header and is missing rounds
  1–1969.** See `v2_crates.md`.
- **`train.py` does not record the training scenario in `meta.json`**, which is
  why v3a's is unknown. Worth fixing before the next run.
- **`--json` does not persist `--trace` output.** Every position trace cited in
  these files is terminal-only; each file lists the exact command to regenerate.
- **No saved tool produces Q-value dumps or weight breakdowns.** These were the
  single most decisive diagnostics in v3b and v3c and exist only as ad-hoc
  scripts. Each file documents what to rebuild.

## Reproducing the evaluations

```bash
python3 analysis/evaluate.py --agent mayank_agent --run <run> \
        --scenario <scenario> -n 30 --histogram --quiet \
        --json agent_code/mayank_agent/runs/<run>/eval_<scenario>.json
```

All seeds are `20240101 + i`, so any two runs evaluated with the same `-n` and
`--seed-base` face identical arenas. `rule_based_agent` on `classic` scores
**8.500 ± 0.170** as a reference baseline.
