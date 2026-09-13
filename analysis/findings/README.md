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

\* v1 is measured over **100** rounds on `coin-heaven`, not 30 on `loot-crate` —
its number is **not comparable** to the rest of the column. It is also the only
run whose evaluation cannot be reproduced (see below).

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

## Open items

- **v3d is not yet run.** v3c's conclusion proposes raising
  `MOVED_TOWARD_CRATE` and/or penalising `WAIT` when `danger` is 0.
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
