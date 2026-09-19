"""
eval_mixed.py -- evaluate one model against an arbitrary opponent lineup.

A generalisation of evaluate.py: the three opponent seats can be filled with
any agent folders (rule_based_agent, random_agent, peaceful_agent, ...), and
the whole thing can run inside a *snapshot tree* -- an exported copy of the
repository at a given commit -- so a model is always scored by the feature
code it was trained against, whatever the working tree currently holds.

Reproducibility. Round i plays seed_base+i. Three streams are pinned per round:
  - world.rng                      -> the arena (crates, coins, start tiles)
  - $BOMBERMAN_OPPONENT_SEED       -> our local rule_based_agent's private RNG
  - np.random.seed(seed)           -> random_agent / peaceful_agent, which draw
                                      from the global numpy stream
The evaluated agent runs greedy (train=False) and draws nothing, so the
global stream is left to the stock opponents alone.

Example:
    python analysis/eval_mixed.py --tree /path/to/snap_v6 --agent mayank_agent \
        --run v9_urgency --lineup rule_based_agent,random_agent,random_agent \
        -n 200 --seed-base 20240101 --json out.json
"""

import argparse
import importlib
import json
import math
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tree', default=None,
                   help='repository root to run inside (default: this checkout)')
    p.add_argument('--agent', required=True)
    p.add_argument('--run', default=None, help='run name under agent_code/<agent>/runs')
    p.add_argument('--lineup', required=True,
                   help='comma-separated opponent folders, e.g. rule_based_agent,random_agent,random_agent')
    p.add_argument('--scenario', default='classic')
    p.add_argument('-n', '--n-rounds', type=int, default=200)
    p.add_argument('--seed-base', type=int, default=20240101)
    p.add_argument('--json', default=None)
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


def mean_sem(values):
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, float('nan')
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var / n)


def main():
    args = parse_args()
    root = os.path.abspath(args.tree) if args.tree else \
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    os.chdir(root)

    import numpy as np
    from environment import BombeRLeWorld, WorldArgs

    lineup = [x.strip() for x in args.lineup.split(',') if x.strip()]
    api = importlib.import_module(f'agent_code.{args.agent}.callbacks')
    if args.run is not None:
        os.environ[api.RUN_ENV_VAR] = args.run
    meta = api.read_meta(args.run) if hasattr(api, 'read_meta') else {}
    print(f"[eval_mixed] tree={root}", flush=True)
    print(f"[eval_mixed] agent={args.agent} run={args.run} "
          f"feature_version(code)={getattr(api, 'FEATURE_VERSION', None)} "
          f"feature_version(model)={meta.get('feature_version')} "
          f"lineup={lineup}", flush=True)

    wargs = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                      save_replay=False, replay=None, make_video=False,
                      continue_without_training=True,
                      log_dir=os.path.join(root, 'logs'), save_stats=False,
                      match_name=None, seed=None, silence_errors=False,
                      scenario=args.scenario)
    os.makedirs(wargs.log_dir, exist_ok=True)
    world = BombeRLeWorld(wargs, [(args.agent, False)] + [(o, False) for o in lineup])
    agent = world.agents[0]

    rounds = []
    t0 = time.time()
    try:
        for i in range(args.n_rounds):
            seed = args.seed_base + i
            world.rng = np.random.default_rng(seed)
            os.environ['BOMBERMAN_OPPONENT_SEED'] = str(seed)
            np.random.seed(seed)
            world.new_round()
            while world.running:
                world.do_step()
            st = agent.statistics
            opp = [a.statistics['score'] for a in world.agents if a is not agent]
            rank = 1 + sum(1 for s_ in opp if s_ > st['score'])
            rounds.append({
                'seed': seed, 'score': st['score'], 'coins': st['coins'],
                'kills': st['kills'], 'crates': st['crates'], 'steps': st['steps'],
                'suicide': 1.0 if st['suicides'] > 0 else 0.0,
                'died': 1.0 if agent.dead else 0.0,
                'invalid': st['invalid'], 'rank': rank,
                'won': 1.0 if rank == 1 else 0.0,
                'opponent_scores': opp,
            })
            if not args.quiet and ((i + 1) % 20 == 0 or i + 1 == args.n_rounds):
                print(f"  {i + 1}/{args.n_rounds} rounds  ({time.time() - t0:.0f}s)",
                      file=sys.stderr, flush=True)
    finally:
        world.end()

    col = lambda k: [r[k] for r in rounds]
    sm, se = mean_sem(col('score'))
    summary = {
        'n_rounds': len(rounds),
        'mean_score': sm, 'sem_score': se,
        'mean_coins': mean_sem(col('coins'))[0],
        'mean_kills': mean_sem([float(k) for k in col('kills')])[0],
        'mean_crates': mean_sem([float(k) for k in col('crates')])[0],
        'mean_steps': mean_sem(col('steps'))[0],
        'suicide_rate': sum(col('suicide')) / len(rounds),
        'death_rate': sum(col('died')) / len(rounds),
        'win_rate': sum(col('won')) / len(rounds),
        'mean_rank': mean_sem([float(r) for r in col('rank')])[0],
        'invalid_rate': sum(col('invalid')) / max(1, sum(col('steps'))),
        'wall_seconds': time.time() - t0,
    }
    print(json.dumps({'agent': args.agent, 'run': args.run, 'lineup': lineup,
                      'summary': summary}, indent=1), flush=True)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump({'agent': args.agent, 'run': args.run, 'tree': root,
                       'feature_version_code': getattr(api, 'FEATURE_VERSION', None),
                       'feature_version_model': meta.get('feature_version'),
                       'lineup': lineup, 'scenario': args.scenario,
                       'seed_base': args.seed_base, 'summary': summary,
                       'rounds': rounds}, f, indent=1)
        print(f"wrote {args.json}", flush=True)


if __name__ == '__main__':
    main()
