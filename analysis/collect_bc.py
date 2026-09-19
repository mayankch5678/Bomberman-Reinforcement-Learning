"""
collect_bc.py -- behavioural-cloning data from the rule-based agent.

Plays N rounds of 4x rule_based_agent on a scenario and, for ONE designated
seat, logs at every step the pair

    (state_to_features(game_state) at the tree's feature version, action)

where game_state is exactly the dict the engine handed to that agent's act()
and action is what act() returned. The engine-recorded action (which can
differ if a step timed out) is stored alongside so the two can be compared.

Output: a compressed .npz with
    X            float32 (n, N_FEATURES)   feature vectors
    y            int8    (n,)              index into ACTIONS, -1 if act() returned None
    action       str     (n,)              the same, as text ('<none>' for None)
    engine_action str    (n,)              what the engine actually applied
    round, step, seed    int32 (n,)        provenance
    plus scalar metadata: feature_version, actions, agent_name, seed_base, layout

Example:
    python analysis/collect_bc.py --tree /path/to/snap_v6 -n 300 \
        --seed-base 20240101 --out analysis/results/bc_rule_based_v6.npz
"""

import argparse
import collections
import importlib
import json
import os
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tree', default=None)
    p.add_argument('--features-from', default='mayank_agent',
                   help='agent folder whose callbacks.state_to_features defines the vector')
    p.add_argument('--expert', default='rule_based_agent')
    p.add_argument('--scenario', default='classic')
    p.add_argument('-n', '--n-rounds', type=int, default=300)
    p.add_argument('--seed-base', type=int, default=20240101)
    p.add_argument('--seat', type=int, default=0, help='which of the 4 seats to log')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out_path = os.path.abspath(args.out)
    root = os.path.abspath(args.tree) if args.tree else \
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    os.chdir(root)

    import numpy as np
    from environment import BombeRLeWorld, WorldArgs

    fmod = importlib.import_module(f'agent_code.{args.features_from}.callbacks')
    ACTIONS = list(fmod.ACTIONS)
    print(f"[collect_bc] tree={root} feature_version={fmod.FEATURE_VERSION} "
          f"n_features={fmod.N_FEATURES}", flush=True)

    wargs = WorldArgs(no_gui=True, fps=15, turn_based=False, update_interval=0.1,
                      save_replay=False, replay=None, make_video=False,
                      continue_without_training=True,
                      log_dir=os.path.join(root, 'logs'), save_stats=False,
                      match_name=None, seed=None, silence_errors=False,
                      scenario=args.scenario)
    os.makedirs(wargs.log_dir, exist_ok=True)
    world = BombeRLeWorld(wargs, [(args.expert, False)] * 4)
    agent = world.agents[args.seat]

    # Capture what the designated seat's act() returns. All four seats share
    # one imported module, so the wrapper keys on the runner's fake_self.
    runner = agent.backend.runner
    orig_act = runner.callbacks.act
    captured = {}

    def act_wrapped(self_, game_state):
        a = orig_act(self_, game_state)
        if self_ is runner.fake_self:
            captured['action'] = a
        return a
    runner.callbacks.act = act_wrapped

    X, y, act_s, eng_s, rnd, stp, sd = [], [], [], [], [], [], []
    t0 = time.time()
    try:
        for i in range(args.n_rounds):
            seed = args.seed_base + i
            world.rng = np.random.default_rng(seed)
            os.environ['BOMBERMAN_OPPONENT_SEED'] = str(seed)
            np.random.seed(seed)
            world.new_round()
            log = world.replay['actions'][agent.name]
            while world.running:
                n_before = len(log)
                captured.pop('action', None)
                world.do_step()
                if len(log) == n_before:
                    continue                       # seat was dead: not polled
                state = agent.last_game_state      # the dict act() received
                feats = fmod.state_to_features(state)
                a = captured.get('action', None)
                X.append(np.asarray(feats, dtype=np.float32))
                y.append(ACTIONS.index(a) if a in ACTIONS else -1)
                act_s.append('<none>' if a is None else str(a))
                eng_s.append('<none>' if log[-1] is None else str(log[-1]))
                rnd.append(i + 1); stp.append(state['step']); sd.append(seed)
            if (i + 1) % 25 == 0 or i + 1 == args.n_rounds:
                print(f"  {i + 1}/{args.n_rounds} rounds, {len(X)} rows "
                      f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    finally:
        world.end()

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int8)
    layout = fmod.state_to_features.__doc__ or ''
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path, X=X, y=y,
        action=np.asarray(act_s), engine_action=np.asarray(eng_s),
        round=np.asarray(rnd, dtype=np.int32), step=np.asarray(stp, dtype=np.int32),
        seed=np.asarray(sd, dtype=np.int32),
        feature_version=np.int32(fmod.FEATURE_VERSION),
        n_features=np.int32(fmod.N_FEATURES),
        actions=np.asarray(ACTIONS), agent_name=np.str_(agent.name),
        expert=np.str_(args.expert), scenario=np.str_(args.scenario),
        seed_base=np.int32(args.seed_base), n_rounds=np.int32(args.n_rounds),
        layout=np.str_(layout))

    dist = collections.Counter(act_s)
    mismatch = int(sum(1 for a, b in zip(act_s, eng_s) if a != b))
    print(json.dumps({
        'out': out_path, 'rows': int(len(y)), 'n_features': int(X.shape[1]),
        'feature_version': int(fmod.FEATURE_VERSION), 'agent_name': agent.name,
        'rounds': args.n_rounds, 'seed_base': args.seed_base,
        'action_distribution': {a: dist.get(a, 0) for a in ACTIONS + ['<none>']},
        'engine_vs_policy_mismatches': mismatch,
        'file_bytes': os.path.getsize(out_path),
        'wall_seconds': time.time() - t0,
    }, indent=1), flush=True)


if __name__ == '__main__':
    main()
