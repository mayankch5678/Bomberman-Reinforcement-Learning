"""
verify_gbt_numpy.py -- prove the numpy port reproduces the scikit-learn model.

Collects real game states by actually playing mayank_gbt (so the states are the
ones the agent visits, not synthetic vectors), then scores every state twice:
once through the fitted scikit-learn estimators and once through the exported
numpy arrays. Reports the largest absolute difference in Q and whether the two
ever disagree about which action is best -- the second is what actually decides
play, and it must hold on every single state.

Must run under a scikit-learn that can still read the pickle (1.8.0).

    python3 analysis/verify_gbt_numpy.py --run gbt_v1_r20000 -n 10000
"""

import argparse
import importlib
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np


def collect_states(run, n_target, seed_base, lineups, scenario='classic'):
    """Play mayank_gbt and return the feature vectors it actually saw."""
    from environment import BombeRLeWorld, WorldArgs
    gbt = importlib.import_module('agent_code.mayank_gbt.callbacks')
    os.environ[gbt.RUN_ENV_VAR] = run

    rows, provenance = [], []
    for lineup in lineups:
        wargs = WorldArgs(no_gui=True, fps=15, turn_based=False,
                          update_interval=0.1, save_replay=False, replay=None,
                          make_video=False, continue_without_training=True,
                          log_dir=os.path.join(REPO_ROOT, 'logs'), save_stats=False,
                          match_name=None, seed=None, silence_errors=False,
                          scenario=scenario)
        os.makedirs(wargs.log_dir, exist_ok=True)
        world = BombeRLeWorld(wargs, [('mayank_gbt', False)] + [(o, False) for o in lineup])
        agent = world.agents[0]
        try:
            i = 0
            while len(rows) < n_target * (lineups.index(lineup) + 1) // len(lineups):
                seed = seed_base + i
                i += 1
                world.rng = np.random.default_rng(seed)
                os.environ['BOMBERMAN_OPPONENT_SEED'] = str(seed)
                np.random.seed(seed)
                world.new_round()
                log = world.replay['actions'][agent.name]
                while world.running:
                    n_before = len(log)
                    world.do_step()
                    if len(log) == n_before:
                        continue                 # agent was dead, not polled
                    state = agent.last_game_state
                    if state is not None:
                        rows.append(gbt.state_to_features(state))
                        provenance.append(lineup[0])
                print(f"  [{lineup[0]:<17}] round {i}: {len(rows)} states",
                      file=sys.stderr, flush=True)
        finally:
            world.end()
    X = np.asarray(rows, dtype=np.float64)
    return X[:n_target], provenance[:n_target]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='gbt_v1_r20000')
    p.add_argument('--npz', default='agent_code/mayank_gbt_np/model.npz')
    p.add_argument('-n', '--n-states', type=int, default=10000)
    p.add_argument('--seed-base', type=int, default=20250101)
    p.add_argument('--extra-npz', default=None,
                   help='additional real states to check, e.g. the cloning dataset')
    args = p.parse_args()

    import joblib
    import sklearn
    print(f"[verify] scikit-learn {sklearn.__version__}, numpy {np.__version__}")

    src = f'agent_code/mayank_gbt/runs/{args.run}/model.joblib'
    models = joblib.load(src)
    print(f"[verify] reference: {src} ({len(models)} estimators)")

    npmod = importlib.import_module('agent_code.mayank_gbt_np.callbacks')
    model = npmod.load_model(args.npz)
    npmod.check_compatible(model, args.run)
    print(f"[verify] port: {args.npz} -- {len(model['tree_root'])} trees, "
          f"{len(model['node_value'])} nodes, max depth {int(model['max_depth'])}")

    print(f"[verify] collecting {args.n_states} real game states ...")
    X, prov = collect_states(args.run, args.n_states, args.seed_base,
                             [['rule_based_agent'] * 3, ['random_agent'] * 3])
    import collections
    print(f"[verify] collected {len(X)} states: {dict(collections.Counter(prov))}")
    check(models, model, npmod, X, "real game states (mayank_gbt trajectories)")

    if args.extra_npz and os.path.exists(args.extra_npz):
        d = np.load(args.extra_npz)
        Xe = d['X'].astype(np.float64)
        print(f"\n[verify] extra coverage: {args.extra_npz} ({len(Xe)} states)")
        check(models, model, npmod, Xe, "rule_based cloning states")


def check(models, model, npmod, X, label):
    n = len(X)
    t0 = time.time()
    ref = np.column_stack([m.predict(X) for m in models])
    t_sklearn = time.time() - t0

    t0 = time.time()
    got = np.empty_like(ref)
    for i in range(n):
        got[i] = npmod.q_values(model, X[i])
    t_numpy = time.time() - t0

    diff = np.abs(ref - got)
    max_abs = float(diff.max())
    ref_arg = ref.argmax(axis=1)
    got_arg = got.argmax(axis=1)
    disagree = int((ref_arg != got_arg).sum())

    print(f"--- {label}: {n} states x {ref.shape[1]} actions")
    print(f"    max |Q_sklearn - Q_numpy| : {max_abs:.6e}   (threshold 1e-5)")
    print(f"    mean |difference|         : {float(diff.mean()):.6e}")
    print(f"    Q range in reference      : [{ref.min():.3f}, {ref.max():.3f}]")
    print(f"    argmax disagreements      : {disagree} / {n}")
    print(f"    exact bitwise equality    : {bool(np.array_equal(ref, got))}")
    print(f"    timing: sklearn batch {t_sklearn:.2f}s, numpy per-row loop "
          f"{t_numpy:.2f}s ({1000 * t_numpy / n:.3f} ms/state)")
    ok = max_abs < 1e-5 and disagree == 0
    print(f"    VERDICT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        worst = int(diff.max(axis=1).argmax())
        print(f"    worst state index {worst}: ref={ref[worst]} got={got[worst]}")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
