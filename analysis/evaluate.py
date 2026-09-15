"""
evaluate.py -- measure a trained agent's performance with learning switched off.

Runs the agent for N rounds, one fixed seed per round, and reports
mean +/- standard error for:

    score, coins, steps survived, suicide rate, invalid-action rate

Training is disabled: the agent is constructed with train=False, so the
framework never imports train.py. No weight updates happen, no rows are
appended to training_log.csv, and act() takes the greedy branch because
`self.train` is False (see callbacks.py act()).

Examples:
    python analysis/evaluate.py --agent mayank_agent --scenario coin-heaven -n 100
    python analysis/evaluate.py --agent mayank_agent --scenario classic -n 200 --opponents 3
"""

import argparse
import collections
import importlib
import json
import math
import os
import sys

# This script lives in analysis/, but the framework expects to be imported and
# run from the repository root (agent_code.<name>.callbacks, relative log dirs).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np

import settings as s
from environment import BombeRLeWorld, WorldArgs

DEFAULT_SEED_BASE = 20240101
OPPONENT = 'rule_based_agent'
# Must match OPPONENT_SEED_ENV in agent_code/rule_based_agent/callbacks.py.
OPPONENT_SEED_ENV = 'BOMBERMAN_OPPONENT_SEED'
# Agents that predate the run layout keep a bare weights file in their own
# directory. Either format may be present there; the run layout itself is
# always .npy (see callbacks.MODEL_FILE).
LEGACY_MODELS = ('model.npy', 'model.pt')
# The game's action space, used to order the histogram so that an action the
# agent NEVER picks still shows up as a zero row -- which is the interesting case.
ALL_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# An agent may return None (no decision); the engine records that verbatim and
# perform_agent_action() then falls through to INVALID_ACTION.
NO_ACTION = '<none>'


def action_label(action):
    return NO_ACTION if action is None else str(action)


def run_api(agent_name):
    """
    The agent's own run helpers, if it uses the run layout.

    Resolution is deliberately delegated to the agent's callbacks.py rather
    than reimplemented here: that module is the single source of truth, so
    evaluate.py, train.py and the acting agent cannot disagree about which
    file is loaded. Agents without the layout (rule_based_agent, ...) return
    None and fall back to a plain model.pt.
    """
    try:
        mod = importlib.import_module(f'agent_code.{agent_name}.callbacks')
    except ImportError:
        return None
    needed = ('resolve_run', 'model_path', 'announce_run', 'RUN_ENV_VAR')
    return mod if all(hasattr(mod, n) for n in needed) else None


def resolve_model(agent_name, requested_run):
    """
    Work out which model file to evaluate and make the framework agree.

    Returns (model_path, run_name, meta). Setting RUN_ENV_VAR here is what
    makes the agent's own setup() -- called later, inside the world -- resolve
    to this very same path.
    """
    api = run_api(agent_name)
    if api is None:
        if requested_run:
            raise SystemExit(
                f"agent {agent_name!r} does not use the run layout, so --run "
                f"{requested_run!r} means nothing to it")
        for candidate in LEGACY_MODELS:
            path = os.path.join(REPO_ROOT, 'agent_code', agent_name, candidate)
            if os.path.isfile(path):
                print(f"[evaluate] agent {agent_name!r} has no run layout -> {path}",
                      flush=True)
                return path, None, {}
        path = os.path.join(REPO_ROOT, 'agent_code', agent_name, LEGACY_MODELS[0])
        # Hand-coded baselines (rule_based_agent, ...) carry no weights at all
        # and are still worth evaluating as a reference row in the report.
        print(f"[evaluate] agent {agent_name!r} has no model file "
              f"-- treating it as hand-coded", flush=True)
        return None, None, {}

    run = api.resolve_run(requested_run)
    os.environ[api.RUN_ENV_VAR] = run
    api.announce_run('evaluate', run)
    meta = api.read_meta(run) if hasattr(api, 'read_meta') else {}
    return api.model_path(run), run, meta


def make_seeds(n, base):
    """
    The fixed seed list: base, base+1, ..., base+n-1.

    Round i always gets the same seed, so two agents evaluated with the same
    -n and --seed-base face exactly the same N arenas and are directly
    comparable. Report the base together with your numbers.
    """
    return [base + i for i in range(n)]


def build_world(agent_name, scenario, n_opponents):
    args = WorldArgs(
        no_gui=True,
        fps=15,
        turn_based=False,
        update_interval=0.1,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,   # nobody is training
        log_dir=os.path.join(REPO_ROOT, 'logs'),
        save_stats=False,
        match_name=None,
        seed=None,                        # re-seeded per round in run_round()
        silence_errors=False,
        scenario=scenario,
    )
    os.makedirs(args.log_dir, exist_ok=True)

    # train=False for every agent -> train.py is never loaded.
    agents = [(agent_name, False)] + [(OPPONENT, False)] * n_opponents
    return BombeRLeWorld(args, agents)


def run_round(world, agent, seed, trace=None):
    """
    Play one round on the arena determined by `seed`; return its stats.

    If `trace` is a list, one record per step is appended to it describing
    where the agent stood, what it chose, and where it ended up.
    """
    # build_arena() is the only consumer of world.rng at round start, so
    # re-seeding here makes the arena a pure function of `seed`.
    world.rng = np.random.default_rng(seed)
    # LOCAL HARNESS CHANGE: publish the round's seed so our local copy of
    # rule_based_agent can seed itself from it (see the note at the top of
    # agent_code/rule_based_agent/callbacks.py). Without this the arena is
    # reproducible but the opponents are not, and the same model scores
    # differently on every rerun of the same seeds.
    os.environ[OPPONENT_SEED_ENV] = str(seed)
    world.new_round()

    log = world.replay['actions'][agent.name]   # the engine records every choice
    while world.running:
        before = (int(agent.x), int(agent.y))
        step_no = world.step + 1
        n_before = len(log)
        world.do_step()
        # A dead agent is polled no further, so nothing is appended for it.
        chosen = log[n_before] if len(log) > n_before else None
        if trace is not None:
            trace.append({'step': step_no, 'from': before, 'action': chosen,
                          'to': (int(agent.x), int(agent.y))})

    # agent.statistics is reset by start_round(), so these are per-round counts.
    st = agent.statistics
    steps = st['steps']
    return {
        'seed': seed,
        'score': st['score'],
        'coins': st['coins'],
        'steps': steps,
        'suicide': 1.0 if st['suicides'] > 0 else 0.0,
        'invalid_rate': (st['invalid'] / steps) if steps > 0 else 0.0,
        'invalid': st['invalid'],
    }


def mean_sem(values):
    """Sample mean and standard error of the mean (ddof=1)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float('nan')
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def proportion_sem(values):
    """SEM for a 0/1 indicator: sqrt(p(1-p)/n)."""
    n = len(values)
    p = sum(values) / n
    return p, math.sqrt(p * (1 - p) / n)


def summarise(rounds):
    col = lambda k: [r[k] for r in rounds]

    score_m, score_e = mean_sem(col('score'))
    coins_m, coins_e = mean_sem(col('coins'))
    steps_m, steps_e = mean_sem(col('steps'))
    suic_m, suic_e = proportion_sem(col('suicide'))
    inv_m, inv_e = mean_sem(col('invalid_rate'))

    total_steps = sum(col('steps'))
    pooled_invalid = sum(col('invalid')) / total_steps if total_steps else 0.0

    return {
        'n_rounds': len(rounds),
        'mean_score': score_m, 'sem_score': score_e,
        'mean_coins': coins_m, 'sem_coins': coins_e,
        'mean_steps': steps_m, 'sem_steps': steps_e,
        'suicide_rate': suic_m, 'sem_suicide_rate': suic_e,
        'invalid_action_rate': inv_m, 'sem_invalid_action_rate': inv_e,
        'pooled_invalid_action_rate': pooled_invalid,
        'total_steps': total_steps,
    }


def print_histogram(counts):
    """How often each action was chosen across the whole evaluation."""
    total = sum(counts.values())
    print()
    print(f"action histogram ({total} actions chosen)")
    print("-" * 44)
    print(f"{'action':<10}{'count':>10}{'share':>10}")
    print("-" * 44)
    # Canonical actions first (including the ones never chosen), then anything
    # unexpected the engine substituted, e.g. a timeout WAIT or ERROR.
    extra = sorted(a for a in counts if a not in ALL_ACTIONS)
    for a in ALL_ACTIONS + extra:
        n = counts.get(a, 0)
        share = n / total if total else 0.0
        flag = '   <- never chosen' if n == 0 else ''
        print(f"{a:<10}{n:>10}{share:>9.1%}{flag}")
    print("-" * 44)
    if NO_ACTION in counts:
        print(f"{NO_ACTION} = the agent returned no action; "
              f"the engine scores that as INVALID_ACTION.")


def fmt_pos(pos):
    return f"({pos[0]},{pos[1]})"


def print_trace(trace, seed, round_index):
    """Step-by-step positions for one round, with a movement summary."""
    print()
    print(f"position trace -- evaluation round {round_index} (seed {seed}), "
          f"{len(trace)} steps")
    print("-" * 56)
    print(f"{'step':>5}  {'from':<9}{'action':<8}{'to':<9}  note")
    print("-" * 56)
    for i, r in enumerate(trace):
        note = ''
        if r['from'] == r['to']:
            note = 'did not move'
        elif i >= 1 and r['to'] == trace[i - 1]['from'] and trace[i - 1]['from'] != trace[i - 1]['to']:
            note = 'stepped back'
        act = action_label(r['action'])
        print(f"{r['step']:>5}  {fmt_pos(r['from']):<9}{act:<8}"
              f"{fmt_pos(r['to']):<9}  {note}")
    print("-" * 56)

    positions = [r['from'] for r in trace] + ([trace[-1]['to']] if trace else [])
    stationary = sum(1 for r in trace if r['from'] == r['to'])
    back = sum(1 for i, r in enumerate(trace)
               if i >= 1 and r['from'] != r['to']
               and r['to'] == trace[i - 1]['from']
               and trace[i - 1]['from'] != trace[i - 1]['to'])
    distinct = len(set(positions))
    counts = collections.Counter(positions)
    print(f"distinct tiles visited : {distinct}")
    print(f"steps without moving   : {stationary} / {len(trace)}")
    print(f"steps stepping back    : {back} / {len(trace)}  "
          f"(A->B->A oscillation)")
    if counts:
        tile, hits = counts.most_common(1)[0]
        print(f"most-visited tile      : {fmt_pos(tile)} ({hits} visits)")
    if stationary == len(trace) and len(trace) > 4:
        print("VERDICT: the agent never moves -- it stands still all round.")
    elif distinct <= 2 and len(trace) > 4:
        print("VERDICT: the agent is stuck -- it never leaves these tiles.")
    elif back > len(trace) * 0.4:
        print("VERDICT: the agent is oscillating -- most steps undo the previous one.")


def report(summary, agent_name, scenario, n_opponents, seed_base, run=None, meta=None):
    n = summary['n_rounds']
    meta = meta or {}
    print()
    print(f"agent={agent_name}  scenario={scenario}  opponents={n_opponents}")
    if run is not None:
        bits = [f"run={run}"]
        if meta.get('feature_version') is not None:
            bits.append(f"feature_version={meta['feature_version']}")
        if meta.get('rounds_trained') is not None:
            bits.append(f"trained_rounds={meta['rounds_trained']}")
        if meta.get('trained_scenario'):
            bits.append(f"trained_on={meta['trained_scenario']}")
        print("  " + "  ".join(bits))
    print(f"rounds={n}  seeds={seed_base}..{seed_base + n - 1}  (training disabled)")
    print("-" * 58)
    print(f"{'metric':<26}{'mean':>12}{'std. error':>14}")
    print("-" * 58)
    print(f"{'score':<26}{summary['mean_score']:>12.3f}{summary['sem_score']:>14.3f}")
    print(f"{'coins':<26}{summary['mean_coins']:>12.3f}{summary['sem_coins']:>14.3f}")
    print(f"{'steps survived':<26}{summary['mean_steps']:>12.3f}{summary['sem_steps']:>14.3f}")
    print(f"{'suicide rate':<26}{summary['suicide_rate']:>12.4f}{summary['sem_suicide_rate']:>14.4f}")
    print(f"{'invalid-action rate':<26}"
          f"{summary['invalid_action_rate']:>12.4f}{summary['sem_invalid_action_rate']:>14.4f}")
    print("-" * 58)
    print(f"invalid-action rate, pooled over all {summary['total_steps']} steps: "
          f"{summary['pooled_invalid_action_rate']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', required=True,
                        help='agent folder name under agent_code/, e.g. mayank_agent')
    parser.add_argument('--scenario', default='coin-heaven', choices=sorted(s.SCENARIOS),
                        help='game mode (default: coin-heaven)')
    parser.add_argument('-n', '--n-rounds', type=int, default=100,
                        help='number of evaluation rounds (default: 100)')
    parser.add_argument('--opponents', type=int, default=0,
                        choices=range(0, s.MAX_AGENTS), metavar='{0..3}',
                        help=f'number of {OPPONENT}s to play against (default: 0)')
    parser.add_argument('--run', default=None,
                        help='archived run to evaluate, i.e. '
                             'agent_code/<agent>/runs/<run>/model.pt '
                             '(default: $BOMBERMAN_RUN, else the agent\'s default run)')
    parser.add_argument('--seed-base', type=int, default=DEFAULT_SEED_BASE,
                        help=f'first seed; round i uses seed_base+i (default: {DEFAULT_SEED_BASE})')
    parser.add_argument('--json', default=None,
                        help='also write the summary and per-round rows to this JSON file')
    parser.add_argument('--histogram', action='store_true',
                        help='count every action the agent chooses and print the totals')
    parser.add_argument('--trace', nargs='?', type=int, const=1, default=None,
                        metavar='ROUND',
                        help='print the position at every step of one evaluation '
                             'round (default: round 1) to show whether the agent '
                             'is stationary or oscillating')
    parser.add_argument('--quiet', action='store_true', help='no per-round progress output')
    args = parser.parse_args()

    agent_dir = os.path.join(REPO_ROOT, 'agent_code', args.agent)
    if not os.path.isdir(agent_dir):
        raise SystemExit(f"no such agent: agent_code/{args.agent}")

    # Resolve and announce BEFORE the world is built: build_world() constructs
    # the agent, whose setup() reads the environment variable set here.
    model, run, meta = resolve_model(args.agent, args.run)
    if model is not None and not os.path.isfile(model):
        raise SystemExit(f"no model at {model} -- train this run first")

    seeds = make_seeds(args.n_rounds, args.seed_base)
    world = build_world(args.agent, args.scenario, args.opponents)
    agent = world.agents[0]   # our agent was added first

    if args.trace is not None and not (1 <= args.trace <= len(seeds)):
        raise SystemExit(f"--trace {args.trace} is outside the evaluated "
                         f"rounds 1..{len(seeds)}")

    rounds = []
    action_counts = collections.Counter()
    trace = None
    try:
        for i, seed in enumerate(seeds, start=1):
            this_trace = [] if i == args.trace else None
            rounds.append(run_round(world, agent, seed, this_trace))
            if this_trace is not None:
                trace = this_trace
            if args.histogram:
                action_counts.update(action_label(a)
                                     for a in world.replay['actions'][agent.name])
            if not args.quiet and (i % 10 == 0 or i == len(seeds)):
                print(f"  {i}/{len(seeds)} rounds", file=sys.stderr)
    finally:
        world.end()

    summary = summarise(rounds)
    report(summary, args.agent, args.scenario, args.opponents, args.seed_base, run, meta)

    if args.histogram:
        print_histogram(action_counts)
        summary['action_counts'] = dict(action_counts)
    if trace is not None:
        print_trace(trace, seeds[args.trace - 1], args.trace)

    if args.json:
        path = args.json if os.path.isabs(args.json) else os.path.join(REPO_ROOT, args.json)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'agent': args.agent, 'run': run, 'model': model,
                'feature_version': meta.get('feature_version'),
                'trained_rounds': meta.get('rounds_trained'),
                'scenario': args.scenario,
                'opponents': args.opponents, 'seed_base': args.seed_base,
                'summary': summary, 'rounds': rounds,
            }, f, indent=2)
        print(f"wrote {path}")


if __name__ == '__main__':
    main()
