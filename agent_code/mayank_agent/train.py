"""
train.py -- only loaded when the framework runs with --train.

Implements tabular-style Q-learning on top of a LINEAR function approximator:

    Q(s, a) = w_a . phi(s)

    td_error = r + gamma * max_a' Q(s', a')  -  Q(s, a)
    w_a     += alpha * td_error * phi(s)
"""

import csv
import os
import subprocess
from collections import deque
from datetime import datetime

import numpy as np

import events as e
from .callbacks import (ACTIONS, FEATURE_VERSION, IDX_BOMB_ESCAPE, IDX_BOMB_HITS_OPP,
                        IDX_DANGER, N_FEATURES, announce_run, blast_coords, bfs_distance,
                        crate_approach_tiles, log_path, model_path, other_positions,
                        read_meta, run_dir, run_name, state_to_features, viable_bomb_tiles,
                        write_meta)

# --- hyperparameters (these are what you tune for the report) --------------
ALPHA = 0.01          # learning rate
GAMMA = 0.95          # discount factor
EPSILON_START = 1.0   # exploration at the very beginning
EPSILON_END = 0.05    # exploration floor
# Multiplied after every round. 0.999 hits the 0.05 floor at round ~2994, which
# was far too early for the 20000-round runs: the agent spent most of its
# training greedy. 0.9998 stretches that to round ~14977.
#   0.999  -- v1 .. v3e
#   0.9998 -- v4 .. v5_classic (20000 rounds, floor at ~15000, i.e. 75% in)
#   0.9999 -- v7_escape (40000 rounds, floor at ~29956, i.e. 75% in)
# Kept at the same three-quarters-of-training proportion as v5_classic, so the
# longer run buys more exploration rather than just a longer greedy tail:
# solve 0.05 = decay ** n for n = 0.75 * rounds.
EPSILON_DECAY = 0.9999

# Paid once per step, on top of whatever the events award. It puts a clock on
# every round: dithering is never free, so any behaviour that makes no progress
# is strictly worse than one that does.
LIVING_COST = -0.1

# --- reward-regime toggles -------------------------------------------------
# The v3d experiment changed two things at once. These switches keep both
# regimes reproducible from the same file instead of one overwriting the other:
#
#   v3c, v3e : both False  -- plain WAITED penalty, no bomb-spot shaping
#   v3d      : both True   -- WAITED free + STALLED -3.0, bomb-spot shaping on
#
# The viable-bomb-spot FEATURE ([21:25]) is always present either way; only its
# shaping is gated here, which is what makes v3e a clean control for v3d.
USE_BOMB_SPOT_SHAPING = False
USE_STALL_PENALTY = False

# --- offence (v12) -----------------------------------------------------------
# Up to v9 the agent was rewarded for surviving and for collecting coins, and
# KILLED_OPPONENT was absent from reward_from_events entirely -- a kill reached
# the agent only as the engine's +5 flowing into the round score, with nothing
# teaching it to seek one. That is the wrong emphasis for this game: settings.py
# pays REWARD_KILL = 5 against REWARD_COIN = 1, and across a tournament those
# totals simply accumulate, so one kill is worth five coins.
#
# Off reproduces v9_urgency's reward table exactly; on adds the three terms
# below. Overridable so both regimes stay runnable from one file.
USE_OFFENCE = os.environ.get('BOMBERMAN_OFFENCE', '1') != '0'

# --- custom events ---------------------------------------------------------
# Each pair is one axis of behaviour, scored symmetrically (see reward_from_events).
MOVED_TOWARD_COIN = 'MOVED_TOWARD_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'

ESCAPED_BLAST = 'ESCAPED_BLAST'            # stepped out of a bomb's blast radius
MOVED_INTO_BLAST = 'MOVED_INTO_BLAST'      # stepped into one

BOMB_NEXT_TO_CRATE = 'BOMB_NEXT_TO_CRATE'  # dropped a bomb that will clear >= 1 crate
USELESS_BOMB = 'USELESS_BOMB'              # dropped a bomb that will clear nothing

MOVED_TOWARD_CRATE = 'MOVED_TOWARD_CRATE'  # only while no coin is reachable
MOVED_AWAY_FROM_CRATE = 'MOVED_AWAY_FROM_CRATE'

# A viable bombing tile hits a crate AND leaves an escape. Preferred over the
# plain crate target, which may be unbombable from anywhere the agent can stand.
MOVED_TOWARD_BOMB_SPOT = 'MOVED_TOWARD_BOMB_SPOT'
MOVED_AWAY_FROM_BOMB_SPOT = 'MOVED_AWAY_FROM_BOMB_SPOT'

STALLED = 'STALLED'                        # waited while in no danger at all
# A move the board refuses while standing in a blast radius. The engine already
# reports INVALID_ACTION, but it scores the same whether the agent wasted a step
# in open ground or burned one of the four it had to get clear. Separating the
# two is the point: the forensics on v5_classic found that 77-89% of its
# self-kills contained at least one refused move inside the blast -- every one of
# them caused by an opponent's body -- and the agent kept re-picking the blocked
# direction because nothing distinguished it from an ordinary wasted turn.
INVALID_WHILE_IN_BLAST = 'INVALID_WHILE_IN_BLAST'

# --- offensive shaping events ------------------------------------------------
OFFENSIVE_BOMB = 'OFFENSIVE_BOMB'          # bomb dropped onto an opponent, with an escape
MOVED_TOWARD_OPPONENT = 'MOVED_TOWARD_OPPONENT'
MOVED_AWAY_FROM_OPPONENT = 'MOVED_AWAY_FROM_OPPONENT'

# How much that costs, over and above the flat INVALID_ACTION penalty. Settable
# per process so a sweep can run several arms in parallel from one working tree
# without editing this file between launches; the value lands in meta.json, so
# every run records the arm it belongs to.
#   -15.0 -- v7_escape   (the default, and what v7 was trained with)
#    -6.0 -- v8_inblast6
#   -10.0 -- v8_inblast10
INVALID_IN_BLAST_PENALTY = float(os.environ.get('BOMBERMAN_INBLAST_PENALTY', -15.0))


def git_commit():
    """Which version of the code produced this model (best effort)."""
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=os.path.dirname(__file__), capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def setup_training(self):
    """Called after setup() when the agent is in training mode."""
    self.epsilon = EPSILON_START
    self.round_rewards = 0.0
    self.transitions = deque(maxlen=1000)   # kept for later experiments

    name = announce_run('train')
    os.makedirs(run_dir(name), exist_ok=True)
    self.logger.info(f"Training run {name!r} (feature version {FEATURE_VERSION}).")

    # The log lives beside the model it produced, so the two cannot drift apart.
    self.log_file = log_path(name)
    if not os.path.isfile(self.log_file):
        with open(self.log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['round', 'steps', 'score', 'reward', 'epsilon'])

    # Stamp the run. Resuming a run keeps its original creation time and
    # round count, so meta.json always describes the model actually on disk.
    meta = read_meta(name)
    self.rounds_trained = meta.get('rounds_trained', 0)
    meta.update({
        'run': name,
        'feature_version': FEATURE_VERSION,
        'n_features': N_FEATURES,
        'actions': list(ACTIONS),
        'hyperparameters': {
            'alpha': ALPHA, 'gamma': GAMMA,
            'epsilon_start': EPSILON_START, 'epsilon_end': EPSILON_END,
            'epsilon_decay': EPSILON_DECAY,
        },
        'reward_regime': {
            'invalid_in_blast': INVALID_IN_BLAST_PENALTY,
            'use_offence': USE_OFFENCE,
            'killed_opponent': 30.0 if USE_OFFENCE else 0.0,
            'offensive_bomb': 8.0 if USE_OFFENCE else 0.0,
            'opponent_nav': [1.0, -1.5] if USE_OFFENCE else [0.0, 0.0],
            'use_bomb_spot_shaping': USE_BOMB_SPOT_SHAPING,
            'use_stall_penalty': USE_STALL_PENALTY,
            'waited': 0.0 if USE_STALL_PENALTY else -1.0,
            'living_cost': LIVING_COST,
        },
        'git_commit': git_commit(),
        'created': meta.get('created') or datetime.now().isoformat(timespec='seconds'),
    })
    write_meta(meta, name)
    self.meta = meta


def step_reward(self, events) -> float:
    """Reward for one step: the events plus the flat cost of having taken it."""
    return reward_from_events(self, events) + LIVING_COST


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Called after every step except the last one. This is where learning happens."""
    if old_game_state is None:
        return

    events = list(events) + auxiliary_events(old_game_state, self_action, new_game_state, events)
    reward = step_reward(self, events)
    self.round_rewards += reward

    update_weights(self, old_game_state, self_action, reward, new_game_state)
    self.transitions.append((old_game_state, self_action, new_game_state, reward))


def end_of_round(self, last_game_state, last_action, events):
    """Called once at the end of each round."""
    reward = step_reward(self, list(events))
    self.round_rewards += reward

    # Terminal state: no future value to bootstrap from.
    update_weights(self, last_game_state, last_action, reward, None)

    # Decay exploration.
    self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    # Log this round.
    with open(self.log_file, 'a', newline='') as f:
        csv.writer(f).writerow([
            last_game_state['round'],
            last_game_state['step'],
            last_game_state['self'][1],
            round(self.round_rewards, 2),
            round(self.epsilon, 4),
        ])
    self.round_rewards = 0.0

    # Save the model next to the log that produced it, and keep meta.json in
    # step so the archived run is self-describing. model_path() already ends in
    # .npy, so np.save writes exactly that name.
    np.save(model_path(), self.weights)

    self.rounds_trained += 1
    self.meta['rounds_trained'] = self.rounds_trained
    self.meta['final_epsilon'] = round(self.epsilon, 4)
    self.meta['updated'] = datetime.now().isoformat(timespec='seconds')
    write_meta(self.meta)


# --------------------------------------------------------------------------
# The Q-learning update
# --------------------------------------------------------------------------

def update_weights(self, old_state, action, reward, new_state):
    if action is None:
        return

    a = ACTIONS.index(action)
    phi = state_to_features(old_state)

    q_current = self.weights[a] @ phi

    if new_state is None:
        q_target = reward
    else:
        phi_next = state_to_features(new_state)
        q_target = reward + GAMMA * np.max(self.weights @ phi_next)

    td_error = q_target - q_current
    self.weights[a] += ALPHA * td_error * phi


# --------------------------------------------------------------------------
# Rewards
# --------------------------------------------------------------------------

def in_blast(state, pos):
    """Is `pos` inside the blast radius of any bomb currently on the board?"""
    return any(pos in blast_coords(state['field'], bx, by)
               for (bx, by), _ in state['bombs'])


def hunt_events(old_state, new_state, old_pos, new_pos):
    """
    The toward/away pair for closing on the nearest opponent.

    Gated on two conditions, both read from the state the agent acted in:
    a bomb must be available -- walking at someone with nothing to threaten them
    with is not hunting, it is just walking into danger -- and the agent must not
    already be standing in a blast, where getting out ranks above everything.

    Distance is the BFS distance to the nearest opponent, the same measure the
    coin and crate pairs use. The bucketed slots [29:32] would have been the
    obvious alternative, but a bucket only changes at its 3/7 boundaries, so the
    pair would stay silent through most of an approach; [25:29] gives a direction
    rather than a distance and cannot say whether a step closed the gap at all.
    """
    if not old_state['self'][2]:                       # no bomb to threaten with
        return []
    phi = state_to_features(old_state)
    if phi[IDX_DANGER]:                                # escaping outranks hunting
        return []

    old_o = other_positions(old_state)
    new_o = other_positions(new_state)
    if not old_o or not new_o:
        return []
    old_d = bfs_distance(old_state['field'], old_pos, old_o)
    new_d = bfs_distance(new_state['field'], new_pos, new_o)
    if old_d is None or new_d is None:
        return []
    if new_d < old_d:
        return [MOVED_TOWARD_OPPONENT]
    if new_d > old_d:
        return [MOVED_AWAY_FROM_OPPONENT]
    return []


def auxiliary_events(old_state, self_action, new_state, events):
    """Derive the shaping events that the engine does not provide itself."""
    if old_state is None or new_state is None:
        return []

    aux = []
    old_pos = old_state['self'][3]
    new_pos = new_state['self'][3]

    # --- a refused move while standing in fire -----------------------------
    # Judged on old_state: that is where the agent was when it chose, and the
    # blast it was standing in is the one that was about to kill it.
    if e.INVALID_ACTION in events and in_blast(old_state, tuple(old_pos)):
        aux.append(INVALID_WHILE_IN_BLAST)

    # --- closing in on a coin, or failing that, on a crate -----------------
    # Same mechanism for both; the crate version only takes over once no coin
    # is reachable, which mirrors the gate on the crate-direction feature.
    old_d = bfs_distance(old_state['field'], old_pos, old_state['coins'])
    new_d = bfs_distance(new_state['field'], new_pos, new_state['coins'])
    if old_d is not None and new_d is not None:
        if new_d < old_d:
            aux.append(MOVED_TOWARD_COIN)
        elif new_d > old_d:
            aux.append(MOVED_AWAY_FROM_COIN)
    else:
        # Strict precedence: coin > viable bombing tile > any crate. Exactly one
        # navigation pair fires per step, so the three never fight each other.
        old_s = new_s = None
        if USE_BOMB_SPOT_SHAPING:
            # Same occupancy-aware tile set the features use, so the shaping
            # cannot reward walking towards a spot state_to_features has already
            # ruled out as unescapable.
            old_s = bfs_distance(old_state['field'], old_pos,
                                 viable_bomb_tiles(old_state['field'],
                                                   other_positions(old_state)))
            new_s = bfs_distance(new_state['field'], new_pos,
                                 viable_bomb_tiles(new_state['field'],
                                                   other_positions(new_state)))
        if old_s is not None and new_s is not None:
            if new_s < old_s:
                aux.append(MOVED_TOWARD_BOMB_SPOT)
            elif new_s > old_s:
                aux.append(MOVED_AWAY_FROM_BOMB_SPOT)
        else:
            # Fall back to plain crate proximity when nothing is safely bombable.
            old_c = bfs_distance(old_state['field'], old_pos,
                                 crate_approach_tiles(old_state['field']))
            new_c = bfs_distance(new_state['field'], new_pos,
                                 crate_approach_tiles(new_state['field']))
            if old_c is not None and new_c is not None:
                if new_c < old_c:
                    aux.append(MOVED_TOWARD_CRATE)
                elif new_c > old_c:
                    aux.append(MOVED_AWAY_FROM_CRATE)
            elif USE_OFFENCE:
                # Last tier: nothing left to collect and nothing left to blow
                # up, which is exactly the endgame where the only points still
                # on the board are the opponents. Placing it here rather than
                # higher keeps the promise that at most one navigation pair
                # fires per step, and stops hunting from competing with coins.
                aux += hunt_events(old_state, new_state, old_pos, new_pos)

    # --- stepping out of / into a blast radius -----------------------------
    # Only when I actually changed tile. Dropping a bomb also puts me inside a
    # blast radius without moving, and that decision is scored below instead --
    # scoring it here too would punish every bomb twice.
    if new_pos != old_pos:
        was_in = in_blast(old_state, old_pos)
        now_in = in_blast(new_state, new_pos)
        if was_in and not now_in:
            aux.append(ESCAPED_BLAST)
        elif now_in and not was_in:
            aux.append(MOVED_INTO_BLAST)

    # --- stalling: waited with nothing threatening me ----------------------
    # Mirrors feature [9]: danger means a ticking blast radius or a live
    # explosion on my tile. Waiting there can be the only correct move, so it
    # goes unpunished; waiting anywhere else is pure time-wasting.
    if USE_STALL_PENALTY and e.WAITED in events:
        in_danger = (in_blast(old_state, old_pos)
                     or old_state['explosion_map'][old_pos] > 0)
        if not in_danger:
            aux.append(STALLED)

    # --- a bomb aimed at a player ------------------------------------------
    # Read off the very slots the policy sees, so the reward cannot disagree
    # with the features: [32] says this blast covers an opponent where it
    # stands, [16] says an escape would remain. Both are required -- rewarding
    # [32] alone would pay for suicide bombing, which the -10 in-blast penalty
    # and the -50 KILLED_SELF are simultaneously trying to stamp out.
    if USE_OFFENCE and e.BOMB_DROPPED in events:
        phi = state_to_features(old_state)
        if phi[IDX_BOMB_HITS_OPP] and phi[IDX_BOMB_ESCAPE]:
            aux.append(OFFENSIVE_BOMB)

    # --- was this bomb worth dropping? -------------------------------------
    if e.BOMB_DROPPED in events:
        blast = blast_coords(old_state['field'], old_pos[0], old_pos[1])
        hits_crate = any(old_state['field'][t] == 1 for t in blast)
        aux.append(BOMB_NEXT_TO_CRATE if hits_crate else USELESS_BOMB)

    return aux


def reward_from_events(self, events) -> float:
    """
    Map events to numbers.

    Opposing pairs are scored symmetrically (+x / -x) so the agent cannot farm
    the positive half by cycling: stepping toward a coin and back again, or in
    and out of a blast radius, sums to exactly zero.

    The crate pair is the deliberate exception -- see below. Together with the
    per-step LIVING_COST added in step_reward(), that makes every repeated
    two-tile cycle strictly negative rather than merely break-even.
    """
    rewards = {
        # --- what actually scores points -----------------------------------
        e.COIN_COLLECTED: 10.0,      # the objective; everything else is scaled to this
        e.CRATE_DESTROYED: 3.0,      # instrumental: crates hide the coins
        e.COIN_FOUND: 2.0,           # a crate that actually revealed one

        # The engine pays REWARD_KILL = 5 against REWARD_COIN = 1, so a kill is
        # worth five coins; at COIN_COLLECTED = 10 that fixes a kill at +50 on
        # the same scale. +30 is deliberately short of it. The shaped reward is
        # not the game's score, it is a training signal, and a kill is a rarer
        # and far riskier event than a coin: pricing it at the full ratio makes
        # the BOMB row chase kills through blast radii that KILLED_SELF (-50) is
        # simultaneously punishing, which is how v7 learned to stop scoring.
        # +30 keeps a kill clearly the biggest prize on the board while leaving
        # dying strictly worse than killing.
        e.KILLED_OPPONENT: 30.0 if USE_OFFENCE else 0.0,

        # Down-payment on that kill, paid for the decision rather than the
        # outcome: this bomb covers an opponent AND leaves an escape. Without it
        # the +30 arrives four steps late and only by luck, which is precisely
        # why the agent never learned to hunt.
        OFFENSIVE_BOMB: 8.0 if USE_OFFENCE else 0.0,

        # Same asymmetry as the crate and bomb-spot pairs, so closing in and
        # backing off again is strictly loss-making rather than free.
        MOVED_TOWARD_OPPONENT: 1.0 if USE_OFFENCE else 0.0,
        MOVED_AWAY_FROM_OPPONENT: -1.5 if USE_OFFENCE else 0.0,

        # --- symmetric shaping pairs ---------------------------------------
        MOVED_TOWARD_COIN: 1.0,
        MOVED_AWAY_FROM_COIN: -1.0,

        # Deliberately ASYMMETRIC, unlike every other pair here: stepping away
        # costs more than stepping closer pays. A two-tile oscillation earns
        # +1.0 - 1.5 = -0.5 per cycle instead of breaking even, so the 2-cycle
        # the v2 agent fell into is now strictly loss-making rather than free.
        MOVED_TOWARD_CRATE: 1.0,
        MOVED_AWAY_FROM_CRATE: -1.5,

        # Same asymmetry, bigger numbers: this target is the one the agent can
        # actually act on, so it has to outweigh the standing-still habit that
        # v3c acquired (a WAIT bias of roughly +25).
        MOVED_TOWARD_BOMB_SPOT: 2.0,
        MOVED_AWAY_FROM_BOMB_SPOT: -3.0,

        ESCAPED_BLAST: 3.0,          # outranks coin-chasing: never walk into fire for a coin
        MOVED_INTO_BLAST: -3.0,

        BOMB_NEXT_TO_CRATE: 2.0,     # immediate credit for a decision that pays off later
        USELESS_BOMB: -2.0,

        # --- costs ----------------------------------------------------------
        # With USE_STALL_PENALTY on, waiting itself is free -- sitting out a
        # blast behind a corner is often the only survivable move -- and only
        # STALLED (waiting while nothing threatens me) is punished, hard.
        # With it off, we are back to v3c's flat penalty on every wait.
        e.WAITED: 0.0 if USE_STALL_PENALTY else -1.0,
        STALLED: -3.0,
        e.INVALID_ACTION: -3.0,
        # Stacks on top of the -3.0 above. Scaled against KILLED_SELF (-50): a
        # refused move in a blast is not a wasted step, it is a sizeable step
        # towards dying, and the agent has at most four moves to spend.
        # Deliberately short of -50 so that trying and being blocked still beats
        # standing still and burning the timer. The magnitude is the thing being
        # swept -- at -15 the v7 agent stopped dying but also stopped scoring.
        INVALID_WHILE_IN_BLAST: INVALID_IN_BLAST_PENALTY,
        e.BOMB_DROPPED: 0.0,         # was -5 in task 1; the pair above judges bombs now
        e.KILLED_SELF: -50.0,
        e.SURVIVED_ROUND: 0.0,
        # GOT_KILLED fires alongside KILLED_SELF, so leave it at 0 -- giving it a
        # value would double-count every suicide.
        e.GOT_KILLED: 0.0,
    }
    total = sum(rewards.get(ev, 0.0) for ev in events)
    self.logger.debug(f"Events {events} -> reward {total}")
    return total
