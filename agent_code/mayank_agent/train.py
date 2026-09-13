"""
train.py -- only loaded when the framework runs with --train.

Implements tabular-style Q-learning on top of a LINEAR function approximator:

    Q(s, a) = w_a . phi(s)

    td_error = r + gamma * max_a' Q(s', a')  -  Q(s, a)
    w_a     += alpha * td_error * phi(s)
"""

import csv
import os
import pickle
import subprocess
from collections import deque
from datetime import datetime

import numpy as np

import events as e
from .callbacks import (ACTIONS, FEATURE_VERSION, N_FEATURES, announce_run, blast_coords,
                        bfs_distance, crate_approach_tiles, log_path, model_path, read_meta,
                        run_dir, run_name, state_to_features, write_meta)

# --- hyperparameters (these are what you tune for the report) --------------
ALPHA = 0.01          # learning rate
GAMMA = 0.95          # discount factor
EPSILON_START = 1.0   # exploration at the very beginning
EPSILON_END = 0.05    # exploration floor
EPSILON_DECAY = 0.999  # multiplied after every round

# Paid once per step, on top of whatever the events award. It puts a clock on
# every round: dithering is never free, so any behaviour that makes no progress
# is strictly worse than one that does.
LIVING_COST = -0.1

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
    # step so the archived run is self-describing.
    with open(model_path(), 'wb') as f:
        pickle.dump(self.weights, f)

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


def auxiliary_events(old_state, self_action, new_state, events):
    """Derive the shaping events that the engine does not provide itself."""
    if old_state is None or new_state is None:
        return []

    aux = []
    old_pos = old_state['self'][3]
    new_pos = new_state['self'][3]

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
        # Targets are the free tiles NEXT to a crate -- you cannot stand on one.
        old_c = bfs_distance(old_state['field'], old_pos,
                             crate_approach_tiles(old_state['field']))
        new_c = bfs_distance(new_state['field'], new_pos,
                             crate_approach_tiles(new_state['field']))
        if old_c is not None and new_c is not None:
            if new_c < old_c:
                aux.append(MOVED_TOWARD_CRATE)
            elif new_c > old_c:
                aux.append(MOVED_AWAY_FROM_CRATE)

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

        # --- symmetric shaping pairs ---------------------------------------
        MOVED_TOWARD_COIN: 1.0,
        MOVED_AWAY_FROM_COIN: -1.0,

        # Deliberately ASYMMETRIC, unlike every other pair here: stepping away
        # costs more than stepping closer pays. A two-tile oscillation earns
        # +1.0 - 1.5 = -0.5 per cycle instead of breaking even, so the 2-cycle
        # the v2 agent fell into is now strictly loss-making rather than free.
        MOVED_TOWARD_CRATE: 1.0,
        MOVED_AWAY_FROM_CRATE: -1.5,

        ESCAPED_BLAST: 3.0,          # outranks coin-chasing: never walk into fire for a coin
        MOVED_INTO_BLAST: -3.0,

        BOMB_NEXT_TO_CRATE: 2.0,     # immediate credit for a decision that pays off later
        USELESS_BOMB: -2.0,

        # --- costs ----------------------------------------------------------
        e.WAITED: -1.0,
        e.INVALID_ACTION: -3.0,
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
