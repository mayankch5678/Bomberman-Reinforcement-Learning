"""
callbacks.py -- always loaded by the framework.

The gradient-boosted sibling of mayank_agent. Everything below the "Features"
heading is a byte-for-byte copy of mayank_agent/callbacks.py at feature version
6: the same 38 slots, the same urgency floor, the same escape reasoning. The
ONLY thing that differs is the shape of Q.

  mayank_agent : Q(s, a) = w_a . phi(s)          -- one weight vector per action
  mayank_gbt   : Q(s, a) = GBT_a(phi(s))         -- one regression forest per action

Keeping the features identical is the whole point of the comparison: any
difference in play is attributable to the function class and nothing else.
"""

import os

# --- thread pinning: MUST run before numpy/scikit-learn load an OpenMP runtime -
# Measured, not assumed. Predicting ONE row from six boosted models costs, on
# this machine:
#
#     default threads : mean 15.4 ms, p99 28.0 ms, max 51.5 ms
#     OMP_NUM_THREADS=1 : mean  2.3 ms, p99  2.6 ms, max  2.8 ms
#
# 6.7x faster and dramatically tighter in the tail. A single row gives OpenMP
# nothing to parallelise, so the entire difference is threads spinning up and
# contending -- and the tail is what the 0.5 s per-step budget is judged on.
#
# It has to be an environment variable: threadpoolctl cannot see this build's
# OpenMP runtime (threadpool_info() comes back empty), so threadpool_limits() is
# a no-op. The variable is read when the runtime loads, which is why this sits
# above every import and why scikit-learn is imported lazily, inside functions,
# rather than at module scope.
#
# The cost is that fitting is single-threaded too. That is the right trade: fits
# happen ~160 times in a run, whereas predict happens on every single step of
# every round, in training as well as in play.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
from collections import deque

import numpy as np

import settings as s

# joblib ships with scikit-learn; both are imported lazily in setup() so that a
# tournament run which only ever calls act() still pays the import cost once,
# and so that importing this module for its feature code alone (as the analysis
# scripts do) does not require sklearn to be installed at all.

# Order matters: the index of an action here is its row in the weight matrix.
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# How each move changes (x, y). Note: image coordinates, so y grows downwards.
MOVES = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}
DIRECTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

# --- feature-set version ---------------------------------------------------
# Bump this whenever state_to_features() changes what its slots MEAN. A model
# is only ever loaded back into the feature set it was trained on; the check in
# setup() refuses anything else rather than silently scoring garbage.
#   1 -- task 1: bias, walls, coin direction                        (9 features)
#   2 -- task 2: + danger, neighbour safety, bomb value, crates    (21 features)
#   3 -- v3d:    + direction to the nearest VIABLE bombing tile    (25 features)
#   4 -- v6:     + opponent direction / distance / blast overlap   (33 features)
#                and opponents block the escape-route search
#   5 -- v7:     + per-direction escape continuation, recomputed     (38 features)
#                every step against live opponent positions
#   6 -- v9:     [10] rescaled -- a ticking bomb is near-maximally    (38 features)
#                urgent at every timer value, not just the last one
FEATURE_VERSION = 6
N_FEATURES = 38         # must match state_to_features() below

# --- run layout ------------------------------------------------------------
# Every training run owns a directory holding its model, its log and its
# metadata, so a model can never drift apart from the log that produced it:
#
#   agent_code/mayank_agent/runs/<run>/model.pt
#                                     /training_log.csv
#                                     /meta.json
#
# The run name is resolved in exactly one place, resolve_run(), so training,
# acting and evaluation can never disagree about which model is in play:
#   an explicit name (evaluate.py --run) > $BOMBERMAN_RUN > no run at all
#   BOMBERMAN_RUN=v2_crates python main.py play --agents mayank_agent --train 1 ...
# The weight matrix is a plain float ndarray, so it is stored with np.save and
# read back with allow_pickle=False -- no object graph, and stable across NumPy
# versions. np.save appends '.npy' unless the path already ends in it, so the
# constant carries the extension and every path built from model_path() names
# the file exactly. callbacks.py, train.py and evaluate.py all go through
# model_path(), so the three cannot drift apart.
# joblib, not .npy: the model is a list of six fitted estimators, not an array.
MODEL_FILE = 'model.joblib'
LOG_FILE = 'training_log.csv'
META_FILE = 'meta.json'
RUNS_DIR = 'runs'
RUN_ENV_VAR = 'BOMBERMAN_RUN'

# Written into meta.json so a run directory says what kind of model it holds.
# mayank_agent's runs predate the field and are implicitly 'linear'.
MODEL_CLASS = 'HistGradientBoostingRegressor'

# --- capacity, chosen for INFERENCE cost -------------------------------------
# Every step calls predict() once per action, so the per-step cost is
# 6 x (tree traversal over MAX_ITER trees). The tournament allows 0.5 s per step
# on an AMD Ryzen 5 2600, and the linear agent it is being compared against
# spends about 2 ms. These caps keep the boosted model in the same order of
# magnitude rather than trading the whole budget for accuracy; MAX_ITER is the
# single knob to turn if measurement says otherwise.
MAX_ITER = 100          # boosting rounds per action-model
MAX_LEAF_NODES = 31     # sklearn's default; depth is what makes traversal slow
LEARNING_RATE = 0.1

# --- threads: see the OMP_NUM_THREADS block at the top of this file ----------
INFERENCE_THREADS = 1


def resolve_run(run=None):
    """
    The single source of truth for which run is active.

    Precedence: an explicitly passed name (e.g. evaluate.py --run) beats the
    BOMBERMAN_RUN environment variable, which beats no run at all.

    None means "no run": the model sits directly beside this file, which is the
    tournament-submission layout -- one agent directory holding callbacks.py,
    train.py and model.npy, with no runs/ tree. That is deliberately the
    DEFAULT, so a submitted agent loads its own weights with nothing set in the
    environment.
    """
    return run or os.environ.get(RUN_ENV_VAR) or None


def run_name(run=None):
    """Which run this process is reading from / writing to."""
    return resolve_run(run)


def run_dir(name=None):
    here = os.path.dirname(os.path.abspath(__file__))
    run = resolve_run(name)
    return here if run is None else os.path.join(here, RUNS_DIR, run)


def announce_run(source, name=None):
    """
    One-line startup banner. Every entry point prints this, so a mismatch
    between training and evaluation is visible in the first line of output.
    """
    name = resolve_run(name)
    shown = name if name is not None else '<none: agent directory>'
    print(f"[{source}] run {shown!r} -> {model_path(name)}", flush=True)
    return name


def model_path(name=None):
    return os.path.join(run_dir(name), MODEL_FILE)


def log_path(name=None):
    return os.path.join(run_dir(name), LOG_FILE)


def meta_path(name=None):
    return os.path.join(run_dir(name), META_FILE)


def read_meta(name=None):
    """Run metadata, or {} if the run has none yet."""
    try:
        with open(meta_path(name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_meta(meta, name=None):
    os.makedirs(run_dir(name), exist_ok=True)
    with open(meta_path(name), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def check_compatible(models, meta, name):
    """Refuse a model that was not trained against the current feature set."""
    trained_version = meta.get('feature_version')
    trained_features = meta.get('n_features')
    trained_class = meta.get('model_class')

    if len(models) != len(ACTIONS):
        raise ValueError(
            f"run {name!r} holds {len(models)} estimators, but there are "
            f"{len(ACTIONS)} actions. That directory was not written by this agent."
        )
    if trained_class is not None and trained_class != MODEL_CLASS:
        raise ValueError(
            f"run {name!r} holds a {trained_class!r} model, but this code expects "
            f"{MODEL_CLASS!r}. Point {RUN_ENV_VAR} at a run trained by this agent."
        )
    # A boosted model carries no shape to check, so the feature width has to come
    # from the estimator itself -- otherwise a stale run would be caught only by
    # sklearn raising deep inside predict(), one step into the first round.
    seen = {getattr(m, 'n_features_in_', None) for m in models} - {None}
    if seen and seen != {N_FEATURES}:
        raise ValueError(
            f"run {name!r} was fitted on {sorted(seen)} features, but feature "
            f"version {FEATURE_VERSION} produces {N_FEATURES}. Retrain it under a "
            f"new run name."
        )
    if trained_features is not None and trained_features != N_FEATURES:
        raise ValueError(
            f"run {name!r} records {trained_features} features, this code has "
            f"{N_FEATURES}. Retrain under a new run name."
        )
    if trained_version is not None and trained_version != FEATURE_VERSION:
        raise ValueError(
            f"run {name!r} was trained against feature version {trained_version}, "
            f"but this code is version {FEATURE_VERSION}. The feature count happens "
            f"to match, which makes this the dangerous case: the inputs would be "
            f"silently misinterpreted. Retrain under a new run name."
        )

# Timing, derived from the rules in environment.py / items.py:
#   A bomb observed with timer t detonates at the END of the step in which it
#   is observed with timer 0.  Counting the action I am about to choose as
#   move 1, that blast lands after move t + 1, and the explosion lingers for
#   one further step, so the tile is also lethal on move t + 2.
#   A bomb I drop *now* is observed as t = BOMB_TIMER - 1 on my next turn,
#   which leaves me BOMB_TIMER moves to get clear.
FRESH_BOMB_TIMER = s.BOMB_TIMER      # 4 -- effective timer of a bomb dropped this step

# --- urgency floor (version 6) ---------------------------------------------
# Slot [10] used to be the raw fraction (BOMB_TIMER - t)/BOMB_TIMER, so the four
# observable timers read 0.25 / 0.50 / 0.75 / 1.00. That understated the first
# one badly. A bomb grants exactly BOMB_TIMER = 4 moves, and clearing a
# BOMB_POWER = 3 blast down a straight corridor needs all four: there is no
# slack anywhere in the window, so t = 3 is as urgent as t = 0.
#
# The v7 weights showed the cost of pretending otherwise. [10] carries the
# largest weight in that model (-61.3 on WAIT), which is what eventually drives
# the agent out of a blast -- but at 0.25 it is scaled down to about -15, not
# enough to beat WAIT's structural head start from [0] and [9]. Measured over
# 100 rounds, 122 of v7's 184 in-blast WAITs happened at exactly urgency 0.25,
# the one step where waiting costs the escape.
#
# Rescaling to [URGENCY_FLOOR, 1.0] rather than flattening to a constant 1.0:
# a constant would make [10] perfectly collinear with [9] (both 1 in a blast,
# both 0 outside), leaving the model one direction where it had two. Keeping the
# ordering costs nothing and preserves the ability to tell a fresh bomb from one
# about to detonate, which still matters for decisions other than escaping.
URGENCY_FLOOR = 0.9


def urgency(timer):
    """Slot [10] for a bomb observed with `timer` steps left: near 1.0 always."""
    raw = (s.BOMB_TIMER - timer) / s.BOMB_TIMER
    return URGENCY_FLOOR + (1.0 - URGENCY_FLOOR) * raw

# Named slots into the feature vector, so code that reads features does not
# carry magic numbers. See state_to_features() for the full layout.
IDX_DANGER = 9        # am I standing in a blast radius / live explosion?
IDX_SAFE = 11         # [11:15] is stepping UP / RIGHT / DOWN / LEFT survivable?
IDX_BOMB_CRATE = 15   # would a bomb dropped here destroy a crate?
IDX_BOMB_ESCAPE = 16  # would a bomb dropped here still leave me an escape?
IDX_CRATE_DIR = 17    # [17:21] direction to the nearest crate
IDX_SPOT_DIR = 21     # [21:25] direction to the nearest VIABLE bombing tile
IDX_OPP_DIR = 25      # [25:29] direction to the nearest opponent
IDX_OPP_DIST = 29     # [29:32] one-hot bucket of the BFS distance to that opponent
IDX_BOMB_HITS_OPP = 32   # would a bomb dropped here catch an opponent where it stands?
IDX_ESCAPE_DIR = 33   # [33:37] does stepping UP/RIGHT/DOWN/LEFT continue a live escape?
IDX_TRAPPED = 37      # in a blast radius AND at least one escape direction exists

# [11:15] answers "is that tile survivable", which is computed once, before the
# bomb is dropped, and never revisited. [33:37] is the same question asked from
# the position the agent is actually in, every single step, with the other agents
# treated as the obstacles they are. That is the difference that matters after a
# move is refused: the old feature still describes the plan the agent made four
# steps ago, this one describes the board in front of it now.

# Buckets for the opponent distance. A linear model cannot bend a raw distance
# into "too close / about right / irrelevant", so the distance is spent as three
# indicator slots instead. The near edge is one step beyond a bomb's reach
# (BOMB_POWER = 3), i.e. the range in which an opponent can already be hit; the
# far edge is roughly where a chase stops being worth planning around.
OPP_NEAR = 3          # bucket 0: distance <= 3   -- in or at bombing range
OPP_MID = 7           # bucket 1: 4..7            -- worth closing in on
N_OPP_DIST_BUCKETS = 3   # bucket 2: >= 8         -- far away


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

def setup(self):
    """Called once before the first round."""
    import joblib                      # see the note at the top of the file

    name = announce_run('callbacks')
    self.run = name
    path = model_path(name)

    if self.train and not os.path.isfile(path):
        # Fresh start. Unlike a weight matrix there is no "small random" boosted
        # model to begin from: an unfitted estimator cannot predict at all. The
        # agent therefore holds None and acts at random until train.py has run
        # its first fit -- see act().
        self.logger.info(f"Run {name!r}: no model yet -- acting at random until "
                         f"the first fit.")
        os.makedirs(run_dir(name), exist_ok=True)
        self.models = None
    else:
        if not os.path.isfile(path):
            where = repr(name) if name is not None else 'the agent directory'
            raise FileNotFoundError(
                f"no model for {where} at {path}. Set {RUN_ENV_VAR} to an "
                f"existing run, or train one first."
            )
        self.logger.info(f"Run {name!r}: loading model from {path}.")
        self.models = joblib.load(path)
        check_compatible(self.models, read_meta(name), name)


# --------------------------------------------------------------------------
# Acting
# --------------------------------------------------------------------------

def explorable_actions(features):
    """
    The actions exploration is allowed to sample, given the current features.

    Random exploration is what teaches the bomb features, but unrestricted
    random bombing is self-defeating: the escape that follows is random too, so
    almost every exploratory bomb ends in KILLED_SELF (-50) and the BOMB row
    learns that bombing is fatal before it can learn that bombing pays. Masking
    the two provably-fatal choices removes that bias without telling the agent
    which of the remaining actions is good.

    Excluded:
      - BOMB, when no escape route would remain (feature [16] == 0)
      - a move into a tile the safety features mark unsurvivable ([11:15] == 0)

    WAIT is always available, so the result is never empty.
    """
    allowed = [name for i, name in enumerate(DIRECTIONS)
               if features[IDX_SAFE + i] > 0]
    allowed.append('WAIT')
    if features[IDX_BOMB_ESCAPE] > 0:
        allowed.append('BOMB')
    return allowed


def q_values(models, features):
    """
    Q(s, .) as a vector -- one call per action-model.

    predict() wants a 2-D array, and building that view once and reusing it for
    all six calls matters: at one step per 0.5 s budget the arithmetic is free,
    but sklearn's per-call input validation is not, and it is paid six times.
    """
    row = np.asarray(features, dtype=np.float64).reshape(1, -1)
    return np.array([m.predict(row)[0] for m in models])


def act(self, game_state: dict) -> str:
    """Called once per step. Must return one of ACTIONS."""
    features = state_to_features(game_state)

    # Before the first fit there is no Q to be greedy about. Sampling from the
    # same mask the linear agent explores with keeps the two agents' random
    # behaviour identical, so the comparison starts from the same place.
    if getattr(self, 'models', None) is None:
        return str(np.random.choice(explorable_actions(features)))

    # Exploration: with probability epsilon, act randomly -- uniformly over the
    # actions that are not already known to be fatal.
    if self.train and np.random.rand() < self.epsilon:
        return str(np.random.choice(explorable_actions(features)))

    # Greedy selection stays unrestricted: the mask shapes what the agent
    # TRIES while learning, never what it is allowed to conclude.
    return ACTIONS[int(np.argmax(q_values(self.models, features)))]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def state_to_features(game_state: dict) -> np.ndarray:
    """
    Turn the full game_state dict into a short vector of length N_FEATURES.

    Layout:
      [0]      bias term (always 1)
      [1:5]    is the tile UP / RIGHT / DOWN / LEFT of me blocked?  (1 = blocked)
      [5:9]    one-hot: which direction is the first step towards the nearest coin?
      [9]      am I standing in the blast radius of a ticking bomb / live explosion?
      [10]     how urgent is that danger?  0 = safe, otherwise >= URGENCY_FLOOR --
               with only BOMB_TIMER moves to clear a BOMB_POWER blast there is no
               slack, so every ticking bomb reads as near-maximally urgent
      [11:15]  for UP / RIGHT / DOWN / LEFT: is stepping there survivable?
      [15]     would dropping a bomb here destroy at least one crate?
      [16]     would dropping a bomb here still leave me a reachable escape tile?
      [17:21]  one-hot: direction to the nearest crate, *only* when no coin is reachable
      [21:25]  one-hot: direction to the nearest VIABLE bombing tile -- one where
               a bomb would BOTH hit a crate AND leave an escape. Same gate.
      [25:29]  one-hot: direction to the nearest opponent (all zero if none reachable)
      [29:32]  one-hot: how far that opponent is -- <=3 / 4..7 / >=8 steps
      [32]     would a bomb dropped here catch an opponent where it now stands?
      [33:37]  for UP / RIGHT / DOWN / LEFT: does stepping there continue a live
               escape from the danger on the board, given where opponents stand
               this step? Recomputed every step, so it refreshes after a refused move.
      [37]     am I inside a blast radius AND is at least one escape direction open?
    """
    if game_state is None:
        return np.zeros(N_FEATURES)

    field = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = other_positions(game_state)

    features = np.zeros(N_FEATURES)
    features[0] = 1.0

    # --- surroundings ------------------------------------------------------
    for i, name in enumerate(DIRECTIONS):
        dx, dy = MOVES[name]
        features[1 + i] = 0.0 if field[x + dx, y + dy] == 0 else 1.0

    # --- direction to nearest coin ----------------------------------------
    step = bfs_next_step(field, (x, y), coins)
    if step is not None:
        dx, dy = step[0] - x, step[1] - y
        for i, name in enumerate(DIRECTIONS):
            if MOVES[name] == (dx, dy):
                features[5 + i] = 1.0

    # --- danger on my own tile --------------------------------------------
    danger = danger_map(field, bombs)
    bomb_tiles = {tuple(pos) for pos, _ in bombs}
    horizon = blast_horizon(danger, explosion_map)

    timer_here = danger.get((x, y))
    if explosion_map[x, y] > 0:
        features[9], features[10] = 1.0, 1.0
    elif timer_here is not None:
        features[9] = 1.0
        # Every ticking bomb reads as near-maximally urgent; see URGENCY_FLOOR.
        features[10] = urgency(timer_here)

    # --- is each neighbouring tile survivable? -----------------------------
    for i, name in enumerate(DIRECTIONS):
        dx, dy = MOVES[name]
        nxt = (x + dx, y + dy)
        if field[nxt] != 0 or nxt in bomb_tiles:
            continue                      # cannot go there at all
        if lethal_at(nxt, 1, danger, explosion_map):
            continue                      # stepping there dies immediately
        if survivable(field, nxt, danger, explosion_map, bomb_tiles, horizon, start_move=1):
            features[11 + i] = 1.0

    # --- consequences of dropping a bomb right here ------------------------
    if bombs_left:
        blast = blast_coords(field, x, y)
        features[15] = 1.0 if any(field[t] == 1 for t in blast) else 0.0

        hypothetical = dict(danger)
        for t in blast:
            hypothetical[t] = min(hypothetical.get(t, FRESH_BOMB_TIMER), FRESH_BOMB_TIMER)
        hyp_horizon = max(horizon, FRESH_BOMB_TIMER + 2)
        # Move 1 is the BOMB action itself: I stay put, and from then on the
        # tile I am standing on is blocked for re-entry.
        features[16] = 1.0 if survivable(field, (x, y), hypothetical, explosion_map,
                                         bomb_tiles | {(x, y)}, hyp_horizon,
                                         start_move=1) else 0.0

        # Would that bomb catch somebody? Measured against where the opponents
        # stand right now, which is a lower bound on the real thing -- they get
        # BOMB_TIMER moves to walk out. It is still the signal that separates a
        # bomb aimed at a player from one aimed at a crate.
        blast_set = set(blast)
        features[IDX_BOMB_HITS_OPP] = 1.0 if any(o in blast_set for o in others) else 0.0

    # --- direction to nearest crate, only if no coin is reachable ----------
    if step is None:
        crate_step = bfs_next_step(field, (x, y), crate_approach_tiles(field))
        if crate_step is not None:
            dx, dy = crate_step[0] - x, crate_step[1] - y
            for i, name in enumerate(DIRECTIONS):
                if MOVES[name] == (dx, dy):
                    features[IDX_CRATE_DIR + i] = 1.0

        # --- direction to the nearest tile worth bombing FROM ---------------
        # The crate direction above points at crates the agent may have no safe
        # way to bomb; this points only at tiles where a bomb both pays and is
        # survivable. Goes to zero once the agent is standing on such a tile,
        # which is exactly when [15] and [16] both switch on.
        spot_step = bfs_next_step(field, (x, y), viable_bomb_tiles(field, others))
        if spot_step is not None:
            dx, dy = spot_step[0] - x, spot_step[1] - y
            for i, name in enumerate(DIRECTIONS):
                if MOVES[name] == (dx, dy):
                    features[IDX_SPOT_DIR + i] = 1.0

    # --- can I still walk out of the danger I am in? -----------------------
    # Asked fresh every step against live opponent positions. [11:15] answered
    # this once, for the board as it was before the bomb; after a body plugs the
    # corridor only this one notices, and only this one can tell the agent to
    # turn around instead of repeating a move the board keeps refusing.
    escape = escape_directions(field, (x, y), danger, explosion_map,
                               bomb_tiles, horizon, occupied=others)
    for i in range(4):
        features[IDX_ESCAPE_DIR + i] = escape[i]
    # In danger, but not yet cornered: the state where walking the right way is
    # the whole game. Distinct from [9], which says only that I am in a blast.
    features[IDX_TRAPPED] = 1.0 if (features[IDX_DANGER] and any(escape)) else 0.0

    # --- where is the nearest opponent? ------------------------------------
    # Both slots are read off one BFS, so the direction and the distance can
    # never disagree about which opponent is meant. Opponents are targets here,
    # not obstacles: they sit on free tiles, and a path to one must be allowed
    # to end on the tile it occupies. (Their blocking role is confined to the
    # escape-route search, where a body in a corridor is what kills the agent.)
    opp_goal, _, opp_dist = _bfs(field, (x, y), others)
    if opp_goal is not None:
        opp_step = bfs_next_step(field, (x, y), others)
        if opp_step is not None:
            dx, dy = opp_step[0] - x, opp_step[1] - y
            for i, name in enumerate(DIRECTIONS):
                if MOVES[name] == (dx, dy):
                    features[IDX_OPP_DIR + i] = 1.0

        bucket = 0 if opp_dist <= OPP_NEAR else (1 if opp_dist <= OPP_MID else 2)
        features[IDX_OPP_DIST + bucket] = 1.0

    return features


# --------------------------------------------------------------------------
# Bomb / blast reasoning
# --------------------------------------------------------------------------

def blast_coords(field, x, y):
    """
    Tiles hit by a bomb at (x, y).

    Mirrors Bomb.get_blast_coords() in items.py: the blast reaches BOMB_POWER
    tiles along each axis and is stopped by walls only -- it passes straight
    *through* crates, which is why a single bomb can clear several of them.
    """
    tiles = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, s.BOMB_POWER + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == -1:
                break
            tiles.append((nx, ny))
    return tiles


def danger_map(field, bombs):
    """tile -> observed timer of the soonest bomb whose blast covers it."""
    danger = {}
    for pos, timer in bombs:
        for tile in blast_coords(field, pos[0], pos[1]):
            if timer < danger.get(tile, np.inf):
                danger[tile] = timer
    return danger


def lethal_at(tile, move, danger, explosion_map):
    """Would standing on `tile` at the end of move number `move` kill me?"""
    # A blast already on the board is deadly for the coming step only.
    if move == 1 and explosion_map[tile] > 0:
        return True
    timer = danger.get(tile)
    return timer is not None and move in (timer + 1, timer + 2)


def blast_horizon(danger, explosion_map):
    """How many moves ahead I must stay alive for the board to be clear again."""
    horizon = 1 if explosion_map.max() > 0 else 0
    for timer in danger.values():
        horizon = max(horizon, timer + 2)
    return int(horizon)


def survivable(field, start, danger, explosion_map, bomb_tiles, horizon, start_move=0,
               occupied=()):
    """
    Is there ANY sequence of moves from `start` that stays out of every blast
    until all current bombs have finished exploding?

    Breadth-first over (tile, move) pairs; waiting in place is allowed, which
    matters because sitting still one tile around a corner is often the only
    way out. This answers "do I have a future here", not "where should I go".
    """
    if start_move >= horizon:
        return True

    occupied = set(occupied)
    seen = {(start, start_move)}
    frontier = deque([(start, start_move)])

    while frontier:
        (cx, cy), d = frontier.popleft()
        nd = d + 1
        for nxt in ((cx, cy - 1), (cx + 1, cy), (cx, cy + 1), (cx - 1, cy), (cx, cy)):
            nx, ny = nxt
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if nxt != (cx, cy) and (field[nx, ny] != 0 or nxt in bomb_tiles
                                    or nxt in occupied):
                continue            # walls, crates, bombs and bodies all block
            if lethal_at(nxt, nd, danger, explosion_map):
                continue
            if nd >= horizon:
                return True         # outlived every bomb on the board
            if (nxt, nd) not in seen:
                seen.add((nxt, nd))
                frontier.append((nxt, nd))

    return False


def escape_directions(field, pos, danger, explosion_map, bomb_tiles, horizon, occupied=()):
    """
    For UP/RIGHT/DOWN/LEFT: does stepping there leave a live escape from the
    danger currently on the board?

    This is survivable() asked once per neighbour, from where the agent stands
    right now, with the other agents counted as obstacles. It is deliberately
    recomputed from scratch every step: the whole point is that it answers for
    the board as it is, so a route that a body has just plugged stops being
    offered on the very next step rather than four steps later.
    """
    out = [0.0, 0.0, 0.0, 0.0]
    occupied = set(occupied)
    px, py = pos
    for i, name in enumerate(DIRECTIONS):
        dx, dy = MOVES[name]
        nxt = (px + dx, py + dy)
        if not (0 <= nxt[0] < field.shape[0] and 0 <= nxt[1] < field.shape[1]):
            continue
        if field[nxt] != 0 or nxt in bomb_tiles or nxt in occupied:
            continue                      # cannot go there at all
        if lethal_at(nxt, 1, danger, explosion_map):
            continue                      # stepping there dies immediately
        if survivable(field, nxt, danger, explosion_map, bomb_tiles, horizon,
                      start_move=1, occupied=occupied):
            out[i] = 1.0
    return out


def _escape_reachable(field, start, blast, budget=s.BOMB_TIMER, occupied=()):
    """
    Could I step off `start` onto a tile outside `blast` within `budget` moves?

    A cheaper special case of survivable(): with only a freshly dropped bomb on
    the board, the sole hazard is `blast`, so waiting can never help and simple
    breadth-first reachability to depth `budget` settles it.

    `occupied` holds the tiles other agents are standing on. They are not in
    `field` -- the engine keeps agents out of the arena array -- but they block
    movement just as a crate does, so an escape route that runs through one does
    not exist. Counting it would be the dangerous direction of wrong: the agent
    drops a bomb believing it has an exit and then finds the corridor plugged.
    An opponent may of course step aside on the very next turn; treating them as
    solid is the conservative reading, and a bomb not dropped costs far less
    than a bomb dropped into a dead end.
    """
    occupied = set(occupied)
    seen = {start}
    frontier = deque([(start, 0)])
    while frontier:
        (cx, cy), d = frontier.popleft()
        if d >= budget:
            continue
        for nxt in ((cx, cy - 1), (cx + 1, cy), (cx, cy + 1), (cx - 1, cy)):
            nx, ny = nxt
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nx, ny] != 0 or nxt == start or nxt in seen:
                continue          # walls, crates, and the bomb tile itself
            if nxt in occupied:
                continue          # another agent is standing there
            if nxt not in blast:
                return True       # made it clear with a move to spare
            seen.add(nxt)
            frontier.append((nxt, d + 1))
    return False


# The viable-tile set depends on the arena, which changes just a few times a
# round (when crates are destroyed), and -- since version 4 -- on where the
# other agents stand, which changes every step. state_to_features still runs
# several times per step, so the cache keeps earning its place within a step
# even though opponent movement now turns it over between steps. Key on the raw
# board bytes plus the occupied tiles.
_VIABLE_CACHE = {}
_VIABLE_CACHE_MAX = 256


def viable_bomb_tiles(field, occupied=()):
    """
    Free tiles where dropping a bomb would BOTH destroy at least one crate AND
    leave a reachable escape -- the positions actually worth walking to.

    `occupied` holds the tiles other agents stand on. They cannot be bombed
    FROM (the agent cannot walk onto them) and they do not count as escape
    route, so both the candidate set and the reachability test skip them.
    """
    occupied = frozenset(occupied)
    key = (field.tobytes(), occupied)
    cached = _VIABLE_CACHE.get(key)
    if cached is not None:
        return cached

    tiles = []
    xs, ys = np.where(field == 0)
    for x, y in zip(xs, ys):
        x, y = int(x), int(y)
        if (x, y) in occupied:
            continue                                  # cannot stand there
        blast = set(blast_coords(field, x, y))
        if not any(field[t] == 1 for t in blast):
            continue                                  # nothing to gain
        if _escape_reachable(field, (x, y), blast, occupied=occupied):
            tiles.append((x, y))

    if len(_VIABLE_CACHE) >= _VIABLE_CACHE_MAX:
        _VIABLE_CACHE.clear()
    _VIABLE_CACHE[key] = tiles
    return tiles


def other_positions(game_state):
    """Tiles the other agents currently stand on."""
    return [tuple(pos) for _, _, _, pos in game_state.get('others', ())]


def crate_approach_tiles(field):
    """Free tiles standing next to at least one crate -- i.e. bombing spots."""
    targets = []
    xs, ys = np.where(field == 0)
    for x, y in zip(xs, ys):
        for dx, dy in MOVES.values():
            if field[x + dx, y + dy] == 1:
                targets.append((int(x), int(y)))
                break
    return targets


# --------------------------------------------------------------------------
# Pathfinding helpers (also imported by train.py)
# --------------------------------------------------------------------------

def _bfs(field, start, targets):
    """Breadth-first search over free tiles. Returns (goal, parent_map, dist)."""
    targets = set(map(tuple, targets))
    if not targets:
        return None, {}, None

    parent = {start: None}
    dist = {start: 0}
    frontier = deque([start])

    while frontier:
        cur = frontier.popleft()
        if cur in targets:
            return cur, parent, dist[cur]
        cx, cy = cur
        for nxt in ((cx, cy - 1), (cx + 1, cy), (cx, cy + 1), (cx - 1, cy)):
            nx, ny = nxt
            inside = 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
            if inside and field[nx, ny] == 0 and nxt not in parent:
                parent[nxt] = cur
                dist[nxt] = dist[cur] + 1
                frontier.append(nxt)

    return None, parent, None


def bfs_next_step(field, start, targets):
    """First tile to step onto in order to reach the closest target."""
    goal, parent, _ = _bfs(field, start, targets)
    if goal is None or goal == start:
        return None
    cur = goal
    while parent[cur] is not None and parent[cur] != start:
        cur = parent[cur]
    return None if cur == start else cur


def bfs_distance(field, start, targets):
    """Number of steps to the closest target, or None if unreachable."""
    _, _, dist = _bfs(field, start, targets)
    return dist