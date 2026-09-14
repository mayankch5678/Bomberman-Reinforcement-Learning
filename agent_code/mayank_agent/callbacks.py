"""
callbacks.py -- always loaded by the framework.

Contains:
  - the feature extraction (game_state dict  ->  short numeric vector)
  - the linear Q-model (one weight vector per action)
  - the action selection (epsilon-greedy during training, greedy otherwise)
"""

import json
import os
from collections import deque

import numpy as np

import settings as s

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
FEATURE_VERSION = 3
N_FEATURES = 25         # must match state_to_features() below

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
MODEL_FILE = 'model.npy'
LOG_FILE = 'training_log.csv'
META_FILE = 'meta.json'
RUNS_DIR = 'runs'
RUN_ENV_VAR = 'BOMBERMAN_RUN'


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


def check_compatible(weights, meta, name):
    """Refuse a model that was not trained against the current feature set."""
    expected = (len(ACTIONS), N_FEATURES)
    trained_version = meta.get('feature_version')

    if weights.shape != expected:
        raise ValueError(
            f"run {name!r} holds weights of shape {weights.shape}, but feature "
            f"version {FEATURE_VERSION} needs {expected}. That model was trained "
            f"against feature version {trained_version} -- evaluate it from its "
            f"archived results, or retrain it under a new run name."
        )
    if trained_version is not None and trained_version != FEATURE_VERSION:
        raise ValueError(
            f"run {name!r} was trained against feature version {trained_version}, "
            f"but this code is version {FEATURE_VERSION}. The slot count happens to "
            f"match, which makes this the dangerous case: the weights would be "
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

# Named slots into the feature vector, so code that reads features does not
# carry magic numbers. See state_to_features() for the full layout.
IDX_DANGER = 9        # am I standing in a blast radius / live explosion?
IDX_SAFE = 11         # [11:15] is stepping UP / RIGHT / DOWN / LEFT survivable?
IDX_BOMB_CRATE = 15   # would a bomb dropped here destroy a crate?
IDX_BOMB_ESCAPE = 16  # would a bomb dropped here still leave me an escape?
IDX_CRATE_DIR = 17    # [17:21] direction to the nearest crate
IDX_SPOT_DIR = 21     # [21:25] direction to the nearest VIABLE bombing tile


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

def setup(self):
    """Called once before the first round."""
    name = announce_run('callbacks')
    self.run = name
    path = model_path(name)

    if self.train and not os.path.isfile(path):
        # Fresh start: small random weights, shape (n_actions, n_features)
        self.logger.info(f"Run {name!r}: no model yet -- starting from scratch.")
        os.makedirs(run_dir(name), exist_ok=True)
        self.weights = np.random.rand(len(ACTIONS), N_FEATURES) * 0.01
    else:
        if not os.path.isfile(path):
            legacy = os.path.join(run_dir(name), 'model.pt')
            hint = (" That run still has the old pickled model.pt -- convert it "
                    "with analysis/convert_weights.py."
                    if os.path.isfile(legacy) else "")
            where = repr(name) if name is not None else 'the agent directory'
            raise FileNotFoundError(
                f"no model for {where} at {path}. Set {RUN_ENV_VAR} to an "
                f"existing run, or train one first.{hint}"
            )
        self.logger.info(f"Run {name!r}: loading model from {path}.")
        # allow_pickle=False is the point: a weights file can only ever be a
        # plain array, never a pickled object graph.
        self.weights = np.load(path, allow_pickle=False)
        check_compatible(self.weights, read_meta(name), name)


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


def act(self, game_state: dict) -> str:
    """Called once per step. Must return one of ACTIONS."""
    features = state_to_features(game_state)

    # Exploration: with probability epsilon, act randomly -- uniformly over the
    # actions that are not already known to be fatal.
    if self.train and np.random.rand() < self.epsilon:
        return str(np.random.choice(explorable_actions(features)))

    # Greedy selection stays unrestricted: the mask shapes what the agent
    # TRIES while learning, never what it is allowed to conclude.
    q_values = self.weights @ features       # one Q-value per action
    return ACTIONS[int(np.argmax(q_values))]


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
      [10]     how urgent is that danger?  0 = safe, 1 = it goes off after my next move
      [11:15]  for UP / RIGHT / DOWN / LEFT: is stepping there survivable?
      [15]     would dropping a bomb here destroy at least one crate?
      [16]     would dropping a bomb here still leave me a reachable escape tile?
      [17:21]  one-hot: direction to the nearest crate, *only* when no coin is reachable
      [21:25]  one-hot: direction to the nearest VIABLE bombing tile -- one where
               a bomb would BOTH hit a crate AND leave an escape. Same gate.
    """
    if game_state is None:
        return np.zeros(N_FEATURES)

    field = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']

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
        # t = BOMB_TIMER-1 (just dropped) -> small; t = 0 (about to go off) -> 1.0
        features[10] = (s.BOMB_TIMER - timer_here) / s.BOMB_TIMER

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
        spot_step = bfs_next_step(field, (x, y), viable_bomb_tiles(field))
        if spot_step is not None:
            dx, dy = spot_step[0] - x, spot_step[1] - y
            for i, name in enumerate(DIRECTIONS):
                if MOVES[name] == (dx, dy):
                    features[IDX_SPOT_DIR + i] = 1.0

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


def survivable(field, start, danger, explosion_map, bomb_tiles, horizon, start_move=0):
    """
    Is there ANY sequence of moves from `start` that stays out of every blast
    until all current bombs have finished exploding?

    Breadth-first over (tile, move) pairs; waiting in place is allowed, which
    matters because sitting still one tile around a corner is often the only
    way out. This answers "do I have a future here", not "where should I go".
    """
    if start_move >= horizon:
        return True

    seen = {(start, start_move)}
    frontier = deque([(start, start_move)])

    while frontier:
        (cx, cy), d = frontier.popleft()
        nd = d + 1
        for nxt in ((cx, cy - 1), (cx + 1, cy), (cx, cy + 1), (cx - 1, cy), (cx, cy)):
            nx, ny = nxt
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if nxt != (cx, cy) and (field[nx, ny] != 0 or nxt in bomb_tiles):
                continue            # walls, crates and bombs block movement
            if lethal_at(nxt, nd, danger, explosion_map):
                continue
            if nd >= horizon:
                return True         # outlived every bomb on the board
            if (nxt, nd) not in seen:
                seen.add((nxt, nd))
                frontier.append((nxt, nd))

    return False


def _escape_reachable(field, start, blast, budget=s.BOMB_TIMER):
    """
    Could I step off `start` onto a tile outside `blast` within `budget` moves?

    A cheaper special case of survivable(): with only a freshly dropped bomb on
    the board, the sole hazard is `blast`, so waiting can never help and simple
    breadth-first reachability to depth `budget` settles it.
    """
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
            if nxt not in blast:
                return True       # made it clear with a move to spare
            seen.add(nxt)
            frontier.append((nxt, d + 1))
    return False


# The viable-tile set depends only on the arena, which changes just a few times
# a round (when crates are destroyed), while state_to_features runs several
# times per step. Cache on the raw board bytes.
_VIABLE_CACHE = {}
_VIABLE_CACHE_MAX = 256


def viable_bomb_tiles(field):
    """
    Free tiles where dropping a bomb would BOTH destroy at least one crate AND
    leave a reachable escape -- the positions actually worth walking to.
    """
    key = field.tobytes()
    cached = _VIABLE_CACHE.get(key)
    if cached is not None:
        return cached

    tiles = []
    xs, ys = np.where(field == 0)
    for x, y in zip(xs, ys):
        x, y = int(x), int(y)
        blast = set(blast_coords(field, x, y))
        if not any(field[t] == 1 for t in blast):
            continue                                  # nothing to gain
        if _escape_reachable(field, (x, y), blast):
            tiles.append((x, y))

    if len(_VIABLE_CACHE) >= _VIABLE_CACHE_MAX:
        _VIABLE_CACHE.clear()
    _VIABLE_CACHE[key] = tiles
    return tiles

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