"""
train.py -- only imported when the framework is asked for a training agent.

This agent is a FROZEN EXPORT. Its model is a table of numpy arrays extracted
from mayank_gbt's fitted scikit-learn ensemble by analysis/export_gbt_numpy.py,
and there is no gradient, no replay buffer and no estimator to refit here: the
whole point of the export is that scikit-learn is not installed at all, so the
boosted trees cannot be grown further in this process.

Training therefore happens in mayank_gbt, and the result is re-exported:

    BOMBERMAN_RUN=<run> python main.py play --agents mayank_gbt --train 1 ...
    python3 analysis/export_gbt_numpy.py \
        --model agent_code/mayank_gbt/runs/<run>/model.joblib \
        --out   agent_code/mayank_gbt_np/model.npz

The three callbacks below exist because the framework's AGENT_API requires them
to be present whenever an agent is constructed with train=True (agents.py checks
for them by name and by argument count, and raises NotImplementedError if either
is missing). They deliberately do nothing rather than raise: a tournament game
runs every agent with train=False, so this file is never even imported there,
and turning an accidental `--train 1` into a hard crash would be a worse failure
than playing correctly without learning. The warning is written to the agent log
AND to the console so that it cannot be mistaken for silent training.
"""

WARNING = ("mayank_gbt_np is a frozen numpy export and cannot be trained. "
           "It will play with its exported weights; nothing will be updated. "
           "Train agent_code/mayank_gbt instead, then re-run "
           "analysis/export_gbt_numpy.py.")


def setup_training(self):
    """Called once when the agent is constructed with train=True."""
    self.logger.warning(WARNING)
    print(f"[mayank_gbt_np] WARNING: {WARNING}", flush=True)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events):
    """Called once per step during training. Nothing to update."""
    return


def end_of_round(self, last_game_state: dict, last_action: str, events):
    """Called once at the end of each training round. Nothing to update."""
    return
