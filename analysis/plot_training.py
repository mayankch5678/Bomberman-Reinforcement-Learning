"""
plot_training.py -- plot rolling-mean score curves from one or more training logs.

Usage:
    python analysis/plot_training.py agent_code/mayank_agent/training_log.csv
    python analysis/plot_training.py a/training_log.csv b/training_log.csv --window 50

Each log is a CSV written by train.py with columns:
    round, steps, score, reward, epsilon
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')          # save to file, no display needed
import matplotlib.pyplot as plt

DEFAULT_WINDOW = 100
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')


def read_log(path):
    """Return (rounds, scores) as parallel lists of numbers."""
    rounds, scores = [], []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rounds.append(int(row['round']))
            scores.append(float(row['score']))
    return rounds, scores


def rolling_mean(xs, ys, window):
    """
    Trailing rolling mean. The first point is emitted once `window` samples
    exist, and is plotted at the x of the last sample in the window.
    """
    if len(ys) < window:
        return [], []

    out_x, out_y = [], []
    total = sum(ys[:window])
    out_x.append(xs[window - 1])
    out_y.append(total / window)

    for i in range(window, len(ys)):
        total += ys[i] - ys[i - window]
        out_x.append(xs[i])
        out_y.append(total / window)

    return out_x, out_y


def label_for(path):
    """Logs are all called training_log.csv, so name the line after its folder."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent or os.path.basename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logs', nargs='+', help='one or more training_log.csv files')
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW,
                        help=f'rolling mean window in rounds (default: {DEFAULT_WINDOW})')
    parser.add_argument('--out', default=None,
                        help='output filename inside analysis/plots/ '
                             '(default: score_rolling<window>.png)')
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0

    for path in args.logs:
        rounds, scores = read_log(path)
        x, y = rolling_mean(rounds, scores, args.window)
        if not x:
            print(f"skipping {path}: only {len(scores)} rounds, "
                  f"need at least {args.window}")
            continue
        ax.plot(x, y, label=f"{label_for(path)} ({len(scores)} rounds)", linewidth=1.8)
        plotted += 1

    if plotted == 0:
        raise SystemExit("nothing to plot -- no log had enough rounds")

    ax.set_xlabel('round')
    ax.set_ylabel(f'score (rolling mean, window {args.window})')
    ax.set_title('Training progress: score per round')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_name = args.out or f'score_rolling{args.window}.png'
    out_path = os.path.join(PLOT_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
