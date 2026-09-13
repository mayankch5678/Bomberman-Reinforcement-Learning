"""
convert_weights.py -- one-off: rewrite pickled weight files as .npy.

The weight matrix is a plain float ndarray, so it never needed pickle. Pickled
arrays also carry the NumPy version's object layout, which is what breaks them
across upgrades. This walks every place a weights file lives, reads the pickle,
and writes `<stem>.npy` beside it with np.save.

Originals are never deleted: the .pt file stays exactly where it was.

Searched locations:
    agent_code/*/runs/*/*.pt
    models/archive/*.pt

Usage:
    python3 analysis/convert_weights.py            # convert what is missing
    python3 analysis/convert_weights.py --force    # rewrite existing .npy too
    python3 analysis/convert_weights.py --dry-run  # report, change nothing
"""

import argparse
import glob
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

SEARCH = [
    os.path.join(REPO_ROOT, 'agent_code', '*', 'runs', '*', '*.pt'),
    os.path.join(REPO_ROOT, 'models', 'archive', '*.pt'),
]


def find_pickles():
    found = []
    for pattern in SEARCH:
        found.extend(sorted(glob.glob(pattern)))
    return found


def load_pickled_array(path):
    """Read the pickle and insist it really is a plain numeric array."""
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, np.ndarray):
        raise TypeError(f"not an ndarray but {type(obj).__name__}")
    if obj.dtype == object:
        raise TypeError("object-dtype array -- would still need pickle")
    if not np.issubdtype(obj.dtype, np.number):
        raise TypeError(f"non-numeric dtype {obj.dtype}")
    return obj


def convert(path, force=False, dry_run=False):
    """Returns (status, detail)."""
    target = os.path.splitext(path)[0] + '.npy'
    if os.path.isfile(target) and not force:
        return 'skipped', 'target exists (use --force)'

    try:
        arr = load_pickled_array(path)
    except Exception as exc:                      # noqa: BLE001 - report and move on
        return 'FAILED', str(exc)

    if dry_run:
        return 'would write', f"{arr.shape} {arr.dtype} -> {os.path.basename(target)}"

    np.save(target, arr)                          # target already ends in .npy

    # Read back with the same settings the agent will use and compare exactly.
    back = np.load(target, allow_pickle=False)
    if back.shape != arr.shape or back.dtype != arr.dtype:
        return 'FAILED', 'round-trip changed shape/dtype'
    if not np.array_equal(back, arr):
        return 'FAILED', 'round-trip changed values'
    if back.tobytes() != arr.tobytes():
        return 'FAILED', 'round-trip not bitwise identical'
    return 'converted', f"{arr.shape} {arr.dtype}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--force', action='store_true',
                    help='rewrite .npy files that already exist')
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would happen, write nothing')
    args = ap.parse_args()

    pickles = find_pickles()
    if not pickles:
        print("no pickled weight files found")
        return

    print(f"found {len(pickles)} pickled weight file(s)\n")
    width = max(len(os.path.relpath(p, REPO_ROOT)) for p in pickles)
    counts = {}
    for path in pickles:
        status, detail = convert(path, args.force, args.dry_run)
        counts[status] = counts.get(status, 0) + 1
        rel = os.path.relpath(path, REPO_ROOT)
        print(f"  {rel:<{width}}  {status:<11} {detail}")

    print("\n" + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print("originals left in place; nothing deleted")
    if counts.get('FAILED'):
        sys.exit(1)


if __name__ == '__main__':
    main()
