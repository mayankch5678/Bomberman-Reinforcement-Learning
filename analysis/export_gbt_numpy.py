"""
export_gbt_numpy.py -- turn a fitted HistGradientBoostingRegressor ensemble into
plain numpy arrays that reproduce its predictions without scikit-learn.

Why this exists. The submitted model.joblib is a pickle of six fitted
scikit-learn estimators, and a pickle is only readable by a compatible version
of the library that wrote it: the file records _sklearn_version 1.8.0 and
scikit-learn >= 1.9.0 refuses it outright (ModuleNotFoundError: No module named
'_loss'), which kills the agent inside setup() before the first step. An
unpinned `scikit-learn` in requirements.txt resolves to the newest release, so
the graders would hit exactly that. Exporting the tree structure into numpy
arrays removes the dependency altogether: the agent then needs nothing but
numpy, which every environment already has, and there is no pickle to go stale.

What is extracted. For each of the six action-models, every tree's node table:

    feature index, numerical threshold, left child, right child, leaf value,
    missing-goes-left flag, is-leaf flag

plus the per-action baseline prediction. Children are rewritten from
per-tree-local indices into indices of one globally concatenated node table, so
all 600 trees live in a single flat array and all six actions can be walked in
one vectorised traversal.

The summation. scikit-learn's HistGradientBoosting applies the learning rate as
shrinkage when a leaf is finalised during fitting, NOT at prediction time, so
the stored leaf values already carry the factor 0.1 and the prediction is a
plain sum:

    raw(s, a) = baseline[a] + sum over that action's trees of leaf_value
    Q(s, a)   = raw(s, a)

That identity is asserted below rather than assumed, and `learning_rate` and
`leaf_scale` are both written to the archive so the arithmetic stays auditable:
leaf_scale is 1.0 precisely because the shrinkage is already inside the values.
The loss is HalfSquaredError, whose link is the identity, so predict() and
_raw_predict() coincide and no inverse link is needed.

Run under the scikit-learn version that can still read the pickle (1.8.0):
    python3 analysis/export_gbt_numpy.py \
        --model agent_code/mayank_gbt/runs/gbt_v1_r20000/model.joblib \
        --out agent_code/mayank_gbt_np/model.npz
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def export(models, feature_version, actions):
    """Flatten every tree of every action-model into one node table."""
    node_feature, node_threshold, node_left, node_right = [], [], [], []
    node_value, node_is_leaf, node_missing_left = [], [], []
    tree_root, tree_action = [], []
    baseline = np.empty(len(models), dtype=np.float64)

    offset = 0
    for a, est in enumerate(models):
        baseline[a] = float(np.asarray(est._baseline_prediction).ravel()[0])
        for iteration in est._predictors:
            if len(iteration) != 1:
                raise SystemExit(
                    f"action {a}: {len(iteration)} trees per iteration; this "
                    f"exporter handles single-output regression only")
            nodes = iteration[0].nodes
            if nodes['is_categorical'].any():
                # A categorical split is answered from a bitset, not a
                # threshold, so the numeric traversal below would silently take
                # the wrong branch. Refuse rather than export something wrong.
                raise SystemExit(
                    f"action {a}: tree contains a categorical split, which this "
                    f"exporter does not support")
            n = len(nodes)
            tree_root.append(offset)
            tree_action.append(a)
            node_feature.append(nodes['feature_idx'].astype(np.int32))
            node_threshold.append(nodes['num_threshold'].astype(np.float64))
            # Children are tree-local in scikit-learn; make them global.
            node_left.append(nodes['left'].astype(np.int64) + offset)
            node_right.append(nodes['right'].astype(np.int64) + offset)
            node_value.append(nodes['value'].astype(np.float64))
            node_is_leaf.append(nodes['is_leaf'].astype(np.uint8))
            node_missing_left.append(nodes['missing_go_to_left'].astype(np.uint8))
            offset += n

    out = {
        'node_feature': np.concatenate(node_feature).astype(np.int32),
        'node_threshold': np.concatenate(node_threshold),
        'node_left': np.concatenate(node_left).astype(np.int32),
        'node_right': np.concatenate(node_right).astype(np.int32),
        'node_value': np.concatenate(node_value),
        'node_is_leaf': np.concatenate(node_is_leaf),
        'node_missing_left': np.concatenate(node_missing_left),
        'tree_root': np.asarray(tree_root, dtype=np.int32),
        'tree_action': np.asarray(tree_action, dtype=np.int32),
        'baseline': baseline,
        'n_actions': np.int32(len(models)),
        'n_features': np.int32(int(models[0].n_features_in_)),
        'feature_version': np.int32(feature_version),
        'actions': np.asarray(actions),
        'learning_rate': np.float64(models[0].learning_rate),
        # 1.0: the learning rate is applied to the leaves during fitting, so the
        # prediction-time summation is unweighted. Kept explicit so predict()
        # never has to hard-code the assumption.
        'leaf_scale': np.float64(1.0),
    }
    # Deepest traversal any tree can need; predict() uses it as a loop bound.
    depth = 0
    for root in out['tree_root']:
        stack = [(int(root), 1)]
        while stack:
            idx, d = stack.pop()
            depth = max(depth, d)
            if not out['node_is_leaf'][idx]:
                stack.append((int(out['node_left'][idx]), d + 1))
                stack.append((int(out['node_right'][idx]), d + 1))
    out['max_depth'] = np.int32(depth)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--feature-version', type=int, default=None)
    args = p.parse_args()

    import joblib
    import sklearn
    print(f"[export] scikit-learn {sklearn.__version__}, numpy {np.__version__}")
    models = joblib.load(args.model)
    print(f"[export] loaded {len(models)} estimators from {args.model}")

    from agent_code.mayank_gbt.callbacks import ACTIONS, FEATURE_VERSION
    fv = args.feature_version if args.feature_version is not None else FEATURE_VERSION

    data = export(models, fv, list(ACTIONS))
    n_trees = len(data['tree_root'])
    print(f"[export] {n_trees} trees, {len(data['node_value'])} nodes, "
          f"max depth {int(data['max_depth'])}")
    print(f"[export] baselines: {np.round(data['baseline'], 6).tolist()}")
    print(f"[export] learning_rate={float(data['learning_rate'])} "
          f"leaf_scale={float(data['leaf_scale'])}")

    # Assert the summation identity on random inputs BEFORE writing anything:
    # if shrinkage were applied at predict time this would fail loudly here.
    rng = np.random.default_rng(0)
    X = rng.random((64, int(data['n_features'])))
    for a, est in enumerate(models):
        ref = est.predict(X)
        manual = np.full(len(X), data['baseline'][a])
        for iteration in est._predictors:
            manual += iteration[0].predict(
                X, np.zeros((0, 8), dtype=np.uint32),
                np.zeros(0, dtype=np.uint32), 1)
        err = np.max(np.abs(ref - manual))
        if err > 1e-12:
            raise SystemExit(f"action {a}: summation identity fails, max err {err}")
    print("[export] summation identity baseline + sum(leaves) == predict(): OK")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"[export] wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == '__main__':
    main()
