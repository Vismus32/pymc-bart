# pymc_bart/decision_table.py
# Standalone DecisionTable prototype (no external deps besides numpy)
from __future__ import annotations
from typing import List, Optional, Any
import numpy as np


class SimpleSplitRule:
    """
    Minimal split-rule for continuous variables.
    divide(col_values, split_val) -> boolean mask: True = go left, False = go right
    """
    def divide(self, col_values: np.ndarray, split_val: Any) -> np.ndarray:
        # split_val is a scalar threshold -> left if <= threshold
        return col_values <= split_val


class DecisionTable:
    """
    Balanced binary tree where every depth uses a single split predicate (variable + threshold).
    Leaves = 2**depth. Vectorized predict.
    Minimal, numpy-only implementation for quick testing.
    """
    def __init__(
        self,
        depth: int,
        n_features: int,
        shape: int = 1,
        split_rules: Optional[List[Any]] = None,
    ):
        self.depth = int(depth)
        self.n_features = int(n_features)
        self.shape = int(shape)
        self.n_leaves = 2 ** self.depth
        # For each depth: index of variable to split on (-1 means uninitialized)
        self.split_vars: List[int] = [-1] * self.depth
        # For each depth: split value (threshold) or None
        self.split_vals: List[Optional[Any]] = [None] * self.depth
        # leaf values: shape (n_leaves, shape)
        self.leaf_values = np.zeros((self.n_leaves, self.shape), dtype=float)
        # small record of counts per leaf (not required)
        self.leaf_n = np.zeros(self.n_leaves, dtype=int)
        # split rules per variable (default: SimpleSplitRule for all)
        if split_rules is None:
            self.split_rules = [SimpleSplitRule() for _ in range(self.n_features)]
        else:
            self.split_rules = split_rules

    @classmethod
    def new_table(cls, depth: int, n_features: int, shape: int = 1):
        return cls(depth=depth, n_features=n_features, shape=shape)

    def copy(self) -> "DecisionTable":
        dt = DecisionTable(self.depth, self.n_features, self.shape, self.split_rules)
        dt.split_vars = self.split_vars.copy()
        dt.split_vals = [None if v is None else (v if isinstance(v, (int, float)) else v.copy())
                         for v in self.split_vals]
        dt.leaf_values = self.leaf_values.copy()
        dt.leaf_n = self.leaf_n.copy()
        return dt

    def n_leaves_count(self) -> int:
        return self.n_leaves

    def _path_indices(self, X: np.ndarray) -> np.ndarray:
        """
        Compute leaf index (0..2^D-1) for each row in X.
        We treat left as 0, right as 1, and build bits from top depth->bottom.
        """
        X = np.asarray(X)
        n = X.shape[0]
        idx = np.zeros(n, dtype=int)
        # iterate depths
        for d in range(self.depth):
            sv = self.split_vars[d]
            sval = self.split_vals[d]
            if sv < 0 or sval is None:
                # treat as all-left
                to_left = np.ones(n, dtype=bool)
            else:
                col = X[:, sv]
                rule = self.split_rules[sv]
                to_left = rule.divide(col, sval)
            bit = 1 << (self.depth - 1 - d)
            idx = idx + (~to_left).astype(int) * bit
        return idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_samples, n_features) or (n_features,) -> returns (n_samples, shape) or (shape,) for single sample
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        idx = self._path_indices(X)
        out = self.leaf_values[idx]
        # update leaf_n counts (optional)
        # reset and count
        self.leaf_n[:] = 0
        unique, counts = np.unique(idx, return_counts=True)
        self.leaf_n[unique] = counts
        return out.squeeze()

    def recompute_output_from_assignments(self, X: np.ndarray):
        """
        Places are not cached separately; predict updates leaf_n.
        This helper simply calls predict to refresh leaf_n.
        """
        self.predict(X)

    def draw_leaf_values_from_residuals(self, residuals: np.ndarray, X: np.ndarray, sigma: float = 1.0):
        """
        Simple conjugate-like draw: for each leaf, compute mean of residuals in the leaf and sample normal around it.
        residuals: vector (n_obs,) representing (y - predictions_others) that this tree should fit (i.e., target)
        sigma: observation noise (used to scale variance of draw)
        """
        X = np.asarray(X)
        residuals = np.asarray(residuals).reshape(-1)
        idx = self._path_indices(X)
        for leaf in range(self.n_leaves):
            mask = idx == leaf
            k = mask.sum()
            if k == 0:
                # keep previous value (optionally jitter)
                continue
            # posterior mean ~ mean(resid) and variance ~ sigma^2 / k
            mean = residuals[mask].mean()
            sd = sigma / np.sqrt(k)
            draw = np.random.normal(loc=mean, scale=sd)
            self.leaf_values[leaf] = float(draw)
        # refresh counts
        self.recompute_output_from_assignments(X)

    def set_random_splits_from_X(self, X: np.ndarray, seed: Optional[int] = None):
        """
        Initialize split_vars and split_vals with random choices from X columns and observed values.
        """
        if seed is not None:
            np.random.seed(seed)
        X = np.asarray(X)
        for d in range(self.depth):
            var = np.random.randint(0, self.n_features)
            self.split_vars[d] = int(var)
            # choose random threshold as mid-point between two random values of that column
            col = X[:, var]
            a, b = np.random.choice(np.unique(col), size=min(2, len(np.unique(col))), replace=False)
            self.split_vals[d] = float((a + b) / 2.0) if a != b else float(a)

    def __repr__(self):
        return f"<DecisionTable depth={self.depth} n_leaves={self.n_leaves}>"

