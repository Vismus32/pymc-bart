# pymc_bart/dt_sampler.py
# Minimal MH-style sampler for symmetric binary trees (CatBoost-style) with MH moves: grow, prune, change
import numpy as np
from typing import Tuple
from .decision_table import DecisionTable  # переименуй в SymmetricTree, если хочешь

def sse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sum((y_true - y_pred) ** 2))


class SimpleSymmetricTreeMetropolis:
    """
    Metropolis-Hastings sampler for one symmetric binary tree.
    Moves:
      - GROW: add a new level
      - PRUNE: remove last level
      - CHANGE: change feature or threshold at a random level
    Leaf values are drawn conditioned on residuals.
    """

    def __init__(self, tree: DecisionTable, X: np.ndarray, y: np.ndarray, sigma: float = 1.0, rng: int = None):
        self.tree = tree
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(-1)
        self.sigma = float(sigma)
        if rng is not None:
            np.random.seed(rng)

    def current_prediction(self) -> np.ndarray:
        return self.tree.predict(self.X).reshape(-1)

    def log_likelihood(self, pred: np.ndarray) -> float:
        resid = self.y - pred
        return -0.5 * np.sum((resid / self.sigma) ** 2)

    # --- MH moves ---
    def _grow(self):
        """Add new level to the tree"""
        n_features = self.X.shape[1]
        new_var = np.random.randint(0, n_features)
        col = self.X[:, new_var]
        uniq = np.unique(col)
        if len(uniq) == 1:
            new_val = float(uniq[0])
        else:
            a, b = np.random.choice(uniq, size=2, replace=False)
            new_val = float((a + b) / 2.0)
        self.tree.split_vars.append(new_var)
        self.tree.split_vals.append(new_val)
        self.tree.depth += 1
        # update leaf values: 2^depth leaves
        n_leaves = 2 ** self.tree.depth
        self.tree.leaf_values = np.zeros(n_leaves)

    def _prune(self):
        """Remove last level from the tree"""
        if self.tree.depth <= 1:
            return  # cannot prune below depth 1
        self.tree.split_vars.pop()
        self.tree.split_vals.pop()
        self.tree.depth -= 1
        # update leaf values
        n_leaves = 2 ** self.tree.depth
        self.tree.leaf_values = np.zeros(n_leaves)

    def _change(self):
        """Change feature or threshold at a random level"""
        depth = self.tree.depth
        n_features = self.X.shape[1]
        d = np.random.randint(0, depth)
        if np.random.rand() < 0.5:
            # change variable
            new_var = np.random.randint(0, n_features)
            self.tree.split_vars[d] = new_var
            col = self.X[:, new_var]
            uniq = np.unique(col)
            if len(uniq) == 1:
                self.tree.split_vals[d] = float(uniq[0])
            else:
                a, b = np.random.choice(uniq, size=2, replace=False)
                self.tree.split_vals[d] = float((a + b) / 2.0)
        else:
            # change threshold
            var = self.tree.split_vars[d]
            col = self.X[:, var]
            uniq = np.unique(col)
            if len(uniq) == 1:
                self.tree.split_vals[d] = float(uniq[0])
            else:
                a, b = np.random.choice(uniq, size=2, replace=False)
                self.tree.split_vals[d] = float((a + b) / 2.0)

# --- Propose a new tree ---
    def propose(self) -> Tuple[DecisionTable, np.ndarray]:
        prop = self.tree.copy()
        move = np.random.choice(['grow', 'prune', 'change'])
        if move == 'grow':
            prop_grow = SimpleSymmetricTreeMetropolis(prop, self.X, self.y, self.sigma)
            prop_grow._grow()
        elif move == 'prune':
            prop_prune = SimpleSymmetricTreeMetropolis(prop, self.X, self.y, self.sigma)
            prop_prune._prune()
        else:
            prop_change = SimpleSymmetricTreeMetropolis(prop, self.X, self.y, self.sigma)
            prop_change._change()
        # draw leaf values from residuals (single-tree)
        residuals = self.y
        prop.draw_leaf_values_from_residuals(residuals, self.X, sigma=self.sigma)
        pred = prop.predict(self.X).reshape(-1)
        return prop, pred

    def step(self) -> bool:
        cur_pred = self.current_prediction()
        cur_ll = self.log_likelihood(cur_pred)
        prop, prop_pred = self.propose()
        prop_ll = self.log_likelihood(prop_pred)
        log_alpha = prop_ll - cur_ll
        if np.log(np.random.rand()) < log_alpha:
            # accept
            self.tree.split_vars = prop.split_vars
            self.tree.split_vals = prop.split_vals
            self.tree.leaf_values = prop.leaf_values
            self.tree.recompute_output_from_assignments(self.X)
            return True
        return False

    def run(self, n_iter: int = 100, verbose: bool = False) -> dict:
        accepts = 0
        history = {"ll": [], "accepted": []}
        for i in range(n_iter):
            accepted = self.step()
            accepts += int(accepted)
            cur_pred = self.current_prediction()
            cur_ll = self.log_likelihood(cur_pred)
            history["ll"].append(cur_ll)
            history["accepted"].append(bool(accepted))
            if verbose and (i % max(1, n_iter // 10) == 0):
                print(f"iter {i:4d}: ll={cur_ll:.3f} accepted={accepted}")
        history["accept_rate"] = accepts / float(n_iter)
        return history
