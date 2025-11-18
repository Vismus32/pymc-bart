# pymc_bart/dt_sampler.py
# Minimal MH-style sampler that proposes change to DecisionTable splits and accepts/rejects using SSE likelihood.
import numpy as np
from typing import Tuple
from .decision_table import DecisionTable


def sse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sum((y_true - y_pred) ** 2))


class SimpleDTMetropolis:
    """
    Simple Metropolis-Hastings optimizer for one DecisionTable.
    Not PyMC — standalone quick prototype:
      - proposes changes to split variables or split values at random depth
      - draws leaf values conditioned on residuals (simple normal draw)
      - accepts/rejects using exponential(-SSE/(2*sigma^2)) likelihood (equivalent to Gaussian noise)
    """

    def __init__(self, dt: DecisionTable, X: np.ndarray, y: np.ndarray, sigma: float = 1.0, rng: int = None):
        self.dt = dt
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(-1)
        self.sigma = float(sigma)
        if rng is not None:
            np.random.seed(rng)

    def current_prediction(self) -> np.ndarray:
        return self.dt.predict(self.X).reshape(-1)

    def log_likelihood(self, pred: np.ndarray) -> float:
        # Gaussian log-likelihood up to constant
        resid = self.y - pred
        return -0.5 * np.sum((resid / self.sigma) ** 2)

    def propose(self) -> Tuple[DecisionTable, np.ndarray]:
        """
        Returns (proposed_dt, proposed_pred)
        Two types of proposals:
          - change variable at a random depth (and set threshold randomly from observed values)
          - change threshold at a random depth for existing variable
        """
        prop = self.dt.copy()
        depth = prop.depth
        n_features = prop.n_features
        # pick depth
        d = np.random.randint(0, depth)
        if np.random.rand() < 0.5:
            # change variable
            new_var = np.random.randint(0, n_features)
            prop.split_vars[d] = int(new_var)
            col = self.X[:, new_var]
            uniq = np.unique(col)
            if len(uniq) == 1:
                prop.split_vals[d] = float(uniq[0])
            else:
                a, b = np.random.choice(uniq, size=2, replace=False)
                prop.split_vals[d] = float((a + b) / 2.0)
        else:
            # change split value if var set, else set var too
            sv = prop.split_vars[d]
            if sv < 0:
                new_var = np.random.randint(0, n_features)
                prop.split_vars[d] = int(new_var)
                col = self.X[:, new_var]
                uniq = np.unique(col)
                prop.split_vals[d] = float(uniq[np.random.randint(0, len(uniq))])
            else:
                col = self.X[:, sv]
                uniq = np.unique(col)
                if len(uniq) == 1:
                    prop.split_vals[d] = float(uniq[0])
                else:
                    a, b = np.random.choice(uniq, size=2, replace=False)
                    prop.split_vals[d] = float((a + b) / 2.0)
        # draw leaf values for prop to best fit residuals (here residuals = y, since we optimize single tree)
        # In ensemble use-case you'd pass residuals = y - predictions_other_trees
        pred_other = np.zeros_like(self.y)  # simple single-tree test
        residuals = self.y - pred_other
        prop.draw_leaf_values_from_residuals(residuals, self.X, sigma=self.sigma)
        prop_pred = prop.predict(self.X).reshape(-1)
        return prop, prop_pred

    def step(self) -> bool:
        """
        Do one MH step: propose and accept/reject. Returns True if accepted.
        """
        cur_pred = self.current_prediction()
        cur_ll = self.log_likelihood(cur_pred)
        prop, prop_pred = self.propose()
        prop_ll = self.log_likelihood(prop_pred)
        log_alpha = prop_ll - cur_ll
        if np.log(np.random.rand()) < log_alpha:
            # accept
            self.dt.split_vars = prop.split_vars
            self.dt.split_vals = prop.split_vals
            self.dt.leaf_values = prop.leaf_values
            self.dt.recompute_output_from_assignments(self.X)
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
