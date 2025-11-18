
import numpy as np
from typing import List, Tuple
from .decision_table import DecisionTable

def sse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum((y_true - y_pred) ** 2))

class EnsembleDTMetropolis:
    """
    Metropolis-Hastings sampler for an ensemble of DecisionTables (BART-style).
    """
    def __init__(self, trees: List[DecisionTable], X: np.ndarray, y: np.ndarray, sigma: float = 1.0, rng: int = None):
        self.trees = trees
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(-1)
        self.sigma = float(sigma)
        if rng is not None:
            np.random.seed(rng)
        self.n_trees = len(trees)

    def current_prediction(self) -> np.ndarray:
        # Сумма предсказаний всех деревьев
        pred = np.zeros_like(self.y, dtype=float)
        for tree in self.trees:
            pred += tree.predict(self.X)
        return pred

    def log_likelihood(self, pred: np.ndarray) -> float:
        resid = self.y - pred
        return -0.5 * np.sum((resid / self.sigma) ** 2)

    def propose_tree(self, k: int) -> Tuple[DecisionTable, np.ndarray]:
        """
        Propose a new version of tree k, draw leaf values using residuals
        """
        tree = self.trees[k]
        prop = tree.copy()

        # Случайный сплит proposal (можно использовать ту же логику, что и SimpleDTMetropolis)
        d = np.random.randint(0, prop.depth)
        n_features = prop.n_features
        if np.random.rand() < 0.5:
            # смена переменной
            new_var = np.random.randint(0, n_features)
            prop.split_vars[d] = int(new_var)
            col = self.X[:, new_var]
            uniq = np.unique(col)
            if len(uniq) == 1:
                prop.split_vals[d] = float(uniq[0])
            else:
                a, b = np.random.choice(uniq, size=2, replace=False)
                prop.split_vals[d] = float((a + b)/2)
        else:
            # смена порога
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
                    prop.split_vals[d] = float((a + b)/2)

        # residuals = y - sum(other_trees)
        pred_other = np.zeros_like(self.y)
        for i, t in enumerate(self.trees):
            if i != k:
                pred_other += t.predict(self.X)
        residuals = self.y - pred_other

        # draw leaf values
        prop.draw_leaf_values_from_residuals(residuals, self.X, sigma=self.sigma)

        prop_pred = prop.predict(self.X)
        return prop, prop_pred

    def step(self) -> bool:
        """
        Do one MH step: pick a tree at random, propose, accept/reject.
        Returns True if accepted
        """
        k = np.random.randint(0, self.n_trees)
        cur_pred = self.current_prediction()
        cur_ll = self.log_likelihood(cur_pred)

        prop, prop_pred = self.propose_tree(k)
        prop_ll = self.log_likelihood(prop_pred)

        log_alpha = prop_ll - cur_ll
        if np.log(np.random.rand()) < log_alpha:
            # accept
            self.trees[k].split_vars = prop.split_vars
            self.trees[k].split_vals = prop.split_vals
            self.trees[k].leaf_values = prop.leaf_values
            self.trees[k].recompute_output_from_assignments(self.X)
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
