import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import mean_squared_error, r2_score

class FederatedClientNode:
    """Simulates an isolated decentralized healthcare node (district or national PHC network)."""

    def __init__(
        self,
        node_id: str,
        node_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[list] = None
    ):
        self.node_id = node_id
        self.node_name = node_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or []
        self.n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1

        # Local model parameters: weights (d,) and bias (scalar)
        self.local_weights = np.zeros(self.n_features)
        self.local_bias = 0.0

    def train_local(
        self,
        global_weights: Optional[np.ndarray] = None,
        global_bias: float = 0.0,
        epochs: int = 5,
        lr: float = 0.01,
        l2_reg: float = 0.01
    ) -> Dict[str, Any]:
        """
        Trains model parameters locally on private data partition starting from global weights.
        Raw patient data is never transmitted.
        """
        w = global_weights.copy() if global_weights is not None else self.local_weights.copy()
        b = float(global_bias)
        n = len(self.X_train)
        
        if n == 0:
            return {
                "node_id": self.node_id,
                "node_name": self.node_name,
                "weights": w,
                "bias": b,
                "samples": 0,
                "local_rmse": 0.0
            }

        # Mini-batch / Stochastic Gradient Descent on local data
        batch_size = min(64, n)
        for _ in range(epochs):
            indices = np.random.permutation(n)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]
                
                # Forward prediction: y_hat = X * w + b
                preds = np.dot(xb, w) + b
                errors = preds - yb
                
                # Gradients: dL/dw = (2/N) * X^T * err + 2 * l2_reg * w
                dw = (2.0 / len(xb)) * np.dot(xb.T, errors) + 2.0 * l2_reg * w
                db = (2.0 / len(xb)) * np.sum(errors)
                
                # Gradient update
                w -= lr * dw
                b -= lr * db

        self.local_weights = w
        self.local_bias = b
        
        # Calculate local training loss
        train_preds = np.dot(self.X_train, w) + b
        local_rmse = float(np.sqrt(mean_squared_error(self.y_train, train_preds)))
        
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "weights": w.copy(),
            "bias": float(b),
            "samples": int(n),
            "local_rmse": round(local_rmse, 3)
        }

    def evaluate(self, weights: np.ndarray, bias: float) -> Dict[str, float]:
        """Evaluates model parameters against local out-of-sample test partition."""
        if len(self.X_test) == 0:
            return {"rmse": 0.0, "r2": 0.0}
            
        preds = np.dot(self.X_test, weights) + bias
        rmse = float(np.sqrt(mean_squared_error(self.y_test, preds)))
        r2 = float(r2_score(self.y_test, preds))
        
        return {
            "rmse": round(rmse, 3),
            "r2": round(r2, 3)
        }
