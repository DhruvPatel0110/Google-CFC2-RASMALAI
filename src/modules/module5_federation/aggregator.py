import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from .privacy import DifferentialPrivacyEngine

class FederatedAggregator:
    """Central server performing sample-weighted Federated Averaging (FedAvg) with Differential Privacy."""

    def __init__(self, dp_engine: Optional[DifferentialPrivacyEngine] = None, enable_dp: bool = True):
        self.dp_engine = dp_engine or DifferentialPrivacyEngine()
        self.enable_dp = enable_dp
        self.round_number = 0
        self.global_weights: Optional[np.ndarray] = None
        self.global_bias: float = 0.0
        self.aggregation_history: List[Dict[str, Any]] = []

    def aggregate(self, node_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes sample-weighted FedAvg across received client model updates and applies DP noise.
        """
        if not node_updates:
            raise ValueError("No client node updates provided for aggregation.")
            
        total_samples = sum(u['samples'] for u in node_updates)
        if total_samples == 0:
            raise ValueError("Total sample count across nodes is zero.")
            
        # FedAvg: Weight importance by sample count
        n_features = len(node_updates[0]['weights'])
        agg_weights = np.zeros(n_features)
        agg_bias = 0.0
        
        for update in node_updates:
            weight_factor = update['samples'] / total_samples
            agg_weights += weight_factor * np.array(update['weights'])
            agg_bias += weight_factor * float(update['bias'])
            
        # Apply Differential Privacy Laplace Noise if enabled
        privacy_noise_applied = False
        if self.enable_dp:
            # Scale noise inversely to total samples (more data = less noise needed)
            effective_eps = self.dp_engine.epsilon * np.sqrt(max(1.0, total_samples / 1000.0))
            # Clip weights before noise addition
            agg_weights = self.dp_engine.clip_weights(agg_weights, clip_threshold=10.0)
            agg_weights = self.dp_engine.add_laplace_noise(agg_weights, epsilon=effective_eps)
            privacy_noise_applied = True
            
        self.round_number += 1
        self.global_weights = agg_weights
        self.global_bias = agg_bias
        
        round_meta = {
            "round": self.round_number,
            "participating_nodes": [u['node_id'] for u in node_updates],
            "total_samples_trained": int(total_samples),
            "privacy_noise_applied": privacy_noise_applied,
            "privacy_report": self.dp_engine.get_privacy_report() if self.enable_dp else None
        }
        self.aggregation_history.append(round_meta)
        
        return {
            "global_weights": self.global_weights.copy(),
            "global_bias": float(self.global_bias),
            "metadata": round_meta
        }
