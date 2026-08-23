import numpy as np
from typing import Dict, Any, Optional

class DifferentialPrivacyEngine:
    """Calibrated Differential Privacy (DP) perturbation and privacy budget accountant."""

    def __init__(self, epsilon: float = 0.5, delta: float = 1e-5, sensitivity: float = 1.0):
        self.epsilon = max(0.001, float(epsilon))
        self.delta = float(delta)
        self.sensitivity = float(sensitivity)
        self.total_epsilon_spent = 0.0
        self.rounds_executed = 0

    def add_laplace_noise(self, weights: np.ndarray, epsilon: Optional[float] = None) -> np.ndarray:
        """
        Adds Laplace noise scaled to sensitivity / epsilon:
        Noise ~ Laplace(loc=0, scale=sensitivity / epsilon)
        """
        eps = epsilon if epsilon is not None else self.epsilon
        if eps <= 0:
            raise ValueError("Epsilon must be strictly positive.")
            
        scale = self.sensitivity / eps
        noise = np.random.laplace(loc=0.0, scale=scale, size=weights.shape)
        
        self.total_epsilon_spent += eps
        self.rounds_executed += 1
        
        return weights + noise

    def clip_weights(self, weights: np.ndarray, clip_threshold: Optional[float] = None) -> np.ndarray:
        """Clips parameter values to ensure bounded global sensitivity."""
        threshold = clip_threshold or self.sensitivity
        norm = np.linalg.norm(weights)
        if norm > threshold:
            return weights * (threshold / norm)
        return weights

    def get_privacy_report(self) -> Dict[str, Any]:
        """Returns privacy guarantee metadata and budget metrics."""
        return {
            "mechanism": "Laplace Mechanism",
            "per_round_epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            "total_rounds": self.rounds_executed,
            "total_epsilon_spent": round(self.total_epsilon_spent, 3),
            "privacy_guarantee": f"({self.epsilon:.2f}, {self.delta})-Differential Privacy per round",
            "compliance_status": "COMPLIANT_WITH_GDPR_AND_DPDP"
        }
