import pandas as pd
from typing import Dict, List, Any, Optional
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.feature_store import FeatureStore
from .coordinator import FederationCoordinator

class FederationModuleService:
    """End-to-End Service Facade for Module 5 (Federated Learning & Privacy Architecture)."""

    def __init__(self):
        self.cleaned_data: Optional[Dict[str, pd.DataFrame]] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.coordinator: Optional[FederationCoordinator] = None
        self.last_results: Optional[Dict[str, Any]] = None

    def initialize_data(self):
        """Loads and pre-processes feature dataset for federation simulation."""
        if self.feature_df is None:
            raw = DataLoader.load_all()
            self.cleaned_data = DataPreprocessor.clean_all(raw)
            self.feature_df = FeatureStore.build_medicine_features(self.cleaned_data)

    def run_simulation(
        self,
        n_rounds: int = 5,
        epsilon: float = 0.5,
        enable_dp: bool = True
    ) -> Dict[str, Any]:
        """
        Executes multi-round federated training simulation across district nodes.
        """
        self.initialize_data()
        
        self.coordinator = FederationCoordinator(
            epsilon=epsilon,
            enable_dp=enable_dp,
            node_partition_col="district"
        )
        
        self.coordinator.setup_nodes_from_features(self.feature_df)
        self.last_results = self.coordinator.run_federated_rounds(
            n_rounds=n_rounds,
            local_epochs=3,
            learning_rate=0.015
        )
        return self.last_results

    def get_model_card(self) -> Dict[str, Any]:
        """Generates formal Federated Model Card & Governance Report."""
        if self.coordinator is None:
            self.run_simulation(n_rounds=3)
        return self.coordinator.generate_model_card()

    def get_federation_summary(self) -> Dict[str, Any]:
        """Returns condensed metrics for Dashboard widgets."""
        if self.last_results is None:
            self.run_simulation(n_rounds=3)
            
        return {
            "rounds": self.last_results['rounds_completed'],
            "global_rmse": self.last_results['final_global_rmse'],
            "global_r2": self.last_results['final_global_r2'],
            "performance_gain_vs_local": f"{self.last_results['performance_gain_pct']}%",
            "privacy_budget_spent": self.last_results['privacy_report']['total_epsilon_spent'],
            "active_nodes": list(self.coordinator.nodes.keys())
        }
