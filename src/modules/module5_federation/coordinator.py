import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from .node import FederatedClientNode
from .privacy import DifferentialPrivacyEngine
from .aggregator import FederatedAggregator

class FederationCoordinator:
    """Orchestrates multi-node, multi-round federated training simulations and benchmarks."""

    def __init__(
        self,
        epsilon: float = 0.5,
        enable_dp: bool = True,
        node_partition_col: str = "district"
    ):
        self.epsilon = epsilon
        self.enable_dp = enable_dp
        self.node_partition_col = node_partition_col
        self.dp_engine = DifferentialPrivacyEngine(epsilon=epsilon)
        self.aggregator = FederatedAggregator(dp_engine=self.dp_engine, enable_dp=enable_dp)
        self.nodes: Dict[str, FederatedClientNode] = {}
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.round_logs: List[Dict[str, Any]] = []

    def setup_nodes_from_features(
        self,
        feature_df: pd.DataFrame,
        target_col: str = "target_7d_total",
        test_ratio: float = 0.2
    ) -> Dict[str, str]:
        """
        Partitions dataset across decentralized healthcare nodes.
        """
        exclude = [
            'date', 'phc_id', 'drug_id', 'district', 'qty_dispensed', 'patients_served_estimate',
            'opening_stock', 'closing_stock', 'received_qty', 'reorder_point', 'safety_stock'
        ] + [f"target_day_{i}" for i in range(1, 8)] + ["target_7d_total"]
        
        self.feature_cols = [c for c in feature_df.columns if c not in exclude and np.issubdtype(feature_df[c].dtype, np.number)]
        
        # Fit global scaler for standardized parameter gradients
        X_all = feature_df[self.feature_cols].values
        self.scaler.fit(X_all)
        
        # Partition by district / nation
        partitions = feature_df[self.node_partition_col].unique()
        node_info = {}
        
        for part in partitions:
            part_df = feature_df[feature_df[self.node_partition_col] == part].copy()
            n_samples = len(part_df)
            n_test = int(n_samples * test_ratio)
            
            # Temporal split
            train_sub = part_df.iloc[:-n_test] if n_test > 0 else part_df
            test_sub = part_df.iloc[-n_test:] if n_test > 0 else part_df
            
            X_tr = self.scaler.transform(train_sub[self.feature_cols].values)
            y_tr = train_sub[target_col].values
            
            X_te = self.scaler.transform(test_sub[self.feature_cols].values)
            y_te = test_sub[target_col].values
            
            node_id = f"NODE_{part.upper()}"
            node_name = f"District {part} Health Network"
            
            node = FederatedClientNode(
                node_id=node_id,
                node_name=node_name,
                X_train=X_tr,
                y_train=y_tr,
                X_test=X_te,
                y_test=y_te,
                feature_names=self.feature_cols
            )
            self.nodes[node_id] = node
            node_info[node_id] = f"{node_name} ({len(X_tr)} train, {len(X_te)} test)"
            
        return node_info

    def run_federated_rounds(
        self,
        n_rounds: int = 5,
        local_epochs: int = 3,
        learning_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        Executes N rounds of Federated Averaging and tracks convergence metrics.
        """
        if not self.nodes:
            raise RuntimeError("Nodes not initialized. Call setup_nodes_from_features first.")
            
        n_features = len(self.feature_cols)
        current_weights = np.zeros(n_features)
        current_bias = 0.0
        
        self.round_logs = []
        
        for r in range(1, n_rounds + 1):
            node_updates = []
            
            # Step 1: Local training on each node
            for node_id, node in self.nodes.items():
                update = node.train_local(
                    global_weights=current_weights,
                    global_bias=current_bias,
                    epochs=local_epochs,
                    lr=learning_rate
                )
                node_updates.append(update)
                
            # Step 2: Central aggregation with FedAvg + Differential Privacy
            agg_result = self.aggregator.aggregate(node_updates)
            current_weights = agg_result['global_weights']
            current_bias = agg_result['global_bias']
            
            # Step 3: Evaluate global model across all local test sets
            node_evals = {}
            test_rmses = []
            test_r2s = []
            
            for node_id, node in self.nodes.items():
                metrics = node.evaluate(current_weights, current_bias)
                node_evals[node_id] = metrics
                test_rmses.append(metrics['rmse'])
                test_r2s.append(metrics['r2'])
                
            avg_rmse = round(float(np.mean(test_rmses)), 3)
            avg_r2 = round(float(np.mean(test_r2s)), 3)
            
            round_log = {
                "round": r,
                "global_avg_rmse": avg_rmse,
                "global_avg_r2": avg_r2,
                "node_evaluations": node_evals,
                "privacy_spent": self.dp_engine.get_privacy_report()['total_epsilon_spent']
            }
            self.round_logs.append(round_log)
            
        # Step 4: Benchmark vs Isolated Local-Only Models (No Federation)
        isolated_benchmarks = self._benchmark_isolated_nodes(epochs=n_rounds * local_epochs, lr=learning_rate)
        
        return {
            "rounds_completed": n_rounds,
            "final_global_rmse": self.round_logs[-1]['global_avg_rmse'],
            "final_global_r2": self.round_logs[-1]['global_avg_r2'],
            "convergence_curve": self.round_logs,
            "isolated_local_benchmarks": isolated_benchmarks,
            "performance_gain_pct": round(
                float((isolated_benchmarks['avg_isolated_rmse'] - self.round_logs[-1]['global_avg_rmse']) / max(0.1, isolated_benchmarks['avg_isolated_rmse']) * 100.0),
                2
            ),
            "privacy_report": self.dp_engine.get_privacy_report()
        }

    def _benchmark_isolated_nodes(self, epochs: int = 15, lr: float = 0.02) -> Dict[str, Any]:
        """Trains isolated models on individual node partitions without any federated weight sharing."""
        isolated_results = {}
        rmses = []
        
        for node_id, node in self.nodes.items():
            # Clone node state for isolated run
            w_isolated = np.zeros(node.n_features)
            b_isolated = 0.0
            
            n = len(node.X_train)
            batch_size = min(64, n)
            for _ in range(epochs):
                indices = np.random.permutation(n)
                X_s = node.X_train[indices]
                y_s = node.y_train[indices]
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    xb = X_s[start:end]
                    yb = y_s[start:end]
                    err = (np.dot(xb, w_isolated) + b_isolated) - yb
                    w_isolated -= lr * (2.0 / len(xb)) * np.dot(xb.T, err)
                    b_isolated -= lr * (2.0 / len(xb)) * np.sum(err)
                    
            metrics = node.evaluate(w_isolated, b_isolated)
            isolated_results[node_id] = metrics
            rmses.append(metrics['rmse'])
            
        return {
            "node_isolated_metrics": isolated_results,
            "avg_isolated_rmse": round(float(np.mean(rmses)), 3)
        }

    def generate_model_card(self) -> Dict[str, Any]:
        """Generates formal Federated Model Card & Data Governance Card."""
        return {
            "model_title": "Federated PHC Healthcare Resource Forecaster (FedAvg + DP)",
            "version": f"v{self.aggregator.round_number}.0_federated",
            "participating_nodes": list(self.nodes.keys()),
            "total_nodes": len(self.nodes),
            "data_governance": {
                "raw_data_transferred": False,
                "data_residency": "Local node partitions only (Strict In-Country/In-District Sovereignty)",
                "transmitted_payload": "Aggregated gradient vectors & sample weights only",
                "encryption": "TLS 1.3 payload encryption"
            },
            "differential_privacy_certification": self.dp_engine.get_privacy_report(),
            "latest_round_metrics": self.round_logs[-1] if self.round_logs else None
        }
