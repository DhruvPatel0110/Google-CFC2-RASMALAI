import unittest
import numpy as np
from src.modules.module5_federation.privacy import DifferentialPrivacyEngine
from src.modules.module5_federation.node import FederatedClientNode
from src.modules.module5_federation.aggregator import FederatedAggregator
from src.modules.module5_federation.service import FederationModuleService

class TestModule5Federation(unittest.TestCase):

    def test_differential_privacy_engine(self):
        """Verify Laplace noise injection and privacy budget accounting."""
        dp = DifferentialPrivacyEngine(epsilon=0.5, sensitivity=1.0)
        weights = np.zeros((10,))
        noisy_weights = dp.add_laplace_noise(weights)
        
        self.assertEqual(noisy_weights.shape, (10,))
        self.assertFalse(np.allclose(noisy_weights, weights))
        
        report = dp.get_privacy_report()
        self.assertEqual(report['total_rounds'], 1)
        self.assertEqual(report['total_epsilon_spent'], 0.5)

    def test_client_node_training(self):
        """Verify local node training updates weights properly."""
        X_tr = np.random.randn(50, 4)
        y_tr = np.dot(X_tr, np.array([2.0, -1.0, 0.5, 3.0])) + 1.0 + np.random.randn(50) * 0.1
        X_te = np.random.randn(20, 4)
        y_te = np.dot(X_te, np.array([2.0, -1.0, 0.5, 3.0])) + 1.0
        
        node = FederatedClientNode(
            node_id="NODE_TEST",
            node_name="Test District Hospital",
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te
        )
        
        res = node.train_local(epochs=10, lr=0.05)
        self.assertEqual(res['node_id'], "NODE_TEST")
        self.assertEqual(len(res['weights']), 4)
        self.assertGreater(res['samples'], 0)
        
        eval_res = node.evaluate(res['weights'], res['bias'])
        self.assertIn('rmse', eval_res)
        self.assertIn('r2', eval_res)

    def test_federated_aggregator_weighted_average(self):
        """Verify sample-weighted FedAvg aggregation math."""
        agg = FederatedAggregator(enable_dp=False)
        
        update1 = {"node_id": "N1", "weights": np.array([10.0, 20.0]), "bias": 2.0, "samples": 100}
        update2 = {"node_id": "N2", "weights": np.array([20.0, 40.0]), "bias": 4.0, "samples": 300}
        
        # Expected: (100*10 + 300*20)/400 = (1000 + 6000)/400 = 17.5
        # Expected: (100*20 + 300*40)/400 = (2000 + 12000)/400 = 35.0
        # Expected bias: (100*2 + 300*4)/400 = (200 + 1200)/400 = 3.5
        result = agg.aggregate([update1, update2])
        
        np.testing.assert_almost_equal(result['global_weights'], np.array([17.5, 35.0]))
        self.assertAlmostEqual(result['global_bias'], 3.5)
        self.assertEqual(result['metadata']['total_samples_trained'], 400)

    def test_full_federated_simulation_service(self):
        """Verify multi-round federated training convergence across real district partitions."""
        service = FederationModuleService()
        results = service.run_simulation(n_rounds=3, epsilon=0.5, enable_dp=True)
        
        self.assertIsInstance(results, dict)
        self.assertEqual(results['rounds_completed'], 3)
        self.assertIn('final_global_rmse', results)
        self.assertIn('convergence_curve', results)
        self.assertEqual(len(results['convergence_curve']), 3)
        
        # Test model card generation
        card = service.get_model_card()
        self.assertIn('model_title', card)
        self.assertIn('data_governance', card)
        self.assertFalse(card['data_governance']['raw_data_transferred'])

if __name__ == '__main__':
    unittest.main()
