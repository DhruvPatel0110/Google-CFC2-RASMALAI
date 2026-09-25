import unittest
import os
import pandas as pd
from src.modules.module1_medicine.forecaster import MedicineDemandForecaster
from src.modules.module1_medicine.stockout_engine import StockoutEngine
from src.modules.module1_medicine.service import MedicineModuleService

class TestModule1Medicine(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.service = MedicineModuleService()
        cls.metrics = cls.service.train(test_days=30)
        
    def test_model_training_and_metrics(self):
        """Verify model trains successfully and produces high R2 score."""
        self.assertIsNotNone(self.service.forecaster.model)
        self.assertGreater(self.metrics['overall_r2'], 0.60)
        self.assertGreater(self.metrics['7d_total_r2'], 0.75)
        self.assertGreater(self.metrics['train_samples'], 0)
        self.assertGreater(self.metrics['test_samples'], 0)
        
        # Verify model files exist on disk
        self.assertTrue(os.path.exists(self.service.forecaster.model_path))
        self.assertTrue(os.path.exists(self.service.forecaster.meta_path))

    def test_feature_importances(self):
        """Verify feature importances are extracted."""
        fi_df = self.service.forecaster.get_feature_importances()
        self.assertGreater(len(fi_df), 0)
        self.assertIn('feature', fi_df.columns)
        self.assertIn('importance', fi_df.columns)
        self.assertAlmostEqual(fi_df['importance'].sum(), 1.0, places=2)

    def test_stockout_assessment_and_alerts(self):
        """Verify assessment produces alert classifications and risk metrics."""
        risk_df = self.service.run_assessment()
        expected_rows = len(self.service.cleaned_data['facilities']) * len(self.service.cleaned_data['medicines'])
        self.assertEqual(len(risk_df), expected_rows)
        
        expected_cols = [
            'phc_id', 'drug_id', 'closing_stock', 'predicted_consumption_7d',
            'days_to_stockout', 'stockout_probability', 'alert_level',
            'shortfall_qty', 'surplus_qty', 'recommendation'
        ]
        for col in expected_cols:
            self.assertIn(col, risk_df.columns)
            
        self.assertTrue(set(risk_df['alert_level'].unique()).issubset({'CRITICAL', 'WARNING', 'SAFE'}))

    def test_critical_stockouts_and_surplus_helpers(self):
        """Verify helper methods for Module 4 integration."""
        critical_alerts = self.service.get_critical_stockouts()
        self.assertIsInstance(critical_alerts, list)
        
        surplus_donors = self.service.get_surplus_facilities()
        self.assertIsInstance(surplus_donors, list)
        
        facility_summary = self.service.get_facility_summary("PHC_001")
        self.assertEqual(facility_summary['phc_id'], "PHC_001")
        self.assertIn('overall_status', facility_summary)
        self.assertEqual(facility_summary['total_skus'], 15)

    def test_procurement_order_generation(self):
        """Verify purchase orders are structured with valid dates and lead times."""
        risk_df = self.service.run_assessment()
        orders = StockoutEngine.generate_procurement_orders(risk_df, pd.Timestamp("2025-08-31"))
        self.assertIsInstance(orders, list)
        if len(orders) > 0:
            order = orders[0]
            self.assertIn('phc_id', order)
            self.assertIn('recommended_order_qty', order)
            self.assertGreater(order['recommended_order_qty'], 0)
            self.assertIn('estimated_cost_inr', order)

if __name__ == '__main__':
    unittest.main()
