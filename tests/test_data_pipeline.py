import unittest
import pandas as pd
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.feature_store import FeatureStore

class TestDataPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.raw_data = DataLoader.load_all()
        cls.cleaned_data = DataPreprocessor.clean_all(cls.raw_data)
        
    def test_data_loader_completeness(self):
        """Verify all 11 datasets are loaded properly."""
        self.assertEqual(len(self.raw_data), 11)
        self.assertGreater(len(self.raw_data['facilities']), 0)
        self.assertGreater(len(self.raw_data['medicines']), 0)
        self.assertGreater(len(self.raw_data['medicine_transactions']), 0)
        self.assertGreater(len(self.raw_data['inventory_snapshots']), 0)
        self.assertGreater(len(self.raw_data['bed_occupancy']), 0)
        self.assertGreater(len(self.raw_data['distance_matrix']), 0)

    def test_preprocessor_null_handling(self):
        """Verify that transactions nulls are imputed and values are non-negative."""
        tx_df = self.cleaned_data['medicine_transactions']
        self.assertEqual(tx_df['qty_dispensed'].isnull().sum(), 0)
        self.assertTrue((tx_df['qty_dispensed'] >= 0).all())
        
        bed_df = self.cleaned_data['bed_occupancy']
        self.assertTrue((bed_df['occupied_beds'] >= 0).all())
        self.assertTrue((bed_df['available_beds'] >= 0).all())
        self.assertTrue((bed_df['occupancy_pct'] >= 0).all())

    def test_medicine_feature_store(self):
        """Verify feature store builds valid feature matrices for Module 1."""
        med_feats = FeatureStore.build_medicine_features(self.cleaned_data, forecast_horizon=7)
        self.assertGreater(len(med_feats), 0)
        
        # Check presence of key lag and rolling features
        expected_cols = [
            'lag_qty_1d', 'lag_qty_7d', 'lag_qty_30d',
            'rolling_mean_7d', 'rolling_std_7d', 'rolling_mean_14d',
            'consumption_trend_7d', 'target_7d_total',
            'target_day_1', 'target_day_7'
        ]
        for col in expected_cols:
            self.assertIn(col, med_feats.columns)
            self.assertEqual(med_feats[col].isnull().sum(), 0, f"Nulls found in {col}")

    def test_bed_feature_store(self):
        """Verify feature store builds valid feature matrices for Module 2."""
        bed_feats = FeatureStore.build_bed_features(self.cleaned_data, forecast_horizon=7)
        self.assertGreater(len(bed_feats), 0)
        
        expected_cols = [
            'lag_adm_1d', 'lag_adm_7d', 'lag_occ_1d', 'lag_occ_7d',
            'rolling_adm_mean_7d', 'target_occ_day_1', 'target_occ_day_7'
        ]
        for col in expected_cols:
            self.assertIn(col, bed_feats.columns)
            self.assertEqual(bed_feats[col].isnull().sum(), 0, f"Nulls found in {col}")

if __name__ == '__main__':
    unittest.main()
