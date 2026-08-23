import unittest
import os
import pandas as pd
from src.modules.module2_beds.forecaster import BedOccupancyForecaster
from src.modules.module2_beds.los_model import LengthOfStayModel
from src.modules.module2_beds.occupancy_engine import BedOccupancyEngine
from src.modules.module2_beds.service import BedModuleService

class TestModule2Beds(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.service = BedModuleService()
        cls.metrics = cls.service.train(test_days=30)
        
    def test_bed_model_training_and_metrics(self):
        """Verify model trains and yields low MAE and RMSE on occupancy projection."""
        self.assertIsNotNone(self.service.forecaster.model)
        self.assertGreater(self.metrics['occupancy_r2'], 0.0)
        self.assertLess(self.metrics['occupancy_mae'], 3.0)
        self.assertLess(self.metrics['occupancy_rmse'], 3.5)
        self.assertGreater(self.metrics['train_samples'], 0)
        self.assertGreater(self.metrics['test_samples'], 0)
        
        self.assertTrue(os.path.exists(self.service.forecaster.model_path))
        self.assertTrue(os.path.exists(self.service.forecaster.meta_path))

    def test_los_model_benchmarks(self):
        """Verify Length of Stay model calculates valid empirical medians."""
        los_model = self.service.los_model
        self.assertIsNotNone(los_model)
        
        # Test expected stay durations
        los_pneu = los_model.get_expected_los("pneumonia", "moderate")
        self.assertGreater(los_pneu, 0.0)
        self.assertLess(los_pneu, 15.0)

    def test_transferable_patient_extraction(self):
        """Verify patient candidate extraction applies clinical stability constraints."""
        candidates = self.service.get_transferable_patients(
            phc_id="PHC_001",
            as_of_date="2025-05-15"
        )
        self.assertIsInstance(candidates, list)
        for c in candidates:
            self.assertIn(c['severity'], ['mild', 'moderate'])
            self.assertGreaterEqual(c['days_admitted'], 2)
            self.assertGreaterEqual(c['remaining_los_days'], 1)
            self.assertEqual(c['transfer_eligibility'], "SAFE_FOR_TRANSIT")

    def test_bed_assessment_and_7d_curves(self):
        """Verify assessment produces 7-day trajectories across all 15 facilities."""
        risk_df = self.service.run_assessment()
        self.assertEqual(len(risk_df), 15)
        
        for day in range(1, 8):
            self.assertIn(f'proj_occ_day_{day}', risk_df.columns)
            self.assertIn(f'proj_pct_day_{day}', risk_df.columns)
            self.assertIn(f'proj_avail_day_{day}', risk_df.columns)
            
        self.assertTrue(set(risk_df['alert_level'].unique()).issubset({'CRITICAL', 'WARNING', 'SAFE'}))

    def test_module4_integration_helpers(self):
        """Verify helper hooks for Module 4 redistribution and dashboard reporting."""
        overcrowded = self.service.get_overcrowded_facilities()
        self.assertIsInstance(overcrowded, list)
        
        surplus = self.service.get_capacity_surplus_facilities()
        self.assertIsInstance(surplus, list)
        
        report = self.service.get_facility_bed_report("PHC_001")
        self.assertEqual(report['phc_id'], "PHC_001")
        self.assertIn('daily_7d_forecast', report)
        self.assertEqual(len(report['daily_7d_forecast']), 7)

if __name__ == '__main__':
    unittest.main()
