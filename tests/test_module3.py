import unittest
import pandas as pd
from src.modules.module3_staff.evaluator import StaffingEvaluator
from src.modules.module3_staff.shift_forecaster import ShiftCoverageForecaster
from src.modules.module3_staff.service import StaffModuleService

class TestModule3Staff(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.service = StaffModuleService()
        cls.service.initialize_data()
        
    def test_evaluator_severity_scoring(self):
        """Verify demand multipliers elevate shortage severity."""
        # Baseline normal load
        res_normal = StaffingEvaluator.calculate_shortage_severity(
            role_shortfalls={'doctor': 0, 'nurse': 1, 'pharmacist': 0},
            role_required={'doctor': 2, 'nurse': 6, 'pharmacist': 2},
            bed_occupancy_pct=50.0,
            patient_footfall_ratio=1.0
        )
        self.assertIn(res_normal['severity'], ['LOW', 'MEDIUM'])
        
        # High bed occupancy + footfall surge
        res_surge = StaffingEvaluator.calculate_shortage_severity(
            role_shortfalls={'doctor': 1, 'nurse': 2, 'pharmacist': 0},
            role_required={'doctor': 2, 'nurse': 6, 'pharmacist': 2},
            bed_occupancy_pct=90.0,
            patient_footfall_ratio=1.5
        )
        self.assertEqual(res_surge['severity'], 'CRITICAL')
        self.assertTrue(res_surge['doctor_crisis'])

    def test_facility_staffing_assessment(self):
        """Verify assessment evaluates all facilities."""
        df = self.service.run_assessment()
        self.assertEqual(len(df), len(self.service.cleaned_data['facilities']))
        
        expected_cols = [
            'phc_id', 'district', 'doctors_required', 'doctors_present', 'doctors_shortfall',
            'nurses_required', 'nurses_present', 'nurses_shortfall',
            'composite_score', 'severity', 'operationally_feasible', 'recommendation'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
        self.assertTrue(set(df['severity'].unique()).issubset({'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'HEALTHY'}))

    def test_shift_coverage_forecaster(self):
        """Verify 24h shift forecast returns Morning, Evening, Night projections."""
        shifts = self.service.get_shift_forecast("PHC_001")
        self.assertEqual(len(shifts), 3)
        
        shift_names = [s['shift'] for s in shifts]
        self.assertListEqual(shift_names, ["MORNING", "EVENING", "NIGHT"])
        
        for s in shifts:
            self.assertIn('roles', s)
            self.assertIn('doctor', s['roles'])
            self.assertIn('nurse', s['roles'])
            self.assertIn('pharmacist', s['roles'])
            self.assertGreater(s['roles']['doctor']['scheduled'], 0)

    def test_operational_feasibility_gatekeeper(self):
        """Verify gatekeeper function returns boolean feasibility for Module 4."""
        is_feasible = self.service.is_facility_operationally_feasible("PHC_001")
        self.assertIsInstance(is_feasible, bool)

    def test_facility_staffing_report(self):
        """Verify full facility report structure."""
        report = self.service.get_facility_staffing_report("PHC_001")
        self.assertEqual(report['phc_id'], "PHC_001")
        self.assertIn('staffing_breakdown', report)
        self.assertIn('upcoming_shifts_forecast', report)
        self.assertEqual(len(report['upcoming_shifts_forecast']), 3)

if __name__ == '__main__':
    unittest.main()
