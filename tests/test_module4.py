import unittest
import pandas as pd
from src.data.loader import DataLoader
from src.modules.module4_redistribution.network import FacilityNetwork
from src.modules.module4_redistribution.medicine_matcher import MedicineRedistributionMatcher
from src.modules.module4_redistribution.patient_matcher import PatientRedistributionMatcher
from src.modules.module4_redistribution.service import RedistributionService

class TestModule4Redistribution(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.fac_df = DataLoader.load_facilities()
        cls.dist_df = DataLoader.load_distance_matrix()
        cls.network = FacilityNetwork(cls.fac_df, cls.dist_df)
        cls.service = RedistributionService()
        cls.service.initialize()

    def test_facility_network_routing(self):
        """Verify distance and travel time graph lookups."""
        route = self.network.get_route("PHC_001", "PHC_002")
        self.assertGreater(route['distance_km'], 0.0)
        self.assertGreater(route['travel_time_hours'], 0.0)
        
        # Test sorted neighbors within reachable range
        neighbors = self.network.get_sorted_neighbors("PHC_001")
        self.assertGreater(len(neighbors), 0)
        # Check ascending sort order
        travel_times = [n['travel_time_hours'] for n in neighbors]
        self.assertEqual(travel_times, sorted(travel_times))
        
        # Test full network neighbors with unlimited horizon
        all_neighbors = self.network.get_sorted_neighbors("PHC_001", max_travel_hours=999.0)
        self.assertEqual(len(all_neighbors), len(self.fac_df) - 1)

    def test_medicine_matcher_logic(self):
        """Verify nearest neighbor medicine matching and dynamic donor decrementing."""
        matcher = MedicineRedistributionMatcher(self.network)
        
        deficits = [
            {
                "phc_id": "PHC_001", "district": "Vellore", "drug_id": "ORS_SACHET",
                "category": "ORS", "shortfall_qty": 150, "stockout_probability": 0.95
            }
        ]
        
        surpluses = [
            {"phc_id": "PHC_002", "district": "Krishnagiri", "drug_id": "ORS_SACHET", "surplus_qty": 100},
            {"phc_id": "PHC_003", "district": "Tiruvannamalai", "drug_id": "ORS_SACHET", "surplus_qty": 200}
        ]
        
        transfers = matcher.match_transfers(deficits, surpluses)
        self.assertGreater(len(transfers), 0)
        
        # Verify total transferred equals deficit or total available
        total_transferred = sum(t['quantity'] for t in transfers)
        self.assertEqual(total_transferred, 150)
        
        # Verify destination is PHC_001
        for t in transfers:
            self.assertEqual(t['destination_phc'], "PHC_001")
            self.assertEqual(t['drug_id'], "ORS_SACHET")
            self.assertGreater(t['quantity'], 0)

    def test_patient_matcher_staffing_gatekeeper(self):
        """Verify patient matcher strictly avoids facilities with staffing crisis."""
        matcher = PatientRedistributionMatcher(self.network)
        
        overcrowded = [
            {"phc_id": "PHC_001", "district": "Vellore", "peak_projected_occupancy_pct": 92.0, "capacity_deficit": 3}
        ]
        
        surpluses = [
            {"phc_id": "PHC_002", "district": "Krishnagiri", "surplus_beds_available": 5},
            {"phc_id": "PHC_003", "district": "Tiruvannamalai", "surplus_beds_available": 5}
        ]
        
        patient_candidates = {
            "PHC_001": [
                {"patient_id": "P_101", "severity": "mild", "days_admitted": 3, "remaining_los_days": 2},
                {"patient_id": "P_102", "severity": "moderate", "days_admitted": 4, "remaining_los_days": 1}
            ]
        }
        
        # Mock staffing gatekeeper: PHC_002 (nearest) has doctor deficit crisis -> False, PHC_003 -> True
        def mock_staff_check(phc_id: str) -> bool:
            return phc_id != "PHC_002"
            
        transfers = matcher.match_patient_transfers(
            overcrowded_facilities=overcrowded,
            capacity_surplus_facilities=surpluses,
            patient_candidates_by_facility=patient_candidates,
            staff_feasibility_checker=mock_staff_check
        )
        
        self.assertGreater(len(transfers), 0)
        # Verify no patients were sent to unstaffed PHC_002
        for t in transfers:
            self.assertNotEqual(t['destination_phc'], "PHC_002")
            self.assertEqual(t['destination_phc'], "PHC_003")

    def test_end_to_end_redistribution_service(self):
        """Verify complete multi-module redistribution service execution."""
        plan = self.service.generate_plan("2025-08-31")
        self.assertIsInstance(plan, dict)
        self.assertIn("total_recommendations", plan)
        self.assertIn("impact_metrics", plan)
        self.assertIn("medicine_transfers", plan)
        self.assertIn("patient_transfers", plan)
        
        # Verify impact metrics
        impact = plan['impact_metrics']
        self.assertIn("stockouts_prevented", impact)
        self.assertIn("medicine_units_redistributed", impact)
        self.assertIn("average_transit_time_hours", impact)

if __name__ == '__main__':
    unittest.main()
