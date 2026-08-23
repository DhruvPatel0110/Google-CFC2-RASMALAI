import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional
from .network import FacilityNetwork
from .medicine_matcher import MedicineRedistributionMatcher
from .patient_matcher import PatientRedistributionMatcher

class RedistributionOptimizer:
    """Master multi-resource optimization and dispatch orchestrator."""

    def __init__(self, network: FacilityNetwork):
        self.network = network
        self.medicine_matcher = MedicineRedistributionMatcher(network)
        self.patient_matcher = PatientRedistributionMatcher(network)

    def optimize_network_redistribution(
        self,
        medicine_deficits: List[Dict[str, Any]],
        medicine_surpluses: List[Dict[str, Any]],
        overcrowded_facilities: List[Dict[str, Any]],
        bed_surpluses: List[Dict[str, Any]],
        patient_candidates_by_facility: Dict[str, List[Dict[str, Any]]],
        staff_feasibility_checker: Callable[[str], bool],
        snapshot_date: str = "2025-08-31"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-resource matching and produces structured dispatch manifests.
        """
        # 1. Match Medicines
        med_transfers = self.medicine_matcher.match_transfers(
            critical_deficits=medicine_deficits,
            surplus_donors=medicine_surpluses
        )
        
        # 2. Match Patients
        pat_transfers = self.patient_matcher.match_patient_transfers(
            overcrowded_facilities=overcrowded_facilities,
            capacity_surplus_facilities=bed_surpluses,
            patient_candidates_by_facility=patient_candidates_by_facility,
            staff_feasibility_checker=staff_feasibility_checker
        )
        
        # 3. Compute High-Impact Summary
        total_med_units = sum(t['quantity'] for t in med_transfers)
        total_patients = sum(t['patient_count'] for t in pat_transfers)
        
        all_transfers = med_transfers + pat_transfers
        avg_travel_time = round(
            float(np.mean([t['travel_time_hours'] for t in all_transfers])) if all_transfers else 0.0,
            2
        )
        cross_district_count = sum(1 for t in all_transfers if t.get('is_cross_district', False))
        
        summary = {
            "plan_date": snapshot_date,
            "total_recommendations": len(all_transfers),
            "by_type": {
                "medicine_transfers": len(med_transfers),
                "patient_transfers": len(pat_transfers)
            },
            "impact_metrics": {
                "medicine_units_redistributed": total_med_units,
                "stockouts_prevented": len(med_transfers),
                "patients_safely_diverted": total_patients,
                "bed_overflow_crises_averted": len(pat_transfers),
                "cross_district_transfers": cross_district_count,
                "average_transit_time_hours": avg_travel_time
            },
            "medicine_transfers": med_transfers,
            "patient_transfers": pat_transfers,
            "pending_approvals": [t for t in all_transfers if t.get('approval_required', False)]
        }
        
        return summary
