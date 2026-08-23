import pandas as pd
from typing import Dict, List, Any, Callable
from .network import FacilityNetwork

class PatientRedistributionMatcher:
    """Patient load-balancing engine with clinical safety and staffing compliance gatekeeping."""

    def __init__(self, network: FacilityNetwork):
        self.network = network

    def match_patient_transfers(
        self,
        overcrowded_facilities: List[Dict[str, Any]],
        capacity_surplus_facilities: List[Dict[str, Any]],
        patient_candidates_by_facility: Dict[str, List[Dict[str, Any]]],
        staff_feasibility_checker: Callable[[str], bool],
        max_travel_hours: float = 6.0
    ) -> List[Dict[str, Any]]:
        """
        Diverts clinically stable patients from overloaded PHCs to nearest staffed facilities.
        """
        # Dynamic bed capacity pool: phc_id -> available_beds
        bed_pool: Dict[str, int] = {}
        for surplus_fac in capacity_surplus_facilities:
            bed_pool[surplus_fac['phc_id']] = int(surplus_fac['surplus_beds_available'])
            
        recommendations = []
        rec_counter = 1
        
        # Sort overloaded facilities by peak occupancy descending
        sorted_overloaded = sorted(
            overcrowded_facilities,
            key=lambda x: x.get('peak_projected_occupancy_pct', 100.0),
            reverse=True
        )
        
        for over_fac in sorted_overloaded:
            src_phc = over_fac['phc_id']
            transferable_patients = list(patient_candidates_by_facility.get(src_phc, []))
            
            if not transferable_patients:
                continue
                
            # Find candidate receivers with available beds
            candidate_receivers = [
                r_phc for r_phc, avail_beds in bed_pool.items()
                if avail_beds > 0 and r_phc != src_phc
            ]
            
            if not candidate_receivers:
                continue
                
            sorted_neighbors = self.network.get_sorted_neighbors(
                src_phc,
                candidate_ids=candidate_receivers,
                max_travel_hours=max_travel_hours
            )
            
            for neighbor in sorted_neighbors:
                if not transferable_patients:
                    break
                    
                target_phc = neighbor['phc_id']
                
                # Module 3 Staffing Feasibility Gatekeeper
                # If target facility is severely understaffed, do not divert patients there
                if not staff_feasibility_checker(target_phc):
                    continue
                    
                avail_beds = bed_pool.get(target_phc, 0)
                if avail_beds <= 0:
                    continue
                    
                # Allocate batch of patients
                batch_size = min(len(transferable_patients), avail_beds)
                assigned_patients = transferable_patients[:batch_size]
                
                # Update pools
                transferable_patients = transferable_patients[batch_size:]
                bed_pool[target_phc] -= batch_size
                
                rec_id = f"REDIS_PAT_{rec_counter:04d}"
                rec_counter += 1
                
                is_cross = neighbor['is_cross_district']
                
                recommendations.append({
                    "recommendation_id": rec_id,
                    "action": "TRANSFER_PATIENT",
                    "source_phc": src_phc,
                    "destination_phc": target_phc,
                    "patient_count": int(batch_size),
                    "patient_ids": [p['patient_id'] for p in assigned_patients],
                    "patient_details": assigned_patients,
                    "distance_km": float(neighbor['distance_km']),
                    "travel_time_hours": float(neighbor['travel_time_hours']),
                    "is_cross_district": is_cross,
                    "urgency": "HIGH",
                    "approval_required": True,
                    "approval_role": "District CMO / Ambulance Dispatch Authority",
                    "confidence_score": 0.90,
                    "rationale": f"Divert {batch_size} stable patients from {src_phc} (Peak Occupancy: {over_fac.get('peak_projected_occupancy_pct')}%) to {target_phc} (Verified Staffing & Capacity, ETA: {neighbor['travel_time_hours']}h)."
                })
                
        return recommendations
