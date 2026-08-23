import uuid
import pandas as pd
from typing import Dict, List, Any
from .network import FacilityNetwork

class MedicineRedistributionMatcher:
    """Nearest-Neighbor matching engine for cross-facility medicine transfers."""

    def __init__(self, network: FacilityNetwork):
        self.network = network

    def match_transfers(
        self,
        critical_deficits: List[Dict[str, Any]],
        surplus_donors: List[Dict[str, Any]],
        max_travel_hours: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Matches critical drug deficits with nearest surplus facilities, updating donor balances in real time.
        """
        # Build dynamic donor surplus pool: (phc_id, drug_id) -> remaining_surplus
        donor_pool: Dict[tuple, int] = {}
        for donor in surplus_donors:
            key = (donor['phc_id'], donor['drug_id'])
            donor_pool[key] = int(donor['surplus_qty'])
            
        recommendations = []
        rec_counter = 1
        
        # Sort deficits by stockout urgency (highest stockout probability / shortest days to stockout first)
        sorted_deficits = sorted(
            critical_deficits,
            key=lambda x: (x.get('stockout_probability', 1.0), -x.get('shortfall_qty', 0)),
            reverse=True
        )
        
        for deficit in sorted_deficits:
            dest_phc = deficit['phc_id']
            drug_id = deficit['drug_id']
            shortfall = int(deficit['shortfall_qty'])
            
            if shortfall <= 0:
                continue
                
            # Find all donor facilities with available surplus for this specific drug
            candidate_donors = [
                donor_phc for (donor_phc, d_id), avail_qty in donor_pool.items()
                if d_id == drug_id and avail_qty > 0 and donor_phc != dest_phc
            ]
            
            if not candidate_donors:
                continue
                
            # Get candidates sorted by shortest road travel time
            sorted_candidates = self.network.get_sorted_neighbors(
                dest_phc,
                candidate_ids=candidate_donors,
                max_travel_hours=max_travel_hours
            )
            
            for candidate in sorted_candidates:
                if shortfall <= 0:
                    break
                    
                src_phc = candidate['phc_id']
                donor_key = (src_phc, drug_id)
                avail_stock = donor_pool.get(donor_key, 0)
                
                if avail_stock <= 0:
                    continue
                    
                # Allocate transfer
                transfer_qty = min(shortfall, avail_stock)
                donor_pool[donor_key] -= transfer_qty
                shortfall -= transfer_qty
                
                rec_id = f"REDIS_MED_{rec_counter:04d}"
                rec_counter += 1
                
                is_cross = candidate['is_cross_district']
                
                recommendations.append({
                    "recommendation_id": rec_id,
                    "action": "TRANSFER_MEDICINE",
                    "source_phc": src_phc,
                    "destination_phc": dest_phc,
                    "drug_id": drug_id,
                    "category": deficit.get('category', 'General'),
                    "quantity": int(transfer_qty),
                    "distance_km": float(candidate['distance_km']),
                    "travel_time_hours": float(candidate['travel_time_hours']),
                    "is_cross_district": is_cross,
                    "urgency": "CRITICAL",
                    "approval_required": is_cross,
                    "approval_role": "District Chief Medical Officer" if is_cross else "PHC Medical Officer",
                    "confidence_score": 0.95,
                    "rationale": f"Transfer {transfer_qty} units of {drug_id} from {src_phc} (Surplus donor) to {dest_phc} to prevent stockout (ETA: {candidate['travel_time_hours']}h)."
                })
                
        return recommendations
