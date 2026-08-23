import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class FacilityNetwork:
    """Graph and distance matrix query engine for the PHC facility network."""

    def __init__(self, facilities_df: pd.DataFrame, distance_df: pd.DataFrame):
        self.facilities_df = facilities_df.copy()
        self.distance_df = distance_df.copy()
        
        # Build quick lookup table: (from_id, to_id) -> (distance_km, travel_time_hours)
        self.routes: Dict[Tuple[str, str], Dict[str, float]] = {}
        for _, row in distance_df.iterrows():
            key = (row['phc_id_from'], row['phc_id_to'])
            self.routes[key] = {
                "distance_km": float(row['distance_km']),
                "travel_time_hours": float(row['travel_time_hours'])
            }
            
        self.facility_map: Dict[str, Dict[str, Any]] = {}
        for _, fac in facilities_df.iterrows():
            self.facility_map[fac['phc_id']] = {
                "phc_id": fac['phc_id'],
                "name": fac['phc_name'],
                "district": fac['district'],
                "latitude": float(fac['latitude']),
                "longitude": float(fac['longitude']),
                "total_beds": int(fac['total_beds']),
                "population_served": int(fac['population_served'])
            }

    def get_route(self, from_phc: str, to_phc: str) -> Dict[str, float]:
        """Returns distance and travel time between two facilities."""
        if from_phc == to_phc:
            return {"distance_km": 0.0, "travel_time_hours": 0.0}
            
        key = (from_phc, to_phc)
        if key in self.routes:
            return self.routes[key]
            
        # Reverse lookup fallback
        rev_key = (to_phc, from_phc)
        if rev_key in self.routes:
            return self.routes[rev_key]
            
        return {"distance_km": 999.0, "travel_time_hours": 99.0}

    def get_sorted_neighbors(
        self,
        phc_id: str,
        candidate_ids: Optional[List[str]] = None,
        max_travel_hours: float = 12.0
    ) -> List[Dict[str, Any]]:
        """
        Returns nearest neighbor facilities sorted by travel time.
        """
        all_candidates = candidate_ids if candidate_ids is not None else list(self.facility_map.keys())
        neighbors = []
        
        origin_district = self.facility_map.get(phc_id, {}).get("district", "")
        
        for cand in all_candidates:
            if cand == phc_id:
                continue
            route = self.get_route(phc_id, cand)
            if route['travel_time_hours'] <= max_travel_hours:
                cand_info = self.facility_map.get(cand, {})
                neighbors.append({
                    "phc_id": cand,
                    "district": cand_info.get("district", ""),
                    "is_cross_district": (cand_info.get("district") != origin_district),
                    "distance_km": route['distance_km'],
                    "travel_time_hours": route['travel_time_hours']
                })
                
        # Sort by travel time ascending
        neighbors.sort(key=lambda x: x['travel_time_hours'])
        return neighbors
