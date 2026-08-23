import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class ShiftCoverageForecaster:
    """Predicts next 24-48 hour shift attendance and coverage shortfalls from staff reliability profiles."""

    @staticmethod
    def forecast_shift_coverage(
        roster_df: pd.DataFrame,
        phc_id: str,
        forecast_date: pd.Timestamp,
        outbreak_active: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Forecasts staffing headcount and absence probabilities for upcoming shifts (Morning, Evening, Night).
        """
        phc_roster = roster_df[roster_df['phc_id'] == phc_id]
        if len(phc_roster) == 0:
            return []
            
        is_sunday = forecast_date.weekday() == 6
        outbreak_mult = 1.4 if outbreak_active else 1.0
        sunday_mult = 1.3 if is_sunday else 1.0
        
        shifts = [
            {"shift": "MORNING", "prob_share": 0.45, "hours": "08:00 - 14:00"},
            {"shift": "EVENING", "prob_share": 0.35, "hours": "14:00 - 20:00"},
            {"shift": "NIGHT", "prob_share": 0.20, "hours": "20:00 - 08:00"}
        ]
        
        forecasts = []
        for s in shifts:
            shift_name = s['shift']
            shift_results = {}
            
            for role in ['doctor', 'nurse', 'pharmacist']:
                role_staff = phc_roster[phc_roster['role'] == role]
                total_staff = len(role_staff)
                
                # Expected scheduled staff for this shift
                scheduled_expected = max(1, round(total_staff * s['prob_share']))
                
                # Expected present headcount based on reliability scores
                expected_present = 0.0
                for _, member in role_staff.iterrows():
                    p_absence = (1.0 - member['reliability_score']) * outbreak_mult * sunday_mult
                    p_present = max(0.05, 1.0 - min(0.95, p_absence))
                    expected_present += p_present * s['prob_share']
                    
                expected_present_count = round(expected_present, 1)
                shortfall_expected = max(0.0, scheduled_expected - expected_present_count)
                
                shift_results[role] = {
                    "scheduled": scheduled_expected,
                    "expected_present": expected_present_count,
                    "expected_shortfall": round(shortfall_expected, 1),
                    "coverage_ratio": round(expected_present_count / max(0.1, scheduled_expected), 2)
                }
                
            # Determine overall shift alert risk
            shift_alert = (
                shift_results['doctor']['expected_shortfall'] >= 1.0 or
                shift_results['nurse']['coverage_ratio'] < 0.70
            )
            
            forecasts.append({
                "phc_id": phc_id,
                "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                "shift": shift_name,
                "shift_hours": s['hours'],
                "roles": shift_results,
                "alert_expected": bool(shift_alert),
                "confidence_score": 0.85
            })
            
        return forecasts
