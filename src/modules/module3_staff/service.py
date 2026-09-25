import pandas as pd
from typing import Dict, List, Any, Optional
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from .evaluator import StaffingEvaluator
from .shift_forecaster import ShiftCoverageForecaster

class StaffModuleService:
    """End-to-End Service Facade for Module 3 (Medical Personnel Attendance & Staffing Analysis)."""

    def __init__(self):
        self.cleaned_data: Optional[Dict[str, pd.DataFrame]] = None
        self._assessment_cache: Dict[str, pd.DataFrame] = {}

    def initialize_data(self):
        """Loads and pre-processes all staffing and demand datasets."""
        if self.cleaned_data is None:
            raw = DataLoader.load_all()
            self.cleaned_data = DataPreprocessor.clean_all(raw)

    def run_assessment(self, as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Runs comprehensive staffing evaluation across all facilities for given date.
        """
        self.initialize_data()
        eval_date = pd.to_datetime(as_of_date) if as_of_date else self.cleaned_data['staff_attendance']['date'].max()
        cache_key = str(eval_date)
        if cache_key in self._assessment_cache:
            return self._assessment_cache[cache_key]
        
        staff_eval_df = StaffingEvaluator.evaluate_facility_staffing(
            attendance_df=self.cleaned_data['staff_attendance'],
            facilities_df=self.cleaned_data['facilities'],
            current_date=eval_date,
            bed_df=self.cleaned_data['bed_occupancy'],
            tx_df=self.cleaned_data['medicine_transactions']
        )
        self._assessment_cache[cache_key] = staff_eval_df
        return staff_eval_df

    def is_facility_operationally_feasible(self, phc_id: str, as_of_date: Optional[str] = None) -> bool:
        """
        Gatekeeper for Module 4 (Cross-District Redistribution).
        Checks if facility has sufficient medical personnel to safely receive transferred patients.
        """
        df = self.run_assessment(as_of_date)
        match = df[df['phc_id'] == phc_id]
        if len(match) == 0:
            return False
        return bool(match['operationally_feasible'].iloc[0])

    def get_critical_staffing_alerts(self, as_of_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns facilities with CRITICAL or HIGH staff deficits."""
        df = self.run_assessment(as_of_date)
        alerts_df = df[df['severity'].isin(['CRITICAL', 'HIGH'])]
        
        alerts = []
        for _, row in alerts_df.iterrows():
            alerts.append({
                "phc_id": row['phc_id'],
                "district": row['district'],
                "severity": row['severity'],
                "doctors_shortfall": int(row['doctors_shortfall']),
                "nurses_shortfall": int(row['nurses_shortfall']),
                "pharmacists_shortfall": int(row['pharmacists_shortfall']),
                "bed_occupancy_pct": float(row['bed_occupancy_pct']),
                "composite_score": float(row['composite_score']),
                "recommendation": row['recommendation']
            })
        return alerts

    def get_shift_forecast(self, phc_id: str, forecast_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Provides next 24-hour shift coverage projections (Morning, Evening, Night)."""
        self.initialize_data()
        fdate = pd.to_datetime(forecast_date) if forecast_date else self.cleaned_data['staff_attendance']['date'].max() + pd.Timedelta(days=1)
        
        # Check if outbreak active for this PHC's district
        fac_match = self.cleaned_data['facilities'][self.cleaned_data['facilities']['phc_id'] == phc_id]
        district = fac_match['district'].iloc[0] if len(fac_match) > 0 else "Vellore"
        
        sig_df = self.cleaned_data['external_signals']
        sig_match = sig_df[(sig_df['date'] == fdate) & (sig_df['district'] == district)]
        outbreak_active = bool(sig_match['outbreak_active'].iloc[0]) if len(sig_match) > 0 else False
        
        forecasts = ShiftCoverageForecaster.forecast_shift_coverage(
            roster_df=self.cleaned_data['staff_roster'],
            phc_id=phc_id,
            forecast_date=fdate,
            outbreak_active=outbreak_active
        )
        return forecasts

    def get_facility_staffing_report(self, phc_id: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        """Provides complete staffing and compliance summary for a PHC."""
        df = self.run_assessment(as_of_date)
        match = df[df['phc_id'] == phc_id]
        if len(match) == 0:
            return {"error": f"Facility {phc_id} not found."}
            
        row = match.iloc[0]
        shift_forecast = self.get_shift_forecast(phc_id, as_of_date)
        
        return {
            "phc_id": phc_id,
            "district": row['district'],
            "date": row['date'],
            "status": row['severity'],
            "operationally_feasible_for_transfers": bool(row['operationally_feasible']),
            "staffing_breakdown": {
                "doctors": {
                    "required": int(row['doctors_required']),
                    "present": int(row['doctors_present']),
                    "shortfall": int(row['doctors_shortfall'])
                },
                "nurses": {
                    "required": int(row['nurses_required']),
                    "present": int(row['nurses_present']),
                    "shortfall": int(row['nurses_shortfall'])
                },
                "pharmacists": {
                    "required": int(row['pharmacists_required']),
                    "present": int(row['pharmacists_present']),
                    "shortfall": int(row['pharmacists_shortfall'])
                }
            },
            "recommendation": row['recommendation'],
            "upcoming_shifts_forecast": shift_forecast
        }
