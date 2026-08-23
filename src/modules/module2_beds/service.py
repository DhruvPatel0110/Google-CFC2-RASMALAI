import pandas as pd
from typing import Dict, List, Any, Optional
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.feature_store import FeatureStore
from .forecaster import BedOccupancyForecaster
from .los_model import LengthOfStayModel
from .occupancy_engine import BedOccupancyEngine

class BedModuleService:
    """End-to-End Service Facade for Module 2 (Bed Availability & Occupancy Forecasting)."""

    def __init__(self, forecaster: Optional[BedOccupancyForecaster] = None):
        self.forecaster = forecaster or BedOccupancyForecaster()
        self.los_model: Optional[LengthOfStayModel] = None
        self.cleaned_data: Optional[Dict[str, pd.DataFrame]] = None
        self.feature_df: Optional[pd.DataFrame] = None

    def initialize_data(self):
        """Loads, cleans datasets, and initializes the LOS benchmark model."""
        if self.cleaned_data is None:
            raw = DataLoader.load_all()
            self.cleaned_data = DataPreprocessor.clean_all(raw)
            self.feature_df = FeatureStore.build_bed_features(self.cleaned_data)
            self.los_model = LengthOfStayModel(self.cleaned_data['admissions_discharges'])

    def train(self, test_days: int = 30) -> Dict[str, float]:
        """Trains the underlying forecaster and returns validation metrics."""
        self.initialize_data()
        metrics = self.forecaster.train(self.feature_df, test_days=test_days)
        return metrics

    def run_assessment(self, as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Runs model inference and bed capacity risk evaluation for a given snapshot date.
        """
        self.initialize_data()
        
        if as_of_date is not None:
            target_date = pd.to_datetime(as_of_date)
            snapshot_features = self.feature_df[self.feature_df['date'] == target_date]
            if len(snapshot_features) == 0:
                earlier = self.feature_df[self.feature_df['date'] <= target_date]
                if len(earlier) > 0:
                    latest_date = earlier['date'].max()
                    snapshot_features = self.feature_df[self.feature_df['date'] == latest_date]
                else:
                    snapshot_features = self.feature_df[self.feature_df['date'] == self.feature_df['date'].min()]
        else:
            max_date = self.feature_df['date'].max()
            snapshot_features = self.feature_df[self.feature_df['date'] == max_date]
            
        predictions = self.forecaster.predict(snapshot_features)
        
        risk_df = BedOccupancyEngine.evaluate_bed_risks(
            forecast_df=predictions,
            bed_df=self.cleaned_data['bed_occupancy'],
            facilities_df=self.cleaned_data['facilities']
        )
        
        return risk_df

    def get_overcrowded_facilities(self, as_of_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns facilities with imminent bed overflow (>85%) formatted for Module 4."""
        risk_df = self.run_assessment(as_of_date)
        critical = risk_df[risk_df['alert_level'] == 'CRITICAL']
        
        overloaded = []
        for _, row in critical.iterrows():
            overloaded.append({
                "phc_id": row['phc_id'],
                "district": row['district'],
                "total_beds": int(row['total_beds']),
                "current_occupied": int(row['occupied_beds']),
                "current_occupancy_pct": float(row['occupancy_pct']),
                "peak_projected_occupancy_pct": float(row['peak_occupancy_pct']),
                "capacity_deficit": int(row['capacity_deficit']),
                "breach_day": int(row['breach_day']) if pd.notna(row['breach_day']) else 1,
                "urgency": "CRITICAL"
            })
        return overloaded

    def get_capacity_surplus_facilities(self, as_of_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns facilities with available beds under safe capacity to receive patient transfers."""
        risk_df = self.run_assessment(as_of_date)
        surplus = risk_df[risk_df['surplus_beds_available'] > 0]
        
        receivers = []
        for _, row in surplus.iterrows():
            receivers.append({
                "phc_id": row['phc_id'],
                "district": row['district'],
                "total_beds": int(row['total_beds']),
                "current_occupied": int(row['occupied_beds']),
                "peak_projected_occupancy_pct": float(row['peak_occupancy_pct']),
                "surplus_beds_available": int(row['surplus_beds_available'])
            })
        return receivers

    def get_transferable_patients(self, phc_id: str, as_of_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns candidate recovering/stable patients from an overloaded facility."""
        self.initialize_data()
        eval_date = pd.to_datetime(as_of_date) if as_of_date else self.cleaned_data['admissions_discharges']['admission_date'].max()
        
        candidates = LengthOfStayModel.get_transfer_candidates(
            self.cleaned_data['admissions_discharges'],
            phc_id=phc_id,
            current_date=eval_date
        )
        return candidates

    def get_facility_bed_report(self, phc_id: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        """Returns complete 7-day occupancy projection curve and bed stats for a PHC."""
        risk_df = self.run_assessment(as_of_date)
        phc_data = risk_df[risk_df['phc_id'] == phc_id]
        
        if len(phc_data) == 0:
            return {"error": f"PHC {phc_id} not found."}
            
        row = phc_data.iloc[0]
        daily_curve = []
        for day in range(1, 8):
            daily_curve.append({
                "day": day,
                "projected_occupied": int(row[f'proj_occ_day_{day}']),
                "projected_available": int(row[f'proj_avail_day_{day}']),
                "projected_occupancy_pct": float(row[f'proj_pct_day_{day}'])
            })
            
        return {
            "phc_id": phc_id,
            "district": row['district'],
            "total_beds": int(row['total_beds']),
            "current_occupied": int(row['occupied_beds']),
            "current_occupancy_pct": float(row['occupancy_pct']),
            "peak_occupancy_pct": float(row['peak_occupancy_pct']),
            "alert_level": row['alert_level'],
            "recommendation": row['recommendation'],
            "daily_7d_forecast": daily_curve
        }
