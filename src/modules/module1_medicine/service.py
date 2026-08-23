import pandas as pd
from typing import Dict, List, Any, Optional
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.feature_store import FeatureStore
from .forecaster import MedicineDemandForecaster
from .stockout_engine import StockoutEngine

class MedicineModuleService:
    """End-to-End Service Facade for Module 1 (Medicine Demand & Stockout Engine)."""

    def __init__(self, forecaster: Optional[MedicineDemandForecaster] = None):
        self.forecaster = forecaster or MedicineDemandForecaster()
        self.cleaned_data: Optional[Dict[str, pd.DataFrame]] = None
        self.feature_df: Optional[pd.DataFrame] = None

    def initialize_data(self):
        """Loads and pre-processes the underlying datasets if not already cached."""
        if self.cleaned_data is None:
            raw = DataLoader.load_all()
            self.cleaned_data = DataPreprocessor.clean_all(raw)
            self.feature_df = FeatureStore.build_medicine_features(self.cleaned_data)

    def train(self, test_days: int = 30) -> Dict[str, float]:
        """Trains the underlying forecaster and returns validation metrics."""
        self.initialize_data()
        metrics = self.forecaster.train(self.feature_df, test_days=test_days)
        return metrics

    def run_assessment(self, as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Runs model inference and stockout evaluation for a given snapshot date.
        If date is None, uses the latest available date.
        """
        self.initialize_data()
        
        if as_of_date is not None:
            target_date = pd.to_datetime(as_of_date)
            snapshot_features = self.feature_df[self.feature_df['date'] == target_date]
            if len(snapshot_features) == 0:
                # Fallback to nearest earlier date
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
        
        # Evaluate inventory risk
        risk_df = StockoutEngine.evaluate_inventory_risk(
            forecast_df=predictions,
            inventory_df=self.cleaned_data['inventory_snapshots'],
            facilities_df=self.cleaned_data['facilities'],
            medicines_df=self.cleaned_data['medicines']
        )
        
        return risk_df

    def get_critical_stockouts(self, as_of_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns list of critical stockouts formatted specifically for Module 4 redistribution."""
        risk_df = self.run_assessment(as_of_date)
        critical = risk_df[risk_df['alert_level'] == 'CRITICAL']
        
        alerts = []
        for _, row in critical.iterrows():
            alerts.append({
                "phc_id": row['phc_id'],
                "district": row['district'],
                "drug_id": row['drug_id'],
                "category": row['category'],
                "current_stock": int(row['closing_stock']),
                "predicted_consumption_7d": float(row['predicted_consumption_7d']),
                "days_to_stockout": float(row['days_to_stockout']),
                "shortfall_qty": int(row['shortfall_qty']),
                "stockout_probability": float(row['stockout_probability']),
                "urgency": "CRITICAL"
            })
        return alerts

    def get_surplus_facilities(self, as_of_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns list of facilities with surplus inventory available for donor transfers in Module 4."""
        risk_df = self.run_assessment(as_of_date)
        surplus = risk_df[risk_df['surplus_qty'] > 0]
        
        donors = []
        for _, row in surplus.iterrows():
            donors.append({
                "phc_id": row['phc_id'],
                "district": row['district'],
                "drug_id": row['drug_id'],
                "current_stock": int(row['closing_stock']),
                "safety_stock": int(row['safety_stock']),
                "surplus_qty": int(row['surplus_qty'])
            })
        return donors

    def get_facility_summary(self, phc_id: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        """Provides a complete inventory, forecast, and risk report for a single PHC."""
        risk_df = self.run_assessment(as_of_date)
        phc_risk = risk_df[risk_df['phc_id'] == phc_id]
        
        critical_count = int((phc_risk['alert_level'] == 'CRITICAL').sum())
        warning_count = int((phc_risk['alert_level'] == 'WARNING').sum())
        safe_count = int((phc_risk['alert_level'] == 'SAFE').sum())
        
        return {
            "phc_id": phc_id,
            "district": phc_risk['district'].iloc[0] if len(phc_risk) > 0 else "Unknown",
            "total_skus": len(phc_risk),
            "critical_stockouts": critical_count,
            "warning_items": warning_count,
            "safe_items": safe_count,
            "overall_status": "CRITICAL" if critical_count > 0 else ("WARNING" if warning_count > 0 else "HEALTHY"),
            "items": phc_risk.to_dict(orient='records')
        }
