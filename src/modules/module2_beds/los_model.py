import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class LengthOfStayModel:
    """Models patient length-of-stay profiles and filters eligible transfer candidates."""

    def __init__(self, admissions_df: Optional[pd.DataFrame] = None):
        self.los_stats: Dict[str, Dict[str, float]] = {}
        if admissions_df is not None:
            self.fit(admissions_df)

    def fit(self, admissions_df: pd.DataFrame):
        """Computes diagnosis and severity length-of-stay benchmarks."""
        stats = admissions_df.groupby(['diagnosis_category', 'severity'])['los_days'].agg(
            median_los='median',
            mean_los='mean',
            std_los='std',
            count='count'
        ).reset_index()
        
        self.los_stats = {}
        for _, row in stats.iterrows():
            key = f"{row['diagnosis_category']}_{row['severity']}"
            self.los_stats[key] = {
                "median_los": float(row['median_los']),
                "mean_los": float(row['mean_los']),
                "std_los": float(row['std_los']) if not pd.isna(row['std_los']) else 1.0,
                "count": int(row['count'])
            }

    def get_expected_los(self, diagnosis: str, severity: str) -> float:
        """Returns expected stay duration in days."""
        key = f"{diagnosis}_{severity}"
        if key in self.los_stats:
            return self.los_stats[key]['median_los']
        return 3.0  # Default fallback

    @staticmethod
    def get_active_patients(admissions_df: pd.DataFrame, current_date: pd.Timestamp) -> pd.DataFrame:
        """Filters patients currently admitted at facilities on current_date."""
        df = admissions_df.copy()
        mask = (df['admission_date'] <= current_date) & (df['discharge_date'] > current_date)
        active = df[mask].copy()
        active['days_admitted'] = (current_date - active['admission_date']).dt.days
        active['remaining_los'] = (active['discharge_date'] - current_date).dt.days
        return active

    @classmethod
    def get_transfer_candidates(
        cls,
        admissions_df: pd.DataFrame,
        phc_id: str,
        current_date: pd.Timestamp
    ) -> List[Dict[str, Any]]:
        """
        Identifies stable patients eligible for ambulance transfer to neighbor facilities.
        Excludes acute/critical patients and those admitted within the last 24-48 hours.
        """
        active = cls.get_active_patients(admissions_df, current_date)
        phc_patients = active[active['phc_id'] == phc_id]
        
        # Eligibility criteria:
        # 1. Non-critical severity (mild or moderate)
        # 2. Admitted for at least 2 days (past initial acute phase)
        # 3. Remaining length of stay at least 1 day
        # 4. Exclude complex conditions if unstable
        eligible = phc_patients[
            (phc_patients['severity'].isin(['mild', 'moderate'])) &
            (phc_patients['days_admitted'] >= 2) &
            (phc_patients['remaining_los'] >= 1)
        ].copy()
        
        candidates = []
        for _, p in eligible.iterrows():
            candidates.append({
                "patient_id": p['patient_id'],
                "phc_id": p['phc_id'],
                "age": int(p['age']),
                "gender": p['gender'],
                "diagnosis": p['diagnosis_category'],
                "severity": p['severity'],
                "days_admitted": int(p['days_admitted']),
                "remaining_los_days": int(p['remaining_los']),
                "transfer_eligibility": "SAFE_FOR_TRANSIT",
                "transfer_priority": "MEDIUM" if p['severity'] == 'moderate' else "STANDARD"
            })
            
        return candidates
