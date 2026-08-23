import pandas as pd
import numpy as np
from typing import Dict, List, Any
from src import config

class BedOccupancyEngine:
    """Evaluates 7-day forward bed occupancy curves, detects overflow, and computes capacity buffers."""

    @classmethod
    def evaluate_bed_risks(
        cls,
        forecast_df: pd.DataFrame,
        bed_df: pd.DataFrame,
        facilities_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes multi-step predictions and computes capacity metrics and alert classifications.
        """
        # Get latest bed state
        latest_bed = bed_df.sort_values(by='date').groupby('phc_id').last().reset_index()
        
        # Merge with facilities
        latest_bed = pd.merge(
            latest_bed,
            facilities_df[['phc_id', 'district', 'population_served']],
            on='phc_id',
            how='left'
        )
        
        # Merge with forecasts
        eval_df = pd.merge(
            latest_bed,
            forecast_df,
            on=['phc_id', 'total_beds'],
            how='inner',
            suffixes=('', '_pred')
        )
        
        # Compute daily trajectories
        for i in range(1, 8):
            occ_col = f'target_occ_day_{i}'
            eval_df[f'proj_occ_day_{i}'] = eval_df[occ_col].round(0).astype(int)
            eval_df[f'proj_avail_day_{i}'] = np.maximum(0, eval_df['total_beds'] - eval_df[f'proj_occ_day_{i}']).astype(int)
            eval_df[f'proj_pct_day_{i}'] = (eval_df[f'proj_occ_day_{i}'] / eval_df['total_beds'] * 100.0).round(1)
            
        proj_pct_cols = [f'proj_pct_day_{i}' for i in range(1, 8)]
        proj_occ_cols = [f'proj_occ_day_{i}' for i in range(1, 8)]
        
        # Calculate Peak Occupancy Metrics
        eval_df['peak_occupancy_pct'] = eval_df[proj_pct_cols].max(axis=1).round(1)
        eval_df['peak_occupied_beds'] = eval_df[proj_occ_cols].max(axis=1).astype(int)
        eval_df['avg_projected_occupancy_pct'] = eval_df[proj_pct_cols].mean(axis=1).round(1)
        
        # Days until capacity breach (>85%)
        def find_breach_day(row):
            for day in range(1, 8):
                if row[f'proj_pct_day_{day}'] >= config.BED_OCCUPANCY_CRITICAL_PCT:
                    return day
            return None
            
        eval_df['breach_day'] = eval_df.apply(find_breach_day, axis=1)
        
        # Determine Alert Level
        conditions = [
            (eval_df['peak_occupancy_pct'] >= config.BED_OCCUPANCY_CRITICAL_PCT) | (eval_df['occupancy_pct'] >= config.BED_OCCUPANCY_CRITICAL_PCT),
            (eval_df['peak_occupancy_pct'] >= config.BED_OCCUPANCY_WARNING_PCT) | (eval_df['occupancy_pct'] >= config.BED_OCCUPANCY_WARNING_PCT)
        ]
        choices = ['CRITICAL', 'WARNING']
        eval_df['alert_level'] = np.select(conditions, choices, default='SAFE')
        
        # Capacity Deficit / Surplus (Threshold: 75% target safe capacity)
        eval_df['capacity_deficit'] = np.maximum(
            0,
            eval_df['peak_occupied_beds'] - (eval_df['total_beds'] * 0.85).astype(int)
        )
        
        # Surplus beds safe for receiving transferred patients (under 65% utilization)
        eval_df['surplus_beds_available'] = np.maximum(
            0,
            (eval_df['total_beds'] * 0.65).astype(int) - eval_df['peak_occupied_beds']
        )
        
        # Actionable Recommendation Text
        def generate_recommendation(row):
            if row['alert_level'] == 'CRITICAL':
                breach = f"on Day {int(row['breach_day'])}" if pd.notna(row['breach_day']) else "immediately"
                return f"OVERFLOW IMMINENT: Projected {row['peak_occupancy_pct']}% occupancy {breach}. Divert {int(row['capacity_deficit'])} patients to nearby surplus PHCs."
            elif row['alert_level'] == 'WARNING':
                return f"ELEVATED OCCUPANCY: Projected {row['peak_occupancy_pct']}%. Monitor admissions and prepare early discharge protocols."
            else:
                return f"STABLE: Safe capacity available ({int(row['surplus_beds_available'])} beds available for incoming patient transfers)."
                
        eval_df['recommendation'] = eval_df.apply(generate_recommendation, axis=1)
        
        return eval_df
