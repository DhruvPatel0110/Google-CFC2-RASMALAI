import pandas as pd
import numpy as np
from typing import Dict, Tuple

class FeatureStore:
    """Builds model-ready feature matrices with temporal lags, rolling windows, and cross-module signals."""

    @staticmethod
    def build_medicine_features(
        cleaned_data: Dict[str, pd.DataFrame],
        forecast_horizon: int = 7
    ) -> pd.DataFrame:
        """
        Constructs time-series features for Module 1 (Medicine Demand Forecasting).
        """
        tx_df = cleaned_data['medicine_transactions'].copy()
        inv_df = cleaned_data['inventory_snapshots'].copy()
        fac_df = cleaned_data['facilities'].copy()
        med_df = cleaned_data['medicines'].copy()
        sig_df = cleaned_data['external_signals'].copy()
        bed_df = cleaned_data['bed_occupancy'].copy()
        
        # Merge basic transaction and inventory
        df = pd.merge(
            tx_df,
            inv_df[['date', 'phc_id', 'drug_id', 'closing_stock', 'reorder_point', 'safety_stock']],
            on=['date', 'phc_id', 'drug_id'],
            how='left'
        )
        
        # Merge facility metadata
        df = pd.merge(
            df,
            fac_df[['phc_id', 'district', 'population_served', 'total_beds', 'warehouse_lead_time_days', 'distance_to_district_warehouse_km']],
            on='phc_id',
            how='left'
        )
        
        # Merge medicine metadata
        df = pd.merge(
            df,
            med_df[['drug_id', 'category', 'shelf_life_days', 'unit_cost_inr']],
            on='drug_id',
            how='left'
        )
        
        # Merge external signals by date and district
        df = pd.merge(
            df,
            sig_df[['date', 'district', 'rainfall_mm', 'temperature_c', 'outbreak_active', 'outbreak_severity', 'campaign_active']],
            on=['date', 'district'],
            how='left'
        )
        
        # Merge bed occupancy signal
        df = pd.merge(
            df,
            bed_df[['date', 'phc_id', 'occupancy_pct', 'occupied_beds']],
            on=['date', 'phc_id'],
            how='left'
        )
        
        # Sort values
        df = df.sort_values(by=['phc_id', 'drug_id', 'date']).reset_index(drop=True)
        
        # 1. Calendar Features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # 2. Time-Series Lag Features (dispensed quantity)
        grouped = df.groupby(['phc_id', 'drug_id'])['qty_dispensed']
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            df[f'lag_qty_{lag}d'] = grouped.shift(lag)
            
        # 3. Rolling Statistics
        df['rolling_mean_7d'] = grouped.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        df['rolling_std_7d'] = grouped.transform(lambda x: x.shift(1).rolling(7, min_periods=1).std()).fillna(0)
        df['rolling_mean_14d'] = grouped.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
        df['rolling_mean_30d'] = grouped.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
        
        # 4. Trend Slope (7-day delta)
        df['consumption_trend_7d'] = (df['lag_qty_1d'] - df['lag_qty_7d']).fillna(0)
        
        # 5. Future Target for Multi-Day Forecast (next 7 days total consumption & individual days)
        for h in range(1, forecast_horizon + 1):
            df[f'target_day_{h}'] = grouped.shift(-h)
            
        # Total 7-day forward demand target
        target_cols = [f'target_day_{h}' for h in range(1, forecast_horizon + 1)]
        df['target_7d_total'] = df[target_cols].sum(axis=1)
        
        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=['category', 'district'], drop_first=False)
        
        # Drop initial rows with unpopulated lags (first 30 days) and end rows missing target
        df_clean = df.dropna(subset=['lag_qty_30d'] + target_cols).reset_index(drop=True)
        
        return df_clean

    @staticmethod
    def build_bed_features(
        cleaned_data: Dict[str, pd.DataFrame],
        forecast_horizon: int = 7
    ) -> pd.DataFrame:
        """
        Constructs time-series features for Module 2 (Bed Occupancy Forecasting).
        """
        bed_df = cleaned_data['bed_occupancy'].copy()
        fac_df = cleaned_data['facilities'].copy()
        sig_df = cleaned_data['external_signals'].copy()
        
        df = pd.merge(
            bed_df,
            fac_df[['phc_id', 'district', 'population_served', 'warehouse_lead_time_days']],
            on='phc_id',
            how='left'
        )
        
        df = pd.merge(
            df,
            sig_df[['date', 'district', 'rainfall_mm', 'temperature_c', 'outbreak_active', 'outbreak_severity', 'campaign_active']],
            on=['date', 'district'],
            how='left'
        )
        
        df = df.sort_values(by=['phc_id', 'date']).reset_index(drop=True)
        
        # Calendar Features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lags on admissions and occupancy
        adm_grouped = df.groupby('phc_id')['admissions']
        occ_grouped = df.groupby('phc_id')['occupied_beds']
        
        for lag in [1, 2, 3, 7, 14]:
            df[f'lag_adm_{lag}d'] = adm_grouped.shift(lag)
            df[f'lag_occ_{lag}d'] = occ_grouped.shift(lag)
            
        # Rolling stats
        df['rolling_adm_mean_7d'] = adm_grouped.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        df['rolling_occ_mean_7d'] = occ_grouped.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        
        # Future targets for 7-day occupancy projection
        for h in range(1, forecast_horizon + 1):
            df[f'target_occ_day_{h}'] = occ_grouped.shift(-h)
            df[f'target_adm_day_{h}'] = adm_grouped.shift(-h)
            
        target_cols = [f'target_occ_day_{h}' for h in range(1, forecast_horizon + 1)]
        df = pd.get_dummies(df, columns=['district'], drop_first=False)
        df_clean = df.dropna(subset=['lag_adm_14d'] + target_cols).reset_index(drop=True)
        
        return df_clean
