import pandas as pd
import numpy as np
from typing import Dict

class DataPreprocessor:
    """Cleans, normalizes, and validates the raw health and supply chain datasets."""

    @staticmethod
    def clean_medicine_transactions(
        transactions_df: pd.DataFrame, 
        inventory_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        df = transactions_df.copy()
        
        # Sort values
        df = df.sort_values(by=['phc_id', 'drug_id', 'date']).reset_index(drop=True)
        
        # If inventory_snapshots is provided, fill missing qty_dispensed from inventory dispensed_qty
        if inventory_df is not None:
            merged = pd.merge(
                df,
                inventory_df[['date', 'phc_id', 'drug_id', 'dispensed_qty']],
                on=['date', 'phc_id', 'drug_id'],
                how='left'
            )
            df['qty_dispensed'] = df['qty_dispensed'].combine_first(merged['dispensed_qty'])
            
        # Group-wise rolling median interpolation for remaining nulls
        df['qty_dispensed'] = df.groupby(['phc_id', 'drug_id'])['qty_dispensed'].transform(
            lambda group: group.ffill().bfill().fillna(0)
        )
        
        # Clip negative dispensed quantities (e.g. data entry error corrections) to 0
        df['qty_dispensed'] = df['qty_dispensed'].clip(lower=0)
        df['patients_served_estimate'] = df['patients_served_estimate'].clip(lower=0)
        
        return df

    @staticmethod
    def clean_inventory_snapshots(inventory_df: pd.DataFrame) -> pd.DataFrame:
        df = inventory_df.copy()
        df = df.sort_values(by=['phc_id', 'drug_id', 'date']).reset_index(drop=True)
        
        # Ensure non-negative stocks
        df['opening_stock'] = df['opening_stock'].clip(lower=0)
        df['received_qty'] = df['received_qty'].clip(lower=0)
        df['dispensed_qty'] = df['dispensed_qty'].clip(lower=0)
        df['closing_stock'] = df['closing_stock'].clip(lower=0)
        
        return df

    @staticmethod
    def clean_bed_occupancy(bed_df: pd.DataFrame) -> pd.DataFrame:
        df = bed_df.copy()
        df = df.sort_values(by=['phc_id', 'date']).reset_index(drop=True)
        
        df['total_beds'] = df['total_beds'].clip(lower=1)
        df['occupied_beds'] = df['occupied_beds'].clip(lower=0)
        # Cap occupied beds at total capacity
        df['occupied_beds'] = np.minimum(df['occupied_beds'], df['total_beds'])
        
        # Recompute derived metrics
        df['available_beds'] = (df['total_beds'] - df['occupied_beds']).clip(lower=0)
        df['occupancy_pct'] = (df['occupied_beds'] / df['total_beds'] * 100.0).round(2)
        df['admissions'] = df['admissions'].clip(lower=0)
        df['discharges'] = df['discharges'].clip(lower=0)
        
        return df

    @staticmethod
    def clean_external_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
        df = signals_df.copy()
        df['outbreak_active'] = df['outbreak_active'].astype(int)
        df['campaign_active'] = df['campaign_active'].astype(int)
        df['rainfall_mm'] = df['rainfall_mm'].clip(lower=0.0)
        df['outbreak_severity'] = df['outbreak_severity'].clip(lower=0.0, upper=1.0)
        
        return df

    @classmethod
    def clean_all(cls, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        cleaned = dict(raw_data)
        
        cleaned['inventory_snapshots'] = cls.clean_inventory_snapshots(raw_data['inventory_snapshots'])
        cleaned['medicine_transactions'] = cls.clean_medicine_transactions(
            raw_data['medicine_transactions'], 
            cleaned['inventory_snapshots']
        )
        cleaned['bed_occupancy'] = cls.clean_bed_occupancy(raw_data['bed_occupancy'])
        cleaned['external_signals'] = cls.clean_external_signals(raw_data['external_signals'])
        
        return cleaned
