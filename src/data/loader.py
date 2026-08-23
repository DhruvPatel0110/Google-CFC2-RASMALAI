import pandas as pd
from typing import Dict
from src import config

class DataLoader:
    """Loads all raw datasets for PHC supply chain resilience."""
    
    @staticmethod
    def load_facilities() -> pd.DataFrame:
        df = pd.read_csv(config.FACILITIES_FILE)
        return df

    @staticmethod
    def load_distance_matrix() -> pd.DataFrame:
        df = pd.read_csv(config.DISTANCE_MATRIX_FILE)
        return df

    @staticmethod
    def load_medicines() -> pd.DataFrame:
        df = pd.read_csv(config.MEDICINES_FILE)
        return df

    @staticmethod
    def load_external_signals() -> pd.DataFrame:
        df = pd.read_csv(config.EXTERNAL_SIGNALS_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df

    @staticmethod
    def load_medicine_transactions() -> pd.DataFrame:
        df = pd.read_csv(config.MEDICINE_TRANSACTIONS_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df

    @staticmethod
    def load_inventory_snapshots() -> pd.DataFrame:
        df = pd.read_csv(config.INVENTORY_SNAPSHOTS_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df

    @staticmethod
    def load_admissions_discharges() -> pd.DataFrame:
        df = pd.read_csv(config.ADMISSIONS_DISCHARGES_FILE)
        df['admission_date'] = pd.to_datetime(df['admission_date'])
        df['discharge_date'] = pd.to_datetime(df['discharge_date'])
        return df

    @staticmethod
    def load_bed_occupancy() -> pd.DataFrame:
        df = pd.read_csv(config.BED_OCCUPANCY_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df

    @staticmethod
    def load_staff_roster() -> pd.DataFrame:
        df = pd.read_csv(config.STAFF_ROSTER_FILE)
        return df

    @staticmethod
    def load_staff_attendance() -> pd.DataFrame:
        df = pd.read_csv(config.STAFF_ATTENDANCE_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df

    @staticmethod
    def load_staffing_summary() -> pd.DataFrame:
        df = pd.read_csv(config.STAFFING_SUMMARY_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df

    @classmethod
    def load_all(cls) -> Dict[str, pd.DataFrame]:
        return {
            "facilities": cls.load_facilities(),
            "distance_matrix": cls.load_distance_matrix(),
            "medicines": cls.load_medicines(),
            "external_signals": cls.load_external_signals(),
            "medicine_transactions": cls.load_medicine_transactions(),
            "inventory_snapshots": cls.load_inventory_snapshots(),
            "admissions_discharges": cls.load_admissions_discharges(),
            "bed_occupancy": cls.load_bed_occupancy(),
            "staff_roster": cls.load_staff_roster(),
            "staff_attendance": cls.load_staff_attendance(),
            "staffing_summary": cls.load_staffing_summary()
        }
