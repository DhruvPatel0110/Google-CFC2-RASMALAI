import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Datasets"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Datasets Files
FACILITIES_FILE = DATASET_DIR / "facilities.csv"
DISTANCE_MATRIX_FILE = DATASET_DIR / "distance_matrix.csv"
MEDICINES_FILE = DATASET_DIR / "medicines.csv"
EXTERNAL_SIGNALS_FILE = DATASET_DIR / "external_signals.csv"
MEDICINE_TRANSACTIONS_FILE = DATASET_DIR / "medicine_transactions.csv"
INVENTORY_SNAPSHOTS_FILE = DATASET_DIR / "inventory_snapshots.csv"
ADMISSIONS_DISCHARGES_FILE = DATASET_DIR / "admissions_discharges.csv"
BED_OCCUPANCY_FILE = DATASET_DIR / "bed_occupancy_daily.csv"
STAFF_ROSTER_FILE = DATASET_DIR / "staff_roster.csv"
STAFF_ATTENDANCE_FILE = DATASET_DIR / "staff_attendance.csv"
STAFFING_SUMMARY_FILE = DATASET_DIR / "staffing_daily_summary.csv"

# Global Constants
START_DATE = "2024-09-01"
END_DATE = "2025-08-31"
N_PHCS = 15
DISTRICTS = ["Vellore", "Krishnagiri", "Tiruvannamalai"]

# Forecasting Horizons
FORECAST_HORIZON_DAYS = 7

# Alert Thresholds
STOCKOUT_CRITICAL_DAYS = 3.0
STOCKOUT_WARNING_DAYS = 7.0
BED_OCCUPANCY_CRITICAL_PCT = 85.0
BED_OCCUPANCY_WARNING_PCT = 75.0
STAFF_DEFICIT_CRITICAL_RATIO = 0.5
