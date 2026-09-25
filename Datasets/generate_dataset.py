"""
PHC Supply Chain Resilience — National-Scale Synthetic Dataset Generator
Generates realistic, multi-state, correlated health and supply chain data across India:
  - 5 States across diverse agro-climatic zones:
      * Tamil Nadu (South: Peninsular / Northeast Monsoon)
      * Maharashtra (West: Deccan Plateau & Coastal Ghats)
      * Rajasthan (North: Arid Thar Desert / Extreme Summer Heat)
      * Odisha (East: Coastal Cyclone & Endemic Malaria Belt)
      * Assam (North-East: Brahmaputra Riverine Flood Plains)
  - 30 Districts (6 districts per state)
  - 90 PHC Facilities (3 PHCs per district)
  - 365 Days of daily temporal history (captures complete seasonal cycle)
  - Correlated cross-module dynamics:
      * Outbreak & heatwave spikes drive medicine consumption, bed surges, and staff loads
      * Regional climate modeling (monsoons, temperatures, disaster-induced surges)
      * Realistic Indian Primary Health Centre staffing, bed capacities, and rural drug formularies
  - Deliberate pipeline messiness (missing values, biometric dropout, stock recount adjustments)
"""

import os
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd

# Set deterministic seed
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

START_DATE = datetime(2024, 9, 1)
N_DAYS = 365
dates = [START_DATE + timedelta(days=i) for i in range(N_DAYS)]

# Default output directory to local Datasets directory
DATASETS_DIR = Path(__file__).resolve().parent
OUT = str(DATASETS_DIR)

# ---------------------------------------------------------------------------
# 1. STATES & DISTRICTS HIERARCHICAL TOPOLOGY
# ---------------------------------------------------------------------------
# 5 Geographically diverse states with 6 representative districts each
STATES_CONFIG = {
    "Tamil Nadu": {
        "climate_type": "northeast_monsoon",
        "districts": {
            "Vellore": (12.9165, 79.1325),
            "Krishnagiri": (12.5186, 78.2137),
            "Tiruvannamalai": (12.2253, 79.0747),
            "Salem": (11.6643, 78.1460),
            "Madurai": (9.9252, 78.1198),
            "Coimbatore": (11.0168, 76.9558),
        },
        "possible_outbreaks": ["dengue", "seasonal_flu", "gastroenteritis"],
        "temp_base": 30.0,
        "summer_temp": 38.5,
    },
    "Maharashtra": {
        "climate_type": "plateau_monsoon",
        "districts": {
            "Pune": (18.5204, 73.8567),
            "Nagpur": (21.1458, 79.0882),
            "Nashik": (19.9975, 73.7898),
            "Chhatrapati Sambhajinagar": (19.8762, 75.3433),
            "Thane": (19.2183, 72.9781),
            "Amravati": (20.9374, 77.7796),
        },
        "possible_outbreaks": ["seasonal_flu", "dengue", "gastroenteritis"],
        "temp_base": 28.5,
        "summer_temp": 42.0,  # Vidarbha extreme heat
    },
    "Rajasthan": {
        "climate_type": "arid_desert",
        "districts": {
            "Jaipur": (26.9124, 75.7873),
            "Jodhpur": (26.2389, 73.0243),
            "Bikaner": (28.0229, 73.3119),
            "Udaipur": (24.5854, 73.7125),
            "Kota": (25.2138, 75.8648),
            "Barmer": (25.7532, 71.4181),
        },
        "possible_outbreaks": ["heatstroke_dehydration", "malaria", "seasonal_flu"],
        "temp_base": 27.0,
        "summer_temp": 45.0,  # Extreme Thar desert heat
    },
    "Odisha": {
        "climate_type": "coastal_cyclone",
        "districts": {
            "Khordha": (20.1809, 85.6212),
            "Cuttack": (20.4625, 85.8828),
            "Puri": (19.8135, 85.8312),
            "Balasore": (21.4934, 86.9135),
            "Mayurbhanj": (21.9287, 86.7454),
            "Ganjam": (19.3800, 85.0500),
        },
        "possible_outbreaks": ["malaria", "gastroenteritis", "dengue"],
        "temp_base": 29.0,
        "summer_temp": 39.0,
    },
    "Assam": {
        "climate_type": "riverine_flood",
        "districts": {
            "Kamrup": (26.1445, 91.7362),
            "Dibrugarh": (27.4728, 94.9120),
            "Cachar": (24.8333, 92.7789),
            "Jorhat": (26.7509, 94.2037),
            "Nagaon": (26.3464, 92.6840),
            "Sonitpur": (26.6338, 92.7926),
        },
        "possible_outbreaks": ["gastroenteritis", "malaria", "japanese_encephalitis"],
        "temp_base": 25.5,
        "summer_temp": 34.0,
    },
}

# ---------------------------------------------------------------------------
# 2. FACILITIES MASTER (3 PHCs per district = 90 PHCs)
# ---------------------------------------------------------------------------
facilities = []
phc_counter = 1

for state_name, state_info in STATES_CONFIG.items():
    for district_name, (lat0, lon0) in state_info["districts"].items():
        for d_phc_idx in range(1, 4):  # 3 PHCs per district
            phc_id = f"PHC_{phc_counter:03d}"
            phc_counter += 1
            
            # Geographic jitter around district centroid (~5-30 km)
            lat = round(lat0 + np.random.uniform(-0.18, 0.18), 5)
            lon = round(lon0 + np.random.uniform(-0.18, 0.18), 5)
            
            total_beds = int(np.random.choice([15, 20, 25, 30, 40], p=[0.25, 0.35, 0.20, 0.15, 0.05]))
            population_served = int(np.random.randint(9000, 45000))
            doctors_required = max(2, round(total_beds / 12))
            nurses_required = max(4, round(total_beds / 4))
            pharmacists_required = max(1, round(total_beds / 20))
            dist_to_warehouse = round(np.random.uniform(8, 65), 1)
            lead_time_days = max(1, round(dist_to_warehouse / 25))
            
            facilities.append({
                "phc_id": phc_id,
                "phc_name": f"{district_name} PHC-{d_phc_idx}",
                "district": district_name,
                "state": state_name,
                "latitude": lat,
                "longitude": lon,
                "total_beds": total_beds,
                "population_served": population_served,
                "doctors_required": doctors_required,
                "nurses_required": nurses_required,
                "pharmacists_required": pharmacists_required,
                "distance_to_district_warehouse_km": dist_to_warehouse,
                "warehouse_lead_time_days": lead_time_days,
            })

facilities_df = pd.DataFrame(facilities)
facilities_df.to_csv(f"{OUT}/facilities.csv", index=False)
phc_ids = facilities_df["phc_id"].tolist()
districts = list(facilities_df["district"].unique())

# ---------------------------------------------------------------------------
# 3. DISTANCE MATRIX (Inter-PHC road distances & transit hours)
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

distance_rows = []
fac_list = facilities_df.to_dict(orient="records")

for a in fac_list:
    for b in fac_list:
        if a["phc_id"] == b["phc_id"]:
            continue
        d_km = round(haversine(a["latitude"], a["longitude"], b["latitude"], b["longitude"]), 1)
        # Rural roads for local (<100km), national corridors for long-distance
        speed = np.random.uniform(32, 42) if d_km < 120 else np.random.uniform(50, 65)
        travel_hours = round(d_km / speed, 2)
        distance_rows.append({
            "phc_id_from": a["phc_id"],
            "phc_id_to": b["phc_id"],
            "distance_km": d_km,
            "travel_time_hours": travel_hours,
        })

distance_df = pd.DataFrame(distance_rows)
distance_df.to_csv(f"{OUT}/distance_matrix.csv", index=False)

# ---------------------------------------------------------------------------
# 4. MEDICINE MASTER (National Essential Medicine List)
# ---------------------------------------------------------------------------
medicines = [
    ("ORS_SACHET", "ORS", "sachet", 200, 400, 730, 5.0),
    ("PARACETAMOL_500MG", "analgesic", "tablet", 500, 1000, 900, 1.0),
    ("AMOXICILLIN_500MG", "antibiotic", "capsule", 300, 600, 540, 3.0),
    ("ANTIMALARIAL_ACT", "antimalarial", "tablet", 100, 250, 730, 12.0),
    ("ANTIVENOM_VIAL", "antivenom", "vial", 10, 25, 365, 450.0),
    ("INSULIN_VIAL", "chronic", "vial", 40, 80, 300, 180.0),
    ("ORAL_REHYDRATION_IV", "IV_fluid", "bottle", 80, 150, 545, 40.0),
    ("MEASLES_VACCINE", "vaccine", "dose", 150, 300, 180, 25.0),
    ("ANTIHISTAMINE", "allergy", "tablet", 200, 400, 900, 2.0),
    ("IBUPROFEN_400MG", "analgesic", "tablet", 300, 600, 900, 1.5),
    ("COUGH_SYRUP", "respiratory", "bottle", 100, 200, 545, 30.0),
    ("ANTIDIARRHEAL", "gastro", "tablet", 150, 300, 730, 2.0),
    ("DENGUE_TEST_KIT", "diagnostic", "kit", 50, 120, 365, 60.0),
    ("TETANUS_TOXOID", "vaccine", "dose", 100, 200, 365, 20.0),
    ("ORAL_ANTIBIOTIC_PEDIATRIC", "antibiotic", "bottle", 150, 300, 540, 35.0),
]
medicines_df = pd.DataFrame(medicines, columns=[
    "drug_id", "category", "unit", "reorder_point", "safety_stock",
    "shelf_life_days", "unit_cost_inr"
])
medicines_df.to_csv(f"{OUT}/medicines.csv", index=False)
drug_ids = medicines_df["drug_id"].tolist()

# ---------------------------------------------------------------------------
# 5. EXTERNAL SIGNALS (Weather, Outbreaks, Health Drives per District)
# ---------------------------------------------------------------------------
# Pre-schedule district-specific outbreak windows based on regional agro-climates
outbreak_windows = {}  # district -> list of (start_idx, end_idx, type, severity)

for state_name, state_info in STATES_CONFIG.items():
    c_type = state_info["climate_type"]
    for dist in state_info["districts"].keys():
        windows = []
        if c_type == "northeast_monsoon":
            # Tamil Nadu: Outbreak during Northeast monsoon (Oct-Dec, days ~35 to 120)
            s1 = int(np.random.randint(40, 80))
            windows.append((s1, s1 + np.random.randint(14, 28), "dengue", round(np.random.uniform(0.65, 0.95), 2)))
            s2 = int(np.random.randint(160, 240))
            windows.append((s2, s2 + np.random.randint(10, 20), "seasonal_flu", round(np.random.uniform(0.5, 0.8), 2)))
        elif c_type == "plateau_monsoon":
            # Maharashtra: Monsoon flu/dengue (Jul-Sep, days ~300 to 360) and winter viral fevers
            s1 = int(np.random.randint(300, 340))
            windows.append((s1, s1 + np.random.randint(12, 24), "dengue", round(np.random.uniform(0.6, 0.9), 2)))
            s2 = int(np.random.randint(110, 160))
            windows.append((s2, s2 + np.random.randint(10, 20), "seasonal_flu", round(np.random.uniform(0.5, 0.75), 2)))
        elif c_type == "arid_desert":
            # Rajasthan: May-June Extreme Heatwave (days ~240 to 290)
            s1 = int(np.random.randint(245, 275))
            windows.append((s1, s1 + np.random.randint(15, 25), "heatstroke_dehydration", round(np.random.uniform(0.75, 1.0), 2)))
            s2 = int(np.random.randint(310, 350))
            windows.append((s2, s2 + np.random.randint(10, 18), "malaria", round(np.random.uniform(0.5, 0.8), 2)))
        elif c_type == "coastal_cyclone":
            # Odisha: Endemic Malaria & Post-cyclone water-borne gastroenteritis
            s1 = int(np.random.randint(30, 75))
            windows.append((s1, s1 + np.random.randint(14, 25), "malaria", round(np.random.uniform(0.7, 0.95), 2)))
            s2 = int(np.random.randint(280, 320))
            windows.append((s2, s2 + np.random.randint(12, 22), "gastroenteritis", round(np.random.uniform(0.65, 0.9), 2)))
        elif c_type == "riverine_flood":
            # Assam: Monsoon flood gastro (Jun-Aug, days ~280 to 350) & Japanese Encephalitis
            s1 = int(np.random.randint(285, 325))
            windows.append((s1, s1 + np.random.randint(16, 28), "gastroenteritis", round(np.random.uniform(0.75, 1.0), 2)))
            s2 = int(np.random.randint(326, 355))
            windows.append((s2, s2 + np.random.randint(10, 20), "japanese_encephalitis", round(np.random.uniform(0.6, 0.85), 2)))
        
        outbreak_windows[dist] = windows

# Build District-to-State Lookup
district_state_map = facilities_df.drop_duplicates(subset=["district"]).set_index("district")["state"].to_dict()

external_rows = []
for d_idx, date in enumerate(dates):
    m = date.month
    dstr = date.strftime("%Y-%m-%d")
    
    for district in districts:
        st_name = district_state_map[district]
        c_type = STATES_CONFIG[st_name]["climate_type"]
        base_temp = STATES_CONFIG[st_name]["temp_base"]
        peak_summer_temp = STATES_CONFIG[st_name]["summer_temp"]
        
        # Climate-specific rainfall and temperature modeling
        if c_type == "northeast_monsoon":
            is_rainy = m in (10, 11, 12)
            is_summer = m in (4, 5, 6)
            rain = np.random.gamma(2.5, 18) if is_rainy else np.random.gamma(1.0, 3.0)
            temp = np.random.normal(peak_summer_temp if is_summer else base_temp, 2.5)
        elif c_type == "plateau_monsoon":
            is_rainy = m in (6, 7, 8, 9)
            is_summer = m in (4, 5)
            rain = np.random.gamma(3.0, 16) if is_rainy else np.random.gamma(0.8, 2.0)
            temp = np.random.normal(peak_summer_temp if is_summer else base_temp, 2.8)
        elif c_type == "arid_desert":
            is_rainy = m in (7, 8)
            is_summer = m in (5, 6)
            rain = np.random.gamma(1.2, 8) if is_rainy else np.random.gamma(0.2, 1.0)
            temp = np.random.normal(peak_summer_temp if is_summer else base_temp, 3.2)
        elif c_type == "coastal_cyclone":
            is_rainy = m in (6, 7, 8, 9, 10)
            is_summer = m in (4, 5)
            rain = np.random.gamma(3.5, 20) if is_rainy else np.random.gamma(1.2, 4.0)
            temp = np.random.normal(peak_summer_temp if is_summer else base_temp, 2.4)
        else:  # riverine_flood (Assam)
            is_rainy = m in (5, 6, 7, 8, 9)
            is_summer = m in (6, 7)
            rain = np.random.gamma(4.0, 24) if is_rainy else np.random.gamma(1.5, 5.0)
            temp = np.random.normal(peak_summer_temp if is_summer else base_temp, 2.0)
            
        rainfall_mm = round(max(0.0, float(rain)), 1)
        temperature_c = round(float(temp), 1)
        
        # Outbreak status
        active_wins = [w for w in outbreak_windows[district] if w[0] <= d_idx <= w[1]]
        outbreak_active = len(active_wins) > 0
        outbreak_type = active_wins[0][2] if outbreak_active else "none"
        severity = active_wins[0][3] if outbreak_active else 0.0
        
        # Periodic vaccination and health drives (~monthly 5-day windows)
        campaign_active = (d_idx % 30) < 5
        campaign_type = np.random.choice(
            ["measles_drive", "tetanus_drive", "general_immunization"]
        ) if campaign_active else "none"
        
        external_rows.append({
            "date": dstr,
            "district": district,
            "rainfall_mm": rainfall_mm,
            "temperature_c": temperature_c,
            "outbreak_active": outbreak_active,
            "outbreak_type": outbreak_type,
            "outbreak_severity": severity,
            "campaign_active": campaign_active,
            "campaign_type": campaign_type,
        })

external_df = pd.DataFrame(external_rows)
external_df.to_csv(f"{OUT}/external_signals.csv", index=False)
ext_lookup = external_df.set_index(["date", "district"])

# ---------------------------------------------------------------------------
# 6. STAFF ROSTER + ATTENDANCE (Module 3)
# ---------------------------------------------------------------------------
roster_rows = []
staff_id_counter = 1
staff_by_phc = {}

for _, phc in facilities_df.iterrows():
    staff_list = []
    for role, n_req in [
        ("doctor", phc["doctors_required"]),
        ("nurse", phc["nurses_required"]),
        ("pharmacist", phc["pharmacists_required"])
    ]:
        n_staff = n_req + np.random.randint(0, 2)
        for _ in range(n_staff):
            sid = f"STAFF_{staff_id_counter:04d}"
            staff_id_counter += 1
            reliability = round(np.random.beta(8, 2), 2)
            staff_list.append({"staff_id": sid, "role": role, "reliability_score": reliability})
    staff_by_phc[phc["phc_id"]] = staff_list
    for s in staff_list:
        roster_rows.append({"phc_id": phc["phc_id"], **s})

roster_df = pd.DataFrame(roster_rows)
roster_df.to_csv(f"{OUT}/staff_roster.csv", index=False)

attendance_rows = []
shifts = ["MORNING", "EVENING", "NIGHT"]

for _, phc in facilities_df.iterrows():
    pid = phc["phc_id"]
    district = phc["district"]
    for d_idx, date in enumerate(dates):
        dstr = date.strftime("%Y-%m-%d")
        try:
            outbreak_active = ext_lookup.loc[(dstr, district), "outbreak_active"]
        except KeyError:
            outbreak_active = False
            
        for s in staff_by_phc[pid]:
            # ~6 days/week scheduling
            if np.random.rand() > (6 / 7):
                continue
            shift = np.random.choice(shifts, p=[0.45, 0.35, 0.20])
            absence_prob = (1 - s["reliability_score"]) * (1.35 if outbreak_active else 1.0)
            if date.weekday() == 6:
                absence_prob *= 1.25
            present = np.random.rand() > absence_prob
            leave_type = "none"
            if not present:
                leave_type = np.random.choice(["sick", "planned", "unplanned"], p=[0.5, 0.3, 0.2])
                
            attendance_rows.append({
                "date": dstr,
                "phc_id": pid,
                "staff_id": s["staff_id"],
                "role": s["role"],
                "shift": shift,
                "scheduled": True,
                "present": bool(present),
                "leave_type": leave_type,
            })

attendance_df = pd.DataFrame(attendance_rows)
# Introduce ~1.2% missing biometric logging rows (real-world hardware packet drops)
drop_idx = attendance_df.sample(frac=0.012, random_state=SEED).index
attendance_df = attendance_df.drop(drop_idx)
attendance_df.to_csv(f"{OUT}/staff_attendance.csv", index=False)

# Precomputed staffing daily summary table
staffing_summary = (
    attendance_df.groupby(["date", "phc_id", "role"])
    .agg(scheduled_count=("staff_id", "count"), present_count=("present", "sum"))
    .reset_index()
)
staffing_summary["shortfall"] = staffing_summary["scheduled_count"] - staffing_summary["present_count"]
staffing_summary.to_csv(f"{OUT}/staffing_daily_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 7. ADMISSIONS / DISCHARGES + BED OCCUPANCY (Module 2)
# ---------------------------------------------------------------------------
diagnosis_los = {
    "pneumonia": (5, 2), "dengue": (6, 2.5), "gastroenteritis": (3, 1),
    "malaria": (4, 1.5), "maternal_delivery": (2, 1), "trauma_minor": (2, 1),
    "diabetes_complication": (7, 3), "respiratory_infection": (4, 1.5),
    "snakebite": (5, 2), "other": (3, 1.5),
}
severities = ["mild", "moderate", "critical"]

admission_rows = []
active_patients = {p: [] for p in phc_ids}
patient_counter = 1
bed_occupancy_rows = []

for d_idx, date in enumerate(dates):
    dstr = date.strftime("%Y-%m-%d")
    for _, phc in facilities_df.iterrows():
        pid = phc["phc_id"]
        district = phc["district"]
        total_beds = phc["total_beds"]
        
        try:
            row = ext_lookup.loc[(dstr, district)]
            outbreak_active = bool(row["outbreak_active"])
            outbreak_type = str(row["outbreak_type"])
            severity_ext = float(row["outbreak_severity"])
        except KeyError:
            outbreak_active, outbreak_type, severity_ext = False, "none", 0.0
            
        # Discharge patients whose stay completed
        still_active = []
        discharges_today = 0
        for p in active_patients[pid]:
            if p[1] <= d_idx:
                discharges_today += 1
            else:
                still_active.append(p)
        active_patients[pid] = still_active
        
        # Admissions generation
        base_rate = phc["population_served"] / 400000
        outbreak_mult = (1.0 + 1.8 * severity_ext) if outbreak_active else 1.0
        weekday_mult = 1.15 if date.weekday() in (0, 1) else 1.0
        lam = max(0.2, base_rate * outbreak_mult * weekday_mult * 8)
        n_admissions = np.random.poisson(lam)
        
        for _ in range(n_admissions):
            patient_id = f"P_{patient_counter:06d}"
            patient_counter += 1
            
            # Map outbreak types to clinical diagnosis categories
            if outbreak_active and np.random.rand() < 0.65:
                if outbreak_type in diagnosis_los:
                    diagnosis = outbreak_type
                elif outbreak_type == "seasonal_flu":
                    diagnosis = "respiratory_infection"
                elif outbreak_type in ("heatstroke_dehydration", "gastroenteritis"):
                    diagnosis = "gastroenteritis"
                elif outbreak_type == "japanese_encephalitis":
                    diagnosis = "other"
                else:
                    diagnosis = "other"
            else:
                diagnosis = np.random.choice(list(diagnosis_los.keys()))
                
            mean_los, std_los = diagnosis_los.get(diagnosis, (3, 1.5))
            los_days = max(1, int(round(np.random.normal(mean_los, std_los))))
            age = int(np.clip(np.random.normal(38, 20), 0, 95))
            gender = np.random.choice(["M", "F"])
            sev = np.random.choice(severities, p=[0.55, 0.35, 0.10])
            
            admission_rows.append({
                "patient_id": patient_id,
                "phc_id": pid,
                "admission_date": dstr,
                "discharge_date": (date + timedelta(days=los_days)).strftime("%Y-%m-%d"),
                "age": age,
                "gender": gender,
                "diagnosis_category": diagnosis,
                "severity": sev,
                "los_days": los_days,
                "admission_source": np.random.choice(["OPD", "emergency", "referral"], p=[0.5, 0.3, 0.2]),
            })
            active_patients[pid].append((patient_id, d_idx + los_days))
            
        occupied = len(active_patients[pid])
        occupancy_pct = round(100.0 * occupied / total_beds, 1)
        bed_occupancy_rows.append({
            "date": dstr,
            "phc_id": pid,
            "total_beds": total_beds,
            "occupied_beds": occupied,
            "available_beds": max(0, total_beds - occupied),
            "occupancy_pct": occupancy_pct,
            "admissions": n_admissions,
            "discharges": discharges_today,
        })

admissions_df = pd.DataFrame(admission_rows)
admissions_df.to_csv(f"{OUT}/admissions_discharges.csv", index=False)

bed_occupancy_df = pd.DataFrame(bed_occupancy_rows)
bed_occupancy_df.to_csv(f"{OUT}/bed_occupancy_daily.csv", index=False)

# ---------------------------------------------------------------------------
# 8. MEDICINE CONSUMPTION + INVENTORY (Module 1) — Multi-Signal Cross-Module Coupling
# ---------------------------------------------------------------------------
bed_occ_lookup = bed_occupancy_df.set_index(["date", "phc_id"])

drug_outbreak_affinity = {
    "dengue": ["DENGUE_TEST_KIT", "PARACETAMOL_500MG", "ORAL_REHYDRATION_IV"],
    "seasonal_flu": ["COUGH_SYRUP", "ANTIHISTAMINE", "PARACETAMOL_500MG", "AMOXICILLIN_500MG"],
    "gastroenteritis": ["ORS_SACHET", "ANTIDIARRHEAL", "ORAL_REHYDRATION_IV", "ORAL_ANTIBIOTIC_PEDIATRIC"],
    "malaria": ["ANTIMALARIAL_ACT", "PARACETAMOL_500MG", "ORAL_REHYDRATION_IV"],
    "heatstroke_dehydration": ["ORS_SACHET", "ORAL_REHYDRATION_IV", "PARACETAMOL_500MG"],
    "japanese_encephalitis": ["PARACETAMOL_500MG", "ORAL_REHYDRATION_IV", "ORAL_ANTIBIOTIC_PEDIATRIC"],
}

consumption_rows = []
inventory_rows = []
stock_state = {}

for _, phc in facilities_df.iterrows():
    for drug in medicines_df.itertuples():
        stock_state[(phc["phc_id"], drug.drug_id)] = int(drug.safety_stock * np.random.uniform(1.2, 2.0))

for d_idx, date in enumerate(dates):
    dstr = date.strftime("%Y-%m-%d")
    for _, phc in facilities_df.iterrows():
        pid = phc["phc_id"]
        district = phc["district"]
        occ_row = bed_occ_lookup.loc[(dstr, pid)]
        
        try:
            ext_row = ext_lookup.loc[(dstr, district)]
            outbreak_active = bool(ext_row["outbreak_active"])
            outbreak_type = str(ext_row["outbreak_type"])
            campaign_active = bool(ext_row["campaign_active"])
            campaign_type = str(ext_row["campaign_type"])
        except KeyError:
            outbreak_active, outbreak_type, campaign_active, campaign_type = False, "none", False, "none"
            
        footfall = int(occ_row["occupied_beds"] * np.random.uniform(2.5, 4.0) + np.random.poisson(phc["population_served"] / 6000))
        
        for drug in medicines_df.itertuples():
            base = {
                "ORS_SACHET": 8, "PARACETAMOL_500MG": 15, "AMOXICILLIN_500MG": 6,
                "ANTIMALARIAL_ACT": 2, "ANTIVENOM_VIAL": 0.1, "INSULIN_VIAL": 1.5,
                "ORAL_REHYDRATION_IV": 3, "MEASLES_VACCINE": 1, "ANTIHISTAMINE": 4,
                "IBUPROFEN_400MG": 10, "COUGH_SYRUP": 3, "ANTIDIARRHEAL": 5,
                "DENGUE_TEST_KIT": 1, "TETANUS_TOXOID": 1, "ORAL_ANTIBIOTIC_PEDIATRIC": 3,
            }[drug.drug_id]
            
            mult = 1.0
            if outbreak_active and drug.drug_id in drug_outbreak_affinity.get(outbreak_type, []):
                mult *= np.random.uniform(2.2, 3.8)
            if campaign_active and (
                (drug.drug_id == "MEASLES_VACCINE" and campaign_type == "measles_drive") or
                (drug.drug_id == "TETANUS_TOXOID" and campaign_type == "tetanus_drive")
            ):
                mult *= np.random.uniform(4.0, 7.0)
                
            footfall_mult = 0.7 + 0.6 * (footfall / (phc["population_served"] / 6000 + 1))
            dow_mult = 0.6 if date.weekday() == 6 else 1.0
            
            qty = max(0, int(np.random.poisson(base * mult * footfall_mult * dow_mult)))
            
            key = (pid, drug.drug_id)
            opening_stock = stock_state[key]
            
            # Reorder fulfillment logic
            received = 0
            if opening_stock < drug.reorder_point and np.random.rand() < (1.0 / phc["warehouse_lead_time_days"]):
                received = int(drug.reorder_point * np.random.uniform(2.0, 3.0))
            closing_stock = max(0, opening_stock + received - qty)
            stock_state[key] = closing_stock
            
            consumption_rows.append({
                "date": dstr,
                "phc_id": pid,
                "drug_id": drug.drug_id,
                "qty_dispensed": qty,
                "patients_served_estimate": footfall,
            })
            inventory_rows.append({
                "date": dstr,
                "phc_id": pid,
                "drug_id": drug.drug_id,
                "opening_stock": opening_stock,
                "received_qty": received,
                "dispensed_qty": qty,
                "closing_stock": closing_stock,
                "reorder_point": drug.reorder_point,
                "safety_stock": drug.safety_stock,
            })

consumption_df = pd.DataFrame(consumption_rows)
inventory_df = pd.DataFrame(inventory_rows)

# Deliberate data cleaning challenges
# 1. 0.8% missing values in consumption (null entries from paper log transitions)
miss_idx = consumption_df.sample(frac=0.008, random_state=SEED).index
consumption_df.loc[miss_idx, "qty_dispensed"] = np.nan

# 2. 0.2% negative stock recount anomalies (physical inventory reconciliations)
outlier_idx = inventory_df.sample(frac=0.002, random_state=SEED + 1).index
inventory_df.loc[outlier_idx, "closing_stock"] = inventory_df.loc[outlier_idx, "closing_stock"] * -1

consumption_df.to_csv(f"{OUT}/medicine_transactions.csv", index=False)
inventory_df.to_csv(f"{OUT}/inventory_snapshots.csv", index=False)

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUCCESS: National-scale Indian PHC dataset generated!")
print(f"Total States:    {len(STATES_CONFIG)} ({', '.join(STATES_CONFIG.keys())})")
print(f"Total Districts: {len(districts)}")
print(f"Total PHCs:      {len(facilities_df)}")
print("=" * 60)
for f in [
    "facilities.csv", "medicines.csv", "distance_matrix.csv", "external_signals.csv",
    "staff_roster.csv", "staff_attendance.csv", "staffing_daily_summary.csv",
    "admissions_discharges.csv", "bed_occupancy_daily.csv",
    "medicine_transactions.csv", "inventory_snapshots.csv"
]:
    df = pd.read_csv(f"{OUT}/{f}")
    size_mb = round(os.path.getsize(f"{OUT}/{f}") / (1024 * 1024), 2)
    print(f"  {f:35s} {len(df):>9,} rows  {len(df.columns):>2} cols  ({size_mb} MB)")
print("=" * 60 + "\n")
