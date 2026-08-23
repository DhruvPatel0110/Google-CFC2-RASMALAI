"""
PHC Supply Chain Resilience — Synthetic Dataset Generator
Generates realistic, correlated data for Modules 1-4:
  1. Medicine consumption + inventory
  2. Bed admissions/discharges + occupancy
  3. Staff attendance
  4. Facility network (for redistribution)

Design principles:
- 15 PHCs across 3 districts (Tamil Nadu-style layout, since that's a plausible
  state to anchor a BRICS/India hackathon demo in)
- 365 days of daily history (1 full year -> captures seasonality)
- Correlated signals: outbreak weeks push up admissions AND medicine consumption
  AND bed occupancy simultaneously (this is what makes Module 4's cross-module
  logic demoable — you need the modules to actually agree with each other)
- Deliberate messiness: missing days, a few negative-corrected stock entries,
  duplicate-ish rows — so your "Data Cleaning" pipeline slide has real work to show
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

START_DATE = datetime(2024, 9, 1)
N_DAYS = 365
dates = [START_DATE + timedelta(days=i) for i in range(N_DAYS)]

OUT = "/home/claude/phc_data"

# ---------------------------------------------------------------------------
# 1. FACILITIES MASTER
# ---------------------------------------------------------------------------
districts = ["Vellore", "Krishnagiri", "Tiruvannamalai"]
# rough lat/lon anchors per district (Tamil Nadu region), PHCs jittered around them
district_anchor = {
    "Vellore": (12.9165, 79.1325),
    "Krishnagiri": (12.5186, 78.2137),
    "Tiruvannamalai": (12.2253, 79.0747),
}

n_phcs = 15
facilities = []
for i in range(1, n_phcs + 1):
    phc_id = f"PHC_{i:03d}"
    district = districts[(i - 1) % 3]
    lat0, lon0 = district_anchor[district]
    lat = round(lat0 + np.random.uniform(-0.35, 0.35), 5)
    lon = round(lon0 + np.random.uniform(-0.35, 0.35), 5)
    total_beds = int(np.random.choice([15, 20, 25, 30, 40], p=[0.25, 0.3, 0.25, 0.15, 0.05]))
    population_served = int(np.random.randint(8000, 45000))
    doctors_required = max(2, round(total_beds / 12))
    nurses_required = max(4, round(total_beds / 4))
    pharmacists_required = max(1, round(total_beds / 20))
    dist_to_warehouse = round(np.random.uniform(8, 65), 1)
    lead_time_days = max(1, round(dist_to_warehouse / 25))
    facilities.append({
        "phc_id": phc_id, "phc_name": f"{district} PHC-{i}", "district": district,
        "state": "Tamil Nadu", "latitude": lat, "longitude": lon,
        "total_beds": total_beds, "population_served": population_served,
        "doctors_required": doctors_required, "nurses_required": nurses_required,
        "pharmacists_required": pharmacists_required,
        "distance_to_district_warehouse_km": dist_to_warehouse,
        "warehouse_lead_time_days": lead_time_days,
    })
facilities_df = pd.DataFrame(facilities)
facilities_df.to_csv(f"{OUT}/facilities.csv", index=False)

phc_ids = facilities_df["phc_id"].tolist()

# ---------------------------------------------------------------------------
# 2. DISTANCE MATRIX (for Module 4 redistribution)
# ---------------------------------------------------------------------------
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

rows = []
for i, a in facilities_df.iterrows():
    for j, b in facilities_df.iterrows():
        if a.phc_id == b.phc_id:
            continue
        d_km = round(haversine(a.latitude, a.longitude, b.latitude, b.longitude), 1)
        travel_hours = round(d_km / np.random.uniform(28, 40), 2)  # rural road speeds
        rows.append({"phc_id_from": a.phc_id, "phc_id_to": b.phc_id,
                      "distance_km": d_km, "travel_time_hours": travel_hours})
distance_df = pd.DataFrame(rows)
distance_df.to_csv(f"{OUT}/distance_matrix.csv", index=False)

# ---------------------------------------------------------------------------
# 3. MEDICINE MASTER
# ---------------------------------------------------------------------------
medicines = [
    ("ORS_SACHET", "ORS", "sachet", 200, 400, 730, 5),
    ("PARACETAMOL_500MG", "analgesic", "tablet", 500, 1000, 900, 1),
    ("AMOXICILLIN_500MG", "antibiotic", "capsule", 300, 600, 540, 3),
    ("ANTIMALARIAL_ACT", "antimalarial", "tablet", 100, 250, 730, 12),
    ("ANTIVENOM_VIAL", "antivenom", "vial", 10, 25, 365, 450),
    ("INSULIN_VIAL", "chronic", "vial", 40, 80, 300, 180),
    ("ORAL_REHYDRATION_IV", "IV_fluid", "bottle", 80, 150, 545, 40),
    ("MEASLES_VACCINE", "vaccine", "dose", 150, 300, 180, 25),
    ("ANTIHISTAMINE", "allergy", "tablet", 200, 400, 900, 2),
    ("IBUPROFEN_400MG", "analgesic", "tablet", 300, 600, 900, 1.5),
    ("COUGH_SYRUP", "respiratory", "bottle", 100, 200, 545, 30),
    ("ANTIDIARRHEAL", "gastro", "tablet", 150, 300, 730, 2),
    ("DENGUE_TEST_KIT", "diagnostic", "kit", 50, 120, 365, 60),
    ("TETANUS_TOXOID", "vaccine", "dose", 100, 200, 365, 20),
    ("ORAL_ANTIBIOTIC_PEDIATRIC", "antibiotic", "bottle", 150, 300, 540, 35),
]
medicines_df = pd.DataFrame(medicines, columns=[
    "drug_id", "category", "unit", "reorder_point", "safety_stock",
    "shelf_life_days", "unit_cost_inr"])
medicines_df.to_csv(f"{OUT}/medicines.csv", index=False)
drug_ids = medicines_df["drug_id"].tolist()

# ---------------------------------------------------------------------------
# 4. EXTERNAL SIGNALS (weather, outbreaks, campaigns) — drives everything else
# ---------------------------------------------------------------------------
# Monsoon: Oct-Dec (higher dengue/diarrheal risk in TN); summer heat: Apr-Jun
def month_of(d):
    return d.month

external_rows = []
outbreak_windows = {}  # district -> list of (start_idx, end_idx, type)
for district in districts:
    # 2 random outbreak windows per district across the year
    starts = sorted(np.random.choice(range(20, N_DAYS - 30), size=2, replace=False))
    windows = []
    for s in starts:
        length = np.random.randint(10, 25)
        outbreak_type = np.random.choice(["dengue", "seasonal_flu", "gastroenteritis"])
        windows.append((s, s + length, outbreak_type))
    outbreak_windows[district] = windows

for d_idx, date in enumerate(dates):
    m = month_of(date)
    is_monsoon = m in (10, 11, 12)
    is_summer = m in (4, 5, 6)
    base_rain = np.random.gamma(2, 15) if is_monsoon else np.random.gamma(1, 3)
    rainfall_mm = round(max(0, base_rain + (40 if is_monsoon else 0)), 1)
    temperature_c = round(np.random.normal(38 if is_summer else 29, 3), 1)

    for district in districts:
        active_windows = [w for w in outbreak_windows[district] if w[0] <= d_idx <= w[1]]
        outbreak_active = len(active_windows) > 0
        outbreak_type = active_windows[0][2] if outbreak_active else "none"
        severity = round(np.random.uniform(0.5, 1.0), 2) if outbreak_active else 0.0
        # vaccination campaigns: ~monthly 5-day windows
        campaign_active = (d_idx % 30) < 5
        campaign_type = np.random.choice(["measles_drive", "tetanus_drive", "general_immunization"]) if campaign_active else "none"

        external_rows.append({
            "date": date.strftime("%Y-%m-%d"), "district": district,
            "rainfall_mm": rainfall_mm, "temperature_c": temperature_c,
            "outbreak_active": outbreak_active, "outbreak_type": outbreak_type,
            "outbreak_severity": severity,
            "campaign_active": campaign_active, "campaign_type": campaign_type,
        })
external_df = pd.DataFrame(external_rows)
external_df.to_csv(f"{OUT}/external_signals.csv", index=False)

ext_lookup = external_df.set_index(["date", "district"])

# ---------------------------------------------------------------------------
# 5. STAFF ROSTER + ATTENDANCE (Module 3)
# ---------------------------------------------------------------------------
roster_rows = []
staff_id_counter = 1
staff_by_phc = {}
for _, phc in facilities_df.iterrows():
    staff_list = []
    for role, n_req in [("doctor", phc.doctors_required),
                         ("nurse", phc.nurses_required),
                         ("pharmacist", phc.pharmacists_required)]:
        n_staff = n_req + np.random.randint(0, 2)  # slight overstaff buffer some places
        for _ in range(n_staff):
            sid = f"STAFF_{staff_id_counter:04d}"
            staff_id_counter += 1
            reliability = round(np.random.beta(8, 2), 2)  # most staff reliable, some flaky
            staff_list.append({"staff_id": sid, "role": role, "reliability_score": reliability})
    staff_by_phc[phc.phc_id] = staff_list
    for s in staff_list:
        roster_rows.append({"phc_id": phc.phc_id, **s})
roster_df = pd.DataFrame(roster_rows)
roster_df.to_csv(f"{OUT}/staff_roster.csv", index=False)

attendance_rows = []
shifts = ["MORNING", "EVENING", "NIGHT"]
for _, phc in facilities_df.iterrows():
    district = phc.district
    for d_idx, date in enumerate(dates):
        dstr = date.strftime("%Y-%m-%d")
        try:
            outbreak_active = ext_lookup.loc[(dstr, district), "outbreak_active"]
        except KeyError:
            outbreak_active = False
        for s in staff_by_phc[phc.phc_id]:
            # each staff scheduled ~5-6 days/week, one shift/day
            if np.random.rand() > (6 / 7):
                continue  # off-roster day, not an absence
            shift = np.random.choice(shifts, p=[0.45, 0.35, 0.20])
            absence_prob = (1 - s["reliability_score"]) * (1.4 if outbreak_active else 1.0)
            # slightly higher leave rate on Sundays
            if date.weekday() == 6:
                absence_prob *= 1.3
            present = np.random.rand() > absence_prob
            leave_type = "none"
            if not present:
                leave_type = np.random.choice(["sick", "planned", "unplanned"], p=[0.5, 0.3, 0.2])
            attendance_rows.append({
                "date": dstr, "phc_id": phc.phc_id, "staff_id": s["staff_id"],
                "role": s["role"], "shift": shift, "scheduled": True,
                "present": bool(present), "leave_type": leave_type,
            })
attendance_df = pd.DataFrame(attendance_rows)
# introduce ~1.5% missing rows (real biometric systems drop pings)
drop_idx = attendance_df.sample(frac=0.015, random_state=SEED).index
attendance_df = attendance_df.drop(drop_idx)
attendance_df.to_csv(f"{OUT}/staff_attendance.csv", index=False)

# Daily staffing summary per PHC/role (precomputed convenience table for Module 3)
staffing_summary = (attendance_df.groupby(["date", "phc_id", "role"])
                     .agg(scheduled_count=("staff_id", "count"),
                          present_count=("present", "sum"))
                     .reset_index())
staffing_summary["shortfall"] = staffing_summary["scheduled_count"] - staffing_summary["present_count"]
staffing_summary.to_csv(f"{OUT}/staffing_daily_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 6. ADMISSIONS / DISCHARGES + BED OCCUPANCY (Module 2)
# ---------------------------------------------------------------------------
diagnosis_los = {
    "pneumonia": (5, 2), "dengue": (6, 2.5), "gastroenteritis": (3, 1),
    "malaria": (4, 1.5), "maternal_delivery": (2, 1), "trauma_minor": (2, 1),
    "diabetes_complication": (7, 3), "respiratory_infection": (4, 1.5),
    "snakebite": (5, 2), "other": (3, 1.5),
}
severities = ["mild", "moderate", "critical"]

admission_rows = []
occupancy_state = {p: 0 for p in phc_ids}  # currently occupied beds (rough running tally)
active_patients = {p: [] for p in phc_ids}  # list of (patient_id, discharge_day_idx)
patient_counter = 1

bed_occupancy_rows = []

for d_idx, date in enumerate(dates):
    dstr = date.strftime("%Y-%m-%d")
    for _, phc in facilities_df.iterrows():
        pid = phc.phc_id
        district = phc.district
        try:
            row = ext_lookup.loc[(dstr, district)]
            outbreak_active, outbreak_type, severity_ext = row.outbreak_active, row.outbreak_type, row.outbreak_severity
        except KeyError:
            outbreak_active, outbreak_type, severity_ext = False, "none", 0.0

        # discharge patients whose discharge day has arrived
        still_active = []
        discharges_today = 0
        for p in active_patients[pid]:
            if p[1] <= d_idx:
                discharges_today += 1
            else:
                still_active.append(p)
        active_patients[pid] = still_active

        # base admission rate scales with population served
        base_rate = phc.population_served / 400000  # tuned so avg ~1-4/day
        outbreak_mult = (1 + 1.8 * severity_ext) if outbreak_active else 1.0
        weekday_mult = 1.15 if date.weekday() in (0, 1) else 1.0  # Mon/Tue higher OPD->admission
        lam = max(0.2, base_rate * outbreak_mult * weekday_mult * 8)
        n_admissions = np.random.poisson(lam)

        for _ in range(n_admissions):
            patient_id = f"P_{patient_counter:06d}"
            patient_counter += 1
            if outbreak_active and np.random.rand() < 0.6:
                diagnosis = outbreak_type if outbreak_type in diagnosis_los else "other"
                if outbreak_type == "seasonal_flu":
                    diagnosis = "respiratory_infection"
                if outbreak_type == "gastroenteritis":
                    diagnosis = "gastroenteritis"
            else:
                diagnosis = np.random.choice(list(diagnosis_los.keys()))
            mean_los, std_los = diagnosis_los.get(diagnosis, (3, 1.5))
            los_days = max(1, int(round(np.random.normal(mean_los, std_los))))
            age = int(np.clip(np.random.normal(38, 20), 0, 95))
            gender = np.random.choice(["M", "F"])
            sev = np.random.choice(severities, p=[0.55, 0.35, 0.10])

            admission_rows.append({
                "patient_id": patient_id, "phc_id": pid, "admission_date": dstr,
                "discharge_date": (date + timedelta(days=los_days)).strftime("%Y-%m-%d"),
                "age": age, "gender": gender, "diagnosis_category": diagnosis,
                "severity": sev, "los_days": los_days,
                "admission_source": np.random.choice(["OPD", "emergency", "referral"], p=[0.5, 0.3, 0.2]),
            })
            active_patients[pid].append((patient_id, d_idx + los_days))

        occupied = len(active_patients[pid])
        occupancy_pct = round(100 * occupied / phc.total_beds, 1)
        bed_occupancy_rows.append({
            "date": dstr, "phc_id": pid, "total_beds": phc.total_beds,
            "occupied_beds": occupied, "available_beds": max(0, phc.total_beds - occupied),
            "occupancy_pct": occupancy_pct, "admissions": n_admissions,
            "discharges": discharges_today,
        })

admissions_df = pd.DataFrame(admission_rows)
admissions_df.to_csv(f"{OUT}/admissions_discharges.csv", index=False)

bed_occupancy_df = pd.DataFrame(bed_occupancy_rows)
bed_occupancy_df.to_csv(f"{OUT}/bed_occupancy_daily.csv", index=False)

# ---------------------------------------------------------------------------
# 7. MEDICINE CONSUMPTION + INVENTORY (Module 1) — correlated with occupancy/outbreaks
# ---------------------------------------------------------------------------
bed_occ_lookup = bed_occupancy_df.set_index(["date", "phc_id"])

drug_outbreak_affinity = {
    "dengue": ["DENGUE_TEST_KIT", "PARACETAMOL_500MG", "ORAL_REHYDRATION_IV"],
    "seasonal_flu": ["COUGH_SYRUP", "ANTIHISTAMINE", "PARACETAMOL_500MG"],
    "gastroenteritis": ["ORS_SACHET", "ANTIDIARRHEAL", "ORAL_REHYDRATION_IV"],
}

consumption_rows = []
inventory_rows = []
stock_state = {}  # (phc_id, drug_id) -> current stock

for _, phc in facilities_df.iterrows():
    for drug in medicines_df.itertuples():
        stock_state[(phc.phc_id, drug.drug_id)] = int(drug.safety_stock * np.random.uniform(1.2, 2.0))

for d_idx, date in enumerate(dates):
    dstr = date.strftime("%Y-%m-%d")
    for _, phc in facilities_df.iterrows():
        pid = phc.phc_id
        district = phc.district
        occ_row = bed_occ_lookup.loc[(dstr, pid)]
        try:
            ext_row = ext_lookup.loc[(dstr, district)]
            outbreak_active, outbreak_type = ext_row.outbreak_active, ext_row.outbreak_type
            campaign_active, campaign_type = ext_row.campaign_active, ext_row.campaign_type
        except KeyError:
            outbreak_active, outbreak_type, campaign_active, campaign_type = False, "none", False, "none"

        footfall = int(occ_row.occupied_beds * np.random.uniform(2.5, 4.0) + np.random.poisson(phc.population_served / 6000))

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
                mult *= np.random.uniform(2.2, 3.5)
            if campaign_active and (
                (drug.drug_id == "MEASLES_VACCINE" and campaign_type == "measles_drive") or
                (drug.drug_id == "TETANUS_TOXOID" and campaign_type == "tetanus_drive")
            ):
                mult *= np.random.uniform(4, 7)
            footfall_mult = 0.7 + 0.6 * (footfall / (phc.population_served / 6000 + 1))
            dow_mult = 0.6 if date.weekday() == 6 else 1.0  # lower Sunday dispensing

            qty = max(0, int(np.random.poisson(base * mult * footfall_mult * dow_mult)))

            key = (pid, drug.drug_id)
            opening_stock = stock_state[key]
            # reorder trigger: if stock below reorder point, receive a delivery after lead_time
            received = 0
            if opening_stock < drug.reorder_point and np.random.rand() < (1 / phc.warehouse_lead_time_days):
                received = int(drug.reorder_point * np.random.uniform(2, 3))
            closing_stock = max(0, opening_stock + received - qty)
            stock_state[key] = closing_stock

            consumption_rows.append({
                "date": dstr, "phc_id": pid, "drug_id": drug.drug_id,
                "qty_dispensed": qty, "patients_served_estimate": footfall,
            })
            inventory_rows.append({
                "date": dstr, "phc_id": pid, "drug_id": drug.drug_id,
                "opening_stock": opening_stock, "received_qty": received,
                "dispensed_qty": qty, "closing_stock": closing_stock,
                "reorder_point": drug.reorder_point, "safety_stock": drug.safety_stock,
            })

consumption_df = pd.DataFrame(consumption_rows)
inventory_df = pd.DataFrame(inventory_rows)

# --- deliberately introduce messiness for the "data cleaning" pipeline slide ---
# 1. ~0.8% missing qty_dispensed (nulls)
miss_idx = consumption_df.sample(frac=0.008, random_state=SEED).index
consumption_df.loc[miss_idx, "qty_dispensed"] = np.nan
# 2. a few negative-correction outlier rows (stock recount adjustments)
outlier_idx = inventory_df.sample(frac=0.002, random_state=SEED + 1).index
inventory_df.loc[outlier_idx, "closing_stock"] = inventory_df.loc[outlier_idx, "closing_stock"] * -1

consumption_df.to_csv(f"{OUT}/medicine_transactions.csv", index=False)
inventory_df.to_csv(f"{OUT}/inventory_snapshots.csv", index=False)

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("Generated files:")
for f in ["facilities.csv", "medicines.csv", "distance_matrix.csv", "external_signals.csv",
          "staff_roster.csv", "staff_attendance.csv", "staffing_daily_summary.csv",
          "admissions_discharges.csv", "bed_occupancy_daily.csv",
          "medicine_transactions.csv", "inventory_snapshots.csv"]:
    df = pd.read_csv(f"{OUT}/{f}")
    print(f"  {f:35s} {len(df):>8,} rows  {len(df.columns):>2} cols")
