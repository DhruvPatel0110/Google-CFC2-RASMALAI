# PHC Supply Chain Resilience — Detailed Architecture (5 Modules)

**Hackathon Track:** BRICS Theme: Resilience  
**Problem:** Federated AI platform for national-scale health resource and supply chain management  
**Target:** Real-time visibility + forecasting + early warnings + automated redistribution across PHC networks

---

## Module 1: Medicine Stockout Forecasting & Inventory Management

### 1.1 Problem Statement (Module-Level)
Predict medicine stock depletion per PHC facility and per drug SKU to enable proactive procurement and redistribution decisions. Avoid stock-outs during health emergencies.

### 1.2 Data Inputs

**Source 1: Historical Consumption Data**
- Daily/weekly medicine issue rates per drug per PHC
- Quantity dispensed per patient visit
- Drug category (antibiotic, ORS, vaccine, etc.)
- Supplier and batch information
- Time period: minimum 12 months (to capture seasonality)

**Source 2: Current Inventory State**
- Current stock level per drug per PHC
- Reorder point thresholds (safety stock)
- Lead time from district warehouse (in days)
- Expiry date / shelf-life for each batch
- Storage capacity constraints per facility

**Source 3: External Demand Signals**
- Patient footfall trends (OPD registrations per day)
- Disease incidence data (IDSP reports, surveillance bulletins)
- Seasonal outbreak patterns (flu season, monsoon-linked conditions, malaria/dengue)
- Health campaign calendars (planned vaccination drives, awareness campaigns)
- Population served by each PHC (demographic profile)

**Source 4: Supply Chain Factors**
- Distance to nearest district warehouse (km)
- Average transport lead time (days)
- Supplier procurement cycle delays
- Road/weather disruption history (monsoon, floods)
- Neighboring PHC stock levels (for redistribution feasibility)

### 1.3 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Data Ingestion Layer                                         │
├─────────────────────────────────────────────────────────────┤
│ • ERP/HMIS database (medicine ledgers)                       │
│ • Manual CSV uploads (PHC attendance records)                │
│ • API connectors (IDSP disease surveillance)                 │
│ • Weather APIs (rainfall, temperature data)                  │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Data Cleaning & Normalization                               │
├─────────────────────────────────────────────────────────────┤
│ • Handle missing values (forward fill, interpolation)        │
│ • Remove outliers (sudden stock corrections)                 │
│ • Standardize units (convert mL to tablets, etc.)            │
│ • Timestamp alignment (sync multiple data sources)           │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Feature Engineering                                          │
├─────────────────────────────────────────────────────────────┤
│ Time-series features:                                        │
│  • Rolling averages (7-day, 14-day, 30-day consumption)     │
│  • Trend (slope of consumption line)                         │
│  • Seasonality (day-of-week, month-of-year patterns)        │
│  • Lag features (consumption t-1, t-7, t-30)                │
│                                                              │
│ External features:                                           │
│  • Outbreak indicator (binary: yes/no)                       │
│  • Campaign intensity (0-10 scale)                           │
│  • Days since last stockout                                  │
│  • Facility capacity utilization %                           │
│  • Staff attendance rate                                     │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Feature Store (cache for model retraining)                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 Model Architecture

**Primary Model: XGBoost Time Series Regression**

```python
# Input shape: (samples, features)
# Output: next_7_day_consumption_forecast per drug per PHC

XGBoost_Regressor:
  - n_estimators: 200
  - max_depth: 8
  - learning_rate: 0.1
  - subsample: 0.8
  - colsample_bytree: 0.8
  - objective: 'reg:squarederror'
  - early_stopping: 50 rounds on validation set
  
  Input Features (25-30 total):
    ├─ Time-series: rolling averages, trends, lags
    ├─ Seasonal: day_of_week, month, is_holiday
    ├─ External: outbreak_alert, campaign_active, weather
    ├─ Facility: bed_occupancy%, staff_attendance%, population
    └─ Drug: category, reorder_point, lead_time_days
```

**Alternative Model: LSTM (if trend prediction is critical)**

```
LSTM Architecture (for longer-horizon forecasting):
  - Input: (seq_len=30, n_features=25)  # 30 days history
  - Layer 1: LSTM(128 units, return_sequences=True) → Dropout(0.2)
  - Layer 2: LSTM(64 units, return_sequences=False) → Dropout(0.2)
  - Dense(32, activation='relu') → Dropout(0.1)
  - Dense(7)  # Output: 7-day forecast
  - Compile: optimizer='adam', loss='mse'
```

**Model Comparison & Selection:**
- **XGBoost**: Faster training, interpretable feature importance, handles non-linear patterns → **RECOMMENDED for MVP**
- **LSTM**: Better for capturing long-term dependencies, but slower, harder to debug → use if XGBoost underperforms

### 1.5 Model Outputs

**Output 1: Consumption Forecast (next 7 days)**
```json
{
  "phc_id": "PHC_001",
  "drug_id": "ASPIRIN_500MG",
  "forecast_date": "2024-08-30",
  "predicted_consumption": {
    "day_1": 45,
    "day_2": 52,
    "day_3": 48,
    "day_4": 55,
    "day_5": 50,
    "day_6": 58,
    "day_7": 51
  },
  "confidence_interval": [0.85, 0.95],
  "model_version": "xgboost_v2.1"
}
```

**Output 2: Stockout Risk Score (0-1, higher = more critical)**
```json
{
  "phc_id": "PHC_001",
  "drug_id": "ASPIRIN_500MG",
  "current_stock": 120,
  "predicted_consumption_7d": 359,
  "days_to_stockout": 2.8,
  "stockout_probability": 0.92,
  "alert_level": "CRITICAL",
  "recommendation": "URGENT: Reorder from warehouse or request redistribution from PHC_005"
}
```

**Output 3: Reorder Suggestion**
```json
{
  "action": "REORDER",
  "phc_id": "PHC_001",
  "drug_id": "ASPIRIN_500MG",
  "recommended_quantity": 500,
  "lead_time_days": 3,
  "order_by_date": "2024-08-29",
  "estimated_arrival": "2024-09-01"
}
```

### 1.6 Integration Points

- **Input from Module 3:** Staff attendance rate (affects patient volume → affects consumption)
- **Input from Module 2:** Bed occupancy (correlates with patient footfall → affects medicine usage)
- **Output to Module 4:** Stockout alerts + reorder quantities (trigger redistribution engine)
- **Output to Dashboard:** Real-time stock levels, alerts, historical trends

### 1.7 Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Data storage | PostgreSQL + TimescaleDB | Time-series optimized, handles high-frequency updates |
| Feature pipeline | Python (pandas + scikit-learn) | Standard ML preprocessing |
| Model training | XGBoost (Python wrapper) | Fast, interpretable, production-ready |
| Batch predictions | Apache Airflow (DAG scheduler) | Recompute forecasts daily at 2 AM |
| Serving | Flask API + Redis cache | Low-latency inference, cached forecasts |
| Monitoring | Prometheus + Grafana | Track model drift, prediction latency |

---

## Module 2: Bed Availability & Occupancy Forecasting

### 2.1 Problem Statement (Module-Level)
Forecast bed occupancy rates per PHC and predict available beds for next 7 days to enable patient admission planning and facility coordination during emergencies.

### 2.2 Data Inputs

**Source 1: Admission & Discharge Data**
- Daily admissions per PHC (count, by department/ward if available)
- Discharge dates and times
- Average length of stay (LOS) by condition/age group
- Readmission rates (patients returning within 30 days)
- Admission source (OPD, emergency, referral)

**Source 2: Current Facility State**
- Total bed count per PHC (by ward type if available: isolation, general, maternal)
- Current occupancy count (real-time)
- Bed capacity constraints (max capacity planning)
- ICU/ventilator availability (if applicable)

**Source 3: Patient-Level Signals**
- Patient demographics (age, gender — affects LOS)
- Diagnosis code (ICD-10 or local coding)
- Severity/acuity level (critical, moderate, mild)
- Comorbidities (correlates with longer stays)

**Source 4: External Demand Factors**
- Seasonal outbreak alerts (will surge admissions)
- Local events (accidents, heat waves — drive ED visits)
- Health campaigns (vaccination drives may reduce admissions)
- Holiday calendars (affects staffing + admission patterns)
- Neighboring facility status (full → patients diverted to your facility)

### 2.3 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Data Ingestion Layer                                         │
├─────────────────────────────────────────────────────────────┤
│ • HMIS/Hospital Management System (admission registers)      │
│ • Real-time census dashboards (bed occupancy snapshots)      │
│ • EDR/Disease surveillance (outbreak predictions)            │
│ • Weather/disaster APIs (heat wave alerts, flood risk)       │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Data Cleaning & Validation                                   │
├─────────────────────────────────────────────────────────────┤
│ • Remove outliers (LOS > 365 days → likely data errors)     │
│ • Impute missing discharge dates (use median LOS)            │
│ • Validate: admission_date < discharge_date                  │
│ • Handle same-day admissions (high variance, smooth)         │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Feature Engineering                                          │
├─────────────────────────────────────────────────────────────┤
│ Admission patterns:                                          │
│  • Rolling admission counts (7-day, 14-day)                  │
│  • Day-of-week patterns (Monday admits vs. Sunday)           │
│  • Admission rate trend (increasing/decreasing)              │
│                                                              │
│ LOS-based features:                                          │
│  • Median LOS by diagnosis (pre-computed lookup)             │
│  • Median LOS by age group                                   │
│  • LOS percentiles (p25, p50, p75 — for variance)            │
│                                                              │
│ Occupancy dynamics:                                          │
│  • Current occupancy %                                       │
│  • Occupancy trend (slope)                                   │
│  • Beds available (total - current)                          │
│                                                              │
│ External signals:                                            │
│  • Outbreak active (binary flag)                             │
│  • Days until major holiday                                  │
│  • Weather severity index (0-10)                             │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Feature Store                                                │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Model Architecture

**Two-Stage Forecasting System:**

**Stage 1: Admission Forecast (XGBoost Regression)**
```
Input: Historical admissions + external signals
Output: Predicted admissions for next 7 days

XGBoost_Admissions:
  - n_estimators: 150
  - max_depth: 7
  - learning_rate: 0.1
  
  Features:
    ├─ Time-series: rolling admissions, trends, seasonality
    ├─ External: outbreak_active, holiday_proximity, weather
    ├─ Facility: bed_capacity, current_occupancy%
    └─ Lagged: admissions_t-1, t-7, t-30
```

**Stage 2: Length of Stay Distribution (SVM or Gradient Boosting)**
```
Input: Patient demographics + diagnosis + admission type
Output: Predicted LOS (duration in days)

Median_LOS = LOS_lookup_table[age_group][diagnosis]
Variance = Historical_std_dev[age_group][diagnosis]

# For probabilistic forecast:
LOS_samples ~ Normal(Median_LOS, Variance)
```

**Stage 3: Occupancy Projection (Deterministic)**
```
For each day d in [1, 7]:
  Predicted_Admissions[d] = XGBoost_admission_forecast[d]
  Discharges[d] = {patients admitted on day (d - LOS[i]) for each patient i}
  
  Occupancy[d] = Occupancy[d-1] 
                 + Admissions[d] 
                 - Discharges[d]
  
  Available_Beds[d] = Total_Capacity - Occupancy[d]
```

### 2.5 Model Outputs

**Output 1: Occupancy Forecast (next 7 days)**
```json
{
  "phc_id": "PHC_001",
  "forecast_date": "2024-08-30",
  "total_beds": 50,
  "daily_forecast": [
    {
      "day": 1,
      "predicted_occupancy": 38,
      "occupancy_percent": 76,
      "predicted_admissions": 12,
      "predicted_discharges": 6,
      "available_beds": 12,
      "confidence_interval": [35, 42]
    },
    {
      "day": 2,
      "predicted_occupancy": 41,
      "occupancy_percent": 82,
      "predicted_admissions": 10,
      "predicted_discharges": 7,
      "available_beds": 9,
      "confidence_interval": [37, 45]
    }
    // ... days 3-7
  ],
  "alert_level": "CAUTION",
  "alert_reason": "Day 4 forecast shows 94% occupancy"
}
```

**Output 2: Bed Allocation Recommendation**
```json
{
  "phc_id": "PHC_001",
  "action": "REQUEST_TRANSFER",
  "target_phc": "PHC_005",
  "reason": "Occupancy forecast: 95% by day 5",
  "estimated_surplus_needed": 8,
  "urgency": "MEDIUM",
  "suggested_transfer_patients": [
    {
      "patient_id": "P_001",
      "condition": "recovering pneumonia",
      "recommended_target": "PHC_005",
      "transfer_readiness": "48_hours"
    }
  ]
}
```

### 2.6 Integration Points

- **Input from Module 1:** Medicine availability (if critical drug out of stock → length of stay changes)
- **Input from Module 3:** Staff availability (low staffing → discharge delays → higher occupancy)
- **Output to Module 4:** Occupancy predictions (trigger redistribution if one PHC overloaded, neighbor has capacity)
- **Output to Dashboard:** Bed availability heatmap, admission trends

### 2.7 Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Data storage | TimescaleDB (PostgreSQL extension) | Optimized for time-series (admission timestamps) |
| Feature pipeline | Python (pandas, scikit-learn) | Standard ML preprocessing |
| Model training | XGBoost + scikit-learn SVM | Gradient boosting for admissions, SVM for LOS classification |
| Batch predictions | Apache Airflow | Daily occupancy forecast generation |
| Dashboard | Streamlit | Real-time bed availability visualization |
| Serving | FastAPI + Redis | High-performance API for live bed queries |

---

## Module 3: Medical Personnel Attendance & Scheduling

### 3.1 Problem Statement (Module-Level)
Track medical staff (doctors, nurses, pharmacists) attendance in real time and flag staffing gaps that impact facility operations. Alert when staffing levels fall below minimum thresholds, affecting service capacity.

### 3.2 Data Inputs

**Source 1: Attendance Records**
- Daily check-in/check-out logs per staff member
- Scheduled shift (morning/evening/night)
- Actual hours worked
- Absences, leaves (sick, planned, unplanned)
- On-call duty availability

**Source 2: Staffing Requirements**
- Minimum staff required per shift (by role: doctor, nurse, pharmacist)
- Minimum staffing ratio per bed (e.g., 1 nurse per 5 beds)
- Staff-to-patient ratio during emergencies
- Cross-functional capability (can a nurse administer certain tasks if doctor unavailable?)

**Source 3: Facility Demand (from Modules 1 & 2)**
- Current bed occupancy (from Module 2)
- Medicine dispensing load (from Module 1)
- OPD patient footfall

**Source 4: Historical Patterns**
- Staff absence patterns (who's frequently absent? On what days?)
- Seasonal leave usage (holiday season, monsoon)
- Staff performance metrics (if available)

### 3.3 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Data Ingestion Layer                                         │
├─────────────────────────────────────────────────────────────┤
│ • Biometric/RFID attendance system (real-time clocking)      │
│ • Manual attendance sheets (backup/validation)               │
│ • Leave management system (approved absences)                │
│ • Shift scheduling database                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Data Validation & Reconciliation                             │
├─────────────────────────────────────────────────────────────┤
│ • Match biometric records with leave approvals               │
│ • Detect anomalies (logged in 2 places at once)              │
│ • Reconcile time zone differences (multi-site facilities)    │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Feature Engineering                                          │
├─────────────────────────────────────────────────────────────┤
│ Real-time metrics (updated every shift):                     │
│  • Current staff on duty (by role)                           │
│  • Staffing % of minimum required                            │
│  • Staff-to-bed ratio vs. target                             │
│  • Staff-to-patient ratio vs. capacity                       │
│                                                              │
│ Historical patterns:                                         │
│  • Average absence rate (%)                                  │
│  • Day-of-week absence probability                           │
│  • Seasonal absence trends                                   │
│  • Staff member reliability score (0-1)                      │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ Threshold Store (for alert generation)                       │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Logic Architecture (Rule-Based, No ML Model)

**Reason:** Staff scheduling is rule-driven (compliance, regulations). Simple threshold-based alerts are more interpretable and maintainable than ML models.

```
┌────────────────────────────────────────────────────────────┐
│ Real-Time Staffing Check (triggered every hour)             │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ For each PHC:                                               │
│   current_staff = {count_doctors, count_nurses, count_pharm}│
│   min_required = get_minimum_staffing_rules(phc_id)         │
│                                                             │
│   For each role in [doctor, nurse, pharmacist]:             │
│     if current_staff[role] < min_required[role]:            │
│       Alert.create(                                         │
│         severity = calculate_severity(                      │
│           shortfall=min_required[role]-current_staff[role], │
│           current_bed_occupancy,                            │
│           current_patient_load                              │
│         ),                                                  │
│         message = f"{role} shortage: {shortfall} needed"    │
│       )                                                     │
│     end if                                                  │
│   end for                                                   │
│                                                             │
│   # Cross-functional checks (if applicable)                 │
│   if current_staff[doctor] < min_required[doctor] / 2:      │
│     Alert.create(severity='CRITICAL', ...)                 │
│   end if                                                    │
│                                                             │
│ end for                                                     │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Severity Scoring Logic:**
```python
def calculate_severity(shortfall, bed_occupancy, patient_load):
    base_score = shortfall / min_required  # 0-1 scale
    
    # Adjust based on demand (occupancy % from Module 2)
    if bed_occupancy > 85:
        base_score *= 1.5  # Higher occupancy = higher risk
    
    # Adjust based on patient load (footfall from Module 1)
    if patient_load > historical_avg * 1.2:
        base_score *= 1.3  # Higher load = higher risk
    
    if base_score > 0.8:
        return 'CRITICAL'
    elif base_score > 0.5:
        return 'HIGH'
    elif base_score > 0.2:
        return 'MEDIUM'
    else:
        return 'LOW'
```

### 3.5 Outputs

**Output 1: Real-Time Staffing Dashboard**
```json
{
  "phc_id": "PHC_001",
  "timestamp": "2024-08-30T14:30:00Z",
  "current_shift": "MORNING",
  "staffing_status": {
    "doctors": {
      "required": 3,
      "current": 2,
      "shortfall": 1,
      "status": "ALERT"
    },
    "nurses": {
      "required": 8,
      "current": 7,
      "shortfall": 1,
      "status": "CAUTION"
    },
    "pharmacists": {
      "required": 2,
      "current": 2,
      "shortfall": 0,
      "status": "OK"
    }
  },
  "overall_alert_level": "HIGH",
  "reason": "Doctor shortage during high occupancy (89%)"
}
```

**Output 2: Staffing Alert**
```json
{
  "alert_id": "ALERT_PHC001_20240830_001",
  "phc_id": "PHC_001",
  "timestamp": "2024-08-30T14:35:00Z",
  "severity": "HIGH",
  "role_affected": "doctor",
  "shortfall": 1,
  "trigger_reason": "Staff absence + High bed occupancy (89%) + Patient load 120% of average",
  "recommendations": [
    "Call on-call doctor from backup list",
    "Request visiting doctor from district hospital",
    "Defer non-emergency procedures until evening shift"
  ],
  "estimated_resolution_time": "2 hours"
}
```

**Output 3: Shift Coverage Prediction (for next 24 hours)**
```json
{
  "phc_id": "PHC_001",
  "forecast_date": "2024-08-31",
  "predicted_coverage": [
    {
      "shift": "MORNING",
      "predicted_doctors": 2,
      "required": 3,
      "confidence": 0.85,
      "alert_expected": true
    },
    {
      "shift": "EVENING",
      "predicted_doctors": 3,
      "required": 3,
      "confidence": 0.75,
      "alert_expected": false
    }
  ]
}
```

### 3.6 Integration Points

- **Input from Module 2:** Bed occupancy forecast (to adjust staffing requirements dynamically)
- **Input from Module 1:** Medicine dispensing load (affects pharmacist workload)
- **Output to Module 4:** Staffing constraints (affects redistribution recommendations — don't transfer patients to understaffed facilities)
- **Output to Dashboard:** Real-time staffing heatmap

### 3.7 Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Attendance data | Biometric system API or CSV ingestion | Real-time or batch integration |
| Rule engine | Python (conditional logic) | No ML needed, pure business rules |
| Alert generation | Python + Celery (task queue) | Asynchronous alert dispatch |
| Notification | Twilio SMS + email + in-app | Multi-channel alerts to supervisors |
| Dashboard | Streamlit | Real-time staffing heatmap |

---

## Module 4: Cross-District Redistribution Engine

### 4.1 Problem Statement (Module-Level)
Recommend automated resource transfers (medicines, patient transfers) between PHC facilities to optimize resource utilization and prevent stock-outs or bed shortages during emergencies.

### 4.2 Data Inputs

**Source 1: Facility Network Topology**
- PHC locations (latitude, longitude)
- Distance matrix between facilities (km)
- Travel time estimates (by road, weather, time of day)
- Transport availability (vehicle capacity, frequency)

**Source 2: Resource States (Real-Time)**
- From Module 1: Medicine stock levels, stockout alerts per facility
- From Module 2: Bed availability, occupancy forecasts per facility
- From Module 3: Staff availability per facility

**Source 3: Resource Transfer Constraints**
- Medicine shelf-life (can't transfer expired stock)
- Storage capacity at receiving facility
- Transport cost (if applicable)
- Approval workflows (need supervisor sign-off?)
- Regulatory constraints (if any drugs can't cross district boundaries)

**Source 4: Historical Transfer Data**
- Past transfer frequency between facility pairs
- Success rate (did transferred stock arrive on time?)
- Average transfer turnaround time

### 4.3 Data Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Input Aggregation                                            │
├──────────────────────────────────────────────────────────────┤
│ • Module 1 alerts: {phc_id, drug_id, stockout_risk, urgency} │
│ • Module 2 alerts: {phc_id, occupancy_forecast, capacity_gap}│
│ • Module 3 alerts: {phc_id, staff_available, workload}       │
│ • Facility DB: {phc_id, location, capacity, transport_time}  │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│ Graph Construction                                            │
├──────────────────────────────────────────────────────────────┤
│ Build facility network:                                       │
│  - Nodes: Each PHC                                            │
│  - Edges: (PHC_i, PHC_j) with weight = transfer_time_ij      │
│  - Node attributes: stock_levels, bed_availability, staffing │
│                                                               │
│ Constraint propagation:                                       │
│  - Filter out transfer pairs with regulatory constraints     │
│  - Remove edges where receiving facility has no capacity     │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│ Optimization/Matching Engine                                  │
├──────────────────────────────────────────────────────────────┤
│ (See next section for algorithm details)                      │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│ Recommendation Output                                         │
└──────────────────────────────────────────────────────────────┘
```

### 4.4 Algorithm Architecture

**Greedy Nearest-Neighbor Matching (Fast, Hackathon-Friendly)**

**Why greedy over optimal solvers:**
- Optimal solvers (mixed-integer programming) require hours of computation for >50 facilities
- Greedy algorithm produces **good enough** solutions in seconds
- Judges reward working demo + fast response time over theoretical optimality

```python
def generate_redistribution_recommendations():
    """
    Generate transfer recommendations for all resources
    across the PHC network.
    """
    
    # Step 1: Identify all facilities with CRITICAL stockouts
    critical_stockouts = []
    for phc in all_phcs:
        for drug in critical_drugs:
            if module1.stockout_risk[phc][drug] > 0.9:
                critical_stockouts.append({
                    'deficit_phc': phc,
                    'drug': drug,
                    'quantity_needed': calculate_shortfall(phc, drug),
                    'urgency': 'CRITICAL',
                    'deadline_hours': 24
                })
    
    # Step 2: For each critical deficit, find nearest surplus facility
    recommendations = []
    for deficit in critical_stockouts:
        # Find all facilities with surplus of this drug
        surplus_facilities = [
            phc for phc in all_phcs
            if phc != deficit['deficit_phc']
            and module1.stock[phc][deficit['drug']] > safety_threshold
        ]
        
        if not surplus_facilities:
            continue  # Can't fulfill this transfer
        
        # Sort by distance (nearest-neighbor)
        surplus_facilities.sort(
            key=lambda phc: distance_matrix[phc][deficit['deficit_phc']]
        )
        
        # Try top 3 candidates (in case of transfer failure)
        for supplier_phc in surplus_facilities[:3]:
            transfer_qty = min(
                module1.stock[supplier_phc][deficit['drug']] - safety_threshold,
                deficit['quantity_needed']
            )
            
            transfer_time = travel_time_matrix[supplier_phc][deficit['deficit_phc']]
            
            if transfer_time <= deficit['deadline_hours']:  # Feasible?
                recommendations.append({
                    'action': 'TRANSFER_MEDICINE',
                    'source_phc': supplier_phc,
                    'destination_phc': deficit['deficit_phc'],
                    'drug': deficit['drug'],
                    'quantity': transfer_qty,
                    'estimated_arrival_hours': transfer_time,
                    'priority': 'URGENT',
                    'confidence_score': 0.95
                })
                break  # Found feasible supplier, move to next deficit
    
    # Step 3: Identify overcrowded facilities (Module 2 > 85% occupancy)
    overcrowded = [
        phc for phc in all_phcs
        if module2.occupancy_forecast[phc][day_1] > 0.85
    ]
    
    # Step 4: For each overcrowded facility, find underutilized neighbors
    for source_phc in overcrowded:
        # Get transfer candidates (patients who can safely travel)
        transferable_patients = [
            p for p in module2.current_patients[source_phc]
            if p.condition in ['recovery', 'stable'] 
            and p.days_admitted > 2  # Don't move acute cases
        ]
        
        # Find nearest facility with bed capacity
        nearby_phcs = sorted(
            [phc for phc in all_phcs if phc != source_phc],
            key=lambda phc: distance_matrix[phc][source_phc]
        )
        
        for target_phc in nearby_phcs:
            available_beds = module2.available_beds_forecast[target_phc][day_1]
            transferable_count = len(transferable_patients)
            
            if available_beds >= transferable_count and \
               module3.staffing_alert[target_phc] != 'CRITICAL':
                
                recommendations.append({
                    'action': 'TRANSFER_PATIENT',
                    'source_phc': source_phc,
                    'destination_phc': target_phc,
                    'patient_count': min(transferable_count, available_beds),
                    'patient_ids': [p.id for p in transferable_patients[:available_beds]],
                    'transfer_time_hours': travel_time_matrix[source_phc][target_phc],
                    'priority': 'HIGH',
                    'confidence_score': 0.88
                })
                break  # Found target, move to next overcrowded facility
    
    return recommendations
```

**Alternative: Linear Sum Assignment (if 5-10 facilities)**

```python
from scipy.optimize import linear_sum_assignment

def optimized_medicine_redistribution():
    """
    Use Hungarian algorithm for minimum-cost matching
    (only if number of facilities < 20).
    """
    
    # Cost matrix: C[i][j] = cost of transferring from facility i to j
    # Cost = distance + urgency penalty + time penalty
    cost_matrix = np.zeros((n_facilities, n_facilities))
    
    for i in range(n_facilities):
        for j in range(n_facilities):
            if i == j:
                cost_matrix[i][j] = 1e9  # Can't transfer to self
            else:
                distance_cost = distance_matrix[i][j]
                urgency_penalty = max(0, urgency[j] - urgency[i]) * 100
                feasibility_cost = 1e9 if not is_feasible(i, j) else 0
                
                cost_matrix[i][j] = distance_cost + urgency_penalty + feasibility_cost
    
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    
    # Extract recommendations from assignment
    recommendations = []
    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r][c] < 1e9:  # Valid assignment
            recommendations.append({
                'source': r,
                'destination': c,
                'cost': cost_matrix[r][c]
            })
    
    return recommendations
```

### 4.5 Outputs

**Output 1: Medicine Redistribution Recommendation**
```json
{
  "recommendation_id": "REDIS_MED_001",
  "timestamp": "2024-08-30T14:40:00Z",
  "action": "TRANSFER_MEDICINE",
  "source_phc": "PHC_005",
  "destination_phc": "PHC_001",
  "drug": "ASPIRIN_500MG",
  "quantity": 300,
  "urgency": "CRITICAL",
  "estimated_transfer_time_hours": 2.5,
  "confidence_score": 0.95,
  "rationale": "PHC_001 predicted stockout in 2.8 days; PHC_005 has surplus",
  "approval_required": true,
  "approval_contact": "District_Supervisor_ID_123"
}
```

**Output 2: Patient Transfer Recommendation**
```json
{
  "recommendation_id": "REDIS_PAT_001",
  "timestamp": "2024-08-30T14:41:00Z",
  "action": "TRANSFER_PATIENT",
  "source_phc": "PHC_001",
  "destination_phc": "PHC_005",
  "patient_count": 3,
  "patient_ids": ["P_001", "P_002", "P_003"],
  "transfer_time_hours": 1.5,
  "urgency": "HIGH",
  "reason": "PHC_001 occupancy forecast: 94% on day 4; PHC_005 has 15 available beds",
  "priority_patients": [
    {
      "patient_id": "P_001",
      "condition": "pneumonia_recovery",
      "transfer_readiness": "immediate"
    }
  ],
  "confidence_score": 0.88
}
```

**Output 3: Daily Redistribution Summary Report**
```json
{
  "report_date": "2024-08-30",
  "total_recommendations": 14,
  "by_type": {
    "medicine_transfers": 8,
    "patient_transfers": 4,
    "staffing_reassignments": 2
  },
  "by_urgency": {
    "CRITICAL": 3,
    "HIGH": 7,
    "MEDIUM": 4
  },
  "critical_issues_identified": [
    "PHC_001 aspirin stockout predicted in 48 hours",
    "PHC_003 bed occupancy 94% by day 4"
  ],
  "estimated_impact": {
    "stockouts_prevented": 2,
    "patient_waits_reduced": 8,
    "resource_utilization_improved": "12%"
  }
}
```

### 4.6 Integration Points

- **Input from Module 1:** Stockout alerts, transfer quantities
- **Input from Module 2:** Bed availability forecasts, patient transfer candidates
- **Input from Module 3:** Staff availability (don't transfer to understaffed facilities)
- **Output to Dashboard:** Redistribution recommendations, approval workflow
- **Output to Module 5:** Audit trail for federation consensus (if implementing federated approval)

### 4.7 Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Graph DB | Neo4j or NetworkX (Python) | Efficient network/facility representation |
| Optimization | SciPy (linear_sum_assignment) + custom greedy | Fast, no external solver needed |
| API | FastAPI | Real-time recommendation serving |
| Workflow | Celery + Redis | Queue transfer approvals asynchronously |
| Audit log | PostgreSQL | Track all recommendations + approvals |

---

## Module 5: Federated Learning & Privacy Architecture

### 5.1 Problem Statement (Module-Level)
Enable predictive model training and sharing across BRICS nations' health systems **without centralizing sensitive patient data**. Each nation maintains data sovereignty while collaborating on model improvement.

### 5.2 Architecture Overview

**Traditional (Non-Federated) Approach [❌ Not acceptable]**
```
Country A Hospital Data ──┐
Country B Hospital Data ──┼──→ Centralized Server ──→ Train Model ──→ Deploy
Country C Hospital Data ──┘

❌ Privacy violation: Raw patient data crosses borders
❌ Compliance issue: Violates GDPR, local data protection laws
❌ Security risk: Centralized honeypot for attackers
```

**Federated Learning Approach [✅ Recommended]**
```
Country A: Local model training on local data ──┐
                                                │
Country B: Local model training on local data ──┼──→ Federate model parameters
                                                │   (not data)
Country C: Local model training on local data ──┘
                    │
                    ▼
            [Aggregation Server]
              (Average weights)
                    │
                    ▼
        Improved global model → Redeploy to all countries
        
✅ Data never leaves country
✅ Privacy-preserving
✅ Compliant with GDPR, India's DPDP Act
```

### 5.3 Federated Learning Framework: Fed-BioMed

**Why Fed-BioMed:**
- Already used by real hospital networks (Unicancer, cancer centers)
- Production-ready open-source (github.com/fedbiomed/fedbiomed)
- Supports privacy-preserving techniques (differential privacy, secure aggregation)
- Built for healthcare (HIPAA-aware)

**Architecture:**

```
┌──────────────────────────────────────────────────────────────────┐
│ COUNTRY A (India) PHC Network                                    │
├──────────────────────────────────────────────────────────────────┤
│ ┌────────────────────────────────────────────────────────────┐   │
│ │ Fed-BioMed Node (Local)                                    │   │
│ │ ├─ Data: Local HMIS database (never exported)              │   │
│ │ ├─ Model Training: XGBoost on local data only              │   │
│ │ └─ Contribution: Model weights only (no raw data)          │   │
│ └────────────────────────────────────────────────────────────┘   │
└────────────────┬───────────────────────────────────────────────────┘
                 │
                 │ Model weights (encrypted)
                 ▼
         [Aggregation Server]
         (Kubernetes cluster)
         - Secure aggregation
         - Differential privacy
         - Version control
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌────────┐ ┌────────┐ ┌────────┐
│Country │ │Country │ │Country │
│   B    │ │   C    │ │   D    │
│(Brazil)│ │(China) │ │(Russia)│
└────────┘ └────────┘ └────────┘
(Similar Fed-BioMed nodes)
```

### 5.4 Federation Components

**Component 1: Local Training Node (Per-Country)**
```python
# Pseudo-code: Running at each country's PHC network

from fedbiomed.common.data import DataManager
from fedbiomed.fedbiomed_fedavg import FedAvgClient
import xgboost as xgb

class HealthcareFederatedNode:
    def __init__(self, country_id, local_data_path):
        self.country_id = country_id
        self.data_manager = DataManager(local_data_path)
        
    def train_local_model(self, global_model_weights, epochs=10):
        """
        Train model on local data, starting from global weights.
        Never export raw data.
        """
        # Load local training data (stays local)
        X_local, y_local = self.data_manager.get_training_data()
        
        # Initialize model with global weights
        local_model = xgb.XGBRegressor()
        local_model.load_model(global_model_weights)
        
        # Train on local data only
        local_model.fit(X_local, y_local, epochs=epochs)
        
        # Extract model weights
        local_weights = local_model.get_booster().get_dump()
        
        return {
            'country_id': self.country_id,
            'weights': local_weights,
            'training_samples': len(X_local),
            'loss': local_model.evals_result()['validation']['rmse'][-1]
        }
```

**Component 2: Aggregation Server (Central)**
```python
# Pseudo-code: Running at central aggregation point (cloud)

import numpy as np
from fedbiomed.fedbiomed_fedavg import FedAvgAggregator

class FederatedModelAggregator:
    def __init__(self):
        self.aggregator = FedAvgAggregator()
        self.round_num = 0
        
    def aggregate_weights(self, weight_updates_from_countries):
        """
        Average weights from all countries (Federated Averaging).
        Optionally add differential privacy noise.
        """
        
        # Weight averaging (importance-weighted by sample count)
        sample_counts = [update['training_samples'] for update in weight_updates_from_countries]
        total_samples = sum(sample_counts)
        
        aggregated_weights = None
        for update in weight_updates_from_countries:
            weight = update['training_samples'] / total_samples
            
            if aggregated_weights is None:
                aggregated_weights = weight * update['weights']
            else:
                aggregated_weights += weight * update['weights']
        
        # Optional: Add differential privacy noise
        noise = np.random.laplace(0, scale=0.1, size=aggregated_weights.shape)
        aggregated_weights_private = aggregated_weights + noise
        
        self.round_num += 1
        return {
            'round': self.round_num,
            'aggregated_weights': aggregated_weights_private,
            'countries_participated': len(weight_updates_from_countries),
            'privacy_budget_used': 0.05  # ε = 0.05 (strong privacy)
        }
```

**Component 3: Communication Protocol (Encrypted)**
```
Country A                 Aggregation Server           Country B
    │                            │                        │
    ├────(Request weights)──────→│                        │
    │←─(Encrypted global model)──┤                        │
    │                            ├────(Request weights)───→
    │                            │←─(Encrypted update)─────┤
    │                            │
    ├─(Encrypted local update)──→│
    │                            ├─(Encrypted local update)→
    │←─(New global model)────────┤
    │
    [Repeat: 10-20 rounds until convergence]
    
Encryption: TLS 1.3 for transit + end-to-end encryption for weights
```

### 5.5 Privacy-Preserving Techniques

**Technique 1: Federated Averaging (Data never leaves)**
- Each country trains locally on own data
- Only model weights sent to server
- Weights averaged, no raw data exposed

**Technique 2: Differential Privacy**
```python
# Add noise to weights before aggregation
# Ensures individual patient privacy even if attacker sees model updates

def add_differential_privacy(weights, epsilon=0.1, delta=1e-5):
    """
    Add Laplace noise proportional to model gradient sensitivity.
    epsilon: privacy budget (lower = more private, but noisier model)
    delta: failure probability
    """
    sensitivity = 1.0  # Max change from one sample
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, weights.shape)
    return weights + noise
```

**Technique 3: Secure Aggregation**
```
Multiple servers jointly aggregate weights such that:
  - No single server sees individual updates
  - Only final aggregated model revealed
  - Requires cryptographic protocols (threshold secret sharing)
  
Example: Server 1 gets Country A weights, Server 2 gets Country B weights
Neither server alone can reconstruct full model.
```

### 5.6 Audit & Compliance

**Data Governance**
```json
{
  "federation_metadata": {
    "participating_countries": ["India", "Brazil", "Russia", "China", "South Africa"],
    "data_retention_policy": "Local only (never centralized)",
    "model_update_frequency": "Weekly (Sundays 2 AM UTC)",
    "privacy_certification": "GDPR + India DPDP Act + Local regulations"
  },
  "round_1": {
    "date": "2024-09-01",
    "countries_participated": 4,
    "weights_aggregated": true,
    "privacy_epsilon": 0.15,
    "model_improvement": "2.3% accuracy gain",
    "audit_log": [
      {
        "timestamp": "2024-09-01T02:00:00Z",
        "action": "weights_received",
        "country": "India",
        "samples": 50000,
        "status": "encrypted"
      },
      // ...more entries
    ]
  }
}
```

**Compliance Checklist**
- ✅ No personal health data (PHI) crosses borders
- ✅ Each country retains data sovereignty
- ✅ Differential privacy protects against re-identification
- ✅ Audit trail tracks all aggregation rounds
- ✅ Consent obtained from each country upfront
- ✅ Regular privacy audits by independent auditors

### 5.7 Outputs (Dashboard & Reports)

**Output 1: Federation Status Dashboard**
```json
{
  "federation_status": {
    "active_countries": 5,
    "total_participating_phcs": 450,
    "model_version": "global_v3.2",
    "last_aggregation": "2024-08-30T02:00:00Z",
    "next_aggregation": "2024-09-06T02:00:00Z",
    "global_model_performance": {
      "accuracy": 0.92,
      "precision": 0.87,
      "recall": 0.89,
      "improvement_vs_local": "3.5%"
    }
  },
  "participating_countries": [
    {
      "country": "India",
      "phcs_contributing": 120,
      "last_contribution": "2024-08-30T01:45:00Z",
      "local_model_accuracy": 0.90,
      "contribution_status": "HEALTHY"
    },
    // ... other countries
  ]
}
```

**Output 2: Federated Model Card**
```
Model Card: PHC Medicine Demand Forecaster v3.2 (Federated)

Training Data:
  - Sources: 5 countries, 450+ PHCs
  - Total samples: 500K+ (across all countries)
  - Timespan: Jan 2022 - Aug 2024
  - Note: Each country trains locally; only weights aggregated

Model Performance (on private test sets per country):
  - India: RMSE 12.5 units, MAE 8.2 units
  - Brazil: RMSE 14.1 units, MAE 9.7 units
  - Russia: RMSE 11.8 units, MAE 7.9 units
  - [Aggregated global performance: RMSE 13.1, MAE 8.6]

Privacy Guarantees:
  - Differential privacy: ε=0.1, δ=1e-5
  - No centralized data: All raw data stays local
  - Secure aggregation: Used for model weights

Known Limitations:
  - Global model may not fit local conditions perfectly
  - Requires local fine-tuning for new PHCs
  - Aggregation adds privacy noise (slight accuracy drop)

Recommendations:
  - Use global model as baseline, fine-tune locally
  - Reretrain monthly with new local data
  - Monitor for model drift in each country
```

### 5.8 Deployment Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Kubernetes Cluster (Cloud: AWS/Azure/GCP)                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Namespace: fedbiomed                                            │
│ ├─ Pod: aggregation-server (Flask API)                          │
│ │   - Receives weights from countries                           │
│ │   - Performs secure aggregation                               │
│ │   - Publishes new global model                                │
│ │                                                                │
│ ├─ Pod: privacy-engine (Differential Privacy service)           │
│ │   - Adds noise to aggregated weights                          │
│ │   - Tracks privacy budget (ε)                                 │
│ │                                                                │
│ ├─ Pod: audit-logger (Immutable audit trail)                    │
│ │   - Logs all aggregation rounds                               │
│ │   - Records which countries participated                      │
│ │   - Stores model versions                                     │
│ │                                                                │
│ ├─ PersistentVolume: model-storage                              │
│ │   - Stores model weights (versions v1, v2, v3, ...)           │
│ │   - Encrypted at rest                                         │
│ │                                                                │
│ └─ Service: federation-api                                      │
│     - Exposes /aggregate, /get_model, /status endpoints        │
│     - TLS 1.3 + API key authentication                          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 5.9 Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Federated Learning | Fed-BioMed (open-source) | Hospital-tested, privacy-focused |
| Model aggregation | Custom Python + NumPy | Federated averaging with privacy |
| Communication | gRPC + TLS 1.3 | Encrypted, efficient RPC |
| Differential Privacy | Opacus / TensorFlow Privacy | Industry-standard privacy library |
| Audit logging | PostgreSQL + immutable schema | Compliance + audit trail |
| Orchestration | Kubernetes + Helm | Multi-country deployment |
| Monitoring | Prometheus + ELK Stack | Track federation health |

### 5.10 Hackathon Scope (What You Actually Build)

**For a 36-hour hackathon:**

✅ **Build:**
- Aggregation server skeleton (Flask API)
- Federated averaging algorithm (working code)
- Privacy-aware weight aggregation (add DP noise)
- Architecture diagram (data flow across countries)
- Documentation + deployment guide

⚠️ **Simulate/Mock:**
- Fed-BioMed integration (show how it would connect, but use simplified local training)
- Kubernetes deployment (document, don't deploy live)
- Secure aggregation (explain cryptography, use simplified version)
- Multiple real country nodes (simulate with 3-4 mock nodes)

❌ **Don't build:**
- Real encrypted communication (use HTTPS mock)
- Full Fed-BioMed installation (too time-consuming)
- Real differential privacy library integration (show calculation code instead)
- Multi-cloud deployment (run locally on Docker)

**Deliverable:**
- Working Python federation server
- Demo: "Run aggregation on 3 mock country nodes, show model improves"
- Slide: "Here's how this scales to 5 BRICS countries with Fed-BioMed"
- Docs: "Deployment guide + privacy guarantees"

---

## End-to-End Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ Real-Time Data Ingestion (Kafka/API polling)                       │
├─────────────────────────────────────────────────────────────────────┤
│ ERP → Medicine consumption  │ HMIS → Admissions/discharges         │
│ Biometric → Staff check-ins │ Weather API → Outbreak alerts        │
└────────┬────────────────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────────────┐
│ Data Lake (TimescaleDB / PostgreSQL)                                │
│ ├─ medicine_transactions (timestamp, phc_id, drug_id, qty)          │
│ ├─ bed_movements (timestamp, phc_id, patient_id, in/out)            │
│ ├─ staff_attendance (timestamp, phc_id, role, present/absent)       │
│ └─ external_signals (timestamp, alert_type, severity)               │
└────────┬────────────────────────────────────────────────────────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    │          │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Module1│ │Module2│ │Module3│ │Module4│
│ (Med) │ │ (Bed) │ │(Staff)│ │(Redis)│
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
    │          │          │          │
    └──────────┼──────────┼──────────┘
               │          │
        ┌──────▼──────────▼──────┐
        │ Recommendation Engine  │
        │ (Aggregate all alerts) │
        └──────┬─────────────────┘
               │
        ┌──────▼──────────┐
        │  Module 5       │
        │ (Federated Agg) │
        └──────┬──────────┘
               │
        ┌──────▼──────────────────┐
        │ API Gateway + Cache     │
        │ (FastAPI + Redis)       │
        └──────┬──────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼────┐ ┌──▼────┐ ┌──▼───┐
│Dashbrd │ │Mobile │ │ Email│
│(Strm.) │ │(React)│ │Alert │
└────────┘ └───────┘ └──────┘
```

---

## Summary: Module Dependencies & Build Order

```
Suggested hackathon build sequence:

Week 1-2 Planning & Setup:
  ├─ Set up data pipeline (Kafka + TimescaleDB)
  ├─ Create mock datasets (1 year history for 10 PHCs)
  └─ Define API contracts between modules

Day 1 (Hackathon):
  ├─ Start Module 1 & 2 in parallel (independent ML work)
  ├─ Module 3: Simple rule-based stubs
  └─ Module 4: Greedy algorithm (no optimization solver)

Day 1 PM:
  ├─ Integrate Modules 1-4 to single API
  ├─ Build Streamlit dashboard (live forecasts + alerts)
  └─ Module 5: Write architecture slide + Fed-BioMed reference

Day 2:
  ├─ End-to-end demo (data → forecasts → recommendations)
  ├─ Polish UI/UX
  ├─ Prepare pitch (problem → solution → impact)
  └─ Final testing & documentation

Judging:
  • Show working demo (Modules 1-4 live)
  • Explain Module 5 architecture + roadmap
  • Answer questions on feasibility + scalability
```

---

**This architecture is production-ready but simplified for hackathon constraints. All 5 modules work independently and can be extended post-hackathon.**
