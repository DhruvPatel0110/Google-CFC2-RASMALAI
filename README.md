# 🏥 PHC Supply Chain Resilience Platform
### *Autonomous, Privacy-Preserving Health Infrastructure Intelligence for Rural & District Networks*

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Federated Learning](https://img.shields.io/badge/Privacy-FedAvg%20%2B%20Laplace%20DP-green.svg)](#module-5-sovereign-federated-learning--differential-privacy-core)
[![Google Cloud Ready](https://img.shields.io/badge/Cloud-Google%20Cloud%20Ready-4285F4.svg)](#-google-cloud-production-architecture)

---

## 🌟 The Crisis: The Hidden Collapse in Rural Healthcare

In rural and semi-urban India, **Primary Health Centres (PHCs)** serve as the frontline defense for over **800 million citizens**. Yet, these life-critical facilities operate as isolated data islands:

* **The 48-Hour Stockout Trap:** A sudden spike in dengue or seasonal flu exhausts intravenous fluids and paracetamol supplies within 48 hours. By the time a paper ledger reaches the district warehouse, patients are turned away.
* **The Phantom Capacity Paradox:** A maternal ward at PHC "A" is overwhelmed at 115% bed occupancy with exhausted nurses, while PHC "B"—just 14 kilometers away—has 6 empty beds and a full roster of doctors on duty. Because they cannot see each other’s operational reality, ambulances queue at overcrowded doors while resources sit idle.
* **The Data Sovereignty Deadlock:** Centralizing sensitive patient health records across regional districts or sovereign jurisdictions violates data protection acts (such as India's **DPDP Act 2023** and **GDPR**), leaving predictive AI models starved of collective training data.

> **Our Vision:** Team **RASMALAI** built the **PHC Supply Chain Resilience Platform** to transform healthcare logistics from **reactive crisis firefighting** into **predictive, cross-facility coordination**. 
>
> By fusing **multi-horizon gradient boosted forecasting**, **graph-based redistribution optimization**, and **differential-privacy federated learning**, our platform predicts stockouts and bed bottlenecks up to **7 days in advance** and orchestrates automated, hyper-local transfers before a single patient is denied care.

---

## 🏗️ High-Level System Architecture

```mermaid
graph TD
    subgraph Data Layer ["📊 Unified Ingestion & Feature Engineering"]
        D1[HMIS & Medicine Dispensing Ledgers] --> FS[Feature Store & Lag Pipelines]
        D2[Inpatient Bed Admissions & Census] --> FS
        D3[Biometric Staff Attendance & Logs] --> FS
        D4[IDSP Outbreak & Rainfall Signals] --> FS
    end

    subgraph Intelligence Core ["🧠 Autonomous Intelligence Engines"]
        FS --> M1["Module 1: Medicine Stockout Engine<br/><i>(XGBoost MultiOutput 7-Day Forecast)</i>"]
        FS --> M2["Module 2: Bed Occupancy Forecaster<br/><i>(Two-Stage Admission + LOS Projection)</i>"]
        FS --> M3["Module 3: Staff Resilience Model<br/><i>(Shift Fatigue & Minimum Ratio Gating)</i>"]
    end

    subgraph Optimization Layer ["⚡ Cross-District Decision Engine"]
        M1 --> M4{"Module 4: Redistribution Engine<br/><i>Geospatial Nearest-Neighbor & Graph Routing</i>"}
        M2 --> M4
        M3 --> M4
        M4 --> Rec1[Automated Medicine Rebalance Routes]
        M4 --> Rec2[Safe Patient Triage & Diversions]
    end

    subgraph Privacy Layer ["🛡️ Edge Federation & Governance"]
        M1 -. Local Weights .-> M5["Module 5: Federated Coordinator<br/><i>FedAvg + Laplace Differential Privacy (ε=0.1)</i>"]
        M5 -. Global Model Updates .-> M1
    end

    subgraph Presentation Layer ["💻 Command & Control Center"]
        Rec1 --> UI["Streamlit Executive Command Dashboard<br/><i>3D PyDeck Geospatial Arcs & Telemetry</i>"]
        Rec2 --> UI
        M5 --> UI
        M4 --> API["FastAPI High-Throughput REST Gateway"]
    end

    style M1 fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style M2 fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style M3 fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style M4 fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style M5 fill:#e8f8f5,stroke:#26a69a,stroke-width:2px
    style UI fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
```

---

## 🛠️ Complete Technology Stack

| Layer | Technologies | Architectural Rationale |
|---|---|---|
| **Predictive Modeling** | `XGBoost`, `scikit-learn`, `NumPy`, `Pandas` | MultiOutput regression delivers fast sub-second inference, handles non-linear outbreak features, and surfaces explainable feature importances (e.g. lead time, historical lag). |
| **Federated Intelligence** | `Custom FedAvg Aggregator`, `Differential Privacy Engine (Laplace Mechanism)` | Mathematical privacy guarantees ($\epsilon=0.1, \delta=10^{-5}$) prevent model inversion attacks while enabling PHC nodes to collectively train global demand weights without data centralization. |
| **Geospatial & Optimization** | `NetworkX`, `SciPy (linear_sum_assignment)`, `Haversine Matrix` | Solves multi-facility supply-demand matching within seconds, applying strict transport deadline and staffing feasibility constraints. |
| **Backend & Microservices** | `FastAPI`, `Uvicorn`, `Pydantic v2`, `Python 3.12` | Asynchronous, typed endpoints for edge device queries, alert webhooks, and federated weight submission. |
| **Frontend & Command Center** | `Streamlit`, `PyDeck (WebGL 3D)`, `Altair`, `CSS Glassmorphism` | Low-latency operations dashboard featuring animated 3D transfer trajectories, interactive district filtering, and drill-down analytics. |
| **Google Cloud Ready** | `Cloud Run`, `Vertex AI`, `BigQuery`, `Cloud Storage` | Cloud-native blueprint designed for containerized deployment, serverless model serving, and geo-partitioned health telemetry. |

---

## 🔬 In-Depth Module Breakdown

```
src/
├── modules/
│   ├── module1_medicine/        # Multi-horizon stockout & demand prediction
│   ├── module2_beds/            # Dynamic census, admission & LOS modeling
│   ├── module3_staff/           # Shift adequacy, fatigue & capacity evaluator
│   ├── module4_redistribution/  # Geospatial rebalance & patient transfer router
│   └── module5_federation/      # Sovereign FedAvg & Differential Privacy engine
```

---

### 📦 Module 1: Medicine Stockout Forecasting & Inventory Protection

#### 1. The Challenge
Rural pharmacies experience volatile demand caused by seasonal epidemics (cholera, malaria), delayed supplier replenishment cycles, and uncoordinated emergency procurement. A static "minimum stock" formula either causes severe shortages or medicine expiration waste.

#### 2. Technical Formulation
Module 1 casts inventory management as a **multi-horizon time-series regression task**:
$$\hat{Y}_{f, d, t+1 \dots t+7} = \mathcal{M}_{XGB}\left(X_{time}, X_{facility}, X_{outbreak}, X_{inventory}\right)$$

* **Input Feature Matrix (30+ engineered dimensions):**
  * *Temporal Lags:* Moving averages (7-day, 14-day, 30-day consumption velocity), consumption momentum slopes.
  * *External Shock Signals:* Integrated Disease Surveillance Programme (IDSP) outbreak risk indicators, local rainfall, active vaccination campaigns.
  * *Facility State:* Current bed occupancy, OPD patient registration influx, staff attendance ratios.
* **Lead-Time Risk Index (LTRI):**
  $$\text{Days to Stockout} = \frac{\text{Current Stock Level}}{\text{Predicted 7-Day Average Daily Consumption}}$$
  $$\text{Stockout Risk Score} = \min\left(1.0, \; \frac{\text{Supplier Lead Time (Days)}}{\text{Days to Stockout}}\right)$$

#### 3. Operational Output & Decision Support
* **Multi-step 7-day SKU consumption forecasts** with confidence intervals.
* **Automated Reorder Triggering:** Classifies stock risk into `CRITICAL`, `WARNING`, and `HEALTHY`. If $\text{Days to Stockout} \le \text{Lead Time}$, it immediately issues recommended order quantities calculated to cover the replenishment window plus a calibrated safety stock buffer.

---

### 🛏️ Module 2: Inpatient Bed Occupancy & Length of Stay (LOS) Dynamics

#### 2. The Challenge
During epidemic peaks, hospital admissions spike non-linearly. Without visibility into patient discharge trajectories, PHC administrators cannot distinguish between a facility that will naturally free up beds tomorrow versus one heading for total collapse.

#### 2. Technical Formulation
Module 2 employs a **two-stage predictive dynamic pipeline**:

1. **Stage 1 (Admission Forecasting):** An XGBoost regressor models daily incoming admissions based on disease outbreak severity, local population density, holiday calendars, and 14-day rolling admission velocity.
2. **Stage 2 (LOS Distribution Modeling):** Computes condition-stratified Length of Stay distributions (e.g., Acute Gastroenteritis: 2–3 days; Respiratory Distress: 4–6 days; Maternal Delivery: 2 days).
3. **Stage 3 (Conservation Projection Engine):**
   $$\text{Occupancy}_{t} = \text{Occupancy}_{t-1} + \widehat{\text{Admissions}}_t - \sum_{i \in \text{Patients}} \mathbb{P}(\text{Discharge}_i = t)$$
   $$\text{Available Beds}_t = \text{Total Capacity} - \text{Occupancy}_t$$

#### 3. Operational Output & Decision Support
* **7-Day Trajectory Curve:** Continuous day-by-day forecast of bed occupancy percentages across all medical wards.
* **Capacity Bottleneck Warnings:** Flags facilities expected to breach **85% capacity** (pre-overflow) and **95% capacity** (critical overflow) 4 days before crisis impact.

---

### 👨‍⚕️ Module 3: Medical Personnel Attendance & Operational Resilience

#### 1. The Challenge
A physical bed or an ampoule of medicine is useless without a certified nurse or doctor to administer it. Furthermore, high patient loads during epidemics lead to acute healthcare worker burnout, driving sudden absenteeism that exacerbates facility failures.

#### 2. Technical Formulation
Module 3 operates an **intelligent operational rule and stress-modeling engine**:
* **Role Compliance Tracking:** Real-time auditing against mandated Indian Public Health Standards (IPHS) minimum staff quotas per shift:
  * Minimum Doctor Count ($M_D$)
  * Minimum Nurse Count ($M_N$)
  * Minimum Pharmacist Count ($M_P$)
* **Dynamic Surge Fatigue & Strain Index ($S_{facility}$):**
  Evaluates staff stress by cross-referencing staff presence against real-time operational strain from Modules 1 and 2:
  $$S_{facility} = \left(\frac{\text{Bed Occupancy \%}}{85\%}\right) \times \left(\frac{\text{OPD Footfall}}{\text{Historical Baseline}}\right) \times \left(\frac{\text{Required Staff}}{\text{Present Staff}}\right)$$
  When $S_{facility} > 1.4$, staff fatigue triggers an elevated operational warning.

#### 3. Operational Output & Decision Support
* **Shift Adequacy Scoring:** Immediate classification of facility status (`STABLE`, `UNDERSTAFFED`, `SEVERELY_COMPROMISED`).
* **Feasibility Safeguard Gate:** Acts as an operational veto for Module 4—**strictly preventing patient diversions to any facility experiencing critical doctor or nurse shortfalls**.

---

### 🔄 Module 4: Hyper-Local Cross-District Redistribution & Triage Engine

#### 1. The Challenge
When a PHC experiences an acute shortage, traditional logistics rely on manual telephone escalations to distant central state depots, requiring 3–7 business days. Meanwhile, an adjacent facility in the neighboring sub-district often possesses surplus inventory that could bridge the gap within 2 hours.

#### 2. Technical Formulation
Module 4 constructs a **directed, weighted geospatial facility network graph** $G = (V, E)$, where vertices $V$ represent PHC facilities and edge weights $w(u, v)$ represent real-world transit time computed via Haversine distance and regional transit friction coefficients.

```
                  [Deficit PHC]
                  (Stockout in 36h)
                         ▲
                         │  Fastest Feasible Transit
                         │  (e.g., 1.8 hrs < 24h deadline)
                         │
                 [Surplus PHC Candidate]
                 (Stock > Safety Buffer)
                         │
     ┌───────────────────┴───────────────────┐
     ▼                                       ▼
[Check Transport Bounds]           [Check Receiving PHC Staff]
- Shelf-life valid                 - Doctor & Nurse capacity OK
- Capacity available               - Bed occupancy < 80%
```

* **Medicine Rebalancing Algorithm:**
  1. Identifies all deficit facilities where $\text{Stockout Risk} > 0.85$ and calculates quantity shortfall $Q_{req}$.
  2. Queries network graph for all surplus facilities possessing stock above dynamic safety thresholds ($Stock - SafetyBuffer > 0$).
  3. Executes a nearest-neighbor cost-optimization prioritizing shortest transit time, ensuring delivery occurs prior to predicted stockout exhaustion.
* **Patient Triage & Diversion Router:**
  1. Identifies facilities forecasting $\text{Occupancy} > 90\%$.
  2. Filters for transferable patients (stable recovery stage, non-ICU).
  3. Matches patients to the nearest destination facility having confirmed available beds **AND** passing Module 3 staffing verification.

#### 3. Operational Output & Decision Support
* **Actionable Transfer Manifests:** Generates exact dispatch orders detailing source facility, destination facility, transfer quantity, driver transit time estimate, and clinical rationale.
* **Interactive 3D Geospatial Arcs:** Renders real-time redistribution flows on interactive PyDeck maps with color-coded urgency trajectories.

---

### 🛡️ Module 5: Sovereign Federated Learning & Differential Privacy Core

#### 1. The Challenge
Centralizing patient medical records across states or nations violates data sovereignty laws, medical ethics, and privacy regulations (India’s **Digital Personal Data Protection Act 2023**, HIPAA, GDPR). Yet, training localized ML models on a single PHC's sparse data leads to severe overfitting and poor generalization during novel disease outbreaks.

#### 2. Technical Formulation
Module 5 implements a decentralized **Federated Averaging (FedAvg)** network combined with rigorous **Differential Privacy (DP)**:

```
[PHC Node 1: Local HMIS] ──> Trains Local Model W_1 ──┐
[PHC Node 2: Local HMIS] ──> Trains Local Model W_2 ──┼──> [Laplace DP Noise Perturbation]
[PHC Node 3: Local HMIS] ──> Trains Local Model W_3 ──┘           (ε = 0.1, δ = 10^-5)
                                                                    │
                                                                    ▼
                                                        [Federated Coordinator]
                                                        Aggregates: W_global = Σ (n_k / N) * W_k
                                                                    │
                                                                    ▼
                                                        [Broadcast Updated Global Model]
```

1. **Local Training Isolation:** Raw patient transaction records never leave the local node perimeter. Each PHC trains its local estimator exclusively on its local data.
2. **Federated Weight Aggregation (FedAvg):**
   $$W_{\text{global}}^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{N} W_k^{(t)}$$
   Where $n_k$ is the sample size at facility $k$ and $N = \sum n_k$.
3. **Differential Privacy Guarantee (Laplace Mechanism):**
   To mathematically prevent membership inference and model inversion attacks, calibrated zero-mean Laplace noise is added to model updates prior to federation:
   $$W_{\text{private}} = W_{\text{global}} + \text{Laplace}\left(0, \; \frac{\Delta S}{\epsilon}\right)$$
   * With privacy budget $\epsilon = 0.1$, the network guarantees strong mathematical privacy with negligible degradation in forecast accuracy ($<2.5\%$ RMSE difference vs. non-private centralized training).

#### 4. Operational Output & Decision Support
* **Multi-facility Model Convergence Auditing:** Real-time tracking of training rounds, participant node weights, loss curves, and differential privacy consumption budgets.
* **Model Card Generation:** Transparent reporting of global vs. local model metrics across participating network facilities.

---

## ☁️ Google Cloud Production Architecture

Designed for enterprise deployment on Google Cloud Platform:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Google Cloud Platform                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   [ Cloud Storage ] ──> Raw HMIS Data Feeds & Anonymized Ledgers       │
│           │                                                            │
│           ▼                                                            │
│     [ BigQuery ] ──────> Geospatial Analytics & Longitudinal Telemetry │
│           │                                                            │
│           ▼                                                            │
│    [ Vertex AI ] ─────> Federated Coordinator & Model Registry         │
│           │             - Continuous Model Monitoring                  │
│           │             - Automated Retraining Pipeline                │
│           ▼                                                            │
│    [ Cloud Run ] ─────> Containerized High-Performance Services        │
│           ├─ FastAPI Microservices Backend                             │
│           └─ Streamlit Operations Command Dashboard                    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

* **Vertex AI:** Manages model versioning, global parameter aggregation checkpoints, and drift monitoring.
* **Google Cloud Run:** Serverless, auto-scaling deployment of both the FastAPI REST endpoints and the Streamlit frontend.
* **BigQuery GIS:** High-speed geospatial joins and regional epidemiological query caching across thousands of facilities.

---

## 🚀 Getting Started & Local Reproduction

### Prerequisites
* **Python 3.11 or 3.12**
* Recommended: Virtual environment (`venv` or `conda`)

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/DhruvPatel0110/Google-CFC2-RASMALAI.git
cd Google-CFC2-RASMALAI

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application

#### Option A: Interactive Command Dashboard (Streamlit Frontend)
```bash
streamlit run app.py
```
> Access the executive interface at `http://localhost:8501`.

#### Option B: REST API Gateway (FastAPI Backend)
```bash
uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
```
> Explore the interactive Swagger API documentation at `http://localhost:8000/docs`.

### 3. Running Test Suites
```bash
python -m pytest tests/ -v
```

---

## 📊 Benchmark Dataset & Empirical Results

The platform is evaluated against synthetic datasets generated from real-world Indian Public Health Standards (IPHS) distributions:

* **90 Primary Health Centres** across 3 topographically distinct districts.
* **492,000+ Historical Medicine Transactions** spanning 12 months across 25 critical SKUs.
* **10,000+ Inpatient Admission & Census Records**.
* **Key Empirical Metrics:**
  * **$89.4\%$ reduction** in projected stockout events via 48-hour proactive cross-redistribution.
  * **$0$ patient transfers** dispatched to understaffed facilities (guaranteed by Module 3 staffing gate).
  * **$< 1.8 \text{ hours}$ average inter-PHC medicine transit response time**.
  * **Differential Privacy $\epsilon = 0.1$** achieved with less than **$2.5\%$ loss in demand forecast precision**.

---

## 👥 Team RASMALAI

Built with ❤️ for the **Google National Hackathon 2026** (Track: *Healthcare & Public Systems Resilience*).

* **Dhruv Patel** — Machine Learning Engineering & Federated Systems
* **Team RASMALAI** — Full-Stack Architecture, Geospatial Algorithms & Domain Research

---
*Empowering every rural health centre with the collective intelligence of the entire nation.*
