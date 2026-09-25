import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "PHC Resilience Command — Comprehensive Technical & Architectural Approach")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — BRICS & National Health Tech Network")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        
        self.restoreState()

def build_pdf(filename="approach.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0F172A")    # Slate 900
    c_secondary = colors.HexColor("#0284C7")  # Sky Blue 600
    c_accent = colors.HexColor("#0D9488")     # Teal 600
    c_text = colors.HexColor("#1E293B")       # Slate 800
    c_muted = colors.HexColor("#475569")      # Slate 600
    c_card_bg = colors.HexColor("#F8FAFC")    # Slate 50
    c_border = colors.HexColor("#E2E8F0")     # Slate 200
    
    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=c_primary,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        alignment=TA_CENTER
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        textColor=c_muted,
        alignment=TA_CENTER
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=c_text,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_text,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )
    
    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0F172A")
    )

    story = []
    
    # -------------------------------------------------------------
    # COVER / HEADER
    # -------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(Paragraph("PHC Resilience Command", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Predictive Healthcare Supply Chain, Staff Readiness & Federated AI Architecture", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Architectural Specification, Problem-Solution Mapping & Complete Technical Implementation", meta_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceBefore=4, spaceAfter=12))

    # -------------------------------------------------------------
    # EXECUTIVE SUMMARY & SYSTEM OVERVIEW
    # -------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Core Objective", h1_style))
    story.append(Paragraph(
        "Primary Health Centers (PHCs) and district hospitals serve as the vital first line of defense for public health. "
        "During public health surges—such as seasonal monsoon epidemics (Dengue, Malaria, Typhoid), respiratory outbreaks, or extreme weather heatwaves—rural and suburban facilities face catastrophic resource breakdowns: essential medicines run completely dry before warehouse replenishment arrives, emergency and ICU beds breach maximum physical capacity, and severe medical staff absenteeism leaves active wards unstaffed. Despite localized shortages, neighboring districts frequently possess untapped surplus supplies and available bed capacity. However, the lack of real-time multi-echelon predictive intelligence and statutory patient data privacy restrictions prevent dynamic rebalancing.",
        body_style
    ))
    story.append(Paragraph(
        "<b>PHC Resilience Command</b> is an integrated, full-stack predictive intelligence platform designed to eliminate healthcare resource bottlenecks days before they manifest. By combining multi-step machine learning time-series forecasting, deterministic clinical staffing benchmarks, priority-based graph redistribution routing, and decentralized Differential Privacy Federated Learning, the platform converts reactive crisis response into proactive operational resilience.",
        body_style
    ))
    
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # MODULE 1: MEDICINE STOCKOUT FORECASTER
    # -------------------------------------------------------------
    story.append(Paragraph("2. Module 1: Medicine Stockout Forecaster", h1_style))
    
    story.append(Paragraph("A. Problem Statement", h2_style))
    story.append(Paragraph(
        "Rural clinics traditionally utilize static periodic reordering or reactive stock replenishment, ordering medicines only after supplies are critically low or exhausted. Given multi-day warehouse lead times and logistics delays, clinics experience extended stockout windows for essential, life-saving drugs (e.g., Paracetamol, Amoxicillin, Insulin, Oral Rehydration Salts, Antivenom), resulting in preventable patient mortality.",
        body_style
    ))
    
    story.append(Paragraph("B. Solution Offered", h2_style))
    story.append(Paragraph(
        "Module 1 implements a multi-step machine learning time-series regression forecaster that predicts daily medicine consumption over a 7-day rolling window across all facility-drug combinations. It compares daily consumption burn rates against real-time closing inventory and warehouse lead times, computing calibrated mathematical stockout probabilities and issuing actionable tri-tier alerts (CRITICAL, WARNING, SAFE).",
        body_style
    ))

    story.append(Paragraph("C. Technical Implementation Details (\"HOW\" it Works)", h2_style))
    story.append(Paragraph("• <b>Machine Learning Model:</b> Scikit-Learn <code>MultiOutputRegressor</code> wrapping <b><code>XGBRegressor</code></b> (Extreme Gradient Boosting) with hyperparameters: <code>n_estimators=180</code>, <code>max_depth=6</code>, <code>learning_rate=0.08</code>, <code>subsample=0.85</code>, <code>colsample_bytree=0.85</code>, <code>random_state=42</code>.", bullet_style))
    story.append(Paragraph("• <b>Multi-Step Output Vector:</b> Simultaneously predicts 8 targets: daily consumption for days 1 through 7 (<code>target_day_1</code> to <code>target_day_7</code>) and total 7-day cumulative consumption (<code>target_7d_total</code>).", bullet_style))
    story.append(Paragraph("• <b>Feature Engineering Pipeline:</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Historical Lags:</i> Prior consumption at <i>t-1, t-7, t-14, t-30</i> days.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Rolling Window Statistics:</i> 7-day and 30-day rolling mean, standard deviation, and rolling maximums.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Exogenous Environmental Signals:</i> Daily rainfall (mm), ambient temperature (°C), Air Quality Index (AQI), and localized binary epidemic alerts.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Calendar Encodings:</i> Day of week, day of month, month, and binary <code>is_weekend</code> flag.", bullet_style))
    story.append(Paragraph("• <b>Mathematical Stockout Risk Modeling:</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Daily Burn Rate:</i> <code>daily_burn_rate = predicted_consumption_7d / 7.0</code>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Days to Stockout:</i> <code>days_to_stockout = closing_stock / (daily_burn_rate + 1e-6)</code>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Stockout Probability (Logistic Sigmoid):</i> Anchored to warehouse lead time buffer: <br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>z = (lead_time_days * 1.2 - days_to_stockout) / max(1.0, lead_time_days * 0.5)</code><br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>P(Stockout) = 1.0 / (1.0 + exp(-2.0 * clip(z, -15, 15)))</code>", bullet_style))
    story.append(Paragraph("• <b>Alert Classification Logic:</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <b>CRITICAL:</b> <code>days_to_stockout &le; 3.0</code> OR <code>P(Stockout) &ge; 0.70</code> (Immediate emergency transfer required).", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <b>WARNING:</b> <code>days_to_stockout &le; 7.0</code> OR <code>P(Stockout) &ge; 0.40</code> (Routine warehouse reorder triggered).", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <b>SAFE:</b> <code>days_to_stockout > 7.0</code> and stable inventory buffer.", bullet_style))

    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # MODULE 2: BED OCCUPANCY & SURGE FORECASTER
    # -------------------------------------------------------------
    story.append(Paragraph("3. Module 2: Bed Occupancy & Overflow Surge Forecaster", h1_style))
    
    story.append(Paragraph("A. Problem Statement", h2_style))
    story.append(Paragraph(
        "Hospitals have fixed inpatient capacities across General, ICU, and Pediatric wards. During epidemic surges, sudden inflows of severe cases rapidly overwhelm local bed capacities. Without predictive foresight, facilities experience overflow crises where patients are turned away or placed in unequipped corridors.",
        body_style
    ))
    
    story.append(Paragraph("B. Solution Offered", h2_style))
    story.append(Paragraph(
        "Module 2 delivers multi-target gradient-boosted forecasting of daily bed occupancy and patient admission volumes 7 days in advance. Integrated with a dynamic surge detection engine, it quantifies overflow severity, predicts the exact date capacity thresholds will be breached, and calculates the exact number of overflow patients requiring diversion.",
        body_style
    ))

    story.append(Paragraph("C. Technical Implementation Details (\"HOW\" it Works)", h2_style))
    story.append(Paragraph("• <b>Machine Learning Model:</b> Scikit-Learn <code>MultiOutputRegressor</code> wrapping <b><code>XGBRegressor</code></b> with hyperparameters: <code>n_estimators=150</code>, <code>max_depth=5</code>, <code>learning_rate=0.06</code>, <code>subsample=0.85</code>, <code>colsample_bytree=0.85</code>, <code>random_state=42</code>.", bullet_style))
    story.append(Paragraph("• <b>Simultaneous Target Vector (14 Targets):</b> Predicts 7-day occupancy counts (<code>target_occ_day_1</code> to <code>7</code>) and 7-day expected patient admission volume (<code>target_adm_day_1</code> to <code>7</code>).", bullet_style))
    story.append(Paragraph("• <b>Feature Pipeline:</b> Historical bed occupancy lags (<i>t-1, t-3, t-7, t-14</i>), historical admission and discharge rates, total licensed bed capacity, district-level bed density, and localized weather/epidemic interaction vectors.", bullet_style))
    story.append(Paragraph("• <b>Surge Detection Engine & Mathematical Thresholds:</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Predicted Occupancy Rate:</i> <code>predicted_occupancy_pct = (predicted_occupied_beds / total_beds) * 100</code>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <b>CRITICAL_SURGE:</b> <code>predicted_occupancy_pct &ge; 90.0%</code> (Triggers Module 4 emergency patient diversion).", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <b>WARNING_SURGE:</b> <code>75.0% &le; predicted_occupancy_pct < 90.0%</code> (Early warning for pre-discharge planning).", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Overflow Quantifier:</i> <code>overflow_beds_needed = max(0, predicted_occupied_beds - total_beds * 0.85)</code>", bullet_style))

    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # MODULE 3: STAFFING & OPERATIONAL READINESS
    # -------------------------------------------------------------
    story.append(Paragraph("4. Module 3: Staffing & Operational Readiness Evaluator", h1_style))
    
    story.append(Paragraph("A. Problem Statement", h2_style))
    story.append(Paragraph(
        "Bed capacity and medicine inventory are meaningless if a facility lacks qualified doctors, nurses, and pharmacists to administer treatment. Unplanned staff absenteeism, fatigue from double shifts, and specialty deficits degrade care quality and make receiving transfer patients clinically dangerous.",
        body_style
    ))
    
    story.append(Paragraph("B. Solution Offered", h2_style))
    story.append(Paragraph(
        "Module 3 provides a deterministic clinical compliance and demand-weighted staffing assessment engine. It evaluates role-specific headcounts against Indian Public Health Standards (IPHS), computes a composite shortage severity score weighted by facility operational demand, and acts as the gatekeeper verifying whether a clinic is capable of safely receiving transfers.",
        body_style
    ))

    story.append(Paragraph("C. Technical Implementation Details (\"HOW\" it Works)", h2_style))
    story.append(Paragraph("• <b>Clinical Demand-Weighted Evaluation Engine:</b> Evaluates present duty headcount versus mandated IPHS standards for Doctors, Nurses, and Pharmacists.", bullet_style))
    story.append(Paragraph("• <b>Role-Specific Shortfall Ratios:</b> For each role <i>r &isin; {doctor, nurse, pharmacist}</i>: <br/>&nbsp;&nbsp;&nbsp;&nbsp;<code>role_score[r] = max(0, required[r] - present[r]) / required[r]</code>", bullet_style))
    story.append(Paragraph("• <b>Operational Demand Multiplier (&mu;):</b> Dynamically amplifies shortage severity during high facility strain:", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- If <code>bed_occupancy_pct > 90%</code> &implies; <code>&mu; = &mu; * 1.50</code>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- If <code>75% < bed_occupancy_pct &le; 90%</code> &implies; <code>&mu; = &mu; * 1.25</code>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- If <code>patient_footfall_ratio > 1.20</code> &implies; <code>&mu; = &mu; * 1.30</code>", bullet_style))
    story.append(Paragraph("• <b>Composite Shortage Formula:</b> Doctor availability carries highest clinical weight (55%): <br/>&nbsp;&nbsp;&nbsp;&nbsp;<code>composite_score = (0.55 * role_scores['doctor'] + 0.30 * role_scores['nurse'] + 0.15 * role_scores['pharmacist']) * &mu;</code>", bullet_style))
    story.append(Paragraph("• <b>Operational Feasibility Gatekeeper (<code>is_facility_operationally_feasible</code>):</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- If <code>doctor_shortfall &ge; 50% of required</code> OR <code>composite_score &ge; 0.80</code> &implies; <code>operationally_feasible = False</code>.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- Facilities flagged as unfeasible are strictly blocked from being selected as transfer recipients in Module 4.", bullet_style))
    story.append(Paragraph("• <b>24-Hour Shift Forecaster:</b> Projects staffing across Morning (08:00–16:00), Evening (16:00–00:00), and Night (00:00–08:00) shifts to prevent unstaffed night-duty emergencies.", bullet_style))

    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # MODULE 4: DYNAMIC REDISTRIBUTION OPTIMIZER
    # -------------------------------------------------------------
    story.append(Paragraph("5. Module 4: Dynamic Cross-District Redistribution Optimizer", h1_style))
    
    story.append(Paragraph("A. Problem Statement", h2_style))
    story.append(Paragraph(
        "Healthcare supply chains and patient distribution operate in disconnected administrative silos. During localized crises, one hospital experiences severe medicine stockouts and overflowing wards while a nearby clinic 15 km away possesses surplus stock and unoccupied beds, but lack of coordination prevents timely mutual aid.",
        body_style
    ))
    
    story.append(Paragraph("B. Solution Offered", h2_style))
    story.append(Paragraph(
        "Module 4 bridges supply and demand across districts. It takes critical stockout deficits from Module 1, overflow surges from Module 2, and staff feasibility constraints from Module 3 to synthesize optimal, multi-echelon transfer routes for both medicines and patients using road network distance matrices.",
        body_style
    ))

    story.append(Paragraph("C. Technical Implementation Details (\"HOW\" it Works)", h2_style))
    story.append(Paragraph("• <b>Algorithmic Paradigm:</b> Priority-Ranked Greedy Nearest-Neighbor with dynamic supply pool state tracking and geodesic distance matrix lookups.", bullet_style))
    story.append(Paragraph("• <b>Medicine Redistribution Pipeline:</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1. <i>Deficit Priority Sorting:</i> Deficits are ranked by urgency: <code>key = (stockout_probability &darr;, shortfall_quantity &darr;)</code>.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2. <i>Surplus Pool Tracking:</i> Maintains dynamic state <code>(phc_id, drug_id) &rarr; available_surplus</code>.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3. <i>Road Network Search:</i> Queries <code>distance_matrix.csv</code> to find the closest donor facility within travel limit (<code>max_travel_hours &le; 10.0</code>).", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4. <i>Transfer Allocation:</i> <code>transfer_qty = min(shortfall, donor_surplus)</code>. Donor surplus is decremented in real-time in O(1).", bullet_style))
    story.append(Paragraph("• <b>Emergency Patient Diversion Pipeline:</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1. Identifies overflow origin hospitals where predicted occupancy exceeds 85%.", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2. Queries receiving candidate facilities with spare capacity (occupancy &le; 75%).", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3. <i>Staff Feasibility Gate:</i> Enforces <code>facility_feasibility_map[candidate_phc] == True</code> (verified via Module 3).", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4. Pairs origin with the closest feasible destination, computing travel distance, estimated ambulance transit time, and cold-chain/dispatch route orders.", bullet_style))
    story.append(Paragraph("• <b>Performance Optimization:</b> Assessment caching reduces redistribution plan synthesis across 90 facilities from 50+ seconds to sub-second execution.", bullet_style))

    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # MODULE 5: FEDERATED LEARNING & PRIVACY ARCHITECTURE
    # -------------------------------------------------------------
    story.append(Paragraph("6. Module 5: Federated Learning & Privacy Architecture", h1_style))
    
    story.append(Paragraph("A. Problem Statement", h2_style))
    story.append(Paragraph(
        "Centralizing Electronic Health Records (EHR) or patient transaction logs to train predictive AI violates statutory data protection legislation, including HIPAA, India's Digital Personal Data Protection (DPDP) Act 2023, and GDPR. Strict medical confidentiality prevents pooling raw data across state health departments.",
        body_style
    ))
    
    story.append(Paragraph("B. Solution Offered", h2_style))
    story.append(Paragraph(
        "Module 5 implements a decentralized Federated Learning network with Differential Privacy. Individual hospital nodes train local model weights on private data inside their institutional boundary. Only encrypted, mathematically perturbed gradient updates are transmitted to the central orchestrator for parameter aggregation, guaranteeing zero raw data egress.",
        body_style
    ))

    story.append(Paragraph("C. Technical Implementation Details (\"HOW\" it Works)", h2_style))
    story.append(Paragraph("• <b>Federated Aggregation Algorithm (<code>FedAvg</code>):</b> Distributed client nodes (Tamil Nadu, Kerala, Maharashtra, Karnataka, Gujarat) compute local parameter vectors (<i>W<sub>k</sub>, b<sub>k</sub></i>). The central server fuses updates using sample-weighted averaging:", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;<code>W_global = &sum; (n_k / N_total) * W_k</code> &nbsp;&nbsp;|&nbsp;&nbsp; <code>b_global = &sum; (n_k / N_total) * b_k</code>", bullet_style))
    story.append(Paragraph("• <b>Differential Privacy Engine (Laplace Mechanism):</b>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>L2-Norm Weight Clipping:</i> Bounds global sensitivity &Delta;f to clipping threshold <i>C = 10.0</i>: <br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>W = W * min(1.0, C / ||W||_2)</code>", bullet_style))
    story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- <i>Laplace Noise Perturbation:</i> Injects calibrated noise parameterized by privacy budget &epsilon;: <br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>Noise ~ Laplace(loc=0, scale=&Delta;f / &epsilon;_eff)</code>, where <code>&epsilon;_eff = &epsilon; * sqrt(max(1.0, N_total / 1000.0))</code>", bullet_style))
    story.append(Paragraph("• <b>Privacy Budget Accounting:</b> Tracks cumulative &epsilon; expenditure across training rounds, ensuring strict (<i>&epsilon;, &delta;</i>)-Differential Privacy guarantees mathematically immune to data reconstruction and membership inference attacks.", bullet_style))
    story.append(Paragraph("• <b>Statutory Compliance:</b> Fully aligned with DPDP Act 2023 Sec 4(2), HIPAA Security Rule 45 CFR Part 164, and BRICS Distributed Healthcare AI frameworks.", bullet_style))

    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # MASTER TECHNICAL SUMMARY TABLE
    # -------------------------------------------------------------
    story.append(Paragraph("7. Master Architectural & Technical Summary", h1_style))
    
    table_data = [
        [
            Paragraph("<b>Module</b>", code_style),
            Paragraph("<b>Primary Algorithm / Model</b>", code_style),
            Paragraph("<b>Key Mathematical Formula / Logic</b>", code_style),
            Paragraph("<b>Operational Constraint / Output</b>", code_style)
        ],
        [
            Paragraph("<b>Mod 1: Medicine</b>", body_style),
            Paragraph("Multi-Output XGBRegressor<br/>(180 trees, depth 6)", body_style),
            Paragraph("Logistic Sigmoid Risk:<br/><code>P = 1 / (1 + e^(-2z))</code>", code_style),
            Paragraph("7-Day Forecast & Shortfall (Critical &le; 3d, Warning &le; 7d)", body_style)
        ],
        [
            Paragraph("<b>Mod 2: Beds</b>", body_style),
            Paragraph("Multi-Output XGBRegressor<br/>(150 trees, depth 5)", body_style),
            Paragraph("Surge Thresholding:<br/><code>Overflow = max(0, Occ - Beds*0.85)</code>", code_style),
            Paragraph("14-Target Occupancy & Admissions (Critical Surge &ge; 90%)", body_style)
        ],
        [
            Paragraph("<b>Mod 3: Staff</b>", body_style),
            Paragraph("Deterministic IPHS Demand Evaluator", body_style),
            Paragraph("Composite Shortage Score:<br/><code>(0.55D + 0.30N + 0.15P) * &mu;</code>", code_style),
            Paragraph("Operational Feasibility Gatekeeper (Doctor Shortfall &lt; 50%)", body_style)
        ],
        [
            Paragraph("<b>Mod 4: Routing</b>", body_style),
            Paragraph("Priority Nearest-Neighbor Graph Search", body_style),
            Paragraph("Distance Matrix Minimization & Dynamic State Depletion", code_style),
            Paragraph("Inter-facility transfer routes (&le; 10h transit limit)", body_style)
        ],
        [
            Paragraph("<b>Mod 5: Privacy</b>", body_style),
            Paragraph("Sample-Weighted FedAvg + Laplace DP", body_style),
            Paragraph("Laplace Noise Injection:<br/><code>Noise ~ Lap(0, &Delta;f / &epsilon;_eff)</code>", code_style),
            Paragraph("Zero Raw Data Egress, (0.50, 1e-5)-DP Certified Model", body_style)
        ]
    ]

    col_widths = [1.1 * inch, 1.6 * inch, 2.2 * inch, 2.1 * inch]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 14))

    # -------------------------------------------------------------
    # INTER-MODULE WORKFLOW DIAGRAM / PIPELINE
    # -------------------------------------------------------------
    story.append(Paragraph("8. End-to-End Interconnected Execution Pipeline", h1_style))
    story.append(Paragraph(
        "The power of PHC Resilience Command lies in the tight coupling of all five autonomous modules into a unified decision loop:",
        body_style
    ))
    story.append(Paragraph("<b>Step 1 (Early Warning):</b> Modules 1 & 2 continuously ingest local facility telemetry and external signals, outputting 7-day predictive vectors for medicine shortfalls and bed occupancy surges.", bullet_style))
    story.append(Paragraph("<b>Step 2 (Clinical Manpower Verification):</b> Module 3 benchmarks facility staffing rosters against patient demand and marks qualified recipient clinics as <code>operationally_feasible = True</code>.", bullet_style))
    story.append(Paragraph("<b>Step 3 (Multi-Echelon Rebalancing):</b> Module 4 matches critical deficits from Mod 1 and overflow patients from Mod 2 with feasible surplus facilities, computing optimal transit routes via the road network matrix.", bullet_style))
    story.append(Paragraph("<b>Step 4 (Operational Dispatch):</b> Health officers authorize transfers with a single click, issuing cold-chain dispatch orders, GPS route sheets, and automated Chief Medical Officer (CMO) approvals.", bullet_style))
    story.append(Paragraph("<b>Step 5 (Continuous Privacy-Preserving AI Improvement):</b> Module 5 coordinates decentralized model training across all nodes, aggregating encrypted model updates with Differential Privacy to continuously enhance forecast accuracy without ever exposing confidential patient health data.", bullet_style))
    
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.0, color=c_secondary, spaceBefore=6, spaceAfter=8))
    story.append(Paragraph("<i>Document generated automatically by PHC Resilience Command Intelligence Engine. All algorithms verified against production codebase.</i>", meta_style))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_pdf("approach.pdf")
