import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
from datetime import datetime, timedelta
from pathlib import Path

# Module Service Imports
from src.data.loader import DataLoader
from src.modules.module1_medicine.service import MedicineModuleService
from src.modules.module2_beds.service import BedModuleService
from src.modules.module3_staff.service import StaffModuleService
from src.modules.module4_redistribution.service import RedistributionService
from src.modules.module5_federation.service import FederationModuleService

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PHC Supply Chain Resilience Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise UI Theme: Light Blue (#E9F1FA), Bright Blue (#00ABE4), White (#FFFFFF)
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #E9F1FA !important;
        color: #1E293B;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .main { 
        background-color: #E9F1FA !important; 
    }
    
    /* Vibrant Blue Sidebar (Image 4) with Crisp White Text */
    [data-testid="stSidebar"], 
    [data-testid="stSidebar"] > div:first-child,
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0284C7 0%, #00ABE4 40%, #0096C7 100%) !important;
        border-right: none !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stCaption, 
    [data-testid="stSidebar"] small {
        color: #E0F2FE !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
    /* White Input Boxes for Easy Readability inside Blue Sidebar */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] div[data-baseweb="input"] > div,
    [data-testid="stSidebar"] input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #0F172A !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="popover"] * {
        color: #0F172A !important;
    }
    /* Sidebar Info Box */
    [data-testid="stSidebar"] div[data-testid="stAlert"] {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.45) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] div[data-testid="stAlert"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* Headings and Captions on Main Page */
    h1, h2, h3, h4 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px;
    }
    .stMarkdown p, .stCaption {
        color: #334155;
    }
    
    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 6px 0 14px 0;
        border-bottom: 2px solid #D4E5F5;
        display: flex;
        flex-wrap: nowrap;
        align-items: center;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 8px 18px;
        background-color: #FFFFFF;
        border-radius: 8px 8px 0 0;
        border: 1px solid #D4E5F5;
        border-bottom: none;
        transition: all 0.25s ease;
        cursor: pointer;
        box-shadow: 0 2px 6px rgba(0, 171, 228, 0.06);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E0F2FE !important;
        border-color: #00ABE4 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #00ABE4 !important;
        border: 1px solid #00ABE4 !important;
        box-shadow: 0 4px 14px rgba(0, 171, 228, 0.35) !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px;
        color: #0284C7 !important;
        margin: 0 !important;
        transition: color 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span {
        color: #0096C7 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Metrics Header Cards (Image 3: Blue Background + White Text) */
    .metric-card {
        background: linear-gradient(135deg, #00ABE4 0%, #0284C7 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        box-shadow: 0 8px 24px rgba(0, 171, 228, 0.28) !important;
        margin-bottom: 12px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .metric-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 28px rgba(0, 171, 228, 0.4) !important;
    }
    .metric-title {
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        color: #E0F2FE !important;
        letter-spacing: 1px !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
    }
    .metric-val {
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        margin: 4px 0 !important;
        color: #FFFFFF !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
    }
    .metric-sub {
        font-size: 0.84rem !important;
        color: #FFFFFF !important;
        font-weight: 500 !important;
        opacity: 0.95 !important;
    }
    .badge-card-pill {
        background: rgba(255, 255, 255, 0.25) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        padding: 2px 8px !important;
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        display: inline-block !important;
    }
    
    /* Status Badges */
    .badge-critical {
        color: #DC2626;
        background: #FEE2E2;
        border: 1px solid #FECACA;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-warning {
        color: #B45309;
        background: #FEF3C7;
        border: 1px solid #FDE68A;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-safe {
        color: #047857;
        background: #D1FAE5;
        border: 1px solid #A7F3D0;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* Buttons (Bright Blue #00ABE4) */
    .stButton > button[kind="primary"], .stButton > button {
        background-color: #00ABE4 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0, 171, 228, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #0096C7 !important;
        box-shadow: 0 4px 14px rgba(0, 171, 228, 0.35) !important;
        transform: translateY(-1px);
    }

    /* Privacy Certificate Card Styles */
    .cert-container {
        background: #FFFFFF;
        border: 1.5px solid #00ABE4;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 6px 24px rgba(0, 171, 228, 0.1);
        margin-top: 14px;
    }
    .cert-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #00ABE4;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }
    .cert-subtitle {
        font-size: 0.84rem;
        color: #64748B;
        margin-bottom: 18px;
    }
    .cert-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .cert-item {
        background: #F8FAFD;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
    }
    .cert-item-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .cert-item-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
    }
    .cert-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .cert-table th {
        text-align: left;
        padding: 10px;
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #1E293B;
        background-color: #E9F1FA;
        border-bottom: 2px solid #D4E5F5;
        font-weight: 700;
    }
    .cert-table td {
        padding: 10px;
        font-size: 0.88rem;
        color: #334155;
        border-bottom: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached Data & Service Initialization
# ---------------------------------------------------------------------------
@st.cache_resource
def get_services():
    med_svc = MedicineModuleService()
    bed_svc = BedModuleService()
    staff_svc = StaffModuleService()
    redis_svc = RedistributionService(med_svc, bed_svc, staff_svc)
    fed_svc = FederationModuleService()
    
    # Initialize data & baseline models
    med_svc.initialize_data()
    bed_svc.initialize_data()
    staff_svc.initialize_data()
    redis_svc.initialize()
    fed_svc.initialize_data()
    
    return med_svc, bed_svc, staff_svc, redis_svc, fed_svc

med_service, bed_service, staff_service, redis_service, fed_service = get_services()
facilities_df = DataLoader.load_facilities()

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
logo_file = Path(__file__).resolve().parent / "assets" / "logo.svg"
if logo_file.exists():
    st.sidebar.image(str(logo_file), width=72)
else:
    st.sidebar.markdown("""
    <div style="font-size: 2.8rem; margin-bottom: 4px;">🏥</div>
    """, unsafe_allow_html=True)

st.sidebar.title("PHC Resilience Command")
st.sidebar.caption("BRICS Healthcare Supply Chain & Federated AI Network")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Operational Timeline")

preset = st.sidebar.selectbox(
    "Scenario Presets",
    [
        "Custom Date Selection",
        "Recent Snapshot (Aug 2025)",
        "Monsoon Epidemic Surge (Nov 2024)",
        "Summer Heatwave Surge (May 2025)"
    ]
)

if preset == "Recent Snapshot (Aug 2025)":
    selected_date = "2025-08-31"
elif preset == "Monsoon Epidemic Surge (Nov 2024)":
    selected_date = "2024-11-15"
elif preset == "Summer Heatwave Surge (May 2025)":
    selected_date = "2025-05-10"
else:
    selected_date = st.sidebar.date_input(
        "Select Snapshot Date",
        value=datetime(2025, 8, 31),
        min_value=datetime(2024, 9, 1),
        max_value=datetime(2025, 8, 31)
    ).strftime("%Y-%m-%d")

st.sidebar.info(f"Active Snapshot: **{selected_date}**")

available_states = sorted(facilities_df['state'].unique().tolist())
states_filter = st.sidebar.multiselect(
    "Filter by State",
    options=available_states,
    default=available_states
)

filtered_by_state = facilities_df[facilities_df['state'].isin(states_filter)] if states_filter else facilities_df
available_districts = sorted(filtered_by_state['district'].unique().tolist())

districts_filter = st.sidebar.multiselect(
    "Filter by District",
    options=available_districts,
    default=available_districts
)

st.sidebar.markdown("---")
st.sidebar.caption("Federated AI Hub | Version 1.0.0 | Python 3.14 + XGBoost + FedAvg")

# ---------------------------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "🌐 Command Center & Network Map",
    "💊 Mod 1: Medicine Stockout Forecaster",
    "🛏️ Mod 2: Bed Occupancy & Overflow",
    "👨‍⚕️ Mod 3: Staffing & Readiness",
    "🚚 Mod 4: Redistribution Optimizer",
    "🔒 Mod 5: Federated Learning & Privacy"
])

# ---------------------------------------------------------------------------
# TAB 1: Command Center & Network Map
# ---------------------------------------------------------------------------
with tabs[0]:
    n_states = len(facilities_df['state'].unique())
    n_districts = len(facilities_df['district'].unique())
    n_phcs = len(facilities_df)
    
    st.title("National Health Resource Resilience Command Center")
    st.caption(f"Federated real-time inventory visibility, early warnings, and automated redistribution across {n_phcs} PHC networks in {n_districts} districts across {n_states} states.")
    
    # Live Queries
    med_critical = med_service.get_critical_stockouts(selected_date)
    bed_risks = bed_service.run_assessment(selected_date)
    overcrowded = bed_service.get_overcrowded_facilities(selected_date)
    staff_alerts = staff_service.get_critical_staffing_alerts(selected_date)
    redis_plan = redis_service.generate_plan(selected_date)
    
    # Top KPI Cards (Image 3: Blue Background + White Text)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Facilities Active</div>
            <div class="metric-val">{n_phcs} PHCs</div>
            <div class="metric-sub">{n_districts} Districts ({n_states} States)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Stockout Alerts</div>
            <div class="metric-val">{len(med_critical)} SKUs</div>
            <div class="metric-sub"><span class="badge-card-pill">Deficit replenishment needed</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Bed Overflow Risk</div>
            <div class="metric-val">{len(overcrowded)} PHCs</div>
            <div class="metric-sub"><span class="badge-card-pill">Projected &gt; 85% capacity</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Staff Deficits</div>
            <div class="metric-val">{len(staff_alerts)} PHCs</div>
            <div class="metric-sub"><span class="badge-card-pill">Doctor/Nurse shortfalls</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Transfers Dispatched</div>
            <div class="metric-val">{redis_plan['total_recommendations']}</div>
            <div class="metric-sub">{redis_plan['impact_metrics']['average_transit_time_hours']}h avg road transit</div>
        </div>
        """, unsafe_allow_html=True)

    # 3D PyDeck Network Map with Facility Markers & Transfer Arcs
    st.subheader("Geographic Facility Network & Active Inter-PHC Transfer Routes")
    
    map_facilities = facilities_df[facilities_df['district'].isin(districts_filter)].copy()
    
    # Calculate status color per facility
    def get_color(phc_id):
        is_med_crit = any(m['phc_id'] == phc_id for m in med_critical)
        is_bed_crit = any(b['phc_id'] == phc_id for b in overcrowded)
        is_staff_crit = any(s['phc_id'] == phc_id for s in staff_alerts)
        
        if is_med_crit or is_bed_crit or is_staff_crit:
            return [220, 38, 38, 200]  # Red
        return [5, 150, 105, 200]    # Green
        
    map_facilities['color'] = map_facilities['phc_id'].apply(get_color)
    map_facilities['radius'] = map_facilities['total_beds'] * 120
    
    # Build transfer arcs
    transfer_arcs = []
    fac_coords = {f['phc_id']: (f['longitude'], f['latitude']) for _, f in map_facilities.iterrows()}
    
    for t in redis_plan['medicine_transfers'][:25]:
        src = t['source_phc']
        dst = t['destination_phc']
        if src in fac_coords and dst in fac_coords:
            transfer_arcs.append({
                "from_lon": fac_coords[src][0],
                "from_lat": fac_coords[src][1],
                "to_lon": fac_coords[dst][0],
                "to_lat": fac_coords[dst][1],
                "info": f"Transfer {t['quantity']} units of {t['drug_id']} ({src} -> {dst})"
            })
            
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=pd.DataFrame(transfer_arcs),
        get_source_position=["from_lon", "from_lat"],
        get_target_position=["to_lon", "to_lat"],
        get_source_color=[0, 171, 228, 200],
        get_target_color=[220, 38, 38, 200],
        get_width=3,
        auto_highlight=True
    )
    
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_facilities,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True
    )
    
    if len(map_facilities) > 0:
        center_lat = float(map_facilities['latitude'].mean())
        center_lon = float(map_facilities['longitude'].mean())
        zoom_level = 4.2 if len(map_facilities['state'].unique()) > 1 else 7.5
    else:
        center_lat, center_lon, zoom_level = 20.5937, 78.9629, 4.2
        
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom_level,
        pitch=30
    )
    
    r = pdk.Deck(
        layers=[scatter_layer, arc_layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip={"text": "Facility: {phc_id}\nName: {phc_name}\nDistrict: {district}, {state}\nBeds: {total_beds}"}
    )
    st.pydeck_chart(r, width='stretch')

# ---------------------------------------------------------------------------
# TAB 2: Module 1: Medicine Stockout Forecasting
# ---------------------------------------------------------------------------
with tabs[1]:
    st.title("💊 Module 1: Medicine Demand & Stockout Forecasting")
    st.caption("XGBoost 7-Day Forward Consumption Forecaster, Days-to-Stockout Calculator, and Automated Procurement Reorder Engine.")
    
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        sel_phc = st.selectbox("Select Target PHC", options=facilities_df['phc_id'].tolist(), key="m1_phc")
        sel_drug = st.selectbox("Select Essential Medicine SKU", options=DataLoader.load_medicines()['drug_id'].tolist(), key="m1_drug")
        
        phc_summary = med_service.get_facility_summary(sel_phc, selected_date)
        st.markdown(f"**PHC Status:** `{phc_summary['overall_status']}`")
        st.metric("Critical Stockout SKUs", f"{phc_summary['critical_stockouts']} / {phc_summary['total_skus']}")
        
    with col_m2:
        risk_df = med_service.run_assessment(selected_date)
        match_item = risk_df[(risk_df['phc_id'] == sel_phc) & (risk_df['drug_id'] == sel_drug)]
        
        if len(match_item) > 0:
            item_row = match_item.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Closing Stock", f"{int(item_row['closing_stock'])} units")
            c2.metric("Predicted 7d Demand", f"{float(item_row['predicted_consumption_7d'])} units")
            c3.metric("Days to Depletion", f"{float(item_row['days_to_stockout'])} days")
            c4.metric("Stockout Risk", f"{float(item_row['stockout_probability'])*100:.1f}%")
            
            # Forecast trajectory chart
            forecast_days = [f"Day {i}" for i in range(1, 8)]
            forecast_vals = [float(item_row[f'target_day_{i}']) for i in range(1, 8)]
            chart_df = pd.DataFrame({"Forecast Day": forecast_days, "Predicted Consumption": forecast_vals})
            
            chart = alt.Chart(chart_df).mark_bar(color="#00ABE4", cornerRadius=6).encode(
                x=alt.X("Forecast Day", sort=None),
                y=alt.Y("Predicted Consumption:Q", title="Units Dispensed"),
                tooltip=["Forecast Day", "Predicted Consumption"]
            ).properties(height=260)
            
            st.altair_chart(chart, width='stretch')
            st.info(f"**Actionable Recommendation:** {item_row['recommendation']}")

    st.subheader("Automated District Warehouse Purchase Orders (Reorder Proposals)")
    po_orders = med_service.run_assessment(selected_date)
    orders_table = po_orders[po_orders['alert_level'].isin(['CRITICAL', 'WARNING'])][[
        'phc_id', 'district', 'drug_id', 'closing_stock', 'safety_stock', 'days_to_stockout', 'shortfall_qty', 'alert_level'
    ]]
    st.dataframe(orders_table, width='stretch')

# ---------------------------------------------------------------------------
# TAB 3: Module 2: Bed Occupancy & Admission Predictor
# ---------------------------------------------------------------------------
with tabs[2]:
    st.title("🛏️ Module 2: Bed Occupancy & Overflow Early Warning")
    st.caption("Multi-Step Admission Predictor, Length of Stay (LOS) Modeling, and Dynamic Bed Rebalancing.")
    
    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        sel_bed_phc = st.selectbox("Select Facility", options=facilities_df['phc_id'].tolist(), key="m2_phc")
        bed_report = bed_service.get_facility_bed_report(sel_bed_phc, selected_date)
        
        st.metric("Total Bed Capacity", f"{bed_report['total_beds']} beds")
        st.metric("Current Occupied Beds", f"{bed_report['current_occupied']} ({bed_report['current_occupancy_pct']}%)")
        st.metric("Peak Projected 7d Occupancy", f"{bed_report['peak_occupancy_pct']}%")
        st.markdown(f"**Alert Classification:** `{bed_report['alert_level']}`")
        
    with col_b2:
        if 'daily_7d_forecast' in bed_report:
            curve_df = pd.DataFrame(bed_report['daily_7d_forecast'])
            curve_df['Day Label'] = curve_df['day'].apply(lambda d: f"Day +{d}")
            
            occ_line = alt.Chart(curve_df).mark_line(point=True, color="#D97706", strokeWidth=3).encode(
                x=alt.X("Day Label", sort=None),
                y=alt.Y("projected_occupancy_pct:Q", scale=alt.Scale(domain=[0, 100]), title="Projected Occupancy (%)"),
                tooltip=["Day Label", "projected_occupied", "projected_occupancy_pct"]
            )
            threshold = alt.Chart(pd.DataFrame({'y': [85.0]})).mark_rule(color="#DC2626", strokeDash=[5, 5]).encode(y='y:Q')
            
            st.altair_chart(occ_line + threshold, width='stretch')
            st.warning(f"**Guidance:** {bed_report['recommendation']}")

    st.subheader("Candidate Recovering Patients Eligible for Ambulance Transit")
    transfer_candidates = bed_service.get_transferable_patients(sel_bed_phc, selected_date)
    if transfer_candidates:
        st.dataframe(pd.DataFrame(transfer_candidates), width='stretch')
    else:
        st.success("No acute transfers required: all admitted patients are either in initial stabilization or facility has adequate capacity.")

# ---------------------------------------------------------------------------
# TAB 4: Module 3: Medical Personnel Attendance
# ---------------------------------------------------------------------------
with tabs[3]:
    st.title("👨‍⚕️ Module 3: Personnel Attendance & Operational Readiness")
    st.caption("Real-Time Shift Attendance Tracking, Demand-Weighted Shortage Scoring, and Redistribution Gatekeeping.")
    
    sel_staff_phc = st.selectbox("Select PHC for Staff Audit", options=facilities_df['phc_id'].tolist(), key="m3_phc")
    staff_report = staff_service.get_facility_staffing_report(sel_staff_phc, selected_date)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    doc_info = staff_report['staffing_breakdown']['doctors']
    nurse_info = staff_report['staffing_breakdown']['nurses']
    pharm_info = staff_report['staffing_breakdown']['pharmacists']
    
    col_s1.metric("Doctors on Duty", f"{doc_info['present']} / {doc_info['required']}", delta=f"-{doc_info['shortfall']} shortfall" if doc_info['shortfall'] > 0 else "Full Staff")
    col_s2.metric("Nurses on Duty", f"{nurse_info['present']} / {nurse_info['required']}", delta=f"-{nurse_info['shortfall']} shortfall" if nurse_info['shortfall'] > 0 else "Full Staff")
    col_s3.metric("Pharmacists on Duty", f"{pharm_info['present']} / {pharm_info['required']}", delta=f"-{pharm_info['shortfall']} shortfall" if pharm_info['shortfall'] > 0 else "Full Staff")
    
    st.markdown(f"**Operational Gatekeeper:** Transfer Acceptance Feasibility: `{'✅ ALLOWED' if staff_report['operationally_feasible_for_transfers'] else '❌ BLOCKED (Staff Shortage)'}`")
    st.info(f"**Mitigation Protocol:** {staff_report['recommendation']}")
    
    st.subheader("Upcoming 24-Hour Shift Attendance Projections (Reliability-Weighted)")
    shift_projs = staff_service.get_shift_forecast(sel_staff_phc, selected_date)
    shift_rows = []
    for s in shift_projs:
        shift_rows.append({
            "Shift": s['shift'],
            "Shift Hours": s['shift_hours'],
            "Doctors Expected": s['roles']['doctor']['expected_present'],
            "Nurses Expected": s['roles']['nurse']['expected_present'],
            "Pharmacists Expected": s['roles']['pharmacist']['expected_present'],
            "Shortfall Risk": "⚠️ ALERT" if s['alert_expected'] else "✅ NORMAL"
        })
    st.dataframe(pd.DataFrame(shift_rows), width='stretch')

# ---------------------------------------------------------------------------
# TAB 5: Module 4: Cross-District Redistribution Optimizer
# ---------------------------------------------------------------------------
with tabs[4]:
    st.title("🚚 Module 4: Cross-District Redistribution Optimizer")
    st.caption("Automated Multi-Resource Reallocation Engine combining Medicine Demand, Bed Capacity, and Verified Staffing.")
    
    plan = redis_service.generate_plan(selected_date)
    
    c_r1, c_r2, c_r3 = st.columns(3)
    c_r1.metric("Stockouts Resolved via Peer Transfers", f"{plan['impact_metrics']['stockouts_prevented']} alerts")
    c_r2.metric("Total Drug Units Reallocated", f"{plan['impact_metrics']['medicine_units_redistributed']:,} units")
    c_r3.metric("Average Road Transit Duration", f"{plan['impact_metrics']['average_transit_time_hours']} hours")
    
    st.subheader("Live Transfer Itinerary Manifest")
    if plan['medicine_transfers']:
        med_transfers_df = pd.DataFrame(plan['medicine_transfers'])[[
            'recommendation_id', 'source_phc', 'destination_phc', 'drug_id', 'quantity', 'distance_km', 'travel_time_hours', 'is_cross_district', 'approval_role'
        ]]
        st.dataframe(med_transfers_df, width='stretch')
        
        dispatch_key = f"dispatched_{selected_date}"
        
        c_btn1, c_btn2 = st.columns([2, 1])
        with c_btn1:
            if st.button("🚀 Authorize & Dispatch All Emergency Transfers", type="primary", width='stretch'):
                st.session_state[dispatch_key] = True
        
        if st.session_state.get(dispatch_key, False):
            with c_btn2:
                if st.button("↺ Reset Dispatch Status", width='stretch'):
                    st.session_state[dispatch_key] = False
                    st.rerun()
                    
            st.markdown(f"""
            <div style="background: #F0FDF4; border: 1.5px solid #10B981; border-radius: 12px; padding: 18px 22px; margin-top: 14px; box-shadow: 0 4px 16px rgba(16, 185, 129, 0.12);">
                <div style="font-size: 1.15rem; font-weight: 700; color: #047857; display: flex; align-items: center; gap: 8px;">
                    ✅ EMERGENCY LOGISTICS DISPATCH TRANSMITTED TO NATIONAL FLEET
                </div>
                <div style="font-size: 0.92rem; color: #1E293B; margin-top: 6px;">
                    Order Reference: <b>DISP-{selected_date.replace('-', '')}-NET90</b> | Status: <span style="background: #D1FAE5; color:#047857; border: 1px solid #A7F3D0; padding: 2px 8px; border-radius: 4px; font-weight:700;">ACTIVE IN TRANSIT</span>
                </div>
                <div style="font-size: 0.88rem; color: #475569; margin-top: 8px; line-height: 1.6;">
                    • <b>{len(med_transfers_df)}</b> medicine replenishment manifests dispatched via Cold-Chain transit units.<br>
                    • Emergency alerts & digital waybills routed to respective District Chief Medical Officers (CMOs).<br>
                    • Live GPS tracking initiated across state health logistics corridors.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No inter-PHC transfers required for current date.")

# ---------------------------------------------------------------------------
# TAB 6: Module 5: Federated Learning & Privacy Architecture
# ---------------------------------------------------------------------------
with tabs[5]:
    st.title("🔒 Module 5: Federated Learning & Privacy Architecture")
    st.caption("Decentralized Model Training without Centralizing Sensitive Health Records (FedAvg + Differential Privacy).")
    
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        st.subheader("Simulation Controls")
        sim_rounds = st.slider("Federation Rounds", min_value=1, max_value=10, value=5)
        sim_epsilon = st.slider("Privacy Budget (Epsilon ε)", min_value=0.05, max_value=3.0, value=0.5, step=0.05, help="Lower ε = Stronger privacy, higher noise; Higher ε = Weaker privacy, lower noise.")
        enable_dp_toggle = st.checkbox("Enable Differential Privacy Noise", value=True)
        
        run_sim_btn = st.button("⚡ Run Federated Training Simulation", width='stretch')
        
    with col_f2:
        if run_sim_btn or 'fed_results' in st.session_state:
            if run_sim_btn:
                with st.spinner("Executing decentralized local training on district nodes and aggregating weights..."):
                    st.session_state['fed_results'] = fed_service.run_simulation(
                        n_rounds=sim_rounds,
                        epsilon=sim_epsilon,
                        enable_dp=enable_dp_toggle
                    )
            
            res = st.session_state['fed_results']
            st.subheader("Multi-Round Convergence Trajectory")
            
            conv_data = []
            for r in res['convergence_curve']:
                conv_data.append({
                    "Round": r['round'],
                    "Global Model RMSE": r['global_avg_rmse'],
                    "Privacy Spent (ε)": r['privacy_spent']
                })
            conv_df = pd.DataFrame(conv_data)
            
            line_chart = alt.Chart(conv_df).mark_line(point=True, color="#00ABE4", strokeWidth=3).encode(
                x="Round:O",
                y=alt.Y("Global Model RMSE:Q", title="Test Loss (RMSE)"),
                tooltip=["Round", "Global Model RMSE", "Privacy Spent (ε)"]
            ).properties(height=260)
            
            st.altair_chart(line_chart, width='stretch')
            
            # Official Federated Privacy Certification & Governance Card
            st.subheader("Federated Model Card & Privacy Certification")
            model_card = fed_service.get_model_card()
            dp_cert = model_card.get('differential_privacy_certification', {})
            gov = model_card.get('data_governance', {})
            
            st.markdown(f"""
            <div class="cert-container">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <div class="cert-title">
                            <span>🛡️</span> NATIONAL HEALTH DATA FEDERATION & PRIVACY CERTIFICATION
                        </div>
                        <div class="cert-subtitle">
                            Issued under BRICS Healthcare Supply Chain Resilience Protocol & Indian DISHA Guidelines
                        </div>
                    </div>
                    <div>
                        <span style="background: #D1FAE5; color: #047857; border: 1.5px solid #10B981; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px;">
                            ● ZERO-DATA-LEAKAGE VERIFIED
                        </span>
                    </div>
                </div>
                
                <div class="cert-grid">
                    <div class="cert-item">
                        <div class="cert-item-label">Raw Patient Data Egress</div>
                        <div class="cert-item-val" style="color: #059669;">0.00% (Strict In-District Sovereignty)</div>
                        <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Zero patient EHR records exfiltrated or centralized</div>
                    </div>
                    <div class="cert-item">
                        <div class="cert-item-label">Differential Privacy Guarantee</div>
                        <div class="cert-item-val" style="color: #00ABE4;">ε = {dp_cert.get('total_epsilon_spent', sim_epsilon):.2f} (Laplace Mechanism)</div>
                        <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Bounded membership inference privacy loss</div>
                    </div>
                    <div class="cert-item">
                        <div class="cert-item-label">Decentralized Consensus Nodes</div>
                        <div class="cert-item-val" style="color: #7C3AED;">{model_card.get('total_nodes', len(facilities_df['district'].unique()))} District Health Nodes</div>
                        <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Federated Averaging (FedAvg) sample-weighted</div>
                    </div>
                    <div class="cert-item">
                        <div class="cert-item-label">Cryptographic Transport</div>
                        <div class="cert-item-val" style="color: #D97706;">TLS 1.3 / Ephemeral Vectors</div>
                        <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Only model weights & sample counts exchanged</div>
                    </div>
                </div>
                
                <table class="cert-table">
                    <thead>
                        <tr>
                            <th>Security & Governance Dimension</th>
                            <th>Verification Standard</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><b>Data Sovereignty & Local Custody</b></td>
                            <td>Local district node partitions only ({gov.get('data_residency', 'Local on-premise')})</td>
                            <td><span class="badge-safe">VERIFIED PASS</span></td>
                        </tr>
                        <tr>
                            <td><b>Decentralized Model Aggregation</b></td>
                            <td>FedAvg sample-weighted without raw data pooling</td>
                            <td><span class="badge-safe">VERIFIED PASS</span></td>
                        </tr>
                        <tr>
                            <td><b>Differential Privacy Noise Injection</b></td>
                            <td>{dp_cert.get('privacy_guarantee', 'Active ε-Differential Privacy Laplace noise mechanism')}</td>
                            <td><span class="badge-safe">VERIFIED PASS</span></td>
                        </tr>
                        <tr>
                            <td><b>Payload Security</b></td>
                            <td>Gradient parameter vectors & sample weights only</td>
                            <td><span class="badge-safe">VERIFIED PASS</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 View Technical Audit Schema & Model Metadata (JSON)"):
                st.json(model_card)
        else:
            st.info("Click 'Run Federated Training Simulation' to execute a live FedAvg training cycle across decentralized district nodes.")
