import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="UK Road Safety ML Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom Premium CSS (Dark Theme Accentuated)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title and Headers Styling */
    h1, h2, h3 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #ef8a62 0%, #b2182b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        font-weight: 800;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-bottom: 25px;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, rgba(31, 59, 87, 0.6) 0%, rgba(15, 30, 48, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(4px);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(239, 138, 98, 0.4);
    }
    .metric-value {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 5px 0;
    }
    .val-yellow { color: #e0a106; }
    .val-red { color: #b2182b; }
    .val-teal { color: #2a9d8f; }
    .val-orange { color: #ef8a62; }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Risk Cards */
    .risk-banner {
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin: 20px 0;
    }
    .risk-low {
        background: linear-gradient(135deg, rgba(42, 157, 143, 0.15) 0%, rgba(42, 157, 143, 0.05) 100%);
        border: 2px solid #2a9d8f;
        color: #2a9d8f;
    }
    .risk-medium {
        background: linear-gradient(135deg, rgba(224, 161, 6, 0.15) 0%, rgba(224, 161, 6, 0.05) 100%);
        border: 2px solid #e0a106;
        color: #e0a106;
    }
    .risk-high {
        background: linear-gradient(135deg, rgba(178, 24, 43, 0.15) 0%, rgba(178, 24, 43, 0.05) 100%);
        border: 2px solid #b2182b;
        color: #b2182b;
    }
    
    /* Sidebar styling override */
    .css-1d391kg {
        background-color: #0b132b !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Cache Data Loading Functions for Performance
@st.cache_data
def load_collision_data():
    collision_path = "dft-road-casualty-statistics-collision-provisional-2025 (1).csv"
    if not os.path.exists(collision_path):
        st.error(f"Could not find collision data at {collision_path}. Please place the file in the workspace.")
        st.stop()
        
    df = pd.read_csv(collision_path, low_memory=False)
    
    # Preprocessing
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['date_only'] = df['datetime'].dt.date
    
    # Categorical decodings
    severity_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
    day_map = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
    urban_rural_map = {1: "Urban", 2: "Rural", 3: "Unallocated"}
    light_map = {
        1: "Daylight",
        4: "Darkness - lights lit",
        5: "Darkness - lights unlit",
        6: "Darkness - no lights",
        7: "Darkness - lights detail unknown"
    }
    weather_map = {
        1: "Fine no high winds",
        2: "Raining no high winds",
        3: "Snowing no high winds",
        4: "Fine + high winds",
        5: "Raining + high winds",
        6: "Snowing + high winds",
        7: "Fog or mist",
        8: "Other",
        9: "Unknown"
    }
    surface_map = {
        1: "Dry",
        2: "Wet or damp",
        3: "Snow",
        4: "Frost or ice",
        5: "Flood over 3cm deep",
        6: "Oil or wet mud",
        7: "Road sign"
    }
    
    df['urban_or_rural_area'] = df['urban_or_rural_area'].replace(-1, np.nan)
    df['light_conditions'] = df['light_conditions'].replace(-1, np.nan)
    df['weather_conditions'] = df['weather_conditions'].replace(-1, np.nan)
    df['road_surface_conditions'] = df['road_surface_conditions'].replace(-1, np.nan)
    df['speed_limit'] = df['speed_limit'].replace(-1, np.nan)
    
    df['collision_severity_label'] = df['collision_severity'].map(severity_map).fillna("Unknown")
    df['day_of_week_label'] = df['day_of_week'].map(day_map).fillna("Unknown")
    df['urban_or_rural_label'] = df['urban_or_rural_area'].map(urban_rural_map).fillna("Unknown")
    df['light_conditions_label'] = df['light_conditions'].map(light_map).fillna("Unknown")
    df['weather_conditions_label'] = df['weather_conditions'].map(weather_map).fillna("Unknown")
    df['road_surface_conditions_label'] = df['road_surface_conditions'].map(surface_map).fillna("Unknown")
    df['speed_limit'] = pd.to_numeric(df['speed_limit'], errors='coerce')
    
    # Binary Severity outcome
    df['is_severe'] = df['collision_severity'].isin([1, 2]).astype(int)
    
    return df

@st.cache_data
def load_casualty_data():
    casualty_path = "dft-road-casualty-statistics-casualty-provisional-2025.csv"
    if not os.path.exists(casualty_path):
        return None
    df = pd.read_csv(casualty_path, low_memory=False)
    
    # Decodings
    severity_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
    casualty_class_map = {1: "Driver or rider", 2: "Passenger", 3: "Pedestrian"}
    sex_map = {1: "Male", 2: "Female", 3: "Unknown", 9: "Unknown"}
    age_band_map = {
        1: "0 - 5", 
        2: "6 - 10", 
        3: "11 - 15", 
        4: "16 - 20", 
        5: "21 - 25", 
        6: "26 - 35", 
        7: "36 - 45", 
        8: "46 - 55", 
        9: "56 - 65", 
        10: "66 - 75", 
        11: "Over 75"
    }
    
    df['casualty_severity_label'] = df['casualty_severity'].map(severity_map)
    df['casualty_class_label'] = df['casualty_class'].replace(-1, np.nan).map(casualty_class_map)
    df['sex_of_casualty_label'] = df['sex_of_casualty'].replace(-1, np.nan).map(sex_map)
    df['age_of_casualty'] = df['age_of_casualty'].replace(-1, np.nan)
    df['age_band_of_casualty'] = df['age_band_of_casualty'].replace(-1, np.nan)
    df['age_band_of_casualty_label'] = df['age_band_of_casualty'].map(age_band_map)
    df['casualty_imd_decile'] = pd.to_numeric(df['casualty_imd_decile'].replace(-1, np.nan), errors='coerce')
    
    return df

@st.cache_data
def load_vehicle_data():
    vehicle_path = "dft-road-casualty-statistics-vehicle-provisional-2025.csv"
    if not os.path.exists(vehicle_path):
        return None
    df = pd.read_csv(vehicle_path, low_memory=False)
    
    vehicle_type_map = {
        1: "Pedal cycle", 2: "Motorcycle 50cc and under", 3: "Motorcycle 50cc-125cc",
        4: "Motorcycle 125cc-500cc", 5: "Motorcycle over 500cc", 8: "Taxi/Private hire",
        9: "Car", 10: "Minibus", 11: "Bus or coach", 16: "Ridden horse",
        17: "Agricultural vehicle", 18: "Tram", 19: "Van/Light Goods",
        20: "Medium Goods (3.5t-7.5t)", 21: "Heavy Goods (7.5t+)", 22: "Mobility scooter",
        23: "E-scooter", 90: "Other vehicle", 97: "Motorcycle", 98: "Goods vehicle"
    }
    df['vehicle_type_label'] = df['vehicle_type'].replace(-1, np.nan).map(vehicle_type_map).fillna("Other")
    df['propulsion_code'] = df['propulsion_code'].replace(-1, np.nan)
    prop_map = {1: "Petrol", 2: "Heavy oil", 3: "Electric", 8: "Hybrid electric"}
    df['propulsion_label'] = df['propulsion_code'].map(prop_map).fillna("Other/Unknown")
    df['journey_purpose_of_driver'] = df['journey_purpose_of_driver'].replace(-1, np.nan)
    jp_map = {
        1: "Journey as part of work",
        2: "Commuting to or from work",
        7: "Education and escort",
        8: "Emergency vehicle",
        9: "Personal/Leisure"
    }
    df['journey_purpose_label'] = df['journey_purpose_of_driver'].map(jp_map).fillna("Other/Leisure")
    
    return df

# Load datasets
df_col = load_collision_data()
df_cas = load_casualty_data()
df_veh = load_vehicle_data()

# 4. Load ML Model and Metadata
@st.cache_resource
def load_ml_model():
    model_file = 'road_safety_model.joblib'
    meta_file = 'model_metadata.json'
    
    if os.path.exists(model_file) and os.path.exists(meta_file):
        model = joblib.load(model_file)
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        return model, metadata
    else:
        return None, None

model, metadata = load_ml_model()

# 5. Sidebar Branding & Info
with st.sidebar:
    st.markdown("<div style='text-align: center; padding-top: 10px;'><h2 style='color: #ef8a62; margin-bottom: 0px;'>UK Road Safety</h2><span style='color: #a0aec0; font-size: 0.95rem;'>Jan - Jun 2025 Data</span></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📊 Dataset Overview")
    st.markdown(f"**Total Collisions:** `{len(df_col):,}`")
    if df_cas is not None:
        st.markdown(f"**Total Casualties:** `{len(df_cas):,}`")
    if df_veh is not None:
        st.markdown(f"**Total Vehicles:** `{len(df_veh):,}`")
        
    st.markdown("### 🔧 Model Details")
    if metadata is not None:
        st.markdown(f"**Primary Model:** `{metadata['model_name']}`")
        st.markdown(f"**CV F1 Score:** `{metadata['cv_results'][metadata['model_name']]['cv_f1']:.3f}`")
        st.markdown(f"**Test ROC AUC:** `{metadata['cv_results'][metadata['model_name']]['test_roc_auc']:.3f}`")
    else:
        st.warning("Model and metadata files not found. Please run 'train_model.py' first.")
        
    st.markdown("---")
    st.markdown("### 📘 Project Context")
    st.markdown("Developed as part of the MSc Data Science program. Grounded in visual encoding theories (Cleveland & McGill, 1985; Tufte, 1983) and interactive details-on-demand (Shneiderman, 1996).")

# 6. Main Interface Layout
st.markdown("<h1 class='main-title'>UK Road Safety Machine Learning Platform</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Accident hotspots, vulnerability profiles, and predictive collision risk analytics</div>", unsafe_allow_html=True)

# Define Tabs
tab_overview, tab_predictor, tab_model, tab_vulnerable = st.tabs([
    "📂 Collisions Overview", 
    "🔮 Interactive Predictor", 
    "📈 Model Performance", 
    "🚴 Vulnerability & Demographics"
])

# ----------------- TAB 1: OVERVIEW -----------------
with tab_overview:
    # Key KPI Metrics
    c_acc, c_cas, c_ksi, c_avg_speed = st.columns(4)
    
    total_collisions = len(df_col)
    total_casualties = df_col['number_of_casualties'].sum()
    ksi_rate = df_col['is_severe'].mean() * 100
    avg_speed = df_col['speed_limit'].mean()
    
    with c_acc:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Accidents</div>
            <div class='metric-value val-orange'>{total_collisions:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_cas:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Casualties</div>
            <div class='metric-value val-yellow'>{total_casualties:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_ksi:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>KSI (Severe) Share</div>
            <div class='metric-value val-red'>{ksi_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_avg_speed:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Avg Speed Limit</div>
            <div class='metric-value val-teal'>{avg_speed:.1f} mph</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Map & Heatmap
    st.markdown("### 🗺️ Geographic & Temporal Distribution")
    col1, col2 = st.columns([1.1, 0.9])
    
    with col1:
        st.markdown("#### Collision Hotspots (Sample of 3,000 Accidents)")
        # Sample data for fast mapping
        map_sample = df_col.dropna(subset=['latitude', 'longitude']).sample(min(3000, len(df_col)), random_state=42)
        fig_map = px.scatter_mapbox(
            map_sample,
            lat="latitude",
            lon="longitude",
            color="collision_severity_label",
            size="number_of_casualties",
            color_discrete_map={"Fatal": "#b2182b", "Serious": "#ef8a62", "Slight": "#67a9cf"},
            mapbox_style="carto-darkmatter",
            zoom=5,
            center={"lat": 54.3, "lon": -2.5},
            hover_data={
                "date": True, "time": True, "speed_limit": True, 
                "number_of_vehicles": True, "number_of_casualties": True, 
                "weather_conditions_label": True, "latitude": False, "longitude": False
            }
        )
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(11, 19, 43, 0.7)")
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col2:
        st.markdown("#### Heatmap: Weekday vs Hour of Collision")
        piv = (df_col.dropna(subset=['hour'])
               .pivot_table(index='day_of_week_label', columns='hour', values='collision_index', aggfunc='count')
               .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']))
        
        fig_heat = px.imshow(
            piv,
            labels=dict(x="Hour (24h)", y="Day of Week", color="Accidents"),
            color_continuous_scale="magma_r"
        )
        fig_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(title="Accidents", thickness=15),
            margin=dict(t=30, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Time series trends
    st.markdown("### 📈 Daily Collision Trends & Speed Limit Analysis")
    col3, col4 = st.columns([1.1, 0.9])
    
    with col3:
        st.markdown("#### Daily Collisions Timeline (with Range Slider)")
        daily_trend = df_col.groupby('date_only').size().reset_index(name='collision_count')
        fig_ts = px.line(
            daily_trend,
            x='date_only',
            y='collision_count',
            labels={'date_only': 'Date', 'collision_count': 'Daily Collisions'},
            color_discrete_sequence=['#ef8a62']
        )
        fig_ts.update_layout(
            xaxis=dict(rangeslider_visible=True),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
    with col4:
        st.markdown("#### Severity Share (%) by Speed Limit Zone")
        speed_crosstab = pd.crosstab(df_col['speed_limit'], df_col['collision_severity_label'], normalize='index') * 100
        speed_crosstab = speed_crosstab.reindex(columns=['Slight', 'Serious', 'Fatal']).dropna()
        
        fig_speed = go.Figure()
        colors_sev = {'Slight': '#67a9cf', 'Serious': '#ef8a62', 'Fatal': '#b2182b'}
        for col_name in ['Slight', 'Serious', 'Fatal']:
            fig_speed.add_trace(go.Bar(
                name=col_name,
                x=speed_crosstab.index.astype(int).astype(str),
                y=speed_crosstab[col_name],
                marker_color=colors_sev[col_name]
            ))
            
        fig_speed.update_layout(
            barmode='stack',
            xaxis_title="Speed Limit (mph)",
            yaxis_title="Percentage (%)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_speed, use_container_width=True)

# ----------------- TAB 2: PREDICTOR -----------------
with tab_predictor:
    st.markdown("### 🔮 Real-Time Collision Severity Predictor")
    st.markdown("Input environmental and collision details below to estimate the probability of the accident resulting in a **Severe** (Fatal or Serious) outcome.")
    
    if model is None:
        st.error("Predictive model not loaded. Please train the model first.")
    else:
        # Create input form
        with st.form("predictor_form"):
            col_in1, col_in2, col_in3 = st.columns(3)
            
            with col_in1:
                in_speed = st.selectbox("Speed Limit (mph)", [20, 30, 40, 50, 60, 70], index=1)
                in_hour = st.slider("Hour of Day (0-23)", 0, 23, 17)
                in_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=20, value=2)
                
            with col_in2:
                in_urban = st.selectbox("Urban or Rural Area", ["Urban", "Rural", "Unallocated"], index=0)
                in_light = st.selectbox("Light Conditions", [
                    "Daylight", 
                    "Darkness - lights lit", 
                    "Darkness - lights unlit", 
                    "Darkness - no lights", 
                    "Darkness - lights detail unknown"
                ], index=0)
                in_day = st.selectbox("Day of Week", [
                    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
                ], index=4)
                
            with col_in3:
                in_weather = st.selectbox("Weather Conditions", [
                    "Fine no high winds",
                    "Raining no high winds",
                    "Snowing no high winds",
                    "Fine + high winds",
                    "Raining + high winds",
                    "Snowing + high winds",
                    "Fog or mist",
                    "Other",
                    "Unknown"
                ], index=0)
                in_surface = st.selectbox("Road Surface Conditions", [
                    "Dry", 
                    "Wet or damp", 
                    "Snow", 
                    "Frost or ice", 
                    "Flood over 3cm deep", 
                    "Oil or wet mud", 
                    "Road sign"
                ], index=0)
            
            st.markdown("##### 🚴 Involved Parties & Demographics")
            col_vru1, col_vru2, col_vru3, col_vru4 = st.columns(4)
            with col_vru1:
                in_motorcycle = st.checkbox("Motorcycle Involved", value=False)
            with col_vru2:
                in_pedal_cycle = st.checkbox("Bicycle involved", value=False)
            with col_vru3:
                in_pedestrian = st.checkbox("Pedestrian involved", value=False)
            with col_vru4:
                in_hgv_or_bus = st.checkbox("HGV / Bus involved", value=False)
                
            col_age1, col_age2 = st.columns(2)
            with col_age1:
                in_driver_age = st.slider("Driver Age Profile (Min, Max)", min_value=16, max_value=99, value=(25, 65))
            with col_age2:
                in_casualty_age = st.slider("Casualty Age Profile (Min, Max)", min_value=0, max_value=99, value=(18, 70))
                
            submit_pred = st.form_submit_button("🔥 Calculate Collision Risk", use_container_width=True)
            
        if submit_pred:
            # Prepare Input DataFrame
            input_df = pd.DataFrame([{
                'speed_limit': float(in_speed),
                'hour': float(in_hour),
                'number_of_vehicles': float(in_vehicles),
                'urban_or_rural_label': in_urban,
                'light_conditions_label': in_light,
                'weather_conditions_label': in_weather,
                'road_surface_conditions_label': in_surface,
                'day_of_week_label': in_day,
                'has_motorcycle': int(in_motorcycle),
                'has_pedal_cycle': int(in_pedal_cycle),
                'has_hgv_or_bus': int(in_hgv_or_bus),
                'has_pedestrian': int(in_pedestrian),
                'driver_age_min': float(in_driver_age[0]),
                'driver_age_max': float(in_driver_age[1]),
                'casualty_age_min': float(in_casualty_age[0]),
                'casualty_age_max': float(in_casualty_age[1])
            }])
            
            # Predict
            prob = model.predict_proba(input_df)[0][1]
            risk_pct = prob * 100
            
            # Risk Category determination
            # Base KSI rate is ~26%. We scale the risk levels:
            # Low: < 22%
            # Medium: 22% - 38%
            # High: > 38%
            if risk_pct < 22:
                risk_class = "risk-low"
                risk_label = "LOW RISK"
                risk_color = "#2a9d8f"
            elif risk_pct <= 38:
                risk_class = "risk-medium"
                risk_label = "MEDIUM RISK"
                risk_color = "#e0a106"
            else:
                risk_class = "risk-high"
                risk_label = "HIGH RISK"
                risk_color = "#b2182b"
                
            col_res1, col_res2 = st.columns([1, 1])
            
            with col_res1:
                st.markdown(f"""
                <div class='risk-banner {risk_class}'>
                    <div style='font-size: 1.1rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.8;'>Severity Risk Assessment</div>
                    <div style='font-size: 3.2rem; font-weight: 900; margin: 10px 0;'>{risk_label}</div>
                    <div style='font-size: 1.8rem; font-weight: 700; opacity: 0.9;'>{risk_pct:.1f}% Probability</div>
                    <div style='font-size: 0.85rem; margin-top: 15px; opacity: 0.7; font-style: italic;'>
                        (Proportion of similar accidents that result in a Fatal or Serious injury)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_res2:
                st.markdown("#### ⚙️ Input Conditions Summary")
                
                # Format involved parties
                parties = []
                if in_motorcycle: parties.append("Motorcycle")
                if in_pedal_cycle: parties.append("Bicycle")
                if in_pedestrian: parties.append("Pedestrian")
                if in_hgv_or_bus: parties.append("HGV/Bus")
                parties_str = ", ".join(parties) if parties else "None (Car-only/Other)"
                
                st.markdown(f"""
                - **Speed Limit:** `{in_speed} mph`
                - **Area Class:** `{in_urban}`
                - **Time of Day:** `{in_hour:02d}:00`
                - **Vehicles Involved:** `{in_vehicles} vehicle(s)`
                - **Ambient Lighting:** `{in_light}`
                - **Atmospheric Weather:** `{in_weather}`
                - **Tarmac Surface:** `{in_surface}`
                - **Temporal Slot:** `{in_day}`
                - **Involved Parties:** `{parties_str}`
                - **Driver Age Profile:** `{in_driver_age[0]} - {in_driver_age[1]} yrs`
                - **Casualty Age Profile:** `{in_casualty_age[0]} - {in_casualty_age[1]} yrs`
                """)
                
                # Gauge plot
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_pct,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "KSI Risk Probability (%)", 'font': {'size': 16, 'color': '#ffffff'}},
                    number = {'suffix': "%", 'font': {'size': 24, 'color': risk_color}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                        'bar': {'color': risk_color},
                        'bgcolor': "rgba(255, 255, 255, 0.1)",
                        'borderwidth': 2,
                        'bordercolor': "#ffffff",
                        'steps': [
                            {'range': [0, 22], 'color': 'rgba(42, 157, 143, 0.2)'},
                            {'range': [22, 38], 'color': 'rgba(224, 161, 6, 0.2)'},
                            {'range': [38, 100], 'color': 'rgba(178, 24, 43, 0.2)'}
                        ],
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=200,
                    margin=dict(t=30, b=0, l=10, r=10)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

# ----------------- TAB 3: MODEL PERFORMANCE -----------------
with tab_model:
    st.markdown("### 📈 Model Evaluation Metrics")
    
    if metadata is None:
        st.error("No model metadata found. Train the model to generate reports.")
    else:
        # Display CV summary and Model selector comparison
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("#### Model Classifier Selection & Training Summary")
            cv_data = metadata['cv_results']
            
            # Format CV results for visual comparison table
            cv_df = pd.DataFrame({
                'Model Name': list(cv_data.keys()),
                'CV Mean F1-Score': [cv_data[m]['cv_f1'] for m in cv_data],
                'Test F1-Score': [cv_data[m]['test_f1'] for m in cv_data],
                'Test ROC-AUC': [cv_data[m]['test_roc_auc'] for m in cv_data]
            })
            
            st.dataframe(cv_df.style.highlight_max(axis=0, color='rgba(239, 138, 98, 0.2)'), hide_index=True)
            
            st.markdown("##### 📝 Classification Report (Holdout Test Set)")
            rep = metadata['classification_report']
            rep_df = pd.DataFrame(rep).T.iloc[:-3, :-1] # Select Slight, Severe rows, drop support and averages
            st.dataframe(rep_df.style.format("{:.3f}"), use_container_width=True)
            
            st.markdown("""
            > [!NOTE]
            > Due to the extreme noise in road collision reports, a **F1 score of ~0.40** for the Severe class represents a substantial, statistically meaningful predictive power (verified via Stratified Cross-Validation) compared to random guessing.
            """)
            
        with col_m2:
            st.markdown("#### Confusion Matrix Heatmap")
            cm = np.array(metadata['confusion_matrix'])
            fig_cm = px.imshow(
                cm,
                x=['Predicted Slight', 'Predicted Severe (KSI)'],
                y=['Actual Slight', 'Actual Severe (KSI)'],
                color_continuous_scale="Blues",
                text_auto=True
            )
            fig_cm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                margin=dict(t=20, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        st.markdown("---")
        
        # Feature importances & ROC Curves
        col_m3, col_m4 = st.columns(2)
        
        with col_m3:
            st.markdown("#### 🏆 Feature Importance (Gini/Information Gain)")
            feat_imp = pd.DataFrame(metadata['feature_importances']).head(10)
            fig_imp = px.bar(
                feat_imp,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale="GnBu_r",
                labels={'importance': 'Relative Importance', 'feature': 'Feature'}
            )
            fig_imp.update_layout(
                yaxis={'categoryorder':'total ascending'},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                margin=dict(t=20, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with col_m4:
            st.markdown("#### 📈 Receiver Operating Characteristic (ROC)")
            fpr = metadata['roc_curve']['fpr']
            tpr = metadata['roc_curve']['tpr']
            auc_val = metadata['roc_curve']['auc']
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc_val:.3f})', line=dict(color='#ef8a62', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline (0.50)', line=dict(color='grey', dash='dash')))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=20, b=10, l=10, r=10),
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

# ----------------- TAB 4: DEMOGRAPHICS -----------------
with tab_vulnerable:
    st.markdown("### 🚴 Casualty Demographics & Road User Vulnerability")
    
    if df_cas is None:
        st.warning("Casualties dataset not found. Demographic views are disabled.")
    else:
        # Columns
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("#### Age & Sex Distribution of Casualties")
            # Filter valid age/sex
            ca_df = df_cas[df_cas['sex_of_casualty_label'].isin(['Male','Female']) & df_cas['age_band_of_casualty_label'].notna()]
            age_order = ["0 - 5", "6 - 10", "11 - 15", "16 - 20", "21 - 25", "26 - 35", "36 - 45", "46 - 55", "56 - 65", "66 - 75", "Over 75"]
            
            ca_grouped = ca_df.groupby(['age_band_of_casualty_label', 'sex_of_casualty_label']).size().reset_index(name='count')
            # Sort age bands properly
            ca_grouped['age_band_of_casualty_label'] = pd.Categorical(ca_grouped['age_band_of_casualty_label'], categories=age_order, ordered=True)
            ca_grouped = ca_grouped.dropna().sort_values('age_band_of_casualty_label')
            
            fig_dem = px.bar(
                ca_grouped,
                x='age_band_of_casualty_label',
                y='count',
                color='sex_of_casualty_label',
                barmode='group',
                color_discrete_map={'Male': '#1f3b57', 'Female': '#ef8a62'},
                labels={'age_band_of_casualty_label': 'Age Band', 'count': 'Casualties', 'sex_of_casualty_label': 'Sex'}
            )
            fig_dem.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=20, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_dem, use_container_width=True)
            
        with col_v2:
            st.markdown("#### Road User Type Casualties")
            user_counts = df_cas['casualty_class_label'].value_counts().reset_index()
            user_counts.columns = ['class', 'count']
            
            fig_class = px.pie(
                user_counts,
                names='class',
                values='count',
                color_discrete_sequence=['#1f3b57', '#ef8a62', '#2a9d8f'],
                hole=0.4
            )
            fig_class.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=20, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_class, use_container_width=True)
            
        st.markdown("---")
        
        # Vehicle propulsion analysis & journey purpose
        col_v3, col_v4 = st.columns(2)
        
        with col_v3:
            st.markdown("#### Journey Purpose Severity Share (%)")
            if df_veh is not None:
                # Merge vehicle type with collision severity
                merged_v = pd.merge(df_veh, df_col[['collision_index', 'collision_severity_label']], on='collision_index', how='inner')
                
                jp_cross = pd.crosstab(merged_v['journey_purpose_label'], merged_v['collision_severity_label'], normalize='index') * 100
                jp_cross = jp_cross.reindex(columns=['Slight', 'Serious', 'Fatal']).dropna()
                
                fig_jp = go.Figure()
                for c_name in ['Slight', 'Serious', 'Fatal']:
                    fig_jp.add_trace(go.Bar(
                        name=c_name,
                        x=jp_cross.index,
                        y=jp_cross[c_name],
                        marker_color=colors_sev[c_name]
                    ))
                fig_jp.update_layout(
                    barmode='stack',
                    xaxis_title="Journey Purpose",
                    yaxis_title="Percentage (%)",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=20, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_jp, use_container_width=True)
            else:
                st.warning("Vehicle dataset missing. Journey purpose analysis disabled.")
                
        with col_v4:
            st.markdown("#### Deprivation Profile (IMD Decile 1=Most Deprived)")
            # Deprivation decile analysis
            dep_df = df_cas.dropna(subset=['casualty_imd_decile'])
            dep_counts = dep_df['casualty_imd_decile'].value_counts().sort_index().reset_index()
            dep_counts.columns = ['decile', 'count']
            
            fig_dep = px.bar(
                dep_counts,
                x=dep_counts['decile'].astype(int).astype(str),
                y='count',
                color='count',
                color_continuous_scale="reds",
                labels={'x': 'IMD Decile', 'count': 'Casualties'}
            )
            fig_dep.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                margin=dict(t=20, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_dep, use_container_width=True)
