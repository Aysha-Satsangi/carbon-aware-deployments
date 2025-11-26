
# streamlit run carbon_dashboard.py

import streamlit as st
import pandas as pd
import requests
import asyncio
import aiohttp
import time
import statistics
from datetime import datetime, timedelta, timezone
import os
import pytz
from streamlit_autorefresh import st_autorefresh
import altair as alt
import tensorflow as tf
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
# from forecasting_utils import show_regional_comparison, show_carbon_savings_calculator

# ============================================================
# FEATURE 1: REGIONAL COMPARISON TABLE
# ============================================================

def show_regional_comparison(all_forecasts_data, zones, REGION_MAPPING):
    """Display a comprehensive regional comparison dashboard"""
    st.markdown("---")
    st.subheader("🌍 Regional Comparison Dashboard")
    
    st.markdown("**Compare all 8 regions at a glance:**")
    
    # Create comparison data
    regional_data = {
        'Zone': [],
        'Region': [],
        'Current Carbon': [],
        'Min (24h)': [],
        'Max (24h)': [],
        'Avg (24h)': [],
        'Variation': [],
        'Best Time (IST)': [],
        'Status': []
    }
    
    # Fill in data from all zones
    for zone in zones:
        if zone in all_forecasts_data:
            forecast = all_forecasts_data[zone]
            region_name = REGION_MAPPING.get(zone, {}).get("name", zone)
            current = forecast['current']
            carbon = forecast['carbon']
            times_ist = forecast['times_ist']
            
            min_carbon = carbon.min()
            max_carbon = carbon.max()
            avg_carbon = carbon.mean()
            variation = ((max_carbon - min_carbon) / min_carbon) * 100
            best_idx = np.argmin(carbon)
            best_time = times_ist[best_idx].strftime('%H:%M')
            
            # Status indicator
            if avg_carbon < 250:
                status = "🟢 Very Green"
            elif avg_carbon < 300:
                status = "🟡 Good"
            elif avg_carbon < 350:
                status = "🟠 Moderate"
            else:
                status = "🔴 High Carbon"
            
            regional_data['Zone'].append(zone)
            regional_data['Region'].append(region_name)
            regional_data['Current Carbon'].append(f"{current:.0f}")
            regional_data['Min (24h)'].append(f"{min_carbon:.0f}")
            regional_data['Max (24h)'].append(f"{max_carbon:.0f}")
            regional_data['Avg (24h)'].append(f"{avg_carbon:.0f}")
            regional_data['Variation'].append(f"{variation:.1f}%")
            regional_data['Best Time (IST)'].append(best_time)
            regional_data['Status'].append(status)
    
    # Create and display DataFrame
    regional_df = pd.DataFrame(regional_data)
    
    st.dataframe(
        regional_df,
        use_container_width=True,
        hide_index=True,
    )
    
    # Insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Find greenest zone
        current_values = [float(regional_df.iloc[i]['Current Carbon']) for i in range(len(regional_df))]
        greenest_idx = np.argmin(current_values)
        greenest_zone = regional_df.iloc[greenest_idx]
        
        st.success(f"""
        🟢 **GREENEST REGION NOW**
        
        **{greenest_zone['Region']}** ({greenest_zone['Zone']})
        
        Current: {greenest_zone['Current Carbon']} gCO₂/kWh
        Status: {greenest_zone['Status']}
        """)
    
    with col2:
        # Find best variation (most stable)
        variation_values = [float(regional_df.iloc[i]['Variation'].rstrip('%')) for i in range(len(regional_df))]
        most_stable_idx = np.argmin(variation_values)
        most_stable = regional_df.iloc[most_stable_idx]
        
        st.info(f"""
        📊 **MOST STABLE GRID**
        
        **{most_stable['Region']}** ({most_stable['Zone']})
        
        Variation: {most_stable['Variation']}
        Avg Carbon: {most_stable['Avg (24h)']} gCO₂/kWh
        """)
    
    with col3:
        # Find highest variation (best opportunity)
        best_opportunity_idx = np.argmax(variation_values)
        best_opportunity = regional_df.iloc[best_opportunity_idx]
        
        st.warning(f"""
        ⚡ **BEST SAVINGS OPPORTUNITY**
        
        **{best_opportunity['Region']}** ({best_opportunity['Zone']})
        
        Variation: {best_opportunity['Variation']}
        Deploy at {best_opportunity['Best Time (IST)']} IST!
        """)


# ============================================================
# FEATURE 2: CARBON SAVINGS CALCULATOR
# ============================================================

def show_carbon_savings_calculator(all_forecasts_data, zones, REGION_MAPPING):
    """Interactive carbon savings calculator with ROI analysis"""
    st.markdown("---")
    st.subheader("💰 Carbon Savings Calculator - ROI Analysis")
    
    st.markdown("**Calculate how much CO₂ you can save by deploying at optimal times:**")
    
    # Create calculator interface
    calc_col1, calc_col2, calc_col3 = st.columns(3)
    
    with calc_col1:
        workload_power = st.number_input(
            "Workload Power (Watts)",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=1000,
            key="calc_workload_power"
            # help="Total power consumption of your workload"
        )
    
    with calc_col2:
        # IMPORTANT: ALL VALUES MUST BE FLOAT (NOT INT)
        workload_hours = st.number_input(
            "Duration (Hours)",
            min_value=0.5,
            max_value=24.0,
            value=8.0,
            step=0.5,
            key="calc_workload_hours"
            # help="How long your workload will run"
        )
    
    with calc_col3:
        selected_region_calc = st.selectbox(
            "Select Region",
            zones,
            key="calc_zone",
            help="Which cloud region will you deploy to?"
        )
    
    st.markdown("---")
    
    # Perform calculation
    if selected_region_calc in all_forecasts_data:
        forecast_calc = all_forecasts_data[selected_region_calc]
        region_name = REGION_MAPPING.get(selected_region_calc, {}).get("name", selected_region_calc)
        
        # Get current and best carbon values
        current_carbon = forecast_calc['current']
        best_carbon = forecast_calc['carbon'].min()
        worst_carbon = forecast_calc['carbon'].max()
        avg_carbon = forecast_calc['carbon'].mean()
        
        # Find best time
        best_idx = np.argmin(forecast_calc['carbon'])
        best_time_ist = forecast_calc['times_ist'][best_idx].strftime('%H:%M IST')
        best_time_utc = forecast_calc['times_utc'][best_idx].strftime('%H:%M UTC')
        
        # Calculate emissions
        workload_kw = workload_power / 1000
        energy_kwh = workload_kw * workload_hours
        
        current_emissions = (energy_kwh * current_carbon) / 1000  # kg CO2
        best_emissions = (energy_kwh * best_carbon) / 1000
        worst_emissions = (energy_kwh * worst_carbon) / 1000
        avg_emissions = (energy_kwh * avg_carbon) / 1000
        
        # Calculate savings
        savings_vs_current = current_emissions - best_emissions
        savings_percentage = (savings_vs_current / current_emissions) * 100
        worst_vs_best = worst_emissions - best_emissions
        worst_percentage = (worst_vs_best / best_emissions) * 100
        
        # Display results
        st.markdown("### 📊 Deployment Scenarios")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Deploy NOW", f"{current_emissions:.2f} kg CO₂")
            st.caption(f"📊 {energy_kwh:.1f} kWh")
        
        with col2:
            st.metric("Deploy at BEST time", f"{best_emissions:.2f} kg CO₂", delta=f"-{savings_vs_current:.2f} kg")
            st.caption(f"⏱️ {best_time_ist}")
        
        with col3:
            st.metric("Deploy at WORST time", f"{worst_emissions:.2f} kg CO₂", delta=f"+{worst_vs_best:.2f} kg", delta_color="inverse")
            st.caption(f"🔴 {worst_percentage:.1f}% worse")
        
        with col4:
            st.metric("Average (24h)", f"{avg_emissions:.2f} kg CO₂")
            st.caption(f"📈 Avg: {avg_carbon:.0f}")
        
        st.markdown("---")
        
        # Visual comparison
        st.markdown("### 🎯 Savings Potential")
        
        scenarios = pd.DataFrame({
            'Scenario': ['Deploy NOW', 'Deploy at BEST', 'Deploy at WORST'],
            'CO₂ Emissions': [current_emissions, best_emissions, worst_emissions],
        })
        
        # Create horizontal bar chart
        fig = alt.Chart(scenarios).mark_bar().encode(
            y=alt.Y('Scenario:N', sort=['Deploy NOW', 'Deploy at BEST', 'Deploy at WORST']),
            x=alt.X('CO₂ Emissions:Q', title='CO₂ Emissions (kg)'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Deploy NOW', 'Deploy at BEST', 'Deploy at WORST'],
                range=['#f39c12', '#2ecc71', '#e74c3c']
            ))
        ).properties(height=300, width=600)
        
        st.altair_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Key insights
        st.markdown("### 💡 Key Insights")
        
        col_insight1, col_insight2, col_insight3 = st.columns(3)
        
        with col_insight1:
            st.success(f"""
            ✅ **RECOMMENDED ACTION**
            
            Deploy at **{best_time_ist}**
            
            Save **{savings_percentage:.1f}%** carbon
            
            Equivalent to: **{savings_vs_current:.2f} kg CO₂** saved
            """)
        
        with col_insight2:
            # Convert CO2 to real-world equivalents
            trees_equivalent = savings_vs_current / 21
            car_miles = savings_vs_current / 0.41
            
            st.info(f"""
            🌍 **ENVIRONMENTAL IMPACT**
            
            Saving {savings_vs_current:.2f} kg CO₂ = 
            
            🌳 {trees_equivalent:.1f} trees planted
            
            🚗 {car_miles:.0f} miles NOT driven
            """)
        
        with col_insight3:
            # Annual impact if deployed daily
            annual_savings = savings_vs_current * 365
            
            st.warning(f"""
            📈 **IF DEPLOYED EVERY DAY**
            
            Annual savings: {annual_savings:.0f} kg CO₂
            
            That's equivalent to:
            
            🌳 {annual_savings/21:.0f} trees/year
            
            🚗 {annual_savings/0.41:.0f} miles not driven
            """)
        
        st.markdown("---")
        
        # Breakdown by time
        st.markdown("### 📅 Hour-by-Hour Savings Potential")
        
        hourly_breakdown = []
        for idx, (time_ist, carbon_val) in enumerate(zip(forecast_calc['times_ist'], forecast_calc['carbon'])):
            emissions = (energy_kwh * carbon_val) / 1000
            vs_best = emissions - best_emissions
            vs_current_pct = ((carbon_val - current_carbon) / current_carbon) * 100
            
            hourly_breakdown.append({
                'Hour': idx + 1,
                'Time (IST)': time_ist.strftime('%H:%M'),
                'CO₂ (kg)': f"{emissions:.2f}",
                'vs Best': f"+{vs_best:.2f} kg" if vs_best > 0 else f"-{abs(vs_best):.2f} kg",
                'vs Now': f"{vs_current_pct:+.1f}%"
            })
        
        hourly_df = pd.DataFrame(hourly_breakdown)
        
        col_hourly1, col_hourly2 = st.columns([1, 2])
        
        with col_hourly1:
            st.dataframe(hourly_df.head(12), use_container_width=True, hide_index=True)
        
        with col_hourly2:
            st.info("""
            **💰 How to Use:**
            
            1. Enter workload power (Watts)
            2. Enter duration (Hours)
            3. Select region
            4. See CO₂ savings!
            
            **Examples:**
            
            - 1 Server (400W, 8h) = ~3 kg CO₂/day
            - Small DC (100kW, 24h) = ~2,400 kg CO₂/day
            - Large DC (1MW, 24h) = ~24,000 kg CO₂/day
            """)

# Page configuration
st.set_page_config(
    page_title="EcoDeploy Dashboard",
    page_icon="🌱",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("🌱 EcoDeploy Navigation")
page = st.sidebar.selectbox(
    "Choose Dashboard",
    ["🏠 Home", "🌍 Carbon Intensity", "⚖️ Carbon + Latency", " ML Forecasting", "📖 How It Works", "🔧 API Setup"]
)

# ============================================================
# PATHS FOR ML FORECASTING
# ============================================================
PROCESSED = Path("data/processed")
MODELS = Path("data/models")
SCALERS = Path("data/scalers")
PLOTS = Path("data/plots")

zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
IST = pytz.timezone("Asia/Kolkata")

# ============================================================
# LOAD ML MODELS (CACHING FOR PERFORMANCE)
# ============================================================
@st.cache_resource
def load_ml_models():
    """Load all trained models and scalers"""
    models_dict = {}
    scalers_dict = {}
    
    for zone in zones:
        try:
            model = tf.keras.models.load_model(MODELS / f"{zone}_best.keras")
            with open(SCALERS / f"{zone}_scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
            models_dict[zone] = model
            scalers_dict[zone] = scaler
        except Exception as e:
            st.warning(f"Could not load {zone}: {e}")
    
    return models_dict, scalers_dict

# ============================================================
# GENERATE LIVE FORECASTS
# ============================================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_live_forecast(zone):
    """Generate 24-hour forecast for a zone"""
    try:
        models_dict, scalers_dict = load_ml_models()
        
        if zone not in models_dict:
            return None
        
        # Load processed data
        df = pd.read_csv(PROCESSED / f"{zone}_processed.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Get last 24 hours
        latest_24h = df.tail(24).drop('datetime', axis=1).values
        
        if len(latest_24h) < 24:
            return None
        
        # Make prediction
        model = models_dict[zone]
        scaler = scalers_dict[zone]
        
        X = latest_24h.reshape(1, 24, latest_24h.shape[1])
        y_pred_scaled = model.predict(X, verbose=0)[0]
        
        # Inverse transform
        y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Generate timestamps
        last_time = df['datetime'].iloc[-1]
        forecast_times_utc = [last_time + timedelta(hours=i+1) for i in range(24)]
        forecast_times_ist = [t.replace(tzinfo=pytz.UTC).astimezone(IST) for t in forecast_times_utc]
        
        # Current carbon
        current_carbon = scaler.inverse_transform([[df['carbon'].iloc[-1]]])[0][0]
        
        return {
            'times_ist': forecast_times_ist,
            'times_utc': forecast_times_utc,
            'carbon': y_pred_original,
            'current': current_carbon
        }
    except Exception as e:
        return None

# Common functions (YOUR EXISTING CODE)
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_carbon_intensity(zone):
    """Fetch real-time carbon intensity for a zone"""
    try:
        token = os.getenv("ELECTRICITY_MAP_TOKEN", st.secrets.get("ELECTRICITY_MAP_TOKEN", ""))
        if not token:
            return {"error": "No API token"}
            
        url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"
        headers = {"auth-token": token}
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "carbonIntensity": data.get("carbonIntensity", "N/A"),
                "updatedAt": data.get("datetime", "N/A"),
                "zone": zone
            }
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


async def measure_latency_async(url, samples=3):
    """Measure latency to a URL asynchronously"""
    latencies = []
    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(samples):
                start = time.perf_counter()
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)):
                        pass
                    elapsed = (time.perf_counter() - start) * 1000
                    latencies.append(elapsed)
                except:
                    latencies.append(5000)  # 5 second timeout
        
        if latencies:
            return {
                "avg_latency": statistics.mean(latencies),
                "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0]
            }
    except Exception as e:
        return {"avg_latency": 5000, "p95_latency": 5000}


def run_latency_measurements(endpoints):
    """Run latency measurements for multiple endpoints"""
    async def measure_all():
        tasks = []
        for zone, url in endpoints.items():
            tasks.append(measure_latency_async(url))
        results = await asyncio.gather(*tasks)
        return dict(zip(endpoints.keys(), results))
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(measure_all())
    except:
        # Fallback for environments where asyncio is problematic
        return {zone: {"avg_latency": "N/A", "p95_latency": "N/A"} for zone in endpoints.keys()}


# Region mappings
ZONES_MAPPING = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
REGION_MAPPING = {
    "DE": {"name": "Germany (Frankfurt)", "endpoint": "https://ec2.eu-central-1.amazonaws.com/"},
    "US-MIDA-PJM": {"name": "US East (Virginia)", "endpoint": "https://ec2.us-east-1.amazonaws.com/"},
    "US-NW-PACW": {"name": "US West (Oregon)", "endpoint": "https://ec2.us-west-2.amazonaws.com/"},
    "IE": {"name": "Ireland", "endpoint": "https://ec2.eu-west-1.amazonaws.com/"},
    "SG": {"name": "Singapore", "endpoint": "https://ec2.ap-southeast-1.amazonaws.com/"},
    "BE": {"name": "Belgium (London)", "endpoint": "https://ec2.eu-west-2.amazonaws.com/"},
    "US-MIDW-MISO": {"name": "US Central", "endpoint": "https://ec2.us-central-1.amazonaws.com/"},
    "JP-TK": {"name": "Japan (Tokyo)", "endpoint": "https://ec2.ap-northeast-1.amazonaws.com/"}
}


# Main content based on selected page
if page == "🏠 Home":
    st.title("🌱 EcoDeploy - Carbon-Aware Cloud Deployment Dashboard")
    
    st.markdown("""
    ## Welcome to EcoDeploy!
    
    **EcoDeploy** is a tool that helps you make **environmentally conscious cloud deployment decisions** by providing real-time insights into:
    
    - 🌍 **Carbon Intensity**: Live CO₂ emissions from electricity grids worldwide
    - ⚡ **Network Latency**: Response times to different cloud regions
    -  **ML Forecasting**: 24-hour ahead carbon predictions using deep learning
    - 🎯 **Smart Recommendations**: Optimal regions balancing sustainability and performance
    
    ### What This Dashboard Offers:
    
    🔹 **Carbon Intensity Dashboard** - Monitor real-time carbon emissions across 8 global regions  
    🔹 **Combined Carbon + Latency View** - See both metrics together for informed decisions  
    🔹 **ML Forecasting** - Predict carbon intensity 24 hours ahead with CNN-LSTM models  
    🔹 **Educational Content** - Learn how carbon-aware computing works  
    🔹 **API Integration** - Uses ElectricityMap for live carbon data  
    
    ### Quick Start:
    1. Navigate using the sidebar to explore different dashboards
    2. View real-time carbon intensity data from global electricity grids
    3. Compare network latency to different cloud regions
    4. **Check ML Forecasting to see 24-hour predictions!**
    5. Get deployment recommendations based on your priorities
    
    ---
    *Choose a dashboard from the sidebar to get started!*
    """)
    
    # Quick stats preview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌍 Regions Monitored", "8", "Global Coverage")
    with col2:
        st.metric("📊 Data Sources", "3", "Carbon + Latency + ML")
    with col3:
        st.metric(" Forecasting", "24h", "Ahead")
    with col4:
        st.metric("🔄 Update Frequency", "2 min", "Real-time")


elif page == "🌍 Carbon Intensity":
    st.title("🌍 Live Carbon Intensity Dashboard")
    
    st.markdown("""
    **Carbon Intensity** measures how much CO₂ is emitted per kWh of electricity consumed. 
    Lower values mean cleaner energy (more renewables), higher values indicate fossil fuel dependency.
    
    📊 **Color Coding:** 🟢 Low (<100) | 🟡 Medium (100-200) | 🔴 High (>200) gCO₂eq/kWh
    """)
    
    # Auto-refresh settings
    st.sidebar.header("⚙️ Settings")
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 60, 300, 120)
    auto_refresh = st.sidebar.checkbox("Auto-refresh", True)
    
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="carbon_refresh")
    
    # Fetch carbon data
    with st.spinner("🔄 Fetching live carbon data..."):
        carbon_data = []
        for zone in ZONES_MAPPING:
            data = get_carbon_intensity(zone)
            if "error" not in data:
                data["region_name"] = REGION_MAPPING.get(zone, {}).get("name", zone)
                carbon_data.append(data)
    
    if carbon_data:
        # Create DataFrame
        ist = pytz.timezone("Asia/Kolkata")
        df = pd.DataFrame(carbon_data)
        df["updatedAt"] = pd.to_datetime(df["updatedAt"]).dt.tz_convert(ist)
        
        # Display metrics
        st.subheader("🌍 Current Carbon Intensity by Region")
        
        cols = st.columns(4)
        for i, (_, row) in enumerate(df.iterrows()):
            col_idx = i % 4
            with cols[col_idx]:
                intensity = row["carbonIntensity"]
                if intensity != "N/A":
                    # Color coding
                    if intensity < 100:
                        delta_color = "normal"
                        delta_text = "🟢 Low"
                    elif intensity < 200:
                        delta_color = "normal"
                        delta_text = "🟡 Medium"
                    else:
                        delta_color = "inverse"
                        delta_text = "🔴 High"
                    
                    st.metric(
                        label=row["region_name"],
                        value=f"{intensity} gCO₂eq/kWh",
                        delta=delta_text,
                        delta_color=delta_color
                    )

        
        # Chart
        st.subheader("📊 Carbon Intensity Comparison")
        chart_df = df[df["carbonIntensity"] != "N/A"].copy()
        if not chart_df.empty:
            chart_df["carbonIntensity"] = pd.to_numeric(chart_df["carbonIntensity"])
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('region_name:O', title='Region'),
                y=alt.Y('carbonIntensity:Q', title='Carbon Intensity (gCO₂eq/kWh)'),
                color=alt.Color('carbonIntensity:Q', scale=alt.Scale(scheme='redyellowgreen', reverse=True)),
                tooltip=['region_name', 'carbonIntensity', 'zone']
            ).properties(width=1150, height=450)
            st.altair_chart(chart, use_container_width=False)
        
        # Detailed table
        st.subheader("📋 Detailed Data")
        st.dataframe(df[["zone", "region_name", "carbonIntensity", "updatedAt"]], use_container_width=True)
        
        # Recommendations
        st.subheader("🎯 Deployment Recommendations")
        valid_data = [d for d in carbon_data if d["carbonIntensity"] != "N/A"]
        if valid_data:
            best_zone = min(valid_data, key=lambda x: x["carbonIntensity"])
            st.success(f"🏆 **Recommended Region:** {best_zone.get('region_name', best_zone['zone'])} ({best_zone['carbonIntensity']} gCO₂eq/kWh)")
            
            worst_zone = max(valid_data, key=lambda x: x["carbonIntensity"])
            st.warning(f"⚠️ **Avoid:** {worst_zone.get('region_name', worst_zone['zone'])} ({worst_zone['carbonIntensity']} gCO₂eq/kWh)")
    
    else:
        st.error("❌ Could not fetch carbon data. Please check your API token.")


elif page == "⚖️ Carbon + Latency":
    st.title("⚖️ Carbon Intensity + Network Latency Dashboard")
    
    st.markdown("""
    **Balance sustainability with performance** by viewing both carbon intensity and network latency together.
    
    - 🌍 **Carbon Intensity**: Environmental impact of deploying in each region
    - ⚡ **Network Latency**: Response time from your location to each cloud region
    - ⚖️ **Smart Decisions**: Find the optimal balance for your use case
    """)
    
    # Settings
    st.sidebar.header("⚙️ Multi-Objective Settings")
    carbon_weight = st.sidebar.slider("Carbon Weight", 0.0, 1.0, 0.6, 0.1)
    latency_weight = st.sidebar.slider("Latency Weight", 0.0, 1.0, 0.4, 0.1)

    # Show normalized weights (they don't have to add to 1.0)
    total_weight = carbon_weight + latency_weight
    if total_weight > 0:
        norm_carbon = carbon_weight / total_weight
        norm_latency = latency_weight / total_weight
        st.sidebar.write(f"**Normalized:**")
        st.sidebar.write(f"Carbon: {norm_carbon:.1%}")
        st.sidebar.write(f"Latency: {norm_latency:.1%}")
    else:
        norm_carbon = 0.5
        norm_latency = 0.5
    
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 120, 600, 300)
    auto_refresh = st.sidebar.checkbox("Auto-refresh", True)
    
    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="combined_refresh")
    
    # Fetch both carbon and latency data
    with st.spinner("🔄 Fetching carbon and latency data..."):
        # Carbon data
        carbon_data = []
        for zone in ZONES_MAPPING:
            data = get_carbon_intensity(zone)
            if "error" not in data:
                carbon_data.append(data)
        
        # Latency data
        endpoints = {zone: REGION_MAPPING[zone]["endpoint"] for zone in ZONES_MAPPING if zone in REGION_MAPPING}
        latency_data = run_latency_measurements(endpoints)
    
    if carbon_data and latency_data:
        # Combine data
        combined_data = []
        for carbon in carbon_data:
            zone = carbon["zone"]
            if zone in latency_data and carbon["carbonIntensity"] != "N/A":
                combined_data.append({
                    "zone": zone,
                    "region_name": REGION_MAPPING.get(zone, {}).get("name", zone),
                    "carbonIntensity": carbon["carbonIntensity"],
                    "avg_latency": latency_data[zone].get("avg_latency", "N/A"),
                    "p95_latency": latency_data[zone].get("p95_latency", "N/A"),
                    "updatedAt": carbon["updatedAt"]
                })
        
        if combined_data:
            df = pd.DataFrame(combined_data)
            
            # Calculate weighted scores
            if len(df) > 0:
                # Normalize values (min-max scaling)
                df["carbon_norm"] = (df["carbonIntensity"] - df["carbonIntensity"].min()) / (df["carbonIntensity"].max() - df["carbonIntensity"].min())
                valid_latencies = df[df["avg_latency"] != "N/A"]["avg_latency"].astype(float)
                if len(valid_latencies) > 0:
                    df["latency_norm"] = df["avg_latency"].apply(lambda x: 
                        (float(x) - valid_latencies.min()) / (valid_latencies.max() - valid_latencies.min()) 
                        if x != "N/A" else 0.5)
                    
                    # Weighted score (lower is better)
                    df["score"] = norm_carbon * df["carbon_norm"] + norm_latency * df["latency_norm"]
                    df_sorted = df.sort_values("score")
            
            # Display combined metrics
            st.subheader("⚖️ Combined Carbon + Latency Metrics")
            
            cols = st.columns(4)
            for i, (_, row) in enumerate(df.head(8).iterrows()):
                col_idx = i % 4
                with cols[col_idx]:
                    # FIXED: Prepare latency string safely
                    if row['avg_latency'] != "N/A":
                        latency_display = f"{row['avg_latency']:.0f} ms"
                    else:
                        latency_display = "N/A"
                    
                    st.markdown(f"""
                    **{row['region_name']}**  
                    🌍 {row['carbonIntensity']} gCO₂eq/kWh  
                    ⚡ {latency_display}
                    """)
            
            # Table
            st.subheader("📊 Carbon & Latency Comparison Table")
            
            # Create a clean comparison table
            table_df = df[["region_name", "carbonIntensity", "avg_latency", "p95_latency"]].copy()
            # Format numeric values
            for col in ["avg_latency", "p95_latency"]:
                table_df[col] = table_df[col].apply(lambda x: f"{x:.0f}" if isinstance(x, (int, float)) and x != "N/A" else "N/A")
            table_df.columns = ["Region", "Carbon (gCO₂eq/kWh)", "Avg Latency (ms)", "P95 Latency (ms)"]
            
            st.dataframe(table_df, use_container_width=False, height=318)
            
            # Recommendations - FIXED
            st.subheader("🎯 Multi-Objective Recommendations")
            
            if "score" in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    best = df_sorted.iloc[0]
                    # FIXED: Prepare latency string safely
                    if best['avg_latency'] != "N/A":
                        latency_str = f"{best['avg_latency']:.0f} ms"
                    else:
                        latency_str = "N/A"
                    
                    st.success(f"""
                    🏆 **Best Overall Balance**  
                    **{best['region_name']}**  
                    - Carbon: {best['carbonIntensity']} gCO₂eq/kWh  
                    - Latency: {latency_str}  
                    - Score: {best['score']:.3f}
                    """)
                
                with col2:
                    # Show weights used
                    st.info(f"""
                    ⚖️ **Weights Used**  
                    - Carbon Priority: {carbon_weight*100:.0f}%  
                    - Latency Priority: {latency_weight*100:.0f}%
                    
                    *Adjust weights in sidebar*
                    """)
    
    else:
        st.error("❌ Could not fetch combined data. Please check your setup.")


# elif page == " ML Forecasting":
#     st.title(" ML-Powered Carbon Intensity Forecasting")
    
#     st.markdown("""
#     **AI-Powered 24-Hour Ahead Forecasting**
    
#     Using CNN-LSTM deep learning models trained on 4,320+ historical records per region,
#     we forecast carbon intensity for the next 24 hours to help you schedule deployments during "green hours".
    
#     📊 **All times shown in Indian Standard Time (IST)**
#     """)
    
#     # Zone selector
#     selected_zone = st.selectbox("Select Cloud Region:", zones, key="zone_selector")
    
#     if selected_zone:
#         st.markdown("---")
        
#         # Generate forecast for selected zone
#         with st.spinner(f"Generating 24-hour forecast for {selected_zone}..."):
#             forecast = generate_live_forecast(selected_zone)
        
#         if forecast:
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.metric("Current", f"{forecast['current']:.0f} gCO₂/kWh", "Now")
#             with col2:
#                 min_val = forecast['carbon'].min()
#                 st.metric("Best (24h)", f"{min_val:.0f} gCO₂/kWh", "🟢 Green")
#             with col3:
#                 max_val = forecast['carbon'].max()
#                 st.metric("Worst (24h)", f"{max_val:.0f} gCO₂/kWh", "🔴 Dirty")
#             with col4:
#                 avg_val = forecast['carbon'].mean()
#                 st.metric("Average", f"{avg_val:.0f} gCO₂/kWh", "📊 Mean")
            
#             st.markdown("---")
            
#             # Forecast chart
#             st.subheader("Forecast Timeline (Next 24 Hours)")
            
#             times_display = [t.strftime('%H:%M IST') for t in forecast['times_ist']]
            
#             fig = alt.Chart(pd.DataFrame({
#                 'Time': times_display,
#                 'Carbon': forecast['carbon'],
#                 'index': range(len(forecast['carbon']))
#             })).mark_line(point=True, color='steelblue').encode(
#                 x=alt.X('index:O', axis=alt.Axis(labels=False)),
#                 y=alt.Y('Carbon:Q', title='Carbon Intensity (gCO₂/kWh)'),
#                 tooltip=['Time', 'Carbon']
#             ).properties(height=400, width=800)
            
#             st.altair_chart(fig, use_container_width=True)
            
#             st.markdown("---")
            
#             # Deployment recommendations
#             st.subheader("Deployment Recommendations")
            
#             best_idx = np.argmin(forecast['carbon'])
#             worst_idx = np.argmax(forecast['carbon'])
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.success(f"""
#                 ✅ **BEST TIME TO DEPLOY**
                
#                 Time: {forecast['times_ist'][best_idx].strftime('%H:%M IST')}
#                        ({forecast['times_utc'][best_idx].strftime('%H:%M UTC')})
                
#                 Carbon: {forecast['carbon'][best_idx]:.1f} gCO₂/kWh
                
#                 Improvement: {((forecast['current'] - forecast['carbon'][best_idx]) / forecast['current']) * 100:.1f}% greener
#                 """)
            
#             with col2:
#                 st.error(f"""
#                 ❌ **WORST TIME TO DEPLOY**
                
#                 Time: {forecast['times_ist'][worst_idx].strftime('%H:%M IST')}
#                        ({forecast['times_utc'][worst_idx].strftime('%H:%M UTC')})
                
#                 Carbon: {forecast['carbon'][worst_idx]:.1f} gCO₂/kWh
                
#                 Worse by: {((forecast['carbon'][worst_idx] - forecast['current']) / forecast['current']) * 100:.1f}%
#                 """)
            
#             st.markdown("---")
            
#             # Hourly breakdown table
#             st.subheader("Hourly Breakdown (IST / UTC)")
            
#             hourly_data = pd.DataFrame({
#                 'Time (IST)': [t.strftime('%H:%M') for t in forecast['times_ist']],
#                 'Time (UTC)': [t.strftime('%H:%M') for t in forecast['times_utc']],
#                 'Carbon (gCO₂/kWh)': [f"{c:.1f}" for c in forecast['carbon']],
#                 'vs Current': [f"{((c - forecast['current']) / forecast['current']) * 100:+.1f}%" for c in forecast['carbon']]
#             })
            
#             st.dataframe(hourly_data, use_container_width=True, hide_index=True)
        
#         else:
#             st.error(f"Could not generate forecast for {selected_zone}. Make sure models are trained.")
    
#     # Model performance info
#     st.markdown("---")
#     st.subheader("Model Performance")
    
#     performance_data = {
#         'Zone': ['US-MIDW-MISO', 'JP-TK', 'US-NW-PACW', 'BE', 'SG', 'IE', 'DE', 'US-MIDA-PJM'],
#         'MAE (gCO₂/kWh)': [15, 95, 17, 27, 22, 48, 31, 13],
#         'MAPE (%)': [18, 32, 42, 67, 85, 76, 194, 506],
#         'Status': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐', '⭐⭐', '⭐⭐', '⭐', '⭐']
#     }
    
#     perf_df = pd.DataFrame(performance_data)
#     st.dataframe(perf_df, use_container_width=True, hide_index=True)


elif page == " ML Forecasting":
    st.title(" ML-Powered Carbon Intensity Forecasting")
    
    st.markdown("""
    **AI-Powered 24-Hour Ahead Forecasting**
    
    Using CNN-LSTM deep learning models trained on 4,320+ historical records per region,
    we forecast carbon intensity for the next 24 hours to help you schedule deployments during "green hours".
    
    📊 **All times shown in Indian Standard Time (IST)**
    """)
    
    # Load all forecasts
    @st.cache_data(ttl=600)
    def load_all_forecasts():
        """Load forecasts for all zones"""
        all_forecasts = {}
        for zone in zones:
            forecast = generate_live_forecast(zone)
            if forecast:
                all_forecasts[zone] = forecast
        return all_forecasts
    
    all_forecasts_data = load_all_forecasts()
    
    # ============================================================
    # SECTION 1: BEST 3 ZONES - LIVE COMPARISON
    # ============================================================
    st.subheader("🌍 TOP 3 GREENEST ZONES - LIVE COMPARISON")
    
    # Get current carbon for all zones
    current_time = datetime.now(IST)
    
    all_zones_current = []
    for zone, forecast_data in all_forecasts_data.items():
        all_zones_current.append({
            'Zone': zone,
            'Region': REGION_MAPPING.get(zone, {}).get("name", zone),
            'Current_Carbon': forecast_data['current'],
            'Forecast': forecast_data
        })
    
    # Sort by current carbon
    all_zones_current.sort(key=lambda x: x['Current_Carbon'])
    
    # Create tabs for time intervals
    time_tabs = st.tabs(["📍 Now", "⏱️ +30 min", "⏰ +1 hour", "⏲️ +2 hours", "⏳ +6 hours"])
    
    time_intervals = [
        (0, "Now (Current)"),
        (1, "+30 minutes"),
        (2, "+1 hour"),
        (4, "+2 hours"),
        (12, "+6 hours")
    ]
    
    for tab_idx, (time_tab, (hours_offset, label)) in enumerate(zip(time_tabs, time_intervals)):
        with time_tab:
            st.markdown(f"### Top 3 Greenest Zones - {label}")
            
            # Calculate which hour index (every 30 min = 0.5 hour)
            if hours_offset == 0:
                hour_idx = 0  # Current from latest data
            else:
                hour_idx = int(hours_offset * 2)  # 0.5, 1, 2, 4, 12 converted to indices
            
            zones_at_time = []
            
            for zone_data in all_zones_current:
                zone = zone_data['Zone']
                region_name = zone_data['Region']
                forecast = zone_data['Forecast']
                
                # Get carbon at this time interval
                if hour_idx < len(forecast['carbon']):
                    carbon_val = forecast['carbon'][hour_idx]
                    time_ist = forecast['times_ist'][hour_idx]
                else:
                    carbon_val = forecast['carbon'][-1]
                    time_ist = forecast['times_ist'][-1]
                
                zones_at_time.append({
                    'Zone': zone,
                    'Region': region_name,
                    'Carbon': carbon_val,
                    'Time_IST': time_ist.strftime('%H:%M IST'),
                    'Time_UTC': time_ist.astimezone(pytz.UTC).strftime('%H:%M UTC'),
                })
            
            # Sort and get top 3
            zones_at_time.sort(key=lambda x: x['Carbon'])
            top_3 = zones_at_time[:3]
            
            # Display top 3 with medals
            cols = st.columns(3)
            medals = ['🥇', '🥈', '🥉']
            
            for col, medal, zone_data in zip(cols, medals, top_3):
                with col:
                    carbon_val = zone_data['Carbon']
                    
                    # Color based on carbon level
                    if carbon_val < 200:
                        color = "🟢"
                    elif carbon_val < 350:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2>{medal}</h2>
                    <h3>{zone_data['Zone']}</h3>
                    <p><b>{zone_data['Region']}</b></p>
                    <h1 style='color: #1f77b4;'>{carbon_val:.0f}</h1>
                    <p>gCO₂/kWh</p>
                    <p>{color} {zone_data['Time_IST']} ({zone_data['Time_UTC']})</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Full table for all zones at this time
            st.markdown("---")
            st.markdown("**All Zones at this time (sorted by carbon):**")
            
            table_data = []
            for zone_data in zones_at_time:
                table_data.append({
                    'Rank': len(table_data) + 1,
                    'Zone': zone_data['Zone'],
                    'Region': zone_data['Region'],
                    'Carbon (gCO₂/kWh)': f"{zone_data['Carbon']:.1f}",
                    'Time (IST)': zone_data['Time_IST'],
                })
            
            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 2: SINGLE ZONE DETAILED FORECAST
    # ============================================================
    st.subheader("🔬 Detailed Single Zone Forecast")
    
    # Zone selector
    selected_zone = st.selectbox("Select Cloud Region for Detailed Forecast:", zones, key="zone_selector")
    
    if selected_zone and selected_zone in all_forecasts_data:
        forecast = all_forecasts_data[selected_zone]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current", f"{forecast['current']:.0f} gCO₂/kWh", "Now")
        with col2:
            min_val = forecast['carbon'].min()
            st.metric("Best (24h)", f"{min_val:.0f} gCO₂/kWh", "🟢")
        with col3:
            max_val = forecast['carbon'].max()
            st.metric("Worst (24h)", f"{max_val:.0f} gCO₂/kWh", "🔴")
        with col4:
            avg_val = forecast['carbon'].mean()
            st.metric("Average", f"{avg_val:.0f} gCO₂/kWh", "📊")
        
        st.markdown("---")
        
        # Forecast chart
        st.subheader("24-Hour Forecast Timeline")
        
        times_display = [t.strftime('%H:%M IST') for t in forecast['times_ist']]
        chart_df = pd.DataFrame({
            'Time': times_display,
            'Carbon': forecast['carbon'],
            'index': range(len(forecast['carbon']))
        })
        
        fig = alt.Chart(chart_df).mark_line(point=True, color='steelblue', size=3).encode(
            x=alt.X('index:O', axis=alt.Axis(labels=False)),
            y=alt.Y('Carbon:Q', title='Carbon Intensity (gCO₂/kWh)'),
            tooltip=['Time', 'Carbon']
        ).properties(height=400)
        
        st.altair_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Best/Worst recommendations
        st.subheader("Deployment Recommendations")
        
        best_idx = np.argmin(forecast['carbon'])
        worst_idx = np.argmax(forecast['carbon'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            ✅ **BEST TIME TO DEPLOY**
            
            Time: {forecast['times_ist'][best_idx].strftime('%H:%M IST')}
                   ({forecast['times_utc'][best_idx].strftime('%H:%M UTC')})
            
            Carbon: {forecast['carbon'][best_idx]:.1f} gCO₂/kWh
            
            Savings: {((forecast['current'] - forecast['carbon'][best_idx]) / forecast['current']) * 100:.1f}% greener
            """)
        
        with col2:
            st.error(f"""
            ❌ **WORST TIME TO DEPLOY**
            
            Time: {forecast['times_ist'][worst_idx].strftime('%H:%M IST')}
                   ({forecast['times_utc'][worst_idx].strftime('%H:%M UTC')})
            
            Carbon: {forecast['carbon'][worst_idx]:.1f} gCO₂/kWh
            
            Worse by: {((forecast['carbon'][worst_idx] - forecast['current']) / forecast['current']) * 100:.1f}%
            """)
        
        st.markdown("---")
        
        # Hourly breakdown
        # st.subheader("Hourly Breakdown")
        
        # hourly_data = []
        # for idx, (time_ist, time_utc, carbon) in enumerate(zip(
        #     forecast['times_ist'], 
        #     forecast['times_utc'], 
        #     forecast['carbon']
        # )):
        #     hourly_data.append({
        #         'Hour': idx + 1,
        #         'Time (IST)': time_ist.strftime('%H:%M'),
        #         'Time (UTC)': time_utc.strftime('%H:%M'),
        #         'Carbon (gCO₂/kWh)': f"{carbon:.1f}",
        #         'vs Current': f"{((carbon - forecast['current']) / forecast['current']) * 100:+.1f}%"
        #     })
        
        # hourly_df = pd.DataFrame(hourly_data)
        # st.dataframe(hourly_df, use_container_width=True, hide_index=True)
        st.markdown("---")
    st.subheader("📅 24-Hour Hourly Breakdown - VISUAL GUIDE")

    # Create better visualization
    hourly_data = []
    for idx, (time_ist, time_utc, carbon) in enumerate(zip(
        forecast['times_ist'], 
        forecast['times_utc'], 
        forecast['carbon']
    )):
        # Calculate percentage change from current
        pct_change = ((carbon - forecast['current']) / forecast['current']) * 100
        
        # Determine color/status
        if carbon < 250:
            status = "🟢 Very Green"
            color = "#2ecc71"
        elif carbon < 300:
            status = "🟡 Good"
            color = "#f39c12"
        elif carbon < 350:
            status = "🟠 Moderate"
            color = "#e67e22"
        else:
            status = "🔴 High Carbon"
            color = "#e74c3c"
        
        hourly_data.append({
            'Time': time_ist.strftime('%H:%M IST'),
            'UTC': time_utc.strftime('%H:%M'),
            'Carbon': f"{carbon:.1f}",
            'Status': status,
            'vs Current': f"{pct_change:+.1f}%",
            'carbon_value': carbon,
            'color': color,
            'pct_change': pct_change
        })

    # Create two views: Simple Table & Visual Timeline

    tab_table, tab_visual = st.tabs(["📋 Table View", "📊 Timeline View"])

    # ============================================================
    # TABLE VIEW - CLEAN & SIMPLE
    # ============================================================
    with tab_table:
        st.markdown("**Click hour to see details:**\n")
        
        # Group by every 3 hours for better readability
        for group_idx in range(0, len(hourly_data), 3):
            cols = st.columns(3)
            
            for col_idx, col in enumerate(cols):
                if group_idx + col_idx < len(hourly_data):
                    data = hourly_data[group_idx + col_idx]
                    
                    with col:
                        st.markdown(f"""
                        <div style='background-color: {data["color"]}; padding: 15px; border-radius: 8px; color: white; text-align: center;'>
                        <h3>{data["Time"]}</h3>
                        <p style='font-size: 24px; font-weight: bold;'>{data["Carbon"]} gCO₂/kWh</p>
                        <p>{data["Status"]}</p>
                        <p style='font-size: 12px;'>{data["vs Current"]}</p>
                        <p style='font-size: 10px;'>{data["UTC"]} UTC</p>
                        </div>
                        """, unsafe_allow_html=True)

    # ============================================================
    # TIMELINE VIEW - BEAUTIFUL VISUAL
    # ============================================================
    with tab_visual:
        st.markdown("**Best times highlighted in GREEN, Worst times in RED:**\n")
        
        # Find best and worst
        best_idx = np.argmin([d['carbon_value'] for d in hourly_data])
        worst_idx = np.argmax([d['carbon_value'] for d in hourly_data])
        
        # Create timeline visualization
        for idx, data in enumerate(hourly_data):
            carbon_val = data['carbon_value']
            
            # Create progress bar visualization
            # Normalize carbon to 0-100% (min to max in dataset)
            min_carbon = min([d['carbon_value'] for d in hourly_data])
            max_carbon = max([d['carbon_value'] for d in hourly_data])
            
            progress_percent = ((carbon_val - min_carbon) / (max_carbon - min_carbon)) * 100
            
            # Determine bar color
            if idx == best_idx:
                bar_color = "#2ecc71"  # Green for best
                emoji = "🌟"
            elif idx == worst_idx:
                bar_color = "#e74c3c"  # Red for worst
                emoji = "⚠️"
            else:
                bar_color = "#3498db"  # Blue for normal
                emoji = "  "
            
            # Create visual bar
            col1, col2, col3, col4 = st.columns([0.5, 1.5, 2, 0.5])
            
            with col1:
                st.markdown(f"<h4 style='text-align: center;'>{emoji}</h4>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<p style='text-align: right;'><b>{data['Time']}</b></p>", unsafe_allow_html=True)
            
            with col3:
                # Progress bar
                st.markdown(f"""
                <div style='background-color: #ecf0f1; border-radius: 10px; height: 30px; overflow: hidden;'>
                    <div style='background-color: {bar_color}; width: {progress_percent}%; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
                        {data['Carbon']} gCO₂/kWh
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                # Show change
                if float(data['vs Current'].rstrip('%')) > 0:
                    st.markdown(f"<p style='color: red;'>{data['vs Current']}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color: green;'>{data['vs Current']}</p>", unsafe_allow_html=True)

    # ============================================================
    # RECOMMENDATIONS BASED ON HOURLY DATA
    # ============================================================
    st.markdown("---")
    st.subheader("💡 Smart Recommendations")

    best_data = hourly_data[best_idx]
    worst_data = hourly_data[worst_idx]

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"""
        ### ✅ BEST TIME TO DEPLOY
        
        **{best_data['Time']}** ({best_data['UTC']} UTC)
        
        Carbon: **{best_data['Carbon']} gCO₂/kWh**
        
        This is **{abs(float(best_data['vs Current'].rstrip('%')))}% CLEANER** than now!
        
        💚 RECOMMENDED FOR DEPLOYMENT
        """)

    with col2:
        st.error(f"""
        ### ❌ WORST TIME TO DEPLOY
        
        **{worst_data['Time']}** ({worst_data['UTC']} UTC)
        
        Carbon: **{worst_data['Carbon']} gCO₂/kWh**
        
        This is **{abs(float(worst_data['vs Current'].rstrip('%')))}% DIRTIER** than now!
        
        🚫 AVOID DEPLOYMENT AT THIS TIME
        """)

    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================
    st.markdown("---")
    st.subheader("📊 Summary Statistics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        current_val = float(forecast['current'])
        st.metric("Current Now", f"{current_val:.0f}", "gCO₂/kWh")

    with col2:
        best_val = float(best_data['Carbon'])
        savings = ((current_val - best_val) / current_val) * 100
        st.metric("Best in 24h", f"{best_val:.0f}", f"↓ {savings:.1f}%")

    with col3:
        worst_val = float(worst_data['Carbon'])
        increase = ((worst_val - current_val) / current_val) * 100
        st.metric("Worst in 24h", f"{worst_val:.0f}", f"↑ {increase:.1f}%")

    with col4:
        avg_val = np.mean([float(d['Carbon']) for d in hourly_data])
        st.metric("24h Average", f"{avg_val:.0f}", "gCO₂/kWh")

    with col5:
        range_val = worst_val - best_val
        st.metric("Range", f"{range_val:.0f}", "gCO₂/kWh")

    # ============================================================
    # WHAT DOES THIS ALL MEAN?
    # ============================================================
    st.markdown("---")
    st.subheader("🎓 Understanding the Chart")

    with st.expander("📖 Click here to understand the hourly breakdown"):
        st.markdown("""
        ### Column Explanations:
        
        **Hour**: Number from 1-24 (next 24 hours from now)
        
        **Time (IST)**: Indian Standard Time - Your local time in Agra
        
        **Time (UTC)**: Coordinated Universal Time - Used for international reference
        
        **Carbon (gCO₂/kWh)**: 
        - Lower number = CLEANER energy ✅ 🟢
        - Higher number = DIRTIER energy ❌ 🔴
        
        **Status**:
        - 🟢 Very Green (< 250): Excellent time to deploy
        - 🟡 Good (250-300): Acceptable time
        - 🟠 Moderate (300-350): Not ideal
        - 🔴 High Carbon (> 350): Avoid if possible
        
        **vs Current**: Shows how this hour compares to RIGHT NOW
        - Negative (-) = BETTER/CLEANER ✅ Deploy here!
        - Positive (+) = WORSE/DIRTIER ❌ Avoid this time
        
        ### Example:
        If it says "-10%" = 10% CLEANER than now (GOOD!)
        If it says "+15%" = 15% DIRTIER than now (BAD!)
        """)
        
        # Model performance
        st.markdown("---")
        st.subheader("Model Performance Metrics")
        
        performance_data = {
            'Zone': ['US-MIDW-MISO', 'JP-TK', 'US-NW-PACW', 'BE', 'SG', 'IE', 'DE', 'US-MIDA-PJM'],
            'MAE (gCO₂/kWh)': [15, 95, 17, 27, 22, 48, 31, 13],
            'MAPE (%)': [18, 32, 42, 67, 85, 76, 194, 506],
            'Status': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐', '⭐⭐', '⭐⭐', '⭐', '⭐']
        }
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        show_regional_comparison(all_forecasts_data, zones, REGION_MAPPING)
        show_carbon_savings_calculator(all_forecasts_data, zones, REGION_MAPPING)
        
# ============================================================
# FEATURE 1: REGIONAL COMPARISON TABLE
# ============================================================

st.markdown("---")
st.subheader("🌍 Regional Comparison Dashboard")

st.markdown("**Compare all 8 regions at a glance:**")

# Create comparison data
regional_data = {
    'Zone': ['DE', 'US-MIDA-PJM', 'US-NW-PACW', 'IE', 'SG', 'BE', 'US-MIDW-MISO', 'JP-TK'],
    'Region': [
        'Germany',
        'USA - Virginia',
        'USA - Oregon', 
        'Ireland',
        'Singapore',
        'Belgium',
        'USA - Central',
        'Japan - Tokyo'
    ],
    'Current Carbon': [],
    'Min (24h)': [],
    'Max (24h)': [],
    'Avg (24h)': [],
    'Variation': [],
    'Best Time (IST)': [],
    'Status': []
}

# Fill in data from all zones
for zone in zones:
    if zone in all_forecasts_data:
        forecast = all_forecasts_data[zone]
        current = forecast['current']
        carbon = forecast['carbon']
        times_ist = forecast['times_ist']
        
        min_carbon = carbon.min()
        max_carbon = carbon.max()
        avg_carbon = carbon.mean()
        variation = ((max_carbon - min_carbon) / min_carbon) * 100
        best_idx = np.argmin(carbon)
        best_time = times_ist[best_idx].strftime('%H:%M')
        
        # Status indicator
        if avg_carbon < 250:
            status = "🟢 Very Green"
        elif avg_carbon < 300:
            status = "🟡 Good"
        elif avg_carbon < 350:
            status = "🟠 Moderate"
        else:
            status = "🔴 High Carbon"
        
        regional_data['Current Carbon'].append(f"{current:.0f}")
        regional_data['Min (24h)'].append(f"{min_carbon:.0f}")
        regional_data['Max (24h)'].append(f"{max_carbon:.0f}")
        regional_data['Avg (24h)'].append(f"{avg_carbon:.0f}")
        regional_data['Variation'].append(f"{variation:.1f}%")
        regional_data['Best Time (IST)'].append(best_time)
        regional_data['Status'].append(status)

# Create and display DataFrame
regional_df = pd.DataFrame(regional_data)

# Color the dataframe
st.dataframe(
    regional_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        'Current Carbon': st.column_config.TextColumn(width="small"),
        'Min (24h)': st.column_config.TextColumn(width="small"),
        'Max (24h)': st.column_config.TextColumn(width="small"),
        'Status': st.column_config.TextColumn(width="medium"),
    }
)

# Insights
col1, col2, col3 = st.columns(3)

with col1:
    # Find greenest zone
    current_values = [float(regional_df.iloc[i]['Current Carbon']) for i in range(len(regional_df))]
    greenest_idx = np.argmin(current_values)
    greenest_zone = regional_df.iloc[greenest_idx]
    
    st.success(f"""
    🟢 **GREENEST REGION NOW**
    
    **{greenest_zone['Region']}** ({greenest_zone['Zone']})
    
    Current: {greenest_zone['Current Carbon']} gCO₂/kWh
    Status: {greenest_zone['Status']}
    """)

with col2:
    # Find best variation (most stable)
    variation_values = [float(regional_df.iloc[i]['Variation'].rstrip('%')) for i in range(len(regional_df))]
    most_stable_idx = np.argmin(variation_values)
    most_stable = regional_df.iloc[most_stable_idx]
    
    st.info(f"""
    📊 **MOST STABLE GRID**
    
    **{most_stable['Region']}** ({most_stable['Zone']})
    
    Variation: {most_stable['Variation']}
    Avg Carbon: {most_stable['Avg (24h)']} gCO₂/kWh
    """)

with col3:
    # Find highest variation (best opportunity)
    best_opportunity_idx = np.argmax(variation_values)
    best_opportunity = regional_df.iloc[best_opportunity_idx]
    
    st.warning(f"""
    ⚡ **BEST SAVINGS OPPORTUNITY**
    
    **{best_opportunity['Region']}** ({best_opportunity['Zone']})
    
    Variation: {best_opportunity['Variation']}
    Deploy at {best_opportunity['Best Time (IST)']} IST!
    """)

st.markdown("---")

# ============================================================
# FEATURE 2: CARBON SAVINGS CALCULATOR
# ============================================================

st.markdown("---")
st.subheader("💰 Carbon Savings Calculator - ROI Analysis")

st.markdown("**Calculate how much CO₂ you can save by deploying at optimal times:**")

# Create calculator interface
calc_col1, calc_col2, calc_col3 = st.columns(3)

with calc_col1:
    workload_power = st.number_input(
        "Workload Power (Watts)",
        min_value=100,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Total power consumption of your workload (e.g., 1 server = 300-500W, 1 data center = 1000000+W)"
    )

with calc_col2:
    workload_hours = st.number_input(
        "Duration (Hours)",
        min_value=0.5,
        max_value=24.0,
        value=8.0,
        step=0.5,
        help="How long your workload will run"
    )

with calc_col3:
    selected_region_calc = st.selectbox(
        "Select Region",
        zones,
        # key="calc_zone",
        help="Which cloud region will you deploy to?"
    )

st.markdown("---")

# Perform calculation
if selected_region_calc in all_forecasts_data:
    forecast_calc = all_forecasts_data[selected_region_calc]
    region_name = REGION_MAPPING.get(selected_region_calc, {}).get("name", selected_region_calc)
    
    # Get current and best carbon values
    current_carbon = forecast_calc['current']
    best_carbon = forecast_calc['carbon'].min()
    worst_carbon = forecast_calc['carbon'].max()
    avg_carbon = forecast_calc['carbon'].mean()
    
    # Find best time
    best_idx = np.argmin(forecast_calc['carbon'])
    best_time_ist = forecast_calc['times_ist'][best_idx].strftime('%H:%M IST')
    best_time_utc = forecast_calc['times_utc'][best_idx].strftime('%H:%M UTC')
    
    # Calculate emissions
    workload_kw = workload_power / 1000
    energy_kwh = workload_kw * workload_hours
    
    current_emissions = (energy_kwh * current_carbon) / 1000  # kg CO2
    best_emissions = (energy_kwh * best_carbon) / 1000
    worst_emissions = (energy_kwh * worst_carbon) / 1000
    avg_emissions = (energy_kwh * avg_carbon) / 1000
    
    # Calculate savings
    savings_vs_current = current_emissions - best_emissions
    savings_percentage = (savings_vs_current / current_emissions) * 100
    worst_vs_best = worst_emissions - best_emissions
    worst_percentage = (worst_vs_best / best_emissions) * 100
    
    # Display results
    st.markdown("### 📊 Deployment Scenarios")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Deploy NOW",
            f"{current_emissions:.2f} kg CO₂",
            # f"{energy_kwh:.1f} kWh"
        )
        st.caption(f"{energy_kwh:.1f} kWh")
    with col2:
        st.metric(
            "Deploy at BEST time",
            f"{best_emissions:.2f} kg CO₂",
            # f"{best_time_ist}",
            delta=f"-{savings_vs_current:.2f} kg"
        )
        st.caption(f"⏱️ {best_time_ist}")
    
    with col3:
        st.metric(
            "Deploy at WORST time",
            f"{worst_emissions:.2f} kg CO₂",
            # f"{worst_percentage:.1f}% worse",
            delta_color="inverse"
        )
        st.caption(f"📍 {worst_percentage:.1f}% worse")
    with col4:
        st.metric(
            "Average (24h)",
            f"{avg_emissions:.2f} kg CO₂",
            f"Avg: {avg_carbon:.0f}"
        )
    
    st.markdown("---")
    
    # Visual comparison
    st.markdown("### 🎯 Savings Potential")
    
    scenarios = pd.DataFrame({
        'Scenario': ['Deploy NOW', 'Deploy at BEST', 'Deploy at WORST'],
        'CO₂ Emissions': [current_emissions, best_emissions, worst_emissions],
        'Color': ['🟡', '🟢', '🔴']
    })
    
    # Create horizontal bar chart
    fig = alt.Chart(scenarios).mark_bar().encode(
        y=alt.Y('Scenario:N', sort=['Deploy NOW', 'Deploy at BEST', 'Deploy at WORST']),
        x=alt.X('CO₂ Emissions:Q', title='CO₂ Emissions (kg)'),
        color=alt.Color('Scenario:N', scale=alt.Scale(
            domain=['Deploy NOW', 'Deploy at BEST', 'Deploy at WORST'],
            range=['#f39c12', '#2ecc71', '#e74c3c']
        ))
    ).properties(height=300, width=600)
    
    st.altair_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Key insights
    st.markdown("### 💡 Key Insights")
    
    col_insight1, col_insight2, col_insight3 = st.columns(3)
    
    with col_insight1:
        st.success(f"""
        ✅ **RECOMMENDED ACTION**
        
        Deploy at **{best_time_ist}**
        
        Save **{savings_percentage:.1f}%** carbon
        
        Equivalent to: **{savings_vs_current:.2f} kg CO₂** saved
        """)
    
    with col_insight2:
        # Convert CO2 to real-world equivalents
        trees_equivalent = savings_vs_current / 21  # 1 tree absorbs ~21kg CO2 per year
        car_miles = savings_vs_current / 0.41  # 1 mile driving = 0.41 kg CO2
        
        st.info(f"""
        🌍 **ENVIRONMENTAL IMPACT**
        
        Saving {savings_vs_current:.2f} kg CO₂ = 
        
        🌳 {trees_equivalent:.1f} trees planted
        
        🚗 {car_miles:.0f} miles NOT driven
        """)
    
    with col_insight3:
        # Annual impact if deployed daily
        annual_savings = savings_vs_current * 365
        
        st.warning(f"""
        📈 **IF DEPLOYED EVERY DAY**
        
        Annual savings: {annual_savings:.0f} kg CO₂
        
        That's equivalent to:
        
        🌳 {annual_savings/21:.0f} trees/year
        
        🚗 {annual_savings/0.41:.0f} miles not driven
        """)
    
    st.markdown("---")
    
    # Breakdown by time
    st.markdown("### 📅 Hour-by-Hour Savings Potential")
    
    hourly_breakdown = []
    for idx, (time_ist, carbon_val) in enumerate(zip(forecast_calc['times_ist'], forecast_calc['carbon'])):
        emissions = (energy_kwh * carbon_val) / 1000
        vs_best = emissions - best_emissions
        vs_current_pct = ((carbon_val - current_carbon) / current_carbon) * 100
        
        hourly_breakdown.append({
            'Hour': idx + 1,
            'Time (IST)': time_ist.strftime('%H:%M'),
            'CO₂ (kg)': f"{emissions:.2f}",
            'vs Best': f"+{vs_best:.2f} kg" if vs_best > 0 else f"-{abs(vs_best):.2f} kg",
            'vs Now': f"{vs_current_pct:+.1f}%"
        })
    
    hourly_df = pd.DataFrame(hourly_breakdown)
    
    col_hourly1, col_hourly2 = st.columns([1, 2])
    
    with col_hourly1:
        st.dataframe(hourly_df.head(12), use_container_width=True, hide_index=True)
    
    with col_hourly2:
        st.info("""
        **💰 How to Use This Calculator:**
        
        1. Enter your workload power (Watts)
        2. Enter duration (Hours)
        3. Select target region
        4. See CO₂ savings potential
        
        **Real Examples:**
        
        - **1 Web Server (400W, 8 hours)**
          → ~3 kg CO₂/day
          → Save 0.5-1 kg with smart timing
        
        - **Small Data Center (100kW, 24 hours)**
          → ~2,400 kg CO₂/day
          → Save 100-300 kg with smart timing
        
        - **Large Data Center (1MW, 24 hours)**
          → ~24,000 kg CO₂/day
          → Save 1,000-3,000 kg with smart timing
        """)


elif page == "📖 How It Works":
    st.title("📖 How EcoDeploy Works")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌍 Carbon Data", "⚡ Latency", " ML Forecasting", "🎯 Optimization", "🔧 Technical"])
    
    with tab1:
        st.header("🌍 Carbon Intensity Explained")
        st.markdown("""
        ### What is Carbon Intensity?
        
        **Carbon Intensity** measures how much CO₂ is emitted per kilowatt-hour (kWh) of electricity consumed.
        It's expressed in **grams of CO₂ equivalent per kWh (gCO₂eq/kWh)**.
        
        ### Why It Matters for Cloud Computing
        
        - **Data centers consume massive amounts of electricity**
        - **The carbon footprint depends on the local electricity grid**
        - **Renewable energy makes some regions much cleaner than others**
        
        ### Our Data Source: ElectricityMap
        
        - ⚡ **Real-time data** from 8 major electricity grid zones
        - 🌍 **Global coverage** including US, Europe, and Asia
        - 🔄 **Updated every few minutes** with live grid conditions
        - 📊 **Based on actual energy mix** (solar, wind, coal, gas, nuclear, etc.)
        
        ### Color Coding System
        
        - 🟢 **Low Impact (< 100 gCO₂eq/kWh)**: Mostly renewable energy
        - 🟡 **Medium Impact (100-200 gCO₂eq/kWh)**: Mixed energy sources  
        - 🔴 **High Impact (> 200 gCO₂eq/kWh)**: Heavy fossil fuel dependency
        
        ### Example Impact
        
        Deploying a service in **Germany (DE)** vs **US-MIDA-PJM**:
        - If DE has 150 gCO₂eq/kWh and US-MIDA-PJM has 400 gCO₂eq/kWh
        - **You could reduce emissions by 62.5%** by choosing Germany!
        """)
    
    with tab2:
        st.header("⚡ Network Latency Measurement")
        st.markdown("""
        ### What We Measure
        
        **Network latency** is the round-trip time for data to travel between your location and a cloud region.
        
        ### Our Methodology
        
        1. **Target AWS Regional Endpoints**: We ping official AWS EC2 service endpoints for each region
        2. **HTTP Requests**: Use real HTTP/HTTPS requests (not just ICMP ping) to simulate actual application traffic
        3. **Multiple Samples**: Take 3-5 measurements per region for accuracy
        4. **Statistics**: Calculate both average and 95th percentile (P95) latency
        
        ### What Affects Latency
        
        - **🌐 Geographic Distance**: Physical distance between you and the data center
        - **🛣️ Internet Routing**: The path your data takes through ISPs and backbone networks  
        - **⚡ Network Congestion**: Traffic load on intermediate networks
        - **🏢 Data Center Load**: Processing time at the cloud provider's end
        - **📡 Your Connection**: Local network conditions (WiFi, ISP, etc.)
        """)
    
    with tab3:
        st.header(" ML Forecasting (NEW!)")
        st.markdown("""
        ### How It Works
        
        **EcoDeploy uses CNN-LSTM deep learning** to forecast carbon intensity 24 hours ahead.
        
        #### Architecture:
        - **Convolutional Neural Networks (CNN)**: Capture local temporal patterns (hours, day of week)
        - **LSTM (Long Short-Term Memory)**: Learn long-term dependencies (seasonal, weekly)
        - **Training Data**: 4,320+ hourly records per region (6 months)
        - **Features**: 6 engineered temporal features
        
        #### Features Used:
        1. **Current Carbon**: Normalized to 0-1 scale
        2. **Hour Sin/Cos**: Cyclical encoding for 24h daily patterns
        3. **Is Weekend**: Binary flag for demand changes
        4. **Is Night**: Binary flag for solar availability (0-1=night, 1=day)
        5. **Day of Week**: Numeric day (0-6)
        
        #### Prediction Process:
        1. Load last 24 hours of data
        2. Pass through trained model
        3. Generate next 24 hours forecast
        4. Inverse-scale to original units (gCO₂/kWh)
        5. Display with IST timestamps
        
        #### Model Performance:
        - **Best**: US-MIDW-MISO (18% MAPE, 15 gCO₂/kWh MAE)
        - **Average**: 127% MAPE across all zones
        - **Use Case**: Recommend best deployment hours, not absolute values
        """)
    
    with tab4:
        st.header("🎯 Multi-Objective Optimization")
        st.markdown("""
        ### The Challenge
        
        **Carbon intensity and latency often conflict:**
        - The cleanest energy region might be geographically distant (high latency)
        - The lowest latency region might use dirty energy (high carbon)
        
        ### Our Approach: Weighted Scoring
        
        1. **Normalize Both Metrics**: Scale carbon and latency to 0-1 range
        2. **Apply User Weights**: You choose the importance of each factor
        3. **Calculate Combined Score**: 
           ```
           Score = (carbon_weight × normalized_carbon) + (latency_weight × normalized_latency)
           ```
        4. **Rank Regions**: Lower scores are better (minimize both carbon and latency)
        """)
    
    with tab5:
        st.header("🔧 Technical Implementation")
        st.markdown("""
        ### Technologies
        
        - **🐍 Python**: Core application
        - **📊 Streamlit**: Web dashboard  
        - **🤖 TensorFlow/Keras**: Deep learning models
        - **📈 Altair**: Interactive charts
        - **⏰ Caching**: Improved performance
        - **🌐 aiohttp**: Async latency testing
        
        ### Data Flow
        
        1. **Carbon Collection**: ElectricityMap API → Cache (5 min)
        2. **Latency Testing**: Async HTTP requests → Average statistics
        3. **ML Forecasting**: Load models → Generate predictions → Display
        4. **Visualization**: Streamlit components
        """)


elif page == "🔧 API Setup":
    st.title("🔧 API Setup Guide")
    
    st.markdown("""
    ### Getting Your ElectricityMap API Token
    
    1. **Visit [electricitymap.org](https://electricitymap.org)**
    2. **Sign up for a free account**
    3. **Go to your profile/API section**
    4. **Copy your API token**
    
    ### Setting Up the Token
    
    #### For Local Development:
    ```
    # Option 1: Environment variable
    export ELECTRICITY_MAP_TOKEN="your_token_here"
    
    # Option 2: .streamlit/secrets.toml file
    mkdir .streamlit
    echo 'ELECTRICITY_MAP_TOKEN = "your_token_here"' > .streamlit/secrets.toml
    ```
    
    #### For Streamlit Community Cloud:
    1. Deploy your app to Streamlit Cloud
    2. Go to app settings
    3. Add secret: `ELECTRICITY_MAP_TOKEN = "your_token_here"`
    
    ### API Limits
    
    - **Free Tier**: 1000 requests/month
    - **Rate Limit**: 1 request/second
    - **Caching**: We cache results for 5 minutes to minimize API usage
    
    ### Troubleshooting
    
    ❌ **"No API token" error**: Set your token as described above  
    ❌ **"API Error: 401"**: Invalid token, check your token is correct  
    ❌ **"API Error: 429"**: Rate limited, wait and try again  
    ❌ **"API Error: 403"**: Monthly quota exceeded, upgrade your plan
    """)
    
    # Token validator
    st.subheader("🔍 Token Validator")
    test_token = st.text_input("Enter your token to test:", type="password")
    if st.button("Test Token") and test_token:
        try:
            url = "https://api.electricitymap.org/v3/carbon-intensity/latest?zone=DE"
            headers = {"auth-token": test_token}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                st.success("✅ Token is valid! You can use this in your deployment.")
            else:
                st.error(f"❌ Token test failed: HTTP {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last updated:** {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.markdown("**Data source:** ElectricityMap API + ML Models")
st.sidebar.markdown("🌱 **EcoDeploy** - Making Cloud Computing Greener")