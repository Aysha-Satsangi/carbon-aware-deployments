#carbon_dashboard.py

import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import os
import pytz
from streamlit_autorefresh import st_autorefresh


st.title("EcoDeploy - Live Carbon Aware Dashboard")

# Function to fetch real-time carbon data
def get_carbon_intensity(zone):
    """Fetch real-time carbon intensity for a zone"""
    try:
        token = os.getenv("ELECTRICITY_MAP_TOKEN", st.secrets.get("ELECTRICITY_MAP_TOKEN", ""))
        if not token:
            st.error("ElectricityMap API token not found")
            return None
            
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
            st.error(f"Error fetching data for {zone}: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Exception fetching data for {zone}: {e}")
        return None

# Zones to monitor (from your region mapper)
ZONES = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]

# Sidebar for configuration
st.sidebar.header("Dashboard Settings")
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 60, 300, 120)
auto_refresh = st.sidebar.checkbox("Auto-refresh", True)

if auto_refresh:
    st_autorefresh(interval=refresh_interval * 1000, key="carbon_refresh")

# Fetch real-time data
st.header("Live Carbon Intensity Data")
st.write("Fetching real-time carbon intensity from ElectricityMap API...")

carbon_data = []
for zone in ZONES:
    data = get_carbon_intensity(zone)
    if data:
        carbon_data.append(data)

if carbon_data:
    # Create dataframe

    ist = pytz.timezone("Asia/Kolkata")
    df = pd.DataFrame(carbon_data)
    df["updatedAt"] = pd.to_datetime(df["updatedAt"]).dt.tz_convert(ist)
    
    # df["updatedAt"] = pd.to_datetime(df["updatedAt"])

    log_file = "carbon_history.csv"
    if not df.empty:
        df.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)
    
    # Display current data
    st.subheader("Current Carbon Intensity (gCO₂eq/kWh)")
    
    # Create metrics in columns
    cols = st.columns(3)
    for i, (_, row) in enumerate(df.iterrows()):
        col_idx = i % 3
        with cols[col_idx]:
            intensity = row["carbonIntensity"]
            if intensity != "N/A":
                # Color code based on carbon intensity
                if intensity < 100:
                    color = "green"
                elif intensity < 200:
                    color = "orange"
                else:
                    color = "red"
                
                st.metric(
                    label=row["zone"],
                    value=f"{intensity} gCO₂",
                    delta="Low" if intensity < 100 else "Medium" if intensity < 200 else "High",
                    delta_color="normal" if intensity == "N/A" else ("inverse" if intensity > 200 else "normal")
                )
    
    # Show detailed table
    st.subheader("Detailed Data")
    st.dataframe(df[["zone", "carbonIntensity", "updatedAt"]])
    
    # Show chart
    st.subheader("Carbon Intensity Comparison")
    chart_df = df[df["carbonIntensity"] != "N/A"].copy()
    if not chart_df.empty:
        chart_df["carbonIntensity"] = pd.to_numeric(chart_df["carbonIntensity"])
        st.bar_chart(chart_df.set_index("zone")["carbonIntensity"])
else:
    st.warning("Could not fetch carbon data. Using sample data for demonstration.")
    
    # Fallback to sample data
    sample_data = [
        {"zone": "DE", "carbonIntensity": 320, "updatedAt": datetime.now()},
        {"zone": "US-MIDA-PJM", "carbonIntensity": 450, "updatedAt": datetime.now()},
        {"zone": "US-NW-PACW", "carbonIntensity": 180, "updatedAt": datetime.now()},
        {"zone": "IE", "carbonIntensity": 280, "updatedAt": datetime.now()},
    ]
    
    df = pd.DataFrame(sample_data)
    st.dataframe(df)

# Deployment recommendations
st.header("Deployment Recommendations")
st.write("Based on current carbon intensity:")

if carbon_data:
    # Filter out zones with no data
    valid_data = [d for d in carbon_data if d["carbonIntensity"] != "N/A" and isinstance(d["carbonIntensity"], (int, float))]
    
    if valid_data:
        # Find the zone with lowest carbon intensity
        best_zone = min(valid_data, key=lambda x: x["carbonIntensity"])
        
        st.success(f"**Recommended deployment zone:** {best_zone['zone']} ({best_zone['carbonIntensity']} gCO₂eq/kWh)")
        
        # Show all zones sorted by carbon intensity
        sorted_zones = sorted(valid_data, key=lambda x: x["carbonIntensity"])
        
        st.subheader("All Zones (Sorted by Carbon Intensity)")
        for i, zone_data in enumerate(sorted_zones):
            intensity = zone_data["carbonIntensity"]
            if i == 0:
                st.write(f"🏆 **{zone_data['zone']}**: {intensity} gCO₂eq/kWh (Best)")
            else:
                st.write(f"{i+1}. {zone_data['zone']}: {intensity} gCO₂eq/kWh")
    else:
        st.warning("No valid carbon data available for recommendations")
else:
    st.warning("No carbon data available for recommendations")

# How it works section
st.header("How This Works")
st.write("""
This dashboard fetches real-time carbon intensity data from the ElectricityMap API:

1. **Real-time Data**: The dashboard queries ElectricityMap's API every 2 minutes (configurable)
2. **Zone Monitoring**: Tracks carbon intensity in 8 regions worldwide
3. **Deployment Recommendations**: Suggests the optimal deployment region based on current carbon intensity
4. **Visualization**: Shows data in both numerical and graphical formats

**Note**: To use this dashboard, you need to:
1. Set your ElectricityMap API token as an environment variable: `ELECTRICITY_MAP_TOKEN`
2. Or add it to Streamlit secrets (if deploying to Streamlit Cloud)
""")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("Data source: ElectricityMap API")