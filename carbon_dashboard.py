# streamlit run carbon_dashboard.py

import streamlit as st
import pandas as pd
import requests
import asyncio
import aiohttp
import time
import statistics
from datetime import datetime
import os
import pytz
from streamlit_autorefresh import st_autorefresh
import altair as alt

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
    ["🏠 Home", "🌍 Carbon Intensity", "⚖️ Carbon + Latency", "📖 How It Works", "🔧 API Setup"]
)

# Common functions
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
ZONES = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
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
    - 🎯 **Smart Recommendations**: Optimal regions balancing sustainability and performance
    
    ### What This Dashboard Offers:
    
    🔹 **Carbon Intensity Dashboard** - Monitor real-time carbon emissions across 8 global regions  
    🔹 **Combined Carbon + Latency View** - See both metrics together for informed decisions  
    🔹 **Educational Content** - Learn how carbon-aware computing works  
    🔹 **API Integration** - Uses ElectricityMap for live carbon data  
    
    ### Quick Start:
    1. Navigate using the sidebar to explore different dashboards
    2. View real-time carbon intensity data from global electricity grids
    3. Compare network latency to different cloud regions
    4. Get deployment recommendations based on your priorities
    
    ---
    *Choose a dashboard from the sidebar to get started!*
    """)
    
    # Quick stats preview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌍 Regions Monitored", "8", "Global Coverage")
    with col2:
        st.metric("📊 Data Sources", "2", "Carbon + Latency")
    with col3:
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
        for zone in ZONES:
            data = get_carbon_intensity(zone)
            if "error" not in data:
                data["region_name"] = REGION_MAPPING.get(zone, {}).get("name", zone)
                carbon_data.append(data)
    
    if carbon_data:
        # Create DataFrame
        ist = pytz.timezone("Asia/Kolkata")
        df = pd.DataFrame(carbon_data)
        # df["updatedAt"] = pd.to_datetime(df["updatedAt"]).dt.strftime("%d-%m-%Y %H:%M:%S")
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
                    # st.markdown(
                    #     f"""
                    #     <div style='text-align: center; font-size:20px; font-weight:600;'>{row['region_name']}</div>
                    #     <div style='text-align: center; font-size:24px; font-weight:700;'>{intensity} gCO<sub>2</sub>eq/kWh</div>
                    #     <div style='text-align: center; font-size:16px; background-color:{"#b7eb8f" if delta_text=="🟢 Low" else "#ffe58f" if delta_text=="🟡 Medium" else "#ffa39e"}; border-radius:12px; display:inline-block; padding:2px 10px; margin-top:0.4em;'>{delta_text}</div>
                    #     """,
                    #     unsafe_allow_html=True
                    # )

        
        # Chart - SMALLER SIZE
        st.subheader("📊 Carbon Intensity Comparison")
        chart_df = df[df["carbonIntensity"] != "N/A"].copy()
        if not chart_df.empty:
            chart_df["carbonIntensity"] = pd.to_numeric(chart_df["carbonIntensity"])
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('region_name:O', title='Region'),
                y=alt.Y('carbonIntensity:Q', title='Carbon Intensity (gCO₂eq/kWh)'),
                color=alt.Color('carbonIntensity:Q', scale=alt.Scale(scheme='redyellowgreen', reverse=True)),
                tooltip=['region_name', 'carbonIntensity', 'zone']
            ).properties(width=1150, height=450)  # FIXED SIZE
            st.altair_chart(chart, use_container_width=False)
        
        # Detailed table
        st.subheader("📋 Detailed Data")
        st.dataframe(df[["zone", "region_name", "carbonIntensity", "updatedAt"]], use_container_width=True)
        # Add custom CSS
        # st.markdown(
        #     """
        #     <style>
        #     .center-table {
        #         display: flex;
        #         justify-content: center;
        #     }
        #     </style>
        #     """,
        #     unsafe_allow_html=True
        # )

        # # Wrap dataframe inside the div
        # st.markdown('<div class="center-table">', unsafe_allow_html=True)
        # st.dataframe(df[["zone", "region_name", "carbonIntensity", "updatedAt"]], use_container_width=True)
        # st.markdown('</div>', unsafe_allow_html=True)
        
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
    # carbon_weight = st.sidebar.slider("Carbon Weight", 0.0, 1.0, 0.6, 0.1)
    # latency_weight = 1.0 - carbon_weight
    # st.sidebar.write(f"Latency Weight: {latency_weight}")
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
        for zone in ZONES:
            data = get_carbon_intensity(zone)
            if "error" not in data:
                carbon_data.append(data)
        
        # Latency data
        endpoints = {zone: REGION_MAPPING[zone]["endpoint"] for zone in ZONES if zone in REGION_MAPPING}
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
                    # df["score"] = carbon_weight * df["carbon_norm"] + latency_weight * df["latency_norm"]
                    df["score"] = norm_carbon * df["carbon_norm"] + norm_latency * df["latency_norm"]
                    df_sorted = df.sort_values("score")
            
            # Display combined metrics - FIXED
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
            
            # COMPACT TABLE INSTEAD OF BIG CHART
            st.subheader("📊 Carbon & Latency Comparison Table")
            
            # Create a clean comparison table
            table_df = df[["region_name", "carbonIntensity", "avg_latency", "p95_latency"]].copy()
            # Format numeric values
            for col in ["avg_latency", "p95_latency"]:
                table_df[col] = table_df[col].apply(lambda x: f"{x:.0f}" if isinstance(x, (int, float)) and x != "N/A" else "N/A")
            table_df.columns = ["Region", "Carbon (gCO₂eq/kWh)", "Avg Latency (ms)", "P95 Latency (ms)"]
            
            st.dataframe(table_df, use_container_width=False, height=318)
            
            # OPTIONAL: Small side-by-side chart
            st.subheader("📈 Visual Comparison (Compact)")
            
            # Prepare chart data
            chart_data = []
            for _, row in df.iterrows():
                if row['avg_latency'] != 'N/A':
                    chart_data.extend([
                        {"region": row["region_name"], "metric": "Carbon", "value": row["carbonIntensity"]},
                        {"region": row["region_name"], "metric": "Latency", "value": row["avg_latency"]}
                    ])
            
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X('region:O', title='Region'),
                    y=alt.Y('value:Q', title='Value'),
                    color=alt.Color('metric:N', title='Metric'),
                    column=alt.Column('metric:N', title='Metric Type'),
                    tooltip=['region', 'metric', 'value']
                ).resolve_scale(y='independent').properties(width=450, height=250)  # SMALLER
                st.altair_chart(chart, use_container_width=False)
            
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

elif page == "📖 How It Works":
    st.title("📖 How EcoDeploy Works")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Carbon Data", "⚡ Latency", "🎯 Optimization", "🔧 Technical"])
    
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
        
        ### Why Latency Varies
        
        - **Internet routes can change** dynamically due to congestion, outages, or ISP policies
        - **Time of day affects** network congestion patterns
        - **Geographic proximity doesn't guarantee** lowest latency due to routing
        
        ### Metrics Explained
        
        - **Average Latency**: Typical response time you can expect
        - **P95 Latency**: 95% of requests are faster than this (shows worst-case performance)
        """)
    
    with tab3:
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
        
        ### Example Calculation
        
        If you set **60% carbon priority, 40% latency priority**:
        - Region A: Low carbon (0.2), High latency (0.8) → Score = 0.6×0.2 + 0.4×0.8 = 0.44
        - Region B: High carbon (0.9), Low latency (0.1) → Score = 0.6×0.9 + 0.4×0.1 = 0.58
        - **Region A wins** because environmental impact was prioritized
        
        ### Alternative: Pareto Frontier Analysis
        
        Instead of weighting, we could show all **non-dominated solutions** - regions where you can't improve one metric without worsening the other. This gives you the full set of optimal trade-offs to choose from.
        
        ### Real-World Application
        
        - **Development/Testing**: Prioritize carbon (60-80%) since latency is less critical
        - **Production**: Balance equally (50/50) or slightly favor latency (40/60)
        - **Batch Processing**: Heavily prioritize carbon (80%+) since latency doesn't matter
        """)
    
    with tab4:
        st.header("🔧 Technical Implementation")
        st.markdown("""
        ### System Architecture
        
        ```
        ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
        │ ElectricityMap  │    │   EcoDeploy      │    │   Streamlit     │
        │      API        │◄──►│   Dashboard      │◄──►│   Community     │
        │                 │    │                  │    │     Cloud       │
        └─────────────────┘    └──────────────────┘    └─────────────────┘
                                        │
                                        ▼
                               ┌──────────────────┐
                               │ AWS Regional     │
                               │   Endpoints      │
                               │ (Latency Tests)  │
                               └──────────────────┘
        ```
        
        ### Key Technologies
        
        - **🐍 Python**: Core application logic
        - **📊 Streamlit**: Web dashboard framework  
        - **🌐 aiohttp**: Async HTTP client for latency measurement
        - **📈 Altair**: Interactive visualizations
        - **🔄 asyncio**: Concurrent latency testing
        - **⏰ Caching**: Reduces API calls and improves performance
        
        ### Data Flow
        
        1. **Carbon Data Collection**:
           - Fetch from ElectricityMap API every 5 minutes (cached)
           - Parse JSON response and extract carbon intensity
           - Handle API errors and rate limiting gracefully
        
        2. **Latency Measurement**:
           - Async HTTP requests to AWS regional endpoints
           - Multiple samples per region for statistical accuracy
           - Timeout handling for unreachable regions
        
        3. **Data Processing**:
           - Normalize both metrics to 0-1 scale
           - Apply user-defined weights
           - Calculate combined scores
           - Rank and recommend optimal regions
        
        4. **Visualization**:
           - Real-time metrics cards
           - Interactive charts with Altair
           - Responsive multi-column layout
           - Auto-refresh capabilities
        
        ### Security & Best Practices
        
        - **🔐 API Token Security**: Never hardcode tokens, use environment variables or Streamlit secrets
        - **⚡ Rate Limiting**: Respect ElectricityMap API limits with caching
        - **🛡️ Error Handling**: Graceful fallbacks when APIs are unavailable
        - **📱 Responsive Design**: Works on desktop, tablet, and mobile
        - **🔄 Auto-refresh**: Configurable refresh intervals for live monitoring
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
st.sidebar.markdown("**Data source:** ElectricityMap API")
# st.sidebar.markdown("**Made with ❤️ by Aysha**")
