import streamlit as st
import pandas as pd
from datetime import datetime

st.title("EcoDeploy - Carbon Aware Dashboard (Simulation)")

# Simulated deployment history
deployments = [
    {"timestamp": datetime.now(), "region": "simulated-us-west", "carbon_savings": "58%", "status": "✅ Success"},
    {"timestamp": datetime.now(), "region": "simulated-eu-central", "carbon_savings": "42%", "status": "✅ Success"},
    {"timestamp": datetime.now(), "region": "simulated-us-east", "carbon_savings": "22%", "status": "⚠️  High Carbon"}
]

df = pd.DataFrame(deployments)
st.dataframe(df)

st.metric("Total CO₂ Saved (Simulated)", "122g", "Equivalent to 5km car travel avoided")
st.metric("Cost Savings", "$100%", "Using free GitHub Actions resources")

st.write("""
### How This Works
1. **Carbon Intelligence**: Real ElectricityMap API data
2. **Region Selection**: Actual carbon-aware logic  
3. **Deployment Simulation**: Minikube instead of real cloud
4. **Cost**: $0.00 (100% free)
""")