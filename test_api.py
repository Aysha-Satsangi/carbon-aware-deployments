import requests
import os

# Get your token from environment variable
API_TOKEN = os.getenv("ELECTRICITY_MAP_TOKEN")

if not API_TOKEN:
    print("❌ Error: ELECTRICITY_MAP_TOKEN not set")
    print("Set it with: $Env:ELECTRICITY_MAP_TOKEN = 'your_token_here'")
    exit(1)

# Test the API
url = "https://api.electricitymaps.com/v3/carbon-intensity/latest"
params = {"zone": "DE"}
headers = {"auth-token": API_TOKEN}

print(f"🔍 Testing API with token: {API_TOKEN[:10]}***")
print(f"📍 Zone: DE\n")

try:
    response = requests.get(url, params=params, headers=headers, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS! API is working.")
        print(f"📊 Current Carbon Intensity (DE): {data.get('carbonIntensity')} gCO₂/kWh")
        print(f"⏰ Time: {data.get('datetime')}")
        print(f"\n🎉 Your API token is valid and working!")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        
except requests.exceptions.RequestException as e:
    print(f"❌ Connection error: {e}")
