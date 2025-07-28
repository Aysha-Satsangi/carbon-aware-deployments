import requests
import time

token = "zCwoSGaD1ftozi6z3ZRi"  # Your actual token
headers = {"auth-token": token}

zones = ["DE", "JP-TK", "US-MIDA-PJM", "US-NW-PACW", "IE", "BE", "US-MIDW-MISO", "SG"]

def get_carbon_data():
    results = []
    for zone in zones:
        try:
            url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                intensity = data.get("carbonIntensity", "N/A")
                updated = data.get("datetime", "N/A")
                results.append((zone, intensity, updated))
            else:
                print(f"⚠️ Error {response.status_code} for {zone}")
                results.append((zone, "Error", "N/A"))
                
            time.sleep(0.5)  # Avoid rate limits
            
        except Exception as e:
            print(f"🚨 Critical error for {zone}: {str(e)}")
            results.append((zone, "Failed", "N/A"))
    
    results.sort(key=lambda x: x[1] if isinstance(x[1], (int, float)) else float('inf'))
    return results

# Test the function
if __name__ == "__main__":
    print("Testing carbon data fetch...")
    data = get_carbon_data()
    print("\nResults:")
    print(f"{'Zone':<15} | {'gCO2/kWh':<10} | Updated")
    print("-" * 45)
    for zone, intensity, updated in data:
        print(f"{zone:<15} | {intensity:<10} | {updated}")