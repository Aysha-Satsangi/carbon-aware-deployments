"""
COLLECT 365 DAYS FOR DE (DEBUG VERSION 02)
==========================================
Single-zone collector with detailed logging so we can see exactly
what the API returns for each 10-day window.
"""

import requests
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

TOKEN_ENV = "ELECTRICITY_MAP_TOKEN"

BASE_URL = "https://api.electricitymaps.com/v3"
ZONE = "DE"
RAW_DIR = Path("forecasting/data/raw_data_02")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def collect_de_365(days_back=365):
    token = os.getenv(TOKEN_ENV)
    if not token:
        print(f"❌ Environment variable {TOKEN_ENV} not set.")
        print(f"   PowerShell:  $Env:{TOKEN_ENV} = 'your_token_here'")
        return
    
    headers = {"auth-token": token}
    
    print("="*70)
    print(f"🚀 DEBUG: COLLECTING {days_back} DAYS FOR {ZONE} USING past-range")
    print("="*70)
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    
    print(f"\n🗓️ Date range:")
    print(f"   Start: {start_date.isoformat()}")
    print(f"   End:   {end_date.isoformat()}")
    
    carbon_records = []
    power_records = []
    
    current_start = start_date
    chunk = 0
    
    while current_start < end_date:
        chunk += 1
        current_end = current_start + timedelta(days=10)
        if current_end > end_date:
            current_end = end_date
        
        start_iso = current_start.isoformat().replace("+00:00","Z")
        end_iso = current_end.isoformat().replace("+00:00","Z")
        
        print("\n" + "-"*70)
        print(f"📦 Chunk {chunk}: {start_iso} → {end_iso}")
        
        # ---------- Carbon ----------
        carbon_url = f"{BASE_URL}/carbon-intensity/past-range"
        carbon_params = {
            "zone": ZONE,
            "start": start_iso,
            "end": end_iso,
            "temporalGranularity": "hourly"
        }
        try:
            c_resp = requests.get(carbon_url, params=carbon_params, headers=headers, timeout=30)
            print(f"   Carbon status: {c_resp.status_code}")
            if c_resp.status_code == 200:
                c_json = c_resp.json()
                if isinstance(c_json, dict) and "data" in c_json:
                    n = len(c_json["data"])
                    print(f"   Carbon data length: {n}")
                    if n > 0:
                        print(f"   Carbon example datetime: {c_json['data'][0].get('datetime','N/A')}")
                        carbon_records.extend(c_json["data"])
                else:
                    print(f"   Carbon response keys: {list(c_json.keys())}")
            else:
                print(f"   Carbon response text (first 200 chars): {c_resp.text[:200]}")
        except Exception as e:
            print(f"   Carbon error: {e}")
        
        # ---------- Power ----------
        power_url = f"{BASE_URL}/power-breakdown/past-range"
        power_params = {
            "zone": ZONE,
            "start": start_iso,
            "end": end_iso,
            "temporalGranularity": "hourly"
        }
        try:
            p_resp = requests.get(power_url, params=power_params, headers=headers, timeout=30)
            print(f"   Power status:  {p_resp.status_code}")
            if p_resp.status_code == 200:
                p_json = p_resp.json()
                if isinstance(p_json, dict) and "data" in p_json:
                    n = len(p_json["data"])
                    print(f"   Power data length: {n}")
                    if n > 0:
                        print(f"   Power example datetime: {p_json['data'][0].get('datetime','N/A')}")
                        power_records.extend(p_json["data"])
                else:
                    print(f"   Power response keys: {list(p_json.keys())}")
            else:
                print(f"   Power response text (first 200 chars): {p_resp.text[:200]}")
        except Exception as e:
            print(f"   Power error: {e}")
        
        current_start = current_end
        time.sleep(0.5)
    
    # ---------- Deduplicate & save ----------
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY BEFORE SAVE")
    print("="*70)
    print(f"   Raw carbon records collected: {len(carbon_records)}")
    print(f"   Raw power records collected:  {len(power_records)}")
    
    if carbon_records:
        carbon_by_dt = {r["datetime"]: r for r in carbon_records if "datetime" in r}
        carbon_records = sorted(carbon_by_dt.values(), key=lambda x: x["datetime"])
        print(f"   Unique carbon records: {len(carbon_records)}")
    
    if power_records:
        power_by_dt = {r["datetime"]: r for r in power_records if "datetime" in r}
        power_records = sorted(power_by_dt.values(), key=lambda x: x["datetime"])
        print(f"   Unique power records:  {len(power_records)}")
    
    carbon_path = RAW_DIR / f"{ZONE}_carbon_365d_02.json"
    power_path  = RAW_DIR / f"{ZONE}_power_365d_02.json"
    
    with open(carbon_path, "w") as f:
        json.dump({"zone": ZONE, "data": carbon_records}, f, indent=2)
    with open(power_path, "w") as f:
        json.dump({"zone": ZONE, "data": power_records}, f, indent=2)
    
    print(f"\n💾 Saved carbon to: {carbon_path}  (records: {len(carbon_records)})")
    print(f"💾 Saved power  to: {power_path}   (records: {len(power_records)})")
    print("\n✅ Done.")

if __name__ == "__main__":
    collect_de_365()
