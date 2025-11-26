"""
COLLECT 365 DAYS FOR ALL 8 ZONES (FIXED VERSION 03)
===================================================
Uses corrected date formatting from debug script
"""

import requests
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

TOKEN_ENV = "ELECTRICITY_MAP_TOKEN"
BASE_URL = "https://api.electricitymaps.com/v3"
ZONES = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
RAW_DIR = Path("forecasting/data/raw_data_02")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def collect_zone(zone, days_back=365):
    """Collect 365 days for a single zone"""
    token = os.getenv(TOKEN_ENV)
    if not token:
        print(f"❌ {TOKEN_ENV} not set")
        return 0
    
    headers = {"auth-token": token}
    
    print(f"\n{'='*70}")
    print(f"📥 COLLECTING {days_back} DAYS FOR {zone}")
    print("="*70)
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    
    carbon_records = []
    power_records = []
    
    current_start = start_date
    chunk = 0
    total_chunks = (days_back + 9) // 10
    
    while current_start < end_date:
        chunk += 1
        current_end = current_start + timedelta(days=10)
        if current_end > end_date:
            current_end = end_date
        
        # FIXED: Proper ISO format conversion
        start_iso = current_start.isoformat().replace("+00:00", "Z")
        end_iso = current_end.isoformat().replace("+00:00", "Z")
        
        progress = (chunk / total_chunks) * 100
        print(f"   [{chunk:2d}/{total_chunks}] {progress:5.1f}% | {current_start.strftime('%Y-%m-%d')}", end="", flush=True)
        
        # Carbon
        try:
            c_resp = requests.get(
                f"{BASE_URL}/carbon-intensity/past-range",
                params={"zone": zone, "start": start_iso, "end": end_iso, "temporalGranularity": "hourly"},
                headers=headers,
                timeout=30
            )
            if c_resp.status_code == 200:
                c_json = c_resp.json()
                if "data" in c_json:
                    carbon_records.extend(c_json["data"])
                    c_count = len(c_json["data"])
                else:
                    c_count = 0
            else:
                c_count = 0
        except:
            c_count = 0
        
        # Power
        try:
            p_resp = requests.get(
                f"{BASE_URL}/power-breakdown/past-range",
                params={"zone": zone, "start": start_iso, "end": end_iso, "temporalGranularity": "hourly"},
                headers=headers,
                timeout=30
            )
            if p_resp.status_code == 200:
                p_json = p_resp.json()
                if "data" in p_json:
                    power_records.extend(p_json["data"])
                    p_count = len(p_json["data"])
                else:
                    p_count = 0
            else:
                p_count = 0
        except:
            p_count = 0
        
        print(f" ✅ C:{c_count} P:{p_count}")
        
        current_start = current_end
        time.sleep(0.5)
    
    # Deduplicate
    if carbon_records:
        carbon_by_dt = {r["datetime"]: r for r in carbon_records if "datetime" in r}
        carbon_records = sorted(carbon_by_dt.values(), key=lambda x: x["datetime"])
    
    if power_records:
        power_by_dt = {r["datetime"]: r for r in power_records if "datetime" in r}
        power_records = sorted(power_by_dt.values(), key=lambda x: x["datetime"])
    
    # Save
    carbon_path = RAW_DIR / f"{zone}_carbon_365d_02.json"
    power_path = RAW_DIR / f"{zone}_power_365d_02.json"
    
    with open(carbon_path, "w") as f:
        json.dump({"zone": zone, "data": carbon_records}, f, indent=2)
    with open(power_path, "w") as f:
        json.dump({"zone": zone, "data": power_records}, f, indent=2)
    
    n_carbon = len(carbon_records)
    n_power = len(power_records)
    
    print(f"\n   💾 {carbon_path.name} ({n_carbon} records)")
    print(f"   💾 {power_path.name} ({n_power} records)")
    
    return n_carbon

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 COLLECTING 365 DAYS FOR ALL 8 ZONES (FIXED VERSION 02)")
    print("="*70)
    print(f"\n⏱️ Estimated: ~40-50 minutes for all zones")
    
    start_time = datetime.now()
    total_records = 0
    successful = 0
    
    for zone in ZONES:
        try:
            n = collect_zone(zone, days_back=365)
            total_records += n
            successful += 1
            
            if zone != ZONES[-1]:
                print(f"\n⏳ Waiting 3 seconds before next zone...")
                time.sleep(3)
        except Exception as e:
            print(f"\n❌ Failed for {zone}: {e}")
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    
    print(f"\n" + "="*70)
    print("🎉 COLLECTION COMPLETE!")
    print("="*70)
    print(f"\n📊 SUMMARY:")
    print(f"   Zones: {successful}/{len(ZONES)}")
    print(f"   Total records: {total_records:,}")
    print(f"   Time: {duration:.1f} minutes")
    print(f"\n✅ Output: {RAW_DIR.absolute()}")
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. python inspect_all_zones_02.py  (verify all zones)")
    print(f"   2. python preprocess_02.py         (feature engineering)")
    print(f"   3. python train_02.py              (train ML models)")
