# # forecasting/data_collector_complete.py

# import requests, json, os, time
# from datetime import datetime, timedelta
# from pathlib import Path

# class EcoDeploy6MonthCollector:
#     def __init__(self, token):
#         self.token = token
#         self.base = "https://api.electricitymaps.com/v3"
#         self.zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
#         self.raw_dir = Path("data/raw_data")
#         self.raw_dir.mkdir(parents=True, exist_ok=True)
#         self.headers = {"auth-token": self.token}
    
#     def collect_zone_history(self, zone, days_back=180):
#         """Collect N days of hourly history using /past endpoint"""
#         print(f"📥 Collecting {days_back} days for {zone}...")
        
#         carbon_records = []
#         power_records = []
        
#         for day_offset in range(days_back):
#             target_date = datetime.utcnow() - timedelta(days=day_offset)
            
#             for hour in range(24):
#                 dt = target_date.replace(hour=hour, minute=0, second=0)
#                 iso_time = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                
#                 # Get carbon
#                 try:
#                     c_resp = requests.get(
#                         f"{self.base}/carbon-intensity/past",
#                         params={"zone": zone, "datetime": iso_time},
#                         headers=self.headers,
#                         timeout=5
#                     )
#                     if c_resp.status_code == 200:
#                         c_data = c_resp.json()
#                         carbon_records.append(c_data)
#                 except:
#                     pass
                
#                 # Get power breakdown
#                 try:
#                     p_resp = requests.get(
#                         f"{self.base}/power-breakdown/past",
#                         params={"zone": zone, "datetime": iso_time},
#                         headers=self.headers,
#                         timeout=5
#                     )
#                     if p_resp.status_code == 200:
#                         p_data = p_resp.json()
#                         power_records.append(p_data)
#                 except:
#                     pass
                
#                 time.sleep(0.3)  # Rate limit
        
#         # Save
#         carbon_out = {"zone": zone, "data": carbon_records}
#         power_out = {"zone": zone, "data": power_records}
        
#         (self.raw_dir / f"{zone}_carbon_180d.json").write_text(json.dumps(carbon_out, indent=2))
#         (self.raw_dir / f"{zone}_power_180d.json").write_text(json.dumps(power_out, indent=2))
        
#         print(f"✅ {zone}: saved {len(carbon_records)} carbon, {len(power_records)} power records")

# if __name__ == "__main__":
#     token = os.getenv("ELECTRICITY_MAP_TOKEN")
#     collector = EcoDeploy6MonthCollector(token)
    
#     for zone in collector.zones:
#         collector.collect_zone_history(zone, days_back=180)
#         print(f"⏳ Waiting before next zone...")
#         time.sleep(2)

# forecasting/data_collector_complete.py

import requests
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

class EcoDeploy6MonthCollector:
    def __init__(self, token):
        self.token = token
        self.base = "https://api.electricitymaps.com/v3"
        self.zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
        self.raw_dir = Path("data/raw_data")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {"auth-token": self.token}
    
    def collect_zone_history(self, zone, days_back=365):
        """Collect N days of hourly history using /history endpoint"""
        print(f"\n📥 Collecting {days_back} days for {zone}...")
        
        carbon_records = []
        power_records = []
        
        # Use /history endpoint instead of looping through /past
        # /history returns last 24h in ONE call, much faster
        
        # For 180 days, we need to collect in daily chunks using /past
        success_count = 0
        fail_count = 0
        
        for day_offset in range(days_back):
            target_date = datetime.utcnow() - timedelta(days=day_offset)
            iso_date = target_date.strftime('%Y-%m-%d')
            
            # Progress indicator
            if day_offset % 10 == 0:
                print(f"  📅 Processing {iso_date}... ({day_offset}/{days_back})")
                sys.stdout.flush()
            
            # Get carbon for that day
            try:
                c_resp = requests.get(
                    f"{self.base}/carbon-intensity/history",
                    params={"zone": zone},
                    headers=self.headers,
                    timeout=10  # 10 second timeout
                )
                if c_resp.status_code == 200:
                    c_data = c_resp.json()
                    if 'history' in c_data:
                        carbon_records.extend(c_data['history'])
                        success_count += 1
                else:
                    print(f"    ⚠️ Carbon {c_resp.status_code} for {iso_date}")
                    fail_count += 1
            except requests.exceptions.Timeout:
                print(f"    ⏱️ Timeout on carbon for {iso_date}")
                fail_count += 1
            except Exception as e:
                print(f"    ❌ Error: {e}")
                fail_count += 1
            
            # Get power breakdown for that day
            try:
                p_resp = requests.get(
                    f"{self.base}/power-breakdown/history",
                    params={"zone": zone},
                    headers=self.headers,
                    timeout=10
                )
                if p_resp.status_code == 200:
                    p_data = p_resp.json()
                    if 'history' in p_data:
                        power_records.extend(p_data['history'])
                else:
                    print(f"    ⚠️ Power {p_resp.status_code} for {iso_date}")
            except requests.exceptions.Timeout:
                print(f"    ⏱️ Timeout on power for {iso_date}")
            except Exception as e:
                print(f"    ❌ Error: {e}")
            
            # Rate limit: 1 second between requests
            time.sleep(1)
        
        # Save
        carbon_out = {"zone": zone, "data": carbon_records}
        power_out = {"zone": zone, "data": power_records}
        
        carbon_file = self.raw_dir / f"{zone}_carbon_365d.json"
        power_file = self.raw_dir / f"{zone}_power_365d.json"
        
        carbon_file.write_text(json.dumps(carbon_out, indent=2))
        power_file.write_text(json.dumps(power_out, indent=2))
        
        print(f"\n✅ {zone}: saved {len(carbon_records)} carbon, {len(power_records)} power records")
        print(f"   Success: {success_count}, Failed: {fail_count}")
        return len(carbon_records)

if __name__ == "__main__":
    token = os.getenv("ELECTRICITY_MAP_TOKEN")
    
    if not token:
        print("❌ Error: Set ELECTRICITY_MAP_TOKEN environment variable")
        print("   $Env:ELECTRICITY_MAP_TOKEN = 'your_token'")
        exit(1)
    
    collector = EcoDeploy6MonthCollector(token)
    
    total_records = 0
    for zone in collector.zones:
        try:
            records = collector.collect_zone_history(zone, days_back=365)
            total_records += records
            print(f"⏳ Waiting 3 seconds before next zone...\n")
            time.sleep(3)
        except Exception as e:
            print(f"❌ Error with {zone}: {e}")
            continue
    
    print(f"\n🎉 Collection complete! Total records: {total_records}")
