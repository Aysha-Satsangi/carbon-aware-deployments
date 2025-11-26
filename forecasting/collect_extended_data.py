# collect_extended_data.py
# Collects NEXT 6 months (Nov 2025 - May 2026) for 1-year dataset
# Run this starting Nov 17, 2025

import requests
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import toml
# try:
#     secrets = toml.load('.streamlit/secrets.toml')
#     token = secrets.get('ELECTRICITY_MAP_TOKEN')
# except:
#     token = os.getenv("ELECTRICITY_MAP_TOKEN")

class EcoDeployExtendedCollector:
    """Collect extended 6-month data (next half year)"""
    
    def __init__(self, token):
        self.token = token
        self.base = "https://api.electricitymap.com/v3"
        self.zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
        
        # NEW FOLDER for extended data
        self.raw_dir = Path("data/raw/2025_2H_collection")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.headers = {"auth-token": self.token}
        self.log_file = self.raw_dir / "collection_log.txt"
    
    def log(self, message):
        """Log to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def collect_zone_history_extended(self, zone, days_back=180):
        """Collect N days of history for EXTENDED collection"""
        self.log(f"\n📥 EXTENDED COLLECTION: {days_back} days for {zone}...")
        
        carbon_records = []
        power_records = []
        success_count = 0
        fail_count = 0
        
        for day_offset in range(days_back):
            target_date = datetime.utcnow() - timedelta(days=day_offset)
            iso_date = target_date.strftime('%Y-%m-%d')
            
            # Progress indicator every 10 days
            if day_offset % 10 == 0:
                progress = f"  📅 Processing {iso_date}... ({day_offset}/{days_back})"
                self.log(progress)
                sys.stdout.flush()
            
            # Get carbon data
            try:
                c_resp = requests.get(
                    f"{self.base}/carbon-intensity/history",
                    params={"zone": zone},
                    headers=self.headers,
                    timeout=10
                )
                if c_resp.status_code == 200:
                    c_data = c_resp.json()
                    if 'history' in c_data:
                        carbon_records.extend(c_data['history'])
                        success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                self.log(f"    ❌ Carbon error for {iso_date}: {e}")
                fail_count += 1
            
            # Get power breakdown data
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
            except Exception as e:
                pass  # Non-critical
            
            # Rate limiting: 1 second between requests
            time.sleep(1)
        
        # Save to JSON files in NEW directory
        carbon_out = {"zone": zone, "data": carbon_records, "collection_date": datetime.now().isoformat()}
        power_out = {"zone": zone, "data": power_records, "collection_date": datetime.now().isoformat()}
        
        carbon_file = self.raw_dir / f"{zone}_carbon_extended_180d.json"
        power_file = self.raw_dir / f"{zone}_power_extended_180d.json"
        
        carbon_file.write_text(json.dumps(carbon_out, indent=2))
        power_file.write_text(json.dumps(power_out, indent=2))
        
        self.log(f"✅ {zone}: saved {len(carbon_records)} carbon, {len(power_records)} power records")
        self.log(f"   Success: {success_count}, Failed: {fail_count}")
        
        return len(carbon_records)


def main():
    token = os.getenv("ELECTRICITY_MAP_TOKEN")
    
    if not token:
        print("❌ Error: Set ELECTRICITY_MAP_TOKEN environment variable")
        print("   Windows: $Env:ELECTRICITY_MAP_TOKEN = 'your_token'")
        print("   Linux: export ELECTRICITY_MAP_TOKEN='your_token'")
        exit(1)
    
    collector = EcoDeployExtendedCollector(token)
    
    total_records = 0
    for zone in collector.zones:
        try:
            records = collector.collect_zone_history_extended(zone, days_back=180)
            total_records += records
            collector.log(f"⏳ Waiting 3 seconds before next zone...\n")
            time.sleep(3)
        except Exception as e:
            collector.log(f"❌ Error with {zone}: {e}")
            continue
    
    collector.log(f"\n🎉 EXTENDED COLLECTION COMPLETE! Total records: {total_records}")


if __name__ == "__main__":
    main()
