# forecasting/data_collector.py

import requests, json, os, time
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st

class DataCollector:
    def __init__(self, token, raw_dir="data/raw"):
        self.token = token
        self.base = "https://api.electricitymap.org/v3"
        self.zones = ["DE","US-MIDA-PJM","US-NW-PACW","IE","SG","BE","US-MIDW-MISO","JP-TK"]
        self.raw_dir = Path(raw_dir); self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {"auth-token": self.token}

    def collect_zone(self, zone, total_days=30, chunk_days=10):
        print(f"\n📥 Collecting {total_days} days for {zone} in {chunk_days}-day chunks")
        all_records = []
        end = datetime.utcnow()
        chunks = total_days // chunk_days
        for i in range(chunks):
            chunk_end = end - timedelta(days=chunk_days * i)
            chunk_start = chunk_end - timedelta(days=chunk_days)
            params = {
                "zone": zone,
                "start": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end":   chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "temporalGranularity": "hourly"
            }
            url = f"{self.base}/carbon-intensity/past-range"
            r = requests.get(url, headers=self.headers, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json().get("data", [])
                print(f"  Chunk {i+1}: {len(data)} records")
                all_records.extend(data)
            else:
                print(f"  Error {r.status_code}: {r.text}")
            time.sleep(1)

        # Save merged JSON
        out = {"zone": zone, "history": all_records}
        fn = f"{zone}_past_{len(all_records)}h.json"
        (self.raw_dir / fn).write_text(json.dumps(out, indent=2))
        print(f"✅ {zone}: saved {len(all_records)} total records")
        return out

    def run_all(self):
        token = self.token
        if not token:
            raise RuntimeError("Set ELECTRICITY_MAP_TOKEN")
        for z in self.zones:
            self.collect_zone(z, total_days=30, chunk_days=10)

    def collect_power_breakdown(self, zone, days=30):
        """Collect power breakdown history"""
        print(f"\n⚡ Collecting power breakdown for {zone}...")
        
        all_records = []
        end = datetime.now(datetime.UTC)
        chunks = days // 10
        
        for i in range(chunks):
            chunk_end = end - timedelta(days=10 * i)
            chunk_start = chunk_end - timedelta(days=10)
            
            params = {
                "zone": zone,
                "start": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "temporalGranularity": "hourly"
            }
            
            url = f"{self.base}/power-breakdown/past-range"
            r = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if r.status_code == 200:
                data = r.json().get("data", [])
                print(f"  Chunk {i+1}: {len(data)} records")
                all_records.extend(data)
            else:
                print(f"  Error {r.status_code}")
            
            time.sleep(1)
        
        # Save merged JSON
        out = {"zone": zone, "data": all_records}
        fn = f"{zone}_power_{len(all_records)}h.json"
        (self.raw_dir / fn).write_text(json.dumps(out, indent=2))
        print(f"✅ {zone}: saved {len(all_records)} power records")
        return out

if __name__=="__main__":
    token = os.getenv("ELECTRICITY_MAP_TOKEN") or st.secrets.get("ELECTRICITY_MAP_TOKEN")
    dc = DataCollector(token)
    dc.run_all()
