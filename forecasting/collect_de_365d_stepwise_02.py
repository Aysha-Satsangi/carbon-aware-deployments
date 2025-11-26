import requests
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

TOKEN_ENV = "ELECTRICITY_MAP_TOKEN"
BASE_URL = "https://api.electricitymaps.com/v3"
ZONE = "DE"
RAW_DIR = Path("forecasting/data/raw_data_02")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def main(days_back=365, chunk_days=10):
    token = os.getenv(TOKEN_ENV)
    if not token:
        print(f"❌ Environment variable {TOKEN_ENV} not set.")
        return
    
    headers = {"auth-token": token}
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    current_start = start_date
    chunk = 0

    carbon_records = []
    power_records = []

    print("="*70)
    print(f"🚀 STEPWISE ITERATIVE DOWNLOAD FOR {ZONE.upper()}, {days_back} DAYS")
    print("="*70)

    while current_start < end_date:
        chunk += 1
        current_end = current_start + timedelta(days=chunk_days)
        if current_end > end_date:
            current_end = end_date
        start_iso = current_start.isoformat().replace("+00:00", "Z")
        end_iso = current_end.isoformat().replace("+00:00", "Z")

        print(f"\n=== CHUNK {chunk}: {start_iso} → {end_iso}")

        # Carbon request
        c_resp = requests.get(
            f"{BASE_URL}/carbon-intensity/past-range",
            params={"zone": ZONE, "start": start_iso, "end": end_iso, "temporalGranularity": "hourly"},
            headers=headers,
            timeout=30,
        )

        # Power request
        p_resp = requests.get(
            f"{BASE_URL}/power-breakdown/past-range",
            params={"zone": ZONE, "start": start_iso, "end": end_iso, "temporalGranularity": "hourly"},
            headers=headers,
            timeout=30,
        )

        # Parse carbon response
        c_data = c_resp.json() if c_resp.status_code == 200 else {}
        p_data = p_resp.json() if p_resp.status_code == 200 else {}

        c_records = c_data.get("data", [])
        p_records = p_data.get("data", [])

        print(f"   Carbon: status={c_resp.status_code}, records={len(c_records)}")
        print(f"   Power:  status={p_resp.status_code}, records={len(p_records)}")

        # Add to collections
        carbon_records.extend(c_records)
        power_records.extend(p_records)

        # Print sample datetime for visual confirmation
        if c_records:
            print("   Carbon sample:", c_records[0].get("datetime"), "...", c_records[-1].get("datetime"))
        if p_records:
            print("   Power sample:", p_records[0].get("datetime"), "...", p_records[-1].get("datetime"))

        # Optional: write chunk temp files for manual inspection
        with open(RAW_DIR / f"{ZONE}_carbon_chunk{chunk:02d}_02.json", "w") as f:
            json.dump({"zone": ZONE, "data": c_records}, f, indent=2)
        with open(RAW_DIR / f"{ZONE}_power_chunk{chunk:02d}_02.json", "w") as f:
            json.dump({"zone": ZONE, "data": p_records}, f, indent=2)

        # PAUSE AFTER EACH CHUNK (remove/comment input for unattended)
        input(f"-> Chunk {chunk} done. Press Enter to continue to the next chunk...")

        current_start = current_end

    print("\n===")
    print(f"Total carbon records collected: {len(carbon_records)}")
    print(f"Total power records collected:  {len(power_records)}")
    print("===")

    # Deduplicate by datetime and save final version
    carbon_by_dt = {r["datetime"]: r for r in carbon_records if "datetime" in r}
    power_by_dt = {r["datetime"]: r for r in power_records if "datetime" in r}
    carbon_final = sorted(carbon_by_dt.values(), key=lambda x: x["datetime"])
    power_final = sorted(power_by_dt.values(), key=lambda x: x["datetime"])

    with open(RAW_DIR / f"{ZONE}_carbon_365d_02.json", "w") as f:
        json.dump({"zone": ZONE, "data": carbon_final}, f, indent=2)
    with open(RAW_DIR / f"{ZONE}_power_365d_02.json", "w") as f:
        json.dump({"zone": ZONE, "data": power_final}, f, indent=2)

    print(f"\n💾 Saved final carbon to: {ZONE}_carbon_365d_02.json ({len(carbon_final)})")
    print(f"💾 Saved final power to: {ZONE}_power_365d_02.json ({len(power_final)})")

if __name__ == "__main__":
    main()
