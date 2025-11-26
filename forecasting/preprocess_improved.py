"""
INSPECT JSON DATA FILES
=======================
Check what data we actually have before preprocessing
"""

import json
from pathlib import Path

# Path to the file you want to inspect
file_path = Path(r"D:\M.tech\Semester_3\Major_Project_1\Project\20_11\ecodeploy\forecasting\data\raw_data\BE_carbon_365d.json")

print(f"📋 INSPECTING: {file_path.name}")
print("="*70)

if not file_path.exists():
    print(f"❌ File not found: {file_path}")
    exit(1)

# Load JSON
with open(file_path) as f:
    data = json.load(f)

# Check structure
print(f"\n🔍 JSON STRUCTURE:")
print(f"   Top-level keys: {list(data.keys())}")

# Get the data array
if 'data' in data:
    records = data['data']
    print(f"\n📊 RECORDS:")
    print(f"   Total records: {len(records)}")
    
    # Show first record structure
    if len(records) > 0:
        first_record = records[0]
        print(f"\n📝 FIRST RECORD FIELDS:")
        for key, value in first_record.items():
            print(f"   {key}: {value}")
        
        # Show last record
        print(f"\n📝 LAST RECORD:")
        last_record = records[-1]
        for key, value in last_record.items():
            print(f"   {key}: {value}")
        
        # Date range
        print(f"\n📅 DATE RANGE:")
        print(f"   First: {records[0].get('datetime', 'N/A')}")
        print(f"   Last: {records[-1].get('datetime', 'N/A')}")
        
        # Check for unique dates
        import pandas as pd
        from datetime import datetime
        
        dates = [pd.to_datetime(r['datetime']) for r in records if 'datetime' in r]
        if dates:
            date_range = (max(dates) - min(dates)).days
            print(f"   Date span: {date_range} days")
            print(f"   Hours covered: {len(records)}")
            print(f"   Expected for full year: ~8,760 hours")
            print(f"   Actual vs Expected: {len(records)/8760:.1%}")

print("\n" + "="*70)
"""
INSPECT ALL AVAILABLE DATA
==========================
Check all datasets you have to choose the best one
"""

import json
from pathlib import Path
import pandas as pd

raw_path = Path(r"D:\M.tech\Semester_3\Major_Project_1\Project\20_11\ecodeploy\data\raw")

print("📋 AVAILABLE DATASETS")
print("="*70)

# Find all JSON files
json_files = list(raw_path.glob("*.json"))

if not json_files:
    print(f"❌ No JSON files found in {raw_path}")
    exit(1)

print(f"✅ Found {len(json_files)} JSON files\n")

# Group by zone and type
datasets = {}

for file in sorted(json_files):
    # Parse filename: {ZONE}_{TYPE}.json
    # Examples: BE_carbon_180d.json, BE_history_6months.json
    
    parts = file.stem.split('_')
    zone = parts[0]
    dataset_type = '_'.join(parts[1:])  # Everything after zone
    
    if zone not in datasets:
        datasets[zone] = []
    
    # Load and analyze
    try:
        with open(file) as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'data' in data:
            records = data['data']
        elif isinstance(data, list):
            records = data
        else:
            records = []
        
        # Parse date range
        if records and 'datetime' in records[0]:
            dates = [pd.to_datetime(r['datetime']) for r in records if 'datetime' in r]
            date_range = (max(dates) - min(dates)).days if dates else 0
            first_date = min(dates) if dates else None
            last_date = max(dates) if dates else None
        else:
            date_range = 0
            first_date = None
            last_date = None
        
        info = {
            'file': file.name,
            'records': len(records),
            'days': date_range,
            'first': first_date,
            'last': last_date,
            'hours_expected': date_range * 24 if date_range > 0 else 0
        }
        
        datasets[zone].append(info)
        
    except Exception as e:
        print(f"⚠️ Error reading {file.name}: {e}")

# Display results
for zone in sorted(datasets.keys()):
    print(f"\n🔹 {zone} (Zone)")
    print("-" * 70)
    
    for info in datasets[zone]:
        file = info['file']
        records = info['records']
        days = info['days']
        first = info['first']
        last = info['last']
        
        # Determine quality
        if days >= 180:
            quality = "⭐⭐⭐⭐⭐ EXCELLENT (6 months+)"
        elif days >= 90:
            quality = "⭐⭐⭐⭐ VERY GOOD (3 months+)"
        elif days >= 30:
            quality = "⭐⭐⭐ GOOD (30 days+)"
        else:
            quality = "⭐ POOR (< 30 days)"
        
        print(f"  📄 {file}")
        print(f"     Records: {records:,}")
        print(f"     Date span: {days} days")
        print(f"     Range: {first.strftime('%Y-%m-%d') if first else 'N/A'} → {last.strftime('%Y-%m-%d') if last else 'N/A'}")
        print(f"     Quality: {quality}")
        print()

# Recommendation
print("\n" + "="*70)
print("💡 RECOMMENDATION:")
print("="*70)

best_datasets = {}
for zone, infos in datasets.items():
    # Find best dataset (most days)
    best = max(infos, key=lambda x: x['days'])
    best_datasets[zone] = best

avg_days = sum(d['days'] for d in best_datasets.values()) / len(best_datasets)
total_records = sum(d['records'] for d in best_datasets.values())

print(f"\n✅ Best available dataset across all zones:")
print(f"   Average days per zone: {avg_days:.0f}")
print(f"   Total records: {total_records:,}")
print(f"   Recommended: Use {best_datasets[list(best_datasets.keys())[0]]['file'].split('_')[1:]}")
