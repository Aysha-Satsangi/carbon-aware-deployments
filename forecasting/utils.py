# forecasting/utils.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

def load_raw_data(zone, data_type='carbon'):
    """Load raw JSON data for a zone"""
    data_dir = Path("data/raw")
    
    if data_type == 'carbon':
        pattern = f"{zone}_carbon_history_*.json"
    elif data_type == 'power':
        pattern = f"{zone}_power_history_*.json"
    
    files = list(data_dir.glob(pattern))
    if not files:
        return None
    
    # Load the most recent file
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def validate_data_quality(zone):
    """Check if we have enough good quality data for training"""
    carbon_data = load_raw_data(zone, 'carbon')
    
    if not carbon_data or 'history' not in carbon_data:
        return False, "No carbon data found"
    
    records = carbon_data['history']
    
    # Check minimum records (need at least 30 days worth)
    if len(records) < 30 * 24:  # 30 days * 24 hours
        return False, f"Only {len(records)} records, need at least 720"
    
    # Check for too many null values
    null_count = sum(1 for record in records if record.get('carbonIntensity') is None)
    null_percentage = null_count / len(records)
    
    if null_percentage > 0.3:  # More than 30% nulls
        return False, f"Too many null values: {null_percentage:.1%}"
    
    return True, f"Good quality: {len(records)} records, {null_percentage:.1%} nulls"

def get_zones_ready_for_training():
    """Check which zones have enough data for training"""
    zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
    ready_zones = []
    
    print("🔍 Checking data quality for training...")
    for zone in zones:
        is_ready, message = validate_data_quality(zone)
        status = "✅" if is_ready else "❌"
        print(f"  {status} {zone}: {message}")
        
        if is_ready:
            ready_zones.append(zone)
    
    return ready_zones
