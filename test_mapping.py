#!/usr/bin/env python3
"""
Test script to verify region mapping works correctly
"""

import sys
import os

# Add the deployment_engine directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'deployment_engine'))

try:
    from deployment_engine.region_mapper import get_cloud_region
    
    # Test zones from your configuration
    test_zones = ["DE", "US-MIDA-PJM", "US-NW-PACW", "IE", "SG", "BE", "US-MIDW-MISO", "JP-TK"]
    
    print("🔍 Testing Simulated Region Mapping")
    print("=" * 50)
    print(f"{'Carbon Zone':<15} | {'Simulated Region':<20}")
    print("-" * 50)
    
    for zone in test_zones:
        region = get_cloud_region(zone)
        print(f"{zone:<15} | {region:<20}")
    
    print("=" * 50)
    print("✅ Test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure your region_mapper.py is in the deployment_engine directory")
except Exception as e:
    print(f"❌ Unexpected error: {e}")