# test_data_collection.py
from forecasting.utils import get_zones_ready_for_training, load_raw_data

print("🔍 Validating collected data...")
ready_zones = get_zones_ready_for_training()

print(f"\n📊 Summary:")
print(f"Zones ready for ML training: {len(ready_zones)}/{8}")
print(f"Ready zones: {', '.join(ready_zones)}")

# Show sample data
if ready_zones:
    sample_zone = ready_zones[0]
    sample_data = load_raw_data(sample_zone, 'carbon')
    print(f"\n📋 Sample data from {sample_zone}:")
    print(f"Total records: {len(sample_data['history'])}")
    print(f"Date range: {sample_data['history'][0]['datetime']} to {sample_data['history'][-1]['datetime']}")
    print(f"Sample carbon intensity: {sample_data['history'][0]['carbonIntensity']} gCO₂eq/kWh")
