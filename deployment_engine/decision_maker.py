from carbon_data.carbon_fetcher import get_carbon_data
from .region_mapper import get_cloud_region

def select_deployment_region(provider="aws"):
    """Select best region based on carbon data"""
    carbon_data = get_carbon_data()
    
    if not carbon_data:
        return "us-east-1"  # Default fallback
    
    # Get top 3 lowest carbon zones
    best_zones = [item[0] for item in carbon_data[:3]]
    
    # Convert to cloud regions
    cloud_regions = [get_cloud_region(zone, provider) for zone in best_zones]
    
    print(f"Top regions: {', '.join(cloud_regions)}")
    return cloud_regions[0]  # Return the best one