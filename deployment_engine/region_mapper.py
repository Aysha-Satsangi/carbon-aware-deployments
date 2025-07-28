# Maps ElectricityMap zones to cloud provider regions
CLOUD_MAP = {
    "DE": {
        "aws": "eu-central-1",
        "gcp": "europe-west3",
        "azure": "germany-west-central"
    },
    "US-MIDA-PJM": {
        "aws": "us-east-1",
        "gcp": "us-east4",
        "azure": "eastus"
    },
    "US-NW-PACW": {
        "aws": "us-west-2",
        "gcp": "us-west1",
        "azure": "westus"
    },
    "IE": {
        "aws": "eu-west-1",
        "gcp": "europe-west1",
        "azure": "northeurope"
    },
    "SG": {
        "aws": "ap-southeast-1",
        "gcp": "asia-southeast1",
        "azure": "southeastasia"
    },
    # Add other zones similarly
}

def get_cloud_region(carbon_zone, cloud_provider="aws"):
    """Convert carbon zone to cloud region"""
    if carbon_zone in CLOUD_MAP:
        return CLOUD_MAP[carbon_zone].get(cloud_provider, "unknown")
    return "unknown"