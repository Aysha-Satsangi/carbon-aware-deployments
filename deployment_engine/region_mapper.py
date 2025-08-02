# Maps ElectricityMap zones to cloud provider regions
# deployment_engine/region_mapper.py

CLOUD_MAP = {
    # Germany (DE) — Frankfurt / Germany West Central
    "DE": {
        "aws":   "eu-central-1",          # AWS Frankfurt
        "gcp":   "europe-west3",          # GCP Frankfurt
        "azure": "germany-west-central"   # Azure Germany West Central
    },

    # Tōkyō (JP-TK) — Asia Pacific North East 1
    "JP-TK": {
        "aws":   "ap-northeast-1",        # AWS Tokyo
        "gcp":   "asia-northeast1",       # GCP Tokyo
        "azure": "japaneast"              # Azure Japan East
    },

    # PJM (US-MIDA-PJM) — US East 1
    "US-MIDA-PJM": {
        "aws":   "us-east-1",             # AWS N. Virginia
        "gcp":   "us-east4",              # GCP N. Virginia
        "azure": "eastus"                 # Azure East US
    },

    # Pacificorp West (US-NW-PACW) — US West 2
    "US-NW-PACW": {
        "aws":   "us-west-2",             # AWS Oregon
        "gcp":   "us-west1",              # GCP Oregon
        "azure": "westus"                 # Azure West US
    },

    # Ireland (IE) — Europe West 1
    "IE": {
        "aws":   "eu-west-1",             # AWS Ireland
        "gcp":   "europe-west1",          # GCP Belgium/Ireland
        "azure": "northeurope"            # Azure North Europe (Dublin)
    },

    # Belgium (BE) — Europe West 3
    "BE": {
        "aws":   "eu-west-3",             # AWS Paris (closest)
        "gcp":   "europe-west4",          # GCP Netherlands (closest)
        "azure": "westeurope"             # Azure West Europe (Netherlands)
    },

    # MISO (US-MIDW-MISO) — US East 2 / Central US
    "US-MIDW-MISO": {
        "aws":   "us-east-2",             # AWS Ohio
        "gcp":   "us-central1",           # GCP Iowa
        "azure": "centralus"              # Azure Central US
    },

    # Singapore (SG) — Asia Southeast 1
    "SG": {
        "aws":   "ap-southeast-1",        # AWS Singapore
        "gcp":   "asia-southeast1",       # GCP Singapore
        "azure": "southeastasia"          # Azure Southeast Asia
    }
}



def get_cloud_region(carbon_zone, cloud_provider="aws"):
    """Convert carbon zone to cloud region with fallback"""
    if carbon_zone in CLOUD_MAP:
        return CLOUD_MAP[carbon_zone].get(cloud_provider, DEFAULT_REGIONS[cloud_provider])
    return DEFAULT_REGIONS[cloud_provider]