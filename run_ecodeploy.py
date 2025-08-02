#!/usr/bin/env python3
import sys
from deployment_engine.decision_maker import select_deployment_region

if __name__ == "__main__":
    region = select_deployment_region()
    print(f"Selected region: {region}")
    with open("region.txt", "w") as f:
        f.write(region)