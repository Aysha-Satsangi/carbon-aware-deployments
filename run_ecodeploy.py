#!/usr/bin/env python3
from deployment_engine.decision_maker import select_deployment_region

if __name__ == "__main__":
    region = select_deployment_region()
    print(f"region={region}")