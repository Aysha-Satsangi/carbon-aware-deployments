import json
import csv
from datetime import datetime
import os
import time

class DeploymentLogger:
    def __init__(self, log_file="deployment_history.csv"):
        self.log_file = log_file
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'service_name', 'region', 
                    'deployment_time_seconds', 'carbon_intensity',
                    'status', 'image_size_mb', 'resource_usage'
                ])
    
    def start_timer(self):
        """Start a timer for deployment measurement"""
        return time.time()

    def log_deployment(self, service_name, region, deployment_time, 
                      carbon_intensity, status="success", image_size=0, resource_usage="low"):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'service_name': service_name,
            'region': region,
            'deployment_time_seconds': deployment_time,
            'carbon_intensity': carbon_intensity,
            'status': status,
            'image_size_mb': image_size,
            'resource_usage': resource_usage
        }
        
        # Append to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                log_entry['timestamp'],
                log_entry['service_name'],
                log_entry['region'],
                log_entry['deployment_time_seconds'],
                log_entry['carbon_intensity'],
                log_entry['status'],
                log_entry['image_size_mb'],
                log_entry['resource_usage']
            ])
        
        # Also append to JSON for easy reading
        json_file = self.log_file.replace('.csv', '.json')
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        data.append(log_entry)
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return log_entry

# Singleton instance
deployment_logger = DeploymentLogger()