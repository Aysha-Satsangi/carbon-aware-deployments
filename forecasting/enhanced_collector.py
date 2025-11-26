# Uses /carbon-intensity/past-range and /carbon-intensity/forecast
class EnhancedCollector:
    def collect_past_range(self, zone, start, end):
        url = f"{self.base_url}/carbon-intensity/past-range"
        params = {
            "zone": zone,
            "start": start,      # ISO format
            "end": end,
            "temporalGranularity": "hourly"
        }
        # auth-token header… GET, save JSON
    def collect_forecast(self, zone, horizon=72):
        url = f"{self.base_url}/carbon-intensity/forecast"
        params = {"zone": zone, "horizonHours": horizon}
        # GET, save JSON
    def collect_power_breakdown_past_range(...): ...
    def collect_all(self):
        # for each zone, call past-range for last 30 days,
        # then forecast to fill future gaps
