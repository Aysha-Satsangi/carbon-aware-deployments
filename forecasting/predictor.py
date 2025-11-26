# forecasting/predictor.py
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from forecasting.data_preprocessor import process_zone

MODELS="data/models"; PROCESSED="data/processed"

def forecast_zone(zone,history_hours=24,forecast_hours=24):
    # load latest processed CSV
    df=pd.read_csv(f"{PROCESSED}/{zone}.csv",index_col=0,parse_dates=True)
    series=df["carbon"].dropna().values
    X=series[-history_hours:].reshape(1,history_hours,1)
    m=load_model(f"{MODELS}/{zone}.h5")
    pred=m.predict(X)[0]
    times=[datetime.now()+timedelta(hours=i+1) for i in range(forecast_hours)]
    return pd.DataFrame({"time":times,"forecast":pred})

if __name__=="__main__":
    for zone in ["DE"]:
        df=forecast_zone(zone)
        print(df.head())
        df.to_csv(f"{zone}_forecast.csv",index=False)
        print(f"Forecast saved to {zone}_forecast.csv")
    print("Forecasting complete.")