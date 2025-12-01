import pandas as pd
import pickle
from datetime import timedelta,datetime

from weather_req_holiday import Weather_Requester,Holidayer

Cabrv = {'IRDUB':'IE','NZAUK':'NZ'} # Country Codes
def ARIMA_MD(loc_id:str,lat:float,long:float) -> pd.DataFrame:
    with open(f"./Dataset/Crowd/arima_models/{loc_id}_arima.pkl", "rb") as f:
        Ar_model = pickle.load(f)# grab right pickel file

    w = Weather_Requester(lat,long)# Grab weather from past and for future
    w.insert(0,'Holiday',0)# Inserting these columns to match indep input
    w.insert(0,'Date',range(len(w))) # Use range to fill in date indexing numbers 
    # Add in the date range from trim point 2025-09-30
    w['Date'] = w['Date'].apply(lambda x: datetime(2025,10,1).date() + timedelta(days=x))

    h = Holidayer(w,Cabrv.get(loc_id.split('_')[0])) # Add in the holiday data
    h = h.set_index('Date').asfreq('D').interpolate(method='linear') # numeric only
    
    fc = pd.DataFrame(Ar_model.get_forecast(exog=h,steps=len(h)).predicted_mean)
    fc = fc.reset_index().rename(columns={
        'index':'Date',
        'predicted_mean':'Avg_Daily_Pedestrian_Count'})

    fc = pd.concat([h.reset_index(drop=True),fc.reset_index(drop=True)],axis='columns') # Combing with Weather data for RC

    fc['Avg_Daily_Pedestrian_Count'] = fc['Avg_Daily_Pedestrian_Count'].apply(lambda x: round(x,0))
    return fc