import pandas as pd
import numpy as np
import pickle
from datetime import timedelta,date
from dateutil import parser
import holidays as hl

from sklearn.preprocessing import LabelEncoder,MinMaxScaler

import openmeteo_requests
import requests_cache
from retry_requests import retry
trim_date = parser.parse('2025-09-30').date()

# Setup the Open-Meteo API client with cache and retry on error # <--- this is from Open Meteo Api Docs
cache_session = requests_cache.CachedSession('.amriacache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

def Weather_Requester(lat:float,long:float,ArvDate:date) -> pd.DataFrame:
    FwdD = 30 + (ArvDate - date.today()).days
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": long,
        "start_date": trim_date.strftime('%Y-%m-%d'),
        "end_date": (date.today()-timedelta(days=1)).strftime('%Y-%m-%d'),
        "daily": ["precipitation_sum", "temperature_2m_mean", "relative_humidity_2m_mean", "wind_gusts_10m_mean"],
        "timezone": "America/New_York",
    }

    responses = openmeteo.weather_api(url,params=params)
    dly = responses[0].Daily()

    T1 = dly.Variables(0).ValuesAsNumpy()
    W1 = dly.Variables(1).ValuesAsNumpy()
    P1 = dly.Variables(2).ValuesAsNumpy()
    R1 = dly.Variables(3).ValuesAsNumpy()

    url = "https://seasonal-api.open-meteo.com/v1/seasonal"
    params = {
        "latitude": lat,
        "longitude": long,
        "forecast_days": FwdD,
        "timezone": "America/New_York",
        "daily": ["temperature_2m_mean", "wind_speed_10m_mean", "precipitation_sum", "relative_humidity_2m_mean"],
    }
    
    response = openmeteo.weather_api(url,params=params)
    dly = response[0].Daily()

    T2 = dly.Variables(0).ValuesAsNumpy()
    W2 = dly.Variables(1).ValuesAsNumpy()
    P2 = dly.Variables(2).ValuesAsNumpy()
    R2 = dly.Variables(3).ValuesAsNumpy()

    T = np.concatenate((T1,T2))
    w = np.concatenate((W1,W2))
    P = np.concatenate((P1,P2))
    R = np.concatenate((R1,R2))
    
    vstk = pd.DataFrame(data = np.vstack((T,w,P,R)).T)

    return vstk

def Holidayer(df:pd.DataFrame,CCode:str) -> pd.DataFrame:
    df.insert(0,'Holiday',0)
    for i in range(0,len(df)):
        dateIndx = trim_date + timedelta(days=i)
        df.loc[i,['Holiday']] = 1 if hl.country_holidays(country=CCode).get(dateIndx.strftime('%Y-%m-%d')) != None else 0
    return df

Cabrv = {'IRDUB':'IE','NZAUK':'NZ'}
def ARIMA_MD(loc_id:str,ArDate:date,lat:float,long:float) -> pd.DataFrame:
    with open(f"./Dataset/Crowd/arima_models/{loc_id}_arima.pkl", "rb") as f:
        Ar_model = pickle.load(f)
    
    w = Weather_Requester(lat,long,ArDate)
    
    h = Holidayer(w,Cabrv.get(loc_id.split('_')[0]))
    
    fc = pd.DataFrame(Ar_model.get_forecast(exog=h,steps=len(h)).predicted_mean)
    fc = pd.concat([fc.reset_index(),h],axis='columns')
    fc = fc.rename(columns={
        'index':'Date',
        'predicted_mean':'Avg_Daily_Pedestrian_Count',
        0:'Weather_Temperature_Avg',
        1:'Weather_Wind_Speed_Avg',
        2:'Weather_Precipitation_Sum',
        3:'Weather_Relative_Humidity_Avg'})
    fc['Avg_Daily_Pedestrian_Count'] = fc['Avg_Daily_Pedestrian_Count'].apply(lambda x: round(x,0))
    return fc

def KNN_MD(NewRCat:list,dfscomb:pd.DataFrame,loc_id:str)->pd.DataFrame:
    with open(f"./Dataset/Crowd/knn_model/loc_knn.pkl", "rb") as f:
        knn_model = pickle.load(f)
    df = dfscomb.copy()
    df['Date'] = df['Date'].apply(lambda x: parser.parse(x).date())
    df.loc[len(df)] = NewRCat
    df = df.reset_index(drop=True)
    df1 = df[df['Location_ID'] != loc_id] # don;t want to predict our self
    df1 = df1[['Country','City','Location_Name','Type_of_Attraction','Attraction_Category','Date','Avg_Daily_Pedestrian_Count']]

    df1['Month_Sin'] = df1['Date'].apply(lambda x: np.sin(2 * np.pi * x.month / 12))
    df1['Month_Cos'] = df1['Date'].apply(lambda x: np.cos(2 * np.pi * x.month / 12))
    df1['Day_Sin']   = df1['Date'].apply(lambda x: np.sin(2 * np.pi * (x.weekday()+1)/ 7))
    df1['Day_Cos']   = df1['Date'].apply(lambda x: np.cos(2 * np.pi * (x.weekday()+1)/ 7))

    for col in df1.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df1[col] = le.fit_transform(df1[col].astype(str))

    df1['CatComb'] = df1['Country'] + df1['City'] + df1['Location_Name'] + df1['Type_of_Attraction'] + df1['Attraction_Category']

    X = df1.drop(columns=['Country','City','Location_Name','Type_of_Attraction','Attraction_Category','Date'])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)[-1]
    
    yPD,yPI = knn_model.kneighbors([X])
    print(yPD)
    print(yPI)
    for i in range(len(yPI[0])):
        idx = yPI[0,i]
        if df['Location_ID'].loc[idx] != loc_id:
            # print(NewRCat)
            # print(df.loc[idx])
            # print(df['Avg_Daily_Pedestrian_Count'].loc[idx])
            return df.loc[idx]