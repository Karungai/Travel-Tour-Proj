import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import os
from datetime import date,timedelta,datetime
from dateutil import parser
import plotly.express as px
from streamlit_plotly_events import plotly_events
from googletrans import Translator

from extra import ARIMA_MD,KNN_MD # Model Exc File

#^ PAGE CONFIGURATION---------------------------- 
st.set_page_config(
    page_title="Start Your Travel Journey", 
    page_icon="🌍", 
    layout="wide"
)

#^ BACKGROUND STYLE---------------------------- 
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://images.unsplash.com/photo-1517760444937-f6397edcbbcd');
    background-size: cover;
    background-attachment: fixed;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
</style>'''
st.markdown(page_bg_img, unsafe_allow_html=True)

#^ GLOBAL VARS & FUNCS-------------------------   
# ---- LOAD DATA ---- 
folder1 = "./Dataset/Crowd/data_weather/Final"
dfs_comb = pd.DataFrame()
for df in [pd.read_csv(os.path.join(folder1,f)) for f in os.listdir(folder1) if f.endswith('.csv')]:
    dfs_comb = pd.concat([dfs_comb,df],axis='rows')
folder2 = "./Dataset/Crowd/Flight/flight_paths_nc.csv"
flights = pd.read_csv(folder2)
flights['apt_time_dt_ds'] = flights['apt_time_dt_ds'].apply(lambda x: parser.parse(x).date())#datetime.date(YYYY, MM, DD)
flights['apt_time_dt_dp'] = flights['apt_time_dt_dp'].apply(lambda x: parser.parse(x).date())

def Dest_Forecastig_Data_Get():
    if st.session_state['sel_org'] != None and st.session_state['sel_Arv_dte'] != None:
        LocData = dfs_comb[dfs_comb['Location_Name'] == st.session_state['sel_locN']]
        MetaData = LocData[['Country','City','Location_ID','Location_Name','Type_of_Attraction','Attraction_Category','Latitude','Longitude']].drop_duplicates().reset_index(drop=True).loc[0]
        FC = ARIMA_MD(MetaData['Location_ID'],st.session_state['sel_Arv_dte'],MetaData['Latitude'],MetaData['Longitude'])
        FC['Date'] = FC['Date'].apply(lambda x : pd.Timestamp(x).date())#datetime.date(YYYY, MM, DD)
        st.session_state['FC_sel_Dest'] = FC
        #Start Date 2025-01-10 to current - 1 day Histrocial weather, current to user selected arivial time Forecast weather
        # Return Dataframe         
        flgData1 = flights[(flights['City_dp'] == st.session_state['sel_org']) & 
                          (flights['City_ds'] == MetaData['City']) & 
                          (flights['apt_time_dt_ds'] >= date.today()) & 
                          (flights['apt_time_dt_ds'] <= FC.loc[len(FC)-1]['Date'])]
        st.session_state['Flght_sel_Dest'] = flgData1
        
        NEwR = [MetaData['Country'],
                MetaData['City'],
                '-',
                MetaData['Location_Name'],
                MetaData['Type_of_Attraction'],
                MetaData['Attraction_Category'],
                MetaData['Latitude'],
                MetaData['Longitude'],
                st.session_state['sel_Arv_dte'],
                FC.loc[len(FC)-1]['Avg_Daily_Pedestrian_Count'],
                0.0,0.0,0.0,0.0,0.0]
        RC = KNN_MD(NEwR,dfs_comb,MetaData['Location_ID'])
        RC['Date'] = RC['Date'].replace(year=date.today().year)
        st.session_state['RC_alt_Dest'] = RC

        flgData2 = flights[(flights['City_dp'] == st.session_state['sel_org']) & 
                          (flights['City_ds'] == RC['City']) & 
                          (flights['apt_time_dt_ds'] >= date.today()) & 
                          (flights['apt_time_dt_ds'] <= RC['Date'])]
        st.session_state['Flght_alt_Dest'] = flgData2
    else:
        st.session_state['FC_sel_Dest'] = pd.DataFrame()
        st.session_state['Flght_sel_Dest'] = pd.DataFrame()
        st.session_state['RC_alt_Dest'] = pd.DataFrame()
        st.session_state['Flght_alt_Dest'] = pd.DataFrame()

def poisUpdate() -> pd.DataFrame:
    Query = st.session_state['user_sel']
    Query = Query[2:]
    MaxC,MedC,MinC = dfs_comb['Avg_Daily_Pedestrian_Count'].max(),dfs_comb['Avg_Daily_Pedestrian_Count'].median(),dfs_comb['Avg_Daily_Pedestrian_Count'].min()
    MaxT,MedT,MinT = dfs_comb['Weather_Temperature_Avg'].max(),dfs_comb['Weather_Temperature_Avg'].median(),dfs_comb['Weather_Temperature_Avg'].min()
    if Query[2] == None or Query[2] == 'HIGH': Query[2] = MaxC
    if Query[2] == 'MEDIUM' : Query[2] = MedC
    if Query[2] == 'LOW' : Query[2] = (MedC - MinC)/2 + MinC
    if Query[3] == None or Query[3] == 'HIGH': Query[3] = MaxT
    if Query[3] == 'MEDIUM' : Query[3] = MedT
    if Query[3] == 'LOW' : Query[3] = (MedT - MinT)/2 + MinT
        
    if Query[0] == None and Query[1] == None: #00
        dfs_c = dfs_comb[(dfs_comb['Avg_Daily_Pedestrian_Count'] <= Query[2]) & 
                         (dfs_comb['Weather_Temperature_Avg'] <= Query[3])]
    
    if Query[0] == None and Query[1] != None: #01
        dfs_c = dfs_comb[(dfs_comb['Type_of_Attraction'] == Query[1]) & 
                         (dfs_comb['Avg_Daily_Pedestrian_Count'] <= Query[2]) & 
                         (dfs_comb['Weather_Temperature_Avg'] <= Query[3])]
        
    if Query[0] != None and Query[1] == None: #10
        dfs_c = dfs_comb[(dfs_comb['Attraction_Category'] == Query[0]) & 
                         (dfs_comb['Avg_Daily_Pedestrian_Count'] <= Query[2]) & 
                         (dfs_comb['Weather_Temperature_Avg'] <= Query[3])]
    
    if Query[0] != None and Query[1] != None: #11
        dfs_c = dfs_comb[(dfs_comb['Attraction_Category'] == Query[0]) &
                         (dfs_comb['Type_of_Attraction'] == Query[1]) &  
                         (dfs_comb['Avg_Daily_Pedestrian_Count'] <= Query[2]) & 
                         (dfs_comb['Weather_Temperature_Avg'] <= Query[3])]
    return dfs_c

#^ SESSION RELATED-----------------------------
# --- HOUSING FORECAST & FLIGHT DATA ---
if 'FC_sel_Dest' not in st.session_state:
    st.session_state['FC_sel_Dest'] = pd.DataFrame()
if 'Flght_sel_Dest' not in st.session_state:
    st.session_state['Flght_sel_Dest'] = pd.DataFrame()
if 'RC_alt_Dest' not in st.session_state:
    st.session_state['RC_alt_Dest'] = pd.DataFrame()
if 'Flght_alt_Dest' not in st.session_state:
    st.session_state['Flght_alt_Dest'] = pd.DataFrame()
# --- HOUSING USER SELECTIONS ---
if 'user_sel' not in st.session_state:
    st.session_state['user_sel'] = [None,None,None,None,None,None]
# ---- SESSION STATE INIT ----
for k in ["sel_att_cat","sel_att_type","sel_org","sel_Arv_dte","sel_crowd","sel_temp","sel_locN"]:
    if k not in st.session_state:
        st.session_state[k] = None
# ---- CALLBACKS ----
def update_user_sel():
    st.session_state['user_sel'][0] = st.session_state['sel_org']
    st.session_state['user_sel'][1] = st.session_state['sel_Arv_dte']
    st.session_state['user_sel'][2] = st.session_state['sel_att_cat']
    if st.session_state['sel_att_cat'] == None:
        st.session_state['sel_att_type'] = None
    st.session_state['user_sel'][3] = st.session_state['sel_att_type']
    st.session_state['user_sel'][4] = st.session_state['sel_crowd']
    st.session_state['user_sel'][5] = st.session_state['sel_temp']
    poisUpdate()

pois = poisUpdate()

#^ LAYOUT STRUCTURE---------------------------- 
O_W = 1
uppR = st.columns([O_W,7,O_W]) 
midR = st.columns([O_W,3,4,O_W],gap='medium')
lowR = st.columns([O_W,2,2.5,2.5,O_W],gap='small')

#* ---------------------------- ROW 1: TITLE
with uppR[1]:
    st.markdown("<h1 style='text-align:center; font-size:60px;'>Start Your Travel Journey</h1>", unsafe_allow_html=True)
    st.divider()

#* ---------------------------- ROW 2: OPTIONS & LOC EDA
with midR[1]:
    ops = st.columns([1]) + st.columns([1,1]) + st.columns([1,1]) + st.columns([1,1,1])
    with ops[0]:
        st.subheader("Itineraries")

    with ops[1]:
        sel_org = st.selectbox("Choose a Orgin:",
                            flights['City_dp'].unique().tolist(),
                            index=None,
                            placeholder="Select...",
                            key="sel_org",
                            on_change=update_user_sel)

    with ops[2]:
        nextday = date.today() + timedelta(days=1)
        MaxD = nextday + timedelta(days=120)
        sel_Arv_dte =  st.date_input(
            "Select Travel Arrival Date",
            min_value=nextday,
            max_value=MaxD,
            format="YYYY-MM-DD",
            key="sel_Arv_dte",
            on_change=update_user_sel
        )

    with ops[3]:
        AttCatL = dfs_comb['Attraction_Category'].unique().tolist()
        sel_att_cat = st.selectbox("Choose Attraction Category:",
                                AttCatL,
                                index=None,
                                key="sel_att_cat",
                                placeholder="Select...",
                                on_change=update_user_sel)

    with ops[4]:
        att_type_list = dfs_comb[dfs_comb['Attraction_Category'] == sel_att_cat]['Type_of_Attraction'].unique().tolist() if sel_att_cat else []
        sel_att_type = st.selectbox("Choose Attraction Type:",
                                att_type_list,
                                index=None,
                                placeholder="Select...",
                                disabled=(sel_att_cat == None),
                                key="sel_att_type",
                                on_change=update_user_sel)

    with ops[5]:
        sel_crowd = st.selectbox("Choose Crowd level:",
                        ['LOW','MEDIUM','HIGH'],
                        index=None,
                        placeholder="Select...",
                        key="sel_crowd",
                        on_change=update_user_sel)

    with ops[6]:
        sel_temp = st.selectbox("Choose Temp level:",
                        ['LOW','MEDIUM','HIGH'],
                        index=None,
                        placeholder="Select...",
                        key="sel_temp",
                        on_change=update_user_sel)
    
    with ops[7]:
        locNL = pois['Location_Name'].unique().tolist()
        sel_locN = st.selectbox("Choose a Destination:",
                        locNL,
                        index=None,
                        placeholder="Select...",
                        key="sel_locN")
        if sel_locN != None: Dest_Forecastig_Data_Get()
with midR[2]:
    if st.session_state['sel_org'] != None and st.session_state['sel_Arv_dte'] != None and st.session_state['sel_locN'] != None:
        pltdata = dfs_comb[dfs_comb['Location_Name'] == st.session_state['sel_locN']]
        pltdata = pd.concat([pltdata,st.session_state['FC_sel_Dest']],axis='index')[['Date','Avg_Daily_Pedestrian_Count']]
        pltdata['Date'] = pltdata['Date'].apply(lambda x: pd.to_datetime(x))
        pltdata = pltdata.set_index('Date').resample('ME').mean().reset_index()
        pltdata = pltdata.rename(columns={'Avg_Daily_Pedestrian_Count':'Avg Monthly Crowd Count'})
        Tinfo = dfs_comb[['City','Country','Location_Name']].loc[dfs_comb['Location_Name'] == st.session_state['sel_locN']].drop_duplicates().reset_index()
        fig = px.line(
                pltdata,
                x='Date',
                y='Avg Monthly Crowd Count',
                title=f"{Tinfo['Location_Name'].loc[0]} — Monthly Trend ---- [{Tinfo['Country'].loc[0]}/{Tinfo['City'].loc[0]}]",
                markers=True
            )
        fig.add_vline(x=parser.parse('2025-09-30').timestamp()*1000, line_width=2, line_dash="dash", line_color="red", annotation_text="Forecast Start")
    else:
        fig = px.line(
                    title=f"Destination-Orgin-Time not Selected",
                    markers=True
                )
    fig.update_layout(title=dict(font=dict(size=24)), height=300, margin=dict(l=10,r=10,t=40,b=10))
    plot = st.plotly_chart(fig, use_container_width=True)

#* ---------------------------- ROW 3: TRANSLATOR & SUGGESTION & RECOMMENDATION
with lowR[1]:
    TransHeaderOps = st.columns([1]) + st.columns([1])
    with TransHeaderOps[0]:
        st.subheader("Ask Translator")
    user_input = ""
    with TransHeaderOps[1]:
        user_input = st.text_area("-", placeholder="Type Here", label_visibility='hidden')
        # res = client.translate_text(
        #     parent=parent,
        #     contents=[user_input],
        #     target_language_code='fr'
        # )
        
        # translator = Translator()
        # translated_text = translator.translate(text = user_input, dest='en').text
        # st.write(translated_text)
        # translated_text = res.translations[0].translated_text
        # st.write(translated_text)
st.markdown("""
    <style>
        .poi-recbox {
            background-color: rgba(131, 131, 131, 0.50);
            padding: 15px;
            border-radius: 15px;
            height: auto;
            font-size:25px;
        }
        .poi-statO {
            font-size:20px;
        }
        .poi-statI {
            font-size:18px;
        }
    </style>
    """, unsafe_allow_html=True)
with lowR[2]:
    st.subheader("Suggestions")
    if st.session_state['sel_org'] != None and st.session_state['sel_Arv_dte'] != None and st.session_state['sel_locN'] != None:
        # Reterving the Forecast at User Arival Time and Flight Path at the date
        FCArv = st.session_state['FC_sel_Dest'].loc[st.session_state['FC_sel_Dest']['Date'] == st.session_state['sel_Arv_dte']].reset_index(drop=True)
        FLArv = st.session_state['Flght_sel_Dest'].loc[st.session_state['Flght_sel_Dest']['apt_time_dt_ds'] == st.session_state['sel_Arv_dte']].reset_index(drop=True)

        FClow = st.session_state['FC_sel_Dest'].loc[st.session_state['FC_sel_Dest']['Avg_Daily_Pedestrian_Count'] < FCArv['Avg_Daily_Pedestrian_Count'].loc[0]]
        FLlow = st.session_state['Flght_sel_Dest'].loc[st.session_state['Flght_sel_Dest']['apt_time_dt_ds'].isin(FClow['Date'].to_list())].reset_index(drop=True)

        StateBuilder = []
        StateBuilder.append(f"""<p class='poi-statO'>Forecast Crowd: {int(FCArv['Avg_Daily_Pedestrian_Count'].loc[0])} people<br></p>""")
        if len(FLArv) > 0:
            OthFlArv = '<br>'.join([f'{tp['apt_name_dp']} -- {tp['apt_time_dt_dp']} --> {tp['apt_name_ds']} -- {tp['apt_time_dt_ds']}  >>> ${tp['price']}' for i,tp in FLArv.nsmallest(n=20, columns='price').iterrows()][:3])
            StateBuilder.append(
                f"""<p class='poi-statO'>Arvival Date Flight Paths <br> {OthFlArv}</p>"""
            )
        else:
            StateBuilder.append(
                """<p class='poi-statO'>No Flights Path For Arvival Date</p>"""
            )
        if len(FClow) > 0:
            OthFCLow = '<br>'.join([f'People: {int(tp['Avg_Daily_Pedestrian_Count'])} -- {tp['Date']}' for i,tp in FClow.nsmallest(n=20, columns='Avg_Daily_Pedestrian_Count').iterrows() if tp['Date'] > date.today()][:3]) 
            StateBuilder.append(
                f"""<p class='poi-statO'>Other Dates With Less Arvival Crowd Forecast<br> {OthFCLow}</p>"""
            ) 
        else:
            StateBuilder.append(
                """<p class='poi-statO'>No Other Dates Less than Arvival Date Crowd Forecast </p>"""
            )
        if len(FLlow) > 0:
            OthFllow = '<br>'.join([f'{tp['apt_name_dp']} -- {tp['apt_time_dt_dp']} --><br> {tp['apt_name_ds']} -- {tp['apt_time_dt_ds']} >>> ${tp['price']}' for i,tp in FLlow.nsmallest(n=20, columns='price').iterrows()][:3])
            StateBuilder.append(
                f"""<p class='poi-statO'>Other Dates Flight Paths <br> {OthFllow}</p>"""
            )
        else:
            StateBuilder.append(
                """<p class='poi-statO'>No Flights Path For Other Dates</p>\n"""
            )
        st.markdown(f"""
            <div class='poi-recbox'>
                    {''.join(StateBuilder)}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='poi-recbox'>
            </div>
            """, unsafe_allow_html=True)

with lowR[3]:
    st.subheader("ALt Destination")
    if st.session_state['sel_org'] != None and st.session_state['sel_Arv_dte'] != None and st.session_state['sel_locN'] != None:
        RCArv = st.session_state['RC_alt_Dest']
        RCFl = st.session_state['Flght_alt_Dest']  
        print(RCArv)
        print(RCFl)
        StateBuilder2 = []
        StateBuilder2.append(f"""<p class='poi-statO'>{RCArv['Location_Name']}, {RCArv['Country']}, {RCArv['City']} with past historical crowd numbers 
                            lower than current selected, one of them being {int(RCArv['Avg_Daily_Pedestrian_Count'])} people<br>You could consider traveling to here during {RCArv['Date'].month}/{RCArv["Date"].day}</p>""")
        st.markdown(f"""
            <div class='poi-recbox'>
                    {''.join(StateBuilder2)}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='poi-recbox'>
            </div>
            """, unsafe_allow_html=True)
            
# st.subheader("Monthly Crowd Trend Per Location")

# loc_ids = pois['Location_ID'].unique().tolist()
# pois_names = dfs_comb[['City','Country','Location_ID','Location_Name']].drop_duplicates().set_index('Location_ID')

# # 3 charts per row
# n_per_row = 3
# for i in range(0, len(loc_ids), n_per_row):
#     row_ids = loc_ids[i:i + n_per_row]
#     cols = st.columns(len(row_ids))

#     for col, loc_id in zip(cols, row_ids):
#         with col:
#             df_loc = dfs_comb[dfs_comb['Location_ID'] == loc_id].copy()
#             df_loc['Date'] = pd.to_datetime(df_loc['Date'])

#             df_month = (
#                 df_loc[['Date', 'Avg_Daily_Pedestrian_Count']]
#                 .set_index('Date')
#                 .resample('ME')
#                 .mean()
#                 .reset_index()
#             )

#             df_month = df_month.rename(columns={'Avg_Daily_Pedestrian_Count':'Avg Monthly Crowd Count'})
#             Tinfo = pois_names.loc[loc_id,['City','Country','Location_Name']]
#             fig = px.line(
#                 df_month,
#                 x='Date',
#                 y='Avg Monthly Crowd Count',
#                 title=f"{Tinfo['Location_Name']} — Monthly Trend ---- [{Tinfo['Country']}/{Tinfo['City']}]",
#                 markers=True
#             )
#             fig.update_layout(title=dict(font=dict(size=24)), height=300, margin=dict(l=10,r=10,t=40,b=10))

#             plot = st.plotly_chart(fig, use_container_width=True)
    
#--------------------------------------------------------------------------------------------

# with lowR[1]:
#     pass
    # with opsTrans[1]:
    #     st.subheader("Ask Transloator")
    
    # with opsTrans[2]:
        
    # with ops[2]:
    #     sel_ctry = st.selectbox("Choose a Country:", 
    #                             pois['Country'].unique().tolist(), 
    #                             index=None, 
    #                             key="sel_ctry",
    #                             placeholder="Select...",
    #                             on_change=reset_city_and_loc)
    # with ops[3]:
    #     city_list = pois[pois['Country'] == sel_ctry]['City'].unique().tolist() if sel_ctry else []
    #     sel_city = st.selectbox("Choose a City:", 
    #                             city_list, 
    #                             index=None, 
    #                             key = "sel_city",
    #                             placeholder="Select...",
    #                             disabled=(sel_ctry == None),
    #                             on_change=reset_loc)
    # with ops[4]:
    #     loc_list = pois[pois['City'] == sel_city]['Location_Name'].unique().tolist() if sel_city else []
    #     sel_loc = st.selectbox("Choose a Location:", 
    #                             loc_list, 
    #                             index=None, 
    #                             placeholder="Select...",
    #                             disabled=(sel_city == None))


    # sel_attract = st.selectbox("Choose Attraction level:",
    #                             ['LOW','MEDIUM','HIGH'],
    #                             index=None,
    #                             placeholder="Select...")
    # sel_hoilday = st.selectbox("Choose Hoilday option:",
    #                            ['YES','NO','BOTH'],
    #                            index=None,
    #                            placeholder="Select...")
    # sel_season = st.selectbox("Choose Travel Season:",
    #                           ['WINTER','SPRING','SUMMER','FALL'],
    #                           index=None,
    #                           placeholder="Select...")
        
    
# ---------------------------- ROW 2: MAP & RECOMMEND 
# with midR[2]:
#     mapRec = st.columns([2])  # Map
#     with mapRec[0]:
#         if sel_loc:
#             latlong = pois.loc[pois.index[pois['Location_Name'] == sel_loc].tolist()[0],['Latitude','Longitude']].tolist()
#             folium.Marker(
#                 location=latlong,
#                 popup=sel_loc,
#                 tooltip=sel_loc,
#                 icon=folium.Icon(color="blue", icon="info-sign")
#             ).add_to(m)
#             lat, lon = latlong
#             m.fit_bounds([[lat - 0.01, lon - 0.01], [lat + 0.01, lon + 0.01]])
#         folium_static(m, width=None)

# ---------------------------- ROW 2: Recommendation & Suggestions
# with midR[3]:
#     st.subheader("Any Flights")
    # st.markdown("""
    #         <style>
    #             .poi-recbox {
    #                 background-color: rgba(131, 131, 131, 0.50);
    #                 padding: 15px;
    #                 border-radius: 15px;
    #                 height: 300px;
    #             }
    #         </style>
    #         """, unsafe_allow_html=True)

    # st.markdown(f"""

    #     <div class='poi-recbox'>
    #         <h3>Recommendations</h3>   
    #     </div>
    #     """, unsafe_allow_html=True)
    
# ---------------------------- ROW 3: LOC EDA
# with lowR[1]:
#     FcCol1 = st.columns([1]) + st.columns([1]) # Top = Header, Bottom = Selected Forecasts

#     with FcCol1[0]:
#         st.subheader("Selected Forecasts")
#     with FcCol1[1]:
#         if sel_loc != None:
#             data = dfs_comb.loc[dfs_comb['Location_Name'] == sel_loc,['Date','Avg_Daily_Pedestrian_Count']]
#             fig1 = px.line(data,x='Date', y='Avg_Daily_Pedestrian_Count', title=f'{sel_loc} Crowd over Time')
#             st.plotly_chart(fig1, use_container_width=True,key=1)

# with lowR[2]:
#     FcCol2 = st.columns([1]) + st.columns([1])# Top = Header, Bottom = Recommendation Selected 
#     with FcCol2[0]:
#         st.subheader("Recommended Forecasts")
#     with FcCol2[1]:
#         if sel_loc != None:
#             fig2 = px.line(dfs_comb.loc[dfs_comb['Location_Name'] == sel_loc,['Date','Avg_Daily_Pedestrian_Count']],
#                            x='Date', y='Avg_Daily_Pedestrian_Count', title=f'{sel_loc} Crowd over Time')
#             st.plotly_chart(fig2, use_container_width=True,key=2)



#
# Create dynamic HTML items
# LocNs = [] 
# if sel_loc != None:
#     LocNs = pois[pois['Loc_Name'] == sel_loc,['Loc_Name']].tolist()

# items_html = "".join([
#     f"<div class='poi-item'>Suggested Crowd Forecast</div>"+
#     "<div class='poi-item'>Recommend Crowd Forecast</div>"
#     for name in [[] if sel_loc == None else [sel_loc]][0]
# ])

# st.markdown(f"""
#     <div class='poi-desbox'>
#         <h3>Destination Forecast</h3>
#         {items_html}
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("""
#     <style>
#         .poi-desbox {
#             background-color: rgba(131, 131, 131, 0.50);
#             padding: 15px;
#             border-radius: 15px;
#             height: 842px;
#             overflow-y: auto;
#         }

#         .poi-item {
#             background: #eee;
#             padding: 10px 15px;
#             border-radius: 10px;
#             margin-bottom: 8px;
#             font-size: 20px;
#             font-weight: bold;
#             color: rgb(0, 0, 255);
#             height: 300px;
#         }
#         .poi-item:hover {
#             background:#dcdcdc;
#             cursor:pointer;
#         }
#     </style>
#     """, unsafe_allow_html=True)


#
# # ----------------------------
# # ROW 1: TITLE
# # ----------------------------
# with rC[1]:
#     st.markdown("<h1 style='text-align:center; font-size:60px;'>Start Your Travel Journey</h1>", unsafe_allow_html=True)

# # ----------------------------
# # ROW 2: SEARCH & TRANSLATOR + MAP & RECOMMENDATION 
# # ----------------------------
# with rC[4]:
#     rC2 = st.columns([2]) # Top = Controls, Bottom = LLM
#     with rC2[0]:

#         poi_list = [
#             "CN Tower, Toronto",
#             "Harbourfront Centre, Toronto",
#             "Royal Ontario Museum, Toronto",
#             "Ripley's Aquarium, Toronto",
#             "Distillery District, Toronto",
#             "Casa Loma, Toronto"
#         ]

#         st.markdown(
#             """<div style="background-color:rgba(255,255,255,0.8);padding: 20px; border-radius: 10px;">
#                     <h3>Search a Point of Interest</h3>
#             """,unsafe_allow_html=True)
        
#         selected_location = st.selectbox("Choose a Location:", poi_list, index=None, placeholder="Select...")

#         st.markdown("""</div>""",unsafe_allow_html=True)

#         # st.markdown(
#         #     """
#         #     <div style="background-color:rgba(255,255,255,0.8);padding: 20px; border-radius: 10px;">
#         #         <h3>Search a Point of Interest</h3>
#         #         <p>This is content within a custom HTML div.</p>
#         #     </div>
#         #     """,
#         #     unsafe_allow_html=True
#         # )

        

#         # AI Translator Feature
#         st.subheader("Your Suggestions")
#         mode = st.radio("Input:", ["Text", "Voice"], horizontal=True)
#         user_input = ""

#         if mode == "Text":
#             user_input = st.text_area("Enter text in any language:", placeholder="Type Here")
#         else:
#             pass

# with rC[5]: # MAP * RECOMMONDATIONS LIVE 
#     rC3 = st.columns([2]) + st.columns([2]) # Top = Map, Bottom = Flight Recommendations 
#     with rC3[0]:
#         m = folium.Map(location=[43.65107, -79.347015], zoom_start=14)
#         poi_coords = {
#             "CN Tower, Toronto": [43.6426, -79.3871],
#             "Harbourfront Centre, Toronto": [43.6387, -79.3823],
#             "Royal Ontario Museum, Toronto": [43.6677, -79.3948],
#             "Ripley's Aquarium, Toronto": [43.6424, -79.3860],
#             "Distillery District, Toronto": [43.6500, -79.3590],
#             "Casa Loma, Toronto": [43.6780, -79.4094]
#         }
#         if selected_location:
#             folium.Marker(
#                 location=poi_coords[selected_location],
#                 popup=selected_location,
#                 tooltip=selected_location,
#                 icon=folium.Icon(color="blue", icon="info-sign")
#             ).add_to(m)
#             lat, lon = poi_coords[selected_location]
#             m.fit_bounds([[lat - 0.01, lon - 0.01], [lat + 0.01, lon + 0.01]])
#         folium_static(m, width=None)
#         st.markdown("</div>", unsafe_allow_html=True)
        
#     # ----------------------------
#     # ROW 3: SCROLLABLE TILES
#     # ----------------------------
#     with rC3[1]:
#         st.markdown(
#             """
#             <h3>Featured Attractions</h3>
#             <div style='background-color:rgba(255,255,255,0.8);padding:15px;border-radius:15px;height:180px;overflow-x:auto;white-space:nowrap;'>
#             <div style='display:flex;gap:20px;'>
#             <div style='background:#eee;padding:10px 20px;border-radius:10px;'>CN Tower</div>
#             <div style='background:#eee;padding:10px 20px;border-radius:10px;'>Royal Ontario Museum</div>
#             <div style='background:#eee;padding:10px 20px;border-radius:10px;'>Casa Loma</div>
#             <div style='background:#eee;padding:10px 20px;border-radius:10px;'>Harbourfront Centre</div>
#             <div style='background:#eee;padding:10px 20px;border-radius:10px;'>Distillery District</div>
#             <div style='background:#eee;padding:10px 20px;border-radius:10px;'>Ripley's Aquarium</div>
#             <div style='background:#eee;padding:10px 20px;border-radius:10px;'>Nathan Phillips Square</div>
#             <div style='background:#eee;padding:10px 20px;border-radius:10px;'>Toronto Islands</div>
#             </div></div>
#             """, unsafe_allow_html=True
#         )

# with rC[6]:
#     rC4 = st.columns([1]) # Loc EDA
#     with rC4[0]:
#         st.markdown(
#             """
#             <div style='background-color:rgba(255,255,255,0.8);padding:15px;border-radius:15px;height:180px;overflow-x:auto;white-space:nowrap;'>
#             <h3>Where Location EDA goes</h3>
#             <div style='display:flex;gap:20px;'>
#             </div></div>
#             """, unsafe_allow_html=True)