import streamlit as st
import pandas as pd
import os
from datetime import date,timedelta
from dateutil import parser
import plotly.express as px

from openai import OpenAI
#Set key from secrets 
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Used for Getting forecasting data from selected location 
from Dest_Forecasting_Data_Get import Dest_Forecastig_Data_Get 

# Function handles itinerary changes 
from poisUpdate import poisUpdate

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
dfs_comb['Date'] = dfs_comb['Date'].apply(lambda x: parser.parse(x).date())#datetime.date(YYYY, MM, DD)


folder2 = "./Dataset/Crowd/Flight/flight_paths.csv"
flights = pd.read_csv(folder2)
flights['apt_time_dt_ds'] = flights['apt_time_dt_ds'].apply(lambda x: parser.parse(x).date())#datetime.date(YYYY, MM, DD)
flights['apt_time_dt_dp'] = flights['apt_time_dt_dp'].apply(lambda x: parser.parse(x).date())

#^ SESSION RELATED-----------------------------
# --- OPENAI DEF MODEL ---
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
if "messages" not in st.session_state:# Initialize chat history
    st.session_state.messages = []
# --- SUGGESTIONS & ALT RECOMMENDATIONS TEXT ---
if "suggest" not in st.session_state:
    st.session_state["suggest"] = []
if "recommend" not in st.session_state:
    st.session_state["recommend"] = []
# --- HOUSING FORECAST & RECOMMEND & FLIGHT DATA ---
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
def update_user_sel():#Updating user_sel list to reflect new setting changes 
    st.session_state['user_sel'][0] = st.session_state['sel_org']
    st.session_state['user_sel'][1] = st.session_state['sel_Arv_dte']
    st.session_state['user_sel'][2] = st.session_state['sel_att_cat']
    if st.session_state['sel_att_cat'] == None:
        st.session_state['sel_att_type'] = None
    st.session_state['user_sel'][3] = st.session_state['sel_att_type']
    st.session_state['user_sel'][4] = st.session_state['sel_crowd']
    st.session_state['user_sel'][5] = st.session_state['sel_temp']
    # poisUpdate()

pois = poisUpdate(dfs_comb) # used by the destination selection

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
        sel_org = st.selectbox("Choose an Orgin:",
                            flights['City_dp'].unique().tolist(),
                            index=None,
                            placeholder="Select...",
                            key="sel_org",
                            on_change=update_user_sel)

    with ops[2]:
        nextday = date.today() + timedelta(days=1)
        MaxD = nextday + timedelta(days=180)
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
        if sel_locN != None: Dest_Forecastig_Data_Get(dfs_comb,flights)

with midR[2]:
    # Update figure with new data if Orgin,Avr Time,Dest have been selected
    if st.session_state['sel_org'] != None and st.session_state['sel_Arv_dte'] != None and st.session_state['sel_locN'] != None:

        # Get Only the selected location, attach the storeded FC session data to historical data
        pltdata = dfs_comb[dfs_comb['Location_Name'] == st.session_state['sel_locN']]
        pltdata = pd.concat([pltdata,st.session_state['FC_sel_Dest']],axis='index')[['Date','Avg_Daily_Pedestrian_Count']]
        pltdata['Date'] = pltdata['Date'].apply(lambda x: pd.to_datetime(x.strftime('%Y-%m-%d')))

        # Resample for monthly from daily, provides a better visual of the older + new data
        pltdata = pltdata.set_index('Date').resample('ME').mean().reset_index()
        pltdata = pltdata.rename(columns={
            'Avg_Daily_Pedestrian_Count':'Avg Monthly Crowd Count',
            })
        Tinfo = dfs_comb[['City','Country','Location_Name']].loc[dfs_comb['Location_Name'] == st.session_state['sel_locN']].drop_duplicates().reset_index()
        fig = px.line(
                pltdata,
                x='Date',
                y='Avg Monthly Crowd Count',
                title=f"{Tinfo['Location_Name'].loc[0]} — Monthly Trend ---- [{Tinfo['Country'].loc[0]}/{Tinfo['City'].loc[0]}]",
                markers=True
            )
        
        # Adding Forecast vertical line 
        fig.add_vline(x=parser.parse('2025-09-30').timestamp()*1000, line_width=2, line_dash="dash", line_color="red", annotation_text="Forecast Start", annotation_position="bottom right")
    
    else: # If user deselectes Orgin,Arv Time,Dest, then reset graph. 
        fig = px.line(
                    title=f"Destination-Orgin-Time not Selected",
                    markers=True
                )
    
    fig.update_layout(title=dict(font=dict(size=24)),
                      font=dict(size=24),
                      xaxis=dict(title_font_size=20,tickfont=dict(size=18)),
                      yaxis=dict(title_font_size=20,tickfont=dict(size=18)),
                      height=300, 
                      margin=dict(l=10,r=10,t=40,b=10))
    plot = st.plotly_chart(fig, use_container_width=True)

#* ---------------------------- ROW 3: TRANSLATOR & SUGGESTION & RECOMMENDATION
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
# Below are the AI Features for Sugesting and Recommending 
with lowR[2]: # Sueggestions
    st.subheader("Suggestions")
    if st.session_state['sel_org'] != None and st.session_state['sel_Arv_dte'] != None and st.session_state['sel_locN'] != None:
        # Reterving the Forecast at User Arival Time and Flight Path at the date
        FCArv = st.session_state['FC_sel_Dest'].loc[st.session_state['FC_sel_Dest']['Date'] == st.session_state['sel_Arv_dte']].reset_index(drop=True)
        FLArv = st.session_state['Flght_sel_Dest'].loc[st.session_state['Flght_sel_Dest']['apt_time_dt_ds'] == st.session_state['sel_Arv_dte']].reset_index(drop=True)

        FClow = st.session_state['FC_sel_Dest'].loc[st.session_state['FC_sel_Dest']['Avg_Daily_Pedestrian_Count'] < FCArv['Avg_Daily_Pedestrian_Count'].loc[0]]
        FLlow = st.session_state['Flght_sel_Dest'].loc[st.session_state['Flght_sel_Dest']['apt_time_dt_ds'].isin(FClow['Date'].to_list())].reset_index(drop=True)

        StateBuilder = [] # Logic Statement Builder

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
        
        st.session_state['suggest'] = StateBuilder # Save in session for OpenAI to translate to user

    else: # Empty div when one of the itinerary selections is deselected
        st.markdown(f"""
            <div class='poi-recbox'>
            </div> 
            """, unsafe_allow_html=True)
        
        st.session_state['suggest'] = [] # Reset for new session info to be saved when user deselects itinerary

with lowR[3]:# Recmmmendation
    st.subheader("ALt Destination")
    if st.session_state['sel_org'] != None and st.session_state['sel_Arv_dte'] != None and st.session_state['sel_locN'] != None:
        RCArv = st.session_state['RC_alt_Dest']
        RCFl = st.session_state['Flght_alt_Dest']  
 
        StateBuilder2 = [] # Logic Satament Builder

        StateBuilder2.append(f"""<p class='poi-statO'>{RCArv['Location_Name']}, {RCArv['Country']}, {RCArv['City']} with past historical crowd numbers 
                            lower than current selected, one of them being {int(RCArv['Avg_Daily_Pedestrian_Count'])} people<br>You could consider traveling to here during {RCArv['Date'].month}/{RCArv["Date"].day}</p>""")
       
        st.markdown(f"""
            <div class='poi-recbox'>
                    {''.join(StateBuilder2)}
            </div>
            """, unsafe_allow_html=True)
        
        st.session_state['recommend'] = StateBuilder2 # Save in session for OpenAI to translate to user

    else: # Empty div when one of the itinerary selections is deselected
        st.markdown(f"""
            <div class='poi-recbox'>
            </div>
            """, unsafe_allow_html=True)
        
        st.session_state['recommend'] = [] # Reset for new session info to be saved when user deselects itinerary


with lowR[1]:
    st.subheader("Language Translator")
    user_input = ""
    language_list = ["English", "French", "Spanish", "German", "Tamil", "Hindi", "Chinese"]
    user_input = st.text_area("-", placeholder=f"Type what language to tranlate to\n Languages like: {','.join(language_list)},.etc", label_visibility='hidden')
    if st.session_state['sel_org'] != None and st.session_state['sel_Arv_dte'] != None and st.session_state['sel_locN'] != None:
        if user_input != "":
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"user","content":f"Translate just the non html of this "+
                                            f"{''.join(st.session_state['suggest'])}{''.join(st.session_state['recommend'])} into {user_input}, "+
                                            "and output only the translated text following the html format"}
                ]
            )

            st.markdown(f"""
                <div class='poi-recbox'>
                        {resp.choices[0].message.content}
                </div>
                """, unsafe_allow_html=True)
    else: # Empty div when one of the itinerary selections is deselected
        st.markdown(f"""
            <div class='poi-recbox'>
            </div>
            """, unsafe_allow_html=True)