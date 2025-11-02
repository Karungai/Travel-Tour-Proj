# ----------------------------
# STREAMLIT APP
# ----------------------------

import numpy as np
import pickle as pk
import pandas as pd
import streamlit as st 

import folium as flm
from streamlit_folium import folium_static

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="Smart Tourism Explorer",
    page_icon="🌍",
    layout="wide",
)

# ----------------------------
# BACKGROUND STYLING
# ----------------------------
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://images.unsplash.com/photo-1517760444937-f6397edcbbcd');
background-size: cover;
background-repeat: no-repeat;
background-attachment: fixed;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stToolbar"] {right: 2rem;}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ----------------------------
# LAYOUT STRUCTURE
# ----------------------------
rC = st.columns([1,5,1]) + st.columns([1,5,1]) + st.columns([1,5,1])

with rC[1]:
  st.markdown(
    "<h1 style='text-align: center; font-size: 60px;'>Start Your Travel Journey</h1>"
    , unsafe_allow_html=True)

with rC[4]:
  col = st.columns([4,2],gap='small')
  with col[0]:
    # ----------------------------
    # PLACEHOLDER ABOVE MAP
    # ----------------------------
    st.markdown(
      "<div  style='height:100px; background-color: rgba(255,255,255,0.7); border-radius: 10px;'>"
      "</div>", unsafe_allow_html=True)

    # ----------------------------
    # SEARCH BAR CONFIGURATION
    # ----------------------------
    poi_list = [
      "CN Tower, Toronto",
      "Harbourfront Centre, Toronto",
      "Royal Ontario Museum, Toronto",
      "Ripley's Aquarium, Toronto",
      "Distillery District, Toronto",
      "Casa Loma, Toronto"
    ]
    selected_location = st.selectbox(
      "Search for a Point of Interest:",
      options=poi_list,
      index=None,
      placeholder="Type or select a location...",
    )

    # ----------------------------
    # MAP SETUP
    # ----------------------------
    m = flm.Map(location=[43.65107, -79.347015], zoom_start=14, max_zoom=17, min_zoom=12)

    poi_coords = {
        "CN Tower, Toronto": [43.6426, -79.3871],
        "Harbourfront Centre, Toronto": [43.6387, -79.3823],
        "Royal Ontario Museum, Toronto": [43.6677, -79.3948],
        "Ripley's Aquarium, Toronto": [43.6424, -79.3860],
        "Distillery District, Toronto": [43.6500, -79.3590],
        "Casa Loma, Toronto": [43.6780, -79.4094]
    }

    if selected_location:
      flm.Marker(
          location=poi_coords[selected_location],
          popup=selected_location,
          tooltip=selected_location,
          icon=flm.Icon(color='blue', icon='info-sign')
      ).add_to(m)

      # Limit map bounds (restrict movement area)
      lat, lon = poi_coords[selected_location]
      m.fit_bounds([[lat - 0.01, lon - 0.01], [lat + 0.01, lon + 0.01]])
    
    # Display map
    map_output = folium_static(m,width=None)

  with col[1]:
    st.markdown(
      "<div style='height:700px; background-color: rgba(255,255,255,0.7); border-radius: 10px;'>"
      "</div>", unsafe_allow_html=True)
    
with rC[7]:
  st.markdown(
    "<div style='height:200px; background-color: rgba(255,255,255,0.7); border-radius: 10px;'>"
    "</div>", unsafe_allow_html=True)

# pickle_in = open("./classifier.pkl","rb")
# classifier=pickle.load(pickle_in)
