import pandas 
import folium
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import plotly.graph_objects as go
import geopandas as gpd
from streamlit_folium import folium_static

# Définissez la largeur de la page Streamlit
st.set_page_config(layout="wide")

st.title("La sécurité routière, même à vélo !")
st.write("Les données expoitées courent depuis 2005")

# Utilisez pd.read_excel() pour lire le fichier Excel dans un DataFrame
@st.cache_data
def import_data():
    data = pandas.read_excel(r"accidentsVelo.xlsx", decimal=",")  
    data = data.loc[data['an'] >= 2015]
    # Créer une correspondance département - région
    departement_region = {
        1: 'Auvergne-Rhône-Alpes',
        2: 'Hauts-de-France',
        3: 'Auvergne-Rhône-Alpes',
        4: 'Provence-Alpes-Côte d\'Azur',
        5: 'Provence-Alpes-Côte d\'Azur',
        6: 'Provence-Alpes-Côte d\'Azur',
        7: 'Auvergne-Rhône-Alpes',
        8: 'Grand Est',
        9: 'Occitanie',
        10: 'Grand Est',
        11: 'Occitanie',
        12: 'Occitanie',
        13: 'Provence-Alpes-Côte d\'Azur',
        14: 'Normandie',
        15: 'Auvergne-Rhône-Alpes',
        16: 'Nouvelle-Aquitaine',
        17: 'Nouvelle-Aquitaine',
        18: 'Centre-Val de Loire',
        19: 'Nouvelle-Aquitaine',
        21: 'Bourgogne-Franche-Comté',
        22: 'Bretagne',
        23: 'Nouvelle-Aquitaine',
        24: 'Nouvelle-Aquitaine',
        25: 'Bourgogne-Franche-Comté',
        26: 'Auvergne-Rhône-Alpes',
        27: 'Normandie',
        28: 'Centre-Val de Loire',
        29: 'Bretagne',
        '2A': 'Corse',
        '2B': 'Corse',
        30: 'Occitanie',
        31: 'Occitanie',
        32: 'Occitanie',
        33: 'Nouvelle-Aquitaine',
        34: 'Occitanie',
        35: 'Bretagne',
        36: 'Centre-Val de Loire',
        37: 'Centre-Val de Loire',
        38: 'Auvergne-Rhône-Alpes',
        39: 'Bourgogne-Franche-Comté',
        40: 'Nouvelle-Aquitaine',
        41: 'Centre-Val de Loire',
        42: 'Auvergne-Rhône-Alpes',
        43: 'Auvergne-Rhône-Alpes',
        44: 'Pays de la Loire',
        45: 'Centre-Val de Loire',
        46: 'Occitanie',
        47: 'Nouvelle-Aquitaine',
        48: 'Occitanie',
        49: 'Pays de la Loire',
        50: 'Normandie',
        51: 'Grand Est',
        52: 'Grand Est',
        53: 'Pays de la Loire',
        54: 'Grand Est',
        55: 'Grand Est',
        56: 'Bretagne',
        57: 'Grand Est',
        58: 'Bourgogne-Franche-Comté',
        59: 'Hauts-de-France',
        60: 'Hauts-de-France',
        61: 'Normandie',
        62: 'Hauts-de-France',
        63: 'Auvergne-Rhône-Alpes',
        64: 'Nouvelle-Aquitaine',
        65: 'Occitanie',
        66: 'Occitanie',
        67: 'Grand Est',
        68: 'Grand Est',
        69: 'Auvergne-Rhône-Alpes',
        70: 'Bourgogne-Franche-Comté',
        71: 'Bourgogne-Franche-Comté',
        72: 'Pays de la Loire',
        73: 'Auvergne-Rhône-Alpes',
        74: 'Auvergne-Rhône-Alpes',
        75: 'Île-de-France',
        76: 'Normandie',
        77: 'Île-de-France',
        78: 'Île-de-France',
        79: 'Nouvelle-Aquitaine',
        80: 'Hauts-de-France',
        81: 'Occitanie',
        82: 'Occitanie',
        83: 'Provence-Alpes-Côte d\'Azur',
        84: 'Provence-Alpes-Côte d\'Azur',
        85: 'Pays de la Loire',
        86: 'Nouvelle-Aquitaine',
        87: 'Nouvelle-Aquitaine',
        88: 'Grand Est',
        89: 'Bourgogne-Franche-Comté',
        90: 'Bourgogne-Franche-Comté',
        91: 'Île-de-France',
        92: 'Île-de-France',
        93: 'Île-de-France',
        94: 'Île-de-France',
        95: 'Île-de-France',
        971: 'Guadeloupe',
        972: 'Martinique',
        973: 'Guyane',
        974: 'La Réunion',
        976: 'Mayotte'
    }
    data['region'] = data['dep'].map(departement_region)
    data = data.dropna(subset=['region'])
    return data

data = import_data()
