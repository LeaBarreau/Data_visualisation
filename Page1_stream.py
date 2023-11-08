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
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.tokenize import word_tokenize
import random
import seaborn as sns
from folium.plugins import HeatMap
import geoviews as gv
import geopandas as gpd
from pandas import merge

# Définissez la largeur de la page Streamlit
st.set_page_config(layout="wide")

st.title("La sécurité routière, même à vélo !")
st.write("Les données expoitées courent depuis 2005")

# Utilisez pd.read_excel() pour lire le fichier Excel dans un DataFrame
@st.cache_data
def import_data():
    data = pandas.read_excel(r"accidentsVelo.xlsx", decimal=",")  
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

datamoy = data.groupby(['grav'])['Num_Acc'].count().reset_index()
datamoy = datamoy.rename(columns={'Num_Acc': 'nb_accidents'})

st.subheader("Quelques statistiques")
T1, T2 = st.columns(2)
with T1:
    st.write("74 689 accidents de vélo recensés mais combien sont restés indemnes ?")
    C7, C8, C9, C10 = st.columns(4)
    with C7 :
        st.image("indemne.png", caption=None, width=80, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(datamoy.loc[datamoy['grav'] == 'Indemne', 'nb_accidents'].values[0])
    with C8 :
        st.image("Blésser léger.png", caption=None, width=80, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(datamoy.loc[datamoy['grav'] == 'Blessé léger', 'nb_accidents'].values[0])
    with C9 :
        st.image("Blésser hospitalisés.png", caption=None, width=80, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(datamoy.loc[datamoy['grav'] == 'Blessé hospitalisé', 'nb_accidents'].values[0])
    with C10 :
        st.image("mort.png", caption=None, width=80, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(datamoy.loc[datamoy['grav'] == 'Tué', 'nb_accidents'].values[0])
    st.write("")
with T2 :
    st.write("Selon les statistiques, quel jour et à quelle heure y-t-il eu le plus d'accidents de vélos ?")
    C2, C3, C4, C5 = st.columns(4)
    with C2 :
        st.image("année.png", caption=None, width=80, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(data['an'].mode().values[0])
    with C3 :
        st.image("mois.png", caption=None, width=80, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(data['mois'].mode().values[0])
    with C4 :
        st.image("jours.png", caption=None, width=80, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(data['jour'].mode().values[0])
    with C5 :
        st.image("heure.png", caption=None, width=80, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(data['hrmn'].mode().values[0])
    st.write("")

# Titre intermédiaire
st.subheader("L'année 2018, un tournant")

# Contenu de la première section
st.write("Ecrire du contenu")

Q1,Q2 = st.columns(2)

with Q1:
    dfQ1 = data.groupby(by = ['an','grav'])['Num_Acc'].count().reset_index()
    dfQ1 = dfQ1.rename(columns={'Num_Acc': 'nb_accidents'})
    figQ1 = px.bar(dfQ1,
                            x='an',
                            y='nb_accidents',
                            color='grav',
                            title='Nombre d\'accidents en fonction de l\'année')
    figQ1.update_layout(plot_bgcolor = "rgba(0,0,0,0)",
                                    xaxis =(dict(showgrid = False,
                                                    title='Année')),
                                    yaxis =(dict(showgrid = False,
                                                    title='Nombre d\'accidents')))
    st.plotly_chart(figQ1,use_container_width=True, figsize=(10, 6))
st.text("    ")

with Q2:
    dfQ2 = data.groupby(by = ['grav'])['Num_Acc'].count().reset_index()
    dfQ2 = dfQ2.rename(columns={'Num_Acc': 'nb_accidents'})
    figQ2 = px.pie(dfQ2,names='grav',values='nb_accidents',title='Représentation du nombre d\'accients par gravité')
    figQ2.update_layout(plot_bgcolor = "rgba(0,0,0,0)")
    st.plotly_chart(figQ2,use_container_width=True)

st.subheader("Visualisation cartographique")

# Définissez les limites géographiques pour la France métropolitaine
# Remarque : Les valeurs de latitude et de longitude sont approximatives et doivent être ajustées selon vos besoins.
min_lat, max_lat = 41.2, 51.1  # Limites approximatives pour la latitude de la France métropolitaine
min_long, max_long = -5.142, 9.561  # Limites approximatives pour la longitude de la France métropolitaine

# Filtrer les données pour supprimer les points en dehors de la France métropolitaine
data_carte = data.loc[data['an'] >= 2019]
data_carte['size'] = data_carte['grav'].apply(lambda x: 20000 if x == 'Tué' else (10000 if x == 'Blessé hospitalisé' else (5000 if x == 'Blessé léger' else 2500)))
data_carte['couleur'] = data_carte['grav'].apply(lambda x: '#8b0000' if x == 'Tué' else ('#b22222' if x == 'Blessé hospitalisé' else ('#dc143c' if x == 'Blessé léger' else '#f08080')))
data_carte = data_carte.dropna(subset=['lat', 'long'])
data_carte = data_carte[(data_carte['lat'] >= min_lat) & (data_carte['lat'] <= max_lat) & (data_carte['long'] >= min_long) & (data_carte['long'] <= max_long)]

# Créez la carte avec des marqueurs colorés en fonction de la gravité et centrez-la sur la moyenne des coordonnées
st.map(data_carte, latitude='lat', longitude='long', size='size', color = 'couleur', zoom=4.5, use_container_width=True)

st.subheader("Nuage de mots")
#Word cloud
data["Full"] = (
    data["lum"].astype(str) + " " +
    data["mois"].astype(str) + " " +
    data["col"].astype(str) + " " +
    data["obsm"].astype(str) + " " +
    data["atm"].astype(str) + " " +
    data["equipement"].astype(str) + " " +
    data["sexe"].astype(str)
)
# Créez une fonction pour afficher le nuage de mots à partir d'une colonne
def show_wordcloud_from_column(data, column_name):
    # Divisez le texte en mots en utilisant l'espace comme séparateur
    texte_original = ' '.join(data[column_name])
    mots = texte_original.split()
    # Mélangez l'ordre des mots
    random.shuffle(mots)
    # Rejoignez les mots mélangés pour former une chaîne de caractères mélangée
    texte_melange = ' '.join(mots)
    exclure_mots = ['avec', 'd', 'du', 'de', 'la', 'des', 'le', 'et', 'est', 'elle', 'une', 'en', 'que', 'aux', 'qui', 'ces', 'les', 'dans', 'sur', 'l', 'un', 'pour', 'par', 'il', 'ou', 'à', 'ce', 'a', 'sont', 'cas', 'plus', 'leur', 'se', 's', 'vous', 'au', 'c', 'aussi', 'toutes', 'autre', 'comme', "Non", "Nan", "Null"]
    # Générez le nuage de mots à partir des mots uniques
    wordcloud = WordCloud(width=800, height=400, background_color='black', stopwords = exclure_mots, max_words=100).generate(texte_melange)

    # Créez un conteneur pour afficher le nuage de mots avec une largeur personnalisée
    container = st.container()
    with container:
        st.image(wordcloud.to_image(), caption="Nuage de mots")

# Sélectionnez une colonne à partir de laquelle générer le nuage de mots
selected_column = "Full"

# Affichez le nuage de mots pour la colonne sélectionnée
show_wordcloud_from_column(data, selected_column)


#HEAT MAP
sf = gpd.read_file('france-geojson/departements-version-simplifiee.geojson')
jf = sf.merge(data, left_on='code', right_on='dep', suffixes=('','_y'))
regions = gv.Polygons(jf, vdims=['nom', 't_total', 'h_total', 'f_total'])
regions.opts(width=600, height=600, toolbar='above', color=dim('t_total')/1e6, 
             colorbar=True, tools=['hover'], aspect='equal')
