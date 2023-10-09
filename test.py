import pandas 
import folium
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
from streamlit_folium import folium_static
from wordcloud import WordCloud
import plotly.graph_objects as go
import geopandas as gpd
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

# Définissez la largeur de la page Streamlit
st.set_page_config(layout="wide")

st.title("Etude des accidents de vélos depuis 2005")
st.write("Bienvenue !")

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

# Créez un bouton de navigation pour chaque page
page = st.sidebar.selectbox("Choisissez une page", ["Statistiques descriptives", "Graphiques interactifs", "Machine Learning"])

if page == "Statistiques descriptives":
    st.title("page accueil")

elif page == "Graphiques interactifs":
    with st.sidebar:
            region_filter = st.selectbox(label='Sélectionnez une région',
                                   options=data['region'].unique(),
                                   index=0) 

    df_filtre = data.query('region == @region_filter')

    Q1,Q2 = st.columns(2)

    with Q1:
        df1 = df_filtre.groupby(by = ['an','grav'])['Num_Acc'].count().reset_index()
        df1 = df1.rename(columns={'Num_Acc': 'nb_accidents'})
        fig1 = px.bar(df1,
                                x='an',
                                y='nb_accidents',
                                color='grav',
                                title='Nombre d\'accidents par années')
        fig1.update_layout(title = {'x' : 0.5},
                                        plot_bgcolor = "rgba(0,0,0,0)",
                                        xaxis =(dict(showgrid = False)),
                                        yaxis =(dict(showgrid = False)))
        st.plotly_chart(fig1,use_container_width=True, figsize=(10, 6))
    st.text("    ")

    with Q2:
        df2 = df_filtre.groupby(by = ['grav'])['Num_Acc'].count().reset_index()
        df2 = df2.rename(columns={'Num_Acc': 'nb_accidents'})
        fig2 = px.pie(df2,names='grav',values='nb_accidents',title='Type d\'accients')
        fig2.update_layout(title = {'x':0.5}, plot_bgcolor = "rgba(0,0,0,0)")
        st.plotly_chart(fig2,use_container_width=True)

    ##NB D'ACCIDENTS SELON L'AGE ET LE TRAJET
    df3 = df_filtre.loc[df_filtre['age']<=100]
    df3 = df3.groupby(by = ['age','trajet'])['Num_Acc'].count().reset_index()
    df3 = df3.rename(columns={'Num_Acc': 'nb_accidents'})
    fig3 = px.line(df3,x='age',y='nb_accidents',color='trajet',title='Nombre d\'accidents par rapport à l\'âge')
    fig3.update_layout(title = {'x':0.5}, plot_bgcolor = "rgba(0,0,0,0)")
    st.plotly_chart(fig3,use_container_width=True)

    Q3,Q4 = st.columns([1, 2])

    with Q3:
        df6 = df_filtre.groupby(by = ['trajet','grav'])['Num_Acc'].count().reset_index()
        df6 = df6.rename(columns={'Num_Acc': 'nb_accidents'})
        fig6 = px.bar(df6,
                                x='trajet',
                                y='nb_accidents',
                                color='grav',
                                title='Nombre d\'accidents par années')
        fig6.update_layout(title = {'x' : 0.5},
                                        plot_bgcolor = "rgba(0,0,0,0)",
                                        xaxis =(dict(tickangle=45,  # Angle d'inclinaison
                                                    title='trajet',  # Titre de l'axe des x
                                                    showgrid=False)),
                                        yaxis =(dict(showgrid = False)))
        st.plotly_chart(fig6,use_container_width=True, figsize=(10, 6))

    with Q4:
        df7 = df_filtre.loc[df_filtre['age']<=100]
        df7 = df7.groupby(by = ['age','grav'])['Num_Acc'].count().reset_index()
        df7 = df7.rename(columns={'Num_Acc': 'nb_accidents'})
        fig7 = px.line(df7,x='age',y='nb_accidents',color='grav',title='Nombre d\'accidents par rapport à l\'âge et la gravité')
        fig7.update_layout(title = {'x':0.5}, plot_bgcolor = "rgba(0,0,0,0)")
        st.plotly_chart(fig7,use_container_width=True)

    ##DIAGRAMME DE FLUX   
    df5bis = df_filtre
    # Créer un dictionnaire de mappage
    group_mapping = {
        'Normale': 'Normale',
        'Neige - grêle': 'Neige - grêle',
        'Temps éblouissant': 'Conditions dégradées',
        'Vent fort - tempête': 'Conditions dégradées',
        'Pluie légère': 'Conditions dégradées',
        'Pluie forte': 'Conditions dégradées',
        'Brouillard - fumée': 'Conditions dégradées',
        'Temps couvert': 'Conditions dégradées',
        'Autre': 'Conditions dégradées'     
    }
    df5bis['atm'] = df5bis['atm'].replace(group_mapping)
    df5 = df5bis.groupby(by=['atm', 'grav'])['Num_Acc'].count().reset_index()
    df5 = df5.rename(columns={'Num_Acc': 'nb_accidents'})
    # Créez le diagramme Sankey
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=['Conditions dégradées', 'Neige - grêle', 'Normale', "Blessés hospitalisés", "Blessés légers", "Indemnes", "Tués"]
        ),
        link=dict(
            source=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            target=[3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6],
            value=df5["nb_accidents"]
        )
    ))
    fig.update_layout(title="Diagramme de flux")
    # Affichez le diagramme dans Streamlit
    st.plotly_chart(fig)

    #SAISONNALITE DES ACCIDENTS PAR GRAVITE
    df4 = df_filtre.groupby(by = ['mois','grav'])['Num_Acc'].count().reset_index()
    df4 = df4.rename(columns={'Num_Acc': 'nb_accidents'})
    # Liste des mois dans l'ordre de l'arrivée
    ordre_des_mois = ["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre", "novembre", "decembre"]
    df4.columns = ['mois', 'grav', 'nb_accidents']
    df4['mois'] = pandas.Categorical(df4['mois'], categories=ordre_des_mois, ordered=True)
    df4 = df4.sort_values(by='mois')
    fig4 = px.line(df4,x='mois',y='nb_accidents',color='grav',title='Saisonnalité des accidents par rapport à la gravité')
    fig4.update_layout(title = {'x':0.5}, plot_bgcolor = "rgba(0,0,0,0)")
    st.plotly_chart(fig4,use_container_width=True)

    ##CARTE
    # Supprimez les lignes avec des valeurs NaN dans les colonnes 'lat' et 'long'
    data_carte = df_filtre.dropna(subset=['lat', 'long'])
    # Créez un menu déroulant pour sélectionner le mois
    selected_month = st.selectbox("Sélectionnez un mois", ["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre", "novembre", "decembre"])
    # Filtrez les données en fonction de l'année et du mois sélectionnés
    data_filtered = data_carte[(data_carte['mois'] == selected_month)]
    # Créez une carte centrée sur la France
    # Créez une carte Folium centrée sur la région
    region_selected_data = gdf_regions[gdf_regions['Nom'] == region_filter]
    m = folium.Map(location=[data_filtered['lat'].values[0], data_filtered['long'].values[0]], zoom_start=7)
    # Ajoutez le polygone de la région à la carte
    folium.GeoJson(data_filtered).add_to(m)
    #m = folium.Map(location=[data_filtered['lat'].max(), data_filtered['long'].max()], zoom_start=7)

    # Parcourez les lignes du DataFrame pour ajouter des marqueurs sur la carte
    for index, row in data_filtered.iterrows():
        latitude, longitude = row['lat'], row['long']
        gravite = row['grav']
        # Détermination de la couleur du marqueur en fonction de la gravité
        if gravite == 'Tué':
            marker_color = 'red'
        elif gravite == 'Indemne':
            marker_color = 'green'
        elif gravite == 'Blessé léger':
            marker_color = 'orange'
        elif gravite == 'Blessé hospitalisé':
            marker_color = 'blue'
        # Ajout d'un marqueur à la carte
        folium.Marker([latitude, longitude], icon=folium.Icon(icon='circle', color=marker_color, min_zoom=10, max_zoom=10)).add_to(m)
    # Affichez la carte dans Streamlit avec bonne largeur
    st.markdown(
        """
        <style>
        #map {
             display: flex;
        justify-content: center;
        align-items: center;
        height: 80vh;  /* Ajustez la hauteur en fonction de vos besoins */
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    # Affichez la carte dans Streamlit
    folium_static(m)
    #st_folium(m)

elif page == "Machine Learning":
    st.title("Page 2")
    st.write("C'est la deuxième page.")

