import pandas 
import folium
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
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

# Créez un bouton de navigation pour chaque page
page = st.sidebar.selectbox("Choisissez une page", ["Statistiques descriptives", "Graphiques interactifs", "Machine Learning"])

if page == "Statistiques descriptives":
    datamoy = data.groupby(['grav'])['Num_Acc'].count().reset_index()
    datamoy = datamoy.rename(columns={'Num_Acc': 'nb_accidents'})

    st.subheader("74 758 accidents de vélo recensés mais combien sont restés indemnes ?")
    C7, C8, C9, C10 = st.columns(4)
    with C7 :
        st.image("indemne.png", caption=None, width=120, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write("Nombre :", datamoy.loc[datamoy['grav'] == 'Indemne', 'nb_accidents'].values[0])
    with C8 :
        st.image("Blésser léger.png", caption=None, width=120, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write("Nombre :", datamoy.loc[datamoy['grav'] == 'Blessé léger', 'nb_accidents'].values[0])
    with C9 :
        st.image("Blésser hospitalisés.png", caption=None, width=120, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write("Nombre :", datamoy.loc[datamoy['grav'] == 'Blessé hospitalisé', 'nb_accidents'].values[0])
    with C10 :
        st.image("mort.png", caption=None, width=120, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write("Nombre :", datamoy.loc[datamoy['grav'] == 'Tué', 'nb_accidents'].values[0])
    
    st.subheader("Selon les statistiques, quel mois, quel jour, à quelle heure a-t-on le plus de chance d'avoir un accident de vélo ?")
    C2, C3, C4, C5 = st.columns(4)
    with C2 :
        st.image("année.png", caption=None, width=120, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(data['an'].mode().values[0])
    with C3 :
        st.image("mois.png", caption=None, width=120, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(data['mois'].mode().values[0])
    with C4 :
        st.image("jours.png", caption=None, width=120, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(data['jour'].mode().values[0])
    with C5 :
        st.image("heure.png", caption=None, width=120, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.write(data['hrmn'].mode().values[0])
    
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


elif page == "Graphiques interactifs":
    with st.sidebar:
            region_filter = st.selectbox(label='Sélectionnez une région',
                                   options=data['region'].unique(),
                                   index=0) 

    df_filtre = data.query('region == @region_filter')

    # Titre intermédiaire
    st.subheader("L'année 2018, un tournant")

    # Contenu de la première section
    st.write("Ecrire du contenu")


    Q1,Q2 = st.columns(2)

    with Q1:
        df1 = df_filtre.groupby(by = ['an','grav'])['Num_Acc'].count().reset_index()
        df1 = df1.rename(columns={'Num_Acc': 'nb_accidents'})
        fig1 = px.bar(df1,
                                x='an',
                                y='nb_accidents',
                                color='grav',
                                title='Nombre d\'accidents en fonction de l\'année')
        fig1.update_layout(plot_bgcolor = "rgba(0,0,0,0)",
                                        xaxis =(dict(showgrid = False,
                                                     title='Année')),
                                        yaxis =(dict(showgrid = False,
                                                     title='Nombre d\'accidents')))
        st.plotly_chart(fig1,use_container_width=True, figsize=(10, 6))
    st.text("    ")

    with Q2:
        df2 = df_filtre.groupby(by = ['grav'])['Num_Acc'].count().reset_index()
        df2 = df2.rename(columns={'Num_Acc': 'nb_accidents'})
        fig2 = px.pie(df2,names='grav',values='nb_accidents',title='Représentation du nombre d\'accients par gravité')
        fig2.update_layout(plot_bgcolor = "rgba(0,0,0,0)")
        st.plotly_chart(fig2,use_container_width=True)
            
    # Titre intermédiaire
    st.subheader("Que cela soit en agglomération ou hors agglomération, n'oubliez pas l'existance du code de la route")

    # Contenu de la première section
    st.write("Ecrire du contenu")

   # Groupez les données par agglomération et gravité
    df_sankeyagg = df_filtre.groupby(['agg', 'grav']).size().reset_index(name='count')
    # Triez les données en fonction de la catégorie 'sexe'
    df_sankeyagg = df_sankeyagg.sort_values(by=['agg', 'grav'])
    # Créez une liste de noms de nœuds uniques en combinant les colonnes 'sexe' et 'gravité'
    nodes1 = pd.concat([df_sankeyagg['agg'], df_sankeyagg['grav']]).unique()
    # Créez un dictionnaire de correspondance entre les noms de nœuds et des valeurs numériques
    node_mapping1 = {node: index for index, node in enumerate(nodes1)}

    # Créez le diagramme Sankey en utilisant les valeurs numériques pour les sources et les cibles
    figagg = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes1  # Utilisez les noms de nœuds comme étiquettes de nœuds
        ),
        link=dict(
            source=df_sankeyagg['agg'].map(node_mapping1),  # Utilisez les valeurs numériques pour les sources
            target=df_sankeyagg['grav'].map(node_mapping1),  # Utilisez les valeurs numériques pour les cibles
            value=df_sankeyagg['count']
        )
    ))
    figagg.update_layout(title="Relation entre le type de lieux et la gravité en fonction du nombre d'accidents")
    # Affichez le diagramme dans Streamlit
    st.plotly_chart(figagg)

    # Titre intermédiaire
    st.subheader("Mesdames, vous semblez plus prudentes que messieurs")

    # Contenu de la première section
    st.write("Ecrire du contenu")

    # Groupez les données par sexe et gravité
    df_sankeysexe = df_filtre.groupby(['sexe', 'grav']).size().reset_index(name='count')
    # Triez les données en fonction de la catégorie 'sexe'
    df_sankeysexe = df_sankeysexe.sort_values(by=['sexe', 'grav'])
    # Créez une liste de noms de nœuds uniques en combinant les colonnes 'sexe' et 'gravité'
    nodes = pd.concat([df_sankeysexe['sexe'], df_sankeysexe['grav']]).unique()
    # Créez un dictionnaire de correspondance entre les noms de nœuds et des valeurs numériques
    node_mapping = {node: index for index, node in enumerate(nodes)}

    # Créez le diagramme Sankey en utilisant les valeurs numériques pour les sources et les cibles
    figsexe = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes  # Utilisez les noms de nœuds comme étiquettes de nœuds
        ),
        link=dict(
            source=df_sankeysexe['sexe'].map(node_mapping),  # Utilisez les valeurs numériques pour les sources
            target=df_sankeysexe['grav'].map(node_mapping),  # Utilisez les valeurs numériques pour les cibles
            value=df_sankeysexe['count']
        )
    ))
    figsexe.update_layout(title="Relation entre le sexe et la gravité en fonction du nombre d'accidents")
    # Affichez le diagramme dans Streamlit
    st.plotly_chart(figsexe)

    # Titre intermédiaire
    st.subheader("Même en promenade à vélo et qu'importe l'âge, restez vigilents")

    # Contenu de la première section
    st.write("Ecrire du contenu")


    ##NB D'ACCIDENTS SELON L'AGE ET LE TRAJET
    df3 = df_filtre.loc[df_filtre['age']<=100]
    df3 = df3.groupby(by = ['age','trajet'])['Num_Acc'].count().reset_index()
    df3 = df3.rename(columns={'Num_Acc': 'nb_accidents'})
    fig3 = px.line(df3,x='age',y='nb_accidents',color='trajet',title='Nombre d\'accidents en fonction de l\'âge en fonction du type de trajet effectué')
    fig3.update_layout(plot_bgcolor = "rgba(0,0,0,0)",
                       xaxis =(dict(showgrid = False,
                                                     title='Âge')),
                                        yaxis =(dict(showgrid = False,
                                                     title='Nombre d\'accidents')))
    st.plotly_chart(fig3,use_container_width=True)

    Q3,Q4 = st.columns(2)

    with Q3:
        df6 = df_filtre.groupby(by = ['trajet','grav'])['Num_Acc'].count().reset_index()
        df6 = df6.rename(columns={'Num_Acc': 'nb_accidents'})
        fig6 = px.bar(df6,
                                x='trajet',
                                y='nb_accidents',
                                color='grav',
                                title='Nombre d\'accidents en fonction du type de trajet et par gravité')
        fig6.update_layout(plot_bgcolor = "rgba(0,0,0,0)",
                                        xaxis =(dict(tickangle=45,  # Angle d'inclinaison
                                                    title='Trajet',  # Titre de l'axe des x
                                                    showgrid=False)),
                                        yaxis =(dict(showgrid = False,
                                                     title='Nombre d\'accidents')))
        st.plotly_chart(fig6,use_container_width=True, figsize=(10, 6))

    with Q4:
        # Créez des catégories d'âge en fonction de votre colonne d'âge
        bins = [0, 15, 25, 40, 50, 60, 70, 80, 100]
        labels = ['0-15', '16-25', '26-40', '41-50', '51-60', '61-70', '71-80', '81-100']
        df_filtre['age_group'] = pd.cut(df_filtre['age'], bins=bins, labels=labels, right=False)

        # Groupez les données par tranche d'âge et gravité
        df_age_gravity = df_filtre.groupby(['age_group', 'grav'])['Num_Acc'].count().reset_index()
        df_age_gravity = df_age_gravity.rename(columns={'Num_Acc': 'nb_accidents'})

        # Créez un histogramme
        fig_age_gravity = px.bar(
            df_age_gravity,
            x='age_group',
            y='nb_accidents',
            color='grav',
            title='Nombre d\'accidents en fonction des tranches d\'âge et par gravité'
        )

        fig_age_gravity.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                      xaxis =(dict(tickangle=45,  # Angle d'inclinaison
                                                    title='Âge',  # Titre de l'axe des x
                                                    showgrid=False)),
                                        yaxis =(dict(showgrid = False,
                                                     title='Nombre d\'accidents')))

        # Affichez l'histogramme dans Streamlit
        st.plotly_chart(fig_age_gravity, use_container_width=True)
    
    # Titre intermédiaire
    st.subheader("L'été, une période à haut risque : zoom sur votre région")

    # Contenu de la première section
    st.write("Ecrire du contenu")

    #SAISONNALITE DES ACCIDENTS PAR GRAVITE
    df4 = df_filtre.groupby(by = ['mois','grav'])['Num_Acc'].count().reset_index()
    df4 = df4.rename(columns={'Num_Acc': 'nb_accidents'})
    # Liste des mois dans l'ordre de l'arrivée
    ordre_des_mois = ["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre", "novembre", "decembre"]
    df4.columns = ['mois', 'grav', 'nb_accidents']
    df4['mois'] = pandas.Categorical(df4['mois'], categories=ordre_des_mois, ordered=True)
    df4 = df4.sort_values(by='mois')
    fig4 = px.line(df4,x='mois',y='nb_accidents',color='grav',title='Saisonnalité du nombre d\'accidents par rapport à la gravité', category_orders={"mois": ordre_des_mois})
    fig4.update_layout(plot_bgcolor = "rgba(0,0,0,0)",
                       xaxis =(dict(tickangle=45,  # Angle d'inclinaison
                                                    title='Mois',  # Titre de l'axe des x
                                                    showgrid=False)),
                                        yaxis =(dict(showgrid = False,
                                                     title='Nombre d\'accidents')))
    st.plotly_chart(fig4,use_container_width=True)

    ##CARTE
    # Supprimez les lignes avec des valeurs NaN dans les colonnes 'lat' et 'long'
    data_carte = df_filtre.dropna(subset=['lat', 'long'])
    # Créez un menu déroulant pour sélectionner le mois
    selected_month = st.selectbox("Sélectionnez un mois", ["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre", "novembre", "decembre"])
    # Filtrez les données en fonction de l'année et du mois sélectionnés
    data_filtered = data_carte[((data_carte['an']==2021) & (data_carte['mois'] == selected_month))]
    col_lat = data_filtered['lat']  # Utilisez la moyenne des latitudes pour centrer la carte
    col_long = data_filtered['long'] # Utilisez la moyenne des longitudes pour centrer la carte
    # Créez une carte centrée sur la France
    m = folium.Map(location=[col_lat.mean(), col_long.mean()], zoom_start=7.5)

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

