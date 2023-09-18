import pandas 
import folium
import streamlit as st

st.title("Ma première application Streamlit")
st.write("Bienvenue sur ma première application Streamlit !")

# Utilisez pd.read_excel() pour lire le fichier Excel dans un DataFrame
data = pandas.read_excel(r"C:\Users\laura\OneDrive\Bureau\ISUP\Data Visualisation\accidentsVelo.xlsx", decimal=",")  
data.head(5)

data=data.dropna(subset=['long'])
data=data.dropna(subset=['lat'])

#réduction de la table pour voir (janvier 2021)
data2021 = data[data.an==2021]
data2021 = data2021[data2021.mois == "janvier"]

# Création de la carte centrée sur la France
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)                                                              

# Parcourir les lignes du DataFrame pour ajouter des marqueurs sur la carte
for index, row in data2021.iterrows():
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
    folium.Marker([latitude, longitude], icon=folium.Icon(icon='circle' , color=marker_color)).add_to(m)

st.write(m)

