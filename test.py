import pandas 
import folium
import streamlit as st
import matplotlib.pyplot as plt

st.title("Etude des accidents de vélos depuis 2005")
st.write("Bienvenue !")

# Utilisez pd.read_excel() pour lire le fichier Excel dans un DataFrame
data = pandas.read_excel(r"accidentsVelo.xlsx", decimal=",")  
data.head(5)

# Compter le nombre d'accident par an
acc_an = data['an'].value_counts().reset_index()
acc_an.columns = ['annee', 'nombre_accidents']

# affichage du nombre d'accident par an
fig = plt.figure(figsize=(10, 6))

plt.bar(acc_an['annee'], acc_an['nombre_accidents'])
plt.xlabel('Année du sinistre')
plt.ylabel('Nombre d accident de vélo')
plt.xticks(acc_an['annee'])
plt.title('Evolution annuelle du nombre d accident de velo en France')

st.pyplot(fig)

# Créez un curseur pour sélectionner une année
annee_selectionnee = st.slider("Sélectionnez une année", min_value=acc_an['annee'].min(), max_value=acc_an['annee'].max(), value=acc_an['annee'].max(), step=1)

# Filtrer les données en fonction de l'année sélectionnée
donnees_filtrees = acc_an[acc_an['annee'] == annee_selectionnee]

# Créez un histogramme des valeurs pour l'année sélectionnée
fig, ax = plt.subplots()
ax.bar(donnees_filtrees['annee'], donnees_filtrees['nombre_accidents'])
ax.set_xlabel('annee')
ax.set_ylabel('nombre_accidents')
ax.set_title(f'Histogramme pour l\'année {annee_selectionnee}')

# Affichez le graphique dans Streamlit
st.pyplot(fig)


